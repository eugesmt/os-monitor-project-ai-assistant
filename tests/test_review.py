import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dev_assistant.review import (
    DEFAULT_REVIEW_MAX_DIFF_CHARS,
    NO_CHANGES_MESSAGE,
    _diff_args,
    _review_max_diff_chars,
    _ReviewProgress,
    _split_diff,
    run_review,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return result.stdout


class _FakeStderr(io.StringIO):
    def __init__(self, interactive: bool):
        super().__init__()
        self.interactive = interactive
        self.writes: list[str] = []
        self.flushes = 0

    def isatty(self) -> bool:
        return self.interactive

    def write(self, value: str) -> int:
        self.writes.append(value)
        return super().write(value)

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


class ReviewTests(unittest.TestCase):
    def _repository(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "project"
        root.mkdir()
        _git(root, "init", "-q", "-b", "develop")
        _git(root, "config", "user.email", "review@example.test")
        _git(root, "config", "user.name", "Review Test")
        (root / "app.py").write_text("return 1\n", encoding="utf-8")
        _git(root, "add", "app.py")
        _git(root, "commit", "-qm", "initial")
        return temporary, root

    def test_committed_review_uses_triple_dot_range_and_rag(self):
        temporary, root = self._repository()
        self.addCleanup(temporary.cleanup)
        _git(root, "switch", "-c", "feature/review", "-q")
        (root / "app.py").write_text("return 2\n", encoding="utf-8")
        _git(root, "add", "app.py")
        _git(root, "commit", "-qm", "change")
        output_root = Path(temporary.name) / "assistant"
        sources = [{"source": "docs/design.md", "start_line": 4, "end_line": 8, "text": "Repository contract"}]
        with patch("dev_assistant.review.embed", return_value=[0.2, 0.8]) as embed, patch(
            "dev_assistant.review.search", return_value=sources
        ) as search, patch("dev_assistant.review.review", return_value="review") as model:
            result = run_review(root, output_root)

        self.assertEqual(result, "review")
        embed.assert_called_once()
        self.assertIn("app.py", embed.call_args.args[0])
        self.assertIn("return 2", embed.call_args.args[0])
        search.assert_called_once_with(
            output_root / "indexes" / "structural" / "index.faiss",
            output_root / "indexes" / "structural" / "chunks.sqlite",
            [0.2, 0.8],
            top_k=5,
            query_text=embed.call_args.args[0],
        )
        model.assert_called_once()
        self.assertIn("return 2", model.call_args.args[0])
        self.assertEqual(model.call_args.args[2], ["app.py"])

    def test_working_tree_contains_staged_and_unstaged_changes(self):
        temporary, root = self._repository()
        self.addCleanup(temporary.cleanup)
        (root / "app.py").write_text("return 'staged'\n", encoding="utf-8")
        _git(root, "add", "app.py")
        (root / "app.py").write_text("return 'unstaged'\n", encoding="utf-8")
        output_root = Path(temporary.name) / "assistant"
        with patch("dev_assistant.review.embed", return_value=[1.0]), patch(
            "dev_assistant.review.search", return_value=[]
        ), patch("dev_assistant.review.review", side_effect=lambda diff, sources, files: diff) as model:
            result = run_review(root, output_root, working_tree=True)

        # ``git diff HEAD`` compares the final worktree state to HEAD, so both
        # the staged and unstaged edits are represented by the resulting patch.
        self.assertIn("-return 1", result)
        self.assertIn("+return 'unstaged'", result)
        self.assertEqual(model.call_args.args[2], ["app.py"])

    def test_clean_working_tree_does_not_call_embeddings_or_model(self):
        temporary, root = self._repository()
        self.addCleanup(temporary.cleanup)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch("dev_assistant.review.embed") as embed, patch("dev_assistant.review.review") as model:
            result = run_review(root, Path(temporary.name) / "assistant", working_tree=True)
        self.assertEqual(result, NO_CHANGES_MESSAGE)
        embed.assert_not_called()
        model.assert_not_called()
        self.assertEqual(stderr.getvalue(), "")

    def test_revision_options_are_rejected(self):
        with self.assertRaises(ValueError):
            _diff_args("--output=bad", "HEAD", working_tree=False)
        with self.assertRaises(ValueError):
            _diff_args("main", "--no-ext-diff", working_tree=False)

    def test_large_diff_is_batched_without_losing_source_or_exceeding_limit(self):
        temporary, root = self._repository()
        self.addCleanup(temporary.cleanup)
        # Each block is a little below the limit; batching should prefer the
        # file boundaries and then split only where a combined batch is too big.
        diff = "".join(
            f"diff --git a/file{number}.go b/file{number}.go\n"
            f"@@ -1 +1 @@\n-previous-{number}\n+updated-{number}\n"
            for number in range(30)
        )
        limit = 256
        captured: list[str] = []
        stderr = io.StringIO()

        def fake_review(part, sources, files):
            captured.append(part)
            return f"Потенциальные баги\nНе найдено\n\nАрхитектурные проблемы\nНе найдено\n\nРекомендации\nНе найдено"

        with contextlib.redirect_stderr(stderr), patch.dict(os.environ, {"REVIEW_MAX_DIFF_CHARS": str(limit)}), patch(
            "dev_assistant.review._git_changes", return_value=(diff, [f"file{number}.go" for number in range(30)])
        ), patch("dev_assistant.review.embed", side_effect=lambda query: [float(len(query))]), patch(
            "dev_assistant.review.search", return_value=[]
        ), patch("dev_assistant.review.review", side_effect=fake_review) as model:
            result = run_review(root, Path(temporary.name) / "assistant")

        self.assertGreater(len(captured), 1)
        self.assertTrue(all(len(part) <= limit for part in captured))
        self.assertEqual("".join(captured), diff)
        self.assertEqual(model.call_count, len(captured))
        self.assertIn("## Часть 1 из", result)
        self.assertIn("## Часть 2 из", result)
        progress = stderr.getvalue().splitlines()
        self.assertEqual(progress, [f"Reviewing batch {number}/{len(captured)}..." for number in range(1, len(captured) + 1)])
        self.assertNotIn("Reviewing batch", result)

    def test_small_diff_keeps_single_model_call(self):
        temporary, root = self._repository()
        self.addCleanup(temporary.cleanup)
        diff = "diff --git a/app.py b/app.py\n+small\n"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), patch.dict(os.environ, {"REVIEW_MAX_DIFF_CHARS": "256"}), patch(
            "dev_assistant.review._git_changes", return_value=(diff, ["app.py"])
        ), patch("dev_assistant.review.embed", return_value=[1.0]) as embed, patch(
            "dev_assistant.review.search", return_value=[]
        ), patch("dev_assistant.review.review", return_value="review") as model:
            result = run_review(root, Path(temporary.name) / "assistant")
        self.assertEqual(result, "review")
        embed.assert_called_once()
        model.assert_called_once_with(diff, [], ["app.py"])
        self.assertEqual(stderr.getvalue(), "Reviewing batch 1/1...\n")
        self.assertNotIn("Reviewing batch", result)

    def test_review_limit_invalid_values_fall_back_to_default(self):
        with patch.dict(os.environ, {"REVIEW_MAX_DIFF_CHARS": "not-a-number"}):
            self.assertEqual(_review_max_diff_chars(), DEFAULT_REVIEW_MAX_DIFF_CHARS)
        with patch.dict(os.environ, {"REVIEW_MAX_DIFF_CHARS": "1"}):
            self.assertEqual(_review_max_diff_chars(), DEFAULT_REVIEW_MAX_DIFF_CHARS)

    def test_progress_tty_reuses_row_and_finishes_with_newline(self):
        stderr = _FakeStderr(interactive=True)
        progress = _ReviewProgress(stderr)
        progress.update(1, 2)
        progress.update(2, 2)
        progress.finish()
        self.assertEqual(
            stderr.writes,
            ["\rReviewing batch 1/2...", "\rReviewing batch 2/2...", "\n"],
        )
        self.assertEqual(stderr.getvalue(), "\rReviewing batch 1/2...\rReviewing batch 2/2...\n")
        self.assertEqual(stderr.flushes, 3)

    def test_progress_non_tty_keeps_one_line_per_batch(self):
        stderr = _FakeStderr(interactive=False)
        progress = _ReviewProgress(stderr)
        progress.update(1, 2)
        progress.update(2, 2)
        progress.finish()
        self.assertEqual(stderr.writes, ["Reviewing batch 1/2...\n", "Reviewing batch 2/2...\n"])
        self.assertEqual(stderr.getvalue(), "Reviewing batch 1/2...\nReviewing batch 2/2...\n")
        self.assertEqual(stderr.flushes, 2)


if __name__ == "__main__":
    unittest.main()

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from dev_assistant import cli


class ReviewCLITests(unittest.TestCase):
    def test_review_command_ensures_index_calls_review_and_writes_utf8_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            output_root = root / "assistant"
            report_path = root / "reports" / "review.md"
            def fake_run_review(**kwargs):
                self.assertEqual(kwargs["project_root"], project)
                self.assertEqual(kwargs["output_root"], output_root)
                self.assertEqual(kwargs["base"], "develop")
                self.assertEqual(kwargs["head"], "feature/demo")
                self.assertFalse(kwargs["working_tree"])
                return "Найдена проблема"

            fake_review = types.SimpleNamespace(run_review=fake_run_review)
            rendered = io.StringIO()
            with patch.object(cli, "ensure_index") as ensure, patch.dict(
                sys.modules, {"dev_assistant.review": fake_review}
            ), contextlib.redirect_stdout(rendered):
                result = cli.review_command(
                    project,
                    output_root,
                    base="develop",
                    head="feature/demo",
                    output=report_path,
                )

            ensure.assert_called_once_with(project, output_root, reindex=False)
            self.assertEqual(result, "Найдена проблема")
            self.assertEqual(rendered.getvalue(), "Найдена проблема\n")
            self.assertEqual(report_path.read_text(encoding="utf-8"), "Найдена проблема")

    def test_main_passes_default_refs_to_review(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            with patch.object(cli, "default_project_root", return_value=project), patch.object(
                cli, "assistant_root", return_value=Path(directory) / "assistant"
            ), patch.object(cli, "review_command") as review, patch.object(
                sys, "argv", ["taskmanager-assistant", "review"]
            ):
                cli.main()

            review.assert_called_once_with(
                project.resolve(),
                Path(directory, "assistant"),
                base="develop",
                head="HEAD",
                working_tree=False,
                output=None,
                reindex=False,
            )

    def test_working_tree_cannot_be_combined_with_refs(self):
        with patch.object(cli, "default_project_root", return_value=Path("/tmp/project")), patch.object(
            sys, "argv", ["taskmanager-assistant", "review", "--working-tree", "--base", "main"]
        ), self.assertRaises(SystemExit) as raised:
            cli.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

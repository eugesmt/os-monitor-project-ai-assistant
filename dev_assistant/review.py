"""Local Git code-review orchestration.

The review flow intentionally keeps the patch itself as the primary model
context.  The existing embedding index is used only to retrieve surrounding
project documentation/code that helps explain the change.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from .deepseek import review
from .ollama import embed
from .storage import search


NO_CHANGES_MESSAGE = "Нет изменений для ревью."
_DIFF_UNIFIED_LINES = 80
_DEFAULT_TOP_K = 5
DEFAULT_REVIEW_MAX_DIFF_CHARS = 60_000
MIN_REVIEW_MAX_DIFF_CHARS = 256
MAX_REVIEW_MAX_DIFF_CHARS = 1_000_000


class _ReviewProgress:
    """Render batch progress appropriately for terminals and CI logs."""

    def __init__(self, stream) -> None:
        self.stream = stream
        self.interactive = bool(stream.isatty())

    def update(self, number: int, total: int) -> None:
        suffix = "" if self.interactive else "\n"
        prefix = "\r" if self.interactive else ""
        self.stream.write(f"{prefix}Reviewing batch {number}/{total}...{suffix}")
        self.stream.flush()

    def finish(self) -> None:
        if self.interactive:
            self.stream.write("\n")
            self.stream.flush()


def _validate_ref(value: str, name: str) -> str:
    """Validate a user-provided revision before putting it in argv.

    ``--end-of-options`` below is the actual protection against Git option
    injection.  Rejecting option-looking and NUL-containing values gives a
    clearer error and avoids constructing surprising revision expressions.
    Git revision expressions such as ``HEAD~1`` remain valid.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty Git revision")
    if "\x00" in value:
        raise ValueError(f"{name} contains NUL")
    if value.startswith("-"):
        raise ValueError(f"{name} must not start with '-' (Git options are not allowed)")
    return value


def _run_git(project_root: Path, args: list[str]) -> str:
    """Run Git without a shell and return stdout, with a useful error."""
    try:
        result = subprocess.run(
            args,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"git diff failed{suffix}") from exc
    return result.stdout


def _diff_args(base: str, head: str, *, working_tree: bool, names_only: bool = False) -> list[str]:
    """Build a safe ``git diff`` argv for committed or working-tree mode."""
    options = ["git", "diff", "--no-ext-diff"]
    if names_only:
        options.append("--name-only")
    else:
        options.append(f"--unified={_DIFF_UNIFIED_LINES}")

    # In working-tree mode ``git diff HEAD`` includes both staged and unstaged
    # changes.  Untracked files intentionally remain outside this review until
    # they are added to the index, because Git has no HEAD diff for them.
    if working_tree:
        revision = "HEAD"
    else:
        revision = f"{_validate_ref(base, 'base')}...{_validate_ref(head, 'head')}"
    options.extend(["--end-of-options", revision, "--"])
    return options


def _git_changes(project_root: Path, base: str, head: str, working_tree: bool) -> tuple[str, list[str]]:
    diff = _run_git(project_root, _diff_args(base, head, working_tree=working_tree))
    names_output = _run_git(project_root, _diff_args(base, head, working_tree=working_tree, names_only=True))
    # Newline-delimited names are the human-friendly Git default.  Empty lines
    # are ignored, while preserving Git's order for deterministic prompts.
    names: list[str] = []
    seen: set[str] = set()
    for name in names_output.splitlines():
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return diff, names


def _index_paths(output_root: Path) -> tuple[Path, Path]:
    index_root = output_root / "indexes" / "structural"
    return index_root / "index.faiss", index_root / "chunks.sqlite"


def _rag_query(diff: str, changed_files: list[str]) -> str:
    """Make a bounded embedding query from the changed area of the patch."""
    files = ", ".join(changed_files) or "(files unavailable)"
    # The complete diff is still sent to DeepSeek.  Bounding only this query
    # keeps local embedding requests responsive on very large changes.
    excerpt = diff[:12000]
    return f"Code review context\nChanged files: {files}\nDiff excerpt:\n{excerpt}"


def _top_k() -> int:
    try:
        value = int(os.getenv("RAG_TOP_K", str(_DEFAULT_TOP_K)))
    except ValueError:
        value = _DEFAULT_TOP_K
    return max(1, min(value, 20))


def _review_max_diff_chars() -> int:
    """Read and validate the per-model diff limit.

    A lower bound prevents accidentally turning one review into thousands of
    tiny API calls, while the upper bound keeps a typo in the environment from
    defeating batching altogether.  Invalid values deliberately fall back to
    the useful default instead of making an otherwise readable diff fail.
    """
    raw = os.getenv("REVIEW_MAX_DIFF_CHARS")
    if raw is None:
        return DEFAULT_REVIEW_MAX_DIFF_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_REVIEW_MAX_DIFF_CHARS
    if not MIN_REVIEW_MAX_DIFF_CHARS <= value <= MAX_REVIEW_MAX_DIFF_CHARS:
        return DEFAULT_REVIEW_MAX_DIFF_CHARS
    return value


def _hard_split(text: str, limit: int) -> list[str]:
    """Split text into exact, bounded pieces, preferring newline boundaries."""
    pieces: list[str] = []
    remainder = text
    while len(remainder) > limit:
        cut = remainder.rfind("\n", 0, limit + 1)
        # A single unusually long source line still must be bounded.
        if cut <= 0:
            cut = limit
        else:
            cut += 1
        pieces.append(remainder[:cut])
        remainder = remainder[cut:]
    if remainder:
        pieces.append(remainder)
    return pieces


def _split_diff(diff: str, limit: int) -> list[str]:
    """Split a diff into exact pieces no longer than *limit* characters.

    Normal diffs are grouped by their ``diff --git`` file blocks, so a model
    usually receives whole files.  A block larger than the limit is split at
    deterministic newline boundaries (or hard boundaries for a long line).
    Concatenating the returned pieces always reproduces ``diff`` byte-for-byte.
    """
    if len(diff) <= limit:
        return [diff]
    starts = [match.start() for match in re.finditer(r"(?m)^diff --git ", diff)]
    if not starts:
        return _hard_split(diff, limit)

    segments: list[str] = []
    if starts[0] > 0:
        segments.append(diff[: starts[0]])
    segments.extend(diff[start:end] for start, end in zip(starts, [*starts[1:], len(diff)]))

    pieces: list[str] = []
    current = ""
    for segment in segments:
        if len(segment) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(segment, limit))
        elif not current:
            current = segment
        elif len(current) + len(segment) <= limit:
            current += segment
        else:
            pieces.append(current)
            current = segment
    if current:
        pieces.append(current)
    return pieces


def run_review(
    project_root: Path,
    output_root: Path,
    base: str = "develop",
    head: str = "HEAD",
    working_tree: bool = False,
) -> str:
    """Review a committed range or current working tree and return Markdown.

    Committed mode compares ``BASE...HEAD`` (with ``develop`` as the default
    base).  Working-tree mode compares
    ``HEAD`` and therefore includes staged and unstaged changes; untracked
    files are deliberately excluded until staged.  A clean tree returns a
    short message and does not invoke embeddings or the LLM.  Large diffs are
    split into bounded per-part reviews using ``REVIEW_MAX_DIFF_CHARS`` and
    joined with explicit part markers; no extra synthesis request is made.
    """
    project_root = Path(project_root)
    output_root = Path(output_root)
    if not project_root.is_dir():
        raise FileNotFoundError(f"project root does not exist: {project_root}")

    diff, changed_files = _git_changes(project_root, base, head, working_tree)
    if not diff.strip():
        return NO_CHANGES_MESSAGE

    index_path, database_path = _index_paths(output_root)
    parts = _split_diff(diff, _review_max_diff_chars())
    reports: list[str] = []
    total = len(parts)
    progress = _ReviewProgress(sys.stderr)
    try:
        for number, part in enumerate(parts, start=1):
            progress.update(number, total)
            query = _rag_query(part, changed_files)
            sources = search(index_path, database_path, embed(query), top_k=_top_k(), query_text=query)
            report = review(part, sources, changed_files)
            if total == 1:
                return report
            reports.append(f"<!-- review-part: {number}/{total} -->\n\n## Часть {number} из {total}\n\n{report}")
        return "\n\n".join(reports)
    finally:
        progress.finish()


# Backward-compatible descriptive name for callers that want to distinguish
# this model operation from the orchestration module itself.
deepseek_review = review

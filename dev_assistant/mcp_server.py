"""Small, local line-delimited JSON-RPC MCP server.

The server deliberately exposes a small set of file tools instead of a shell
tool. Every path supplied by the caller is resolved relative to the fixed
``--project-root`` and checked after resolving symlinks.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MAX_FILE_BYTES = 1_048_576
MAX_RESULT_BYTES = 4 * 1_048_576
MAX_LIST_ENTRIES = 10_000

# These directories are either VCS metadata or routinely contain very large,
# generated dependency/build caches. They are skipped even with
# ``include_hidden=True``; an explicit project-root file tool should inspect
# source, not recursively ingest these implementation details.
SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git", ".hg", ".svn",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".tox", ".nox", ".venv", "venv", "env",
        "build", "dist", "target", "coverage", ".gradle",
    }
)


def git_branch(project_root: Path) -> dict:
    result = subprocess.run(
        ["git", "-C", str(project_root), "branch", "--show-current"],
        check=False, capture_output=True, text=True, timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "project root is not a Git repository")
    branch = result.stdout.strip()
    if branch:
        return {"branch": branch, "detached": False}
    commit = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True, timeout=5,
    )
    if commit.returncode != 0:
        raise RuntimeError("unable to read Git HEAD")
    return {"branch": None, "commit": commit.stdout.strip(), "detached": True}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() or "."


def _safe_path(project_root: Path, user_path: str | None, *, allow_missing: bool = False) -> Path:
    """Resolve a caller path, rejecting traversal and symlink escapes."""
    if user_path is None or user_path == "":
        user_path = "."
    if not isinstance(user_path, str):
        raise ValueError("path must be a relative string")
    candidate_text = user_path.replace("\\", "/")
    candidate = Path(candidate_text)
    # ``Path`` follows the host OS, so explicitly reject Windows drive paths
    # too when this Linux server receives a request from a cross-platform
    # client.
    windows_absolute = len(candidate_text) >= 3 and candidate_text[1] == ":" and candidate_text[2] == "/"
    if candidate.is_absolute() or windows_absolute or any(part == ".." for part in candidate.parts):
        raise ValueError("path must stay inside project root (absolute paths and '..' are forbidden)")
    root = project_root.resolve(strict=True)
    resolved = (root / candidate).resolve(strict=not allow_missing)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("path resolves outside project root") from exc
    if not allow_missing and not resolved.exists():
        raise FileNotFoundError(_relative(resolved, root))
    return resolved


def _ensure_regular_file(path: Path, *, project_root: Path) -> None:
    if not path.is_file():
        raise ValueError(f"not a regular file: {_relative(path, project_root)}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"unable to stat file: {exc}") from exc
    if size > MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {MAX_FILE_BYTES} byte limit")


def _read_text(project_root: Path, path: str | None) -> str:
    resolved = _safe_path(project_root, path)
    _ensure_regular_file(resolved, project_root=project_root)
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read file: {exc}") from exc
    if b"\x00" in raw:
        raise ValueError("binary files are not supported")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("file is not valid UTF-8 text") from exc


def list_files(
    project_root: Path,
    path: str | None = None,
    recursive: bool = True,
    include_hidden: bool = False,
    max_entries: int = MAX_LIST_ENTRIES,
) -> dict:
    root = _safe_path(project_root, path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {_relative(root, project_root)}")
    if root.name in SKIPPED_DIRECTORY_NAMES:
        raise ValueError(f"directory is excluded from recursive file tools: {_relative(root, project_root)}")
    if not isinstance(max_entries, int) or not 0 < max_entries <= MAX_LIST_ENTRIES:
        raise ValueError(f"max_entries must be between 1 and {MAX_LIST_ENTRIES}")
    entries: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    def consider(item: Path) -> None:
        rel = _relative(item, project_root)
        if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
            return
        try:
            item.resolve(strict=True).relative_to(project_root.resolve(strict=True))
        except (OSError, ValueError):
            skipped.append({"path": rel, "reason": "symlink escapes project root"})
            return
        kind = "directory" if item.is_dir() else "file" if item.is_file() else "other"
        entries.append({"path": rel, "type": kind})

    if recursive:
        # os.walk lets us prune generated/cache trees before descending into
        # them. Path.rglob would enumerate their contents first and could
        # become unexpectedly expensive on a real project checkout.
        for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            dirnames.sort()
            filenames.sort()
            kept_dirs: list[str] = []
            for name in dirnames:
                item = Path(directory) / name
                if name in SKIPPED_DIRECTORY_NAMES or (not include_hidden and name.startswith(".")):
                    continue
                consider(item)
                # Keep traversing real directories and internal symlink dirs
                # only when they resolve inside the project.
                try:
                    if item.is_dir() and item.resolve(strict=True).is_relative_to(project_root.resolve(strict=True)):
                        kept_dirs.append(name)
                except (OSError, ValueError):
                    continue
            dirnames[:] = kept_dirs
            for name in filenames:
                consider(Path(directory) / name)
            if len(entries) >= max_entries:
                break
    else:
        for item in sorted(root.iterdir(), key=lambda p: p.name):
            if item.is_dir() and item.name in SKIPPED_DIRECTORY_NAMES:
                continue
            consider(item)
            if len(entries) >= max_entries:
                break
    entries.sort(key=lambda entry: entry["path"])
    return {
        "path": _relative(root, project_root),
        "entries": entries,
        "files": [entry["path"] for entry in entries if entry["type"] == "file"],
        "skipped": skipped,
        "truncated": len(entries) >= max_entries,
    }


def read_file(project_root: Path, path: str) -> dict:
    content = _read_text(project_root, path)
    resolved = _safe_path(project_root, path)
    return {"path": _relative(resolved, project_root), "content": content, "size": len(content.encode("utf-8"))}


def search_text(
    project_root: Path,
    query: str,
    path: str | None = None,
    case_sensitive: bool = True,
    max_matches: int = 1000,
) -> dict:
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    if not isinstance(max_matches, int) or not 0 < max_matches <= MAX_LIST_ENTRIES:
        raise ValueError(f"max_matches must be between 1 and {MAX_LIST_ENTRIES}")
    base = _safe_path(project_root, path)
    if base.is_file():
        candidates = [_relative(base, project_root)]
    else:
        candidates = list_files(project_root, path, recursive=True)["files"]
    needle = query if case_sensitive else query.casefold()
    matches: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for relative in candidates:
        try:
            text = _read_text(project_root, relative)
        except (ValueError, OSError) as exc:
            skipped.append({"path": relative, "reason": str(exc)})
            continue
        for number, line in enumerate(text.splitlines(), 1):
            haystack = line if case_sensitive else line.casefold()
            if needle in haystack:
                matches.append({"path": relative, "line": number, "text": line})
                if len(matches) >= max_matches:
                    return {"query": query, "matches": matches, "skipped": skipped, "truncated": True}
    return {"query": query, "matches": matches, "skipped": skipped, "truncated": False}


def _write_text(project_root: Path, path: str, content: str, *, overwrite: bool) -> dict:
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError(f"content exceeds {MAX_FILE_BYTES} byte limit")
    resolved = _safe_path(project_root, path, allow_missing=True)
    if resolved == project_root.resolve(strict=True):
        raise ValueError("a file path is required")
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"file already exists: {_relative(resolved, project_root)}")
    parent = resolved.parent
    if not parent.is_dir():
        raise ValueError("parent directory does not exist")
    existing_mode = None
    if resolved.exists():
        if not resolved.is_file():
            raise ValueError(f"not a regular file: {_relative(resolved, project_root)}")
        existing_mode = stat.S_IMODE(resolved.stat().st_mode)
    with tempfile.NamedTemporaryFile("wb", dir=parent, prefix=".mcp-", delete=False) as temporary:
        temporary.write(encoded)
        temporary_path = Path(temporary.name)
    try:
        if existing_mode is not None:
            # Apply the old mode before replacement so the target never
            # becomes the temporary file's umask-derived 0600/0666 mode.
            os.chmod(temporary_path, existing_mode)
        os.replace(temporary_path, resolved)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"path": _relative(resolved, project_root), "size": len(encoded), "created": not overwrite}


def write_file(project_root: Path, path: str, content: str, overwrite: bool = False) -> dict:
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    return _write_text(project_root, path, content, overwrite=overwrite)


def apply_patch(
    project_root: Path,
    path: str | None = None,
    old_text: str | None = None,
    new_text: str | None = None,
    content: str | None = None,
    patch: str | None = None,
) -> dict:
    """Apply an exact replacement, or a small unified text patch."""
    if patch is not None:
        if any(value is not None for value in (old_text, new_text, content)):
            raise ValueError("use either patch or old_text/new_text/content")
        target = path
        hunks: list[list[str]] = []
        current_hunk: list[str] | None = None
        for line in patch.splitlines(keepends=True):
            if line.startswith("*** Update File:"):
                target = line.split(":", 1)[1].strip()
                continue
            if line.startswith("+++ "):
                header = line[4:].strip().split("\t", 1)[0]
                target = header[2:] if header.startswith(("a/", "b/")) else header
                continue
            if line.startswith(("*** Begin Patch", "*** End Patch", "--- ")):
                continue
            if line.startswith("@@"):
                current_hunk = []
                hunks.append(current_hunk)
                continue
            if current_hunk is not None and line[:1] in {" ", "+", "-"} and not line.startswith("\\ No newline"):
                current_hunk.append(line)
        if not target or not hunks:
            raise ValueError("patch must contain a target path and at least one text hunk")
        current = _read_text(project_root, target)
        updated = current
        for hunk in hunks:
            old_block = "".join(line[1:] for line in hunk if line.startswith((" ", "-")))
            new_block = "".join(line[1:] for line in hunk if line.startswith((" ", "+")))
            if not old_block or updated.count(old_block) != 1:
                raise ValueError("patch context must occur exactly once")
            updated = updated.replace(old_block, new_block, 1)
        path = target
        result = _write_text(project_root, path, updated, overwrite=True)
        result["changed"] = current != updated
        return result

    if path is None:
        raise ValueError("path is required when patch is not supplied")
    current = _read_text(project_root, path)
    if content is not None:
        if old_text is not None or new_text is not None:
            raise ValueError("use either content or old_text/new_text")
        updated = content
    else:
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ValueError("old_text and new_text are required")
        occurrences = current.count(old_text)
        if occurrences != 1:
            raise ValueError(f"old_text must occur exactly once (found {occurrences})")
        updated = current.replace(old_text, new_text, 1)
    result = _write_text(project_root, path, updated, overwrite=True)
    result["changed"] = current != updated
    return result


def get_diff(project_root: Path, path: str | None = None, staged: bool = False) -> dict:
    relative = None
    if path is not None:
        resolved = _safe_path(project_root, path, allow_missing=True)
        relative = _relative(resolved, project_root)
    command = ["git", "-C", str(project_root), "diff"]
    if staged:
        command.append("--cached")
    command.append("--")
    if relative and relative != ".":
        command.append(relative)
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "unable to read Git diff")
    diff = result.stdout

    # An explicit file request is allowed to inspect an ignored file (for
    # example, documentation generated under an ignored docs/ directory).
    # Git's normal diff and ``ls-files --others --exclude-standard`` both omit
    # it, so use the same bounded no-index path as for ordinary untracked
    # files. A tracked path stays on normal Git diff semantics, including an
    # empty diff for a clean tracked file.
    explicit_untracked = False
    if relative and relative != ".":
        tracked = subprocess.run(
            ["git", "-C", str(project_root), "ls-files", "--error-unmatch", "--", relative],
            check=False, capture_output=True, text=True, timeout=10,
        )
        if tracked.returncode != 0:
            try:
                explicit_path = _safe_path(project_root, relative)
                _ensure_regular_file(explicit_path, project_root=project_root)
                _read_text(project_root, relative)
            except (OSError, ValueError) as exc:
                raise ValueError(f"explicit diff path is not a safe UTF-8 text file: {exc}") from exc
            creation = subprocess.run(
                ["git", "-C", str(project_root), "diff", "--no-index", "--", "/dev/null", relative],
                check=False, capture_output=True, text=True, timeout=10,
            )
            if creation.returncode not in (0, 1):
                raise RuntimeError(creation.stderr.strip() or "unable to create explicit file diff")
            diff += creation.stdout
            explicit_untracked = True

    # ``git diff`` intentionally omits untracked files. Include only
    # untracked regular UTF-8 files that pass the same root/size/binary checks
    # as read_file, using Git's own no-index formatter for a normal patch.
    # Staged diffs retain Git semantics and therefore do not include them.
    if not staged and not explicit_untracked:
        untracked_command = [
            "git", "-C", str(project_root), "ls-files", "--others",
            "--exclude-standard", "-z", "--",
        ]
        if relative and relative != ".":
            untracked_command.append(relative)
        untracked = subprocess.run(
            untracked_command, check=False, capture_output=True, text=True, timeout=10,
        )
        if untracked.returncode != 0:
            raise RuntimeError(untracked.stderr.strip() or "unable to list untracked files")
        untracked_count = 0
        for item in filter(None, untracked.stdout.split("\x00")):
            if untracked_count >= MAX_LIST_ENTRIES:
                break
            if any(part in SKIPPED_DIRECTORY_NAMES for part in Path(item).parts):
                continue
            try:
                untracked_path = _safe_path(project_root, item)
                _ensure_regular_file(untracked_path, project_root=project_root)
                _read_text(project_root, item)
            except (OSError, ValueError):
                # Binary/oversized/unresolvable untracked files are not safe
                # to place in a text diff and are intentionally omitted.
                continue
            creation = subprocess.run(
                ["git", "-C", str(project_root), "diff", "--no-index", "--", "/dev/null", item],
                check=False, capture_output=True, text=True, timeout=10,
            )
            if creation.returncode not in (0, 1):
                raise RuntimeError(creation.stderr.strip() or "unable to create untracked file diff")
            diff += creation.stdout
            untracked_count += 1

    if len(diff.encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError(f"diff exceeds {MAX_RESULT_BYTES} byte limit")
    return {"path": relative, "staged": bool(staged), "diff": diff}


def response(request_id, result=None, error=None) -> dict:
    payload = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


TOOL_DEFINITIONS = [
    {"name": "get_git_branch", "description": "Return the current branch of the configured project root.", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "list_files", "description": "List files and directories inside the configured project root.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}, "include_hidden": {"type": "boolean"}, "max_entries": {"type": "integer"}}, "additionalProperties": False}},
    {"name": "read_file", "description": "Read one UTF-8 text file inside the configured project root.", "inputSchema": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}, "additionalProperties": False}},
    {"name": "search_text", "description": "Search UTF-8 text files inside the configured project root.", "inputSchema": {"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "case_sensitive": {"type": "boolean"}, "max_matches": {"type": "integer"}}, "additionalProperties": False}},
    {"name": "write_file", "description": "Create a UTF-8 text file; overwrite must be explicitly enabled.", "inputSchema": {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"}}, "additionalProperties": False}},
    {"name": "apply_patch", "description": "Apply an exact replacement or unified patch to one UTF-8 text file.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}, "content": {"type": "string"}, "patch": {"type": "string"}}, "additionalProperties": False}},
    {"name": "get_diff", "description": "Return the Git diff for the project or one path.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "staged": {"type": "boolean"}}, "additionalProperties": False}},
]


def _call_tool(project_root: Path, name: str, arguments: dict[str, Any]) -> dict:
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    if name == "get_git_branch":
        if arguments:
            raise ValueError("get_git_branch takes no arguments")
        return git_branch(project_root)
    if name == "list_files":
        return list_files(project_root, **arguments)
    if name == "read_file":
        return read_file(project_root, **arguments)
    if name == "search_text":
        return search_text(project_root, **arguments)
    if name == "write_file":
        return write_file(project_root, **arguments)
    if name == "apply_patch":
        return apply_patch(project_root, **arguments)
    if name == "get_diff":
        return get_diff(project_root, **arguments)
    raise ValueError(f"unknown tool: {name}")


def serve(project_root: Path, input_stream=None, output_stream=None) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    for line in input_stream:
        if not line.strip():
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            method = request.get("method")
            request_id = request.get("id")
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "taskmanager-developer-assistant", "version": "0.1.0"}}
            elif method == "tools/list":
                result = {"tools": TOOL_DEFINITIONS}
            elif method == "tools/call":
                params = request.get("params") or {}
                data = _call_tool(project_root, params.get("name"), params.get("arguments") or {})
                result = {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}], "structuredContent": data, "isError": False}
            elif method == "notifications/initialized":
                continue
            else:
                raise ValueError(f"unsupported MCP method: {method}")
            output_stream.write(json.dumps(response(request_id, result=result), ensure_ascii=False) + "\n")
            output_stream.flush()
        except Exception as exc:  # protocol errors must remain machine-readable
            output_stream.write(json.dumps(response(request.get("id") if isinstance(request, dict) else None, error={"code": -32000, "message": str(exc)}), ensure_ascii=False) + "\n")
            output_stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if not root.is_dir():
        parser.error("project root does not exist")
    serve(root)


if __name__ == "__main__":
    main()

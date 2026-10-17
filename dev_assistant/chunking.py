"""Structure-aware Markdown and Go corpus loading and chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CHUNKING_VERSION = "3"


_GO_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        "vendor",
        "dist",
        "indexes",
        "generated",
        "gen",
        "__pycache__",
    }
)
_GO_DECLARATIONS = frozenset({"func", "type", "var", "const"})


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    title: str
    section: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    text: str
    chunking_strategy: str = "structural"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["heading_path"] = list(self.heading_path)
        return result


def markdown_sources(project_root: Path, include_optional: bool = True) -> list[Path]:
    """Return only the documented, in-scope Markdown corpus files."""

    candidates: list[Path] = []
    readme = project_root / "README.md"
    if readme.is_file():
        candidates.append(readme)
    docs = project_root / "docs"
    if docs.is_dir():
        candidates.extend(sorted(path for path in docs.rglob("*.md") if path.is_file()))
    optional = project_root / "TASKMANAGER_MCP.md"
    if include_optional and optional.is_file():
        candidates.append(optional)
    return candidates


def _is_generated_go(path: Path) -> bool:
    """Return whether *path* looks like a generated Go artifact.

    The standard Go convention is a ``Code generated ... DO NOT EDIT`` comment.
    Filename conventions are checked as well because generated files do not
    consistently carry that comment (and generated files are not useful RAG
    context even when they do).
    """

    name = path.name.lower()
    if name in {"gen.go", "generated.go"} or name.endswith(
        ("_gen.go", "_generated.go", ".gen.go", ".generated.go")
    ):
        return True
    try:
        prefix = path.read_text(encoding="utf-8")[:4096]
    except (OSError, UnicodeDecodeError):
        return False
    return bool(re.search(r"^\s*//\s*code generated .* do not edit\.?\s*$", prefix, re.IGNORECASE | re.MULTILINE))


def go_sources(project_root: Path) -> list[Path]:
    """Return in-scope Go files from a target project.

    Tests (``*_test.go``) intentionally remain in the corpus: they often
    document the expected behavior better than comments do.  Build output,
    vendored code, indexes, and generated artifacts are excluded instead.
    """

    sources: list[Path] = []
    for path in project_root.rglob("*.go"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(project_root).parts
        if any(part.lower() in _GO_EXCLUDED_DIRS for part in relative_parts[:-1]):
            continue
        if _is_generated_go(path):
            continue
        sources.append(path)
    return sorted(sources, key=lambda item: item.relative_to(project_root).as_posix())


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_markdown(path: Path, project_root: Path, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    """Split Markdown by heading boundaries, then cap large sections by words."""

    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("max_words must be positive and overlap must be smaller")
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    title = path.stem
    sections: list[tuple[tuple[str, ...], int, int, list[str]]] = []
    heading_stack: list[str] = []
    section_start = 1
    body: list[str] = []
    fence_marker: str | None = None

    def flush(end_line: int) -> None:
        nonlocal body, section_start
        if any(line.strip() and not re.match(r"^#{1,6}\s+", line) for line in body):
            sections.append((tuple(heading_stack), section_start, end_line, body[:]))
        body = []

    for number, line in enumerate(lines, start=1):
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif fence_marker == marker:
                fence_marker = None
            body.append(line)
            continue
        match = None if fence_marker is not None else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            flush(number - 1)
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1:
                title = heading
            heading_stack = heading_stack[: level - 1] + [heading]
            section_start = number
            body = [line]
        else:
            body.append(line)
    flush(len(lines))

    relative = path.relative_to(project_root).as_posix()
    chunks: list[Chunk] = []
    for heading_path, start, end, section_lines in sections:
        words = _words("\n".join(section_lines))
        if not words:
            continue
        step = max_words - overlap
        for offset in range(0, len(words), step):
            selected = words[offset : offset + max_words]
            if not selected:
                break
            # Keep the complete heading path in the embedded text.  A section
            # such as README's "Системная схема" is nested under the more
            # useful parent heading "Архитектура"; without this context an
            # architecture query cannot match the authoritative section.
            heading_context = " > ".join(heading_path)
            text = " ".join([heading_context, *selected]) if heading_context else " ".join(selected)
            suffix = f"{start}:{end}:{offset}"
            digest = hashlib.sha1(f"{relative}:{suffix}:{text}".encode(), usedforsecurity=False).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"{relative}#chunk-{digest}",
                    source=relative,
                    title=title,
                    section=heading_path[-1] if heading_path else title,
                    heading_path=heading_path,
                    start_line=start,
                    end_line=end,
                    text=text,
                )
            )
            if offset + max_words >= len(words):
                break
    return chunks


def _go_tokens(raw: str) -> list[tuple[str, int, int, tuple[int, int, int]]]:
    """Tokenize enough Go syntax to find declarations at file scope.

    This deliberately is not a Go parser.  It only tracks comments, literals,
    and delimiter depth, which is sufficient to distinguish a top-level
    declaration from a ``func`` nested in a function literal or interface.
    """

    tokens: list[tuple[str, int, int, tuple[int, int, int]]] = []
    index = 0
    brace_depth = paren_depth = bracket_depth = 0
    length = len(raw)
    while index < length:
        char = raw[index]
        if char.isspace():
            index += 1
            continue
        if raw.startswith("//", index):
            newline = raw.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if raw.startswith("/*", index):
            end = raw.find("*/", index + 2)
            index = length if end < 0 else end + 2
            continue
        if char in {'"', "'", "`"}:
            quote = char
            index += 1
            while index < length:
                if quote != "`" and raw[index] == "\\":
                    index += 2
                    continue
                if raw[index] == quote:
                    index += 1
                    break
                index += 1
            continue
        if char.isalpha() or char == "_":
            start = index
            index += 1
            while index < length and (raw[index].isalnum() or raw[index] == "_"):
                index += 1
            tokens.append((raw[start:index], start, index, (brace_depth, paren_depth, bracket_depth)))
            continue
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        index += 1
    return tokens


def _go_declaration_starts(raw: str) -> list[tuple[str, int]]:
    starts: list[tuple[str, int]] = []
    tokens = _go_tokens(raw)
    for token_index, (token, start, _end, depths) in enumerate(tokens):
        if token not in _GO_DECLARATIONS or depths != (0, 0, 0):
            continue
        previous = tokens[token_index - 1] if token_index else None
        line_start = raw.rfind("\n", 0, start) + 1
        line_prefix = raw[line_start:start].strip()
        # Declarations conventionally begin a logical line.  Also accept a
        # declaration after a semicolon/closing import group, while avoiding
        # anonymous ``func`` values in a top-level var initializer.
        if line_prefix and (previous is None or previous[0] not in {"}", ")"}):
            continue
        starts.append((token, _go_doc_start(raw, start)))
    return starts


def _go_doc_start(raw: str, declaration_start: int) -> int:
    """Return the start of comments immediately preceding a declaration."""

    cursor = raw.rfind("\n", 0, declaration_start) + 1
    while cursor > 0:
        previous_end = cursor - 1
        previous_start = raw.rfind("\n", 0, previous_end) + 1
        line = raw[previous_start:previous_end].strip()
        if not line:
            break
        if line.startswith("//"):
            cursor = previous_start
            continue
        if line.endswith("*/"):
            block_end = raw.rfind("*/", 0, cursor)
            block_start = raw.rfind("/*", 0, block_end)
            if block_start >= 0:
                opening_line_start = raw.rfind("\n", 0, block_start) + 1
                # Do not absorb an inline block comment from a previous
                # declaration; only a comment occupying its own lines is a
                # declaration doc comment.
                if not raw[opening_line_start:block_start].strip():
                    cursor = opening_line_start
                    continue
        break
    return cursor


def _go_package_import_context(raw: str) -> str:
    """Build a compact package/import preamble for every code embedding."""

    package = re.search(r"(?m)^\s*package\s+([A-Za-z_]\w*)\s*$", raw)
    lines = [f"package {package.group(1)}" if package else "package (unknown)"]
    imports: list[str] = []
    for match in re.finditer(r"(?ms)^\s*import\s+(\([^)]*\)|[^\n]+)", raw):
        imports.extend(re.findall(r'"([^"\n]+)"', match.group(1)))
    if imports:
        lines.append("imports: " + ", ".join(dict.fromkeys(imports)))
    return "\n".join(lines)


def _go_declaration_section(kind: str, text: str) -> str:
    declaration = re.search(rf"(?m)^\s*{kind}\b", text)
    if declaration:
        text = text[declaration.start() :]
    if kind == "func":
        match = re.match(r"func\s*(\([^)]*\)\s*)?([A-Za-z_]\w*)", text, re.DOTALL)
        if match:
            name = match.group(2)
            receiver = (match.group(1) or "").strip()
            if receiver:
                return f"method {receiver}.{name}"
            return f"func {name}"
    match = re.match(rf"{kind}\s+([A-Za-z_]\w*)", text)
    if match:
        return f"{kind} {match.group(1)}"
    return f"{kind} (group)"


def chunk_go(path: Path, project_root: Path) -> list[Chunk]:
    """Split a Go file into top-level declaration chunks."""

    raw = path.read_text(encoding="utf-8")
    relative = path.relative_to(project_root).as_posix()
    starts = _go_declaration_starts(raw)
    package_import_context = _go_package_import_context(raw)
    chunks: list[Chunk] = []
    for index, (kind, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(raw)
        declaration_source = raw[start:end].rstrip()
        declaration = declaration_source.strip()
        if not declaration:
            continue
        section = _go_declaration_section(kind, declaration)
        title = path.stem
        heading_path = (title, section)
        text = f"{title} > {section}\n{package_import_context}\n{declaration}"
        start_line = raw.count("\n", 0, start) + 1
        end_line = raw.count("\n", 0, start + len(declaration_source)) + 1
        suffix = f"{start_line}:{end_line}:{kind}:{text}"
        digest = hashlib.sha1(f"{relative}:{suffix}".encode(), usedforsecurity=False).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=f"{relative}#chunk-{digest}",
                source=relative,
                title=title,
                section=section,
                heading_path=heading_path,
                start_line=start_line,
                end_line=end_line,
                text=text,
                chunking_strategy="go-structural",
            )
        )
    # Keep a package/import-only file useful to retrieval, while normal files
    # remain strictly declaration-oriented.
    if not chunks and raw.strip():
        title = path.stem
        text = f"{title} > package\n{raw.strip()}"
        digest = hashlib.sha1(f"{relative}:package:{text}".encode(), usedforsecurity=False).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=f"{relative}#chunk-{digest}",
                source=relative,
                title=title,
                section="package",
                heading_path=(title, "package"),
                start_line=1,
                end_line=raw.count("\n") + 1,
                text=text,
                chunking_strategy="go-structural",
            )
        )
    return chunks


def build_corpus(project_root: Path, include_optional: bool = True) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source in markdown_sources(project_root, include_optional=include_optional):
        chunks.extend(chunk_markdown(source, project_root))
    for source in go_sources(project_root):
        chunks.extend(chunk_go(source, project_root))
    if not chunks:
        raise FileNotFoundError("No README.md, docs/**/*.md, or Go source files found")
    return chunks


def corpus_fingerprint(project_root: Path, include_optional: bool = True) -> str:
    """Return a stable fingerprint for the exact in-scope corpus."""

    digest = hashlib.sha256()
    sources = [*markdown_sources(project_root, include_optional=include_optional), *go_sources(project_root)]
    for path in sources:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

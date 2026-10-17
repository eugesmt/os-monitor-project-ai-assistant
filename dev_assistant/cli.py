"""CLI entry point for indexing, /help conversations, and local review."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .chunking import CHUNKING_VERSION, build_corpus, corpus_fingerprint
from .deepseek import answer
from .env import load_local_env
from .mcp_client import MCPClient
from .ollama import embed
from .storage import RETRIEVAL_VERSION, build_index, write_chunks_jsonl


def assistant_root() -> Path:
    """Return the installed standalone assistant's project directory."""
    return Path(__file__).resolve().parents[1]


def default_project_root() -> Path:
    """Resolve the target project without reading its .env file.

    The released assistant lives beside ``taskmanager-ai``.  The fallback to
    the parent itself keeps this staging checkout usable while it is developed
    inside the target repository.
    """
    configured = os.getenv("TASKMANAGER_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser()
    package_root = assistant_root()
    sibling = package_root.parent / "taskmanager-ai"
    if sibling.is_dir():
        return sibling
    if package_root.name == ".uv-assistant-staging" and package_root.parent.is_dir():
        return package_root.parent
    return sibling


def index_paths(output_root: Path) -> tuple[Path, Path]:
    index_dir = output_root / "indexes" / "structural"
    return index_dir / "index.faiss", index_dir / "chunks.sqlite"


def index_manifest_path(output_root: Path) -> Path:
    return output_root / "indexes" / "structural" / "index-manifest.json"


def index_is_ready(output_root: Path, project_root: Path | None = None) -> bool:
    if not all(path.is_file() for path in index_paths(output_root)):
        return False
    try:
        manifest = json.loads(index_manifest_path(output_root).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if manifest.get("chunking_version") != CHUNKING_VERSION or manifest.get("retrieval_version") != RETRIEVAL_VERSION:
        return False
    return project_root is None or manifest.get("corpus_fingerprint") == corpus_fingerprint(project_root)


def ensure_index(project_root: Path, output_root: Path, reindex: bool = False) -> None:
    if reindex or not index_is_ready(output_root, project_root):
        reason = "forced rebuild" if reindex else "index not found"
        print(f"Building RAG index ({reason})...", file=sys.stderr)
        build_index_command(project_root, output_root)


def build_index_command(project_root: Path, output_root: Path) -> None:
    chunks = build_corpus(project_root)
    jsonl = output_root / "data" / "processed" / "chunks.jsonl"
    index = output_root / "indexes" / "structural" / "index.faiss"
    database = output_root / "indexes" / "structural" / "chunks.sqlite"
    write_chunks_jsonl(chunks, jsonl)
    embeddings = [embed(chunk.text) for chunk in chunks]
    build_index([chunk.as_dict() for chunk in chunks], embeddings, index, database)
    index_manifest_path(output_root).write_text(
        json.dumps(
            {
                "chunking_version": CHUNKING_VERSION,
                "retrieval_version": RETRIEVAL_VERSION,
                "corpus_fingerprint": corpus_fingerprint(project_root),
                "chunk_count": len(chunks),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Indexed {len(chunks)} chunks from documentation and Go source (plus optional TASKMANAGER_MCP.md)")


def help_command(question: str, project_root: Path, output_root: Path) -> None:
    top_k = int(os.getenv("RAG_TOP_K", "5"))
    index_path, database_path = index_paths(output_root)
    if not index_is_ready(output_root, project_root):
        raise RuntimeError("RAG index missing or stale; run the assistant without arguments first")
    from .storage import search
    results = search(index_path, database_path, embed(question), top_k=top_k, query_text=question)
    with MCPClient(project_root) as mcp:
        branch = mcp.get_git_branch()
    response = answer(question, results, branch)
    print(response)
    print(f"\nGit branch: {branch.get('branch') or '(detached HEAD)'}")
    print("\nSources:")
    for item in results:
        print(f"- {item['source']}:{item['start_line']} — {item['section']} (score={item['score']:.3f})")


def review_command(
    project_root: Path,
    output_root: Path,
    base: str = "develop",
    head: str = "HEAD",
    working_tree: bool = False,
    output: Path | None = None,
    reindex: bool = False,
) -> str:
    """Run a local code review and optionally save its report.

    ``run_review`` owns the diff and RAG-aware review orchestration.  The CLI
    makes sure the index corresponds to the current target project first, so a
    review never silently uses stale retrieval context.
    """

    ensure_index(project_root, output_root, reindex=reindex)
    # Import lazily so commands such as ``--help`` and ``build-index`` remain
    # usable while the review implementation is optional during development.
    from .review import run_review

    result = run_review(
        project_root=project_root,
        output_root=output_root,
        base=base,
        head=head,
        working_tree=working_tree,
    )
    print(result)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result, encoding="utf-8")
    return result


def main() -> None:
    load_local_env(assistant_root())
    parser = argparse.ArgumentParser(description="Standalone RAG developer assistant")
    parser.add_argument("--project-root", type=Path, dest="project_root", default=None)
    parser.add_argument("--reindex", action="store_true", help="rebuild the RAG index before starting chat")
    subparsers = parser.add_subparsers(dest="command", required=False)
    build_parser = subparsers.add_parser("build-index")
    build_parser.add_argument("--project-root", type=Path, dest="project_root", default=argparse.SUPPRESS)
    help_parser = subparsers.add_parser("help")
    help_parser.add_argument("question", nargs="+")
    help_parser.add_argument("--project-root", type=Path, dest="project_root", default=argparse.SUPPRESS)
    help_parser.add_argument("--reindex", action="store_true", default=argparse.SUPPRESS, help="rebuild the RAG index first")
    chat_parser = subparsers.add_parser("chat", help="read /help commands interactively")
    chat_parser.add_argument("--project-root", type=Path, dest="project_root", default=argparse.SUPPRESS)
    chat_parser.add_argument("--reindex", action="store_true", default=argparse.SUPPRESS, help="rebuild the RAG index first")
    review_parser = subparsers.add_parser("review", help="review local Git changes with RAG context")
    review_parser.add_argument("--project-root", type=Path, dest="project_root", default=argparse.SUPPRESS)
    review_parser.add_argument("--base", help="base Git ref (default: develop)")
    review_parser.add_argument("--head", help="head Git ref (default: HEAD)")
    review_parser.add_argument("--working-tree", action="store_true", help="review uncommitted working-tree changes")
    review_parser.add_argument("--output", type=Path, help="also write the review report as UTF-8 text")
    review_parser.add_argument("--reindex", action="store_true", default=argparse.SUPPRESS, help="rebuild the RAG index first")
    args = parser.parse_args()
    root = (args.project_root or default_project_root()).expanduser().resolve()
    output_root = assistant_root()
    if args.command == "build-index":
        build_index_command(root, output_root)
    elif args.command == "help":
        ensure_index(root, output_root, reindex=args.reindex)
        help_command(" ".join(args.question), root, output_root)
    elif args.command == "review":
        if args.working_tree and (args.base is not None or args.head is not None):
            parser.error("review --working-tree cannot be combined with --base or --head")
        review_command(
            root,
            output_root,
            base=args.base or "develop",
            head=args.head or "HEAD",
            working_tree=args.working_tree,
            output=args.output,
            reindex=getattr(args, "reindex", False),
        )
    else:
        ensure_index(root, output_root, reindex=args.reindex)
        print("Введите /help <вопрос>. Ctrl-D завершает работу.")
        for line in sys.stdin:
            command = line.strip()
            if not command:
                continue
            if command == "/help":
                print("Использование: /help <вопрос>")
                continue
            if command.startswith("/help "):
                try:
                    help_command(command[6:].strip(), root, output_root)
                except Exception as exc:
                    print(f"Ошибка: {exc}")
            else:
                print("Поддерживается команда /help <вопрос>")

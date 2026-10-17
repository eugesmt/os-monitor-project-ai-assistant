"""FAISS + SQLite persistence and retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .chunking import Chunk


RETRIEVAL_VERSION = "3"
_OVERVIEW_PREFIXES = ("архитектур", "структур", "устро", "системн", "компонент")
_CONTEXT_PREFIXES = ("контекст", "памят", "memory", "compiler", "history", "state")
_STOPWORDS = {
    "а", "в", "во", "для", "где", "как", "какая", "какие", "какой", "к", "на",
    "о", "об", "по", "про", "с", "со", "что", "это", "у", "и", "или", "из",
    "проект", "проекта", "проекту", "the", "a", "an", "of", "the", "how", "what",
}


def _numerical_dependencies():
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before building or searching the index") from exc
    return faiss, np


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")


def read_chunks_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_index(records: list[dict], embeddings: list[list[float]], index_path: Path, database_path: Path) -> None:
    faiss, np = _numerical_dependencies()
    if not records or len(records) != len(embeddings):
        raise ValueError("records and embeddings must be non-empty and aligned")
    vectors = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    ids = np.arange(len(records), dtype="int64")
    index.add_with_ids(vectors, ids)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute(
            """CREATE TABLE chunks (
                vector_id INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL, title TEXT NOT NULL, section TEXT NOT NULL,
                heading_path TEXT NOT NULL, start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL, text TEXT NOT NULL)"""
        )
        connection.executemany(
            """INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(
                i, item["chunk_id"], item["source"], item["title"], item["section"],
                json.dumps(item["heading_path"], ensure_ascii=False), item["start_line"],
                item["end_line"], item["text"],
            ) for i, item in enumerate(records)],
        )
        connection.commit()


def _token_keys(text: str) -> set[str]:
    tokens = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
    keys: set[str] = set()
    for token in tokens:
        if token in _STOPWORDS:
            continue
        keys.add(token)
        if len(token) >= 7:
            keys.add(token[:7])
    return keys


def _overview_query(query: str) -> bool:
    query_keys = _token_keys(query)
    broad = any(token.startswith(prefix) for token in query_keys for prefix in _OVERVIEW_PREFIXES)
    raw_tokens = re.findall(r"[^\W_]+", query.lower(), flags=re.UNICODE)
    project_scoped = any(token.startswith(("проект", "прилож", "систем")) for token in raw_tokens)
    context_scoped = any(token.startswith(prefix) for token in query_keys for prefix in _CONTEXT_PREFIXES)
    # The project README is authoritative for questions about the Task Manager
    # itself.  Context-engineering questions remain free to retrieve the long
    # design documents without this source prior.
    return broad and (project_scoped or not context_scoped)


def _rerank_score(query: str, item: dict, vector_score: float, overview: bool) -> float:
    query_keys = _token_keys(query)
    if not query_keys:
        return vector_score
    heading_text = " ".join([item["title"], item["section"], *item["heading_path"]])
    heading_keys = _token_keys(heading_text)
    body_keys = _token_keys(f"{item['source']} {item['text']}")
    heading_overlap = len(query_keys & heading_keys) / len(query_keys)
    body_overlap = len(query_keys & body_keys) / len(query_keys)
    lexical = 0.7 * heading_overlap + 0.3 * body_overlap
    score = vector_score + 0.22 * lexical + 0.10 * heading_overlap
    if overview:
        if item["source"] == "README.md":
            score += 0.14
        elif item["source"] == "TASKMANAGER_MCP.md":
            score += 0.09
        elif item["source"].startswith("docs/"):
            score -= 0.25
        if item["source"] == "TASKMANAGER_MCP.md" and item["section"].casefold() == "architecture":
            score += 0.15
        if item["section"].casefold() in {"содержание", "table of contents"}:
            score -= 0.14
    return score


def search(index_path: Path, database_path: Path, query_vector: list[float], top_k: int = 5, query_text: str | None = None) -> list[dict]:
    faiss, np = _numerical_dependencies()
    index = faiss.read_index(str(index_path))
    vector = np.asarray([query_vector], dtype="float32")
    faiss.normalize_L2(vector)
    candidate_k = min(index.ntotal, max(top_k * 12, 50)) if query_text else top_k
    scores, ids = index.search(vector, candidate_k)
    overview = bool(query_text and _overview_query(query_text))
    with sqlite3.connect(database_path) as connection:
        candidates: list[dict] = []
        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id < 0:
                continue
            row = connection.execute("SELECT chunk_id, source, title, section, heading_path, start_line, end_line, text FROM chunks WHERE vector_id = ?", (int(vector_id),)).fetchone()
            if row is None:
                continue
            item = {"chunk_id": row[0], "source": row[1], "title": row[2], "section": row[3], "heading_path": json.loads(row[4]), "start_line": row[5], "end_line": row[6], "text": row[7], "score": float(score)}
            item["vector_score"] = float(score)
            if query_text:
                item["score"] = _rerank_score(query_text, item, float(score), overview)
            candidates.append(item)
        if not query_text:
            return candidates

        candidates.sort(key=lambda item: item["score"], reverse=True)
        results: list[dict] = []
        seen_sections: set[tuple[str, str]] = set()
        # Adjacent overlapping chunks from one section add little evidence.
        # Prefer distinct sections so a broad question receives an overview,
        # runtime, and MCP architecture evidence when all are available.
        for item in candidates:
            section_key = (item["source"], item["section"])
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)
            results.append(item)
            if len(results) >= top_k:
                break
        return results

"""Ollama embedding client using the qwen3 embedding model."""

from __future__ import annotations

import os

def embed(text: str, *, base_url: str | None = None, model: str | None = None) -> list[float]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before using Ollama embeddings") from exc
    url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/") + "/api/embed"
    payload = {"model": model or os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"), "input": text}
    response = httpx.post(url, json=payload, timeout=120.0)
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings")
    if not embeddings or not isinstance(embeddings[0], list):
        raise ValueError("Ollama returned no embedding")
    return [float(value) for value in embeddings[0]]

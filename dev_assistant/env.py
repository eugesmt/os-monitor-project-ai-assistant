"""Safe loading of the assistant's own optional dotenv file."""

from __future__ import annotations

import os
from pathlib import Path


KNOWN_ENV_KEYS = frozenset(
    {
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_EMBEDDING_MODEL",
        "TASKMANAGER_PROJECT_ROOT",
        "RAG_TOP_K",
    }
)


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_local_env(assistant_root: Path) -> None:
    """Load known keys from the assistant root's .env, preserving env priority."""
    env_path = assistant_root / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key not in KNOWN_ENV_KEYS or key in os.environ:
            continue
        os.environ[key] = _parse_value(raw_value)

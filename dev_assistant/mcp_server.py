"""Small line-delimited JSON-RPC MCP subset for local stdio use.

Supported methods: initialize, tools/list and tools/call.  The only tool is
get_git_branch and its project root is fixed when the server starts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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
    commit = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5)
    if commit.returncode != 0:
        raise RuntimeError("unable to read Git HEAD")
    return {"branch": None, "commit": commit.stdout.strip(), "detached": True}


def response(request_id, result=None, error=None) -> dict:
    payload = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def serve(project_root: Path, input_stream=None, output_stream=None) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    for line in input_stream:
        if not line.strip():
            continue
        request = {}
        try:
            request = json.loads(line)
            method = request.get("method")
            request_id = request.get("id")
            if method == "initialize":
                result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "taskmanager-developer-assistant", "version": "0.1.0"}}
            elif method == "tools/list":
                result = {"tools": [{"name": "get_git_branch", "description": "Return the current branch of the configured project root.", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}}]}
            elif method == "tools/call":
                params = request.get("params") or {}
                if params.get("name") != "get_git_branch" or params.get("arguments") not in (None, {}):
                    raise ValueError("only get_git_branch with empty arguments is supported")
                data = git_branch(project_root)
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

"""MCP stdio client used by the assistant itself."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


class MCPClient:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.process: subprocess.Popen[str] | None = None
        self.request_id = 0

    def __enter__(self):
        self.process = subprocess.Popen([sys.executable, "-m", "dev_assistant.mcp_server", "--project-root", str(self.project_root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.call("initialize", {})
        self.call("notifications/initialized", {}, expect_response=False)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()
            if self.process.stderr:
                self.process.stderr.close()

    def call(self, method: str, params: dict, expect_response: bool = True) -> dict | None:
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("MCP client is not started")
        self.request_id += 1
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if expect_response:
            payload["id"] = self.request_id
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        if not expect_response:
            return None
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"MCP server closed stdout: {error}")
        message = json.loads(line)
        if "error" in message:
            raise RuntimeError(message["error"].get("message", "MCP error"))
        return message["result"]

    def get_git_branch(self) -> dict:
        result = self.call("tools/call", {"name": "get_git_branch", "arguments": {}})
        assert result is not None
        return result["structuredContent"]

    def list_tools(self) -> list[dict]:
        """Return the tools advertised by the configured MCP server."""
        result = self.call("tools/list", {})
        assert result is not None
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Call one advertised MCP tool and return its result envelope."""
        result = self.call("tools/call", {"name": name, "arguments": arguments or {}})
        assert result is not None
        return result

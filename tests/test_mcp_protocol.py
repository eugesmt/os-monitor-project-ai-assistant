import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MCPProtocolTests(unittest.TestCase):
    def test_initialize_list_and_real_git_branch_call(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "test-branch", str(project)], check=True)
            process = subprocess.Popen([sys.executable, "-m", "dev_assistant.mcp_server", "--project-root", str(project)], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

            def call(request):
                assert process.stdin and process.stdout
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                return json.loads(process.stdout.readline())

            self.assertEqual(call({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})["result"]["serverInfo"]["name"], "taskmanager-developer-assistant")
            tools = call({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})["result"]["tools"]
            self.assertEqual(
                [tool["name"] for tool in tools],
                ["get_git_branch", "list_files", "read_file", "search_text", "write_file", "apply_patch", "get_diff"],
            )
            result = call({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_git_branch", "arguments": {}}})
            self.assertEqual(result["result"]["structuredContent"]["branch"], "test-branch")
            process.terminate()
            process.wait(timeout=5)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()


if __name__ == "__main__":
    unittest.main()

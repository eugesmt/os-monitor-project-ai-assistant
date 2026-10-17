import tempfile
import subprocess
import unittest
import copy
import json
from pathlib import Path

from dev_assistant.agent_loop import AgentLimits, FileAgent
from dev_assistant.mcp_client import MCPClient


class FakeMCP:
    def __init__(self):
        self.calls = []
        self.tools = [
            {"name": "list_files", "description": "list files", "inputSchema": {"type": "object"}},
            {"name": "read_file", "description": "read file", "inputSchema": {"type": "object"}},
            {"name": "search_text", "description": "search", "inputSchema": {"type": "object"}},
            {"name": "write_file", "description": "write", "inputSchema": {"type": "object"}},
            {"name": "get_diff", "description": "diff", "inputSchema": {"type": "object"}},
        ]

    def list_tools(self):
        return self.tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "list_files":
            return {"files": ["internal/api.go", "docs/api.md"]}
        if name == "search_text":
            return {"matches": [{"path": "internal/api.go", "line": 10}, {"path": "internal/client.go", "line": 4}]}
        if name == "read_file":
            return {"path": arguments["path"], "text": "package api\n"}
        if name == "write_file":
            return {"path": arguments["path"], "written": True}
        if name == "get_diff":
            return {"diff": "diff --git a/docs/api.md b/docs/api.md\n+updated\n"}
        raise AssertionError(name)


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_search_calls_tools_sequentially_and_records_files(self):
        mcp = FakeMCP()
        responses = iter(
            [
                {"tool_calls": [{"id": "1", "function": {"name": "search_text", "arguments": '{"query":"/api/ai/ask"}'}}]},
                {"content": "Найдено два использования API."},
            ]
        )

        report = FileAgent(self.root, mcp_client=mcp, model=lambda messages, tools: next(responses)).run(
            "Найди использования /api/ai/ask", scenario="search"
        )

        self.assertEqual([name for name, _ in mcp.calls], ["search_text"])
        self.assertEqual(report.read_files, ["internal/api.go", "internal/client.go"])
        self.assertEqual(report.changed_files, [])
        self.assertIn("два использования", report.answer)
        self.assertEqual(report.tool_calls, 1)

    def test_documentation_write_is_followed_by_mcp_diff(self):
        mcp = FakeMCP()
        responses = iter(
            [
                {"tool_calls": [{"id": "1", "function": {"name": "read_file", "arguments": '{"path":"internal/api.go"}'}}]},
                {"tool_calls": [{"id": "2", "function": {"name": "write_file", "arguments": '{"path":"docs/api.md","content":"# API\\n"}'}}]},
                {"content": "Документация обновлена."},
            ]
        )
        report = FileAgent(self.root, mcp_client=mcp, model=lambda messages, tools: next(responses)).run(
            "Обнови документацию API", scenario="documentation"
        )

        self.assertEqual([name for name, _ in mcp.calls], ["read_file", "write_file", "get_diff"])
        self.assertEqual(report.read_files, ["internal/api.go"])
        self.assertEqual(report.changed_files, ["docs/api.md"])
        self.assertIn("updated", report.diff)

    def test_fallback_requests_explicit_diff_for_new_file_when_global_diff_empty(self):
        class PathAwareMCP(FakeMCP):
            def call_tool(self, name, arguments):
                self.calls.append((name, arguments))
                if name == "write_file":
                    return {"path": arguments["path"], "written": True}
                if name == "get_diff":
                    if arguments.get("path"):
                        return {"diff": "diff --no-index /dev/null docs/new.md\n+new documentation\n"}
                    return {"diff": ""}
                return super().call_tool(name, arguments)

        mcp = PathAwareMCP()
        responses = iter(
            [
                {"tool_calls": [{"function": {"name": "write_file", "arguments": '{"path":"docs/new.md","content":"new documentation\\n"}'}}]},
                {"content": "Файл документации создан."},
            ]
        )
        report = FileAgent(self.root, mcp_client=mcp, model=lambda messages, tools: next(responses)).run(
            "Создай документацию", scenario="documentation"
        )

        self.assertEqual(mcp.calls, [("write_file", {"path": "docs/new.md", "content": "new documentation\n"}), ("get_diff", {"path": "docs/new.md"})])
        self.assertIn("new documentation", report.diff)

    def test_shell_tool_is_rejected_even_if_model_requests_it(self):
        mcp = FakeMCP()
        responses = iter(
            [
                {"tool_calls": [{"id": "1", "function": {"name": "run_shell", "arguments": '{"command":"rm -rf /"}'}}]},
                {"content": "Не удалось выполнить."},
            ]
        )
        report = FileAgent(self.root, mcp_client=mcp, model=lambda messages, tools: next(responses)).run("Проверь проект")
        self.assertEqual(mcp.calls, [])
        self.assertTrue(any("не объявлен" in error for error in report.errors))

    def test_search_scenario_is_read_only_even_when_model_requests_write(self):
        mcp = FakeMCP()
        advertised = []
        responses = iter(
            [
                {"tool_calls": [{"function": {"name": "write_file", "arguments": '{"path":"docs/api.md","content":"bad"}'}}]},
                {"content": "Запись запрещена."},
            ]
        )

        def model(messages, tools):
            advertised.extend(tool["function"]["name"] for tool in tools)
            return next(responses)

        report = FileAgent(self.root, mcp_client=mcp, model=model).run(
            "Найди использования API", scenario="search"
        )

        self.assertNotIn("write_file", advertised)
        self.assertEqual(mcp.calls, [])
        self.assertEqual(report.changed_files, [])
        self.assertTrue(any("только для чтения" in error for error in report.errors))

    def test_iteration_limit_returns_report_without_network(self):
        mcp = FakeMCP()
        model = lambda messages, tools: {"tool_calls": [{"function": {"name": "list_files", "arguments": "{}"}}]}
        report = FileAgent(self.root, mcp_client=mcp, model=model, limits=AgentLimits(max_iterations=2)).run("Найди файлы")
        self.assertEqual(report.iterations, 2)
        self.assertTrue(any("лимит итераций" in error for error in report.errors))

    def test_read_budget_is_cumulative_and_stops_next_read(self):
        mcp = FakeMCP()
        responses = iter(
            [
                {"tool_calls": [{"function": {"name": "read_file", "arguments": '{"path":"a.go"}'}}]},
                {"tool_calls": [{"function": {"name": "read_file", "arguments": '{"path":"b.go"}'}}]},
                {"content": "готово"},
            ]
        )
        report = FileAgent(
            self.root,
            mcp_client=mcp,
            model=lambda messages, tools: next(responses),
            limits=AgentLimits(max_read_chars=20),
        ).run("Прочитай два файла")

        # The first serialized result consumes the complete budget; the second
        # request is reported to the model but never reaches MCP.
        self.assertEqual([name for name, _ in mcp.calls], ["read_file"])
        self.assertEqual(report.read_chars, 20)
        self.assertTrue(any("лимит чтения" in error for error in report.errors))
        self.assertLessEqual(len(next(item["result"] for item in report.trace if item["name"] == "read_file")), 20)

    def test_large_tool_rounds_are_compacted_atomically_and_finish_after_budget(self):
        class LargeMCP(FakeMCP):
            def call_tool(self, name, arguments):
                self.calls.append((name, arguments))
                if name == "list_files":
                    return {"path": ".", "files": [f"pkg/file-{i}.go" for i in range(5000)], "entries": [], "truncated": False}
                if name == "read_file":
                    return {"path": arguments["path"], "content": "x" * 20_000}
                return super().call_tool(name, arguments)

        mcp = LargeMCP()
        snapshots = []
        responses = iter(
            [
                {"tool_calls": [{"function": {"name": "list_files", "arguments": "{}"}}]},
                {"tool_calls": [{"function": {"name": "read_file", "arguments": '{"path":"pkg/file-1.go"}'}}]},
                {"tool_calls": [{"function": {"name": "read_file", "arguments": '{"path":"pkg/file-2.go"}'}}]},
                {"content": "Лимит чтения достигнут; итог сформирован по доступному контексту."},
            ]
        )

        def model(messages, tools):
            snapshot = copy.deepcopy(messages)
            snapshots.append(snapshot)
            self.assertLessEqual(len(json.dumps(snapshot, ensure_ascii=False)), 5_000)
            for index, message in enumerate(snapshot):
                if message.get("role") != "assistant" or not message.get("tool_calls"):
                    continue
                expected = [call["id"] for call in message["tool_calls"]]
                following = snapshot[index + 1 : index + 1 + len(expected)]
                self.assertEqual([item.get("role") for item in following], ["tool"] * len(expected))
                self.assertEqual([item.get("tool_call_id") for item in following], expected)
            return next(responses)

        report = FileAgent(
            self.root,
            mcp_client=mcp,
            model=model,
            limits=AgentLimits(max_iterations=6, max_read_chars=5_000, max_prompt_chars=5_000),
        ).run("Проанализируй проект")

        self.assertEqual(report.answer, "Лимит чтения достигнут; итог сформирован по доступному контексту.")
        self.assertNotIn("pkg/file-5000.go", report.read_files)
        self.assertEqual(report.read_files, ["pkg/file-1.go"])
        self.assertTrue(any("лимит чтения" in error for error in report.errors))
        self.assertEqual([name for name, _ in mcp.calls], ["list_files", "read_file"])

    def test_missing_model_tool_id_matches_following_tool_message(self):
        mcp = FakeMCP()
        captured = []
        responses = iter(
            [
                {"tool_calls": [{"function": {"name": "search_text", "arguments": '{"query":"API"}'}}]},
                {"content": "готово"},
            ]
        )

        def model(messages, tools):
            captured.append(messages)
            return next(responses)

        FileAgent(self.root, mcp_client=mcp, model=model).run("Найди API")
        assistant = next(message for message in captured[1] if message["role"] == "assistant")
        tool = next(message for message in captured[1] if message["role"] == "tool")
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], tool["tool_call_id"])

    def test_real_mcp_client_envelope_records_paths_and_diff(self):
        project = self.root / "project"
        project.mkdir()
        (project / "src").mkdir()
        (project / "docs").mkdir()
        (project / "src" / "api.go").write_text("package api\n", encoding="utf-8")
        docs = project / "docs" / "api.md"
        docs.write_text("# Old API\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", "-b", "main", str(project)], check=True)
        subprocess.run(["git", "-C", str(project), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(project), "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-qm", "initial"],
            check=True,
        )
        responses = iter(
            [
                {"tool_calls": [{"function": {"name": "read_file", "arguments": '{"path":"src/api.go"}'}}]},
                {"tool_calls": [{"function": {"name": "write_file", "arguments": '{"path":"docs/api.md","content":"# New API\\n","overwrite":true}'}}]},
                {"content": "Документация обновлена."},
            ]
        )
        with MCPClient(project) as mcp:
            report = FileAgent(project, mcp_client=mcp, model=lambda messages, tools: next(responses)).run(
                "Обнови документацию API", scenario="documentation"
            )

        self.assertEqual(report.read_files, ["src/api.go"])
        self.assertEqual(report.changed_files, ["docs/api.md"])
        self.assertIn("New API", report.diff)


if __name__ == "__main__":
    unittest.main()

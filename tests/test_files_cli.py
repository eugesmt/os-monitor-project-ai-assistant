import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dev_assistant import cli


class FilesCLITests(unittest.TestCase):
    def test_files_command_renders_agent_report_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "taskmanager-ai"
            project.mkdir()
            class Report:
                def render(self):
                    return "Готово\nПрочитанные файлы: README.md\nИзменённые файлы: docs/api.md\nИтоговый git diff:\n+ docs/api.md"

            report = Report()
            def fake_run_file_agent(goal, root):
                self.assertEqual(goal, "найди использования API и обнови docs")
                self.assertEqual(root, project)
                return report

            output = io.StringIO()
            with patch.object(cli, "run_file_agent", side_effect=fake_run_file_agent), contextlib.redirect_stdout(output):
                result = cli.files_command("найди использования API и обнови docs", project)

            self.assertIn("Прочитанные файлы: README.md", result)
            self.assertIn("Изменённые файлы: docs/api.md", output.getvalue())

    def test_main_accepts_files_goal_and_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            captured = {}

            def fake_files(goal, root):
                captured.update(goal=goal, root=root)
                return "ok"

            with patch.object(cli, "files_command", side_effect=fake_files), patch.object(
                sys,
                "argv",
                ["taskmanager-assistant", "files", "найди", "использования", "API", "--project-root", str(project)],
            ):
                cli.main()

            self.assertEqual(captured, {"goal": "найди использования API", "root": project.resolve()})

    def test_main_accepts_read_only_files_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            with patch.object(cli, "default_project_root", return_value=project), patch.object(
                cli, "assistant_root", return_value=Path(directory) / "assistant"
            ), patch.object(cli, "files_command") as files, patch.object(
                sys,
                "argv",
                ["taskmanager-assistant", "files", "найди", "API", "--read-only"],
            ):
                cli.main()

            files.assert_called_once_with("найди API", project.resolve(), read_only=True)

    def test_read_only_files_command_passes_search_scenario_and_mcp_facade(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"

            class ReadOnlyClient:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

            facade = ReadOnlyClient()
            report = type("Report", (), {"render": lambda self: "Только чтение"})()
            with patch.object(cli, "_ReadOnlyMCPClient", return_value=facade), patch.object(
                cli, "run_file_agent", return_value=report
            ) as run, contextlib.redirect_stdout(io.StringIO()):
                cli.files_command("найди API", project, read_only=True)

            run.assert_called_once_with("найди API", project, mcp_client=facade, scenario="search")

    def test_interactive_files_read_does_not_build_rag_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            output_root = Path(directory) / "assistant"
            with patch.object(cli, "default_project_root", return_value=root), patch.object(
                cli, "assistant_root", return_value=output_root
            ), patch.object(cli, "files_command") as files, patch.object(
                sys, "stdin", io.StringIO("/files-read найди API\n")
            ), patch.object(sys, "argv", ["taskmanager-assistant"]), patch.object(
                cli, "ensure_index"
            ) as ensure, contextlib.redirect_stdout(io.StringIO()):
                cli.main()

            ensure.assert_not_called()
            files.assert_called_once_with("найди API", root.resolve(), read_only=True)

    def test_interactive_help_builds_index_lazily_on_first_help(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            output_root = Path(directory) / "assistant"
            with patch.object(cli, "default_project_root", return_value=root), patch.object(
                cli, "assistant_root", return_value=output_root
            ), patch.object(cli, "ensure_index") as ensure, patch.object(
                cli, "help_command"
            ) as help_call, patch.object(
                sys, "stdin", io.StringIO("/files-read найди API\n/help где API\n")
            ), patch.object(sys, "argv", ["taskmanager-assistant"]), patch.object(
                cli, "files_command"
            ), contextlib.redirect_stdout(io.StringIO()):
                cli.main()

            ensure.assert_called_once_with(root.resolve(), output_root, reindex=False)
            help_call.assert_called_once_with("где API", root.resolve(), output_root)


if __name__ == "__main__":
    unittest.main()

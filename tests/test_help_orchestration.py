import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dev_assistant.chunking import corpus_fingerprint
from dev_assistant.cli import CHUNKING_VERSION, RETRIEVAL_VERSION, default_project_root, ensure_index, help_command, index_manifest_path
from dev_assistant.env import load_local_env
from dev_assistant.storage import build_index, search


ROOT = Path(__file__).resolve().parents[1]


class HelpOrchestrationTests(unittest.TestCase):
    def test_local_env_loads_known_values_and_simple_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env").write_text(
                "# comment\nDEEPSEEK_API_KEY='local-key'\nRAG_TOP_K=7 # inline comment\nUNKNOWN=ignored\n",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                load_local_env(Path(directory))
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "local-key")
                self.assertEqual(os.environ["RAG_TOP_K"], "7")
                self.assertNotIn("UNKNOWN", os.environ)

    def test_process_environment_wins_over_local_env(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env").write_text("DEEPSEEK_API_KEY=local-key\n", encoding="utf-8")
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "process-key"}, clear=True):
                load_local_env(Path(directory))
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "process-key")

    def test_loader_only_reads_explicit_assistant_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "taskmanager-ai"
            target.mkdir()
            Path(target, ".env").write_text("DEEPSEEK_API_KEY=target-key\n", encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                load_local_env(root / "taskmanager-ai-assistant")
                self.assertNotIn("DEEPSEEK_API_KEY", os.environ)

    def test_default_target_is_sibling_project(self):
        with patch.dict("os.environ", {}, clear=False):
            with patch.dict("os.environ", {"TASKMANAGER_PROJECT_ROOT": ""}):
                self.assertEqual(default_project_root().name, "taskmanager-ai")

    def test_missing_index_is_built_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with patch("dev_assistant.cli.build_index_command") as build:
                ensure_index(Path(directory) / "project", output_root)
                build.assert_called_once_with(Path(directory) / "project", output_root)

    def test_ready_index_is_not_rebuilt_unless_forced(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            index_dir = output_root / "indexes" / "structural"
            index_dir.mkdir(parents=True)
            (index_dir / "index.faiss").touch()
            (index_dir / "chunks.sqlite").touch()
            index_manifest_path(output_root).write_text(
                '{"chunking_version":"%s","retrieval_version":"%s","corpus_fingerprint":"%s"}'
                % (CHUNKING_VERSION, RETRIEVAL_VERSION, corpus_fingerprint(Path(directory) / "project")),
                encoding="utf-8",
            )
            with patch("dev_assistant.cli.build_index_command") as build:
                ensure_index(Path(directory) / "project", output_root)
                build.assert_not_called()
                ensure_index(Path(directory) / "project", output_root, reindex=True)
                build.assert_called_once_with(Path(directory) / "project", output_root)

    def test_stale_index_manifest_is_rebuilt_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            index_dir = output_root / "indexes" / "structural"
            index_dir.mkdir(parents=True)
            (index_dir / "index.faiss").touch()
            (index_dir / "chunks.sqlite").touch()
            index_manifest_path(output_root).write_text(
                '{"chunking_version":"1","retrieval_version":"1","corpus_fingerprint":"old"}',
                encoding="utf-8",
            )
            with patch("dev_assistant.cli.build_index_command") as build:
                ensure_index(Path(directory) / "project", output_root)
                build.assert_called_once_with(Path(directory) / "project", output_root)

    def test_help_uses_mcp_branch_sources_and_generated_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory) / "project"
            project_root.mkdir()
            subprocess.run(["git", "init", "-q", "-b", "e2e-branch", str(project_root)], check=True)
            output_root = Path(directory) / "assistant"
            index_dir = output_root / "indexes" / "structural"
            index_dir.mkdir(parents=True)
            (index_dir / "index.faiss").touch()
            (index_dir / "chunks.sqlite").touch()
            index_manifest_path(output_root).write_text(
                '{"chunking_version":"%s","retrieval_version":"%s","corpus_fingerprint":"%s"}'
                % (CHUNKING_VERSION, RETRIEVAL_VERSION, corpus_fingerprint(project_root)),
                encoding="utf-8",
            )
            sources = [{"source": "README.md", "start_line": 12, "section": "Architecture", "text": "The server starts in cmd/server/main.go.", "score": 0.91}]

            def fake_answer(question, retrieved, branch):
                self.assertEqual(question, "Where is the server started?")
                self.assertEqual(retrieved, sources)
                self.assertEqual(branch["branch"], "e2e-branch")
                return "The server starts in cmd/server/main.go. [SOURCE 1]"

            output = io.StringIO()
            with patch("dev_assistant.cli.embed", return_value=[0.1, 0.2]), patch("dev_assistant.storage.search", return_value=sources), patch("dev_assistant.cli.answer", side_effect=fake_answer), contextlib.redirect_stdout(output):
                help_command("Where is the server started?", project_root, output_root)

            rendered = output.getvalue()
            self.assertIn("cmd/server/main.go", rendered)
            self.assertIn("Git branch: e2e-branch", rendered)
            self.assertIn("README.md:12", rendered)

    def test_overview_retrieval_prefers_authoritative_project_docs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "chunk_id": "readme-architecture",
                    "source": "README.md",
                    "title": "Task Manager AI",
                    "section": "Системная схема",
                    "heading_path": ["Архитектура", "Системная схема"],
                    "start_line": 80,
                    "end_line": 124,
                    "text": "Архитектура > Системная схема Collector Runtime HTTP API",
                },
                {
                    "chunk_id": "audit-architecture",
                    "source": "docs/context-engineering-audit-task.md",
                    "title": "Audit",
                    "section": "B. Current Architecture",
                    "heading_path": ["B. Current Architecture"],
                    "start_line": 1123,
                    "end_line": 1126,
                    "text": "Current architecture is a template for an audit.",
                },
                {
                    "chunk_id": "mcp-architecture",
                    "source": "TASKMANAGER_MCP.md",
                    "title": "Task Manager MCP Server",
                    "section": "Architecture",
                    "heading_path": ["Architecture"],
                    "start_line": 13,
                    "end_line": 30,
                    "text": "Architecture MCP server API collector.",
                },
            ]
            index, database = root / "index.faiss", root / "chunks.sqlite"
            build_index(records, [[1.0, 0.0]] * len(records), index, database)
            results = search(index, database, [1.0, 0.0], top_k=2, query_text="какая архитектура проекта?")
            self.assertEqual(results[0]["source"], "README.md")
            self.assertEqual(results[0]["section"], "Системная схема")
            self.assertEqual(results[1]["source"], "TASKMANAGER_MCP.md")


if __name__ == "__main__":
    unittest.main()

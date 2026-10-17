import os
import sys
import types
import unittest
from unittest.mock import patch

from dev_assistant.deepseek import answer, complete, review


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "answer"}}]}


class _ErrorResponse:
    status_code = 400
    text = '<html><body>bad request</body></html>'

    def raise_for_status(self):
        raise RuntimeError("400 URL=https://api.deepseek.com/chat/completions")

    def json(self):
        return {"error": {"message": "invalid request", "type": "invalid_request_error"}}


class DeepSeekTests(unittest.TestCase):
    def test_mcp_branch_is_in_generation_context(self):
        captured = {}

        def post(url, **kwargs):
            captured.update(kwargs)
            return _Response()

        fake_httpx = types.SimpleNamespace(post=post)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False), patch.dict(sys.modules, {"httpx": fake_httpx}):
            self.assertEqual(answer("Where?", [{"source": "README.md", "start_line": 1, "text": "context"}], {"branch": "feature/demo", "detached": False}), "answer")
        user_message = captured["json"]["messages"][1]["content"]
        self.assertIn("feature/demo", user_message)
        self.assertIn("get_git_branch", user_message)

    def test_review_prompt_makes_diff_primary_and_requires_russian_sections(self):
        captured = {}

        def post(url, **kwargs):
            captured.update(kwargs)
            return _Response()

        fake_httpx = types.SimpleNamespace(post=post)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False), patch.dict(sys.modules, {"httpx": fake_httpx}):
            self.assertEqual(review("diff text", [{"source": "code.py", "start_line": 3, "text": "context"}], ["code.py"]), "answer")
        messages = captured["json"]["messages"]
        self.assertIn("Потенциальные баги", messages[0]["content"])
        self.assertIn("Архитектурные проблемы", messages[0]["content"])
        self.assertIn("Рекомендации", messages[0]["content"])
        self.assertLess(messages[1]["content"].index("diff text"), messages[1]["content"].index("context"))
        self.assertIn("неподконтрольные данные", messages[0]["content"])

    def test_http_error_is_safe_and_includes_provider_message_for_regular_path(self):
        def post(url, **kwargs):
            return _ErrorResponse()

        fake_httpx = types.SimpleNamespace(post=post)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False), patch.dict(
            sys.modules, {"httpx": fake_httpx}
        ):
            with self.assertRaisesRegex(RuntimeError, r"DeepSeek HTTP 400: invalid request") as raised:
                answer("Where?", [], {"branch": "main"})
        self.assertNotIn("api.deepseek.com", str(raised.exception))
        self.assertNotIn("<html>", str(raised.exception))

    def test_http_error_uses_same_safe_diagnostic_for_tool_call_path(self):
        def post(url, **kwargs):
            return _ErrorResponse()

        fake_httpx = types.SimpleNamespace(post=post)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=False), patch.dict(
            sys.modules, {"httpx": fake_httpx}
        ):
            with self.assertRaisesRegex(RuntimeError, r"DeepSeek HTTP 400: invalid request") as raised:
                complete([{"role": "user", "content": "Where?"}], tools=[])
        self.assertNotIn("api.deepseek.com", str(raised.exception))


if __name__ == "__main__":
    unittest.main()

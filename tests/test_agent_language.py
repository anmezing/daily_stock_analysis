# -*- coding: utf-8 -*-
"""Agent output language is forced by the base agent."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext


class _DummyAgent(BaseAgent):
    agent_name = "dummy"

    def system_prompt(self, ctx: AgentContext) -> str:
        return "You are a test agent."

    def build_user_message(self, ctx: AgentContext) -> str:
        return "test"


class AgentLanguageTest(unittest.TestCase):
    def _agent(self):
        return _DummyAgent(MagicMock(), MagicMock())

    def test_chinese_instruction_is_injected(self) -> None:
        ctx = AgentContext(query="test")
        ctx.meta["report_language"] = "zh"
        messages = self._agent()._build_messages(ctx)
        self.assertIn("所有面向用户的文本必须使用中文", messages[0]["content"])

    def test_english_instruction_is_injected(self) -> None:
        ctx = AgentContext(query="test")
        ctx.meta["report_language"] = "en"
        messages = self._agent()._build_messages(ctx)
        self.assertIn("Write every human-readable value in English", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()

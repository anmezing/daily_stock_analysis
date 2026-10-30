# -*- coding: utf-8 -*-
"""Regression tests for LLMToolAdapter fallback behavior."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.llm_adapter import LLMResponse, LLMToolAdapter


class TestLLMToolAdapterFallback(unittest.TestCase):
    def test_rate_limit_error_falls_back_to_next_model(self) -> None:
        """Cross-provider fallback should NOT sleep (different rate-limit buckets)."""
        adapter = LLMToolAdapter.__new__(LLMToolAdapter)
        adapter._config = SimpleNamespace(
            litellm_model="gemini/primary",
            litellm_fallback_models=["openai/fallback"],
            agent_litellm_model=None,
        )

        responses = [
            RuntimeError("Rate limit exceeded"),
            LLMResponse(content="ok", provider="openai", model="openai/fallback"),
        ]

        def _side_effect(*_args, **_kwargs):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        adapter._call_litellm_model = _side_effect

        with patch("src.agent.llm_adapter.time.sleep") as sleep_mock:
            response = adapter.call_completion([{"role": "user", "content": "hi"}], tools=None)

        self.assertEqual(response.content, "ok")
        # Cross-provider fallback skips sleep — different rate-limit buckets
        sleep_mock.assert_not_called()

    def test_rate_limit_error_same_provider_backs_off(self) -> None:
        """Same-provider fallback SHOULD sleep (same rate-limit bucket)."""
        adapter = LLMToolAdapter.__new__(LLMToolAdapter)
        adapter._config = SimpleNamespace(
            litellm_model="openai/gpt-4",
            litellm_fallback_models=["openai/gpt-3.5-turbo"],
            agent_litellm_model=None,
        )

        responses = [
            RuntimeError("Rate limit exceeded"),
            LLMResponse(content="ok", provider="openai", model="openai/gpt-3.5-turbo"),
        ]

        def _side_effect(*_args, **_kwargs):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        adapter._call_litellm_model = _side_effect

        with patch("src.agent.llm_adapter.time.sleep") as sleep_mock:
            response = adapter.call_completion([{"role": "user", "content": "hi"}], tools=None)

        self.assertEqual(response.content, "ok")
        sleep_mock.assert_called_once()

    def setUp(self) -> None:
        from src.agent import llm_adapter

        llm_adapter.reset_model_health_cache()

    def tearDown(self) -> None:
        from src.agent import llm_adapter

        llm_adapter.reset_model_health_cache()

    def test_removed_model_is_skipped_on_later_calls(self) -> None:
        """404 / no longer available 的模型不该每次分析都再撞一遍。"""
        adapter = LLMToolAdapter.__new__(LLMToolAdapter)
        adapter._config = SimpleNamespace(
            litellm_model="gemini/gemini-2.5-flash",
            litellm_fallback_models=["gemini/gemini-3-flash"],
            agent_litellm_model=None,
        )

        class _NotFoundError(Exception):
            pass

        calls = []

        def _side_effect(_messages, _tools, model, **_kwargs):
            calls.append(model)
            if model == "gemini/gemini-2.5-flash":
                raise _NotFoundError(
                    "litellm.NotFoundError: GeminiException - { 'code': 404, "
                    "'message': 'This model models/gemini-2.5-flash is no longer available' }"
                )
            return LLMResponse(content="ok", provider="gemini", model=model)

        adapter._call_litellm_model = _side_effect

        first = adapter.call_completion([{"role": "user", "content": "hi"}], tools=None)
        second = adapter.call_completion([{"role": "user", "content": "hi"}], tools=None)

        self.assertEqual(first.content, "ok")
        self.assertEqual(second.content, "ok")
        self.assertEqual(calls, [
            "gemini/gemini-2.5-flash", "gemini/gemini-3-flash",
            "gemini/gemini-3-flash",
        ])

    def test_exhausted_quota_fails_fast_without_calling_models(self) -> None:
        """配额打满后，剩下的股票应该立刻失败，而不是每只都重试一遍模型。"""
        adapter = LLMToolAdapter.__new__(LLMToolAdapter)
        adapter._config = SimpleNamespace(
            litellm_model="gemini/gemini-3-flash",
            litellm_fallback_models=[],
            agent_litellm_model=None,
        )

        calls = []

        def _side_effect(_messages, _tools, model, **_kwargs):
            calls.append(model)
            raise RuntimeError(
                "litellm.RateLimitError: 429 You exceeded your current quota\n"
                "Please check your plan and billing details\n"
                "generate_content_free_tier_requests, limit: 20"
            )

        adapter._call_litellm_model = _side_effect

        first = adapter.call_completion([{"role": "user", "content": "hi"}], tools=None)
        second = adapter.call_completion([{"role": "user", "content": "hi"}], tools=None)

        self.assertEqual(first.provider, "error")
        self.assertEqual(second.provider, "error")
        self.assertIn("quota", second.content.lower())
        self.assertEqual(calls, ["gemini/gemini-3-flash"])

    def test_only_dead_models_reports_without_calling_provider(self) -> None:
        from src.agent import llm_adapter

        adapter = LLMToolAdapter.__new__(LLMToolAdapter)
        adapter._config = SimpleNamespace(
            litellm_model="gemini/gemini-2.5-flash",
            litellm_fallback_models=[],
            agent_litellm_model=None,
        )
        llm_adapter._DEAD_MODELS.add("gemini/gemini-2.5-flash")

        adapter._call_litellm_model = lambda *_a, **_k: self.fail("should not call the provider")

        response = adapter.call_completion([{"role": "user", "content": "hi"}], tools=None)

        self.assertEqual(response.provider, "error")
        self.assertIn("unavailable", response.content.lower())


if __name__ == "__main__":
    unittest.main()

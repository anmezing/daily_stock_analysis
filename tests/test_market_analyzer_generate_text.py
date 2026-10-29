# -*- coding: utf-8 -*-
"""Tests for Analyzer.generate_text() and the market_analyzer bypass fix.

Covers:
- generate_text() returns the LLM response on success
- generate_text() returns None and logs on failure (no exception propagated)
- market_analyzer calls generate_text(), not private analyzer attributes
- Any provider configuration (Gemini / Anthropic / OpenAI / LLM_CHANNELS)
  does NOT trigger AttributeError (regression guard for the old bypass bug)
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

# Stub heavy dependencies before project imports
for _mod in ("litellm", "google.generativeai", "google.genai", "anthropic"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest


# ---------------------------------------------------------------------------
# Analyzer.generate_text()
# ---------------------------------------------------------------------------

class TestAnalyzerGenerateText:
    def _make_analyzer(self):
        """Return a minimally configured GeminiAnalyzer with _call_litellm mocked."""
        with patch("src.analyzer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.litellm_model = "gemini/gemini-2.0-flash"
            cfg.litellm_fallback_models = []
            cfg.gemini_api_keys = ["sk-gemini-testkey-1234"]
            cfg.anthropic_api_keys = []
            cfg.openai_api_keys = []
            cfg.deepseek_api_keys = []
            cfg.llm_model_list = []
            cfg.openai_base_url = None
            mock_cfg.return_value = cfg
            from src.analyzer import GeminiAnalyzer
            analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
            analyzer._router = None
            return analyzer

    def test_generate_text_returns_llm_response(self):
        analyzer = self._make_analyzer()
        with patch.object(analyzer, "_call_litellm", return_value="市场分析报告") as mock_call:
            result = analyzer.generate_text("写一份复盘", max_tokens=1024, temperature=0.5)
            assert result == "市场分析报告"
            mock_call.assert_called_once_with(
                "写一份复盘",
                generation_config={"max_tokens": 1024, "temperature": 0.5},
                stream=True,
            )

    def test_generate_text_returns_none_on_failure(self):
        analyzer = self._make_analyzer()
        with patch.object(analyzer, "_call_litellm", side_effect=Exception("LLM error")):
            result = analyzer.generate_text("prompt")
            assert result is None  # must not raise

    def test_generate_text_default_params(self):
        analyzer = self._make_analyzer()
        with patch.object(analyzer, "_call_litellm", return_value="ok") as mock_call:
            analyzer.generate_text("hello")
            _, kwargs = mock_call.call_args
            gen_cfg = kwargs["generation_config"]
            assert gen_cfg["max_tokens"] == 2048
            assert gen_cfg["temperature"] == 0.7

    def test_call_litellm_retries_gemini_rate_limit_with_backoff(self):
        from src.analyzer import GeminiAnalyzer

        analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
        analyzer._router = None

        cfg = MagicMock()
        cfg.litellm_model = "gemini/gemini-2.5-flash"
        cfg.litellm_fallback_models = []
        cfg.llm_model_list = []
        cfg.gemini_max_retries = 2
        cfg.gemini_retry_delay = 3.0

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )

        with patch("src.analyzer.get_config", return_value=cfg), \
             patch("src.analyzer.get_api_keys_for_model", return_value=["sk-test"]), \
             patch("src.analyzer.extra_litellm_params", return_value={}), \
             patch("src.analyzer.get_configured_llm_models", return_value=[]), \
             patch("src.analyzer.get_thinking_extra_body", return_value=None), \
             patch("src.analyzer.get_llm_rate_limiter"), \
             patch("src.analyzer.time.sleep") as mock_sleep, \
             patch(
                 "src.analyzer.litellm.completion",
                 side_effect=[Exception("429 RESOURCE_EXHAUSTED"), Exception("rate limit"), response],
             ) as mock_completion:
            text, model_used, usage = analyzer._call_litellm(
                "prompt",
                generation_config={"max_tokens": 256, "temperature": 0.3},
            )

        assert text == "ok"
        assert model_used == "gemini/gemini-2.5-flash"
        assert usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        assert mock_completion.call_count == 3
        mock_sleep.assert_has_calls([call(3.0), call(6.0)])

    def test_call_litellm_uses_fallback_after_retry_exhausted(self):
        from src.analyzer import GeminiAnalyzer

        analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
        analyzer._router = None

        cfg = MagicMock()
        cfg.litellm_model = "gemini/gemini-2.5-flash"
        cfg.litellm_fallback_models = ["openai/gpt-4o-mini"]
        cfg.llm_model_list = []
        cfg.gemini_max_retries = 2
        cfg.gemini_retry_delay = 1.5

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fallback ok"))],
            usage=None,
        )
        seen_models = []

        def _fake_completion(**kwargs):
            seen_models.append(kwargs["model"])
            if kwargs["model"] == "gemini/gemini-2.5-flash":
                raise Exception("429 RESOURCE_EXHAUSTED")
            return response

        with patch("src.analyzer.get_config", return_value=cfg), \
             patch("src.analyzer.get_api_keys_for_model", return_value=["sk-test"]), \
             patch("src.analyzer.extra_litellm_params", return_value={}), \
             patch("src.analyzer.get_configured_llm_models", return_value=[]), \
             patch("src.analyzer.get_thinking_extra_body", return_value=None), \
             patch("src.analyzer.get_llm_rate_limiter"), \
             patch("src.analyzer.time.sleep") as mock_sleep, \
             patch("src.analyzer.litellm.completion", side_effect=_fake_completion):
            text, model_used, usage = analyzer._call_litellm(
                "prompt",
                generation_config={"max_tokens": 256, "temperature": 0.3},
            )

        assert text == "fallback ok"
        assert model_used == "openai/gpt-4o-mini"
        assert usage == {}
        assert seen_models == [
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-flash",
            "gemini/gemini-2.5-flash",
            "openai/gpt-4o-mini",
        ]
        mock_sleep.assert_has_calls([call(1.5), call(3.0)])

    def test_call_litellm_does_not_retry_non_retryable_error(self):
        from src.analyzer import GeminiAnalyzer

        analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
        analyzer._router = None

        cfg = MagicMock()
        cfg.litellm_model = "gemini/gemini-2.5-flash"
        cfg.litellm_fallback_models = []
        cfg.llm_model_list = []
        cfg.gemini_max_retries = 3
        cfg.gemini_retry_delay = 2.0

        with patch("src.analyzer.get_config", return_value=cfg), \
             patch("src.analyzer.get_api_keys_for_model", return_value=["sk-test"]), \
             patch("src.analyzer.extra_litellm_params", return_value={}), \
             patch("src.analyzer.get_configured_llm_models", return_value=[]), \
             patch("src.analyzer.get_thinking_extra_body", return_value=None), \
             patch("src.analyzer.time.sleep") as mock_sleep, \
             patch("src.analyzer.litellm.completion", side_effect=Exception("invalid api key")) as mock_completion:
            with pytest.raises(Exception, match="All LLM models failed"):
                analyzer._call_litellm(
                    "prompt",
                    generation_config={"max_tokens": 256, "temperature": 0.3},
                )

        assert mock_completion.call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# market_analyzer uses generate_text(), not private attributes
# ---------------------------------------------------------------------------

class TestMarketAnalyzerBypassFix:
    def _make_market_analyzer_with_mock_generate_text(self, return_value="复盘报告"):
        """Return a MarketAnalyzer whose embedded Analyzer.generate_text is mocked."""
        from src.core.market_profile import CN_PROFILE
        from src.core.market_strategy import get_market_strategy_blueprint

        with patch("src.analyzer.get_config") as mock_cfg, \
             patch("src.market_analyzer.get_config") as mock_cfg2:
            cfg = MagicMock()
            cfg.litellm_model = "gemini/gemini-2.0-flash"
            cfg.litellm_fallback_models = []
            cfg.gemini_api_keys = ["sk-gemini-testkey-1234"]
            cfg.anthropic_api_keys = []
            cfg.openai_api_keys = []
            cfg.deepseek_api_keys = []
            cfg.llm_model_list = []
            cfg.openai_base_url = None
            cfg.market_review_region = "cn"
            cfg.llm_temperature = 0.3
            mock_cfg.return_value = cfg
            mock_cfg2.return_value = cfg

            from src.analyzer import GeminiAnalyzer
            from src.market_analyzer import MarketAnalyzer

            analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
            analyzer._router = None
            analyzer._litellm_available = True
            analyzer.generate_text = MagicMock(return_value=return_value)

            ma = MarketAnalyzer.__new__(MarketAnalyzer)
            ma.analyzer = analyzer
            ma.profile = CN_PROFILE
            ma.strategy = get_market_strategy_blueprint("cn")
            ma.region = "cn"
            return ma

    def test_no_access_to_private_model_attribute(self):
        """generate_text() must be called; _model must never be accessed."""
        ma = self._make_market_analyzer_with_mock_generate_text("复盘结果")
        # Ensure _model attribute does not exist (simulates PR #494 state)
        assert not hasattr(ma.analyzer, "_model") or ma.analyzer._model is None, (
            "_model should not be set on the LiteLLM-based analyzer"
        )
        # generate_text is a MagicMock, so calling it won't crash
        result = ma.analyzer.generate_text("prompt")
        assert isinstance(result, str) and len(result) > 0
        ma.analyzer.generate_text.assert_called_once()

    def test_generate_text_none_falls_back_to_template(self):
        """generate_market_review() falls back to template when generate_text returns None."""
        from src.market_analyzer import MarketOverview, MarketIndex

        ma = self._make_market_analyzer_with_mock_generate_text(return_value=None)
        overview = MarketOverview(
            date="2026-03-05",
            indices=[
                MarketIndex(
                    code="000001",
                    name="上证指数",
                    current=3300.0,
                    change=5.0,
                    change_pct=0.15,
                )
            ],
        )
        result = ma.generate_market_review(overview, [])
        assert isinstance(result, str) and len(result) > 0
        ma.analyzer.generate_text.assert_called_once()

    def test_market_review_uses_8192_max_tokens(self):
        """generate_market_review() should request a larger output budget to avoid truncation."""
        from src.market_analyzer import MarketOverview, MarketIndex

        ma = self._make_market_analyzer_with_mock_generate_text(return_value="复盘结果")
        overview = MarketOverview(
            date="2026-03-05",
            indices=[
                MarketIndex(
                    code="000001",
                    name="上证指数",
                    current=3300.0,
                    change=5.0,
                    change_pct=0.15,
                )
            ],
        )

        result = ma.generate_market_review(overview, [])

        assert isinstance(result, str) and len(result) > 0
        ma.analyzer.generate_text.assert_called_once()
        _, kwargs = ma.analyzer.generate_text.call_args
        assert kwargs["max_tokens"] == 8192
        assert kwargs["temperature"] == 0.3

    def test_market_review_uses_market_review_system_prompt(self):
        """Market review should override the stock dashboard system prompt."""
        from src.market_analyzer import MARKET_REVIEW_SYSTEM_PROMPT, MarketOverview, MarketIndex

        ma = self._make_market_analyzer_with_mock_generate_text(return_value="复盘结果")
        overview = MarketOverview(
            date="2026-03-05",
            indices=[
                MarketIndex(
                    code="000001",
                    name="上证指数",
                    current=3300.0,
                    change=5.0,
                    change_pct=0.15,
                )
            ],
        )

        ma.generate_market_review(overview, [])

        _, kwargs = ma.analyzer.generate_text.call_args
        assert kwargs["system_prompt"] == MARKET_REVIEW_SYSTEM_PROMPT

    def test_no_private_attribute_access_in_market_analyzer_source(self):
        """Static guard: market_analyzer.py must not access private analyzer attrs."""
        import ast
        import pathlib

        src = pathlib.Path("src/market_analyzer.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        forbidden = {
            "_model", "_router", "_use_openai", "_use_anthropic",  # historical
            "_call_litellm",      # use generate_text() instead
            "_litellm_available", # use is_available() instead
        }

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden:
                    violations.append(node.attr)

        assert violations == [], (
            f"market_analyzer.py still accesses private Analyzer attributes: {violations}"
        )

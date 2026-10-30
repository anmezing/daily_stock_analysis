# -*- coding: utf-8 -*-
"""LLM 请求体非有限浮点数清洗的回归测试。"""

import math
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import parse_env_float, resolve_unified_llm_temperature
from src.utils.json_sanitize import sanitize_non_finite


class SanitizeNonFiniteTest(unittest.TestCase):
    def test_replaces_top_level_nan_temperature(self) -> None:
        cleaned, replaced = sanitize_non_finite({"temperature": float("nan"), "max_tokens": 100})

        self.assertIsNone(cleaned["temperature"])
        self.assertEqual(cleaned["max_tokens"], 100)
        self.assertEqual(replaced, ["temperature"])

    def test_replaces_infinity_nested_in_messages(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "hi", "extra": {"ratio": float("inf")}},
            ]
        }

        cleaned, replaced = sanitize_non_finite(payload)

        self.assertIsNone(cleaned["messages"][0]["extra"]["ratio"])
        self.assertEqual(cleaned["messages"][0]["content"], "hi")
        self.assertEqual(replaced, ["messages[0].extra.ratio"])

    def test_lists_and_tuples_keep_type(self) -> None:
        cleaned, replaced = sanitize_non_finite({"a": [1.0, float("nan")], "b": (2.0, float("-inf"))})

        self.assertEqual(cleaned["a"], [1.0, None])
        self.assertEqual(cleaned["b"], (2.0, None))
        self.assertEqual(len(replaced), 2)

    def test_finite_values_untouched(self) -> None:
        cleaned, replaced = sanitize_non_finite({"temperature": 0.3, "messages": [{"content": "nan 文本"}]})

        self.assertEqual(cleaned["temperature"], 0.3)
        self.assertEqual(cleaned["messages"][0]["content"], "nan 文本")
        self.assertEqual(replaced, [])

    def test_numpy_floats_become_builtin_floats(self) -> None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            self.skipTest("numpy not installed")

        cleaned, replaced = sanitize_non_finite({"price": np.float64(1.5), "bad": np.float64("nan")})

        self.assertIsInstance(cleaned["price"], float)
        self.assertEqual(cleaned["price"], 1.5)
        self.assertIsNone(cleaned["bad"])
        self.assertEqual(replaced, ["bad"])


class NonFiniteEnvConfigTest(unittest.TestCase):
    def test_parse_env_float_rejects_nan(self) -> None:
        value = parse_env_float("nan", 0.3, field_name="LLM_TEMPERATURE")

        self.assertTrue(math.isfinite(value))
        self.assertEqual(value, 0.3)

    def test_parse_env_float_rejects_inf(self) -> None:
        value = parse_env_float("inf", 0.3, field_name="LLM_TEMPERATURE")

        self.assertEqual(value, 0.3)

    def test_resolve_temperature_rejects_nan(self) -> None:
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "nan"}, clear=False):
            self.assertEqual(resolve_unified_llm_temperature("gemini/gemini-3-flash-preview"), 0.3)

    def test_resolve_temperature_keeps_valid_value(self) -> None:
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "0.2"}, clear=False):
            self.assertEqual(resolve_unified_llm_temperature("gemini/gemini-3-flash-preview"), 0.2)


class ToolResultSerializationTest(unittest.TestCase):
    def test_serialize_tool_result_replaces_nan_with_null(self) -> None:
        from src.agent.runner import serialize_tool_result

        payload = {
            "code": "AAPL",
            "data": [{"date": "2026-10-28", "close": 268.5, "ma5": float("nan"), "ma10": float("inf")}],
        }

        text = serialize_tool_result(payload)

        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertIn("null", text)
        self.assertIn("268.5", text)

    def test_serialize_tool_result_converts_numpy_scalars(self) -> None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            self.skipTest("numpy not installed")

        from src.agent.runner import serialize_tool_result

        text = serialize_tool_result({"close": np.float64(1.25), "volume": np.int64(100), "ok": np.bool_(True)})

        self.assertEqual(text, '{"close": 1.25, "volume": 100, "ok": true}')

    def test_adapter_json_safe_text_handles_nan(self) -> None:
        from src.agent.llm_adapter import LLMToolAdapter

        text = LLMToolAdapter._json_safe_text({"tool": "x", "value": float("nan")})

        self.assertEqual(text, '{"tool": "x", "value": null}')


if __name__ == "__main__":
    unittest.main()

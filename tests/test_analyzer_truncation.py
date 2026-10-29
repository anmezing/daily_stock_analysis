# -*- coding: utf-8 -*-
"""输出被 max_tokens 截断时必须降级，不能当成完整分析入库。"""

import json
import sys
from unittest.mock import MagicMock, patch

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from src.analyzer import GeminiAnalyzer  # noqa: E402

_JSON = json.dumps(
    {
        "stock_name": "Apple Inc.",
        "sentiment_score": 72,
        "trend_prediction": "看多",
        "operation_advice": "买入",
        "decision_type": "buy",
        "confidence_level": "中",
        "analysis_summary": "ok",
    }
)


def _analyzer() -> GeminiAnalyzer:
    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        analyzer = GeminiAnalyzer()
    analyzer._litellm_available = False
    return analyzer


def test_stream_finish_reason_length_marks_truncated():
    analyzer = _analyzer()
    chunk = {
        "choices": [
            {"delta": {"content": '{"sentiment_score": 72'}, "finish_reason": "length"}
        ]
    }
    meta = {}

    text, _ = analyzer._consume_litellm_stream(iter([chunk]), model="m", response_meta=meta)

    assert text == '{"sentiment_score": 72'
    assert meta["truncated"] is True


def test_truncated_response_is_retried_then_accepted():
    analyzer = _analyzer()
    calls = {"count": 0}

    def _fake_call(prompt, generation_config, *, response_meta=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            response_meta["truncated"] = True
        return _JSON, "m", {}

    cfg = MagicMock()
    cfg.llm_temperature = 0.3
    cfg.litellm_model = "gemini/x"
    cfg.report_integrity_enabled = True
    cfg.report_integrity_retry = 1
    cfg.gemini_request_delay = 0

    with patch("src.analyzer.get_config", return_value=cfg), \
         patch.object(analyzer, "is_available", return_value=True), \
         patch.object(analyzer, "_call_litellm", side_effect=_fake_call), \
         patch.object(analyzer, "_check_content_integrity", return_value=(True, [])), \
         patch.object(analyzer, "_format_prompt", return_value="prompt"), \
         patch.object(analyzer, "_build_market_snapshot", return_value={}):
        result = analyzer.analyze({"code": "AAPL", "stock_name": "Apple Inc."})

    assert calls["count"] == 2
    assert result.parse_mode == "structured"


def test_persistent_truncation_stays_degraded():
    analyzer = _analyzer()

    def _fake_call(prompt, generation_config, *, response_meta=None, **kwargs):
        response_meta["truncated"] = True
        return _JSON, "m", {}

    cfg = MagicMock()
    cfg.llm_temperature = 0.3
    cfg.litellm_model = "gemini/x"
    cfg.report_integrity_enabled = True
    cfg.report_integrity_retry = 1
    cfg.gemini_request_delay = 0

    with patch("src.analyzer.get_config", return_value=cfg), \
         patch.object(analyzer, "is_available", return_value=True), \
         patch.object(analyzer, "_call_litellm", side_effect=_fake_call), \
         patch.object(analyzer, "_check_content_integrity", return_value=(True, [])), \
         patch.object(analyzer, "_format_prompt", return_value="prompt"), \
         patch.object(analyzer, "_build_market_snapshot", return_value={}):
        result = analyzer.analyze({"code": "AAPL", "stock_name": "Apple Inc."})

    assert result.parse_mode == "truncated"

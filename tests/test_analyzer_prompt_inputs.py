# -*- coding: utf-8 -*-
"""喂给大模型的输入数据：缺失值不能变成假信号。"""

import sys
from unittest.mock import MagicMock, patch

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from src.analyzer import GeminiAnalyzer  # noqa: E402


def _analyzer() -> GeminiAnalyzer:
    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        analyzer = GeminiAnalyzer()
    analyzer._litellm_available = False
    return analyzer


def _context(**extra):
    ctx = {
        "code": "AAPL",
        "stock_name": "Apple Inc.",
        "date": "2026-03-23",
        "today": {
            "date": "2026-03-20",
            "close": 180.0,
            "open": 178.0,
            "high": 182.0,
            "low": 177.0,
            "volume": 1_000_000,
            "pct_chg": 1.2,
            "ma5": 178.0,
            "ma10": 175.0,
            "ma20": 170.0,
        },
    }
    ctx.update(extra)
    return ctx


def test_missing_chip_data_is_not_rendered_as_zero():
    """筹码缺失时不能显示 0.00%，否则模型会认为筹码极度集中。"""
    prompt = _analyzer()._format_prompt(
        _context(chip={"avg_cost": None, "chip_status": None}),
        "Apple Inc.",
    )

    assert "N/A" in prompt
    assert "0.00%" not in prompt
    assert "0.0%" not in prompt


def test_missing_bias_does_not_claim_safe_range():
    prompt = _analyzer()._format_prompt(
        _context(trend_analysis={"trend_status": "多头"}),
        "Apple Inc.",
    )

    assert "乖离率数据缺失" in prompt
    assert "安全范围" not in prompt


def test_none_quote_fields_are_not_literal_none():
    prompt = _analyzer()._format_prompt(
        _context(realtime={"price": 181.0, "volume_ratio": None, "pe_ratio": None, "pb_ratio": None}),
        "Apple Inc.",
    )

    # 不能把 Python 的 None 直接写进提示词
    assert "None" not in prompt
    assert "N/A" in prompt


def test_prompt_discloses_the_bar_date():
    """行情表必须带上数据日期，避免把缓存里的旧 K 线当"今日"。"""
    prompt = _analyzer()._format_prompt(_context(), "Apple Inc.")

    assert "数据日期" in prompt
    assert "2026-03-20" in prompt


def test_missing_fields_warning_lists_actual_gaps():
    prompt = _analyzer()._format_prompt(
        _context(
            today={"date": "2026-03-20", "close": 180.0, "ma5": None, "ma20": None, "volume": None},
            realtime={"price": 181.0, "volume_ratio": None, "turnover_rate": None},
        ),
        "Apple Inc.",
    )

    assert "数据缺失警告" in prompt
    assert "均线" in prompt
    assert "量比" in prompt
    # 有明确清单时不应再出现兜底文案
    assert "部分实时行情" not in prompt


def test_intraday_price_is_not_called_close():
    """盘中注入的是最新价，标成"收盘价"会让模型误判突破/支撑已确认。"""
    prompt = _analyzer()._format_prompt(
        _context(quote_injected=True, is_intraday=True, session_label="盘中"),
        "Apple Inc.",
    )

    assert "最新价（分析时点快照，非收盘价）" in prompt
    assert "成交量（盘中累计，非全天）" in prompt
    assert "| 收盘价 | 180.00 美元 |" not in prompt


def test_bias_warning_uses_the_same_threshold_as_scoring():
    """提示词不能一边说"超过 5% 别追"，一边按美股 15% 阈值给高分。"""
    prompt = _analyzer()._format_prompt(
        _context(
            market_context={"market": "us", "market_label": "美股"},
            trend_analysis={"trend_status": "多头排列", "bias_ma5": 8.0, "trend_strength": 75},
        ),
        "Apple Inc.",
    )

    assert "处于 5.0%~8.0% 警戒区" in prompt
    assert "超过 5.0%" not in prompt


def test_bias_above_market_threshold_is_flagged():
    prompt = _analyzer()._format_prompt(
        _context(
            market_context={"market": "us", "market_label": "美股"},
            trend_analysis={"trend_status": "多头排列", "bias_ma5": 20.0, "trend_strength": 75},
        ),
        "Apple Inc.",
    )

    assert "超过 8.0%" in prompt


def test_fundamental_context_reaches_the_prompt():
    """基本面数据不能只写进快照：模型看不到就只能写空话或编造。"""
    prompt = _analyzer()._format_prompt(
        _context(
            fundamental_context={
                "status": "partial",
                "growth": {"data": {"revenue_yoy": 20.0, "roe": 35.5}},
                "earnings": {"data": {"latest_filing_form": "10-Q", "latest_filing_date": "2026-02-01"}},
                "institution": {"data": {"insider_form4_count_90d": 3}},
            }
        ),
        "Apple Inc.",
    )

    assert "基本面数据（数据源聚合）" in prompt
    assert "+20.00%" in prompt
    assert "35.50%" in prompt
    assert "10-Q" in prompt
    assert "3 笔" in prompt


def test_empty_fundamental_context_adds_no_noise():
    prompt = _analyzer()._format_prompt(_context(), "Apple Inc.")

    assert "基本面数据（数据源聚合）" not in prompt


def test_numeric_validator_flags_invented_price():
    from src.analyzer import AnalysisResult, validate_result_numbers

    result = AnalysisResult(
        code="AAPL",
        name="Apple Inc.",
        sentiment_score=70,
        trend_prediction="看多",
        operation_advice="买入",
        decision_type="buy",
        dashboard={
            "data_perspective": {
                "price_position": {"current_price": 999.0, "ma5": 178.0}
            }
        },
    )

    issues = validate_result_numbers(result, _context(trend_analysis={"ma5": 178.0}))

    assert any(issue.startswith("price_mismatch:current_price") for issue in issues)


def test_numeric_validator_accepts_matching_numbers():
    from src.analyzer import AnalysisResult, validate_result_numbers

    result = AnalysisResult(
        code="AAPL",
        name="Apple Inc.",
        sentiment_score=70,
        trend_prediction="看多",
        operation_advice="买入",
        decision_type="buy",
        dashboard={
            "data_perspective": {
                "price_position": {"current_price": 180.0, "ma5": 178.0}
            }
        },
    )

    issues = validate_result_numbers(result, _context(trend_analysis={"ma5": 178.0}))

    assert issues == []


def test_numeric_validator_flags_impossible_target_order():
    from src.analyzer import AnalysisResult, validate_result_numbers

    result = AnalysisResult(
        code="AAPL",
        name="Apple Inc.",
        sentiment_score=70,
        trend_prediction="看多",
        operation_advice="买入",
        decision_type="buy",
        dashboard={
            "data_perspective": {
                "price_position": {"current_price": 180.0, "ma5": 178.0}
            },
            "battle_plan": {
                "sniper_points": {"stop_loss": "190", "take_profit": "200"}
            },
        },
    )

    issues = validate_result_numbers(result, _context(trend_analysis={"ma5": 178.0}))

    assert "target_order_invalid" in issues


def test_regime_block_warns_in_risk_off():
    prompt = _analyzer()._format_prompt(
        _context(
            market_regime={
                "code": "SPY",
                "close": 480.0,
                "ma200": 500.0,
                "above_ma200": False,
                "volatility_20d_pct": 1.8,
                "recent_volatility_pct": 2.2,
            }
        ),
        "Apple Inc.",
    )

    assert "市场状态（基准指数）" in prompt
    assert "risk-off" in prompt
    assert "200日线下方" in prompt

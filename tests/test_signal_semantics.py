# -*- coding: utf-8 -*-
"""买卖建议的单一语义表：analyzer 与回测必须给出一致解释。"""

import sys
from unittest.mock import MagicMock

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from src.analyzer import _signal_from_operation_advice  # noqa: E402
from src.core.backtest_engine import BacktestEngine  # noqa: E402
from src.core.signal_semantics import (  # noqa: E402
    infer_direction,
    infer_position,
    infer_signal,
)


def test_previously_misclassified_words_are_recognised():
    """这些词过去在 analyzer 侧识别为空、或与回测解释不一致。"""
    expected = {
        "清仓": ("sell", "down", "cash"),
        "增持": ("buy", "up", "long"),
        "建仓": ("buy", "up", "long"),
        "止盈": ("sell", "not_up", "cash"),
        "止损": ("sell", "not_up", "cash"),
        "reduce": ("sell", "not_up", "cash"),
        "add": ("buy", "up", "long"),
    }
    for advice, (signal, direction, position) in expected.items():
        assert infer_signal(advice) == signal, advice
        assert infer_direction(advice) == direction, advice
        assert infer_position(advice) == position, advice


def test_wait_and_reduce_are_exposure_management_not_bearish():
    for advice in ("观望", "等待", "wait", "减仓", "止盈", "止损", "reduce"):
        assert infer_position(advice) == "cash"
        # 关键：不要求"必须下跌"，只要求"没有显著上涨"
        assert infer_direction(advice) == "not_up", advice


def test_hold_stays_long_but_wait_goes_cash():
    assert infer_position("持有") == "long"
    assert infer_direction("持有") == "not_down"
    assert infer_position("观望") == "cash"
    assert infer_direction("观望") == "not_up"


def test_compound_advice_is_deterministic_by_first_action():
    """「持有观望」过去上游判持有、下游判空仓，现在按最先出现的动作解析。"""
    assert infer_signal("持有观望") == "hold"
    assert infer_direction("持有观望") == "not_down"
    assert infer_direction("观望持有") == "not_up"


def test_negation_is_respected():
    assert infer_direction("不要卖出") != "down"
    assert infer_direction("do not sell") != "down"


def test_unknown_advice_falls_back_to_cash_and_flat():
    assert infer_signal("some gibberish") == ""
    assert infer_direction("some gibberish") == "flat"
    assert infer_position("some gibberish") == "cash"


def test_analyzer_and_backtest_agree_on_every_word():
    words = [
        "买入", "加仓", "增持", "建仓", "持有", "观望", "等待",
        "卖出", "清仓", "减仓", "止盈", "止损",
        "buy", "add", "hold", "wait", "sell", "reduce",
    ]
    for advice in words:
        # analyzer 侧（报告里展示的信号）
        assert _signal_from_operation_advice(advice) == infer_signal(advice), advice
        # 回测侧（评估口径）
        assert BacktestEngine.infer_direction_expected(advice) == infer_direction(advice), advice
        assert BacktestEngine.infer_position_recommendation(advice) == infer_position(advice), advice

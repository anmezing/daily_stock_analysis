# -*- coding: utf-8 -*-
"""长周期均线：数据不足时不能拿 0 冒充，数据足够时必须真的算出来。"""

import pandas as pd

from src.stock_analyzer import StockTrendAnalyzer


def _frame(count: int) -> pd.DataFrame:
    rows = []
    for index in range(count):
        close = 100.0 + index * 0.2
        rows.append(
            {
                "date": f"2024-{1 + index // 28:02d}-{1 + index % 28:02d}",
                "open": close - 0.1,
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


def test_long_moving_averages_are_computed_when_history_is_long_enough():
    result = StockTrendAnalyzer().analyze(_frame(260), "AAPL")

    assert result.ma50 > 0
    assert result.ma60 > 0
    assert result.ma200 > 0
    # 线性上涨序列里，越长的均线越低
    assert result.ma5 > result.ma50 > result.ma200


def test_long_moving_averages_stay_zero_when_history_is_short():
    result = StockTrendAnalyzer().analyze(_frame(100), "AAPL")

    assert result.ma50 > 0
    assert result.ma200 == 0


def test_prompt_does_not_show_zero_moving_averages():
    from tests.test_analyzer_prompt_inputs import _analyzer, _context

    prompt = _analyzer()._format_prompt(_context(today={"date": "2026-03-20", "close": 180.0}), "Apple Inc.")

    assert "| MA200 | N/A |" in prompt
    assert "| MA200 | 0.00 |" not in prompt

# -*- coding: utf-8 -*-
"""盘中数据不能被当成完整交易日来解读。"""

import pandas as pd

from src.stock_analyzer import StockTrendAnalyzer, VolumeStatus


def _frame(*, last_volume: float, down: bool = True) -> pd.DataFrame:
    bars = []
    for index in range(30):
        close = 100.0 + index * 0.5
        bars.append(
            {
                "date": f"2026-03-{index + 1:02d}",
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    last = bars[-1]
    last["close"] = last["close"] - 1.0 if down else last["close"] + 1.0
    last["volume"] = last_volume
    return pd.DataFrame(bars)


def test_intraday_bar_is_not_classified_as_shrink_volume():
    """10:00 的累计成交量只有全天的零头，不能据此判定“缩量回调”。"""
    analyzer = StockTrendAnalyzer()
    df = _frame(last_volume=100_000.0)

    result = analyzer.analyze(df, "AAPL", intraday=True)

    assert result.volume_status is VolumeStatus.INTRADAY_UNKNOWN
    assert "盘中" in result.volume_trend
    assert not any("缩量" in reason for reason in result.signal_reasons)


def test_closed_bar_still_uses_volume_shape():
    """同一根 K 线在收盘后是完整数据，量能判断照常生效。"""
    analyzer = StockTrendAnalyzer()
    df = _frame(last_volume=100_000.0)

    result = analyzer.analyze(df, "AAPL", intraday=False)

    assert result.volume_status is VolumeStatus.SHRINK_VOLUME_DOWN


def test_intraday_volume_does_not_add_shrink_bonus():
    analyzer = StockTrendAnalyzer()
    df = _frame(last_volume=100_000.0)

    closed = analyzer.analyze(df, "AAPL", intraday=False)
    live = analyzer.analyze(df, "AAPL", intraday=True)

    # 缩量回调 15 分 → 盘中按中性 10 分
    assert closed.signal_score - live.signal_score == 5

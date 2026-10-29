# -*- coding: utf-8 -*-
"""Stale daily data must stop the analysis instead of producing a trading call."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.core.pipeline import StockAnalysisPipeline


def _pipeline(latest: date | None):
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace(analysis_max_staleness_days=7)
    pipeline.db = MagicMock()
    pipeline.db.get_latest_data.return_value = (
        [SimpleNamespace(date=latest)] if latest else []
    )
    pipeline.fetch_and_save_stock_data = MagicMock(return_value=(False, "network down"))
    return pipeline


@patch("src.core.pipeline.get_market_today", return_value=date(2026, 10, 29))
@patch("src.core.pipeline.get_market_for_stock", return_value="us")
def test_stale_data_is_refreshed_once_then_rejected(_market, _today):
    pipeline = _pipeline(date(2026, 8, 21))

    fresh, latest, target = pipeline._ensure_data_fresh("AAPL")

    assert fresh is False
    assert latest == date(2026, 8, 21)
    assert target == date(2026, 10, 29)
    pipeline.fetch_and_save_stock_data.assert_called_once_with(
        "AAPL", force_refresh=True
    )


@patch("src.core.pipeline.get_market_today", return_value=date(2026, 10, 29))
@patch("src.core.pipeline.get_market_for_stock", return_value="us")
def test_fresh_data_is_not_refetched(_market, _today):
    pipeline = _pipeline(date(2026, 10, 28))

    fresh, latest, _ = pipeline._ensure_data_fresh("AAPL")

    assert fresh is True
    assert latest == date(2026, 10, 28)
    pipeline.fetch_and_save_stock_data.assert_not_called()

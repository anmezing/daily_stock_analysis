# -*- coding: utf-8 -*-
"""Benchmark series: window returns, regime snapshot and the off switch."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.services.benchmark_service import BenchmarkService


class FakeDB:
    def __init__(self) -> None:
        self.rows = []

    def get_data_range(self, code, start, end):
        return [row for row in self.rows if start <= row.date <= end]

    def save_daily_data(self, frame, code, data_source="unknown"):
        for _, row in frame.iterrows():
            value = row["date"]
            day = value.date() if hasattr(value, "date") else value
            self.rows.append(
                SimpleNamespace(date=day, open=float(row["open"]), close=float(row["close"]))
            )
        return len(frame)


def _config(**overrides):
    base = {
        "backtest_benchmark_enabled": True,
        "backtest_benchmark_us": "SPY",
        "backtest_benchmark_hk": "2800.HK",
        "backtest_benchmark_cn": "510300.SS",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _frame(count: int = 5):
    days = pd.date_range("2024-01-01", periods=count, freq="B")
    closes = [100.0 + index for index in range(count)]
    return pd.DataFrame(
        {
            "date": days,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * count,
        }
    )


def test_window_return_uses_cached_benchmark_bars():
    db = FakeDB()
    service = BenchmarkService(db, config=_config())

    with patch.object(service, "_download", return_value=_frame()):
        value = service.return_pct("us", date(2024, 1, 1), date(2024, 1, 5))

    assert value == 4.0
    assert service.code_for("us") == "SPY"
    assert db.rows  # cached into stock_daily


def test_snapshot_reports_risk_state_without_downloading():
    db = FakeDB()
    service = BenchmarkService(db, config=_config())
    with patch.object(service, "_download", return_value=_frame(30)):
        service.ensure("us", date(2024, 1, 1), date(2024, 2, 9))

    snapshot = service.snapshot("us", date(2024, 2, 9))

    assert snapshot["code"] == "SPY"
    assert snapshot["close"] == 129.0
    assert snapshot["above_ma200"] is False  # 不足 200 根，不能假装在上方
    assert snapshot["volatility_20d_pct"] is not None


def test_disabled_benchmark_returns_none():
    db = FakeDB()
    service = BenchmarkService(db, config=_config(backtest_benchmark_enabled=False))

    assert service.ensure("us", date(2024, 1, 1), date(2024, 1, 5)) is None
    assert service.return_pct("us", date(2024, 1, 1), date(2024, 1, 5)) is None

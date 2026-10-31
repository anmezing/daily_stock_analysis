# -*- coding: utf-8 -*-
"""Relative strength of a stock versus its market benchmark.

The score answers a different question than the absolute trend: a stock can be
rising while underperforming its index (laggard), or falling less than a
falling market (relative leader).  The US/HK pipelines previously had no such
input, so the model could not tell "真领涨" from "跟涨".

Pure computation only: no network, no database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _closes_from_history(history: Sequence[Dict[str, Any]]) -> List[float]:
    closes: List[float] = []
    for bar in history or []:
        if not isinstance(bar, dict):
            continue
        close = _to_float(bar.get("close"))
        if close is not None and close > 0:
            closes.append(close)
    return closes


def _return_pct(closes: Sequence[float], days: int) -> Optional[float]:
    if len(closes) <= days or days <= 0:
        return None
    base = closes[-1 - days]
    if base <= 0:
        return None
    return round((closes[-1] / base - 1.0) * 100.0, 2)


def _label(excess_20d: Optional[float], excess_60d: Optional[float]) -> str:
    excess = excess_20d if excess_20d is not None else excess_60d
    if excess is None:
        return ""
    if excess >= 3.0:
        return "强于基准（领涨）"
    if excess <= -3.0:
        return "弱于基准（跑输）"
    return "与基准同步"


def compute_relative_strength(
    history: Sequence[Dict[str, Any]],
    current_close: Any,
    benchmark: str = "SPY",
    benchmark_return_20d: Any = None,
    benchmark_return_60d: Any = None,
) -> Dict[str, Any]:
    """Return 20d/60d excess return versus the benchmark.

    ``history`` is ascending and normally excludes today; ``current_close`` is
    appended when present so the comparison includes the analysis day.
    """
    closes = _closes_from_history(history)
    close = _to_float(current_close)
    if close is not None and close > 0:
        # Avoid double-counting when the caller already included today.
        if not closes or abs(closes[-1] - close) > 1e-9:
            closes.append(close)
    if len(closes) < 6:
        return {}

    stock_20d = _return_pct(closes, 20)
    stock_60d = _return_pct(closes, 60)
    bench_20d = _to_float(benchmark_return_20d)
    bench_60d = _to_float(benchmark_return_60d)

    excess_20d = (
        round(stock_20d - bench_20d, 2)
        if stock_20d is not None and bench_20d is not None
        else None
    )
    excess_60d = (
        round(stock_60d - bench_60d, 2)
        if stock_60d is not None and bench_60d is not None
        else None
    )
    if stock_20d is None and stock_60d is None:
        return {}
    if excess_20d is None and excess_60d is None:
        # Without a benchmark return the "relative" block would be misleading.
        return {}

    return {
        "benchmark": benchmark or "benchmark",
        "stock_return_20d": stock_20d,
        "benchmark_return_20d": bench_20d,
        "excess_20d": excess_20d,
        "stock_return_60d": stock_60d,
        "benchmark_return_60d": bench_60d,
        "excess_60d": excess_60d,
        "label": _label(excess_20d, excess_60d),
    }

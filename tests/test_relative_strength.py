# -*- coding: utf-8 -*-
"""Tests for 20d/60d relative strength vs a benchmark."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.relative_strength import compute_relative_strength


def _history(slope: float, days: int = 80, start: float = 100.0):
    bars = []
    price = start
    for index in range(days):
        bars.append({"date": f"d{index}", "close": price})
        price *= 1.0 + slope
    return bars


class RelativeStrengthTest(unittest.TestCase):
    def test_outperforming_stock_is_labelled_leader(self) -> None:
        history = _history(0.004)
        current_close = history[-1]["close"] * 1.004

        result = compute_relative_strength(
            history,
            current_close,
            benchmark="SPY",
            benchmark_return_20d=2.0,
            benchmark_return_60d=5.0,
        )

        self.assertEqual(result["benchmark"], "SPY")
        self.assertGreater(result["excess_20d"], 3.0)
        self.assertEqual(result["label"], "强于基准（领涨）")

    def test_underperforming_stock_is_labelled_laggard(self) -> None:
        history = _history(0.0005)
        current_close = history[-1]["close"] * 1.0005

        result = compute_relative_strength(
            history,
            current_close,
            benchmark_return_20d=5.0,
            benchmark_return_60d=10.0,
        )

        self.assertLess(result["excess_20d"], -3.0)
        self.assertEqual(result["label"], "弱于基准（跑输）")

    def test_insufficient_history_returns_empty(self) -> None:
        self.assertEqual(compute_relative_strength(_history(0.01, days=5), 110), {})

    def test_missing_benchmark_return_returns_empty(self) -> None:
        # A "relative" block without a benchmark number is misleading.
        self.assertEqual(
            compute_relative_strength(_history(0.004), None, benchmark_return_20d=None),
            {},
        )


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""Tests for trading-session counting."""

import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.trading_calendar import count_trading_sessions


class TradingSessionCountTest(unittest.TestCase):
    def test_weekend_does_not_add_days(self) -> None:
        # Friday -> Monday is one session, not three calendar days.
        self.assertEqual(
            count_trading_sessions(date(2026, 10, 30), date(2026, 11, 2), "us"),
            1,
        )

    def test_same_day_is_zero(self) -> None:
        self.assertEqual(
            count_trading_sessions(date(2026, 11, 2), date(2026, 11, 2), "us"),
            0,
        )

    def test_reverse_range_is_zero(self) -> None:
        self.assertEqual(
            count_trading_sessions(date(2026, 11, 3), date(2026, 11, 2), "us"),
            0,
        )

    def test_full_week(self) -> None:
        self.assertEqual(
            count_trading_sessions(date(2026, 11, 2), date(2026, 11, 6), "us"),
            4,
        )


if __name__ == "__main__":
    unittest.main()

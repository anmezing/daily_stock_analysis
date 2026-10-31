# -*- coding: utf-8 -*-
"""Tests for intraday volume-ratio normalisation."""

import os
import sys
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.twelvedata_fetcher import TwelveDataFetcher


class VolumeRatioTest(unittest.TestCase):
    def test_30_minutes_is_normalised_to_full_session(self) -> None:
        # 3.2M shares in 30 of 390 minutes is a normal-to-heavy session,
        # not the 0.27 raw ratio.
        ratio = TwelveDataFetcher._normalize_volume_ratio(
            3_200_000,
            12_000_000,
            elapsed_minutes=30,
        )
        self.assertAlmostEqual(ratio, 3.47, places=2)

    def test_full_day_ratio_when_no_elapsed_minutes(self) -> None:
        ratio = TwelveDataFetcher._normalize_volume_ratio(3_200_000, 12_000_000)
        self.assertAlmostEqual(ratio, 0.2667, places=3)

    def test_invalid_ratio_is_dropped(self) -> None:
        self.assertIsNone(
            TwelveDataFetcher._normalize_volume_ratio(1_000, 12_000_000, elapsed_minutes=30)
        )
        self.assertIsNone(
            TwelveDataFetcher._normalize_volume_ratio(500_000_000, 10_000_000, elapsed_minutes=30)
        )
        self.assertIsNone(TwelveDataFetcher._normalize_volume_ratio(None, 10_000_000))

    def test_elapsed_minutes_for_us_session(self) -> None:
        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 10, 30, 10, 0, tzinfo=tz)
        elapsed = TwelveDataFetcher._elapsed_session_minutes(
            "us",
            quote_timestamp=now.timestamp(),
            now=now,
        )
        self.assertEqual(elapsed, 30)

    def test_elapsed_minutes_outside_session_is_none(self) -> None:
        tz = ZoneInfo("America/New_York")
        now = datetime(2026, 10, 30, 9, 0, tzinfo=tz)
        elapsed = TwelveDataFetcher._elapsed_session_minutes(
            "us",
            quote_timestamp=now.timestamp(),
            now=now,
        )
        self.assertIsNone(elapsed)


if __name__ == "__main__":
    unittest.main()

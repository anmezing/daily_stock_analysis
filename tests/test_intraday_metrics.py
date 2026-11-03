# -*- coding: utf-8 -*-
"""Tests for intraday volume-ratio fallback."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.intraday_metrics import normalize_intraday_volume_ratio


class IntradayMetricsTest(unittest.TestCase):
    def test_30_minutes_is_normalised(self) -> None:
        self.assertAlmostEqual(
            normalize_intraday_volume_ratio(3_200_000, 12_000_000, 30),
            3.47,
            places=2,
        )

    def test_full_day_without_elapsed(self) -> None:
        self.assertAlmostEqual(
            normalize_intraday_volume_ratio(3_200_000, 12_000_000, None),
            0.27,
            places=2,
        )

    def test_invalid_values_are_dropped(self) -> None:
        self.assertIsNone(normalize_intraday_volume_ratio(None, 12_000_000, 30))
        self.assertIsNone(normalize_intraday_volume_ratio(1_000, 12_000_000, 30))
        self.assertIsNone(normalize_intraday_volume_ratio(500_000_000, 10_000_000, 30))


if __name__ == "__main__":
    unittest.main()

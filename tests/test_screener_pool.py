# -*- coding: utf-8 -*-
"""Tests for the multi-day screener pool."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.screener_pool_service import ScreenerPoolService


def _result(symbols, sectors=("XLK",), scores=None):
    scores = scores or {}
    picks = []
    for symbol in symbols:
        picks.append(
            {
                "symbol": symbol,
                "name": symbol,
                "sector_etf": "XLK",
                "sector_name": "科技",
                "state": "pullback",
                "state_label": "提前布局",
                "score": scores.get(symbol, 80),
                "rs_15d": 2.0,
                "risk_level": "low",
                "revenue_growth": 0.2,
                "profit_margin": 0.2,
                "return_on_equity": 0.2,
                "free_cashflow": 1_000_000,
                "debt_to_equity": 50,
            }
        )
    return {
        "status": "ok",
        "as_of": None,
        "sectors": [{"symbol": symbol} for symbol in sectors],
        "picks": picks,
    }


class ScreenerPoolTest(unittest.TestCase):
    def test_streak_and_stage_progression(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ScreenerPoolService(path=str(Path(temp_dir) / "pool.json"))
            service.update(_result(["AAA"]), today="2026-10-27")
            item = service.load()["items"]["AAA"]
            self.assertEqual(item["consecutive_days"], 1)
            self.assertEqual(item["stage"], "new")

            service.update(_result(["AAA"]), today="2026-10-28")
            item = service.load()["items"]["AAA"]
            self.assertEqual(item["consecutive_days"], 2)
            self.assertEqual(item["stage"], "watching")

            service.update(_result(["AAA"]), today="2026-10-29")
            item = service.load()["items"]["AAA"]
            self.assertEqual(item["consecutive_days"], 3)
            self.assertEqual(item["stage"], "trigger")
            self.assertEqual(item["fundamental_trend"], "stable")

    def test_missing_ones_then_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ScreenerPoolService(path=str(Path(temp_dir) / "pool.json"))
            service.update(_result(["AAA"]), today="2026-10-27")
            service.update(_result([]), today="2026-10-28")
            item = service.load()["items"]["AAA"]
            self.assertEqual(item["stage"], "watching")
            self.assertEqual(item["missing_days"], 1)

            # The pool is cumulative: a few absent days must NOT drop it.
            from datetime import date as _date, timedelta as _timedelta
            base = _date(2026, 10, 27)
            for offset in range(2, 10):
                service.update(_result([]), today=(base + _timedelta(days=offset)).isoformat())
            item = service.load()["items"]["AAA"]
            self.assertNotEqual(item["stage"], "dropped")

            service.update(_result([]), today=(base + _timedelta(days=10)).isoformat())
            item = service.load()["items"]["AAA"]
            self.assertEqual(item["stage"], "dropped")

    def test_stop_alert_removes_from_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ScreenerPoolService(path=str(Path(temp_dir) / "pool.json"))
            service.update(_result(["AAA"]), today="2026-10-27")
            pool = service.apply_alerts([
                {"symbol": "AAA", "kind": "stop_breach", "message": "跌破止损"}
            ])
            self.assertEqual(pool["items"]["AAA"]["stage"], "dropped")

    def test_notification_only_on_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ScreenerPoolService(path=str(Path(temp_dir) / "pool.json"))
            service.update(_result(["AAA"]), today="2026-10-27")
            self.assertEqual(service.build_notification(service.load()), "")

            service.update(_result(["AAA"]), today="2026-10-28")
            self.assertEqual(service.build_notification(service.load()), "")

            service.update(_result(["AAA"]), today="2026-10-29")
            notice = service.build_notification(service.load())
            self.assertIn("股池新晋关注", notice)
            self.assertIn("AAA", notice)

            # Same stage tomorrow must not repeat the notification.
            service.update(_result(["AAA"]), today="2026-10-30")
            self.assertEqual(service.build_notification(service.load()), "")

    def test_markdown_contains_pool_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ScreenerPoolService(path=str(Path(temp_dir) / "pool.json"))
            service.update(_result(["AAA"]), today="2026-10-27")
            service.update(_result(["AAA"]), today="2026-10-28")
            markdown = service.render_markdown(service.load())
            self.assertIn("股池", markdown)
            self.assertIn("AAA", markdown)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""Tests for post-selection screener monitoring."""

import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.screener_learning_service import ScreenerLearningService, _HistoryRow
from src.services.screener_monitor_service import ScreenerMonitorService


class _Filings:
    def __init__(self, items):
        self.items = items

    def get_recent_filings(self, symbol, limit=10):
        return {"status": "ok", "items": self.items}


class ScreenerMonitorTest(unittest.TestCase):
    def _service(self, temp_dir, filings=None):
        learning = ScreenerLearningService(
            ledger_path=str(Path(temp_dir) / "picks.jsonl"),
            state_path=str(Path(temp_dir) / "learning.json"),
            history_provider=lambda symbols: {},
        )
        pick_day = date.today() - timedelta(days=10)
        learning._write_ledger([
            {
                "screen_date": pick_day.isoformat(),
                "symbol": "AAA",
                "name": "Alpha",
                "sector_etf": "XLK",
                "state": "pullback",
                "plan": {"stop_loss": 90, "target_1": 120},
                "earnings_date": (date.today() + timedelta(days=2)).isoformat(),
            }
        ])
        history = {
            "AAA": [
                _HistoryRow(day=date.today() - timedelta(days=1), close=85),
            ]
        }
        learning._default_history = lambda symbols: history
        monitor = ScreenerMonitorService(
            learning_service=learning,
            filing_provider=_Filings(filings or []),
            notify=False,
        )
        return learning, monitor

    def test_stop_and_earnings_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            learning, monitor = self._service(temp_dir)
            result = monitor.check()
            kinds = {alert["kind"] for alert in result["alerts"]}
            self.assertIn("stop_breach", kinds)
            self.assertIn("earnings_soon", kinds)
            record = learning._read_ledger()[0]
            self.assertTrue(record.get("closed"))

    def test_new_offering_filing_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            filings = [
                {
                    "form": "424B5",
                    "filed": (date.today() - timedelta(days=1)).isoformat(),
                }
            ]
            _, monitor = self._service(temp_dir, filings=filings)
            result = monitor.check()
            kinds = {alert["kind"] for alert in result["alerts"]}
            self.assertIn("dilution", kinds)

    def test_old_filing_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            filings = [
                {
                    "form": "424B5",
                    "filed": (date.today() - timedelta(days=40)).isoformat(),
                }
            ]
            _, monitor = self._service(temp_dir, filings=filings)
            result = monitor.check()
            kinds = {alert["kind"] for alert in result["alerts"]}
            self.assertNotIn("dilution", kinds)


if __name__ == "__main__":
    unittest.main()

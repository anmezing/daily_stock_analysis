# -*- coding: utf-8 -*-
"""Tests for the screener self-learning loop."""

import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.screener_learning_service import (
    ScreenerLearningService,
    _HistoryRow,
    compute_adjustments,
)


def _record(state: str, excess: float, sector: str = "XLK") -> dict:
    return {
        "screen_date": "2026-01-01",
        "symbol": "AAA",
        "sector_etf": sector,
        "state": state,
        "outcomes": {"d10": {"excess_spy": excess}},
    }


class ComputeAdjustmentsTest(unittest.TestCase):
    def test_good_state_gets_bonus_and_bad_state_is_disabled(self) -> None:
        records = [_record("pullback", 2.0) for _ in range(25)]
        records += [_record("breakout", -3.0) for _ in range(35)]

        adjustments = compute_adjustments(records, min_samples=20)

        self.assertEqual(adjustments["state_bonus"].get("pullback"), 3.0)
        self.assertIn("breakout", adjustments["disabled_states"])

    def test_small_sample_does_not_change_weights(self) -> None:
        records = [_record("pullback", 5.0) for _ in range(5)]
        adjustments = compute_adjustments(records, min_samples=20)
        self.assertEqual(adjustments["state_bonus"], {})
        self.assertEqual(adjustments["disabled_states"], [])

    def test_low_overall_hit_rate_raises_min_score(self) -> None:
        records = [_record("watch", -1.0) for _ in range(25)]
        adjustments = compute_adjustments(records, min_samples=20)
        self.assertEqual(adjustments["min_score_delta"], 5.0)


class LearningLedgerTest(unittest.TestCase):
    def test_record_evaluate_and_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "picks.jsonl"
            state = Path(temp_dir) / "learning.json"
            pick_day = date.today() - timedelta(days=40)
            rows = [
                _HistoryRow(day=pick_day + timedelta(days=index), close=100.0 + index)
                for index in range(35)
            ]
            history = {"AAA": rows, "XLK": rows, "SPY": rows}
            service = ScreenerLearningService(
                ledger_path=str(ledger),
                state_path=str(state),
                history_provider=lambda symbols: history,
                min_samples=5,
            )

            written = service.record_picks(
                {
                    "status": "ok",
                    "picks": [
                        {
                            "symbol": "AAA",
                            "name": "Alpha",
                            "sector_etf": "XLK",
                            "sector_name": "科技",
                            "state": "pullback",
                            "score": 80,
                            "close": 100,
                        }
                    ],
                },
                screen_date=pick_day.isoformat(),
            )
            self.assertEqual(written, 1)

            evaluated = service.evaluate_pending()
            self.assertEqual(evaluated, 1)
            records = service._read_ledger()
            self.assertIn("d5", records[0]["outcomes"])
            self.assertIn("excess_spy", records[0]["outcomes"]["d10"])

            adjustments = service.refresh_adjustments()
            self.assertEqual(adjustments["overall"]["samples"], 1)
            self.assertTrue(state.exists())

    def test_apply_adjustments_filters_disabled_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "picks.jsonl"
            state = Path(temp_dir) / "learning.json"
            state.write_text(
                '{"state_bonus":{"pullback":3},"sector_bonus":{},"disabled_states":["weak"],'
                '"disabled_sectors":[],"min_score_delta":5}',
                encoding="utf-8",
            )
            service = ScreenerLearningService(
                ledger_path=str(ledger),
                state_path=str(state),
                history_provider=lambda symbols: {},
            )
            picks, stats = service.apply_adjustments(
                [
                    {"symbol": "AAA", "state": "pullback", "sector_etf": "XLK", "score": 80},
                    {"symbol": "BBB", "state": "weak", "sector_etf": "XLK", "score": 90},
                ]
            )
            self.assertEqual([pick["symbol"] for pick in picks], ["AAA"])
            self.assertEqual(picks[0]["score"], 83)
            self.assertEqual(stats["adjustment_removed"], 1)
            self.assertEqual(stats["min_score_delta"], 5)


if __name__ == "__main__":
    unittest.main()

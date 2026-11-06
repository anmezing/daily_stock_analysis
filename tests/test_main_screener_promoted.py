# -*- coding: utf-8 -*-
"""Tests for auto-analysis of newly promoted screener pool names."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

import main


def _pool(**items):
    return {"items": {symbol: dict(symbol=symbol, **fields) for symbol, fields in items.items()}}


class SelectPromotedTest(unittest.TestCase):
    def test_only_today_promotions_trigger_first(self) -> None:
        pool = _pool(
            AAA={"promoted_today": True, "stage": "ready", "last_score": 80},
            BBB={"promoted_today": True, "stage": "trigger", "last_score": 70},
            CCC={"promoted_today": False, "stage": "ready", "last_score": 90},
            DDD={"promoted_today": True, "stage": "dropped", "last_score": 95},
            EEE={"promoted_today": False, "stage": "watching", "last_score": 99},
        )
        self.assertEqual(main.select_promoted_for_analysis(pool), ["BBB", "AAA"])

    def test_list_is_capped(self) -> None:
        pool = _pool(**{
            f"S{i}": {"promoted_today": True, "stage": "ready", "last_score": i}
            for i in range(12)
        })
        self.assertEqual(len(main.select_promoted_for_analysis(pool)), main.MAX_PROMOTED_ANALYSIS)

    def test_empty_pool(self) -> None:
        self.assertEqual(main.select_promoted_for_analysis({}), [])


class AnalyzePromotedPoolTest(unittest.TestCase):
    def _pool_service(self, pool):
        return SimpleNamespace(load=lambda: pool)

    def test_runs_stock_analysis_without_market_review(self) -> None:
        pool = _pool(AAA={"promoted_today": True, "stage": "ready", "last_score": 80})
        args = argparse.Namespace(no_market_review=False, screen_schedule=True)
        with patch(
            "src.services.screener_pool_service.get_screener_pool_service",
            return_value=self._pool_service(pool),
        ), patch.object(main, "run_full_analysis") as run:
            codes = main.analyze_promoted_pool(SimpleNamespace(), args)

        self.assertEqual(codes, ["AAA"])
        run.assert_called_once()
        _, called_args, called_codes = run.call_args.args
        self.assertTrue(called_args.no_market_review)
        self.assertEqual(called_codes, ["AAA"])
        # The caller's args are not mutated.
        self.assertFalse(args.no_market_review)

    def test_no_promotions_skips_analysis(self) -> None:
        pool = _pool(AAA={"promoted_today": False, "stage": "ready"})
        with patch(
            "src.services.screener_pool_service.get_screener_pool_service",
            return_value=self._pool_service(pool),
        ), patch.object(main, "run_full_analysis") as run:
            self.assertEqual(main.analyze_promoted_pool(SimpleNamespace(), argparse.Namespace()), [])
        run.assert_not_called()

    def test_pool_read_failure_skips_analysis(self) -> None:
        def boom():
            raise OSError("disk")

        with patch(
            "src.services.screener_pool_service.get_screener_pool_service",
            return_value=SimpleNamespace(load=boom),
        ), patch.object(main, "run_full_analysis") as run:
            self.assertEqual(main.analyze_promoted_pool(SimpleNamespace(), argparse.Namespace()), [])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

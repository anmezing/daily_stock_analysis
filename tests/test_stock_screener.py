# -*- coding: utf-8 -*-
"""Tests for the hot-sector stock screener."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timedelta

from src.services.stock_screener_service import (
    Candidate,
    StockScreenerService,
    _next_earnings_day,
    score_metrics,
)


def _series(start: float = 100.0, slope: float = 0.003, days: int = 120, volume: float = 2_000_000):
    closes = []
    highs = []
    lows = []
    volumes = []
    price = start
    for index in range(days):
        closes.append(price)
        highs.append(price * 1.01)
        lows.append(price * 0.99)
        volumes.append(volume)
        price *= 1.0 + slope
    return {"close": closes, "high": highs, "low": lows, "volume": volumes}


class _Themes:
    def __init__(self, status: str = "ok"):
        self.status = status

    def get_snapshot(self, market: str):
        if self.status != "ok":
            return {"status": self.status, "market": market}
        return {
            "status": "ok",
            "market": market,
            "risk_on": True,
            "leaders": [
                {"symbol": "XLK", "name": "科技", "return_5d": 3.0},
                {"symbol": "XLF", "name": "金融", "return_5d": 2.0},
                {"symbol": "XLV", "name": "医疗", "return_5d": 1.0},
            ],
        }


def _candidates(etf: str, name: str, rank: int):
    mapping = {
        "XLK": [("AAA", "Alpha Tech"), ("BBB", "Beta Tech")],
        "XLF": [("CCC", "Gamma Finance")],
        "XLV": [("DDD", "Delta Health")],
    }
    return [Candidate(symbol, label, etf, name, rank) for symbol, label in mapping.get(etf, [])]


def _history(symbols):
    data = {}
    for symbol in symbols:
        if symbol in ("XLK", "XLF", "XLV"):
            data[symbol] = _series(slope=0.0005, volume=5_000_000)
        elif symbol == "DDD":
            data[symbol] = _series(slope=-0.001, volume=2_000_000)
        else:
            data[symbol] = _series(slope=0.004, volume=8_000_000)
    return data


class ScoreMetricsTest(unittest.TestCase):
    def test_pullback_setup_gets_top_bucket(self) -> None:
        result = score_metrics(
            {
                "close": 100,
                "ma50": 90,
                "rs_15d": 2.5,
                "rs_30d": 4.0,
                "dist_ma20_pct": 1.0,
                "volume_ratio_5_20": 0.8,
                "high_20d_proximity": 0.95,
                "dollar_volume_20d": 120_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        self.assertEqual(result["state"], "pullback")
        self.assertGreater(result["score"], 80)

    def test_extended_and_weak_are_penalised(self) -> None:
        extended = score_metrics(
            {
                "close": 100,
                "ma50": 90,
                "rs_15d": 5.0,
                "rs_30d": 8.0,
                "dist_ma20_pct": 12.0,
                "volume_ratio_5_20": 1.2,
                "high_20d_proximity": 1.0,
                "dollar_volume_20d": 120_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        weak = score_metrics(
            {
                "close": 100,
                "ma50": 110,
                "rs_15d": -2.0,
                "rs_30d": -4.0,
                "dist_ma20_pct": 1.0,
                "volume_ratio_5_20": 0.8,
                "high_20d_proximity": 0.95,
                "dollar_volume_20d": 120_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        self.assertEqual(extended["state"], "extended")
        self.assertEqual(weak["state"], "weak")
        self.assertGreater(extended["score"], weak["score"])

    def test_illiquid_stock_is_dropped(self) -> None:
        result = score_metrics(
            {
                "close": 100,
                "ma50": 90,
                "rs_15d": 2.0,
                "rs_30d": 3.0,
                "dist_ma20_pct": 1.0,
                "volume_ratio_5_20": 0.8,
                "high_20d_proximity": 0.95,
                "dollar_volume_20d": 1_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        self.assertEqual(result["score"], 0.0)

    def test_missing_sector_benchmark_is_rejected(self) -> None:
        result = score_metrics(
            {
                "close": 100,
                "ma50": 90,
                "rs_15d": None,
                "rs_30d": None,
                "dist_ma20_pct": 1.0,
                "volume_ratio_5_20": 0.8,
                "high_20d_proximity": 0.95,
                "dollar_volume_20d": 120_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        self.assertEqual(result["score"], 0.0)
        self.assertIn("缺少板块基准", result["reason"])

    def test_risk_off_reduces_score(self) -> None:
        metrics = {
            "close": 100,
            "ma50": 90,
            "rs_15d": 2.5,
            "rs_30d": 4.0,
            "dist_ma20_pct": 1.0,
            "volume_ratio_5_20": 0.8,
            "high_20d_proximity": 0.95,
            "dollar_volume_20d": 120_000_000,
            "atr_pct": 3.0,
        }
        risk_on = score_metrics(metrics, sector_rank=1, risk_on=True)
        risk_off = score_metrics(metrics, sector_rank=1, risk_on=False)
        self.assertLess(risk_off["score"], risk_on["score"])
        self.assertIn("risk-off", risk_off["reason"])


class StockScreenerTest(unittest.TestCase):
    def _service(self, themes_status: str = "ok") -> StockScreenerService:
        return StockScreenerService(
            themes_service=_Themes(themes_status),
            candidate_provider=_candidates,
            history_provider=_history,
            fundamental_provider=lambda symbols: {},
            cache_ttl_seconds=0,
        )

    def test_screen_returns_ranked_picks_with_plan(self) -> None:
        result = self._service().screen(top_n=3)

        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(len(result["picks"]), 3)
        self.assertTrue(result["picks"])
        self.assertIn("今日主线", result["summary"])
        first = result["picks"][0]
        self.assertGreater(first["score"], 0)
        self.assertIn("buy_trigger", first["plan"])
        self.assertIsNotNone(first["plan"]["stop_loss"])

    def test_hk_is_not_supported(self) -> None:
        result = self._service().screen(market="hk")
        self.assertEqual(result["status"], "not_supported")

    def test_missing_themes_is_not_supported(self) -> None:
        result = self._service(themes_status="not_supported").screen()
        self.assertEqual(result["status"], "not_supported")

    def test_render_markdown_contains_table_and_plan(self) -> None:
        markdown = StockScreenerService.render_markdown(self._service().screen(top_n=2))
        self.assertIn("热门板块选股", markdown)
        self.assertIn("| 板块 |", markdown)
        self.assertIn("交易计划", markdown)

    def test_earnings_window_filter_skips_name(self) -> None:
        service = StockScreenerService(
            themes_service=_Themes(),
            candidate_provider=_candidates,
            history_provider=_history,
            fundamental_provider=lambda symbols: {
                symbol: {"days_to_earnings": 2, "quality_ok": True}
                for symbol in symbols
            },
            cache_ttl_seconds=0,
        )
        result = service.screen(top_n=3)
        self.assertEqual(result["picks"], [])
        self.assertGreater(result["filters"]["earnings_skipped"], 0)

    def _earnings_service(self, info):
        return StockScreenerService(
            themes_service=_Themes(),
            candidate_provider=_candidates,
            history_provider=_history,
            fundamental_provider=lambda symbols: {symbol: dict(info) for symbol in symbols},
            cache_ttl_seconds=0,
        )

    def test_earnings_inside_holding_horizon_is_skipped(self) -> None:
        # 12 calendar days out was outside the old 3-day window, but a
        # 10-session time stop would hold through the report.
        result = self._earnings_service({"days_to_earnings": 12, "quality_ok": True}).screen(top_n=3)
        self.assertEqual(result["picks"], [])
        self.assertGreater(result["filters"]["earnings_skipped"], 0)

    def test_earnings_beyond_holding_horizon_is_kept(self) -> None:
        far = (datetime.now() + timedelta(days=60)).date().isoformat()
        result = self._earnings_service(
            {"earnings_date": far, "days_to_earnings": 60, "quality_ok": True}
        ).screen(top_n=3)
        self.assertTrue(result["picks"])
        self.assertEqual(result["filters"]["earnings_skipped"], 0)

    def test_next_earnings_day_skips_past_report(self) -> None:
        past = (datetime.now() - timedelta(days=80)).timestamp()
        upcoming = datetime.now() + timedelta(days=5)
        self.assertEqual(_next_earnings_day(past, upcoming.timestamp()), upcoming.date())
        self.assertIsNone(_next_earnings_day(past, None))

    def test_sector_mismatch_is_skipped(self) -> None:
        # DDD is screened under XLV but its own sector is consumer-defensive
        # (the BTI case): its RS-vs-XLV score is meaningless.
        sectors = {"AAA": "technology", "BBB": "technology", "CCC": "financial-services",
                   "DDD": "consumer-defensive"}
        service = StockScreenerService(
            themes_service=_Themes(),
            candidate_provider=_candidates,
            history_provider=_history,
            fundamental_provider=lambda symbols: {
                symbol: {"quality_ok": True, "sector_key": sectors.get(symbol)}
                for symbol in symbols
            },
            cache_ttl_seconds=0,
        )
        result = service.screen(top_n=5)
        symbols = {pick["symbol"] for pick in result["picks"]}
        self.assertNotIn("DDD", symbols)
        self.assertTrue(symbols)
        self.assertEqual(result["filters"]["sector_mismatch_skipped"], 1)
        self.assertIn("板块不符 1 只", StockScreenerService.render_markdown(result))

    def test_unknown_sector_is_kept(self) -> None:
        result = self._earnings_service({"quality_ok": True}).screen(top_n=5)
        self.assertTrue(result["picks"])
        self.assertEqual(result["filters"]["sector_mismatch_skipped"], 0)

    def test_unverified_fundamentals_are_flagged_in_report(self) -> None:
        # The fixture's correlation filter leaves only AAA; make it unverified.
        result = self._earnings_service({"quality_ok": None}).screen(top_n=5)
        self.assertEqual([pick["symbol"] for pick in result["picks"]], ["AAA"])

        markdown = StockScreenerService.render_markdown(result, top_n=5)
        self.assertIn("基本面未核验 1 只（AAA）", markdown)
        row = next(line for line in markdown.splitlines() if line.startswith("|") and "| AAA |" in line)
        self.assertIn("⚠️未核验", row)
        self.assertIn("- ⚠️ 基本面未核验：质量过滤未生效", markdown)

    def test_fully_verified_report_has_no_warning(self) -> None:
        result = self._earnings_service({"quality_ok": True}).screen(top_n=5)
        self.assertNotIn("未核验", StockScreenerService.render_markdown(result, top_n=5))

    def test_short_term_pop_without_mid_term_leadership_is_skipped(self) -> None:
        def ramp(days: int, flat: int, slope: float, volume: float):
            series = _series(slope=0.0, days=flat, volume=volume)
            tail = _series(start=series["close"][-1], slope=slope, days=days - flat, volume=volume)
            return {key: series[key] + tail[key] for key in series}

        def history(symbols):
            data = {}
            for symbol in symbols:
                if symbol in ("XLK", "XLF", "XLV"):
                    data[symbol] = _series(slope=0.0005, days=200, volume=5_000_000)
                elif symbol == "AAA":
                    # Flat for ~6 months, then a 30-day pop: strong 15/30-day
                    # numbers but it lagged XLK over the 6-1 month window.
                    data[symbol] = ramp(200, 170, 0.004, 8_000_000)
                else:
                    data[symbol] = _series(slope=0.004, days=200, volume=8_000_000)
            return data

        service = StockScreenerService(
            themes_service=_Themes(),
            candidate_provider=_candidates,
            history_provider=history,
            fundamental_provider=lambda symbols: {s: {"quality_ok": True} for s in symbols},
            cache_ttl_seconds=0,
        )
        result = service.screen(top_n=5)
        symbols = {pick["symbol"] for pick in result["picks"]}
        self.assertNotIn("AAA", symbols)
        self.assertTrue(symbols)
        self.assertGreaterEqual(result["filters"]["mid_momentum_skipped"], 1)
        self.assertTrue(all(pick["rs_6_1"] > 0 for pick in result["picks"]))
        self.assertIn("中期动量", StockScreenerService.render_markdown(result))

    def test_short_history_skips_mid_term_gate(self) -> None:
        # Default fixture has 120 bars (< 6 months): the gate must not fire.
        result = self._earnings_service({"quality_ok": True}).screen(top_n=5)
        self.assertTrue(result["picks"])
        self.assertEqual(result["filters"]["mid_momentum_skipped"], 0)
        self.assertIsNone(result["picks"][0]["rs_6_1"])

    def test_low_quality_name_is_skipped(self) -> None:
        service = StockScreenerService(
            themes_service=_Themes(),
            candidate_provider=_candidates,
            history_provider=_history,
            fundamental_provider=lambda symbols: {
                symbol: {"quality_ok": False, "quality_notes": ["净利率为负"]}
                for symbol in symbols
            },
            cache_ttl_seconds=0,
        )
        result = service.screen(top_n=3)
        self.assertEqual(result["picks"], [])
        self.assertGreater(result["filters"]["quality_skipped"], 0)

    def test_correlated_names_are_deduplicated(self) -> None:
        result = self._service().screen(top_n=10)
        symbols = [item["symbol"] for item in result["picks"]]
        # AAA and BBB use the same synthetic series -> only one should survive.
        self.assertFalse({"AAA", "BBB"}.issubset(set(symbols)))
        self.assertGreater(result["filters"]["correlation_skipped"], 0)

    def test_walk_forward_validation_runs(self) -> None:
        service = self._service()
        candidates = [
            Candidate("AAA", "Alpha Tech", "XLK", "科技", 1),
            Candidate("CCC", "Gamma Finance", "XLF", "金融", 2),
        ]
        history = {
            "SPY": _series(slope=0.0005),
            "XLK": _series(slope=0.0005),
            "XLF": _series(slope=0.0005),
            "XLV": _series(slope=0.0005),
            "AAA": _series(slope=0.0015),
            "CCC": _series(slope=0.0010),
        }
        result = service._walk_forward_validation(
            candidates=candidates,
            history=history,
            top_n=2,
            risk_on=True,
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["samples"], 0)
        self.assertIn("hit_rate_10d", result)


if __name__ == "__main__":
    unittest.main()

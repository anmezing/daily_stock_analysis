# -*- coding: utf-8 -*-
"""Tests for the deterministic trade-plan reference."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analyzer import AnalysisResult
from src.core.trade_plan import (
    apply_trade_plan_fallback,
    build_trade_plan,
    compute_atr,
    render_trade_plan_lines,
)


def _bars(start: float, step: float, count: int = 30):
    bars = []
    price = start
    for index in range(count):
        high = price * 1.01
        low = price * 0.99
        bars.append(
            {
                "date": f"2026-10-{index + 1:02d}",
                "open": price,
                "high": high,
                "low": low,
                "close": price,
            }
        )
        price += step
    return bars


class TestTradePlan(unittest.TestCase):
    def test_atr_requires_two_bars(self) -> None:
        self.assertIsNone(compute_atr([]))
        self.assertIsNone(compute_atr([{"high": 10, "low": 9, "close": 9.5}]))

    def test_uptrend_plan_has_actionable_levels(self) -> None:
        context = {
            "code": "AAPL",
            "today": {"close": 100.0, "ma5": 99.0, "ma10": 98.0, "ma20": 95.0},
            "history": _bars(80.0, 0.7, 40),
            "trend_analysis": {"trend_status": "强势多头"},
        }

        plan = build_trade_plan(context)

        self.assertTrue(plan["ready"])
        self.assertIsNotNone(plan["entry_low"])
        self.assertIsNotNone(plan["entry_high"])
        self.assertLessEqual(plan["entry_low"], plan["entry_high"])
        self.assertLessEqual(plan["entry_high"], 100.0)
        self.assertLess(plan["stop_loss"], plan["entry_low"])
        self.assertGreater(plan["target_1"], plan["entry_high"])
        self.assertGreater(plan["target_2"], plan["target_1"])
        self.assertGreater(plan["risk_reward"], 0)
        self.assertIn("回踩", plan["buy_trigger"])
        self.assertIn("止损", plan["sell_trigger"])
        self.assertTrue(plan["wait_condition"])
        self.assertTrue(plan["invalidation"])
        self.assertTrue(plan["time_stop"])

    def test_below_ma20_plan_tells_user_to_wait(self) -> None:
        context = {
            "code": "AAPL",
            "today": {"close": 90.0, "ma5": 92.0, "ma10": 95.0, "ma20": 100.0},
            "history": _bars(110.0, -0.6, 40),
            "trend_analysis": {"trend_status": "强势空头"},
        }

        plan = build_trade_plan(context)

        self.assertTrue(plan["ready"])
        self.assertIn("观望", plan["wait_condition"])
        self.assertIn("站上 MA20", plan["buy_trigger"])
        self.assertLess(plan["stop_loss"], plan["entry_low"])

    def test_missing_price_returns_not_ready(self) -> None:
        plan = build_trade_plan({"code": "AAPL", "today": {}})
        self.assertFalse(plan["ready"])

    def test_fallback_fills_missing_plan_and_sniper_points(self) -> None:
        plan = build_trade_plan(
            {
                "code": "AAPL",
                "today": {"close": 100.0, "ma5": 99.0, "ma10": 98.0, "ma20": 95.0},
                "history": _bars(80.0, 0.7, 40),
                "trend_analysis": {"trend_status": "多头"},
            }
        )
        result = AnalysisResult(
            code="AAPL",
            name="Apple",
            sentiment_score=72,
            trend_prediction="看多",
            operation_advice="买入",
            decision_type="buy",
            dashboard={
                "core_conclusion": {"one_sentence": "回踩买入"},
                "intelligence": {"risk_alerts": []},
                "battle_plan": {"sniper_points": {}, "trade_plan": {}},
            },
        )

        apply_trade_plan_fallback(result, plan)

        trade_plan = result.dashboard["battle_plan"]["trade_plan"]
        self.assertEqual(trade_plan["buy_trigger"], plan["buy_trigger"])
        self.assertIn("–", result.dashboard["battle_plan"]["sniper_points"]["ideal_buy"])
        self.assertTrue(result.dashboard["battle_plan"]["sniper_points"]["stop_loss"])
        self.assertTrue(result.dashboard["battle_plan"]["sniper_points"]["take_profit"])

    def test_fallback_keeps_model_levels(self) -> None:
        plan = build_trade_plan(
            {
                "code": "AAPL",
                "today": {"close": 100.0, "ma10": 98.0, "ma20": 95.0},
                "history": _bars(80.0, 0.7, 40),
            }
        )
        result = AnalysisResult(
            code="AAPL",
            name="Apple",
            sentiment_score=72,
            trend_prediction="看多",
            operation_advice="买入",
            decision_type="buy",
            dashboard={
                "battle_plan": {
                    "sniper_points": {"stop_loss": "90", "take_profit": "130"},
                    "trade_plan": {"buy_trigger": "自定义触发"},
                }
            },
        )

        apply_trade_plan_fallback(result, plan)

        sniper = result.dashboard["battle_plan"]["sniper_points"]
        self.assertEqual(sniper["stop_loss"], "90")
        self.assertEqual(sniper["take_profit"], "130")
        self.assertEqual(
            result.dashboard["battle_plan"]["trade_plan"]["buy_trigger"],
            "自定义触发",
        )

    def test_render_trade_plan_lines(self) -> None:
        dashboard = {
            "battle_plan": {
                "trade_plan": {
                    "buy_trigger": "回踩 98–100 买入",
                    "wait_condition": "高于 105 不追",
                    "sell_trigger": "跌破 90 止损",
                    "invalidation": "跌破 95",
                    "time_stop": "10 日不涨退出",
                    "risk_reward": 2.0,
                }
            }
        }

        lines = render_trade_plan_lines(dashboard)

        text = "\n".join(lines)
        self.assertIn("交易计划", text)
        self.assertIn("回踩 98–100 买入", text)
        self.assertIn("1 : 2", text)

    def test_render_returns_empty_without_plan(self) -> None:
        self.assertEqual(render_trade_plan_lines({}), [])
        self.assertEqual(render_trade_plan_lines(None), [])


if __name__ == "__main__":
    unittest.main()

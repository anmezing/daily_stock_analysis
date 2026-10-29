# -*- coding: utf-8 -*-
"""Portfolio backtest: cross-sectional ranking, sizing and equity curve."""

import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from src.config import Config
from src.services.portfolio_backtest_service import (
    PortfolioBacktestConfig,
    PortfolioBacktestService,
)
from src.storage import AnalysisHistory, DatabaseManager, StockDaily


def _business_days(start: date, count: int):
    days = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


class PortfolioBacktestTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = os.path.join(self._temp_dir.name, "portfolio.db")
        self._old_env = {
            key: os.environ.get(key)
            for key in ("ENV_FILE", "DATABASE_URL", "DATABASE_PATH", "BACKTEST_BENCHMARK_ENABLED")
        }
        os.environ["ENV_FILE"] = os.path.join(self._temp_dir.name, "empty.env")
        os.environ.pop("DATABASE_URL", None)
        os.environ["DATABASE_PATH"] = self._db_path
        os.environ["BACKTEST_BENCHMARK_ENABLED"] = "false"
        Config._instance = None
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()

        codes = (("AAPL", 90), ("MSFT", 80), ("NVDA", 70))
        with self.db.get_session() as session:
            for index, (code, score) in enumerate(codes):
                session.add(
                    AnalysisHistory(
                        query_id=f"q-{code}",
                        code=code,
                        name=code,
                        report_type="simple",
                        sentiment_score=score,
                        operation_advice="买入",
                        trend_prediction="看多",
                        analysis_summary="test",
                        raw_result=json.dumps(
                            {
                                "model_used": "gemini/gemini-2.5-flash",
                                "success": True,
                                "decision_type": "buy",
                                "operation_advice": "买入",
                                "sentiment_score": score,
                            }
                        ),
                        created_at=datetime(2024, 1, 3, 10, 0, 0),
                        context_snapshot='{"enhanced_context": {"date": "2024-01-03"}}',
                    )
                )
            session.commit()

            for code_index, (code, _) in enumerate(codes):
                for day_index, day in enumerate(_business_days(date(2024, 1, 2), 12)):
                    price = 10.0 + code_index + day_index * 0.1
                    session.add(
                        StockDaily(
                            code=code,
                            date=day,
                            open=price,
                            high=price + 0.1,
                            low=price - 0.1,
                            close=price + 0.05,
                            volume=1_000_000,
                        )
                    )
            session.commit()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config._instance = None
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._temp_dir.cleanup()

    def test_portfolio_holds_top_ranked_signals_only(self) -> None:
        service = PortfolioBacktestService(self.db, config=SimpleNamespace())
        cfg = PortfolioBacktestConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 16),
            initial_capital=100_000.0,
            max_positions=2,
            holding_days=3,
            min_score=60,
            commission_bps=0.0,
            slippage_bps=0.0,
            benchmark_market="us",
        )
        with patch.object(service.benchmark, "return_pct", return_value=1.0):
            result = service.run(cfg)

        self.assertEqual(len(result.trades), 2)
        self.assertEqual({trade["code"] for trade in result.trades}, {"AAPL", "MSFT"})
        self.assertGreater(result.metrics["total_return_pct"], 0)
        self.assertLessEqual(result.metrics["max_drawdown_pct"], 0)
        self.assertGreater(result.metrics["avg_exposure_pct"], 0)
        self.assertEqual(result.metrics["benchmark_return_pct"], 1.0)
        self.assertAlmostEqual(
            result.metrics["excess_return_pct"],
            round(result.metrics["total_return_pct"] - 1.0, 2),
        )
        self.assertTrue(result.equity_curve)

    def test_low_score_signals_are_filtered(self) -> None:
        service = PortfolioBacktestService(self.db, config=SimpleNamespace())
        cfg = PortfolioBacktestConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 16),
            max_positions=5,
            holding_days=3,
            min_score=85,
            slippage_bps=0.0,
        )
        with patch.object(service.benchmark, "return_pct", return_value=0.0):
            result = service.run(cfg)

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0]["code"], "AAPL")

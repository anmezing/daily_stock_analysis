# -*- coding: utf-8 -*-
"""Integration tests for backtest service and repository.

These tests run against a temporary SQLite DB (same approach as other tests)
and validate idempotency/force semantics, result field correctness,
summary creation, and query methods.
"""

import json
import os
import tempfile
import unittest
from datetime import date, datetime
from unittest.mock import patch

from src.config import Config
from src.core.backtest_engine import EXECUTION_LOGIC_REVISION, OVERALL_SENTINEL_CODE
from src.services.backtest_service import BacktestService
from src.repositories.backtest_repo import BacktestRepository
from src.storage import AnalysisHistory, BacktestResult, BacktestSummary, DatabaseManager, StockDaily


class BacktestServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = os.path.join(self._temp_dir.name, "test_backtest_service.db")
        self._old_env = {
            key: os.environ.get(key)
            for key in (
                "ENV_FILE",
                "DATABASE_URL",
                "DATABASE_PATH",
                "BACKTEST_ENTRY_MODE",
                "BACKTEST_EVAL_WINDOW_DAYS",
                "BACKTEST_EXTRA_WINDOWS",
                "BACKTEST_SLIPPAGE_BPS",
                "BACKTEST_BENCHMARK_ENABLED",
            )
        }
        os.environ["ENV_FILE"] = os.path.join(self._temp_dir.name, "empty.env")
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("BACKTEST_ENTRY_MODE", None)
        os.environ["DATABASE_PATH"] = self._db_path
        os.environ["BACKTEST_EVAL_WINDOW_DAYS"] = "3"
        os.environ["BACKTEST_EXTRA_WINDOWS"] = ""
        # 这些用例验证执行逻辑，不把成本模型混进来；成本单独测。
        os.environ["BACKTEST_SLIPPAGE_BPS"] = "0"
        os.environ["BACKTEST_BENCHMARK_ENABLED"] = "false"

        Config._instance = None
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()

        # Ensure analysis is old enough for default min_age_days=14
        old_created_at = datetime(2024, 1, 1, 0, 0, 0)

        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id="q1",
                    code="600519",
                    name="贵州茅台",
                    report_type="simple",
                    sentiment_score=80,
                    operation_advice="买入",
                    trend_prediction="看多",
                    analysis_summary="test",
                    raw_result='{"model_used": "gemini/gemini-2.5-flash", "success": true, "parse_mode": "structured", "decision_type": "buy"}',
                    stop_loss=95.0,
                    take_profit=110.0,
                    strategy_id="bull_trend",
                    created_at=old_created_at,
                    context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                )
            )

            # Analysis day close
            session.add(
                StockDaily(
                    code="600519",
                    date=date(2024, 1, 1),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                )
            )

            # Forward bars (3 days) that hit take-profit on day1
            session.add_all(
                [
                    StockDaily(code="600519", date=date(2024, 1, 2), open=100.0, high=111.0, low=100.0, close=105.0),
                    StockDaily(code="600519", date=date(2024, 1, 3), open=105.0, high=108.0, low=103.0, close=106.0),
                    StockDaily(code="600519", date=date(2024, 1, 4), open=106.0, high=109.0, low=104.0, close=107.0),
                ]
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

    def _count_results(self) -> int:
        with self.db.get_session() as session:
            return session.query(BacktestResult).count()

    def test_force_semantics(self) -> None:
        service = BacktestService(self.db)

        stats1 = service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats1["saved"], 1)
        self.assertEqual(self._count_results(), 1)

        # Non-force should be idempotent
        stats2 = service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats2["saved"], 0)
        self.assertEqual(self._count_results(), 1)

        # Force should replace existing result without unique constraint errors
        stats3 = service.run_backtest(code="600519", force=True, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats3["saved"], 1)
        self.assertEqual(self._count_results(), 1)

    def test_incomplete_results_are_retried(self) -> None:
        """Rows left as insufficient_data are retried instead of being skipped forever."""
        with self.db.get_session() as session:
            history = session.query(AnalysisHistory).one()
            session.add(
                BacktestResult(
                    analysis_history_id=history.id,
                    code=history.code,
                    eval_window_days=3,
                    engine_version=f"v1@no-{EXECUTION_LOGIC_REVISION}",
                    eval_status="insufficient_data",
                    evaluated_at=datetime.now(),
                )
            )
            session.commit()

        service = BacktestService(self.db)
        stats = service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        self.assertEqual(stats["saved"], 1)
        self.assertEqual(stats["completed"], 1)
        with self.db.get_session() as session:
            rows = session.query(BacktestResult).all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].eval_status, "completed")

    def test_entry_mode_changes_backtest_identity(self) -> None:
        service = BacktestService(self.db)
        stats1 = service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats1["saved"], 1)

        os.environ["BACKTEST_ENTRY_MODE"] = "analysis_close"
        Config._instance = None
        stats2 = service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        self.assertEqual(stats2["saved"], 1)
        with self.db.get_session() as session:
            versions = {
                row.engine_version
                for row in session.query(BacktestResult).order_by(BacktestResult.engine_version).all()
            }
        self.assertEqual(
            versions,
            {
                f"v1@ac-{EXECUTION_LOGIC_REVISION}",
                f"v1@no-{EXECUTION_LOGIC_REVISION}",
            },
        )

    def _run_and_get_result(self) -> BacktestResult:
        """Helper: run backtest and return the single BacktestResult row."""
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)
        with self.db.get_session() as session:
            return session.query(BacktestResult).one()

    def test_result_fields_correct(self) -> None:
        """Verify BacktestResult row contains correct evaluation values."""
        result = self._run_and_get_result()

        self.assertEqual(result.eval_status, "completed")
        self.assertEqual(result.engine_version, f"v1@no-{EXECUTION_LOGIC_REVISION}")
        self.assertEqual(result.code, "600519")
        self.assertEqual(result.analysis_date, date(2024, 1, 1))
        self.assertEqual(result.operation_advice, "买入")
        self.assertEqual(result.position_recommendation, "long")
        self.assertEqual(result.direction_expected, "up")

        # Prices
        self.assertAlmostEqual(result.start_price, 100.0)
        self.assertAlmostEqual(result.end_close, 107.0)
        self.assertAlmostEqual(result.stock_return_pct, 7.0)

        # Direction & outcome
        self.assertEqual(result.outcome, "win")
        self.assertTrue(result.direction_correct)

        # Target hits -- day2 high=111 >= take_profit=110
        self.assertTrue(result.hit_take_profit)
        self.assertFalse(result.hit_stop_loss)
        self.assertEqual(result.first_hit, "take_profit")
        self.assertEqual(result.first_hit_trading_days, 1)
        self.assertEqual(result.first_hit_date, date(2024, 1, 2))

        # Simulated execution
        self.assertAlmostEqual(result.simulated_entry_price, 100.0)
        self.assertAlmostEqual(result.simulated_exit_price, 110.0)
        self.assertEqual(result.simulated_exit_reason, "take_profit")
        self.assertAlmostEqual(result.simulated_return_pct, 10.0)

    def test_summaries_created_after_run(self) -> None:
        """Verify both overall and per-stock BacktestSummary rows are created."""
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        with self.db.get_session() as session:
            # Overall summary uses sentinel code
            overall = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "overall",
                BacktestSummary.code == OVERALL_SENTINEL_CODE,
            ).first()
            self.assertIsNotNone(overall)
            self.assertEqual(overall.total_evaluations, 1)
            self.assertEqual(overall.completed_count, 1)
            self.assertEqual(overall.win_count, 1)
            self.assertEqual(overall.loss_count, 0)
            self.assertAlmostEqual(overall.win_rate_pct, 100.0)

            # Stock-level summary
            stock = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "stock",
                BacktestSummary.code == "600519",
            ).first()
            self.assertIsNotNone(stock)
            self.assertEqual(stock.total_evaluations, 1)
            self.assertEqual(stock.completed_count, 1)
            self.assertEqual(stock.win_count, 1)

    def test_get_summary_overall_returns_sentinel_as_none(self) -> None:
        """Verify get_summary translates __overall__ sentinel back to None."""
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        summary = service.get_summary(scope="overall", code=None)
        self.assertIsNotNone(summary)
        self.assertIsNone(summary["code"])
        self.assertEqual(summary["scope"], "overall")
        self.assertEqual(summary["win_count"], 1)

    def test_agent_learning_summary_helpers_return_normalized_ratios(self) -> None:
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        global_summary = service.get_global_summary(eval_window_days=3)
        stock_summary = service.get_stock_summary("600519", eval_window_days=3)
        strategy_summary = service.get_strategy_summary("bull_trend", eval_window_days=3)

        self.assertIsNotNone(global_summary)
        self.assertEqual(global_summary["total_evaluations"], 1)
        self.assertAlmostEqual(global_summary["win_rate"], 1.0)
        self.assertAlmostEqual(global_summary["direction_accuracy"], 1.0)
        self.assertAlmostEqual(global_summary["avg_return"], 0.10)

        self.assertIsNotNone(stock_summary)
        self.assertEqual(stock_summary["code"], "600519")
        self.assertAlmostEqual(stock_summary["win_rate"], 1.0)

        self.assertIsNotNone(strategy_summary)
        self.assertEqual(strategy_summary["strategy_id"], "bull_trend")
        self.assertEqual(strategy_summary["source_scope"], "strategy")
        self.assertFalse(strategy_summary["is_fallback"])
        self.assertAlmostEqual(strategy_summary["win_rate"], 1.0)

        fallback_summary = service.get_strategy_summary("new_strategy", eval_window_days=3)
        self.assertIsNotNone(fallback_summary)
        self.assertEqual(fallback_summary["strategy_id"], "new_strategy")
        self.assertEqual(fallback_summary["source_scope"], "overall")
        self.assertTrue(fallback_summary["is_fallback"])

    def test_get_recent_evaluations(self) -> None:
        """Verify get_recent_evaluations returns correct paginated results."""
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        data = service.get_recent_evaluations(code="600519", limit=10, page=1)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["limit"], 10)
        self.assertEqual(len(data["items"]), 1)

        item = data["items"][0]
        self.assertEqual(item["code"], "600519")
        self.assertEqual(item["outcome"], "win")
        self.assertEqual(item["direction_expected"], "up")
        self.assertTrue(item["direction_correct"])

    def test_calibration_lift_is_measured_against_raw_advice(self) -> None:
        """同一条分析同时评估校准前/后：结论变对是校准救回来的，还是本来就对。"""
        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id="q-calibrated",
                    code="000004",
                    name="国华网安",
                    report_type="simple",
                    sentiment_score=50,
                    # 校准后：卖出→持有；股票上涨后持有正确、原始卖出错误
                    operation_advice="持有",
                    trend_prediction="震荡",
                    analysis_summary="calibrated",
                    raw_result=json.dumps(
                        {
                            "model_used": "gemini/gemini-2.5-flash",
                            "success": True,
                            "decision_type": "hold",
                            "operation_advice": "持有",
                            "calibration_info": {
                                "applied": True,
                                "current_signal": "sell",
                                "original_advice": "卖出",
                            },
                        },
                        ensure_ascii=False,
                    ),
                    created_at=datetime(2024, 1, 1, 0, 0, 0),
                    context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                )
            )
            session.add(
                StockDaily(code="000004", date=date(2024, 1, 1), open=10.0, high=10.2, low=9.8, close=10.0)
            )
            session.add_all([
                StockDaily(code="000004", date=date(2024, 1, 2), open=10.2, high=10.5, low=10.1, close=10.4),
                StockDaily(code="000004", date=date(2024, 1, 3), open=10.4, high=10.7, low=10.3, close=10.6),
                StockDaily(code="000004", date=date(2024, 1, 4), open=10.6, high=10.9, low=10.5, close=10.7),
            ])
            session.commit()

        service = BacktestService(db_manager=self.db)
        service.run_backtest()

        with self.db.get_session() as session:
            row = (
                session.query(BacktestResult)
                .filter(BacktestResult.code == "000004")
                .one()
            )
            self.assertEqual(row.raw_operation_advice, "卖出")
            self.assertEqual(row.raw_direction_expected, "down")
            self.assertFalse(row.raw_direction_correct)
            self.assertTrue(row.direction_correct)

        summary = service.get_summary(scope="overall", code=None)
        self.assertIsNotNone(summary)
        # 600519（买入，上涨）原始与最终一致，不计入归因；
        # 000004 校准把"卖出"改成"持有"，最终对、原始错 → lift +100
        self.assertEqual(summary["raw_direction_accuracy_pct"], 50.0)
        self.assertEqual(summary["direction_accuracy_pct"], 100.0)
        self.assertEqual(summary["calibration_lift_pct"], 100.0)
        self.assertEqual(summary["calibration_sample_count"], 1)

    def test_paused_calibration_keeps_shadow_arm_for_recovery(self) -> None:
        """暂停校准时仍记录影子建议，回测能继续判断它是否重新有效。"""
        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id="q-shadow",
                    code="000005",
                    name="世纪星源",
                    report_type="simple",
                    sentiment_score=50,
                    # 暂停：实际结论保持原始"观望"
                    operation_advice="观望",
                    trend_prediction="震荡",
                    analysis_summary="paused",
                    raw_result=json.dumps(
                        {
                            "model_used": "gemini/gemini-2.5-flash",
                            "success": True,
                            "decision_type": "hold",
                            "operation_advice": "观望",
                            "calibration_info": {
                                "applied": False,
                                "paused": True,
                                "current_signal": "hold",
                                "suggested_signal": "buy",
                                "original_advice": "观望",
                            },
                        },
                        ensure_ascii=False,
                    ),
                    created_at=datetime(2024, 1, 1, 0, 0, 0),
                    context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                )
            )
            session.add(
                StockDaily(code="000005", date=date(2024, 1, 1), open=10.0, high=10.2, low=9.8, close=10.0)
            )
            session.add_all([
                StockDaily(code="000005", date=date(2024, 1, 2), open=10.2, high=10.5, low=10.1, close=10.4),
                StockDaily(code="000005", date=date(2024, 1, 3), open=10.4, high=10.7, low=10.3, close=10.6),
                StockDaily(code="000005", date=date(2024, 1, 4), open=10.6, high=10.9, low=10.5, close=10.7),
            ])
            session.commit()

        service = BacktestService(db_manager=self.db)
        service.run_backtest()

        with self.db.get_session() as session:
            row = (
                session.query(BacktestResult)
                .filter(BacktestResult.code == "000005")
                .one()
            )
            self.assertEqual(row.calibration_arm_operation_advice, "买入")
            self.assertTrue(row.calibration_arm_direction_correct)
            self.assertFalse(row.raw_direction_correct)

        summary = service.get_summary(scope="overall", code=None)
        self.assertEqual(summary["calibration_lift_pct"], 100.0)
        self.assertEqual(summary["calibration_sample_count"], 1)

    def test_transaction_costs_reduce_net_return(self) -> None:
        """佣金/滑点必须从模拟收益里扣掉，而不是只展示毛收益。"""
        from src.config import get_config

        cfg = get_config()
        with patch("src.services.backtest_service.get_config", return_value=cfg), \
             patch.object(cfg, "backtest_slippage_bps", 5.0):
            BacktestService(db_manager=self.db).run_backtest()

        with self.db.get_session() as session:
            row = (
                session.query(BacktestResult)
                .filter(BacktestResult.code == "600519")
                .one()
            )
            self.assertAlmostEqual(row.simulated_return_pct_gross, 10.0)
            self.assertAlmostEqual(row.simulated_return_pct, 9.9)

    def test_path_metrics_record_mfe_and_mae(self) -> None:
        BacktestService(db_manager=self.db).run_backtest()

        with self.db.get_session() as session:
            row = (
                session.query(BacktestResult)
                .filter(BacktestResult.code == "600519")
                .one()
            )
            self.assertIsNotNone(row.mfe_pct)
            self.assertIsNotNone(row.mae_pct)
            self.assertGreaterEqual(row.days_to_mfe or 0, 1)
            self.assertGreater(row.mfe_pct, 0)

    def test_benchmark_excess_return_is_recorded(self) -> None:
        with patch(
            "src.services.benchmark_service.BenchmarkService.return_pct",
            return_value=3.0,
        ):
            service = BacktestService(db_manager=self.db)
            service.run_backtest()

        with self.db.get_session() as session:
            row = (
                session.query(BacktestResult)
                .filter(BacktestResult.code == "600519")
                .one()
            )
            self.assertEqual(row.benchmark_return_pct, 3.0)
            self.assertAlmostEqual(row.excess_return_pct, round(row.stock_return_pct - 3.0, 2))

        summary = service.get_summary(scope="overall", code=None)
        self.assertEqual(summary["avg_benchmark_return_pct"], 3.0)
        self.assertIsNotNone(summary["avg_excess_return_pct"])
        self.assertIsNotNone(summary["excess_win_rate_pct"])

    def test_multi_window_evaluates_each_horizon(self) -> None:
        """同一批信号按 1/3/5 日分别评估，短窗口完成、长窗口数据不足。"""
        from src.config import get_config

        cfg = get_config()
        with patch.object(cfg, "backtest_extra_windows", "1,5"), \
             patch.object(BacktestService, "_try_fill_daily_data", return_value=None):
            stats = BacktestService(db_manager=self.db).run_backtest()

        self.assertEqual(stats["windows"], [1, 3, 5])
        self.assertEqual(stats["processed"], 3)
        self.assertEqual(stats["completed"], 2)
        self.assertEqual(stats["insufficient"], 1)

    def test_split_sized_jump_is_not_backtested_as_return(self) -> None:
        """2:1 拆股会在日线里留下 -50%，不能当成真实亏损/收益回测。"""
        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id="q-split",
                    code="000006",
                    name="拆股样本",
                    report_type="simple",
                    sentiment_score=70,
                    operation_advice="买入",
                    trend_prediction="看多",
                    analysis_summary="split",
                    raw_result='{"model_used": "gemini/gemini-2.5-flash", "success": true}',
                    created_at=datetime(2024, 1, 1, 0, 0, 0),
                    context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                )
            )
            session.add(
                StockDaily(code="000006", date=date(2024, 1, 1), open=100.0, high=101.0, low=99.0, close=100.0)
            )
            session.add_all([
                StockDaily(code="000006", date=date(2024, 1, 2), open=50.0, high=51.0, low=49.0, close=50.0),
                StockDaily(code="000006", date=date(2024, 1, 3), open=50.0, high=52.0, low=50.0, close=51.0),
                StockDaily(code="000006", date=date(2024, 1, 4), open=51.0, high=53.0, low=51.0, close=52.0),
            ])
            session.commit()

        stats = BacktestService(db_manager=self.db).run_backtest()

        self.assertEqual(stats["data_quality"], 1)
        with self.db.get_session() as session:
            row = (
                session.query(BacktestResult)
                .filter(BacktestResult.code == "000006")
                .one()
            )
            self.assertEqual(row.eval_status, "data_quality")
            self.assertIsNone(row.stock_return_pct)

    def test_agent_snapshot_analysis_date_is_parsed(self) -> None:
        """Agent 链路没有 enhanced_context，也要能从 analysis_date 定位信号日。"""
        parsed = BacktestRepository.parse_analysis_date_from_snapshot(
            '{"analysis_date": "2026-03-23", "stock_code": "AAPL"}'
        )

        self.assertEqual(parsed, date(2026, 3, 23))

    def test_same_signal_day_is_evaluated_once(self) -> None:
        """同一只股票同一天重复分析，不能按多条独立样本重复加权。"""
        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id="q-dup",
                    code="600519",
                    name="贵州茅台",
                    report_type="simple",
                    sentiment_score=75,
                    operation_advice="买入",
                    trend_prediction="看多",
                    analysis_summary="同一天第二次分析",
                    raw_result='{"model_used": "gemini/gemini-2.5-flash", "success": true}',
                    stop_loss=95.0,
                    take_profit=110.0,
                    created_at=datetime(2024, 1, 1, 12, 0, 0),
                    context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                )
            )
            session.commit()

        stats = BacktestService(db_manager=self.db).run_backtest()

        self.assertEqual(stats["processed"], 1)

    def test_placeholder_analyses_are_not_backtested(self) -> None:
        """LLM 未参与 / 文本降级的记录不参与回测，否则准确率被占位结论稀释。"""
        with self.db.get_session() as session:
            session.add_all(
                [
                    AnalysisHistory(
                        query_id="q-stub",
                        code="000002",
                        name="万科A",
                        report_type="simple",
                        sentiment_score=50,
                        operation_advice="持有",
                        trend_prediction="震荡",
                        analysis_summary="LLM 未配置",
                        raw_result='{"success": false, "operation_advice": "持有"}',
                        created_at=datetime(2024, 1, 1, 0, 0, 0),
                        context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                    ),
                    AnalysisHistory(
                        query_id="q-degraded",
                        code="000003",
                        name="国农科技",
                        report_type="simple",
                        sentiment_score=50,
                        operation_advice="持有",
                        trend_prediction="震荡",
                        analysis_summary="JSON 解析失败",
                        raw_result=(
                            '{"model_used": "gemini/gemini-2.5-flash",'
                            ' "success": true, "parse_mode": "text_fallback"}'
                        ),
                        created_at=datetime(2024, 1, 1, 0, 0, 0),
                        context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                    ),
                ]
            )
            session.commit()

        stats = BacktestService(db_manager=self.db).run_backtest()

        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["saved"], 1)

    def test_multi_stock_summaries(self) -> None:
        """Verify separate summaries for multiple stocks + correct overall aggregate."""
        old_created_at = datetime(2024, 1, 1, 0, 0, 0)

        with self.db.get_session() as session:
            # Second stock with sell advice -- price drops (win for cash/down)
            session.add(
                AnalysisHistory(
                    query_id="q2",
                    code="000001",
                    name="平安银行",
                    report_type="simple",
                    sentiment_score=30,
                    operation_advice="卖出",
                    trend_prediction="看空",
                    analysis_summary="test2",
                    raw_result='{"model_used": "gemini/gemini-2.5-flash", "success": true, "parse_mode": "structured", "decision_type": "sell"}',
                    stop_loss=None,
                    take_profit=None,
                    created_at=old_created_at,
                    context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                )
            )
            session.add(
                StockDaily(code="000001", date=date(2024, 1, 1), open=10.0, high=10.2, low=9.8, close=10.0)
            )
            session.add_all([
                StockDaily(code="000001", date=date(2024, 1, 2), open=10.0, high=10.0, low=9.5, close=9.6),
                StockDaily(code="000001", date=date(2024, 1, 3), open=9.6, high=9.7, low=9.3, close=9.4),
                StockDaily(code="000001", date=date(2024, 1, 4), open=9.4, high=9.5, low=9.0, close=9.1),
            ])
            session.commit()

        service = BacktestService(self.db)
        stats = service.run_backtest(code=None, force=False, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats["saved"], 2)
        self.assertEqual(stats["completed"], 2)

        with self.db.get_session() as session:
            # Each stock has its own summary
            s1 = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "stock", BacktestSummary.code == "600519"
            ).first()
            s2 = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "stock", BacktestSummary.code == "000001"
            ).first()
            self.assertIsNotNone(s1)
            self.assertIsNotNone(s2)
            self.assertEqual(s1.win_count, 1)
            # Cash advice has no simulated trade, so it is not counted as a trade win.
            self.assertEqual(s2.win_count, 0)
            self.assertEqual(s2.cash_count, 1)

            # Overall aggregates both
            overall = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "overall",
                BacktestSummary.code == OVERALL_SENTINEL_CODE,
            ).first()
            self.assertIsNotNone(overall)
            self.assertEqual(overall.total_evaluations, 2)
            self.assertEqual(overall.completed_count, 2)
            self.assertEqual(overall.win_count, 1)


if __name__ == "__main__":
    unittest.main()

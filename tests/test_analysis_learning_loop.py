# -*- coding: utf-8 -*-
"""Tests for the backtest-driven evolution loop."""

import json
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for _mod in ("litellm", "json_repair", "markdown2", "newspaper"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analyzer import AnalysisResult
from src.core.backtest_engine import BacktestEngine
from src.core.pipeline import StockAnalysisPipeline
from src.enums import ReportType
from src.services.analysis_calibration_model import SmallCalibrationModel
from src.services.analysis_calibration_service import AnalysisCalibrationService
from src.services.backtest_service import BacktestService
from src.storage import AnalysisHistory, BacktestResult, DatabaseManager


def _learning_engine_version() -> str:
    """与 BacktestService 写入时一致的引擎版本号（学习样本只吃当前版本）。"""
    return BacktestService._effective_engine_version(
        "v1",
        BacktestEngine._normalize_entry_mode("next_open"),
    )


class AnalysisCalibrationServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = os.path.join(self._temp_dir.name, "learning_loop.db")
        self._old_env = {
            key: os.environ.get(key)
            for key in ("ENV_FILE", "DATABASE_URL", "DATABASE_PATH")
        }
        os.environ["ENV_FILE"] = os.path.join(self._temp_dir.name, "empty.env")
        os.environ.pop("DATABASE_URL", None)
        os.environ["DATABASE_PATH"] = self._db_path

        from src.config import Config

        Config._instance = None
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()
        self.config = SimpleNamespace(
            backtest_enabled=True,
            analysis_learning_min_samples=3,
            analysis_learning_history_limit=100,
            analysis_learning_auto_backtest_limit=50,
            analysis_learning_refresh_interval_minutes=60,
            analysis_learning_model_enabled=True,
            analysis_learning_model_path=os.path.join(self._temp_dir.name, "analysis_calibration_model.json"),
            analysis_learning_model_backend="tree",
            analysis_learning_model_market_split=True,
            analysis_learning_model_scope_min_samples=6,
            analysis_learning_model_retrain_interval_minutes=60,
            analysis_learning_model_train_min_samples=6,
            analysis_learning_model_confidence_threshold=0.55,
            analysis_learning_label_band_pct=2.0,
            backtest_neutral_band_pct=2.0,
        )

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        from src.config import Config

        Config._instance = None
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._temp_dir.cleanup()

    def _insert_learning_record(
        self,
        *,
        query_id: str,
        code: str,
        signal: str,
        direction_correct: bool,
        simulated_return_pct: float,
        stock_return_pct: float,
        model_used: str = "gemini/gemini-2.5-flash",
        market: str = "us",
        sentiment_score: int | None = None,
        change_pct: float | None = None,
        trend_strength: float | None = None,
        signal_score: int | None = None,
        bias_ma5: float | None = None,
        china_exposure_level: str = "",
        analysis_day: date | None = None,
        parse_mode: str = "structured",
    ) -> None:
        advice_map = {
            "buy": "买入",
            "hold": "观望",
            "sell": "减仓/卖出",
        }
        trend_map = {
            "buy": "看多",
            "hold": "震荡",
            "sell": "看空",
        }
        created_at = datetime(2024, 1, 1, 0, 0, 0)
        score_value = sentiment_score if sentiment_score is not None else (
            72 if signal == "buy" else 28 if signal == "sell" else 50
        )
        context_snapshot = self._make_context_snapshot(
            market=market,
            signal=signal,
            change_pct=change_pct,
            trend_strength=trend_strength,
            signal_score=signal_score,
            bias_ma5=bias_ma5,
            china_exposure_level=china_exposure_level,
        )

        with self.db.get_session() as session:
            history = AnalysisHistory(
                query_id=query_id,
                code=code,
                name=code,
                report_type="simple",
                sentiment_score=score_value,
                operation_advice=advice_map[signal],
                trend_prediction=trend_map[signal],
                analysis_summary="test",
                raw_result=json.dumps({
                    "decision_type": signal,
                    "operation_advice": advice_map[signal],
                    "trend_prediction": trend_map[signal],
                    "model_used": model_used,
                    "sentiment_score": score_value,
                    "parse_mode": parse_mode,
                }, ensure_ascii=False),
                context_snapshot=json.dumps(context_snapshot, ensure_ascii=False),
                created_at=created_at,
            )
            session.add(history)
            session.flush()
            session.add(
                BacktestResult(
                    analysis_history_id=history.id,
                    code=code,
                    analysis_date=analysis_day or date(2024, 1, 1),
                    eval_window_days=10,
                    engine_version=_learning_engine_version(),
                    eval_status="completed",
                    evaluated_at=created_at,
                    operation_advice=advice_map[signal],
                    direction_correct=direction_correct,
                    simulated_return_pct=simulated_return_pct,
                    stock_return_pct=stock_return_pct,
                )
            )
            session.commit()

    @staticmethod
    def _make_context_snapshot(
        *,
        market: str,
        signal: str,
        change_pct: float | None = None,
        trend_strength: float | None = None,
        signal_score: int | None = None,
        bias_ma5: float | None = None,
        china_exposure_level: str = "",
    ) -> dict:
        bullish = signal == "buy"
        bearish = signal == "sell"
        return {
            "enhanced_context": {
                "market": market,
                "market_context": {
                    "market": market,
                    "china_exposure": {"level": china_exposure_level} if china_exposure_level else {},
                },
                "realtime": {
                    "change_pct": change_pct if change_pct is not None else (3.6 if bullish else -3.8 if bearish else 0.4),
                    "volume_ratio": 1.4 if bullish else 0.7 if bearish else 1.0,
                    "turnover_rate": 6.0 if bullish else 2.0 if bearish else 3.0,
                },
                "trend_analysis": {
                    "trend_strength": trend_strength if trend_strength is not None else (86 if bullish else 24 if bearish else 50),
                    "signal_score": signal_score if signal_score is not None else (88 if bullish else 22 if bearish else 50),
                    "bias_ma5": bias_ma5 if bias_ma5 is not None else (0.8 if bullish else -0.9 if bearish else 0.0),
                    "bias_ma10": 0.5 if bullish else -0.6 if bearish else 0.0,
                },
                "chip": {
                    "profit_ratio": 78 if bullish else 28 if bearish else 50,
                    "concentration_90": 12 if bullish else 28 if bearish else 18,
                    "concentration_70": 8 if bullish else 20 if bearish else 12,
                },
                "is_index_etf": False,
            }
        }

    def test_calibrate_result_uses_historical_counter_evidence(self) -> None:
        # US market: sell signals have been wrong, buy signals have been right.
        for idx, code in enumerate(("AAPL", "MSFT", "NVDA"), start=1):
            self._insert_learning_record(
                query_id=f"sell-{idx}",
                code=code,
                signal="sell",
                direction_correct=False,
                simulated_return_pct=-6.0,
                stock_return_pct=6.0,
                china_exposure_level="high",
            )
        for idx, code in enumerate(("META", "GOOGL", "AMZN", "TSLA"), start=1):
            self._insert_learning_record(
                query_id=f"buy-{idx}",
                code=code,
                signal="buy",
                direction_correct=True,
                simulated_return_pct=8.0,
                stock_return_pct=8.0,
                china_exposure_level="high",
            )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        result = AnalysisResult(
            code="AAPL",
            name="Apple Inc.",
            sentiment_score=38,
            trend_prediction="看空",
            operation_advice="减仓/卖出",
            decision_type="sell",
            analysis_summary="原始分析认为短线应减仓。",
            success=True,
            model_used="gemini/gemini-2.5-flash",
        )

        calibrated = service.calibrate_result(result)

        self.assertEqual(calibrated.decision_type, "buy")
        self.assertIn("回测校准", calibrated.analysis_summary)
        self.assertIsInstance(calibrated.calibration_info, dict)
        self.assertTrue(calibrated.calibration_info.get("applied"))
        self.assertEqual(calibrated.calibration_info.get("source_scope"), "市场+模型")
        self.assertGreaterEqual(calibrated.sentiment_score, 60)

    def test_learning_rows_exclude_old_engine_versions(self) -> None:
        """旧引擎版本的回测样本不能进入学习集，否则新旧口径互相矛盾。"""
        current_code = "AAPL"
        stale_code = "TSLA"
        self._insert_learning_record(
            query_id="q-current",
            code=current_code,
            signal="buy",
            direction_correct=True,
            simulated_return_pct=3.0,
            stock_return_pct=3.0,
        )
        self._insert_learning_record(
            query_id="q-stale",
            code=stale_code,
            signal="buy",
            direction_correct=False,
            simulated_return_pct=-4.0,
            stock_return_pct=-4.0,
        )

        # 把其中一条改写成旧口径版本号
        with self.db.get_session() as session:
            stale = (
                session.query(BacktestResult)
                .filter(BacktestResult.code == stale_code)
                .one()
            )
            stale.engine_version = "v1"
            session.commit()

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        rows = service._load_learning_rows()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["code"], current_code)

    def test_learning_rows_exclude_placeholder_analyses(self) -> None:
        """LLM 未参与时落库的「持有/震荡/50」占位结果不能进入学习集。"""
        self._insert_learning_record(
            query_id="q-real",
            code="AAPL",
            signal="buy",
            direction_correct=True,
            simulated_return_pct=3.0,
            stock_return_pct=3.0,
            model_used="gemini/gemini-2.5-flash",
        )
        self._insert_learning_record(
            query_id="q-stub",
            code="TSLA",
            signal="hold",
            direction_correct=True,
            simulated_return_pct=0.0,
            stock_return_pct=0.0,
            model_used="",
        )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        rows = service._load_learning_rows()

        self.assertEqual([row["code"] for row in rows], ["AAPL"])

    def _seed_training_timeline(self, *, days: int = 60, start: date | None = None) -> None:
        """按日铺一条可训练时间线：特征与标签强相关，且日期跨度足够留出隔离期。"""
        base = start or date(2024, 1, 1)
        for index in range(days):
            bullish = index % 2 == 0
            self._insert_learning_record(
                query_id=f"seed-{index}",
                code=f"SYM{index:02d}",
                signal="buy" if bullish else "sell",
                direction_correct=True,
                simulated_return_pct=6.0 if bullish else 0.0,
                stock_return_pct=6.0 if bullish else -6.0,
                change_pct=4.0 if bullish else -4.0,
                trend_strength=90 if bullish else 20,
                signal_score=92 if bullish else 18,
                bias_ma5=1.1 if bullish else -1.2,
                china_exposure_level="high" if bullish else "low",
                analysis_day=base + timedelta(days=index),
            )

    def test_learning_labels_use_excess_return(self) -> None:
        """同一天所有标的都涨 5% 时，不能把全部样本标成 buy。"""
        day = date(2024, 1, 15)
        for idx, code in enumerate(("AAPL", "MSFT", "NVDA")):
            self._insert_learning_record(
                query_id=f"mkt-{idx}",
                code=code,
                signal="hold",
                direction_correct=True,
                simulated_return_pct=0.0,
                stock_return_pct=5.0,
                analysis_day=day,
            )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        samples = service._build_training_samples(service._load_learning_rows())

        self.assertTrue(samples)
        # 超额收益 = 5% - 同日中位数 5% = 0 → 全部中性
        self.assertEqual({sample["label"] for sample in samples}, {"hold"})

    def test_learning_rows_dedupe_same_signal_day(self) -> None:
        """同一股票同一天分析两次，收益相同，只能算一条样本。"""
        for index in range(2):
            self._insert_learning_record(
                query_id=f"q-dup-{index}",
                code="AAPL",
                signal="buy",
                direction_correct=True,
                simulated_return_pct=3.0,
                stock_return_pct=3.0,
            )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        rows = service._load_learning_rows()

        self.assertEqual(len(rows), 1)

    def test_learning_rows_exclude_degraded_text_parses(self) -> None:
        """JSON 解析失败时按关键词猜的方向，不能当成模型判断去训练。"""
        self._insert_learning_record(
            query_id="q-structured",
            code="AAPL",
            signal="buy",
            direction_correct=True,
            simulated_return_pct=3.0,
            stock_return_pct=3.0,
        )
        self._insert_learning_record(
            query_id="q-degraded",
            code="TSLA",
            signal="sell",
            direction_correct=False,
            simulated_return_pct=-3.0,
            stock_return_pct=-3.0,
            parse_mode="text_fallback",
        )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        rows = service._load_learning_rows()

        self.assertEqual([row["code"] for row in rows], ["AAPL"])

    def test_calibration_does_not_promote_hold_from_market_drift(self) -> None:
        """牛市里 buy 的高准确率只是市场漂移，不能据此把观望改成买入。"""
        self.config.analysis_learning_model_enabled = False
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)

        rows = []
        # 20 个样本里 18 个上涨：buy 全对，但基准率也是 90%
        for index in range(14):
            rows.append(
                {
                    "signal": "buy",
                    "direction_correct": True,
                    "simulated_return_pct": 5.0,
                    "stock_return_pct": 5.0,
                }
            )
        for index in range(4):
            rows.append(
                {
                    "signal": "hold",
                    "direction_correct": False,
                    "simulated_return_pct": 0.0,
                    "stock_return_pct": 5.0,
                }
            )
        for index in range(2):
            rows.append(
                {
                    "signal": "hold",
                    "direction_correct": True,
                    "simulated_return_pct": 0.0,
                    "stock_return_pct": -5.0,
                }
            )

        decision = service._decide_adjustment(
            rows=rows,
            current_signal="hold",
            threshold=14,
            score=50,
        )

        self.assertIsNone(decision)

    def test_calibration_promotes_hold_when_edge_beats_base_rate(self) -> None:
        self.config.analysis_learning_model_enabled = False
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)

        rows = []
        for index in range(16):
            rows.append(
                {
                    "signal": "buy",
                    "direction_correct": True,
                    "simulated_return_pct": 5.0,
                    "stock_return_pct": 5.0,
                }
            )
        for index in range(14):
            rows.append(
                {
                    "signal": "hold",
                    "direction_correct": True,
                    "simulated_return_pct": 0.0,
                    "stock_return_pct": -3.0,
                }
            )

        decision = service._decide_adjustment(
            rows=rows,
            current_signal="hold",
            threshold=14,
            score=50,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision["suggested_signal"], "buy")

    def test_calibration_ignores_edge_that_only_exists_in_old_samples(self) -> None:
        """近期已经不再成立的历史优势，不能拿来强化当前信号。"""
        self.config.analysis_learning_model_enabled = False
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)

        rows = []
        day = date(2024, 1, 1)
        # 旧样本：buy 全部正确，强势优势明显
        for index in range(14):
            rows.append(
                {
                    "signal": "buy",
                    "direction_correct": True,
                    "simulated_return_pct": 5.0,
                    "stock_return_pct": 5.0,
                    "analysis_date": (day + timedelta(days=index)).isoformat(),
                }
            )
        # 基准样本：下跌
        for index in range(14):
            rows.append(
                {
                    "signal": "hold",
                    "direction_correct": True,
                    "simulated_return_pct": 0.0,
                    "stock_return_pct": -5.0,
                    "analysis_date": (day + timedelta(days=index)).isoformat(),
                }
            )
        # 近期样本：buy 一半对一半错，优势消失
        for index in range(12):
            won = index % 2 == 0
            rows.append(
                {
                    "signal": "buy",
                    "direction_correct": won,
                    "simulated_return_pct": 5.0 if won else -5.0,
                    "stock_return_pct": 5.0 if won else -5.0,
                    "analysis_date": (day + timedelta(days=30 + index)).isoformat(),
                }
            )

        decision = service._decide_adjustment(
            rows=rows,
            current_signal="buy",
            threshold=20,
            score=65,
        )

        self.assertFalse(decision["applied"])

    def test_calibration_stores_pre_calibration_signal_for_training(self) -> None:
        """校准信息要保留校准前的分数/趋势，学习环节才能用原始判断训练。"""
        for idx, code in enumerate(("AAPL", "MSFT", "NVDA"), start=1):
            self._insert_learning_record(
                query_id=f"bad-sell-{idx}",
                code=code,
                signal="sell",
                direction_correct=False,
                simulated_return_pct=-6.0,
                stock_return_pct=6.0,
            )
        for idx, code in enumerate(("META", "GOOGL", "AMZN", "TSLA"), start=1):
            self._insert_learning_record(
                query_id=f"good-buy-{idx}",
                code=code,
                signal="buy",
                direction_correct=True,
                simulated_return_pct=8.0,
                stock_return_pct=8.0,
            )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        result = AnalysisResult(
            code="AAPL",
            name="Apple Inc.",
            sentiment_score=38,
            trend_prediction="看空",
            operation_advice="减仓/卖出",
            decision_type="sell",
            analysis_summary="原始判断偏空。",
            success=True,
            model_used="gemini/gemini-2.5-flash",
        )

        calibrated = service.calibrate_result(result)
        info = calibrated.calibration_info or {}

        self.assertEqual(info.get("current_signal"), "sell")
        self.assertEqual(info.get("original_score"), 38)
        self.assertEqual(info.get("original_trend"), "看空")
        # 结论可以被校准，但原始判断必须保留
        self.assertEqual(calibrated.decision_type, "buy")

    def test_training_features_use_raw_signal_not_calibrated_signal(self) -> None:
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        rows = [
            {
                "code": "AAPL",
                "market": "us",
                "model_used": "gemini/gemini-2.5-flash",
                "signal": "buy",
                "raw_signal": "sell",
                "sentiment_score": 75,
                "raw_score": 30,
                "trend_prediction": "看多",
                "raw_trend": "看空",
                "direction_correct": True,
                "simulated_return_pct": 5.0,
                "stock_return_pct": 5.0,
                "analysis_date": "2024-01-02",
                "window_days": 10,
                "context_payload": {},
            }
        ]

        samples = service._build_training_samples(rows)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["features"]["current_signal_bias"], -1.0)
        self.assertAlmostEqual(samples[0]["features"]["score_norm"], 0.30, places=2)

    def test_calibration_pauses_itself_when_lift_is_negative(self) -> None:
        """历史回测显示校准不如原始判断时，自动停止纠偏。"""
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        result = AnalysisResult(
            code="AAPL",
            name="Apple Inc.",
            sentiment_score=62,
            trend_prediction="看多",
            operation_advice="买入",
            decision_type="buy",
            analysis_summary="原始分析偏多。",
            success=True,
            model_used="gemini/gemini-2.5-flash",
        )

        with patch(
            "src.services.analysis_calibration_service.BacktestService.get_summary",
            return_value={
                "direction_accuracy_pct": 48.0,
                "raw_direction_accuracy_pct": 55.0,
                "calibration_lift_pct": -7.0,
                "calibration_sample_count": 40,
                "completed_count": 40,
            },
        ):
            calibrated = service.calibrate_result(result)

        self.assertFalse(calibrated.calibration_info.get("applied"))
        self.assertIn("暂停", calibrated.calibration_info.get("reason", ""))
        self.assertEqual(calibrated.decision_type, "buy")
        self.assertEqual(calibrated.sentiment_score, 62)

    def test_maybe_refresh_backtests_respects_cooldown(self) -> None:
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config, time_fn=lambda: 1000.0)
        with patch("src.services.analysis_calibration_service.BacktestService.run_backtest", return_value={"processed": 0, "saved": 0, "completed": 0, "insufficient": 0, "errors": 0}) as mock_run:
            service.maybe_refresh_backtests()
            service.maybe_refresh_backtests()
        self.assertEqual(mock_run.call_count, 1)

    def test_learning_model_trains_tree_backend_and_calibrates_hold_to_buy(self) -> None:
        self.config.analysis_learning_min_samples = 10
        self.config.analysis_learning_model_train_min_samples = 6
        self.config.analysis_learning_model_scope_min_samples = 6

        # 按日铺 60 天：70% 训练、隔离 10 天、其余做验证集
        self._seed_training_timeline(days=60)

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        train_stats = service.maybe_refresh_model()
        self.assertTrue(train_stats.get("success"))
        self.assertTrue(os.path.exists(self.config.analysis_learning_model_path))
        self.assertEqual(train_stats.get("backend"), "hist_gradient_boosting")
        self.assertIn("us", train_stats.get("scopes", []))

        result = AnalysisResult(
            code="AAPL",
            name="Apple Inc.",
            sentiment_score=54,
            trend_prediction="震荡",
            operation_advice="观望",
            decision_type="hold",
            analysis_summary="原始分析偏中性。",
            success=True,
            model_used="gemini/gemini-2.5-flash",
        )
        calibrated = service.calibrate_result(
            result,
            context_snapshot=self._make_context_snapshot(
                market="us",
                signal="buy",
                change_pct=4.5,
                trend_strength=94,
                signal_score=95,
                bias_ma5=1.2,
                china_exposure_level="high",
            ),
        )

        self.assertEqual(calibrated.decision_type, "buy")
        self.assertEqual(calibrated.operation_advice, "买入")
        self.assertIsInstance(calibrated.calibration_info, dict)
        self.assertIn("model_prediction", calibrated.calibration_info)
        self.assertIn("学习模型", calibrated.calibration_info.get("source_scope"))
        self.assertEqual(calibrated.calibration_info["model_prediction"].get("engine"), "hist_gradient_boosting")
        self.assertEqual(calibrated.calibration_info["model_prediction"].get("scope"), "us")

    def test_no_model_without_a_clean_validation_set(self) -> None:
        """没有日期信息 / 无法切出隔离验证集时不出模型，而不是硬训练。"""
        samples = []
        for idx in range(18):
            samples.append(
                {
                    "features": {"score_norm": 0.85, "current_signal_bias": 1.0, "change_pct_norm": 0.4},
                    "label": "buy",
                    "market": "us",
                }
            )
        for idx in range(2):
            samples.append(
                {
                    "features": {"score_norm": 0.18, "current_signal_bias": -1.0, "change_pct_norm": -0.5},
                    "label": "sell",
                    "market": "us",
                }
            )

        model = SmallCalibrationModel.fit(
            samples,
            preferred_backend="tree",
            market_split=True,
            min_scope_samples=12,
        )

        # 样本没有 analysis_date，无法按日期切分并留出隔离期 → 不出模型
        self.assertIsNone(model)

    def test_single_class_training_side_yields_no_model(self) -> None:
        samples = []
        for idx, day in enumerate(
            (date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(40)
        ):
            samples.append(
                {
                    "features": {"score_norm": 0.85, "current_signal_bias": 1.0},
                    "label": "buy",
                    "market": "us",
                    "analysis_date": day,
                    "window_days": 10,
                }
            )

        model = SmallCalibrationModel.fit(
            samples,
            preferred_backend="tree",
            market_split=True,
            min_scope_samples=12,
        )

        self.assertIsNone(model)

    def test_calibration_hold_downgrade_keeps_bullish_hold_bias(self) -> None:
        self.config.analysis_learning_model_enabled = False

        for idx, code in enumerate(("AAPL", "MSFT", "NVDA"), start=1):
            self._insert_learning_record(
                query_id=f"weak-buy-{idx}",
                code=code,
                signal="buy",
                direction_correct=False,
                simulated_return_pct=-5.0,
                stock_return_pct=-5.0,
                market="us",
            )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        result = AnalysisResult(
            code="AAPL",
            name="Apple Inc.",
            sentiment_score=67,
            trend_prediction="看多",
            operation_advice="买入",
            decision_type="buy",
            analysis_summary="原始分析偏多。",
            success=True,
            model_used="gemini/gemini-2.5-flash",
        )

        calibrated = service.calibrate_result(result)

        self.assertEqual(calibrated.decision_type, "hold")
        self.assertEqual(calibrated.operation_advice, "持有")
        self.assertEqual(calibrated.trend_prediction, "震荡偏多")
        self.assertTrue(52 <= calibrated.sentiment_score <= 59)

    def test_strategy_profile_conflict_does_not_short_circuit_ml(self) -> None:
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)

        class FakePrediction:
            predicted_signal = "sell"
            confidence = 0.9
            sample_count = 80
            validation_accuracy = None
            validation_count = 0
            baseline_accuracy = None
            engine = "hist_gradient_boosting"
            scope = "us"
            validation_metrics = None

            def to_dict(self) -> dict:
                return {
                    "predicted_signal": self.predicted_signal,
                    "confidence": self.confidence,
                    "probabilities": {"sell": self.confidence, "buy": 0.1},
                    "sample_count": self.sample_count,
                    "engine": self.engine,
                    "scope": self.scope,
                }

        class FakeModel:
            validation_accuracy = 0.70
            validation_count = 20
            baseline_accuracy = 0.50

            def predict(self, features, *, scope_hint=None):
                return FakePrediction()

        service._small_model = FakeModel()

        with patch(
            "src.services.analysis_calibration_service.BacktestService.get_strategy_summary",
            return_value={
                "strategy_id": "bull_trend",
                "source_scope": "strategy",
                "is_fallback": False,
                "total_evaluations": 30,
                "win_rate": 0.8,
                "direction_accuracy": 0.8,
                "avg_return": 0.05,
            },
        ):
            profile = service._build_profile(
                stock_code="AAPL",
                current_signal="buy",
                model_used="gemini/gemini-2.5-flash",
                score=62,
                trend_prediction="看多",
                context_snapshot=self._make_context_snapshot(market="us", signal="buy"),
                strategy_id="bull_trend",
            )

        self.assertTrue(profile.applied)
        self.assertEqual(profile.suggested_signal, "hold")
        self.assertEqual(profile.source_scope, "策略+学习模型")
        self.assertIsNotNone(profile.model_prediction)

    def test_strategy_fallback_summary_does_not_short_circuit_heuristic_profile(self) -> None:
        self.config.analysis_learning_model_enabled = False
        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        rows = [
            {
                "code": code,
                "market": "us",
                "model_used": "gemini/gemini-2.5-flash",
                "signal": "buy",
                "direction_correct": False,
                "simulated_return_pct": -5.0,
                "stock_return_pct": -5.0,
                "evaluated_at_ts": idx,
                "context_payload": {},
            }
            for idx, code in enumerate(("AAPL", "MSFT", "NVDA"), start=1)
        ]

        with patch(
            "src.services.analysis_calibration_service.BacktestService.get_strategy_summary",
            return_value={
                "strategy_id": "new_strategy",
                "source_scope": "overall",
                "is_fallback": True,
                "total_evaluations": 50,
                "win_rate": 0.9,
                "direction_accuracy": 0.9,
                "avg_return": 0.1,
            },
        ), patch.object(service, "_load_learning_rows", return_value=rows):
            profile = service._build_profile(
                stock_code="AAPL",
                current_signal="buy",
                model_used="gemini/gemini-2.5-flash",
                score=62,
                trend_prediction="看多",
                context_snapshot=self._make_context_snapshot(market="us", signal="buy"),
                strategy_id="new_strategy",
            )

        self.assertTrue(profile.applied)
        self.assertEqual(profile.suggested_signal, "hold")
        self.assertEqual(profile.source_scope, "市场+模型")
        self.assertIsNone(profile.strategy_id)


class PipelineLearningLoopTestCase(unittest.TestCase):
    @patch("src.core.pipeline.get_market_today", return_value=date(2026, 3, 24))
    @patch("src.core.pipeline.get_market_for_stock", return_value="us")
    def test_pipeline_calls_refresh_and_calibration_on_standard_path(
        self, _mock_market, _mock_market_today
    ) -> None:
        with patch("src.core.pipeline.get_config") as mock_config, \
             patch("src.core.pipeline.get_db"), \
             patch("src.core.pipeline.DataFetcherManager"), \
             patch("src.core.pipeline.GeminiAnalyzer"), \
             patch("src.core.pipeline.NotificationService"), \
             patch("src.core.pipeline.SearchService"):

            mock_cfg = MagicMock()
            mock_cfg.max_workers = 1
            mock_cfg.agent_mode = False
            mock_cfg.agent_skills = []
            mock_cfg.bocha_api_keys = []
            mock_cfg.tavily_api_keys = []
            mock_cfg.brave_api_keys = []
            mock_cfg.serpapi_keys = []
            mock_cfg.minimax_api_keys = []
            mock_cfg.news_max_age_days = 3
            mock_cfg.enable_realtime_quote = False
            mock_cfg.enable_chip_distribution = False
            mock_cfg.realtime_source_priority = []
            mock_cfg.save_context_snapshot = False
            mock_cfg.backtest_enabled = True
            mock_config.return_value = mock_cfg

            pipeline = StockAnalysisPipeline(config=mock_cfg)
            pipeline.calibration_service = MagicMock()
            pipeline.calibration_service.enabled = True
            pipeline.calibration_service.refresh_interval_seconds = 0
            pipeline.calibration_service._last_refresh_ts = 0.0
            pipeline.calibration_service.maybe_refresh_model = MagicMock()

            pipeline.fetcher_manager.get_stock_name.return_value = "Apple Inc."
            pipeline.fetcher_manager.get_realtime_quote.return_value = None
            pipeline.fetcher_manager.get_chip_distribution.return_value = None
            pipeline.fetcher_manager.get_fundamental_context.return_value = {"source_chain": [], "coverage": {}}
            pipeline.search_service.can_run_comprehensive_intel.return_value = False
            pipeline.search_service.build_market_intel_summary.return_value = {"market": "us", "market_label": "美股"}
            pipeline.db.get_data_range.return_value = []
            pipeline.db.get_analysis_context.return_value = {
                "code": "AAPL",
                "stock_name": "Apple Inc.",
                "date": "2026-03-24",
                "today": {},
                "yesterday": {},
            }

            raw_result = AnalysisResult(
                code="AAPL",
                name="Apple Inc.",
                sentiment_score=62,
                trend_prediction="看多",
                operation_advice="买入",
                decision_type="buy",
                analysis_summary="test",
            )
            pipeline.analyzer.analyze.return_value = raw_result
            pipeline.calibration_service.calibrate_result.side_effect = (
                lambda result, context_snapshot=None: result
            )

            result = pipeline.analyze_stock("AAPL", ReportType.SIMPLE, "q-learn")

            self.assertIsNotNone(result)
            pipeline.db.get_analysis_context.assert_called_once()
            call_args = pipeline.db.get_analysis_context.call_args
            self.assertEqual(call_args[0][0], "AAPL")
            self.assertEqual(call_args[1]["target_date"], date(2026, 3, 24))
            pipeline.calibration_service.maybe_refresh_backtests.assert_called_once()
            pipeline.calibration_service.maybe_refresh_model.assert_called_once()
            pipeline.calibration_service.calibrate_result.assert_called_once()


if __name__ == "__main__":
    unittest.main()

    def test_learning_rows_exclude_placeholder_analyses(self) -> None:
        """LLM 未参与时落库的「持有/震荡/50」占位结果不能进入学习集。"""
        self._insert_learning_record(
            query_id="q-real",
            code="AAPL",
            signal="buy",
            direction_correct=True,
            simulated_return_pct=3.0,
            stock_return_pct=3.0,
            model_used="gemini/gemini-2.5-flash",
        )
        self._insert_learning_record(
            query_id="q-stub",
            code="TSLA",
            signal="hold",
            direction_correct=True,
            simulated_return_pct=0.0,
            stock_return_pct=0.0,
            model_used="",
        )

        service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        rows = service._load_learning_rows()

        self.assertEqual([row["code"] for row in rows], ["AAPL"])

# -*- coding: utf-8 -*-
"""Backtest orchestration service."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, select

from src.config import get_config
from src.core.backtest_engine import (
    EXECUTION_LOGIC_REVISION,
    OVERALL_SENTINEL_CODE,
    BacktestEngine,
    EvaluationConfig,
)
from src.repositories.backtest_repo import BacktestRepository
from src.repositories.stock_repo import StockRepository
from src.services.analysis_calibration_model import SmallCalibrationModel
from src.services.benchmark_service import BenchmarkService
from src.core.trading_calendar import get_market_for_stock
from src.storage import BacktestResult, BacktestSummary, DatabaseManager

logger = logging.getLogger(__name__)


class BacktestService:
    """Service layer to run and query backtests."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = BacktestRepository(self.db)
        self.stock_repo = StockRepository(self.db)
        self.benchmark = BenchmarkService(self.db)

    def run_backtest(
        self,
        *,
        code: Optional[str] = None,
        force: bool = False,
        eval_window_days: Optional[int] = None,
        min_age_days: Optional[int] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Evaluate every configured horizon and aggregate the run stats."""
        config = get_config()

        primary_window = int(
            eval_window_days
            if eval_window_days is not None
            else getattr(config, "backtest_eval_window_days", 10)
        )
        if min_age_days is None:
            min_age_days = getattr(config, "backtest_min_age_days", 14)
        engine_version = getattr(config, "backtest_engine_version", "v1")
        neutral_band_pct = float(getattr(config, "backtest_neutral_band_pct", 2.0))
        entry_mode = BacktestEngine._normalize_entry_mode(
            getattr(config, "backtest_entry_mode", "next_open")
        )
        effective_engine_version = self._effective_engine_version(
            str(engine_version),
            entry_mode,
        )

        windows = self._resolve_windows(primary_window)
        totals = {
            "processed": 0,
            "saved": 0,
            "completed": 0,
            "insufficient": 0,
            "errors": 0,
            "data_quality": 0,
        }
        per_window: Dict[str, Dict[str, Any]] = {}
        for window in windows:
            eval_config = EvaluationConfig(
                eval_window_days=int(window),
                neutral_band_pct=neutral_band_pct,
                engine_version=effective_engine_version,
                entry_mode=entry_mode,
                commission_bps=float(getattr(config, "backtest_commission_bps", 0.0) or 0.0),
                slippage_bps=float(getattr(config, "backtest_slippage_bps", 0.0) or 0.0),
            )
            stats = self._run_window(
                code=code,
                force=force,
                eval_window_days=int(window),
                min_age_days=int(min_age_days),
                limit=int(limit),
                effective_engine_version=effective_engine_version,
                eval_config=eval_config,
                neutral_band_pct=neutral_band_pct,
            )
            per_window[str(window)] = stats
            for key in totals:
                totals[key] += int(stats.get(key, 0) or 0)

        return {
            **totals,
            "windows": windows,
            "per_window": per_window,
        }

    def _resolve_windows(self, primary_window: int) -> List[int]:
        """Return the primary window plus configured extra horizons."""
        primary = max(1, int(primary_window))
        raw = getattr(get_config(), "backtest_extra_windows", "")
        extras: List[int] = []
        if isinstance(raw, str):
            for token in raw.replace("，", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    value = int(token)
                except ValueError:
                    continue
                if 1 <= value <= 120:
                    extras.append(value)
        elif isinstance(raw, (list, tuple)):
            extras = [int(value) for value in raw if str(value).strip().isdigit()]
        windows = sorted({primary, *extras})
        return windows or [primary]

    def _run_window(
        self,
        *,
        code: Optional[str],
        force: bool,
        eval_window_days: int,
        min_age_days: int,
        limit: int,
        effective_engine_version: str,
        eval_config: EvaluationConfig,
        neutral_band_pct: float,
    ) -> Dict[str, Any]:
        candidates = self.repo.get_candidates(
            code=code,
            min_age_days=int(min_age_days),
            limit=int(limit),
            eval_window_days=int(eval_window_days),
            engine_version=effective_engine_version,
            force=force,
        )

        processed = 0
        completed = 0
        insufficient = 0
        errors = 0
        data_quality = 0
        touched_codes: set[str] = set()

        results_to_save: List[BacktestResult] = []

        for analysis in candidates:
            processed += 1
            touched_codes.add(analysis.code)

            try:
                analysis_date = self._resolve_analysis_date(analysis)
                if analysis_date is None:
                    errors += 1
                    results_to_save.append(
                        BacktestResult(
                            analysis_history_id=analysis.id,
                            code=analysis.code,
                            eval_window_days=int(eval_window_days),
                            engine_version=effective_engine_version,
                            eval_status="error",
                            evaluated_at=datetime.now(),
                            operation_advice=analysis.operation_advice,
                            strategy_id=getattr(analysis, 'strategy_id', None),
                        )
                    )
                    continue
                start_daily = self.stock_repo.get_start_daily(code=analysis.code, analysis_date=analysis_date)

                if start_daily is None or start_daily.close is None:
                    self._try_fill_daily_data(code=analysis.code, analysis_date=analysis_date, eval_window_days=eval_window_days)
                    start_daily = self.stock_repo.get_start_daily(code=analysis.code, analysis_date=analysis_date)

                if start_daily is None or start_daily.close is None:
                    insufficient += 1
                    results_to_save.append(
                        BacktestResult(
                            analysis_history_id=analysis.id,
                            code=analysis.code,
                            analysis_date=analysis_date,
                            eval_window_days=int(eval_window_days),
                            engine_version=effective_engine_version,
                            eval_status="insufficient_data",
                            evaluated_at=datetime.now(),
                            operation_advice=analysis.operation_advice,
                            strategy_id=getattr(analysis, 'strategy_id', None),
                        )
                    )
                    continue

                forward_bars = self.stock_repo.get_forward_bars(
                    code=analysis.code,
                    analysis_date=start_daily.date,
                    eval_window_days=int(eval_window_days),
                )

                if len(forward_bars) < int(eval_window_days):
                    self._try_fill_daily_data(code=analysis.code, analysis_date=start_daily.date, eval_window_days=eval_window_days)
                    forward_bars = self.stock_repo.get_forward_bars(
                        code=analysis.code,
                        analysis_date=start_daily.date,
                        eval_window_days=int(eval_window_days),
                    )

                # 拆股/并股会在日线里留下 ±45% 以上的假跳变；这种窗口不能当真实收益回测。
                if self._has_suspicious_jump(
                    forward_bars, reference_close=start_daily.close
                ):
                    data_quality += 1
                    logger.warning(
                        "回测跳过疑似拆股/并股窗口: %s#%s %s",
                        analysis.code,
                        analysis.id,
                        start_daily.date,
                    )
                    results_to_save.append(
                        BacktestResult(
                            analysis_history_id=analysis.id,
                            code=analysis.code,
                            analysis_date=start_daily.date,
                            eval_window_days=int(eval_window_days),
                            engine_version=effective_engine_version,
                            eval_status="data_quality",
                            evaluated_at=datetime.now(),
                            operation_advice=analysis.operation_advice,
                            strategy_id=getattr(analysis, "strategy_id", None),
                        )
                    )
                    continue

                evaluation = BacktestEngine.evaluate_single(
                    operation_advice=analysis.operation_advice,
                    analysis_date=start_daily.date,
                    start_price=float(start_daily.close),
                    forward_bars=forward_bars,
                    stop_loss=analysis.stop_loss,
                    take_profit=analysis.take_profit,
                    config=eval_config,
                )

                # 双轨归因：同一条分析、同一段行情，再判一次"校准前"的原始建议，
                # 才能回答"结论变准是 LLM 本来就对，还是校准救回来的"。
                raw_advice = self._resolve_raw_advice(analysis, evaluation.get("operation_advice"))
                raw_direction = BacktestEngine.infer_direction_expected(raw_advice)
                raw_direction_correct: Optional[bool] = None
                arm_advice = self._resolve_calibration_arm_advice(
                    analysis, evaluation.get("operation_advice")
                )
                arm_direction = BacktestEngine.infer_direction_expected(arm_advice)
                arm_direction_correct: Optional[bool] = None
                if (
                    evaluation.get("eval_status") == "completed"
                    and evaluation.get("stock_return_pct") is not None
                ):
                    _, raw_direction_correct = BacktestEngine.classify_direction(
                        stock_return_pct=float(evaluation["stock_return_pct"]),
                        direction_expected=raw_direction,
                        neutral_band_pct=neutral_band_pct,
                    )
                    _, arm_direction_correct = BacktestEngine.classify_direction(
                        stock_return_pct=float(evaluation["stock_return_pct"]),
                        direction_expected=arm_direction,
                        neutral_band_pct=neutral_band_pct,
                    )

                status = evaluation.get("eval_status")
                benchmark_return: Optional[float] = None
                excess_return: Optional[float] = None
                if status == "completed" and forward_bars:
                    benchmark_return = self.benchmark.return_pct(
                        get_market_for_stock(analysis.code) or "cn",
                        start_daily.date,
                        forward_bars[-1].date,
                    )
                    if (
                        benchmark_return is not None
                        and evaluation.get("stock_return_pct") is not None
                    ):
                        excess_return = round(
                            float(evaluation["stock_return_pct"]) - benchmark_return, 2
                        )
                if status == "insufficient_data":
                    insufficient += 1
                elif status == "completed":
                    completed += 1
                else:
                    errors += 1

                results_to_save.append(
                    BacktestResult(
                        analysis_history_id=analysis.id,
                        code=analysis.code,
                        analysis_date=evaluation.get("analysis_date"),
                        eval_window_days=int(evaluation.get("eval_window_days") or eval_window_days),
                        engine_version=str(evaluation.get("engine_version") or effective_engine_version),
                        eval_status=str(evaluation.get("eval_status") or "error"),
                        evaluated_at=datetime.now(),
                        operation_advice=evaluation.get("operation_advice"),
                        position_recommendation=evaluation.get("position_recommendation"),
                        raw_operation_advice=raw_advice,
                        raw_direction_expected=raw_direction,
                        raw_direction_correct=raw_direction_correct,
                        calibration_arm_operation_advice=arm_advice,
                        calibration_arm_direction_expected=arm_direction,
                        calibration_arm_direction_correct=arm_direction_correct,
                        start_price=evaluation.get("start_price"),
                        end_close=evaluation.get("end_close"),
                        max_high=evaluation.get("max_high"),
                        min_low=evaluation.get("min_low"),
                        stock_return_pct=evaluation.get("stock_return_pct"),
                        direction_expected=evaluation.get("direction_expected"),
                        direction_correct=evaluation.get("direction_correct"),
                        outcome=evaluation.get("outcome"),
                        stop_loss=evaluation.get("stop_loss"),
                        take_profit=evaluation.get("take_profit"),
                        hit_stop_loss=evaluation.get("hit_stop_loss"),
                        hit_take_profit=evaluation.get("hit_take_profit"),
                        first_hit=evaluation.get("first_hit"),
                        first_hit_date=evaluation.get("first_hit_date"),
                        first_hit_trading_days=evaluation.get("first_hit_trading_days"),
                        simulated_entry_price=evaluation.get("simulated_entry_price"),
                        simulated_exit_price=evaluation.get("simulated_exit_price"),
                        simulated_exit_reason=evaluation.get("simulated_exit_reason"),
                        simulated_return_pct=evaluation.get("simulated_return_pct"),
                        simulated_return_pct_gross=evaluation.get("simulated_return_pct_gross"),
                        mfe_pct=evaluation.get("mfe_pct"),
                        mae_pct=evaluation.get("mae_pct"),
                        days_to_mfe=evaluation.get("days_to_mfe"),
                        days_to_mae=evaluation.get("days_to_mae"),
                        benchmark_return_pct=benchmark_return,
                        excess_return_pct=excess_return,
                        strategy_id=getattr(analysis, 'strategy_id', None),
                    )
                )

            except Exception as exc:
                errors += 1
                logger.error(f"回测失败: {analysis.code}#{analysis.id}: {exc}")
                results_to_save.append(
                    BacktestResult(
                        analysis_history_id=analysis.id,
                        code=analysis.code,
                        analysis_date=self._resolve_analysis_date(analysis),
                        eval_window_days=int(eval_window_days),
                        engine_version=effective_engine_version,
                        eval_status="error",
                        evaluated_at=datetime.now(),
                        operation_advice=analysis.operation_advice,
                        strategy_id=getattr(analysis, 'strategy_id', None),
                    )
                )

        saved = 0
        if results_to_save:
            saved = self.repo.save_results_batch(results_to_save, replace_existing=force)

        if saved:
            self._recompute_summaries(
                touched_codes=sorted(touched_codes),
                eval_window_days=int(eval_window_days),
                engine_version=effective_engine_version,
            )

        return {
            "processed": processed,
            "saved": saved,
            "completed": completed,
            "insufficient": insufficient,
            "errors": errors,
            "data_quality": data_quality,
        }

    def get_recent_evaluations(self, *, code: Optional[str], eval_window_days: Optional[int] = None, limit: int = 50, page: int = 1) -> Dict[str, Any]:
        offset = max(page - 1, 0) * limit
        rows, total = self.repo.get_results_paginated(code=code, eval_window_days=eval_window_days, days=None, offset=offset, limit=limit)
        items = [self._result_to_dict(r) for r in rows]
        return {"total": total, "page": page, "limit": limit, "items": items}

    def get_summary(self, *, scope: str, code: Optional[str], eval_window_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        config = get_config()
        entry_mode = BacktestEngine._normalize_entry_mode(
            getattr(config, "backtest_entry_mode", "next_open")
        )
        engine_version = self._effective_engine_version(
            str(getattr(config, "backtest_engine_version", "v1")),
            entry_mode,
        )
        lookup_code = OVERALL_SENTINEL_CODE if scope == "overall" else code
        summary = self.repo.get_summary(
            scope=scope,
            code=lookup_code,
            eval_window_days=eval_window_days,
            engine_version=engine_version,
        )
        if summary is None:
            return None
        return self._summary_to_dict(summary)

    def get_global_summary(self, *, eval_window_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return overall backtest metrics normalized for Agent memory consumers."""
        return self._normalize_learning_summary(
            self.get_summary(scope="overall", code=None, eval_window_days=eval_window_days)
        )

    def get_stock_summary(self, code: str, *, eval_window_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return per-stock backtest metrics normalized for Agent memory consumers."""
        return self._normalize_learning_summary(
            self.get_summary(scope="stock", code=code, eval_window_days=eval_window_days)
        )

    def get_strategy_summary(self, strategy_id: str, *, eval_window_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return real strategy-level backtest summary metrics.

        Queries pre-computed strategy-dimension summaries from the backtest
        storage layer. Falls back to the overall rollup only as a clearly
        marked legacy baseline when no strategy summary has been accumulated.
        """
        summary = self.get_summary(scope="strategy", code=strategy_id, eval_window_days=eval_window_days)
        if summary is not None:
            normalized = self._normalize_learning_summary(summary)
            if normalized is not None:
                normalized["strategy_id"] = strategy_id
                normalized["source_scope"] = "strategy"
                normalized["is_fallback"] = False
            return normalized

        fallback = self.get_global_summary(eval_window_days=eval_window_days)
        if fallback is None:
            return None
        fallback["strategy_id"] = strategy_id
        fallback["source_scope"] = "overall"
        fallback["is_fallback"] = True
        return fallback

    def _resolve_analysis_date(self, analysis) -> Optional[date]:
        parsed = self.repo.parse_analysis_date_from_snapshot(analysis.context_snapshot)
        if parsed:
            return parsed
        if getattr(analysis, "created_at", None):
            return analysis.created_at.date()
        logger.warning(f"无法确定分析日期，跳过记录: {analysis.code}#{getattr(analysis, 'id', '?')}")
        return None

    @staticmethod
    def _has_suspicious_jump(
        bars,
        *,
        reference_close: Optional[float] = None,
        threshold_pct: float = 45.0,
    ) -> bool:
        """Detect a split/merge-sized close-to-close jump inside the window."""
        closes = [float(bar.close) for bar in bars if getattr(bar, "close", None)]
        if reference_close:
            closes.insert(0, float(reference_close))
        for previous, current in zip(closes, closes[1:]):
            if previous <= 0:
                continue
            if abs((current - previous) / previous * 100) > threshold_pct:
                return True
        return False

    @staticmethod
    def _resolve_raw_advice(analysis, final_advice: Optional[str]) -> str:
        """Return the pre-calibration advice for the same analysis record.

        ``calibration_info.original_advice`` is written at analysis time; for
        older rows fall back to the stored raw signal, then to the final advice
        (no calibration applied means raw == final).
        """
        raw_payload = getattr(analysis, "raw_result", None)
        payload: Dict[str, Any] = {}
        if isinstance(raw_payload, dict):
            payload = raw_payload
        elif isinstance(raw_payload, str) and raw_payload.strip():
            try:
                parsed = json.loads(raw_payload)
                if isinstance(parsed, dict):
                    payload = parsed
            except (TypeError, ValueError):
                payload = {}

        calibration = payload.get("calibration_info")
        if not isinstance(calibration, dict) or not calibration.get("applied"):
            return str(final_advice or "")

        original_advice = str(calibration.get("original_advice") or "").strip()
        if original_advice:
            return original_advice

        signal = str(calibration.get("current_signal") or "").strip().lower()
        return {
            "buy": "买入",
            "sell": "减仓/卖出",
            "hold": "持有",
        }.get(signal, str(final_advice or ""))

    @staticmethod
    def _resolve_calibration_arm_advice(analysis, final_advice: Optional[str]) -> str:
        """Return what the calibrator suggested, even when it was paused.

        Applied rows already carry the calibrated advice as ``operation_advice``.
        Paused rows keep the raw advice, so the shadow arm is reconstructed from
        ``calibration_info.suggested_signal``; this keeps lift measurable while
        calibration is disabled, allowing it to recover automatically.
        """
        raw_payload = getattr(analysis, "raw_result", None)
        payload: Dict[str, Any] = {}
        if isinstance(raw_payload, dict):
            payload = raw_payload
        elif isinstance(raw_payload, str) and raw_payload.strip():
            try:
                parsed = json.loads(raw_payload)
                if isinstance(parsed, dict):
                    payload = parsed
            except (TypeError, ValueError):
                payload = {}

        calibration = payload.get("calibration_info")
        if not isinstance(calibration, dict):
            return str(final_advice or "")
        if calibration.get("applied"):
            # 已应用的校准，实际结论就是校准建议
            return str(final_advice or "")

        suggested = str(calibration.get("suggested_signal") or "").strip().lower()
        current = str(calibration.get("current_signal") or "").strip().lower()
        if suggested in {"buy", "hold", "sell"} and suggested != current:
            return {
                "buy": "买入",
                "sell": "减仓/卖出",
                "hold": "持有",
            }.get(suggested, str(final_advice or ""))
        return str(final_advice or "")

    @staticmethod
    def _effective_engine_version(engine_version: str, entry_mode: str) -> str:
        suffix_map = {
            "analysis_close": "ac",
            "next_open": "no",
            "next_close": "nc",
        }
        base = str(engine_version or "v1").strip() or "v1"
        if "@" in base:
            return base
        return f"{base}@{suffix_map.get(entry_mode, 'no')}-{EXECUTION_LOGIC_REVISION}"

    def _try_fill_daily_data(self, *, code: str, analysis_date: date, eval_window_days: int) -> None:
        try:
            from data_provider.base import DataFetcherManager

            # fetch a window that covers start + forward bars
            end_date = analysis_date + timedelta(days=max(eval_window_days * 2, 30))
            manager = DataFetcherManager()
            df, source = manager.get_daily_data(
                stock_code=code,
                start_date=analysis_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                days=eval_window_days * 2,
            )
            if df is None or df.empty:
                return
            self.db.save_daily_data(df, code=code, data_source=source)
        except Exception as exc:
            logger.warning(f"补全日线数据失败({code}): {exc}")

    def _recompute_summaries(self, *, touched_codes: List[str], eval_window_days: int, engine_version: str) -> None:
        with self.db.get_session() as session:
            # overall
            overall_rows = session.execute(
                select(BacktestResult).where(
                    and_(
                        BacktestResult.eval_window_days == eval_window_days,
                        BacktestResult.engine_version == engine_version,
                    )
                )
            ).scalars().all()
            overall_data = BacktestEngine.compute_summary(
                results=overall_rows,
                scope="overall",
                code=OVERALL_SENTINEL_CODE,
                eval_window_days=eval_window_days,
                engine_version=engine_version,
            )
            overall_summary = self._build_summary_model(overall_data)
            self.repo.upsert_summary(overall_summary)

            for code in touched_codes:
                rows = session.execute(
                    select(BacktestResult).where(
                        and_(
                            BacktestResult.code == code,
                            BacktestResult.eval_window_days == eval_window_days,
                            BacktestResult.engine_version == engine_version,
                        )
                    )
                ).scalars().all()
                data = BacktestEngine.compute_summary(
                    results=rows,
                    scope="stock",
                    code=code,
                    eval_window_days=eval_window_days,
                    engine_version=engine_version,
                )
                summary = self._build_summary_model(data)
                self.repo.upsert_summary(summary)

            strategy_ids = sorted(
                {
                    str(row.strategy_id).strip()
                    for row in overall_rows
                    if str(getattr(row, "strategy_id", "") or "").strip()
                }
            )
            for strategy_id in strategy_ids:
                rows = [
                    row
                    for row in overall_rows
                    if str(getattr(row, "strategy_id", "") or "").strip() == strategy_id
                ]
                data = BacktestEngine.compute_summary(
                    results=rows,
                    scope="strategy",
                    code=strategy_id,
                    eval_window_days=eval_window_days,
                    engine_version=engine_version,
                )
                summary = self._build_summary_model(data)
                self.repo.upsert_summary(summary)

    @staticmethod
    def _build_summary_model(summary_data: Dict[str, Any]) -> BacktestSummary:
        return BacktestSummary(
            scope=summary_data.get("scope"),
            code=summary_data.get("code"),
            eval_window_days=summary_data.get("eval_window_days"),
            engine_version=summary_data.get("engine_version"),
            computed_at=datetime.now(),
            total_evaluations=summary_data.get("total_evaluations") or 0,
            completed_count=summary_data.get("completed_count") or 0,
            insufficient_count=summary_data.get("insufficient_count") or 0,
            long_count=summary_data.get("long_count") or 0,
            cash_count=summary_data.get("cash_count") or 0,
            win_count=summary_data.get("win_count") or 0,
            loss_count=summary_data.get("loss_count") or 0,
            neutral_count=summary_data.get("neutral_count") or 0,
            direction_accuracy_pct=summary_data.get("direction_accuracy_pct"),
            raw_direction_accuracy_pct=summary_data.get("raw_direction_accuracy_pct"),
            calibration_lift_pct=summary_data.get("calibration_lift_pct"),
            calibration_sample_count=summary_data.get("calibration_sample_count") or 0,
            win_rate_pct=summary_data.get("win_rate_pct"),
            neutral_rate_pct=summary_data.get("neutral_rate_pct"),
            avg_stock_return_pct=summary_data.get("avg_stock_return_pct"),
            avg_simulated_return_pct=summary_data.get("avg_simulated_return_pct"),
            avg_simulated_return_pct_gross=summary_data.get("avg_simulated_return_pct_gross"),
            avg_mfe_pct=summary_data.get("avg_mfe_pct"),
            avg_mae_pct=summary_data.get("avg_mae_pct"),
            avg_benchmark_return_pct=summary_data.get("avg_benchmark_return_pct"),
            avg_excess_return_pct=summary_data.get("avg_excess_return_pct"),
            excess_win_rate_pct=summary_data.get("excess_win_rate_pct"),
            stop_loss_trigger_rate=summary_data.get("stop_loss_trigger_rate"),
            take_profit_trigger_rate=summary_data.get("take_profit_trigger_rate"),
            ambiguous_rate=summary_data.get("ambiguous_rate"),
            avg_days_to_first_hit=summary_data.get("avg_days_to_first_hit"),
            advice_breakdown_json=json.dumps(summary_data.get("advice_breakdown") or {}, ensure_ascii=False),
            diagnostics_json=json.dumps(summary_data.get("diagnostics") or {}, ensure_ascii=False),
        )

    @staticmethod
    def _result_to_dict(row: BacktestResult) -> Dict[str, Any]:
        return row.to_dict()

    @staticmethod
    def _summary_to_dict(row: BacktestSummary) -> Dict[str, Any]:
        return {
            "scope": row.scope,
            "code": None if row.code == OVERALL_SENTINEL_CODE else row.code,
            "eval_window_days": row.eval_window_days,
            "engine_version": row.engine_version,
            "computed_at": row.computed_at.isoformat() if row.computed_at else None,
            "total_evaluations": row.total_evaluations,
            "completed_count": row.completed_count,
            "insufficient_count": row.insufficient_count,
            "long_count": row.long_count,
            "cash_count": row.cash_count,
            "win_count": row.win_count,
            "loss_count": row.loss_count,
            "neutral_count": row.neutral_count,
            "direction_accuracy_pct": row.direction_accuracy_pct,
            "raw_direction_accuracy_pct": row.raw_direction_accuracy_pct,
            "calibration_lift_pct": row.calibration_lift_pct,
            "calibration_sample_count": int(row.calibration_sample_count or 0),
            "win_rate_pct": row.win_rate_pct,
            "neutral_rate_pct": row.neutral_rate_pct,
            "avg_stock_return_pct": row.avg_stock_return_pct,
            "avg_simulated_return_pct": row.avg_simulated_return_pct,
            "avg_simulated_return_pct_gross": row.avg_simulated_return_pct_gross,
            "avg_mfe_pct": row.avg_mfe_pct,
            "avg_mae_pct": row.avg_mae_pct,
            "avg_benchmark_return_pct": row.avg_benchmark_return_pct,
            "avg_excess_return_pct": row.avg_excess_return_pct,
            "excess_win_rate_pct": row.excess_win_rate_pct,
            "stop_loss_trigger_rate": row.stop_loss_trigger_rate,
            "take_profit_trigger_rate": row.take_profit_trigger_rate,
            "ambiguous_rate": row.ambiguous_rate,
            "avg_days_to_first_hit": row.avg_days_to_first_hit,
            "advice_breakdown": json.loads(row.advice_breakdown_json) if row.advice_breakdown_json else {},
            "diagnostics": json.loads(row.diagnostics_json) if row.diagnostics_json else {},
            "learning_model": BacktestService._learning_model_metrics(),
        }

    @staticmethod
    def _learning_model_metrics() -> Dict[str, Any]:
        """Expose persisted learning calibration model metrics with backtest summaries."""
        config = get_config()
        if not bool(getattr(config, "analysis_learning_model_enabled", True)):
            return {"enabled": False}

        model_path = str(
            getattr(
                config,
                "analysis_learning_model_path",
                "./data/models/analysis_calibration_model.json",
            )
        )
        model = SmallCalibrationModel.load(model_path)
        if model is None:
            return {
                "enabled": True,
                "available": False,
                "path": model_path,
            }

        return {
            "enabled": True,
            "available": True,
            "path": model_path,
            "sample_count": model.sample_count,
            "validation_accuracy": model.validation_accuracy,
            "validation_count": model.validation_count,
            "validation_metrics": model.validation_metrics,
            "label_distribution": model.label_distribution,
            "baseline_accuracy": model.baseline_accuracy,
            "backend": model.preferred_backend,
            "scopes": sorted(model.submodels.keys()),
            "scope_metrics": {
                scope: {
                    "sample_count": submodel.get("sample_count"),
                    "validation_accuracy": submodel.get("validation_accuracy"),
                    "validation_count": submodel.get("validation_count"),
                    "validation_metrics": submodel.get("validation_metrics"),
                    "label_distribution": submodel.get("label_distribution"),
                    "baseline_accuracy": submodel.get("baseline_accuracy"),
                    "engine": submodel.get("engine"),
                }
                for scope, submodel in model.submodels.items()
            },
        }

    @staticmethod
    def _normalize_learning_summary(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize summary metrics to the ratio-based shape expected by Agent memory."""
        if summary is None:
            return None

        normalized = dict(summary)
        normalized["win_rate"] = BacktestService._pct_to_ratio(summary.get("win_rate_pct"), default=0.5)
        normalized["direction_accuracy"] = BacktestService._pct_to_ratio(
            summary.get("direction_accuracy_pct"),
            default=0.5,
        )

        avg_return_pct = summary.get("avg_simulated_return_pct")
        if avg_return_pct is None:
            avg_return_pct = summary.get("avg_stock_return_pct")
        normalized["avg_return"] = BacktestService._pct_to_ratio(avg_return_pct, default=0.0)
        return normalized

    @staticmethod
    def _pct_to_ratio(value: Optional[float], default: float = 0.0) -> float:
        try:
            return float(value) / 100.0
        except (TypeError, ValueError):
            return default

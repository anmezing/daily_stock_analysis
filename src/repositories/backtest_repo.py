# -*- coding: utf-8 -*-
"""Backtest repository.

Provides database access helpers for backtest tables.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, delete, desc, func, select

from src.storage import BacktestResult, BacktestSummary, DatabaseManager, AnalysisHistory

logger = logging.getLogger(__name__)


class BacktestRepository:
    """DB access layer for backtesting."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()

    _PLACEHOLDER_ADVICES = {"", "待补充", "仅拉取数据", "未分析"}

    @staticmethod
    def _is_evaluable(analysis: AnalysisHistory) -> bool:
        """Return True only for analyses that carry a real, structured LLM verdict.

        Placeholder rows (LLM key missing), degraded keyword-only parses and
        "待补充" advisories must never enter direction accuracy or win rate:
        they have no model_used and no structured conclusion, so counting them
        silently drags every metric toward the placeholder "持有/震荡/50".
        """
        raw = getattr(analysis, "raw_result", None)
        payload: Dict[str, Any] = {}
        if isinstance(raw, dict):
            payload = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except (TypeError, ValueError):
                payload = {}

        if not str(payload.get("model_used") or "").strip():
            return False
        if payload.get("success") is False:
            return False
        if str(payload.get("parse_mode") or "structured").strip().lower() in {
            "text_fallback",
            "truncated",
        }:
            return False

        advice = str(
            getattr(analysis, "operation_advice", "")
            or payload.get("operation_advice")
            or ""
        ).strip()
        return advice not in BacktestRepository._PLACEHOLDER_ADVICES

    def get_candidates(
        self,
        *,
        code: Optional[str],
        min_age_days: int,
        limit: int,
        eval_window_days: int,
        engine_version: str,
        force: bool,
    ) -> List[AnalysisHistory]:
        """Return AnalysisHistory rows eligible for backtest."""
        cutoff_dt = datetime.now() - timedelta(days=min_age_days)

        with self.db.get_session() as session:
            conditions = [AnalysisHistory.created_at <= cutoff_dt]
            if code:
                conditions.append(AnalysisHistory.code == code)

            query = select(AnalysisHistory).where(and_(*conditions))

            if not force:
                existing_ids = select(BacktestResult.analysis_history_id).where(
                    and_(
                        BacktestResult.eval_window_days == eval_window_days,
                        BacktestResult.engine_version == engine_version,
                        BacktestResult.eval_status == "completed",
                    )
                )
                query = query.where(AnalysisHistory.id.not_in(existing_ids))

            query = query.order_by(desc(AnalysisHistory.created_at)).limit(limit)
            rows = session.execute(query).scalars().all()

        evaluable = [row for row in rows if self._is_evaluable(row)]
        skipped = len(rows) - len(evaluable)
        if skipped:
            logger.info(
                "回测跳过 %s 条占位/降级分析记录（无 model_used、解析失败或待补充）",
                skipped,
            )

        # 同一天同一只股票可能被手动/定时重复分析多次，结论和收益完全相同。
        # 全部计入会让某一天在统计里被重复加权（还可能是同一天里矛盾的结论）。
        # rows 已按 created_at 倒序，保留每天最新的一条。
        deduped: List[AnalysisHistory] = []
        seen: set = set()
        for row in evaluable:
            signal_day = self.parse_analysis_date_from_snapshot(
                getattr(row, "context_snapshot", None)
            ) or (row.created_at.date() if getattr(row, "created_at", None) else None)
            key = (getattr(row, "code", None), signal_day)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        if len(deduped) != len(evaluable):
            logger.info(
                "回测跳过 %s 条同一信号日的重复分析",
                len(evaluable) - len(deduped),
            )
        return deduped

    def save_result(self, result: BacktestResult) -> None:
        with self.db.get_session() as session:
            session.add(result)
            session.commit()

    def save_results_batch(self, results: List[BacktestResult], *, replace_existing: bool = False) -> int:
        if not results:
            return 0

        with self.db.get_session() as session:
            try:
                analysis_ids = sorted({r.analysis_history_id for r in results if r.analysis_history_id is not None})
                key_pairs = sorted({(r.eval_window_days, r.engine_version) for r in results})

                if replace_existing:
                    if analysis_ids and key_pairs:
                        for window_days, engine_version in key_pairs:
                            session.execute(
                                delete(BacktestResult).where(
                                    and_(
                                        BacktestResult.analysis_history_id.in_(analysis_ids),
                                        BacktestResult.eval_window_days == window_days,
                                        BacktestResult.engine_version == engine_version,
                                    )
                                )
                            )
                elif analysis_ids and key_pairs:
                    # Retried rows (insufficient_data/error) are written again on the
                    # same unique key, so clear the stale non-completed rows first.
                    for window_days, engine_version in key_pairs:
                        session.execute(
                            delete(BacktestResult).where(
                                and_(
                                    BacktestResult.analysis_history_id.in_(analysis_ids),
                                    BacktestResult.eval_window_days == window_days,
                                    BacktestResult.engine_version == engine_version,
                                    BacktestResult.eval_status != "completed",
                                )
                            )
                        )

                session.add_all(results)
                session.commit()
                return len(results)
            except Exception as exc:
                session.rollback()
                logger.error(f"批量保存回测结果失败: {exc}")
                raise

    def get_results_paginated(
        self,
        *,
        code: Optional[str],
        eval_window_days: Optional[int] = None,
        days: Optional[int],
        offset: int,
        limit: int,
    ) -> Tuple[List[BacktestResult], int]:
        with self.db.get_session() as session:
            conditions = []
            if code:
                conditions.append(BacktestResult.code == code)
            if eval_window_days is not None:
                conditions.append(BacktestResult.eval_window_days == eval_window_days)
            if days:
                cutoff = datetime.now() - timedelta(days=int(days))
                conditions.append(BacktestResult.evaluated_at >= cutoff)

            where_clause = and_(*conditions) if conditions else True

            total = session.execute(select(func.count(BacktestResult.id)).where(where_clause)).scalar() or 0
            rows = session.execute(
                select(BacktestResult)
                .where(where_clause)
                .order_by(desc(BacktestResult.evaluated_at))
                .offset(offset)
                .limit(limit)
            ).scalars().all()
            return list(rows), int(total)

    def upsert_summary(self, summary: BacktestSummary) -> None:
        """Insert or replace summary row by unique key."""
        with self.db.get_session() as session:
            existing = session.execute(
                select(BacktestSummary)
                .where(
                    and_(
                        BacktestSummary.scope == summary.scope,
                        BacktestSummary.code == summary.code,
                        BacktestSummary.eval_window_days == summary.eval_window_days,
                        BacktestSummary.engine_version == summary.engine_version,
                    )
                )
                .limit(1)
            ).scalar_one_or_none()

            if existing:
                for attr in (
                    "computed_at",
                    "total_evaluations",
                    "completed_count",
                    "insufficient_count",
                    "long_count",
                    "cash_count",
                    "win_count",
                    "loss_count",
                    "neutral_count",
                    "direction_accuracy_pct",
                    "raw_direction_accuracy_pct",
                    "calibration_lift_pct",
                    "calibration_sample_count",
                    "win_rate_pct",
                    "neutral_rate_pct",
                    "avg_stock_return_pct",
                    "avg_simulated_return_pct",
                    "stop_loss_trigger_rate",
                    "take_profit_trigger_rate",
                    "ambiguous_rate",
                    "avg_days_to_first_hit",
                    "advice_breakdown_json",
                    "diagnostics_json",
                ):
                    setattr(existing, attr, getattr(summary, attr))
                session.commit()
                return

            session.add(summary)
            session.commit()

    def get_summary(
        self,
        *,
        scope: str,
        code: Optional[str],
        eval_window_days: Optional[int] = None,
        engine_version: str,
    ) -> Optional[BacktestSummary]:
        with self.db.get_session() as session:
            conditions = [
                BacktestSummary.scope == scope,
                BacktestSummary.code == code,
                BacktestSummary.engine_version == engine_version,
            ]
            if eval_window_days is not None:
                conditions.append(BacktestSummary.eval_window_days == eval_window_days)

            row = session.execute(
                select(BacktestSummary)
                .where(and_(*conditions))
                .order_by(desc(BacktestSummary.computed_at))
                .limit(1)
            ).scalar_one_or_none()
            return row

    @staticmethod
    def parse_analysis_date_from_snapshot(context_snapshot: Optional[str]) -> Optional[date]:
        if not context_snapshot:
            return None

        try:
            payload = json.loads(context_snapshot)
        except (TypeError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        enhanced = payload.get("enhanced_context")
        if isinstance(enhanced, dict):
            date_str = enhanced.get("date")
        else:
            date_str = None
        if not date_str:
            # Agent 链路保存的是 initial_context，没有 enhanced_context，
            # 但会显式带上 analysis_date；缺少它就只能退回到 created_at，
            # 而 created_at 是服务器本地时间，跨时区时会错一天。
            date_str = payload.get("analysis_date") or payload.get("date")
        if not date_str:
            return None

        try:
            return datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None

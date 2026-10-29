# -*- coding: utf-8 -*-
"""Backtest-driven calibration for the standard analysis pipeline.

The project already contains backtesting and agent memory primitives, but the
classic single-shot analysis path does not consume that feedback.  This
service closes the smallest useful loop:

1. Periodically run incremental backtests for older analysis records.
2. Aggregate historical signal accuracy by stock / market / model.
3. Use those metrics to dampen, downgrade, or occasionally reverse weak
   low-quality signals before the report is persisted and notified.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from sqlalchemy import desc, select

from src.core.backtest_engine import BacktestEngine
from src.core.trading_calendar import get_market_for_stock
from src.services.analysis_calibration_model import SmallCalibrationModel
from src.services.backtest_service import BacktestService
from src.storage import AnalysisHistory, BacktestResult, DatabaseManager

logger = logging.getLogger(__name__)

_DEFAULT_REFRESH_INTERVAL_MINUTES = 60
_DEFAULT_MIN_SAMPLES = 20
_DEFAULT_HISTORY_LIMIT = 800
_DEFAULT_AUTO_BACKTEST_LIMIT = 200


def _safe_ratio(value: Optional[float], default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signal_from_operation_advice(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "hold"

    buy_tokens = ("买入", "加仓", "建仓", "buy", "long", "bull")
    sell_tokens = ("卖出", "减仓", "清仓", "sell", "short", "bear")
    hold_tokens = ("持有", "观望", "等待", "wait", "watch", "hold", "neutral")

    if any(token in text for token in buy_tokens):
        return "buy"
    if any(token in text for token in sell_tokens):
        return "sell"
    if any(token in text for token in hold_tokens):
        return "hold"
    return "hold"


def _pct_to_fraction(value: Any) -> float:
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return 0.0


@dataclass
class SignalStats:
    signal: str
    samples: int = 0
    accuracy: float = 0.5
    avg_return: float = 0.0
    # 该方向在同一批样本里的自然发生率（上涨/下跌/非上涨）。
    # accuracy 减去 base_rate 才是"信号本身"的增量，跨信号直接比 accuracy 没有意义。
    base_rate: float = 0.0

    @property
    def skill(self) -> float:
        return self.accuracy - self.base_rate


@dataclass
class CalibrationProfile:
    enabled: bool = False
    market: str = "cn"
    model_used: Optional[str] = None
    source_scope: str = ""
    samples: int = 0
    threshold: int = _DEFAULT_MIN_SAMPLES
    current_signal: str = "hold"
    # 校准前的原始结论：训练学习模型、双轨回测都要用它
    original_score: int = 50
    original_trend: str = ""
    original_advice: str = ""
    current_signal_stats: Optional[SignalStats] = None
    best_alternative_stats: Optional[SignalStats] = None
    suggested_signal: Optional[str] = None
    paused: bool = False
    score_adjustment: int = 0
    applied: bool = False
    reason: str = ""
    model_prediction: Optional[Dict[str, Any]] = None
    strategy_id: Optional[str] = None
    source_is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, stats in (
            ("current_signal_stats", self.current_signal_stats),
            ("best_alternative_stats", self.best_alternative_stats),
        ):
            if stats is not None:
                stats_payload = asdict(stats)
                stats_payload["skill"] = round(stats.skill, 4)
                payload[key] = stats_payload
        if self.model_prediction is not None:
            payload["model_prediction"] = dict(self.model_prediction)
        return payload


def _median(values: List[float]) -> float:
    """中位数（不引入 numpy 依赖）。"""
    ordered = sorted(values)
    size = len(ordered)
    middle = size // 2
    if size % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


class AnalysisCalibrationService:
    """Use historical backtest outcomes to calibrate current signals."""

    def __init__(
        self,
        *,
        db_manager: Optional[DatabaseManager] = None,
        config: Optional[Any] = None,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager.get_instance()
        self.config = config
        self._time_fn = time_fn or time.time
        self._last_refresh_ts = 0.0
        self._small_model = self._load_small_model()
        self._last_model_train_ts = self._small_model.trained_at_ts if self._small_model else 0.0
        # (timestamp, lift_pct, completed_count) 缓存，避免每只股票都查一次汇总
        self._lift_guard_cache: Optional[tuple] = None

    @property
    def enabled(self) -> bool:
        return isinstance(self.db, DatabaseManager) and bool(getattr(self.config, "backtest_enabled", True))

    @property
    def refresh_interval_seconds(self) -> int:
        minutes = int(getattr(self.config, "analysis_learning_refresh_interval_minutes", _DEFAULT_REFRESH_INTERVAL_MINUTES))
        return max(0, minutes) * 60

    @property
    def min_samples(self) -> int:
        return max(3, int(getattr(self.config, "analysis_learning_min_samples", _DEFAULT_MIN_SAMPLES)))

    @property
    def stock_scope_min_samples(self) -> int:
        return max(12, int(math.ceil(self.min_samples * 0.8)))

    @property
    def model_enabled(self) -> bool:
        return self.enabled and bool(getattr(self.config, "analysis_learning_model_enabled", True))

    @property
    def model_path(self) -> str:
        return str(
            getattr(
                self.config,
                "analysis_learning_model_path",
                "./data/models/analysis_calibration_model.json",
            )
        )

    @property
    def model_backend(self) -> str:
        backend = str(
            getattr(
                self.config,
                "analysis_learning_model_backend",
                "tree",
            )
        ).strip().lower()
        if backend in {"tree", "auto", "hist_gradient_boosting"}:
            return "tree"
        if backend in {"logistic", "logistic_regression", "lr"}:
            return "logistic_regression"
        if backend in {"naive_bayes", "nb"}:
            return "naive_bayes"
        return "tree"

    @property
    def model_market_split(self) -> bool:
        return bool(getattr(self.config, "analysis_learning_model_market_split", True))

    @property
    def model_scope_min_samples(self) -> int:
        return max(
            12,
            int(getattr(self.config, "analysis_learning_model_scope_min_samples", 18)),
        )

    @property
    def model_retrain_interval_seconds(self) -> int:
        minutes = int(
            getattr(
                self.config,
                "analysis_learning_model_retrain_interval_minutes",
                180,
            )
        )
        return max(0, minutes) * 60

    @property
    def model_train_min_samples(self) -> int:
        return max(
            12,
            int(getattr(self.config, "analysis_learning_model_train_min_samples", 60)),
        )

    @property
    def model_confidence_threshold(self) -> float:
        return max(
            0.5,
            min(
                0.95,
                float(
                    getattr(
                        self.config,
                        "analysis_learning_model_confidence_threshold",
                        0.62,
                    )
                ),
            ),
        )

    @property
    def learning_label_band_pct(self) -> float:
        return max(
            0.5,
            float(
                getattr(
                    self.config,
                    "analysis_learning_label_band_pct",
                    getattr(self.config, "backtest_neutral_band_pct", 2.0),
                )
            ),
        )

    def _load_small_model(self) -> Optional[SmallCalibrationModel]:
        if not self.model_enabled:
            return None
        return SmallCalibrationModel.load(self.model_path)

    def maybe_refresh_backtests(self) -> Dict[str, Any]:
        """Run incremental backtest refresh if the cooldown window has elapsed."""
        if not self.enabled:
            return {"success": False, "reason": "disabled"}

        now = self._time_fn()
        if self._last_refresh_ts and now - self._last_refresh_ts < self.refresh_interval_seconds:
            return {"success": True, "reason": "cooldown"}

        try:
            stats = BacktestService(self.db).run_backtest(
                code=None,
                force=False,
                limit=int(getattr(self.config, "analysis_learning_auto_backtest_limit", _DEFAULT_AUTO_BACKTEST_LIMIT)),
            )
            self._last_refresh_ts = now
            logger.info(
                "[LearningLoop] 增量回测刷新完成: processed=%s saved=%s completed=%s insufficient=%s errors=%s",
                stats.get("processed", 0),
                stats.get("saved", 0),
                stats.get("completed", 0),
                stats.get("insufficient", 0),
                stats.get("errors", 0),
            )
            return {"success": True, **stats}
        except Exception as exc:
            logger.warning("[LearningLoop] 增量回测刷新失败: %s", exc)
            return {"success": False, "reason": str(exc)}

    def maybe_refresh_model(self) -> Dict[str, Any]:
        """Retrain the lightweight calibration model when enough history exists."""
        if not self.model_enabled:
            return {"success": False, "reason": "disabled"}

        now = self._time_fn()
        if self._last_model_train_ts and now - self._last_model_train_ts < self.model_retrain_interval_seconds:
            return {"success": True, "reason": "cooldown"}

        rows = self._load_learning_rows()
        samples = self._build_training_samples(rows)
        if len(samples) < self.model_train_min_samples:
            return {
                "success": False,
                "reason": f"insufficient_samples:{len(samples)}",
                "sample_count": len(samples),
            }

        model = SmallCalibrationModel.fit(
            samples,
            preferred_backend=self.model_backend,
            market_split=self.model_market_split,
            min_scope_samples=self.model_scope_min_samples,
        )
        if model is None:
            return {
                "success": False,
                "reason": "fit_failed",
                "sample_count": len(samples),
            }

        try:
            model.save(self.model_path)
        except Exception as exc:
            logger.warning("[LearningLoop] 保存小型校准模型失败: %s", exc)
            return {"success": False, "reason": str(exc), "sample_count": len(samples)}

        self._small_model = model
        self._last_model_train_ts = now
        trained_scopes = sorted(model.submodels.keys())
        logger.info(
            "[LearningLoop] 学习校准模型训练完成: samples=%s backend=%s scopes=%s validation_accuracy=%s baseline_accuracy=%s path=%s",
            model.sample_count,
            model.preferred_backend,
            ",".join(trained_scopes),
            f"{model.validation_accuracy:.3f}" if model.validation_accuracy is not None else "n/a",
            f"{model.baseline_accuracy:.3f}" if model.baseline_accuracy is not None else "n/a",
            self.model_path,
        )
        return {
            "success": True,
            "sample_count": model.sample_count,
            "validation_accuracy": model.validation_accuracy,
            "validation_metrics": model.validation_metrics,
            "label_distribution": model.label_distribution,
            "baseline_accuracy": model.baseline_accuracy,
            "backend": model.preferred_backend,
            "scopes": trained_scopes,
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

    def _calibration_is_helping(self) -> bool:
        """Return False when backtests show calibration is not beating the raw signal.

        The guard is intentionally conservative: no summary / too few samples /
        query failure all keep calibration enabled (fail-open), because the
        first weeks have no attributable data yet.
        """
        ttl_seconds = 600
        now = self._time_fn()
        cached = self._lift_guard_cache
        if cached and now - cached[0] < ttl_seconds:
            return bool(cached[1])

        helping = True
        try:
            summary = BacktestService(self.db).get_summary(scope="overall", code=None)
        except Exception as exc:
            logger.debug("[LearningLoop] 读取校准归因失败，保持启用: %s", exc)
            summary = None

        if summary:
            lift = summary.get("calibration_lift_pct")
            paired = int(summary.get("calibration_sample_count") or 0)
            if lift is not None and paired >= self.min_samples:
                helping = float(lift) > 0
                if not helping:
                    logger.info(
                        "[LearningLoop] 校准未跑赢原始判断（lift=%s, n=%s），暂停应用校准（继续记录影子建议）",
                        lift,
                        paired,
                    )

        self._lift_guard_cache = (now, helping)
        return helping

    def calibrate_result(self, result: Any, context_snapshot: Optional[Dict[str, Any]] = None) -> Any:
        """Calibrate one AnalysisResult in-place and return it."""
        if not self.enabled or result is None or not getattr(result, "success", True):
            return result

        decision_type = str(getattr(result, "decision_type", "") or "").strip().lower()
        if decision_type not in {"buy", "hold", "sell"}:
            decision_type = _signal_from_operation_advice(getattr(result, "operation_advice", ""))

        # 即使暂停，也继续计算"校准器会怎么改"并写入 calibration_info：
        # 回测会把这条建议当作影子对照，等它重新跑赢原始判断后可以自动恢复。
        helping = self._calibration_is_helping()

        profile = self._build_profile(
            stock_code=str(getattr(result, "code", "") or "").strip(),
            current_signal=decision_type,
            model_used=str(getattr(result, "model_used", "") or "").strip() or None,
            score=int(getattr(result, "sentiment_score", 50) or 50),
            trend_prediction=str(getattr(result, "trend_prediction", "") or "").strip(),
            context_snapshot=context_snapshot,
            strategy_id=getattr(result, "strategy_id", None),
        )
        # 记录校准前的分数/趋势/建议，供学习与双轨回测区分原始判断
        profile.original_score = int(getattr(result, "sentiment_score", 50) or 50)
        profile.original_trend = str(getattr(result, "trend_prediction", "") or "").strip()
        profile.original_advice = str(getattr(result, "operation_advice", "") or "").strip()

        if not helping:
            profile.paused = True
            profile.applied = False
            profile.reason = (
                f"{profile.reason} 校准整体未跑赢原始判断，已暂停（仅记录影子建议）。"
            ).strip()
            result.calibration_info = profile.to_dict()
            return result

        result.calibration_info = profile.to_dict()

        if not profile.applied or not profile.suggested_signal:
            return result

        suggested_signal = profile.suggested_signal
        original_signal = decision_type
        hold_bias = "neutral"
        if suggested_signal == "hold":
            preferred_hold_signal = original_signal
            if preferred_hold_signal not in {"buy", "sell"} and profile.best_alternative_stats:
                alt_signal = str(profile.best_alternative_stats.signal or "").strip().lower()
                if alt_signal in {"buy", "sell"}:
                    preferred_hold_signal = alt_signal
            if preferred_hold_signal == "buy":
                hold_bias = "bullish"
            elif preferred_hold_signal == "sell":
                hold_bias = "bearish"
        updated_score = int(getattr(result, "sentiment_score", 50) or 50) + int(profile.score_adjustment or 0)
        updated_score = max(0, min(100, updated_score))
        if suggested_signal == "buy" and updated_score < 60:
            updated_score = 60
        elif suggested_signal == "sell" and updated_score > 39:
            updated_score = 39
        elif suggested_signal == "hold" and not 40 <= updated_score <= 59:
            updated_score = {
                "bullish": 56,
                "neutral": 50,
                "bearish": 44,
            }.get(hold_bias, 50)

        if suggested_signal == "buy":
            result.operation_advice = "买入"
            result.trend_prediction = "看多"
        elif suggested_signal == "sell":
            result.operation_advice = "减仓/卖出"
            result.trend_prediction = "看空"
        else:
            from src.analyzer import (
                _canonical_operation_advice_with_bias,
                _canonical_trend_prediction_with_bias,
            )

            result.operation_advice = _canonical_operation_advice_with_bias("hold", hold_bias)
            result.trend_prediction = _canonical_trend_prediction_with_bias("hold", hold_bias)

        result.decision_type = suggested_signal
        result.sentiment_score = updated_score
        if suggested_signal != original_signal:
            result.confidence_level = "低"

        note = (
            f"回测校准：基于{profile.source_scope}近{profile.samples}笔样本，"
            f"{original_signal}历史准确率约{profile.current_signal_stats.accuracy * 100:.0f}%"
            f"（同口径市场基准{profile.current_signal_stats.base_rate * 100:.0f}%）"
            if profile.current_signal_stats
            else f"回测校准：基于{profile.source_scope}历史样本调整本次信号。"
        )
        if profile.reason:
            note = f"{note} {profile.reason}"

        existing_summary = str(getattr(result, "analysis_summary", "") or "").strip()
        result.analysis_summary = f"{existing_summary} {note}".strip() if existing_summary else note

        existing_warning = str(getattr(result, "risk_warning", "") or "").strip()
        if suggested_signal != original_signal:
            warning_note = f"系统已根据历史回测表现，将本次信号从 {original_signal} 调整为 {suggested_signal}。"
            result.risk_warning = f"{existing_warning} {warning_note}".strip() if existing_warning else warning_note

        dashboard = getattr(result, "dashboard", None)
        if isinstance(dashboard, dict):
            dashboard["learning_calibration"] = profile.to_dict()

        from src.analyzer import normalize_analysis_result_signals

        return normalize_analysis_result_signals(result)

    def _build_profile(
        self,
        *,
        stock_code: str,
        current_signal: str,
        model_used: Optional[str],
        score: int,
        trend_prediction: str,
        context_snapshot: Optional[Dict[str, Any]] = None,
        strategy_id: Optional[str] = None,
    ) -> CalibrationProfile:
        market = get_market_for_stock(stock_code) or "cn"
        profile = CalibrationProfile(
            enabled=self.enabled,
            market=market,
            model_used=model_used,
            current_signal=current_signal,
            threshold=self.min_samples,
        )
        if not stock_code or current_signal not in {"buy", "hold", "sell"}:
            return profile

        # Strategy scope: use pre-computed strategy-level backtest summary as
        # the highest-priority calibration source when a strategy_id is known.
        # When available, strategy profile replaces heuristic as the base;
        # ML model merge + conflict resolution still applies below.
        if strategy_id:
            strategy_profile = self._build_strategy_profile(
                stock_code=stock_code,
                strategy_id=strategy_id,
                current_signal=current_signal,
                market=market,
            )
            if strategy_profile is not None and strategy_profile.applied:
                profile = strategy_profile

        if not profile.applied:
            rows = self._load_learning_rows()
            if rows:
                profile = self._build_heuristic_profile(
                    stock_code=stock_code,
                    current_signal=current_signal,
                    model_used=model_used,
                    score=score,
                    market=market,
                    rows=rows,
                )

        model_decision = self._build_model_profile(
            stock_code=stock_code,
            current_signal=current_signal,
            model_used=model_used,
            score=score,
            trend_prediction=trend_prediction,
            context_snapshot=context_snapshot,
        )
        if model_decision is None:
            return profile

        profile.model_prediction = model_decision["prediction_meta"]
        if profile.applied:
            if model_decision["applied"] and model_decision["suggested_signal"] == profile.suggested_signal:
                profile.source_scope = (
                    f"{profile.source_scope}+学习模型" if profile.source_scope else "学习模型"
                )
                profile.score_adjustment += int(model_decision["score_adjustment"] or 0)
                profile.reason = " ".join(
                    token for token in (profile.reason, model_decision["reason"]) if token
                ).strip()
            elif (
                model_decision["applied"]
                and profile.source_scope in {"全局", "策略"}
                and model_decision["confidence"] >= max(0.72, self.model_confidence_threshold + 0.08)
                and model_decision["suggested_signal"] != profile.suggested_signal
            ):
                profile.source_scope = f"{profile.source_scope}+学习模型"
                profile.suggested_signal = "hold"
                profile.score_adjustment = 0
                profile.reason = " ".join(
                    token
                    for token in (
                        profile.reason,
                        "学习校准模型与全局统计结论冲突，已保守降级为观望。",
                    )
                    if token
                ).strip()
            return profile

        if not model_decision["applied"]:
            return profile

        return CalibrationProfile(
            enabled=self.enabled,
            market=market,
            model_used=model_used,
            source_scope="学习校准模型",
            samples=int(model_decision["prediction_meta"].get("sample_count") or 0),
            threshold=self.model_train_min_samples,
            current_signal=current_signal,
            suggested_signal=model_decision["suggested_signal"],
            score_adjustment=int(model_decision["score_adjustment"] or 0),
            applied=True,
            reason=model_decision["reason"],
            model_prediction=model_decision["prediction_meta"],
        )

        return profile

    def _build_heuristic_profile(
        self,
        *,
        stock_code: str,
        current_signal: str,
        model_used: Optional[str],
        score: int,
        market: str,
        rows: List[Dict[str, Any]],
    ) -> CalibrationProfile:
        profile = CalibrationProfile(
            enabled=self.enabled,
            market=market,
            model_used=model_used,
            current_signal=current_signal,
            threshold=self.min_samples,
        )

        scope_candidates = [
            ("个股", lambda row: row["code"] == stock_code, self.stock_scope_min_samples),
            (
                "市场+模型",
                lambda row: row["market"] == market and model_used and row["model_used"] == model_used,
                self.min_samples,
            ),
            ("市场", lambda row: row["market"] == market, self.min_samples),
            ("全局", lambda row: True, self.min_samples),
        ]

        for scope_name, predicate, threshold in scope_candidates:
            scoped_rows = [row for row in rows if predicate(row)]
            decision = self._decide_adjustment(
                rows=scoped_rows,
                current_signal=current_signal,
                threshold=threshold,
                score=score,
            )
            if decision is None:
                continue

            profile.source_scope = scope_name
            profile.samples = decision["samples"]
            profile.current_signal_stats = decision["current_stats"]
            profile.best_alternative_stats = decision["best_alternative_stats"]
            profile.suggested_signal = decision["suggested_signal"]
            profile.score_adjustment = decision["score_adjustment"]
            profile.applied = bool(decision["applied"])
            profile.reason = decision["reason"]
            return profile

        return profile

    def _build_model_profile(
        self,
        *,
        stock_code: str,
        current_signal: str,
        model_used: Optional[str],
        score: int,
        trend_prediction: str,
        context_snapshot: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.model_enabled or self._small_model is None:
            return None

        # 没有验证集就不能说明模型有泛化能力，直接不应用
        if self._small_model.validation_accuracy is None:
            return None

        if (
            self._small_model.validation_accuracy is not None
            and self._small_model.validation_count >= 12
            and self._small_model.baseline_accuracy is not None
            and self._small_model.validation_accuracy < max(0.35, self._small_model.baseline_accuracy - 0.02)
        ):
            return None

        features = self._extract_model_features(
            stock_code=stock_code,
            current_signal=current_signal,
            model_used=model_used,
            score=score,
            trend_prediction=trend_prediction,
            context_snapshot=context_snapshot,
        )
        market = get_market_for_stock(stock_code) or "cn"
        prediction = self._small_model.predict(features, scope_hint=market)
        if prediction is None:
            return None

        if (
            prediction.validation_accuracy is not None
            and prediction.validation_count >= 6
            and prediction.baseline_accuracy is not None
            and prediction.validation_accuracy < max(0.35, prediction.baseline_accuracy - 0.02)
        ):
            return None

        suggested_signal = prediction.predicted_signal
        confidence = float(prediction.confidence or 0.0)
        applied = False
        score_adjustment = 0
        reason = ""

        if suggested_signal == current_signal:
            if confidence >= self.model_confidence_threshold:
                applied = True
                if suggested_signal == "buy":
                    score_adjustment = 4
                elif suggested_signal == "sell":
                    score_adjustment = -4
                reason = f"学习校准模型同向支持当前 {current_signal} 信号。"
        else:
            high_conf_threshold = max(0.72, self.model_confidence_threshold + 0.10)
            if confidence >= high_conf_threshold:
                applied = True
                if current_signal in {"buy", "sell"} and not 30 <= score <= 70:
                    suggested_signal = "hold"
                    reason = "学习校准模型与当前高强度方向冲突，先保守降级为观望。"
                else:
                    reason = f"学习校准模型认为当前场景更接近 {suggested_signal}。"
                if suggested_signal == "buy":
                    score_adjustment = 9
                elif suggested_signal == "sell":
                    score_adjustment = -9
            elif confidence >= self.model_confidence_threshold and current_signal in {"buy", "sell"}:
                applied = True
                suggested_signal = "hold"
                reason = "学习校准模型对当前方向把握不足，先降级为观望。"

        return {
            "applied": applied,
            "suggested_signal": suggested_signal,
            "score_adjustment": score_adjustment,
            "reason": reason,
            "confidence": confidence,
            "prediction_meta": prediction.to_dict(),
        }

    def _build_training_samples(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 按信号日排序：批量重跑回测后 evaluated_at 会全部相同，
        # 用它排序会让"时间切分"退化成随机切分。
        ordered_rows = sorted(
            rows,
            key=lambda item: (
                str(item.get("analysis_date") or ""),
                float(item.get("evaluated_at_ts") or 0.0),
            ),
        )

        # 横截面中位数作为"市场基准"的替代：同一信号日所有标的的收益中位数。
        # 用它做超额收益标签，可以抵掉大盘涨跌，避免牛市里几乎全是 buy。
        returns_by_date: Dict[str, List[float]] = {}
        for row in ordered_rows:
            day = str(row.get("analysis_date") or "")
            ret = row.get("stock_return_pct")
            if day and ret is not None:
                returns_by_date.setdefault(day, []).append(float(ret))
        median_by_date = {
            day: _median(values)
            for day, values in returns_by_date.items()
            if len(values) >= 3
        }

        samples: List[Dict[str, Any]] = []
        for row in ordered_rows:
            label = self._derive_learning_label(row, median_by_date=median_by_date)
            if label not in {"buy", "hold", "sell"}:
                continue
            features = self._extract_model_features(
                stock_code=str(row.get("code") or ""),
                # 训练特征必须用**校准前**的原始判断：推理时模型拿到的也是刚解析出的
                # 原始信号。用校准后的信号训练会让训练/推理分布不一致。
                current_signal=str(row.get("raw_signal") or row.get("signal") or "hold"),
                model_used=row.get("model_used"),
                score=int(row.get("raw_score") if row.get("raw_score") is not None else row.get("sentiment_score") or 50),
                trend_prediction=str(row.get("raw_trend") or row.get("trend_prediction") or ""),
                context_snapshot=row.get("context_payload"),
            )
            samples.append(
                {
                    "features": features,
                    "label": label,
                    "market": str(
                        row.get("market")
                        or get_market_for_stock(str(row.get("code") or ""))
                        or "cn"
                    ).strip().lower(),
                    "analysis_date": str(row.get("analysis_date") or ""),
                    "window_days": int(row.get("window_days") or 0),
                }
            )
        return samples

    def _derive_learning_label(
        self,
        row: Dict[str, Any],
        *,
        median_by_date: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """标签用**超额收益**（个股收益 − 同日横截面中位数）。

        直接用绝对收益会在趋势市里把绝大多数样本判成 buy，
        基线准确率被抬高、模型学不到任何东西。
        """
        stock_return_pct = row.get("stock_return_pct")
        if stock_return_pct is not None:
            excess = float(stock_return_pct)
            if median_by_date:
                day = str(row.get("analysis_date") or "")
                benchmark = median_by_date.get(day)
                if benchmark is not None:
                    excess -= float(benchmark)
            if excess >= self.learning_label_band_pct:
                return "buy"
            if excess <= -self.learning_label_band_pct:
                return "sell"
            return "hold"

        if row.get("direction_correct") is True:
            return str(row.get("signal") or "hold")
        return None

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clip(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def _unwrap_context_snapshot(context_snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(context_snapshot, dict):
            return {}
        enhanced = context_snapshot.get("enhanced_context")
        if isinstance(enhanced, dict):
            return enhanced
        return context_snapshot

    @staticmethod
    def _provider_bucket(model_used: Optional[str]) -> str:
        provider = str(model_used or "").split("/", 1)[0].strip().lower()
        if provider in {"gemini", "vertex_ai"}:
            return "gemini"
        if provider in {"anthropic", "claude"}:
            return "anthropic"
        if provider in {"openai", "deepseek"}:
            return "openai"
        return "other"

    @staticmethod
    def _trend_bucket(trend_prediction: str) -> str:
        text = str(trend_prediction or "").strip()
        if any(token in text for token in ("看多", "偏多", "bull")):
            return "bull"
        if any(token in text for token in ("看空", "偏空", "bear")):
            return "bear"
        return "neutral"

    @staticmethod
    def _china_exposure_score(level: Any) -> float:
        normalized = str(level or "").strip().lower()
        mapping = {
            "high": 1.0,
            "中": 0.66,
            "medium": 0.66,
            "中等": 0.66,
            "med": 0.66,
            "low": 0.33,
            "低": 0.33,
        }
        return mapping.get(normalized, 0.0)

    def _extract_model_features(
        self,
        *,
        stock_code: str,
        current_signal: str,
        model_used: Optional[str],
        score: int,
        trend_prediction: str,
        context_snapshot: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        enhanced = self._unwrap_context_snapshot(context_snapshot)
        market_context = enhanced.get("market_context") or {}
        realtime = enhanced.get("realtime") or enhanced.get("realtime_quote") or {}
        trend = enhanced.get("trend_analysis") or enhanced.get("trend_result") or {}
        chip = enhanced.get("chip") or enhanced.get("chip_distribution") or {}
        today = enhanced.get("today") if isinstance(enhanced.get("today"), dict) else {}
        yesterday = enhanced.get("yesterday") if isinstance(enhanced.get("yesterday"), dict) else {}
        ma_status = enhanced.get("ma_status") if isinstance(enhanced.get("ma_status"), dict) else {}

        market = str(
            market_context.get("market")
            or enhanced.get("market")
            or get_market_for_stock(stock_code)
            or "cn"
        ).strip().lower()
        signal = current_signal if current_signal in {"buy", "hold", "sell"} else "hold"
        signal_bias = {"buy": 1.0, "hold": 0.0, "sell": -1.0}.get(signal, 0.0)
        provider_bucket = self._provider_bucket(model_used)
        trend_bucket = self._trend_bucket(trend_prediction)
        china_exposure = (
            market_context.get("china_exposure", {})
            if isinstance(market_context.get("china_exposure"), dict)
            else {}
        )

        change_pct = realtime.get("change_pct")
        if change_pct is None:
            change_pct = today.get("pct_chg") or enhanced.get("price_change_ratio")
        volume_ratio = realtime.get("volume_ratio")
        if volume_ratio is None:
            volume_ratio = enhanced.get("volume_change_ratio", 1.0)

        price = self._to_float(realtime.get("price") or today.get("close"))
        open_price = self._to_float(today.get("open"))
        high_price = self._to_float(today.get("high"))
        low_price = self._to_float(today.get("low"))
        prev_close = self._to_float(yesterday.get("close"))
        ma5 = self._to_float(today.get("ma5"))
        ma10 = self._to_float(today.get("ma10"))
        ma20 = self._to_float(today.get("ma20"))
        avg_cost = self._to_float(chip.get("avg_cost"))
        signal_reasons = trend.get("signal_reasons") if isinstance(trend.get("signal_reasons"), list) else []
        risk_factors = trend.get("risk_factors") if isinstance(trend.get("risk_factors"), list) else []
        volume_status = str(trend.get("volume_status") or "").lower()
        buy_signal = str(trend.get("buy_signal") or "").lower()
        trend_status = str(trend.get("trend_status") or "").lower()

        intraday_range_pct = ((high_price - low_price) / price * 100.0) if price > 0 and high_price > low_price else 0.0
        open_gap_pct = ((open_price - prev_close) / prev_close * 100.0) if prev_close > 0 and open_price > 0 else 0.0
        ma5_distance_pct = ((price - ma5) / ma5 * 100.0) if price > 0 and ma5 > 0 else 0.0
        ma10_distance_pct = ((price - ma10) / ma10 * 100.0) if price > 0 and ma10 > 0 else 0.0
        ma20_distance_pct = ((price - ma20) / ma20 * 100.0) if price > 0 and ma20 > 0 else 0.0
        chip_cost_distance_pct = ((price - avg_cost) / avg_cost * 100.0) if price > 0 and avg_cost > 0 else 0.0

        features = {
            "score_norm": self._clip(self._to_float(score) / 100.0, 0.0, 1.0),
            "current_signal_bias": signal_bias,
            "trend_bull": 1.0 if trend_bucket == "bull" else 0.0,
            "trend_neutral": 1.0 if trend_bucket == "neutral" else 0.0,
            "trend_bear": 1.0 if trend_bucket == "bear" else 0.0,
            "trend_status_up": 1.0 if any(token in trend_status for token in ("up", "多", "上升")) else 0.0,
            "trend_status_down": 1.0 if any(token in trend_status for token in ("down", "空", "下降")) else 0.0,
            "buy_signal_strong": 1.0 if any(token in buy_signal for token in ("strong", "强")) else 0.0,
            "buy_signal_weak": 1.0 if any(token in buy_signal for token in ("weak", "弱")) else 0.0,
            "volume_status_high": 1.0 if any(token in volume_status for token in ("high", "放量", "active")) else 0.0,
            "volume_status_low": 1.0 if any(token in volume_status for token in ("low", "缩量", "quiet")) else 0.0,
            "market_cn": 1.0 if market == "cn" else 0.0,
            "market_hk": 1.0 if market == "hk" else 0.0,
            "market_us": 1.0 if market == "us" else 0.0,
            "provider_gemini": 1.0 if provider_bucket == "gemini" else 0.0,
            "provider_anthropic": 1.0 if provider_bucket == "anthropic" else 0.0,
            "provider_openai": 1.0 if provider_bucket == "openai" else 0.0,
            "provider_other": 1.0 if provider_bucket == "other" else 0.0,
            "change_pct_norm": self._clip(self._to_float(change_pct) / 10.0, -3.0, 3.0),
            "volume_ratio_norm": self._clip(self._to_float(volume_ratio, 1.0) / 3.0, 0.0, 3.0),
            "turnover_rate_norm": self._clip(self._to_float(realtime.get("turnover_rate")) / 30.0, 0.0, 3.0),
            "pe_ratio_norm": self._clip(self._to_float(realtime.get("pe_ratio")) / 80.0, 0.0, 3.0),
            "pb_ratio_norm": self._clip(self._to_float(realtime.get("pb_ratio")) / 10.0, 0.0, 3.0),
            "market_cap_norm": self._clip(self._to_float(realtime.get("total_mv")) / 1000000000000.0, 0.0, 5.0),
            "circ_market_cap_norm": self._clip(self._to_float(realtime.get("circ_mv")) / 1000000000000.0, 0.0, 5.0),
            "change_60d_norm": self._clip(self._to_float(realtime.get("change_60d")) / 30.0, -3.0, 3.0),
            "intraday_range_norm": self._clip(intraday_range_pct / 10.0, 0.0, 3.0),
            "open_gap_norm": self._clip(open_gap_pct / 10.0, -3.0, 3.0),
            "ma5_distance_norm": self._clip(ma5_distance_pct / 10.0, -3.0, 3.0),
            "ma10_distance_norm": self._clip(ma10_distance_pct / 10.0, -3.0, 3.0),
            "ma20_distance_norm": self._clip(ma20_distance_pct / 10.0, -3.0, 3.0),
            "ma_bullish_alignment": 1.0 if ma_status.get("bullish_alignment") else 0.0,
            "price_above_ma5": 1.0 if price > 0 and ma5 > 0 and price >= ma5 else 0.0,
            "price_above_ma20": 1.0 if price > 0 and ma20 > 0 and price >= ma20 else 0.0,
            "trend_strength_norm": self._clip(self._to_float(trend.get("trend_strength")) / 100.0, 0.0, 1.0),
            "technical_signal_score_norm": self._clip(self._to_float(trend.get("signal_score")) / 100.0, 0.0, 1.0),
            "bias_ma5_norm": self._clip(self._to_float(trend.get("bias_ma5")) / 10.0, -3.0, 3.0),
            "bias_ma10_norm": self._clip(self._to_float(trend.get("bias_ma10")) / 10.0, -3.0, 3.0),
            "signal_reason_count_norm": self._clip(len(signal_reasons) / 6.0, 0.0, 3.0),
            "risk_factor_count_norm": self._clip(len(risk_factors) / 6.0, 0.0, 3.0),
            "chip_profit_ratio_norm": self._clip(self._to_float(chip.get("profit_ratio")) / 100.0, 0.0, 1.0),
            "chip_concentration90_norm": self._clip(self._to_float(chip.get("concentration_90")) / 100.0, 0.0, 1.0),
            "chip_concentration70_norm": self._clip(self._to_float(chip.get("concentration_70")) / 100.0, 0.0, 1.0),
            "chip_cost_distance_norm": self._clip(chip_cost_distance_pct / 20.0, -3.0, 3.0),
            "china_exposure_norm": self._china_exposure_score(china_exposure.get("level")),
            "news_window_norm": self._clip(self._to_float(enhanced.get("news_window_days"), 3.0) / 7.0, 0.0, 2.0),
            "is_index_etf": 1.0 if enhanced.get("is_index_etf") else 0.0,
            "data_missing": 1.0 if enhanced.get("data_missing") else 0.0,
            "regime_above_ma200": 1.0 if (enhanced.get("market_regime") or {}).get("above_ma200") else 0.0,
            "regime_volatility_norm": self._clip(
                self._to_float((enhanced.get("market_regime") or {}).get("volatility_20d_pct")) / 5.0,
                0.0,
                3.0,
            ),
        }
        return features

    @staticmethod
    def _skill_of(stats: SignalStats) -> float:
        """信号的方向增量：准确率 − 该方向的自然发生率。

        基准率退化（样本里几乎全是同一种结果）时退回 50% 中性先验：
        此时没有横截面差异，扣基准只会把整段趋势当成信号能力。
        """
        if not 0.02 < stats.base_rate < 0.98:
            return stats.accuracy - 0.5
        return stats.skill

    def _recent_slice_confirms(
        self,
        rows: List[Dict[str, Any]],
        *,
        signal: str,
        direction: str,
    ) -> bool:
        """最近一段样本是否仍支持同一结论（样本外检查）。

        只看全量样本的校准器，会在风格/政策切换后继续用旧行情下的结论。
        这里用最近约 40% 的样本复算一次：要求方向优势在近期同样成立。
        日期不足 3 个不同交易日的场景无法切分，按旧行为放行。
        """
        dated = [row for row in rows if row.get("analysis_date")]
        if len({str(row.get("analysis_date")) for row in dated}) < 3:
            return True

        dated.sort(key=lambda item: str(item.get("analysis_date")))
        recent_size = max(3, int(len(dated) * 0.4))
        recent = dated[-recent_size:]
        stats = self._compute_signal_stats(
            recent,
            signal,
            self._direction_base_rates(recent, self.learning_label_band_pct),
        )
        if stats.samples < 3:
            return False

        skill = self._skill_of(stats)
        if direction == "positive":
            return skill >= 0.10
        return skill <= -0.05

    def _decide_adjustment(
        self,
        *,
        rows: List[Dict[str, Any]],
        current_signal: str,
        threshold: int,
        score: int,
    ) -> Optional[Dict[str, Any]]:
        if not rows:
            return None

        # 先算同一批样本里各方向的自然发生率，再比较"超出基准"的部分。
        base_rates = self._direction_base_rates(rows, self.learning_label_band_pct)
        current_stats = self._compute_signal_stats(rows, current_signal, base_rates)
        alt_candidates = [
            self._compute_signal_stats(rows, signal, base_rates)
            for signal in ("buy", "hold", "sell")
            if signal != current_signal
        ]
        alt_candidates = [stats for stats in alt_candidates if stats.samples > 0]
        best_alternative = max(
            alt_candidates,
            key=lambda item: (self._skill_of(item), item.samples),
            default=None,
        )
        current_skill = self._skill_of(current_stats)
        best_skill = self._skill_of(best_alternative) if best_alternative else 0.0

        if current_signal == "hold":
            promotion_threshold = max(threshold + 2, 14)
            if best_alternative is None or best_alternative.samples < promotion_threshold:
                return None
            if best_skill >= 0.25 and 46 <= score <= 58 and self._recent_slice_confirms(
                rows, signal=best_alternative.signal, direction="positive"
            ):
                score_adjustment = 10 if best_alternative.signal == "buy" else -10
                return {
                    "samples": best_alternative.samples,
                    "current_stats": current_stats,
                    "best_alternative_stats": best_alternative,
                    "suggested_signal": best_alternative.signal,
                    "score_adjustment": score_adjustment,
                    "applied": True,
                    "reason": (
                        f"该场景历史上扣除市场基准后，{best_alternative.signal} "
                        f"仍有 {best_skill * 100:.0f} 个百分点的方向优势。"
                    ),
                }
            return None

        if current_stats.samples < threshold:
            return None

        suggested_signal = current_signal
        score_adjustment = 0
        applied = False
        reason = ""

        if (
            current_skill <= -0.15
            and best_alternative
            and best_alternative.samples >= threshold
            and best_skill >= 0.15
            and self._recent_slice_confirms(rows, signal=current_signal, direction="negative")
            and self._recent_slice_confirms(rows, signal=best_alternative.signal, direction="positive")
        ):
            if 35 <= score <= 70:
                suggested_signal = best_alternative.signal if best_skill >= 0.25 else "hold"
                applied = True
                score_adjustment = -12 if current_signal == "buy" else 12
                reason = (
                    f"{current_signal} 在扣除市场基准后仍低于自然发生率，"
                    f"而 {best_alternative.signal} 显著更优。"
                )
        elif current_skill <= -0.10 and self._recent_slice_confirms(
            rows, signal=current_signal, direction="negative"
        ):
            suggested_signal = "hold"
            applied = True
            score_adjustment = -8 if current_signal == "buy" else 8
            reason = f"{current_signal} 扣除市场基准后没有方向优势，先降级为更保守的观望。"
        elif current_skill >= 0.20 and self._recent_slice_confirms(
            rows, signal=current_signal, direction="positive"
        ):
            applied = True
            if current_signal == "buy":
                score_adjustment = 6
            elif current_signal == "sell":
                score_adjustment = -6
            reason = (
                f"{current_signal} 扣除市场基准后仍有 "
                f"{current_skill * 100:.0f} 个百分点的方向优势，适度强化当前结论。"
            )

        if not applied:
            return {
                "samples": current_stats.samples,
                "current_stats": current_stats,
                "best_alternative_stats": best_alternative,
                "suggested_signal": current_signal,
                "score_adjustment": 0,
                "applied": False,
                "reason": "",
            }

        return {
            "samples": current_stats.samples,
            "current_stats": current_stats,
            "best_alternative_stats": best_alternative,
            "suggested_signal": suggested_signal,
            "score_adjustment": score_adjustment,
            "applied": True,
            "reason": reason,
        }


    def _build_strategy_profile(
        self,
        *,
        stock_code: str,
        strategy_id: str,
        current_signal: str,
        market: str,
    ) -> Optional[CalibrationProfile]:
        """Build a CalibrationProfile from pre-computed strategy-level backtest summary.

        Uses ``BacktestService.get_strategy_summary()`` to fetch aggregated
        strategy performance metrics (win_rate, direction_accuracy,
        avg_return) and derives a signal adjustment decision from them.

        Returns ``None`` when no strategy summary exists or the data is
        insufficient — the caller falls back to heuristic/ML calibration.
        """
        try:
            summary = BacktestService(self.db).get_strategy_summary(strategy_id)
        except Exception as exc:
            logger.warning("[LearningLoop] 获取策略回测画像失败(strategy_id=%s): %s", strategy_id, exc)
            return None

        if summary is None:
            return None

        total_evals = int(summary.get("total_evaluations") or 0)
        if total_evals < self.min_samples:
            return None
        is_fallback = bool(summary.get("is_fallback"))
        if is_fallback:
            return None

        win_rate = _safe_ratio(summary.get("win_rate"))
        direction_accuracy = _safe_ratio(summary.get("direction_accuracy"))
        avg_return = _safe_ratio(summary.get("avg_return"), default=0.0)

        # Strategy summary is aggregated (not per-signal), so we build
        # synthetic SignalStats anchored around the current signal.
        current_stats = SignalStats(
            signal=current_signal,
            samples=total_evals,
            accuracy=direction_accuracy,
            avg_return=avg_return,
        )

        applied = False
        suggested_signal = current_signal
        score_adjustment = 0
        reason = ""

        # Conservative strategy-level calibration rules
        if direction_accuracy <= 0.30:
            suggested_signal = "hold"
            applied = True
            score_adjustment = -10 if current_signal == "buy" else 10 if current_signal == "sell" else 0
            reason = "该策略历史方向准确率偏低，降级为观望。"
        elif direction_accuracy <= 0.45:
            if current_signal in {"buy", "sell"}:
                suggested_signal = "hold"
                applied = True
                score_adjustment = -6 if current_signal == "buy" else 6
                reason = "该策略方向准确率较弱，先保守降级为观望。"
        elif direction_accuracy >= 0.65 and win_rate >= 0.55:
            applied = True
            if current_signal == "buy":
                score_adjustment = 5
            elif current_signal == "sell":
                score_adjustment = -5
            reason = f"该策略历史方向准确率较高（{direction_accuracy:.0%}），适度强化当前结论。"

        profile = CalibrationProfile(
            enabled=self.enabled,
            market=market,
            model_used=None,
            source_scope="策略",
            samples=total_evals,
            threshold=self.min_samples,
            current_signal=current_signal,
            current_signal_stats=current_stats,
            best_alternative_stats=None,
            suggested_signal=suggested_signal,
            score_adjustment=score_adjustment,
            applied=applied,
            reason=reason,
            strategy_id=strategy_id,
            source_is_fallback=False,
        )
        return profile


    @staticmethod
    def _direction_base_rates(rows: Iterable[Dict[str, Any]], band_pct: float) -> Dict[str, float]:
        """同一批样本里各"方向口径"的自然发生率。

        比较 buy / hold / sell 的 accuracy 时，分子分母根本不是同一件事：
        buy 正确 = 上涨，观望正确 = 非上涨，持有正确 = 非下跌。牛市里 buy
        天然接近 100%，熊市里观望天然接近 100%。不减去基准率，校准器只是在追涨杀跌。
        """
        returns = [
            float(row.get("stock_return_pct"))
            for row in rows
            if row.get("stock_return_pct") is not None
        ]
        if not returns:
            return {"up": 0.5, "down": 0.5, "not_up": 0.5, "not_down": 0.5}
        band = abs(float(band_pct))
        total = len(returns)
        p_up = sum(1 for value in returns if value > band) / total
        p_down = sum(1 for value in returns if value < -band) / total
        return {
            "up": p_up,
            "down": p_down,
            "not_up": 1.0 - p_up,
            "not_down": 1.0 - p_down,
        }

    @staticmethod
    def _base_rate_for_row(row: Dict[str, Any], base_rates: Dict[str, float]) -> float:
        expected = str(row.get("direction_expected") or "").strip().lower()
        if expected not in base_rates:
            # 老数据 / 测试数据没有 direction_expected，按信号类型回退
            expected = {"buy": "up", "sell": "down"}.get(
                str(row.get("signal") or "").strip().lower(), "not_up"
            )
        return float(base_rates.get(expected, 0.5))

    @staticmethod
    def _compute_signal_stats(
        rows: Iterable[Dict[str, Any]],
        signal: str,
        base_rates: Optional[Dict[str, float]] = None,
    ) -> SignalStats:
        matched = [row for row in rows if row["signal"] == signal]
        if not matched:
            return SignalStats(signal=signal)

        accuracy = sum(1 for row in matched if row.get("direction_correct") is True) / len(matched)
        returns = [float(row.get("simulated_return_pct") or 0.0) for row in matched]
        avg_return = sum(returns) / len(returns) if returns else 0.0
        if base_rates:
            base_rate = sum(
                AnalysisCalibrationService._base_rate_for_row(row, base_rates)
                for row in matched
            ) / len(matched)
        else:
            base_rate = 0.5
        return SignalStats(
            signal=signal,
            samples=len(matched),
            accuracy=accuracy,
            avg_return=avg_return,
            base_rate=base_rate,
        )

    def _learning_engine_version(self) -> str:
        """返回当前生效的回测引擎版本，用于过滤学习样本。

        必须与 ``BacktestService`` 写入时的版本号算法一致，否则过滤条件会
        永远匹配不到任何行（例如漏掉 entry_mode 后缀）。
        """
        raw_version = getattr(self.config, "backtest_engine_version", None)
        base_version = raw_version.strip() if isinstance(raw_version, str) and raw_version.strip() else "v1"
        raw_mode = getattr(self.config, "backtest_entry_mode", None)
        entry_mode = raw_mode if isinstance(raw_mode, str) else "next_open"
        return BacktestService._effective_engine_version(
            base_version,
            BacktestEngine._normalize_entry_mode(entry_mode),
        )

    def _load_learning_rows(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        history_limit = int(getattr(self.config, "analysis_learning_history_limit", _DEFAULT_HISTORY_LIMIT))
        engine_version = self._learning_engine_version()
        with self.db.get_session() as session:
            pairs = session.execute(
                select(BacktestResult, AnalysisHistory)
                .join(AnalysisHistory, AnalysisHistory.id == BacktestResult.analysis_history_id)
                .where(
                    BacktestResult.eval_status == "completed",
                    # 只吃当前引擎版本的样本：旧版本用的是修复前的口径
                    # （long 结果按期末涨跌、现金收益记 0、缺口不成交），
                    # 和新口径混在一起训练会学到互相矛盾的映射。
                    BacktestResult.engine_version == engine_version,
                )
                .order_by(desc(BacktestResult.evaluated_at))
                .limit(history_limit)
            ).all()

        rows: List[Dict[str, Any]] = []
        skipped_placeholder = 0
        skipped_degraded = 0
        skipped_duplicate = 0
        seen_signal_days: set = set()
        for backtest_row, history_row in pairs:
            payload = self._parse_payload(getattr(history_row, "raw_result", None))
            context_payload = self._parse_payload(getattr(history_row, "context_snapshot", None))
            signal = self._extract_signal(payload, history_row)
            if signal not in {"buy", "hold", "sell"}:
                continue

            # 只吃真实 LLM 产出的记录。LLM 不可用时 analyze() 会落一条
            # 「持有/震荡/50」的兜底结果，这类样本没有任何信息量，
            # 混进训练集会让模型学到占位值到涨跌的伪映射。
            model_used = self._extract_model_used(payload)
            if not model_used:
                skipped_placeholder += 1
                continue

            if payload.get("numeric_consistency") is False:
                skipped_degraded += 1
                continue

            # 纯文本降级 / 输出被截断的结果不是完整判断
            if str(payload.get("parse_mode") or "structured").strip().lower() in {
                "text_fallback",
                "truncated",
            }:
                skipped_degraded += 1
                continue

            # 同一天同一只股票重复分析多次，收益完全相同，不是独立样本。
            # pairs 按 evaluated_at 倒序，保留最新一条。
            signal_day = (
                backtest_row.analysis_date.isoformat()
                if getattr(backtest_row, "analysis_date", None)
                else ""
            )
            dedupe_key = (history_row.code, signal_day)
            if signal_day and dedupe_key in seen_signal_days:
                skipped_duplicate += 1
                continue
            if signal_day:
                seen_signal_days.add(dedupe_key)

            rows.append(
                {
                    "code": history_row.code,
                    "market": self._extract_market(history_row.code, payload, context_payload),
                    "model_used": model_used,
                    "signal": signal,
                    "sentiment_score": payload.get("sentiment_score", getattr(history_row, "sentiment_score", 50)),
                    "trend_prediction": payload.get("trend_prediction", getattr(history_row, "trend_prediction", "")),
                    # 校准前的原始判断：学习模型的特征必须和推理时（刚解析出的原始信号）
                    # 保持同一分布，否则等于让模型学习"校准器校准过的输入"。
                    "raw_signal": self._extract_raw_signal(payload, signal),
                    "raw_score": self._extract_raw_score(
                        payload, payload.get("sentiment_score", getattr(history_row, "sentiment_score", 50))
                    ),
                    "raw_trend": self._extract_raw_trend(
                        payload, payload.get("trend_prediction", getattr(history_row, "trend_prediction", ""))
                    ),
                    "direction_correct": backtest_row.direction_correct,
                    "direction_expected": backtest_row.direction_expected,
                    "simulated_return_pct": backtest_row.simulated_return_pct,
                    "stock_return_pct": backtest_row.stock_return_pct,
                    # 训练集必须按“信号日”排序：批量重跑回测后 evaluated_at
                    # 全都是同一时刻，用它排序会把时间切分变成随机切分。
                    "analysis_date": (
                        backtest_row.analysis_date.isoformat()
                        if getattr(backtest_row, "analysis_date", None)
                        else None
                    ),
                    "window_days": int(backtest_row.eval_window_days or 0),
                    "evaluated_at_ts": backtest_row.evaluated_at.timestamp() if backtest_row.evaluated_at else 0.0,
                    "context_payload": context_payload,
                }
            )

        if skipped_placeholder:
            logger.info(
                "[LearningLoop] 跳过 %s 条无 model_used 的占位样本（LLM 未参与的兜底结果）",
                skipped_placeholder,
            )
        if skipped_degraded:
            logger.info(
                "[LearningLoop] 跳过 %s 条降级样本（纯文本解析或被 max_tokens 截断）",
                skipped_degraded,
            )
        if skipped_duplicate:
            logger.info(
                "[LearningLoop] 跳过 %s 条同一信号日的重复分析样本",
                skipped_duplicate,
            )
        return rows

    @staticmethod
    def _parse_payload(raw_result: Any) -> Dict[str, Any]:
        if isinstance(raw_result, dict):
            return dict(raw_result)
        if isinstance(raw_result, str) and raw_result.strip():
            try:
                payload = json.loads(raw_result)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return {}
        return {}

    @staticmethod
    def _extract_signal(payload: Dict[str, Any], history_row: AnalysisHistory) -> str:
        decision_type = str(payload.get("decision_type", "") or "").strip().lower()
        if decision_type in {"buy", "hold", "sell"}:
            return decision_type
        return _signal_from_operation_advice(
            payload.get("operation_advice") or getattr(history_row, "operation_advice", "")
        )

    @staticmethod
    def _extract_model_used(payload: Dict[str, Any]) -> Optional[str]:
        model_used = str(payload.get("model_used", "") or "").strip()
        return model_used or None

    @staticmethod
    def _calibration_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        calibration = payload.get("calibration_info")
        return calibration if isinstance(calibration, dict) else {}

    def _extract_raw_signal(self, payload: Dict[str, Any], fallback: str) -> str:
        calibration = self._calibration_payload(payload)
        candidate = str(calibration.get("current_signal") or "").strip().lower()
        return candidate if candidate in {"buy", "hold", "sell"} else fallback

    def _extract_raw_score(self, payload: Dict[str, Any], fallback: Any) -> Any:
        calibration = self._calibration_payload(payload)
        value = calibration.get("original_score")
        return fallback if value is None else value

    def _extract_raw_trend(self, payload: Dict[str, Any], fallback: Any) -> Any:
        calibration = self._calibration_payload(payload)
        value = calibration.get("original_trend")
        return fallback if not value else value

    @staticmethod
    def _extract_market(code: str, payload: Dict[str, Any], context_payload: Dict[str, Any]) -> str:
        enhanced_payload = context_payload.get("enhanced_context", {}) if isinstance(context_payload, dict) else {}
        market = str((context_payload.get("market_context", {}) or {}).get("market", "")).strip().lower()
        if market in {"cn", "hk", "us"}:
            return market

        enhanced_market = str(
            ((enhanced_payload.get("market_context", {}) or {}).get("market"))
            or enhanced_payload.get("market")
            or ""
        ).strip().lower()
        if enhanced_market in {"cn", "hk", "us"}:
            return enhanced_market

        payload_market = str((payload.get("market_snapshot") or {}).get("market", "")).strip().lower()
        if payload_market in {"cn", "hk", "us"}:
            return payload_market
        return get_market_for_stock(code) or "cn"

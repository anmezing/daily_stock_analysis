# -*- coding: utf-8 -*-
"""
===================================
股票自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 LLM 调用逻辑（通过 LiteLLM 统一调用 Gemini/Anthropic/OpenAI 等）
2. 结合技术面和消息面生成分析报告
3. 解析 LLM 响应为结构化 AnalysisResult
"""

import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, List, Tuple

try:
    import litellm
    from litellm import Router
except Exception:  # pragma: no cover - optional dependency
    litellm = None
    Router = None

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover - optional dependency
    repair_json = None

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import Config, get_config, get_api_keys_for_model, extra_litellm_params, get_configured_llm_models
from src.core.llm_rate_limiter import get_llm_rate_limiter, get_rate_limit_bucket
from src.core.signal_semantics import infer_signal
from src.core.trade_plan import apply_trade_plan_fallback, build_trade_plan
from src.core.trading_calendar import get_market_for_stock
from src.storage import persist_llm_usage
from src.data.stock_mapping import STOCK_NAME_MAP
from src.schemas.report_schema import AnalysisReportSchema
from src.utils.json_sanitize import sanitize_non_finite

logger = logging.getLogger(__name__)


class _LiteLLMStreamError(RuntimeError):
    """Internal error wrapper that records whether any text was streamed."""

    def __init__(self, message: str, *, partial_received: bool = False):
        super().__init__(message)
        self.partial_received = partial_received


_BUY_SIGNAL_HINTS = {
    "buy", "strong_buy", "long", "bullish",
    "买入", "加仓", "强烈买入", "看多", "强烈看多",
}
_SELL_SIGNAL_HINTS = {
    "sell", "strong_sell", "short", "bearish",
    "卖出", "减仓", "强烈卖出", "看空", "强烈看空",
}
_HOLD_SIGNAL_HINTS = {
    "hold", "neutral", "wait", "watch",
    "持有", "观望", "震荡", "中性",
}
_SIGNAL_LEVEL_BY_ADVICE = {
    "强烈买入": ("强烈买入", "💚", "强买"),
    "买入": ("买入", "🟢", "买入"),
    "加仓": ("买入", "🟢", "买入"),
    "持有": ("持有", "🟡", "持有"),
    "观望": ("观望", "⚪", "观望"),
    "减仓": ("减仓", "🟠", "减仓"),
    "卖出": ("卖出", "🔴", "卖出"),
    "强烈卖出": ("卖出", "🔴", "卖出"),
}


def _canonical_decision_signal(value: Any) -> str:
    """Normalize model-facing decision labels into buy/hold/sell."""
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    if not text:
        return ""
    if text in _BUY_SIGNAL_HINTS:
        return "buy"
    if text in _SELL_SIGNAL_HINTS:
        return "sell"
    if text in _HOLD_SIGNAL_HINTS:
        return "hold"
    return ""


def _signal_from_operation_advice(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    # 与回测共用同一份词表（src/core/signal_semantics.py），避免两套关键词错位
    signal = infer_signal(text)
    if signal:
        return signal
    return _canonical_decision_signal(text)


def _split_operation_advice_tokens(value: Any) -> List[str]:
    if not isinstance(value, str):
        return []
    tokens: List[str] = []
    normalized = (
        value.replace("／", "/")
        .replace("|", "/")
        .replace("，", "/")
        .replace(",", "/")
        .replace("、", "/")
    )
    for part in normalized.split("/"):
        token = part.strip()
        if token:
            tokens.append(token)
    return tokens


def _signal_from_trend_prediction(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    if "看空" in text or "bearish" in text.lower():
        return "sell"
    if "看多" in text or "bullish" in text.lower():
        return "buy"
    if any(keyword in text for keyword in ("震荡", "中性", "盘整")):
        return "hold"
    return _canonical_decision_signal(text)


def _signal_from_sentiment_score(score: int) -> str:
    if score >= 60:
        return "buy"
    if score <= 39:
        return "sell"
    return "hold"


def _coerce_sentiment_score_value(value: Any, default: int = 50) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


_CONFIDENCE_LEVELS = {"高", "中", "低"}
_CONFIDENCE_ALIASES = {
    "high": "高",
    "medium": "中",
    "mid": "中",
    "normal": "中",
    "low": "低",
}


def _normalize_confidence_level(value: Any, *, has_conflict: bool = False) -> str:
    """把模型返回的置信度收敛到 高/中/低，避免脏值直接进报告与统计。"""
    text = str(value or "").strip()
    if text not in _CONFIDENCE_LEVELS:
        text = _CONFIDENCE_ALIASES.get(text.lower(), "中")
    if has_conflict:
        # 模型自己的字段互相打架、信号已被降级为观望，就不该再显示高置信度
        return "低"
    return text


def _clamp_sentiment_score_to_signal(score: int, signal: str) -> int:
    return _clamp_sentiment_score_to_signal_with_bias(score, signal, "neutral")


def _clamp_sentiment_score_to_signal_with_bias(score: int, signal: str, hold_bias: str = "neutral") -> int:
    hold_bands = {
        "bullish": (52, 59),
        "neutral": (46, 54),
        "bearish": (40, 47),
    }
    bands = {
        # Upper bound extended from 79 → 100 to allow "强烈买入" (score >= 80)
        "buy": (60, 100),
        "hold": hold_bands.get(hold_bias, hold_bands["neutral"]),
        "sell": (0, 39),
    }
    low, high = bands.get(signal, (0, 100))
    return max(low, min(high, score))


def _canonical_operation_advice(signal: str) -> str:
    return _canonical_operation_advice_with_bias(signal, "neutral")


def _canonical_operation_advice_with_bias(signal: str, hold_bias: str = "neutral") -> str:
    if signal == "hold":
        return "观望" if hold_bias == "bearish" else "持有"
    mapping = {
        "buy": "买入",
        "sell": "减仓/卖出",
    }
    return mapping.get(signal, "观望")


def _canonical_trend_prediction(signal: str) -> str:
    return _canonical_trend_prediction_with_bias(signal, "neutral")


def _canonical_trend_prediction_with_bias(signal: str, hold_bias: str = "neutral") -> str:
    if signal == "hold":
        hold_mapping = {
            "bullish": "震荡偏多",
            "neutral": "震荡",
            "bearish": "震荡偏空",
        }
        return hold_mapping.get(hold_bias, "震荡")
    mapping = {
        "buy": "看多",
        "sell": "看空",
    }
    return mapping.get(signal, "震荡")


def _canonical_signal_type(signal: str) -> str:
    return _canonical_signal_type_with_bias(signal, "neutral")


def _canonical_signal_type_with_bias(signal: str, hold_bias: str = "neutral") -> str:
    if signal == "hold":
        hold_mapping = {
            "bullish": "持有待机",
            "neutral": "持有观望",
            "bearish": "谨慎观望",
        }
        return hold_mapping.get(hold_bias, "持有观望")
    mapping = {
        "buy": "买入信号",
        "sell": "卖出信号",
    }
    return mapping.get(signal, "持有观望")


def get_result_signal_level(result: "AnalysisResult") -> Tuple[str, str, str]:
    """Return stable display tuple from advice/decision_type/score.

    Display layers should trust the machine-stable ``decision_type`` before
    score fallback, while still preserving explicit advice when it is known.
    """
    advice = getattr(result, "operation_advice", "")
    score = _coerce_sentiment_score_value(getattr(result, "sentiment_score", 50), 50)
    decision_signal = _canonical_decision_signal(getattr(result, "decision_type", ""))
    advice_tokens = _split_operation_advice_tokens(advice)

    if isinstance(advice, str) and advice in _SIGNAL_LEVEL_BY_ADVICE:
        return _SIGNAL_LEVEL_BY_ADVICE[advice]

    if advice_tokens:
        recognized_tokens = [token for token in advice_tokens if token in _SIGNAL_LEVEL_BY_ADVICE]
        if recognized_tokens:
            if decision_signal == "buy":
                if "强烈买入" in recognized_tokens:
                    return _SIGNAL_LEVEL_BY_ADVICE["强烈买入"]
                return _SIGNAL_LEVEL_BY_ADVICE["买入"]
            if decision_signal == "sell":
                if "强烈卖出" in recognized_tokens or "卖出" in recognized_tokens:
                    return _SIGNAL_LEVEL_BY_ADVICE["卖出"]
                return _SIGNAL_LEVEL_BY_ADVICE["减仓"]
            if decision_signal == "hold":
                if "持有" in recognized_tokens:
                    return _SIGNAL_LEVEL_BY_ADVICE["持有"]
                return _SIGNAL_LEVEL_BY_ADVICE["观望"]

            component_signals = set()
            for token in recognized_tokens:
                token_signal = _signal_from_operation_advice(token)
                if token_signal:
                    component_signals.add(token_signal)
            if len(component_signals) == 1:
                single_signal = next(iter(component_signals))
                if single_signal == "buy":
                    return _SIGNAL_LEVEL_BY_ADVICE["买入"]
                if single_signal == "sell":
                    if "卖出" in recognized_tokens or "强烈卖出" in recognized_tokens:
                        return _SIGNAL_LEVEL_BY_ADVICE["卖出"]
                    return _SIGNAL_LEVEL_BY_ADVICE["减仓"]
                if "持有" in recognized_tokens:
                    return _SIGNAL_LEVEL_BY_ADVICE["持有"]
                return _SIGNAL_LEVEL_BY_ADVICE["观望"]

    if decision_signal == "buy":
        return _SIGNAL_LEVEL_BY_ADVICE["买入"]
    if decision_signal == "sell":
        return _SIGNAL_LEVEL_BY_ADVICE["卖出"]
    if decision_signal == "hold":
        trend_text = str(getattr(result, "trend_prediction", "") or "")
        if "偏空" in trend_text:
            return _SIGNAL_LEVEL_BY_ADVICE["观望"]
        return _SIGNAL_LEVEL_BY_ADVICE["持有"]

    if score >= 80:
        return _SIGNAL_LEVEL_BY_ADVICE["强烈买入"]
    if score >= 65:
        return _SIGNAL_LEVEL_BY_ADVICE["买入"]
    if score >= 55:
        return _SIGNAL_LEVEL_BY_ADVICE["持有"]
    if score >= 45:
        return _SIGNAL_LEVEL_BY_ADVICE["观望"]
    if score >= 35:
        return _SIGNAL_LEVEL_BY_ADVICE["减仓"]
    return _SIGNAL_LEVEL_BY_ADVICE["卖出"]


def _default_position_advice_for_signal(signal: str) -> Dict[str, str]:
    return _default_position_advice_for_signal_with_bias(signal, "neutral")


def _default_position_advice_for_signal_with_bias(signal: str, hold_bias: str = "neutral") -> Dict[str, str]:
    if signal == "hold":
        hold_mapping = {
            "bullish": {
                "no_position": "暂不追价，等回踩支撑位或下一次确认后再考虑介入。",
                "has_position": "可继续持有，回踩关键支撑不破则耐心观察。",
            },
            "neutral": {
                "no_position": "先观察，等待趋势和消息进一步明朗。",
                "has_position": "控制仓位，等待新的方向选择。",
            },
            "bearish": {
                "no_position": "暂不开新仓，等待风险释放或重新企稳。",
                "has_position": "以防守为主，跌破关键位优先减仓。",
            },
        }
        return hold_mapping.get(hold_bias, hold_mapping["neutral"])
    mapping = {
        "buy": {
            "no_position": "可等待回踩支撑位后分批试仓，避免一次性追高。",
            "has_position": "可继续持有，回踩关键位不破再考虑加仓。",
        },
        "sell": {
            "no_position": "暂不参与，等待风险充分释放后再评估。",
            "has_position": "优先控制回撤，按计划减仓或离场。",
        },
    }
    return mapping.get(signal, _default_position_advice_for_signal_with_bias("hold", hold_bias))


def _resolve_consistent_signal(signals: List[str]) -> Tuple[str, bool]:
    valid_signals = [signal for signal in signals if signal in {"buy", "hold", "sell"}]
    if not valid_signals:
        return "hold", False

    counts = Counter(valid_signals)
    top_count = max(counts.values())
    winners = [signal for signal, count in counts.items() if count == top_count]
    if len(winners) == 1:
        final_signal = winners[0]
    else:
        final_signal = "hold"

    has_conflict = len(set(valid_signals)) > 1
    if "buy" in valid_signals and "sell" in valid_signals:
        final_signal = "hold"
        has_conflict = True

    return final_signal, has_conflict


def _derive_hold_bias(
    signals: List[str],
    *,
    trend_prediction: str = "",
) -> str:
    valid_signals = [signal for signal in signals if signal in {"buy", "hold", "sell"}]
    buy_count = valid_signals.count("buy")
    sell_count = valid_signals.count("sell")
    if buy_count > sell_count:
        return "bullish"
    if sell_count > buy_count:
        return "bearish"
    trend_text = str(trend_prediction or "").strip()
    if "偏多" in trend_text:
        return "bullish"
    if "偏空" in trend_text:
        return "bearish"
    return "neutral"


def _resolve_signal_state(
    *,
    decision_signal: str,
    operation_signal: str,
    trend_signal: str,
    score_signal: str,
    raw_trend_prediction: str = "",
) -> Tuple[str, bool, str]:
    """Resolve the stable buy/hold/sell state from top-level signal fields.

    Directional actions still need support from trend or score. When the model
    emits a weak directional action such as ``buy + 50分 + 震荡``, we downgrade
    to hold instead of mechanically trusting the action field.
    """
    action_signal = decision_signal or operation_signal
    opposite_pairs = {("buy", "sell"), ("sell", "buy")}
    hard_conflict = False

    if decision_signal and operation_signal and (decision_signal, operation_signal) in opposite_pairs:
        hard_conflict = True
    elif action_signal and trend_signal and (action_signal, trend_signal) in opposite_pairs:
        hard_conflict = True
    elif action_signal and score_signal and (action_signal, score_signal) in opposite_pairs:
        hard_conflict = True

    soft_conflict = False
    if not hard_conflict and action_signal in {"buy", "sell"}:
        support_signals = [signal for signal in (trend_signal, score_signal) if signal]
        directional_support = any(signal == action_signal for signal in support_signals)
        if support_signals and not directional_support:
            soft_conflict = True

    has_conflict = hard_conflict or soft_conflict
    if has_conflict and action_signal in {"buy", "sell"}:
        final_signal = "hold"
    elif action_signal:
        final_signal = action_signal
    else:
        final_signal, has_conflict = _resolve_consistent_signal([
            trend_signal,
            score_signal,
        ])

    hold_bias = "neutral"
    if final_signal == "hold":
        hold_bias = _derive_hold_bias(
            [
                decision_signal,
                operation_signal,
                trend_signal,
                score_signal,
            ],
            trend_prediction=raw_trend_prediction,
        )

    return final_signal, has_conflict, hold_bias


def _normalize_dashboard_signal_fields(
    dashboard: Any,
    signal: str,
    *,
    has_conflict: bool,
    hold_bias: str = "neutral",
) -> Any:
    if not isinstance(dashboard, dict):
        return dashboard

    dashboard["decision_type"] = signal
    core = dashboard.get("core_conclusion")
    if not isinstance(core, dict):
        core = {}
        dashboard["core_conclusion"] = core

    core["signal_type"] = _canonical_signal_type_with_bias(signal, hold_bias)
    if has_conflict or not isinstance(core.get("position_advice"), dict):
        core["position_advice"] = _default_position_advice_for_signal_with_bias(signal, hold_bias)
    if has_conflict:
        hold_summary = {
            "bullish": ("多空有分歧，当前先持有等待更优买点", "本周内"),
            "neutral": ("信号存在分歧，当前以观望为主", "不急"),
            "bearish": ("上行动能不足，当前先观望或偏防守", "今日内"),
        }
        summary, time_sensitivity = {
            "buy": ("趋势偏强，但应等待更优买点", "本周内"),
            "hold": hold_summary.get(hold_bias, hold_summary["neutral"]),
            "sell": ("风险偏高，优先减仓防守", "立即行动"),
        }.get(signal, hold_summary["neutral"])
        core["one_sentence"] = summary
        core["time_sensitivity"] = time_sensitivity

    return dashboard


def check_content_integrity(result: "AnalysisResult") -> Tuple[bool, List[str]]:
    """
    Check mandatory fields for report content integrity.
    Returns (pass, missing_fields). Module-level for use by pipeline (agent weak mode).
    """
    missing: List[str] = []
    if result.sentiment_score is None:
        missing.append("sentiment_score")
    advice = result.operation_advice
    if not advice or not isinstance(advice, str) or not advice.strip():
        missing.append("operation_advice")
    summary = result.analysis_summary
    if not summary or not isinstance(summary, str) or not summary.strip():
        missing.append("analysis_summary")
    dash = result.dashboard if isinstance(result.dashboard, dict) else {}
    core = dash.get("core_conclusion")
    core = core if isinstance(core, dict) else {}
    if not (core.get("one_sentence") or "").strip():
        missing.append("dashboard.core_conclusion.one_sentence")
    intel = dash.get("intelligence")
    intel = intel if isinstance(intel, dict) else None
    if intel is None or "risk_alerts" not in intel:
        missing.append("dashboard.intelligence.risk_alerts")
    if result.decision_type in ("buy", "hold"):
        battle = dash.get("battle_plan")
        battle = battle if isinstance(battle, dict) else {}
        sp = battle.get("sniper_points")
        sp = sp if isinstance(sp, dict) else {}
        ideal_buy = sp.get("ideal_buy")
        secondary_buy = sp.get("secondary_buy")
        stop_loss = sp.get("stop_loss")
        if stop_loss is None or (isinstance(stop_loss, str) and not stop_loss.strip()):
            missing.append("dashboard.battle_plan.sniper_points.stop_loss")
        # 只有"买入类"结论才必须具备入场价位；持有/观望本来就没有建仓计划，
        # 硬要求 ideal_buy/secondary_buy 只会让模型编一个数，或者被占位补成"待补充"。
        if result.decision_type == "buy":
            if ideal_buy is None or (isinstance(ideal_buy, str) and not ideal_buy.strip()):
                missing.append("dashboard.battle_plan.sniper_points.ideal_buy")
            if secondary_buy is None or (isinstance(secondary_buy, str) and not secondary_buy.strip()):
                missing.append("dashboard.battle_plan.sniper_points.secondary_buy")
            elif str(ideal_buy).strip() == str(secondary_buy).strip():
                missing.append("dashboard.battle_plan.sniper_points.secondary_buy")
    return len(missing) == 0, missing


def apply_placeholder_fill(result: "AnalysisResult", missing_fields: List[str]) -> None:
    """Fill missing mandatory fields with placeholders (in-place). Module-level for pipeline."""
    for field in missing_fields:
        if field == "sentiment_score":
            result.sentiment_score = 50
        elif field == "operation_advice":
            result.operation_advice = result.operation_advice or "待补充"
        elif field == "analysis_summary":
            result.analysis_summary = result.analysis_summary or "待补充"
        elif field == "dashboard.core_conclusion.one_sentence":
            if not result.dashboard:
                result.dashboard = {}
            if "core_conclusion" not in result.dashboard:
                result.dashboard["core_conclusion"] = {}
            result.dashboard["core_conclusion"]["one_sentence"] = (
                result.dashboard["core_conclusion"].get("one_sentence") or "待补充"
            )
        elif field == "dashboard.intelligence.risk_alerts":
            if not result.dashboard:
                result.dashboard = {}
            if "intelligence" not in result.dashboard:
                result.dashboard["intelligence"] = {}
            if "risk_alerts" not in result.dashboard["intelligence"]:
                result.dashboard["intelligence"]["risk_alerts"] = []
        elif field in (
            "dashboard.battle_plan.sniper_points.stop_loss",
            "dashboard.battle_plan.sniper_points.ideal_buy",
            "dashboard.battle_plan.sniper_points.secondary_buy",
        ):
            if not result.dashboard:
                result.dashboard = {}
            if "battle_plan" not in result.dashboard:
                result.dashboard["battle_plan"] = {}
            if "sniper_points" not in result.dashboard["battle_plan"]:
                result.dashboard["battle_plan"]["sniper_points"] = {}
            if field == "dashboard.battle_plan.sniper_points.stop_loss":
                result.dashboard["battle_plan"]["sniper_points"]["stop_loss"] = "待补充"
            elif field == "dashboard.battle_plan.sniper_points.ideal_buy":
                result.dashboard["battle_plan"]["sniper_points"]["ideal_buy"] = "待补充"
            elif field == "dashboard.battle_plan.sniper_points.secondary_buy":
                result.dashboard["battle_plan"]["sniper_points"]["secondary_buy"] = "待补充"


# ---------- chip_structure fallback (Issue #589) ----------

_CHIP_KEYS: tuple = ("profit_ratio", "avg_cost", "concentration", "chip_health")


def _is_value_placeholder(v: Any) -> bool:
    """True if value is empty or placeholder (N/A, 数据缺失, etc.).

    Note: numeric 0 is NOT treated as placeholder — 0 can be a valid value
    (e.g. bias_ma5 = 0.0%, MACD bar = 0).
    For chip_structure fields, use ``_is_chip_value_placeholder()`` which
    additionally treats numeric 0 as placeholder.
    """
    if v is None:
        return True
    if isinstance(v, (int, float)):
        return False
    s = str(v).strip().lower()
    return s in ("", "n/a", "na", "数据缺失", "未知")


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely convert to float; return default on failure. Private helper for chip fill."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        try:
            return default if math.isnan(float(v)) else float(v)
        except (ValueError, TypeError):
            return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _derive_chip_health(profit_ratio: float, concentration_90: float) -> str:
    """Derive chip_health from profit_ratio and concentration_90."""
    if profit_ratio >= 0.9:
        return "警惕"  # 获利盘极高
    if concentration_90 >= 0.25:
        return "警惕"  # 筹码分散
    if concentration_90 < 0.15 and 0.3 <= profit_ratio < 0.9:
        return "健康"  # 集中且获利比例适中
    return "一般"


def _build_chip_structure_from_data(chip_data: Any) -> Dict[str, Any]:
    """Build chip_structure dict from ChipDistribution or dict."""
    if hasattr(chip_data, "profit_ratio"):
        pr = _safe_float(chip_data.profit_ratio)
        ac = chip_data.avg_cost
        c90 = _safe_float(chip_data.concentration_90)
    else:
        d = chip_data if isinstance(chip_data, dict) else {}
        pr = _safe_float(d.get("profit_ratio"))
        ac = d.get("avg_cost")
        c90 = _safe_float(d.get("concentration_90"))
    chip_health = _derive_chip_health(pr, c90)
    return {
        "profit_ratio": f"{pr:.1%}",
        "avg_cost": ac if (ac is not None and _safe_float(ac) != 0.0) else "N/A",
        "concentration": f"{c90:.2%}",
        "chip_health": chip_health,
    }


def _is_chip_value_placeholder(v: Any) -> bool:
    """Like _is_value_placeholder but also treats numeric 0 as placeholder.

    For chip_structure fields the LLM outputs ``0`` when it has no data,
    so zero should be treated as "needs to be filled from real data source".
    """
    if _is_value_placeholder(v):
        return True
    if isinstance(v, (int, float)) and v == 0:
        return True
    return False


def fill_chip_structure_if_needed(result: "AnalysisResult", chip_data: Any) -> None:
    """When chip_data exists, fill chip_structure placeholder fields from chip_data (in-place)."""
    if not result or not chip_data:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        # Use `or {}` rather than setdefault so that an explicit `null` from LLM is also replaced
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        cs = dp.get("chip_structure") or {}
        filled = _build_chip_structure_from_data(chip_data)
        # Start from a copy of cs to preserve any extra keys the LLM may have added
        merged = dict(cs)
        for k in _CHIP_KEYS:
            if _is_chip_value_placeholder(merged.get(k)):
                merged[k] = filled[k]
        if merged != cs:
            dp["chip_structure"] = merged
            logger.info("[chip_structure] Filled placeholder chip fields from data source (Issue #589)")
    except Exception as e:
        logger.warning("[chip_structure] Fill failed, skipping: %s", e)


_PRICE_POS_KEYS = ("ma5", "ma10", "ma20", "bias_ma5", "bias_status", "current_price", "support_level", "resistance_level")


def fill_price_position_if_needed(
    result: "AnalysisResult",
    trend_result: Any = None,
    realtime_quote: Any = None,
) -> None:
    """Fill missing price_position fields from trend_result / realtime data (in-place)."""
    if not result:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        pp = dp.get("price_position") or {}

        computed: Dict[str, Any] = {}
        if trend_result:
            tr = trend_result if isinstance(trend_result, dict) else (
                trend_result.__dict__ if hasattr(trend_result, "__dict__") else {}
            )
            computed["ma5"] = tr.get("ma5")
            computed["ma10"] = tr.get("ma10")
            computed["ma20"] = tr.get("ma20")
            computed["bias_ma5"] = tr.get("bias_ma5")
            computed["current_price"] = tr.get("current_price")
            support_levels = tr.get("support_levels") or []
            resistance_levels = tr.get("resistance_levels") or []
            if support_levels:
                computed["support_level"] = support_levels[0]
            if resistance_levels:
                computed["resistance_level"] = resistance_levels[0]
        if realtime_quote:
            rq = realtime_quote if isinstance(realtime_quote, dict) else (
                realtime_quote.to_dict() if hasattr(realtime_quote, "to_dict") else {}
            )
            # current_price == 0 is always invalid (stock price cannot be zero),
            # so treat it as placeholder even though generic _is_value_placeholder
            # no longer considers numeric 0 as placeholder.
            if _is_value_placeholder(computed.get("current_price")) or computed.get("current_price") == 0:
                computed["current_price"] = rq.get("price")

        filled = False
        for k in _PRICE_POS_KEYS:
            if _is_value_placeholder(pp.get(k)) and not _is_value_placeholder(computed.get(k)):
                pp[k] = computed[k]
                filled = True
        if filled:
            dp["price_position"] = pp
            logger.info("[price_position] Filled placeholder fields from computed data")
    except Exception as e:
        logger.warning("[price_position] Fill failed, skipping: %s", e)


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """
    多来源获取股票中文名称

    获取策略（按优先级）：
    1. 从传入的 context 中获取（realtime 数据）
    2. 从静态映射表 STOCK_NAME_MAP 获取
    3. 从 DataFetcherManager 获取（各数据源）
    4. 返回默认名称（股票+代码）

    Args:
        stock_code: 股票代码
        context: 分析上下文（可选）
        data_manager: DataFetcherManager 实例（可选）

    Returns:
        股票中文名称
    """
    # 1. 从上下文获取（实时行情数据）
    if context:
        # 优先从 stock_name 字段获取
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name

        # 其次从 realtime 数据获取
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    # 2. 从静态映射表获取
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    # 3. 从数据源获取
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"无法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                # 更新缓存
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"从数据源获取股票名称失败: {e}")

    # 4. 返回默认名称
    return f'股票{stock_code}'


@dataclass
class AnalysisResult:
    """
    AI 分析结果数据类 - 决策仪表盘版

    封装 Gemini 返回的分析结果，包含决策仪表盘和详细分析
    """
    code: str
    name: str

    # ========== 核心指标 ==========
    sentiment_score: int  # 综合评分 0-100 (>70强烈看多, >60看多, 40-60震荡, <40看空)
    trend_prediction: str  # 趋势预测：强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 操作建议：买入/加仓/持有/减仓/卖出/观望
    decision_type: str = "hold"  # 决策类型：buy/hold/sell（用于统计）
    confidence_level: str = "中"  # 置信度：高/中/低

    # ========== 决策仪表盘 (新增) ==========
    dashboard: Optional[Dict[str, Any]] = None  # 完整的决策仪表盘数据

    # ========== 走势分析 ==========
    trend_analysis: str = ""  # 走势形态分析（支撑位、压力位、趋势线等）
    short_term_outlook: str = ""  # 短期展望（1-3日）
    medium_term_outlook: str = ""  # 中期展望（1-2周）

    # ========== 技术面分析 ==========
    technical_analysis: str = ""  # 技术指标综合分析
    ma_analysis: str = ""  # 均线分析（多头/空头排列，金叉/死叉等）
    volume_analysis: str = ""  # 量能分析（放量/缩量，主力动向等）
    pattern_analysis: str = ""  # K线形态分析

    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""  # 基本面综合分析
    sector_position: str = ""  # 板块地位和行业趋势
    company_highlights: str = ""  # 公司亮点/风险点

    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""  # 近期重要新闻/公告摘要
    market_sentiment: str = ""  # 市场情绪分析
    hot_topics: str = ""  # 相关热点话题

    # ========== 综合分析 ==========
    analysis_summary: str = ""  # 综合分析摘要
    key_points: str = ""  # 核心看点（3-5个要点）
    risk_warning: str = ""  # 风险提示
    buy_reason: str = ""  # 买入/卖出理由

    # ========== 元数据 ==========
    market_snapshot: Optional[Dict[str, Any]] = None  # 当日行情快照（展示用）
    market_themes: Optional[Dict[str, Any]] = None  # 市场热点/板块轮动快照（展示用）
    raw_response: Optional[str] = None  # 原始响应（调试用）
    search_performed: bool = False  # 是否执行了联网搜索
    data_sources: str = ""  # 数据来源说明
    success: bool = True
    error_message: Optional[str] = None

    # ========== 价格数据（分析时快照）==========
    current_price: Optional[float] = None  # 分析时的股价
    change_pct: Optional[float] = None     # 分析时的涨跌幅(%)
    analysis_date: Optional[str] = None    # 做出判断的日期（市场当地日期）
    data_date: Optional[str] = None        # 技术指标所用最后一根 K 线的日期

    # ========== 模型标记（Issue #528）==========
    model_used: Optional[str] = None  # 分析使用的 LLM 模型（完整名，如 gemini/gemini-2.0-flash）
    # 结构化解析 / 纯文本降级 / 输出截断。降级和截断结果都不能和正常分析
    # 一起进回测与学习。
    parse_mode: str = "structured"

    # ========== 历史对比（Report Engine P0）==========
    query_id: Optional[str] = None  # 本次分析 query_id，用于历史对比时排除本次记录
    calibration_info: Optional[Dict[str, Any]] = None  # 回测校准/进化信息（如有）
    strategy_id: Optional[str] = None  # 主策略 ID（来自多策略编排器）
    # 数值一致性校验：模型是否引用了输入里不存在的价格/均线
    numeric_consistency: bool = True
    numeric_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,  # 决策仪表盘数据
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'market_themes': self.market_themes,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
            'analysis_date': self.analysis_date,
            'data_date': self.data_date,
            'model_used': self.model_used,
            'parse_mode': self.parse_mode,
            'calibration_info': self.calibration_info,
            'query_id': self.query_id,
            'strategy_id': self.strategy_id,
            'numeric_consistency': self.numeric_consistency,
            'numeric_issues': self.numeric_issues,
        }

    def get_core_conclusion(self) -> str:
        """获取核心结论（一句话）"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        """获取持仓建议"""
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        """获取狙击点位"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        """获取检查清单"""
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        """获取风险警报"""
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        return get_result_signal_level(self)[1]

    def get_confidence_stars(self) -> str:
        """返回置信度星级"""
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


def validate_result_numbers(result: AnalysisResult, context: Dict[str, Any]) -> List[str]:
    """Check that the model did not invent prices or moving averages.

    Only structural mismatches (a field that should mirror the input but is
    off by >1%) invalidate a result. Invalid stop/target ordering is recorded
    but does not by itself mark the analysis unusable.
    """
    import re

    def _to_number(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        text = re.sub(r"MA\d+", " ", value.replace(",", ""), flags=re.I)
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group()) if match else None

    dashboard = result.dashboard if isinstance(result.dashboard, dict) else {}
    perspective = dashboard.get("data_perspective")
    position = perspective.get("price_position") if isinstance(perspective, dict) else {}
    position = position if isinstance(position, dict) else {}

    today = context.get("today") if isinstance(context.get("today"), dict) else {}
    trend = context.get("trend_analysis") if isinstance(context.get("trend_analysis"), dict) else {}
    realtime = context.get("realtime") if isinstance(context.get("realtime"), dict) else {}

    expected = {
        "current_price": today.get("close") or realtime.get("price"),
        "ma5": trend.get("ma5") or today.get("ma5"),
        "ma10": trend.get("ma10") or today.get("ma10"),
        "ma20": trend.get("ma20") or today.get("ma20"),
    }

    issues: List[str] = []
    reference_price: Optional[float] = None
    for key, expected_value in expected.items():
        expected_number = _to_number(expected_value)
        if key == "current_price":
            reference_price = expected_number
        reported = _to_number(position.get(key))
        if reported is None or expected_number is None or expected_number <= 0:
            continue
        if abs(reported - expected_number) / expected_number > 0.01:
            issues.append(f"price_mismatch:{key}")

    battle = dashboard.get("battle_plan")
    sniper = battle.get("sniper_points") if isinstance(battle, dict) else {}
    sniper = sniper if isinstance(sniper, dict) else {}
    stop_loss = _to_number(sniper.get("stop_loss"))
    take_profit = _to_number(sniper.get("take_profit"))
    if reference_price and stop_loss and take_profit and result.decision_type == "buy":
        if not (stop_loss < reference_price < take_profit):
            issues.append("target_order_invalid")

    volume = perspective.get("volume_analysis") if isinstance(perspective, dict) else {}
    volume = volume if isinstance(volume, dict) else {}
    volume_ratio = _to_number(volume.get("volume_ratio"))
    volume_status = str(volume.get("volume_status") or "")
    if volume_ratio is not None:
        if volume_ratio < 1.0 and "放量" in volume_status:
            issues.append("volume_status_conflict")
        elif volume_ratio >= 1.5 and "缩量" in volume_status:
            issues.append("volume_status_conflict")

    return issues


def normalize_analysis_result_signals(result: AnalysisResult) -> AnalysisResult:
    """Normalize an existing AnalysisResult into a self-consistent signal state.

    Used by non-JSON downgrade paths and by agent-mode conversion so every
    runtime path shares the same decision/advice/trend/dashboard contract.
    """
    raw_score = _coerce_sentiment_score_value(getattr(result, "sentiment_score", 50), 50)
    raw_trend_prediction = getattr(result, "trend_prediction", "震荡")
    raw_operation_advice = getattr(result, "operation_advice", "持有")
    raw_decision_type = getattr(result, "decision_type", "")

    decision_signal = _canonical_decision_signal(raw_decision_type)
    operation_signal = _signal_from_operation_advice(raw_operation_advice)
    trend_signal = _signal_from_trend_prediction(raw_trend_prediction)
    score_signal = _signal_from_sentiment_score(raw_score)

    action_signal = decision_signal or operation_signal
    final_signal, has_conflict, hold_bias = _resolve_signal_state(
        decision_signal=decision_signal,
        operation_signal=operation_signal,
        trend_signal=trend_signal,
        score_signal=score_signal,
        raw_trend_prediction=raw_trend_prediction,
    )

    normalized_score = raw_score
    if raw_score < 0 or raw_score > 100 or has_conflict or score_signal not in ("", final_signal):
        normalized_score = _clamp_sentiment_score_to_signal_with_bias(
            max(0, min(100, raw_score)),
            final_signal,
            hold_bias,
        )

    normalized_trend_prediction = raw_trend_prediction
    canonical_trend = _canonical_trend_prediction_with_bias(final_signal, hold_bias)
    if (
        has_conflict
        or trend_signal not in ("", final_signal)
        or (final_signal == "hold" and hold_bias != "neutral" and str(raw_trend_prediction).strip() != canonical_trend)
    ):
        normalized_trend_prediction = canonical_trend

    normalized_operation_advice = raw_operation_advice
    canonical_advice = _canonical_operation_advice_with_bias(final_signal, hold_bias)
    if (
        not action_signal
        or has_conflict
        or operation_signal not in ("", final_signal)
        or (final_signal == "hold" and hold_bias != "neutral" and str(raw_operation_advice).strip() != canonical_advice)
    ):
        normalized_operation_advice = canonical_advice

    result.sentiment_score = normalized_score
    result.trend_prediction = normalized_trend_prediction
    result.operation_advice = normalized_operation_advice
    result.decision_type = final_signal
    result.dashboard = _normalize_dashboard_signal_fields(
        result.dashboard,
        final_signal,
        has_conflict=has_conflict,
        hold_bias=hold_bias,
    )
    return result


class GeminiAnalyzer:
    """
    Gemini AI 分析器

    职责：
    1. 调用 Google Gemini API 进行股票分析
    2. 结合预先搜索的新闻和技术面数据生成分析报告
    3. 解析 AI 返回的 JSON 格式结果

    使用方式：
        analyzer = GeminiAnalyzer()
        result = analyzer.analyze(context, news_context)
    """

    # ========================================
    # 系统提示词 - 决策仪表盘 v3.0
    # 拆成「公共部分 + 市场片段」：按标的所属市场拼装，
    # 避免无关市场的规则稀释注意力，也减少输入 token。
    # ========================================

    SYSTEM_PROMPT_BASE = """你是一位专注于趋势交易的股票投资分析师，负责生成专业的【决策仪表盘】分析报告。

## 硬性约束（优先级高于其它所有规则）

1. **只用给定数据**：价格、均线、乖离率、量比、换手率、PE/PB、筹码等数值必须来自提示中已提供的数据。提示未提供时一律输出 `null`，禁止估算、推测或编造。
2. **市场规则不混用**：只按下方【当前市场规则】执行。不要把 A 股的涨跌停 / T+1 逻辑套到港股或美股，反之亦然。
3. **中国政策权重**：仅当提示标注 `China exposure` 或“中国政策权重高”时才作为核心变量；标注“权重低 / 谨慎使用”时不得作为主要结论来源。
4. **字段一致性**：`decision_type`、`operation_advice`、`trend_prediction`、`sentiment_score` 必须自洽。多空分歧时统一用 `hold` + `震荡偏多/震荡/震荡偏空` 表达，禁止“看空但买入”这类冲突组合。
5. **货币单位**：价格使用标的计价货币（美股 USD、港股 HKD、A 股 CNY），不要用“元”统称。
6. **筹码可能缺失**：未提供筹码分布时，`chip_structure` 整体输出 `null`，且**不得因此下调评分或封顶**。
7. **必须回答“什么时候买、什么时候卖、什么时候等”**：`battle_plan.trade_plan` 的五个字段不可为空，且必须包含可验证的价格或信号条件。禁止“逢低买入”“注意风险”“视情况而定”这类无法执行的空话。
8. **买卖计划必须对称**：`buy` 要给出入场区间、止损、目标；`hold` 要写清“什么条件下我会卖出”和“什么条件下才考虑买入”；`sell` 要给出离场条件与重新入场条件。没有这些内容的结论视为不合格。

## 输出格式：决策仪表盘 JSON

请严格按以下结构输出。`null` 表示该项无数据；必填键不可省略，可选字段无内容时填 `null`。

```json
{
    "stock_name": "股票中文名称",
    "sentiment_score": 0-100整数,
    "trend_prediction": "强烈看多/看多/震荡偏多/震荡/震荡偏空/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",

    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句话核心结论（30字以内，直接告诉用户做什么）",
            "signal_type": "买入信号/持有待机/持有观望/谨慎观望/卖出信号/风险警告",
            "time_sensitivity": "立即行动/今日内/本周内/不急",
            "position_advice": {
                "no_position": "空仓者建议：具体操作指引",
                "has_position": "持仓者建议：具体操作指引"
            }
        },
        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均线排列状态描述",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 当前价格数值或null,
                "ma5": MA5数值或null,
                "ma10": MA10数值或null,
                "ma20": MA20数值或null,
                "bias_ma5": 乖离率百分比数值或null,
                "bias_status": "安全/警戒/危险",
                "support_level": 支撑位价格或null,
                "resistance_level": 压力位价格或null
            },
            "volume_analysis": {
                "volume_ratio": 量比数值或null,
                "volume_status": "放量/缩量/平量/未知",
                "turnover_rate": 换手率百分比或null,
                "volume_meaning": "量能含义解读（如：缩量回调表示抛压减轻）"
            },
            "chip_structure": {
                "profit_ratio": 获利比例,
                "avg_cost": 平均成本,
                "concentration": 筹码集中度,
                "chip_health": "健康/一般/警惕/不适用"
            }
        },
        "intelligence": {
            "latest_news": "【最新消息】近期重要新闻摘要",
            "risk_alerts": ["风险点1：具体描述", "风险点2：具体描述"],
            "positive_catalysts": ["利好1：具体描述", "利好2：具体描述"],
            "earnings_outlook": "业绩预期分析",
            "sentiment_summary": "舆情情绪一句话总结"
        },
        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "理想买入点（用标的计价货币描述价位）",
                "secondary_buy": "次优买入点",
                "stop_loss": "止损位",
                "take_profit": "目标位"
            },
            "trade_plan": {
                "buy_trigger": "买入触发条件：必须写清价格区间 + 确认信号（如：回踩 120–122 且缩量收阳）",
                "wait_condition": "观望条件：什么情况下不买、继续等（如：现价高于 130 不追）",
                "sell_trigger": "卖出条件：持仓者的止损价、趋势失效信号、分批止盈位",
                "invalidation": "买入逻辑失效条件：跌破哪个价位 / 出现什么事件就不再持有",
                "time_stop": "时间止损：持有多少交易日仍不达预期则退出"
            },
            "position_strategy": {
                "suggested_position": "建议仓位：X成",
                "entry_plan": "分批建仓策略描述",
                "risk_control": "风控策略描述"
            },
            "action_checklist": [
            "通过/警示/否决 检查项1：趋势结构成立",
            "通过/警示/否决 检查项2：入场位与波动匹配",
            "通过/警示/否决 检查项3：量能/成交活跃度配合",
            "通过/警示/否决 检查项4：官方披露无重大利空",
            "通过/警示/否决 检查项5：筹码或持仓结构可接受（无数据时标“警示”，不据此否决）",
            "通过/警示/否决 检查项6：估值或事件催化支持"
            ]
        }
    },

    "analysis_summary": "100字以内综合分析摘要",
    "key_points": "3-5个核心看点，逗号分隔",
    "risk_warning": "风险提示",
    "buy_reason": "操作理由，引用当前市场规则",

    "trend_analysis": "走势形态分析（≤120字）",
    "short_term_outlook": "短期1-3日展望（≤60字）",
    "medium_term_outlook": "中期1-2周展望（≤60字）",
    "technical_analysis": "技术面综合分析（≤120字）",
    "ma_analysis": "均线系统分析（≤80字）",
    "volume_analysis": "量能分析（≤80字）",
    "pattern_analysis": "K线形态分析（≤80字）",
    "fundamental_analysis": "基本面分析（≤120字）",
    "sector_position": "板块行业分析（≤80字）",
    "company_highlights": "公司亮点/风险（≤80字）",
    "news_summary": "新闻摘要（≤120字）",
    "market_sentiment": "市场情绪（≤60字）",
    "hot_topics": "相关热点（≤60字）",

    "search_performed": true/false,
    "data_sources": "数据来源说明"
}
```

> 上面这些自由文本字段与 `dashboard` 内容重复时，可精简表述，但不要留空字符串；确实无内容时填 `null`。

## 评分标准

评分依据「趋势 + 入场位置 + 量能 + 催化 + 风险」，任何单一指标都不得单独封顶或否决：

| 分值 | `operation_advice` | 条件 |
|---|---|---|
| 80-100 | 买入 / 加仓（强烈） | 趋势确认（见市场规则）＋ 入场位处于安全区 ＋ 量能配合 ＋ 有明确催化 ＋ 无重大风险 |
| 60-79 | 买入 / 加仓 | 趋势成立但催化一般，或入场位略偏离安全区；允许一项次要条件不满足 |
| 40-59 | 持有 / 观望 | 趋势不明、乖离率进入警戒区、量能不足，或存在待消化的风险事件 |
| 0-39 | 减仓 / 卖出 / 清仓 | 空头排列、关键趋势位被破坏、放量下跌，或重大利空 |

`持有` 与 `观望` 的区别（必须分清）：

- **持有**：已有仓位、继续拿着，倾向 `not_down`（允许波动，但不该显著下跌）。
- **观望**：空仓等待，倾向 `not_up`（不追高，避免在风险位置进场）。

`减仓` / `止盈` / `止损` 属于**仓位管理**，不等于看空：它们表达的是"降低暴露"，不是"预期下跌"。
只有 `卖出` / `清仓` 才表达明确的看空判断。

补充约束：

- 筹码集中度、PE/PB 等字段**只在数据可用时**作为加减分项；缺失按中性处理，不影响评分上限。
- “强烈买入”不要求筹码数据，也不要求同时满足所有条件，但必须写明核心依据。
- 乖离率阈值一律引用下方【当前市场规则】中的区间，不要自行换算成其它市场标准。

## 核心原则

1. **核心结论先行**：一句话说清该买该卖
2. **分持仓建议**：空仓者与持仓者给不同指引
3. **精确狙击点**：给具体价位；无数据时输出 `null`，不要编造
4. **检查清单可视化**：用 通过/警示/否决 标注
5. **风险优先级**：风险点要醒目
6. **可执行交易计划**：直接回答“现在买 / 等 / 卖”，并给出触发条件、止损、目标、时间止损；`trade_plan` 不是复述结论，而是可照着执行的规则
"""

    MARKET_GUIDES = {
        "cn": """
## 当前市场规则：A 股（CNY）

- 交易制度：有涨跌停、T+1、做空受限；涨停不可追。
- 乖离率（(现价 - MA5) / MA5 × 100%）：< 2% 安全区（最佳买点）；2%~5% 可小仓介入；> 5% 观望，不追。
- 趋势：`MA5 > MA10 > MA20` 多头排列是买入级信号的重要条件；均线发散上行优于粘合。
- 买点：缩量回踩 MA5 / MA10；临近涨停或成交约束明显时不给激进方案。
- 优先信息源：交易所 / 巨潮公告、业绩预告、监管动态、资金流、产业政策。
- 风险排查：减持公告、业绩预亏、监管处罚、大额解禁、产业政策变化。
""",
        "hk": """
## 当前市场规则：港股（HKD）

- 交易制度：无 A 股式涨跌停限制；注意流动性与南向资金方向。
- 乖离率（(现价 - MA5) / MA5 × 100%）：≤ 3% 安全区；3%~6% 可分批介入、优先等回踩；> 6% 不宜重仓追价（不等于看空）。
- 趋势：优先看多头排列，同时结合公告催化、回购分红与南向资金。
- 买点：MA10 / MA20 回踩、公告确认位、区间突破回踩、前低防守位；不强制 MA5 / MA10 模板。
- 优先信息源：HKEX 公告、业绩公告、配股/回购/分红安排。
- 风险排查：profit warning、配股摊薄、融资安排、回购变化、监管调查。
""",
        "us": """
## 当前市场规则：美股（USD）

- 交易制度：无涨跌停、T+0；注意财报窗口与盘前盘后跳空。
- 乖离率（(现价 - MA5) / MA5 × 100%）：≤ 5% 安全区；5%~8% 可持有或轻仓并提示回踩更优；> 8% 不宜追高，但强趋势不因此自动看空。
- 趋势：均线趋势重要，但必须结合财报、公司指引、估值与行业主线；不能仅因离均线较远就把强趋势股直接判成看空。
- 买点：breakout retest、earnings gap hold、前高突破确认、VWAP / 关键均线回踩；不强制 MA5 / MA10 模板。
- 优先信息源：SEC 披露、财报电话会、公司指引、诉讼与估值。
- 风险排查：SEC 重大披露、guidance 下修、insider selling、诉讼/调查、关税与出口管制。
- 中国政策：仅当 `China exposure` 明确时才提高权重。
""",
    }

    # 兼容旧引用：不传市场时按 A 股拼装
    SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + MARKET_GUIDES["cn"]

    @classmethod
    def build_system_prompt(cls, market: Optional[str]) -> str:
        """按市场拼装系统提示，避免把无关市场的规则喂给模型。"""
        key = (market or "").strip().lower()
        guide = cls.MARKET_GUIDES.get(key) or cls.MARKET_GUIDES["us"]
        return cls.SYSTEM_PROMPT_BASE + guide

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM Analyzer via LiteLLM.

        Args:
            api_key: Ignored (kept for backward compatibility). Keys are loaded from config.
        """
        self._router = None
        self._litellm_available = False
        self._config_signature = None  # Issue #152: detect config changes for hot-reload
        self._init_litellm()
        if not self._litellm_available:
            logger.warning("No LLM configured (LITELLM_MODEL / API keys), AI analysis will be unavailable")

    def _has_channel_config(self, config: Config) -> bool:
        """Check if multi-channel config (channels / YAML / legacy model_list) is active."""
        return bool(config.llm_model_list) and not all(
            e.get('model_name', '').startswith('__legacy_') for e in config.llm_model_list
        )

    def _init_litellm(self) -> None:
        """Initialize litellm Router from channels / YAML / legacy keys."""
        if litellm is None or Router is None:
            logger.warning("Analyzer LLM: litellm not installed")
            return
        config = get_config()
        litellm_model = config.litellm_model
        if not litellm_model:
            logger.warning("Analyzer LLM: LITELLM_MODEL not configured")
            return

        self._litellm_available = True

        # --- Channel / YAML path: build Router from pre-built model_list ---
        if self._has_channel_config(config):
            model_list = config.llm_model_list
            self._router = Router(
                model_list=model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            unique_models = list(dict.fromkeys(
                e['litellm_params']['model'] for e in model_list
            ))
            logger.info(
                f"Analyzer LLM: Router initialized from channels/YAML — "
                f"{len(model_list)} deployment(s), models: {unique_models}"
            )
            return

        # --- Legacy path: build Router for multi-key, or use single key ---
        keys = get_api_keys_for_model(litellm_model, config)

        if len(keys) > 1:
            # Build legacy Router for primary model multi-key load-balancing
            extra_params = extra_litellm_params(litellm_model, config)
            legacy_model_list = [
                {
                    "model_name": litellm_model,
                    "litellm_params": {
                        "model": litellm_model,
                        "api_key": k,
                        **extra_params,
                    },
                }
                for k in keys
            ]
            self._router = Router(
                model_list=legacy_model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            logger.info(
                f"Analyzer LLM: Legacy Router initialized with {len(keys)} keys "
                f"for {litellm_model}"
            )
        elif keys:
            logger.info(f"Analyzer LLM: litellm initialized (model={litellm_model})")
        else:
            logger.info(
                f"Analyzer LLM: litellm initialized (model={litellm_model}, "
                f"API key from environment)"
            )
        # Issue #152: record config signature for hot-reload detection
        self._config_signature = self._compute_config_signature(config)

    @staticmethod
    def _compute_config_signature(config: Config) -> tuple:
        """Compute a lightweight signature of LLM-relevant config (Issue #152).

        Used to detect runtime config changes (e.g. model switch via WebUI)
        so the Router can be re-initialized without restarting the process.
        """
        model_list = config.llm_model_list or []
        params_tuple = tuple(
            json.dumps(e.get('litellm_params', {}), sort_keys=True, default=str)
            for e in model_list
        )
        return (config.litellm_model or '', params_tuple)

    def _maybe_reinit_router(self) -> None:
        """Re-initialize Router if runtime config has changed (Issue #152).

        When users switch models via WebUI, Config.reset_instance() is called
        and get_config() returns the new config. However, the Router created
        during __init__ still holds the old model_list. This method detects
        the change and rebuilds the Router lazily on the next LLM call.
        """
        config = get_config()
        current_sig = self._compute_config_signature(config)
        previous_sig = getattr(self, "_config_signature", None)
        if previous_sig is not None and current_sig != previous_sig:
            logger.info(
                "[Issue #152] LLM config changed (old=%s, new=%s), re-initializing Router...",
                previous_sig[0], current_sig[0]
            )
            self._router = None
            self._litellm_available = False
            self._init_litellm()
            # _init_litellm updates _config_signature internally


    def is_available(self) -> bool:
        """Check if LiteLLM is properly configured with at least one API key."""
        return self._router is not None or self._litellm_available

    def _get_model_retry_policy(self, model: str, config: Config) -> Tuple[int, float]:
        """Return retry/backoff settings for the current model."""
        provider = (model.split("/", 1)[0] if "/" in model else model).strip().lower()
        if provider in {"gemini", "vertex_ai"}:
            return (
                max(0, int(getattr(config, "gemini_max_retries", 0) or 0)),
                max(0.0, float(getattr(config, "gemini_retry_delay", 0.0) or 0.0)),
            )
        return 0, 0.0

    def _is_retryable_llm_error(self, error: Exception) -> bool:
        """Best-effort detection for transient LLM failures worth retrying."""
        for attr_name in ("status_code", "code"):
            value = getattr(error, attr_name, None)
            if str(value).strip() == "429":
                return True

        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        if str(response_status).strip() == "429":
            return True

        text = str(error).lower()
        retryable_markers = (
            "429",
            "resource_exhausted",
            "rate limit",
            "ratelimit",
            "too many requests",
            "quota",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "deadline exceeded",
            "connection reset",
            "connection aborted",
        )
        return any(marker in text for marker in retryable_markers)

    def _dispatch_litellm_completion(
        self,
        model: str,
        call_kwargs: Dict[str, Any],
        *,
        config: Config,
        use_channel_router: bool,
        router_model_names: set,
    ) -> Any:
        """Dispatch a LiteLLM completion through router or direct fallback."""
        effective_kwargs, replaced_paths = sanitize_non_finite(dict(call_kwargs))
        if replaced_paths:
            logger.warning(
                "LLM 请求体包含 %d 处非法数值（nan/inf），已替换为 null：%s",
                len(replaced_paths),
                ", ".join(replaced_paths[:10]),
            )
        get_llm_rate_limiter().acquire(
            bucket=get_rate_limit_bucket(model),
            min_interval_seconds=getattr(config, "llm_min_interval", 0.0),
        )
        if use_channel_router and self._router and model in router_model_names:
            return self._router.completion(**effective_kwargs)
        if self._router and model == config.litellm_model and not use_channel_router:
            return self._router.completion(**effective_kwargs)

        keys = get_api_keys_for_model(model, config)
        if keys:
            effective_kwargs["api_key"] = keys[0]
        effective_kwargs.update(extra_litellm_params(model, config))
        return litellm.completion(**effective_kwargs)

    def _normalize_usage(self, usage_obj: Any) -> Dict[str, Any]:
        """Normalize usage objects from LiteLLM responses/chunks."""
        if not usage_obj:
            return {}

        def _get_value(key: str) -> int:
            if isinstance(usage_obj, dict):
                return int(usage_obj.get(key) or 0)
            return int(getattr(usage_obj, key, 0) or 0)

        return {
            "prompt_tokens": _get_value("prompt_tokens"),
            "completion_tokens": _get_value("completion_tokens"),
            "total_tokens": _get_value("total_tokens"),
        }

    def _extract_stream_text(self, chunk: Any) -> str:
        """Extract provider-agnostic text delta from a LiteLLM streaming chunk."""
        choices = chunk.get("choices") if isinstance(chunk, dict) else getattr(chunk, "choices", None)
        if not choices:
            return ""

        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)

        content: Any = None
        if isinstance(delta, dict):
            content = delta.get("content")
        elif isinstance(delta, str):
            content = delta
        elif delta is not None:
            content = getattr(delta, "content", None)

        if content is None:
            if isinstance(message, dict):
                content = message.get("content")
            elif message is not None:
                content = getattr(message, "content", None)

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

        return content if isinstance(content, str) else ""

    def _consume_litellm_stream(
        self,
        stream_response: Any,
        *,
        model: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        response_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Consume a LiteLLM stream into a single text payload."""
        chunks: List[str] = []
        usage: Dict[str, Any] = {}
        chars_received = 0
        next_emit_at = 1

        try:
            for chunk in stream_response:
                chunk_usage = chunk.get("usage") if isinstance(chunk, dict) else getattr(chunk, "usage", None)
                normalized_usage = self._normalize_usage(chunk_usage)
                if normalized_usage:
                    usage = normalized_usage

                # finish_reason="length" 表示输出被 max_tokens 截断，
                # 后面的 JSON 解析再"修复"也不等于完整分析。
                choices = chunk.get("choices") if isinstance(chunk, dict) else getattr(chunk, "choices", None)
                if choices:
                    finish_reason = (
                        choices[0].get("finish_reason")
                        if isinstance(choices[0], dict)
                        else getattr(choices[0], "finish_reason", None)
                    )
                    if finish_reason == "length" and response_meta is not None:
                        response_meta["truncated"] = True

                delta_text = self._extract_stream_text(chunk)
                if not delta_text:
                    continue

                chunks.append(delta_text)
                chars_received += len(delta_text)
                if progress_callback and chars_received >= next_emit_at:
                    progress_callback(chars_received)
                    next_emit_at = chars_received + 160
        except Exception as exc:
            raise _LiteLLMStreamError(
                f"{model} stream interrupted: {exc}",
                partial_received=chars_received > 0,
            ) from exc

        response_text = "".join(chunks).strip()
        if not response_text:
            raise _LiteLLMStreamError(
                f"{model} stream returned empty response",
                partial_received=chars_received > 0,
            )

        if progress_callback and chars_received > 0:
            progress_callback(chars_received)

        return response_text, usage

    def _call_litellm(
        self,
        prompt: str,
        generation_config: dict,
        *,
        stream: bool = False,
        stream_progress_callback: Optional[Callable[[int], None]] = None,
        system_prompt: Optional[str] = None,
        response_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Call LLM via litellm with fallback across configured models.

        When channels/YAML are configured, every model goes through the Router
        (which handles per-model key selection, load balancing, and retries).
        In legacy mode, the primary model may use the Router while fallback
        models fall back to direct litellm.completion().

        When stream=True, attempts streaming first for each model; falls back to
        non-stream on _LiteLLMStreamError. Retries for transient errors are
        preserved from the non-stream path.

        Args:
            prompt: User prompt text.
            generation_config: Dict with optional keys: temperature, max_output_tokens, max_tokens.
            stream: Whether to attempt streaming mode first.
            stream_progress_callback: Optional callback(chars_received) fired during stream.

        Returns:
            Tuple of (response text, model_used, usage).
        """
        self._maybe_reinit_router()  # Issue #152: hot-reload Router on config change
        config = get_config()
        max_tokens = (
            generation_config.get('max_output_tokens')
            or generation_config.get('max_tokens')
            or 8192
        )
        temperature = generation_config.get('temperature', 0.7)

        models_to_try = [config.litellm_model] + (config.litellm_fallback_models or [])
        models_to_try = [m for m in models_to_try if m]

        use_channel_router = self._has_channel_config(config)
        router_model_names = set(get_configured_llm_models(config.llm_model_list))

        last_error = None
        for model in models_to_try:
            max_retries, retry_delay = self._get_model_retry_policy(model, config)
            attempt = 0

            while True:
                try:
                    model_short = model.split("/")[-1] if "/" in model else model
                    effective_system_prompt = self.SYSTEM_PROMPT if system_prompt is None else system_prompt
                    messages = [{"role": "user", "content": prompt}]
                    if effective_system_prompt:
                        messages.insert(0, {"role": "system", "content": effective_system_prompt})
                    call_kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    extra = get_thinking_extra_body(model_short)
                    if extra:
                        call_kwargs["extra_body"] = extra

                    # --- Stream path (try first, fallback to non-stream on error) ---
                    if stream:
                        try:
                            stream_response = self._dispatch_litellm_completion(
                                model,
                                {**call_kwargs, "stream": True},
                                config=config,
                                use_channel_router=use_channel_router,
                                router_model_names=router_model_names,
                            )
                            response_text, usage = self._consume_litellm_stream(
                                stream_response,
                                model=model,
                                progress_callback=stream_progress_callback,
                                response_meta=response_meta,
                            )
                            return response_text, model, usage
                        except _LiteLLMStreamError as exc:
                            if exc.partial_received:
                                logger.warning(
                                    "[LiteLLM] %s stream failed after partial output, "
                                    "retrying non-stream for same model: %s",
                                    model, exc,
                                )
                            else:
                                logger.warning(
                                    "[LiteLLM] %s stream unavailable before first chunk, "
                                    "falling back to non-stream: %s",
                                    model, exc,
                                )
                            last_error = exc
                        except Exception as exc:
                            logger.warning(
                                "[LiteLLM] %s stream request failed, falling back to non-stream: %s",
                                model, exc,
                            )
                            last_error = exc

                    # --- Non-stream path ---
                    response = self._dispatch_litellm_completion(
                        model,
                        call_kwargs,
                        config=config,
                        use_channel_router=use_channel_router,
                        router_model_names=router_model_names,
                    )

                    if response and response.choices and response.choices[0].message.content:
                        usage = self._normalize_usage(getattr(response, "usage", None))
                        finish_reason = getattr(response.choices[0], "finish_reason", None)
                        if finish_reason == "length" and response_meta is not None:
                            response_meta["truncated"] = True
                        return (response.choices[0].message.content, model, usage)
                    raise ValueError("LLM returned empty response")

                except Exception as e:
                    last_error = e
                    if self._is_retryable_llm_error(e) and attempt < max_retries:
                        backoff_seconds = retry_delay * (2 ** attempt) if retry_delay > 0 else 0.0
                        logger.warning(
                            "[LiteLLM] %s transient failure, retry %s/%s in %.1fs: %s",
                            model,
                            attempt + 1,
                            max_retries,
                            backoff_seconds,
                            e,
                        )
                        if backoff_seconds > 0:
                            time.sleep(backoff_seconds)
                        attempt += 1
                        continue

                    if attempt > 0:
                        logger.warning(f"[LiteLLM] {model} failed after {attempt} retry(s): {e}")
                    else:
                        logger.warning(f"[LiteLLM] {model} failed: {e}")
                    break

        raise Exception(f"All LLM models failed (tried {len(models_to_try)} model(s)). Last error: {last_error}")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Public entry point for free-form text generation.

        External callers (e.g. MarketAnalyzer) must use this method instead of
        calling _call_litellm() directly or accessing private attributes such as
        _litellm_available, _router, _model, _use_openai, or _use_anthropic.

        Uses streaming mode to avoid long idle waits on high-latency models
        (e.g. market review with max_tokens=8192). Stream failures are
        automatically retried in non-stream mode by ``_call_litellm``.

        Args:
            prompt:      Text prompt to send to the LLM.
            max_tokens:  Maximum tokens in the response (default 2048).
            temperature: Sampling temperature (default 0.7).
            system_prompt: Optional system prompt override for non-dashboard tasks.

        Returns:
            Response text, or None if the LLM call fails (error is logged).
        """
        try:
            call_kwargs: Dict[str, Any] = {
                "generation_config": {"max_tokens": max_tokens, "temperature": temperature},
                "stream": True,
            }
            if system_prompt is not None:
                call_kwargs["system_prompt"] = system_prompt
            result = self._call_litellm(
                prompt,
                **call_kwargs,
            )
            if isinstance(result, tuple):
                text, model_used, usage = result
                persist_llm_usage(usage, model_used, call_type="market_review")
                return text
            return result
        except Exception as exc:
            logger.error("[generate_text] LLM call failed: %s", exc)
            return None

    def analyze(
        self,
        context: Dict[str, Any],
        news_context: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        stream_progress_callback: Optional[Callable[[int], None]] = None,
    ) -> AnalysisResult:
        """
        分析单只股票
        
        流程：
        1. 格式化输入数据（技术面 + 新闻）
        2. 调用 Gemini API（带重试和模型切换）
        3. 解析 JSON 响应
        4. 返回结构化结果
        
        Args:
            context: 从 storage.get_analysis_context() 获取的上下文数据
            news_context: 预先搜索的新闻内容（可选）
            
        Returns:
            AnalysisResult 对象
        """
        code = context.get('code', 'Unknown')
        config = get_config()
        
        # 请求前增加延时（防止连续请求触发限流）
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] 请求前等待 {request_delay:.1f} 秒...")
            time.sleep(request_delay)
        
        # 优先从上下文获取股票名称（由 main.py 传入）
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            # 备选：从 realtime 中获取
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                # 最后从映射表获取
                name = STOCK_NAME_MAP.get(code, f'股票{code}')
        
        # 如果模型不可用，返回默认结果
        if not self.is_available():
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary='AI 分析功能未启用（未配置 API Key）',
                risk_warning='请配置 LLM API Key（GEMINI_API_KEY/ANTHROPIC_API_KEY/OPENAI_API_KEY）后重试',
                success=False,
                error_message='LLM API Key 未配置',
                model_used=None,
            )
        
        try:
            # 格式化输入（包含技术面数据和新闻）
            # 交易计划只算一次：提示词、占位补全、回测点位都用同一份参考值，
            # 避免模型给不出价位时报告里出现空字段。
            analysis_context = dict(context) if isinstance(context, dict) else context
            trade_plan_reference = build_trade_plan(analysis_context)
            if isinstance(analysis_context, dict):
                analysis_context["trade_plan_reference"] = trade_plan_reference
            prompt = self._format_prompt(analysis_context, name, news_context)
            
            config = get_config()
            model_name = config.litellm_model or "unknown"
            logger.info(f"========== AI 分析 {name}({code}) ==========")
            logger.info(f"[LLM配置] 模型: {model_name}")
            logger.info(f"[LLM配置] Prompt 长度: {len(prompt)} 字符")
            logger.info(f"[LLM配置] 是否包含新闻: {'是' if news_context else '否'}")

            # 避免默认 INFO 日志泄露 LLM prompt/response 内容，仅在 DEBUG 级别输出预览与完整内容
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.debug(f"[LLM Prompt 预览]\n{prompt_preview}")
            logger.debug(f"=== 完整 Prompt ({len(prompt)}字符) ===\n{prompt}\n=== End Prompt ===")

            # 设置生成配置
            generation_config = {
                "temperature": config.llm_temperature,
                "max_output_tokens": 8192,
            }

            logger.info(f"[LLM调用] 开始调用 {model_name}...")

            def _emit_progress(progress: int, message: str) -> None:
                if progress_callback is None:
                    return
                try:
                    progress_callback(progress, message)
                except Exception as exc:
                    logger.debug("[analyzer] progress callback skipped: %s", exc)

            # 使用 litellm 调用（支持完整性校验重试）
            current_prompt = prompt
            retry_count = 0
            max_retries = config.report_integrity_retry if config.report_integrity_enabled else 0
            # 按市场拼装系统提示：只喂当前市场的交易规则
            market_guide = self.build_system_prompt(get_market_for_stock(code))

            while True:
                start_time = time.time()
                _emit_progress(68, f"{context.get('code', '')} 正在调用 LLM...")
                response_meta: Dict[str, Any] = {}
                response_text, model_used, llm_usage = self._call_litellm(
                    current_prompt,
                    generation_config,
                    stream=stream_progress_callback is not None,
                    stream_progress_callback=stream_progress_callback,
                    system_prompt=market_guide,
                    response_meta=response_meta,
                )
                elapsed = time.time() - start_time

                # 记录响应信息
                logger.info(
                    f"[LLM返回] {model_used} 响应成功, 耗时 {elapsed:.2f}s, 响应长度 {len(response_text)} 字符"
                )
                response_preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
                logger.debug(f"[LLM返回 预览]\n{response_preview}")
                logger.debug(
                    f"=== {model_used} 完整响应 ({len(response_text)}字符) ===\n{response_text}\n=== End Response ==="
                )

                # 解析响应
                result = self._parse_response(response_text, code, name)
                result.raw_response = response_text
                result.search_performed = bool(news_context)
                result.market_snapshot = self._build_market_snapshot(context)
                result.model_used = model_used
                # 先用规则计划补齐缺失点位，再做完整性校验：模型没写止损时
                # 不必为"必填字段"多跑一轮 LLM（规则位本身就是可执行值）。
                apply_trade_plan_fallback(result, trade_plan_reference)
                numeric_issues = validate_result_numbers(result, context)
                result.numeric_issues = numeric_issues
                result.numeric_consistency = not any(
                    issue.startswith("price_mismatch") for issue in numeric_issues
                )
                if numeric_issues:
                    logger.warning(
                        "[数值校验] %s(%s) 发现异常: %s",
                        name,
                        code,
                        numeric_issues,
                    )

                truncated = bool(response_meta.get("truncated"))
                if truncated:
                    # 被 max_tokens 截断的输出即使被 json_repair "修好"，
                    # 也可能是缺字段的残缺结果，必须标记为降级，不能当完整分析入库。
                    result.parse_mode = "truncated"
                    logger.warning(
                        "[LLM截断] %s 输出达到 max_tokens=%s，已标记为降级结果",
                        model_used,
                        generation_config.get("max_output_tokens"),
                    )

                # 内容完整性校验（可选）
                if not config.report_integrity_enabled and not truncated:
                    break
                pass_integrity, missing_fields = self._check_content_integrity(result)
                if pass_integrity and not truncated:
                    break
                if retry_count < max_retries:
                    if truncated and pass_integrity:
                        current_prompt = self._build_truncation_retry_prompt(prompt)
                    else:
                        current_prompt = self._build_integrity_retry_prompt(
                            prompt,
                            response_text,
                            missing_fields,
                        )
                    retry_count += 1
                    logger.info(
                        "[LLM完整性] 必填字段缺失 %s / 截断=%s，第 %d 次补全重试",
                        missing_fields,
                        truncated,
                        retry_count,
                    )
                else:
                    self._apply_placeholder_fill(result, missing_fields)
                    logger.warning(
                        "[LLM完整性] 必填字段缺失 %s / 截断=%s，已占位补全，不阻塞流程",
                        missing_fields,
                        truncated,
                    )
                    break

            persist_llm_usage(llm_usage, model_used, call_type="analysis", stock_code=code)

            # 模型没给价位时，用规则推算的计划补齐；已有内容不覆盖。
            if result and getattr(result, "success", False):
                apply_trade_plan_fallback(result, trade_plan_reference)

            logger.info(f"[LLM解析] {name}({code}) 分析完成: {result.trend_prediction}, 评分 {result.sentiment_score}")

            return result
            
        except Exception as e:
            logger.error(f"AI 分析 {name}({code}) 失败: {e}")
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction='震荡',
                operation_advice='持有',
                confidence_level='低',
                analysis_summary=f'分析过程出错: {str(e)[:100]}',
                risk_warning='分析失败，请稍后重试或手动分析',
                success=False,
                error_message=str(e),
                model_used=None,
            )
    
    def _format_trade_plan_block(self, context: Dict[str, Any]) -> str:
        """Render the deterministic reference trade plan for the prompt.

        The model must not invent entry/exit levels from scratch: it gets a
        rule-derived anchor and either keeps it or explains the deviation.
        """
        plan = context.get("trade_plan_reference")
        if not isinstance(plan, dict) or not plan.get("ready"):
            plan = build_trade_plan(context)
            if isinstance(context, dict):
                context["trade_plan_reference"] = plan
        if not plan.get("ready"):
            return ""

        rows = [
            ("买入触发", plan.get("buy_trigger")),
            ("观望条件", plan.get("wait_condition")),
            ("卖出/止损", plan.get("sell_trigger")),
            ("失效条件", plan.get("invalidation")),
            ("时间止损", plan.get("time_stop")),
        ]
        table = "\n".join(
            f"| {label} | {text} |"
            for label, text in rows
            if text
        )
        levels = []
        for label, key in (("入场区间", "entry_low"), ("止损", "stop_loss"), ("目标1", "target_1"), ("目标2", "target_2")):
            value = plan.get(key)
            if value is not None:
                levels.append(f"{label}={value}")
        level_text = " | ".join(levels)
        basis = str(plan.get("basis") or "").strip()
        return f"""
### 系统参考交易计划（规则推算，仅作价位锚点）
> 由 ATR14、均线与近 20 日高低点算出，不是最终结论。你要么采纳，要么在 `trade_plan` 里说明为什么调整。
> {level_text}{(' | ' + basis) if basis else ''}

| 场景 | 参考规则 |
|------|----------|
{table}

**硬性要求**：
- 输出的 `dashboard.battle_plan.trade_plan` 必须包含 `buy_trigger` / `wait_condition` / `sell_trigger` / `invalidation` / `time_stop`；
- 每个字段都要有价格或可验证信号，不能写“逢低”“注意风险”；
- 若你判断当前应观望，也必须写清“什么价格/什么信号出现才转为买入”。
"""

    def _format_fundamental_block(self, context: Dict[str, Any]) -> str:
        """把基本面聚合结果渲染成提示词表格。

        此前 `fundamental_context` 只写进快照、从未进入提示词：提示词要求模型
        做基本面分析，却不给营收增速、ROE、财报日期，模型只能写空话或编造。
        """
        fundamental = context.get('fundamental_context')
        if not isinstance(fundamental, dict):
            return ""

        def _data(name: str) -> Dict[str, Any]:
            block = fundamental.get(name)
            if not isinstance(block, dict):
                return {}
            payload = block.get('data')
            return payload if isinstance(payload, dict) else {}

        growth = _data('growth')
        earnings = _data('earnings')
        institution = _data('institution')
        valuation = _data('valuation')

        rows: List[str] = []

        def _add(label: str, value: Any, formatter) -> None:
            if value is None or value == "":
                return
            try:
                rendered = formatter(value)
            except (TypeError, ValueError):
                # 单个字段脏值不应打断整份提示词
                return
            rows.append(f"| {label} | {rendered} |")

        _add('营收同比', growth.get('revenue_yoy'), self._format_signed_percent)
        _add('净利润同比', growth.get('net_profit_yoy'), self._format_signed_percent)
        _add('ROE', growth.get('roe'), self._format_percent)
        _add('毛利率', growth.get('gross_margin'), self._format_percent)
        _add('营业利润率', growth.get('operating_margin'), self._format_percent)
        _add('净利率', growth.get('profit_margin'), self._format_percent)
        _add('PE(TTM)', valuation.get('pe_ratio'), lambda v: f"{float(v):.2f}")
        _add('Forward PE', valuation.get('forward_pe'), lambda v: f"{float(v):.2f}")
        _add('PB', valuation.get('pb_ratio'), lambda v: f"{float(v):.2f}")
        _add('PS(TTM)', valuation.get('ps_ratio'), lambda v: f"{float(v):.2f}")
        _add('PEG', valuation.get('peg_ratio'), lambda v: f"{float(v):.2f}")
        _add('EV/EBITDA', valuation.get('ev_to_ebitda'), lambda v: f"{float(v):.2f}")
        _add('总市值', valuation.get('market_cap'), self._format_compact_money)
        _add('自由现金流', institution.get('free_cashflow'), self._format_compact_money)
        _add('机构持股比例', institution.get('institutional_holding_pct'), self._format_percent)
        _add('分析师覆盖数', institution.get('analyst_coverage_count'), lambda v: f"{float(v):.0f}")
        _add('分析师目标均价', institution.get('target_mean_price'), self._format_price)
        _add('分析师共识', institution.get('recommendation_key'), lambda v: str(v))
        _add('近90天内部人申报(Form 4)', institution.get('insider_form4_count_90d'), lambda v: f"{int(v)} 笔")
        _add('近180天举牌/持股披露', institution.get('ownership_disclosure_count_180d'), lambda v: f"{int(v)} 笔")
        _add('Beta', institution.get('beta'), lambda v: f"{float(v):.2f}")
        _add('行业', (
            institution.get('sector'),
            institution.get('industry'),
        ) if institution.get('sector') or institution.get('industry') else None,
            lambda v: " / ".join(str(item) for item in v if item))
        _add('空头占流通股', institution.get('short_percent_float'), self._format_percent)
        _add('空头回补天数', institution.get('short_ratio_days'), lambda v: f"{float(v):.2f} 天")
        _add('空头数据日期', institution.get('short_interest_date'), lambda v: str(v))
        _add('空头股数环比', institution.get('short_change_pct'), self._format_signed_percent)
        _add('近90天内部人买入', institution.get('insider_buy_count_90d'), lambda v: f"{int(v)} 笔")
        _add('近90天内部人卖出', institution.get('insider_sell_count_90d'), lambda v: f"{int(v)} 笔")
        _add('内部人净买入额(90天, 负=净卖出)', institution.get('insider_net_value_90d'), self._format_compact_money)
        _add('流通股本', institution.get('float_shares'), self._format_compact_money)
        _add('52周高/低', (
            institution.get('week52_high'),
            institution.get('week52_low'),
        ) if institution.get('week52_high') is not None and institution.get('week52_low') is not None else None,
            lambda v: f"{float(v[0]):.2f} / {float(v[1]):.2f}")
        _add('负债权益比', institution.get('debt_to_equity'), lambda v: f"{float(v):.1f}%")
        _add('流动比率', institution.get('current_ratio'), lambda v: f"{float(v):.2f}")
        _add('EPS(TTM/Forward)', (
            earnings.get('eps_trailing'),
            earnings.get('eps_forward'),
        ) if earnings.get('eps_trailing') is not None and earnings.get('eps_forward') is not None else None,
            lambda v: f"{float(v[0]):.2f} / {float(v[1]):.2f}")
        _add('最近财报', earnings.get('latest_filing_form'), lambda v: str(v))
        _add('最近财报日期', earnings.get('latest_filing_date'), lambda v: str(v))
        _add('下次财报日', earnings.get('earnings_date'), lambda v: str(v))
        _add('除息日', earnings.get('ex_dividend_date'), lambda v: str(v))
        if earnings.get('latest_filing_url'):
            rows.append(f"| 最近财报原文 | {earnings['latest_filing_url']} |")

        if not rows:
            return ""

        status = str(fundamental.get('status') or 'unknown')
        return f"""
### 基本面数据（数据源聚合）
> 状态：{status}。下表只列出实际取到的字段；未列出的项目表示数据源没有返回，**不要据此推断或补全**。

| 指标 | 数值 |
|------|------|
{chr(10).join(rows)}
"""

    def _format_prompt(
        self, 
        context: Dict[str, Any], 
        name: str,
        news_context: Optional[str] = None
    ) -> str:
        """
        格式化分析提示词（决策仪表盘 v2.0）
        
        包含：技术指标、实时行情（量比/换手率）、筹码分布、趋势分析、新闻
        
        Args:
            context: 技术面数据上下文（包含增强数据）
            name: 股票名称（默认值，可能被上下文覆盖）
            news_context: 预先搜索的新闻内容
        """
        code = context.get('code', 'Unknown')
        
        # 优先使用上下文中的股票名称（从 realtime_quote 获取）
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'股票{code}')
            
        today = context.get('today', {})
        market_context_block = self._build_market_context_block(context)
        regime_block = self._build_market_regime_block(context)
        market_themes_block = self._build_market_themes_block(context)
        relative_strength_block = self._build_relative_strength_block(context)
        
        # Determine currency label based on market
        market = self._resolve_market_context(context).get("market", "cn")
        currency = {"cn": "元", "hk": "港元", "us": "美元"}.get(market, "元")

        # 盘中/盘前注入的是最新成交价，不是收盘价。标错会直接改变"突破了没有"的判断。
        quote_injected = bool(context.get('quote_injected'))
        is_intraday = bool(context.get('is_intraday'))
        price_label = "最新价（分析时点快照，非收盘价）" if quote_injected else "收盘价"
        volume_label = "成交量（盘中累计，非全天）" if is_intraday else "成交量"
        
        # ========== 构建决策仪表盘格式的输入 ==========
        prompt = f"""# 决策仪表盘分析请求

## 股票基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('analysis_date') or context.get('date', '未知')} |
| 行情数据日期 | {self._display_optional(today.get('date'), context.get('date', '未知'))} |

---

{market_context_block}

{regime_block}

{market_themes_block}

{relative_strength_block}

---

## 技术面数据

### 今日行情
| 指标 | 数值 |
|------|------|
| **数据日期** | **{self._display_optional(today.get('date'), '未知')}**（若落后于分析日期，说明数据源未更新，技术指标可能不代表当前市场） |
| {price_label} | {self._format_price(today.get('close'))} {currency} |
| 开盘价 | {self._format_price(today.get('open'))} {currency} |
| 最高价 | {self._format_price(today.get('high'))} {currency} |
| 最低价 | {self._format_price(today.get('low'))} {currency} |
| 涨跌幅 | {self._format_percent(today.get('pct_chg'))} |
| {volume_label} | {self._format_volume(today.get('volume'), market)} |
| 成交额 | {self._format_amount(today.get('amount'), currency, market)} |

### 均线系统（关键判断指标）
| 均线 | 数值 | 说明 |
|------|------|------|
| MA5 | {self._format_ma(today.get('ma5'))} | 短期趋势线 |
| MA10 | {self._format_ma(today.get('ma10'))} | 中短期趋势线 |
| MA20 | {self._format_ma(today.get('ma20'))} | 中期趋势线 |
| MA50 | {self._format_ma(today.get('ma50'))} | 机构中期趋势线（美股/港股常用） |
| MA60 | {self._format_ma(today.get('ma60'))} | 中期趋势线 |
| MA200 | {self._format_ma(today.get('ma200'))} | 长期趋势线（美股机构常用；N/A 表示历史数据不足 200 个交易日） |
| 均线形态 | {context.get('ma_status', '未知')} | 多头/空头/缠绕 |
"""
        
        # 添加实时行情数据（量比、换手率等）
        if 'realtime' in context:
            rt = context['realtime']
            session_label = context.get('session_label', '')
            raw_volume_ratio = rt.get('volume_ratio')
            if raw_volume_ratio is None and context.get('is_intraday'):
                volume_ratio_text = "N/A（盘中累计量，待收盘确认）"
            elif raw_volume_ratio is None:
                volume_ratio_text = "N/A"
            else:
                volume_ratio_text = self._display_optional(raw_volume_ratio)
            prompt += f"""
### 实时行情增强数据
> **数据时效**：{session_label}
> 说明：上方行情行若标注为"最新价"，是分析时点的最新成交价，不是当日收盘价；盘中价格还会变化，不要把它当成已确认的收盘/突破。量比已按已开市分钟数归一化；换手率为数据源口径；成交量若标注"盘中累计"则不可与全天均量直接比较。

| 指标 | 数值 | 解读 |
|------|------|------|
| 当前价格 | {self._format_price(rt.get('price'))} {currency} | |
| **量比** | **{volume_ratio_text}** | {rt.get('volume_ratio_desc', '')} |
| **换手率** | **{self._format_percent(rt.get('turnover_rate'))}** | |
| 市盈率(动态) | {self._display_optional(rt.get('pe_ratio'))} | |
| 市净率 | {self._display_optional(rt.get('pb_ratio'))} | |
| 总市值 | {self._format_amount(rt.get('total_mv'), currency, market)} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'), currency, market)} | |
| 60日涨跌幅 | {self._format_percent(rt.get('change_60d'))} | 中期表现 |
"""
        
        # 添加筹码分布数据
        if 'chip' in context:
            chip = context['chip']
            prompt += f"""
### 筹码分布数据（效率指标）
| 指标 | 数值 | 健康标准 |
|------|------|----------|
| **获利比例** | **{self._format_ratio(chip.get('profit_ratio'))}** | 70-90%时警惕 |
| 平均成本 | {self._format_price(chip.get('avg_cost'))} {currency} | 现价应高于5-15% |
| 90%筹码集中度 | {self._format_ratio(chip.get('concentration_90'))} | <15%为集中 |
| 70%筹码集中度 | {self._format_ratio(chip.get('concentration_70'))} | |
| 筹码状态 | {self._display_optional(chip.get('chip_status'), '未知')} | |

> 若上表出现 N/A，说明该市场或该标的不提供筹码数据（美股/港股通常没有），**不要据此推断筹码集中或获利情况**。
"""
        
        # 添加趋势分析结果（基于交易理念的预判）
        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            market = self._resolve_market_context(context).get("market", "cn")
            raw_bias = trend.get('bias_ma5')
            try:
                bias_value = float(raw_bias) if raw_bias is not None else None
            except (TypeError, ValueError):
                bias_value = None
            ma_alignment_hint = {
                "cn": "MA5>MA10>MA20 为买入级硬条件",
                "hk": "优先看趋势结构，再结合公告催化和关键支撑位",
                "us": "优先看趋势延续/突破确认，再结合财报、市场环境和估值",
            }.get(market, "按市场规则判断趋势结构")
            # 阈值必须和 stock_analyzer 里打分用的阈值同源，否则提示词说"超 5% 别追"、
            # 系统却按 15% 打满分，模型和评分会给出相反结论。
            from src.stock_analyzer import resolve_market_profile

            bias_profile = resolve_market_profile(code, stock_name)
            bias_threshold = max(
                float(get_config().bias_threshold),
                float(bias_profile["base_threshold_floor"]),
            )
            trend_status_text = str(trend.get('trend_status') or '')
            trend_strength_value = trend.get('trend_strength') or 0
            if trend_status_text == "强势多头" and trend_strength_value >= 70:
                bias_threshold *= float(bias_profile["strong_multiplier"])
            threshold_text = f"{bias_threshold:.1f}%"
            near_band_text = f"{float(bias_profile['near_ma_band']):.1f}%"
            if bias_value is None:
                # 乖离率缺失时不要输出"安全范围"这种判断
                bias_warning = "乖离率数据缺失，无法判断追价风险"
            elif bias_value > bias_threshold:
                bias_warning = f"超过 {threshold_text}，{bias_profile['high_bias_risk']}"
            elif bias_value > float(bias_profile['near_ma_band']):
                bias_warning = (
                    f"处于 {near_band_text}~{threshold_text} 警戒区，"
                    f"{bias_profile['slightly_high_reason']}"
                )
            else:
                bias_warning = f"在 {near_band_text} 安全区内，仍需结合量能与趋势确认"
            prompt += f"""
### 趋势分析预判（基于交易理念）
| 指标 | 数值 | 判定 |
|------|------|------|
| 趋势状态 | {self._display_optional(trend.get('trend_status'), '未知')} | |
| 均线排列 | {self._display_optional(trend.get('ma_alignment'), '未知')} | {ma_alignment_hint} |
| 趋势强度 | {self._display_optional(trend.get('trend_strength'))}/100 | |
| **乖离率(MA5)** | **{self._format_signed_percent(trend.get('bias_ma5'))}** | {bias_warning} |
| 乖离率(MA10) | {self._format_signed_percent(trend.get('bias_ma10'))} | |
| 量能状态 | {self._display_optional(trend.get('volume_status'), '未知')} | {trend.get('volume_trend', '')} |
| 5日量比（日K） | {self._format_volume_ratio_5d(trend.get('volume_ratio_5d'), context)} | 收盘后口径；盘中累计量不可直接与全天均量比较 |
| 系统信号 | {self._display_optional(trend.get('buy_signal'), '未知')} | |
| 系统评分 | {self._display_optional(trend.get('signal_score'))}/100 | |

#### 系统分析理由
**买入理由**：
{chr(10).join('- ' + r for r in trend.get('signal_reasons', ['无'])) if trend.get('signal_reasons') else '- 无'}

**风险因素**：
{chr(10).join('- ' + r for r in trend.get('risk_factors', ['无'])) if trend.get('risk_factors') else '- 无'}
"""

        # 基本面聚合结果（若数据源有返回）
        prompt += self._format_fundamental_block(context)

        # 系统参考交易计划（规则推算，给出可执行的价位锚点）
        prompt += self._format_trade_plan_block(context)
        
        # 添加昨日对比数据
        if 'yesterday' in context:
            volume_change = context.get('volume_change_ratio')
            price_change = context.get('price_change_ratio')
            prompt += f"""
### 量价变化
- 成交量较昨日变化：{self._display_optional(volume_change)} 倍
- 价格较昨日变化：{self._format_signed_percent(price_change)}
"""

        # 添加历史K线数据（用于中期趋势判断：触底反弹 vs 高位回调）
        history = context.get('history')
        if history and len(history) > 0:
            prompt += f"""
### 近{len(history)}日历史K线（用于判断中期趋势、触底反弹或高位回调）
| 日期 | 开盘 | 最高 | 最低 | 收盘 | 涨跌幅 |
|------|------|------|------|------|--------|
"""
            for bar in history:
                pct = bar.get('pct_chg')
                pct_str = f"{pct:+.2f}%" if pct is not None else "N/A"
                prompt += (
                    f"| {bar.get('date', '')} | {self._format_price(bar.get('open'))} "
                    f"| {self._format_price(bar.get('high'))} | {self._format_price(bar.get('low'))} "
                    f"| {self._format_price(bar.get('close'))} | {pct_str} |\n"
                )

        # 添加新闻搜索结果（重点区域）
        prompt += """
---

## 舆情情报
"""
        if news_context:
            prompt += f"""
以下是 **{stock_name}({code})** 近期检索到的情报（以条目自带日期为准，超出时效的不要当作最新消息），请严格区分“近期事件”和“背景资料”：
1. **风险警报**：减持、处罚、利空
2. **利好催化**：业绩、合同、政策
3. **业绩预期**：仅可引用近期财报、公司指引、业绩预告、业绩快报
4. **背景资料约束**：行业分析、百科、公司介绍、旧财年数据只能作背景参考，不得写成“最新消息”“近期催化”或当前业绩展望，除非条目明确给出近期日期且来源为官方披露或可信新闻

```
{news_context}
```
"""
        else:
            prompt += """
未搜索到该股票近期的相关新闻或有效情报。请主要依据技术面数据进行分析。
"""

        # 按实际缺失字段生成警告，而不是只依赖一个布尔标记
        missing_fields: List[str] = []
        if today.get('close') is None:
            missing_fields.append('最新收盘价')
        if today.get('ma5') is None or today.get('ma20') is None:
            missing_fields.append('均线（MA5/MA20）')
        if today.get('volume') is None:
            missing_fields.append('成交量')
        realtime_data = context.get('realtime') or {}
        if realtime_data:
            if realtime_data.get('volume_ratio') is None:
                missing_fields.append('量比')
            if realtime_data.get('turnover_rate') is None:
                missing_fields.append('换手率')
        chip_data = context.get('chip')
        if isinstance(chip_data, dict) and chip_data.get('profit_ratio') is None:
            missing_fields.append('筹码分布')

        if missing_fields or context.get('data_missing'):
            detail = '、'.join(missing_fields) if missing_fields else '部分实时行情'
            prompt += f"""
**数据缺失警告**
以下数据不可用：{detail}。
- 输出中对应字段一律填 `null`，文字里直接说明“数据缺失，无法判断”；
- **严禁编造**价格、均线、量比、换手率、筹码等任何数值；
- 不要因为某项数据缺失而抬高或压低评分，缺失按中性处理。
"""

        # 明确的输出要求
        prompt += f"""
---

## 分析任务

请为 **{stock_name}({code})** 生成【决策仪表盘】，严格按照 JSON 格式输出。
"""
        if context.get('is_index_etf'):
            prompt += """
> **指数/ETF 分析约束**：该标的为指数跟踪型 ETF 或市场指数。
> - 风险分析仅关注：**指数走势、跟踪误差、市场流动性**
> - 严禁将基金公司的诉讼、声誉、高管变动纳入风险警报
> - 业绩预期基于**指数成分股整体表现**，而非基金公司财报
> - `risk_alerts` 中不得出现基金管理人相关的公司经营风险

"""
        prompt += f"""
### 重要：输出正确的股票名称格式
正确的股票名称格式为“股票名称（股票代码）”，例如“贵州茅台（600519）”。
如果上方显示的股票名称为"股票{code}"或不正确，请在分析开头**明确输出该股票的正确中文全称**。

{self._build_market_focus_questions(context)}

### 决策仪表盘要求：
- **股票名称**：必须输出正确的中文全称（如"贵州茅台"而非"股票600519"）
- **核心结论**：一句话说清该买/该卖/该等
- **持仓分类建议**：空仓者怎么做 vs 持仓者怎么做
- **具体狙击点位**：买入价、止损价、目标价（精确到分）
- **检查清单**：每项用 通过/警示/否决 标记

请输出完整的 JSON 格式决策仪表盘。"""
        
        return prompt
    
    def _format_volume(self, volume: Optional[float], market: str = "cn") -> str:
        """格式化成交量显示（市场感知）"""
        if volume is None:
            return 'N/A'
        if market == "us":
            if volume >= 1e6:
                return f"{volume / 1e6:.2f}M shares"
            elif volume >= 1e3:
                return f"{volume / 1e3:.2f}K shares"
            else:
                return f"{volume:.0f} shares"
        # cn / hk 使用中文单位
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 亿股"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f} 万股"
        else:
            return f"{volume:.0f} 股"
    
    def _format_amount(self, amount: Optional[float], currency: str = "元", market: str = "cn") -> str:
        """格式化成交额显示（市场感知，支持多币种标签）"""
        if amount is None:
            return 'N/A'
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return 'N/A'
        if math.isnan(amount):
            return 'N/A'
        if market == "us":
            if amount >= 1e9:
                return f"${amount / 1e9:.2f}B"
            elif amount >= 1e6:
                return f"${amount / 1e6:.2f}M"
            elif amount >= 1e3:
                return f"${amount / 1e3:.2f}K"
            else:
                return f"${amount:.0f}"
        # cn / hk 使用中文单位
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 亿{currency}"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f} 万{currency}"
        else:
            return f"{amount:.0f} {currency}"

    def _format_compact_money(self, value: Optional[float]) -> str:
        """Compact money display for fundamentals (market-neutral units)."""
        if value is None:
            return 'N/A'
        try:
            amount = float(value)
        except (TypeError, ValueError):
            return 'N/A'
        if math.isnan(amount) or math.isinf(amount):
            return 'N/A'
        sign = "-" if amount < 0 else ""
        amount = abs(amount)
        if amount >= 1e12:
            return f"{sign}{amount / 1e12:.2f}T"
        if amount >= 1e9:
            return f"{sign}{amount / 1e9:.2f}B"
        if amount >= 1e6:
            return f"{sign}{amount / 1e6:.2f}M"
        if amount >= 1e3:
            return f"{sign}{amount / 1e3:.2f}K"
        return f"{sign}{amount:.0f}"

    def _format_volume_ratio_5d(self, value: Optional[float], context: Dict[str, Any]) -> str:
        """5-day volume ratio is a full-day metric; intraday it is misleading."""
        if isinstance(context, dict) and context.get("is_intraday"):
            return "盘中待收盘确认"
        try:
            ratio = float(value) if value is not None else None
        except (TypeError, ValueError):
            ratio = None
        if ratio is None or ratio <= 0:
            return "N/A"
        return f"{ratio:.2f}"

    def _format_percent(self, value: Optional[float]) -> str:
        """格式化百分比显示"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_signed_percent(self, value: Optional[float]) -> str:
        """带正负号的百分比，用于乖离率等方向性指标。"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):+.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_ratio(self, value: Optional[float]) -> str:
        """把 0–1 的比例格式化成百分比；缺失返回 N/A（不要用 0 冒充）。"""
        if value is None:
            return 'N/A'
        try:
            return f"{float(value) * 100:.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    @staticmethod
    def _display_optional(value: Any, default: str = 'N/A') -> str:
        """None / 空串显示为 N/A，避免把 "None" 直接写进提示词。"""
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    def _format_price(self, value: Optional[float]) -> str:
        """格式化价格显示"""
        if value is None:
            return 'N/A'
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 'N/A'
        if math.isnan(number):
            return 'N/A'
        return f"{number:.2f}"

    def _format_ma(self, value: Optional[float]) -> str:
        """均线数值：0 / 负数视为数据不足，不要显示成 0.00。"""
        if value is None:
            return 'N/A'
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 'N/A'
        return f"{number:.2f}" if number > 0 else 'N/A'

    @staticmethod
    def _market_label(market: str) -> str:
        return {
            "cn": "A股",
            "hk": "港股",
            "us": "美股",
        }.get(market, market or "未知市场")

    def _resolve_market_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return market metadata injected by pipeline, or infer a minimal fallback."""
        market_context = context.get("market_context")
        if isinstance(market_context, dict) and market_context.get("market"):
            return market_context
        market = get_market_for_stock(context.get("code", "")) or "cn"
        return {
            "market": market,
            "market_label": self._market_label(market),
        }

    def _build_market_regime_block(self, context: Dict[str, Any]) -> str:
        """Benchmark regime block (index trend / volatility)."""
        regime = context.get("market_regime")
        if not isinstance(regime, dict) or regime.get("close") is None:
            return ""
        code = regime.get("code") or "基准"
        ma200 = regime.get("ma200")
        above = regime.get("above_ma200")
        volatility = regime.get("volatility_20d_pct")
        recent = regime.get("recent_volatility_pct")
        lines = [
            "## 市场状态（基准指数）",
            f"- 基准：{code}",
            f"- 收盘：{self._display_optional(regime.get('close'))}",
            f"- 200日均线：{self._display_optional(ma200)}",
            f"- 位置：{'200日线上方' if above else '200日线下方'}",
            f"- 20日波动率：{self._format_percent(volatility)}",
        ]
        if recent is not None:
            lines.append(f"- 近10日波动率：{self._format_percent(recent)}")
        if above:
            lines.append(
                "- 结论：风险偏好正常（risk-on）。仍按个股结构判断，不因大盘强势放宽追高阈值。"
            )
        else:
            lines.append(
                "- 结论：风险偏好偏弱（risk-off）。买入信号应更保守：优先回踩确认、"
                "降低仓位，或等待基准回到均线上方。"
            )
        return "\n".join(lines)

    def _build_market_context_block(self, context: Dict[str, Any]) -> str:
        """Build a compact market-specific guidance block for the analysis prompt."""
        market_context = self._resolve_market_context(context)
        lines = [
            "## 市场规则与政策权重",
            f"- 当前市场：{market_context.get('market_label', '未知')}",
            f"- 官方优先源：{market_context.get('official_source_priority', '按市场默认源处理')}",
            f"- 分析主线：{market_context.get('analysis_focus', '按市场主线处理')}",
            f"- 政策口径：{market_context.get('policy_scope', '未指定')}",
        ]
        china_exposure = market_context.get("china_exposure")
        if isinstance(china_exposure, dict):
            lines.append(f"- China exposure：{china_exposure.get('level_label', '未知')}")
            lines.append(f"- 中国政策权重：{china_exposure.get('policy_weight_label', '谨慎使用')}")
            lines.append(f"- 判定依据：{china_exposure.get('reasoning', '未提供')}")
        return "\n".join(lines)

    def _build_market_themes_block(self, context: Dict[str, Any]) -> str:
        """Render the market-theme / sector-rotation snapshot for the prompt."""
        themes = context.get("market_themes")
        if not isinstance(themes, dict) or themes.get("status") not in ("ok", "partial"):
            return ""

        def _fmt(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):+.2f}%"

        lines = ["## 当前市场热点与板块轮动（日线，15 分钟缓存）"]
        for row in themes.get("benchmarks") or []:
            position = []
            if row.get("above_ma50") is not None:
                position.append("MA50 上方" if row.get("above_ma50") else "MA50 下方")
            if row.get("above_ma200") is not None:
                position.append("MA200 上方" if row.get("above_ma200") else "MA200 下方")
            lines.append(
                f"- 基准 {row.get('name')}({row.get('symbol')})："
                f"1日 {_fmt(row.get('return_1d'))} | 5日 {_fmt(row.get('return_5d'))} | "
                f"20日 {_fmt(row.get('return_20d'))}"
                + (f" | {'、'.join(position)}" if position else "")
            )

        def _row_text(rows: Any) -> str:
            parts = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                parts.append(
                    f"{row.get('name')} {row.get('symbol')} "
                    f"5日 {_fmt(row.get('return_5d'))}（1日 {_fmt(row.get('return_1d'))}）"
                )
            return "、".join(parts)

        leaders = _row_text(themes.get("leaders"))
        laggards = _row_text(themes.get("laggards"))
        if leaders:
            lines.append(f"- 领涨板块：{leaders}")
        if laggards:
            lines.append(f"- 领跌板块：{laggards}")
        lines.append(
            f"- 市场状态：{'risk-on' if themes.get('risk_on') else 'risk-off / 震荡'}"
        )
        lines.append(
            "- 使用方式：个股属于领涨板块时，回踩支撑可更积极；属于领跌板块时，"
            "突破必须同时满足放量与相对强弱转正，不能只因为个股均线好看就追。"
        )
        return "\n".join(lines)

    def _build_relative_strength_block(self, context: Dict[str, Any]) -> str:
        """Render 20d/60d excess return versus the benchmark."""
        rs = context.get("relative_strength")
        if not isinstance(rs, dict) or not rs:
            return ""

        def _fmt(value: Any) -> str:
            return "N/A" if value is None else f"{float(value):+.2f}%"

        lines = [f"## 相对强弱（个股 vs {rs.get('benchmark') or '基准'}）"]
        lines.append(
            f"- 20日：个股 {_fmt(rs.get('stock_return_20d'))} | "
            f"基准 {_fmt(rs.get('benchmark_return_20d'))} | "
            f"超额 {_fmt(rs.get('excess_20d'))}"
        )
        if rs.get("stock_return_60d") is not None or rs.get("excess_60d") is not None:
            lines.append(
                f"- 60日：个股 {_fmt(rs.get('stock_return_60d'))} | "
                f"基准 {_fmt(rs.get('benchmark_return_60d'))} | "
                f"超额 {_fmt(rs.get('excess_60d'))}"
            )
        if rs.get("label"):
            lines.append(f"- 判定：{rs.get('label')}")
        lines.append(
            "- 使用方式：超额为正且扩大 → 趋势可延续；价格新高但超额转负 → 警惕背离，追高需谨慎。"
        )
        return "\n".join(lines)

    def _build_market_focus_questions(self, context: Dict[str, Any]) -> str:
        """Return market-aware checklist questions for the LLM."""
        market = self._resolve_market_context(context).get("market", "cn")
        if market == "us":
            return """### 重点关注（必须明确回答）：
1. SEC/财报/公司指引是否出现超预期或下修？
2. `China exposure` 是高/中/低？中国政策是否应该成为核心因子，还是只能低权重引用？
3. 是否存在 insider selling、诉讼、监管调查、short interest 或 put/call 异动？
4. 估值与业绩增速是否匹配？消息面会不会改变估值锚？
5. 技术面是否支持趋势跟踪、breakout retest、earnings gap hold 或回踩再进？不要机械套用 A股 MA5/MA10 模式。
6. **财报窗口**：下次财报日是否在 3 个交易日内？若是，是否应该等财报后再买 / 降低仓位？
7. **现在到底该买、该等、还是该卖？** 写出触发价格/信号；如果结论是持有或观望，必须写清“什么条件下卖出、什么条件下加仓”。"""
        if market == "hk":
            return """### 重点关注（必须明确回答）：
1. HKEX 公告里是否有业绩、profit warning、配股、回购或分红等关键事项？
2. 公司对内地业务/客户/资产是否敏感？内地政策该给多高权重？
3. 是否存在融资摊薄、诉讼调查或明显的盈利下修风险？
4. 行业景气、南向资金和估值是否支持当前仓位判断？
5. 技术面是否支持区间突破、公告确认位、均线支撑或回测再进？不要机械套用 A股 MA5/MA10 模式。
6. **公告窗口**：业绩公告/财报日是否在 3 个交易日内？若是，是否应等公告后再建仓？
7. **现在到底该买、该等、还是该卖？** 写出触发价格/信号；如果结论是持有或观望，必须写清“什么条件下卖出、什么条件下加仓”。"""
        return """### 重点关注（必须明确回答）：
1. 是否满足 MA5>MA10>MA20 多头排列？
2. 当前乖离率是否在安全范围内（<5%）？超过5%必须标注“严禁追高”
3. 量能是否配合（缩量回调/放量突破）？
4. 筹码结构是否健康？
5. 消息面有无重大利空？（减持、处罚、业绩变脸、政策利空等）
6. 是否存在涨停/交易约束/T+1 等执行层限制，导致“理论买点”在实盘中未必可成交？
7. 是否为特殊交易规则股票？ST股涨跌停 ±5%，科创板/创业板 ±20%，北交所 ±30%——乖离率阈值需相应调整。
8. **现在到底该买、该等、还是该卖？** 写出触发价格/信号；如果结论是持有或观望，必须写清“什么条件下卖出、什么条件下加仓”。"""

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """构建当日行情快照（展示用）"""
        today = context.get('today', {}) or {}
        realtime = context.get('realtime', {}) or {}
        yesterday = context.get('yesterday', {}) or {}

        prev_close = yesterday.get('close')
        close = today.get('close')
        high = today.get('high')
        low = today.get('low')

        amplitude = None
        change_amount = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None
        if prev_close is not None and close is not None:
            try:
                change_amount = float(close) - float(prev_close)
            except (TypeError, ValueError):
                change_amount = None

        market = self._resolve_market_context(context).get("market", "cn")
        currency = {"cn": "元", "hk": "港元", "us": "美元"}.get(market, "元")
        snapshot = {
            "date": context.get('date', '未知'),
            "analysis_date": context.get('analysis_date') or context.get('date', '未知'),
            "close": self._format_price(close),
            "open": self._format_price(today.get('open')),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get('pct_chg')),
            "change_amount": self._format_price(change_amount),
            "amplitude": self._format_percent(amplitude),
            "volume": self._format_volume(today.get('volume'), market),
            "amount": self._format_amount(today.get('amount'), currency, market),
        }

        if realtime:
            snapshot.update({
                "price": self._format_price(realtime.get('price')),
                "volume_ratio": realtime.get('volume_ratio', 'N/A'),
                "turnover_rate": self._format_percent(realtime.get('turnover_rate')),
                "source": getattr(realtime.get('source'), 'value', realtime.get('source', 'N/A')),
            })

        return snapshot

    def _check_content_integrity(self, result: AnalysisResult) -> Tuple[bool, List[str]]:
        """Delegate to module-level check_content_integrity."""
        return check_content_integrity(result)

    def _build_integrity_complement_prompt(self, missing_fields: List[str]) -> str:
        """Build complement instruction for missing mandatory fields."""
        lines = ["### 补全要求：请在上方分析基础上补充以下必填内容，并输出完整 JSON："]
        for f in missing_fields:
            if f == "sentiment_score":
                lines.append("- sentiment_score: 0-100 综合评分")
            elif f == "operation_advice":
                lines.append("- operation_advice: 买入/加仓/持有/减仓/卖出/观望")
            elif f == "analysis_summary":
                lines.append("- analysis_summary: 综合分析摘要")
            elif f == "dashboard.core_conclusion.one_sentence":
                lines.append("- dashboard.core_conclusion.one_sentence: 一句话决策")
            elif f == "dashboard.intelligence.risk_alerts":
                lines.append("- dashboard.intelligence.risk_alerts: 风险警报列表（可为空数组）")
            elif f == "dashboard.battle_plan.sniper_points.stop_loss":
                lines.append("- dashboard.battle_plan.sniper_points.stop_loss: 止损价")
        return "\n".join(lines)

    def _build_integrity_retry_prompt(
        self,
        base_prompt: str,
        previous_response: str,
        missing_fields: List[str],
    ) -> str:
        """Build retry prompt using the previous response as the complement baseline."""
        complement = self._build_integrity_complement_prompt(missing_fields)
        previous_output = previous_response.strip()
        return "\n\n".join([
            base_prompt,
            "### 上一次输出如下，请在该输出基础上补齐缺失字段，并重新输出完整 JSON。不要省略已有字段：",
            previous_output,
            complement,
        ])

    def _build_truncation_retry_prompt(self, base_prompt: str) -> str:
        """Retry after hitting max_tokens: ask for a shorter, complete JSON."""
        return "\n\n".join([
            base_prompt,
            "### 上一次输出因长度上限被截断，JSON 不完整。请压缩各字段叙述，"
            "只输出一个完整、可解析的 JSON；必填键不能省略，未提供的数值填 null。",
        ])

    def _apply_placeholder_fill(self, result: AnalysisResult, missing_fields: List[str]) -> None:
        """Delegate to module-level apply_placeholder_fill."""
        apply_placeholder_fill(result, missing_fields)

    def _parse_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """
        解析 Gemini 响应（决策仪表盘版）
        
        尝试从响应中提取 JSON 格式的分析结果，包含 dashboard 字段
        如果解析失败，尝试智能提取或返回默认结果
        """
        try:
            # 清理响应文本：移除 markdown 代码块标记
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            
            # 尝试找到 JSON 内容
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                
                # 尝试修复常见的 JSON 问题
                json_str = self._fix_json_string(json_str)
                
                data = json.loads(json_str)

                # Schema validation (lenient: on failure, continue with raw dict)
                try:
                    AnalysisReportSchema.model_validate(data)
                except Exception as e:
                    logger.warning(
                        "LLM report schema validation failed, continuing with raw dict: %s",
                        str(e)[:100],
                    )

                # 提取 dashboard 数据
                dashboard = data.get('dashboard', None)

                # 优先使用 AI 返回的股票名称（如果原名称无效或包含代码）
                ai_stock_name = data.get('stock_name')
                if ai_stock_name and (name.startswith('股票') or name == code or 'Unknown' in name):
                    name = ai_stock_name

                # 解析所有字段，优先做一致性校验，避免出现“看空但买入”之类互相打架的结论
                raw_score = _coerce_sentiment_score_value(data.get('sentiment_score', 50))
                raw_trend_prediction = data.get('trend_prediction', '震荡')
                raw_operation_advice = data.get('operation_advice', '持有')
                raw_decision_type = data.get('decision_type', '')

                decision_signal = _canonical_decision_signal(raw_decision_type)
                operation_signal = _signal_from_operation_advice(raw_operation_advice)
                trend_signal = _signal_from_trend_prediction(raw_trend_prediction)
                score_signal = _signal_from_sentiment_score(raw_score)

                action_signal = decision_signal or operation_signal
                final_signal, has_conflict, hold_bias = _resolve_signal_state(
                    decision_signal=decision_signal,
                    operation_signal=operation_signal,
                    trend_signal=trend_signal,
                    score_signal=score_signal,
                    raw_trend_prediction=raw_trend_prediction,
                )

                normalized_score = raw_score
                if raw_score < 0 or raw_score > 100 or has_conflict or score_signal not in ("", final_signal):
                    normalized_score = _clamp_sentiment_score_to_signal_with_bias(
                        max(0, min(100, raw_score)),
                        final_signal,
                        hold_bias,
                    )

                normalized_trend_prediction = raw_trend_prediction
                canonical_trend = _canonical_trend_prediction_with_bias(final_signal, hold_bias)
                if (
                    has_conflict
                    or trend_signal not in ("", final_signal)
                    or (final_signal == "hold" and hold_bias != "neutral" and str(raw_trend_prediction).strip() != canonical_trend)
                ):
                    normalized_trend_prediction = canonical_trend

                normalized_operation_advice = raw_operation_advice
                canonical_advice = _canonical_operation_advice_with_bias(final_signal, hold_bias)
                if (
                    not action_signal
                    or has_conflict
                    or operation_signal not in ("", final_signal)
                    or (final_signal == "hold" and hold_bias != "neutral" and str(raw_operation_advice).strip() != canonical_advice)
                ):
                    normalized_operation_advice = canonical_advice

                dashboard = _normalize_dashboard_signal_fields(
                    dashboard,
                    final_signal,
                    has_conflict=has_conflict,
                    hold_bias=hold_bias,
                )
                
                return AnalysisResult(
                    code=code,
                    name=name,
                    # 核心指标
                    sentiment_score=normalized_score,
                    trend_prediction=normalized_trend_prediction,
                    operation_advice=normalized_operation_advice,
                    decision_type=final_signal,
                    confidence_level=_normalize_confidence_level(
                        data.get('confidence_level'), has_conflict=has_conflict
                    ),
                    # 决策仪表盘
                    dashboard=dashboard,
                    # 走势分析
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    # 技术面
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    # 基本面
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    # 情绪面/消息面
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    # 综合
                    analysis_summary=data.get('analysis_summary', '分析完成'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    # 元数据
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', '技术面数据'),
                    success=True,
                )
            else:
                # 没有找到 JSON，尝试从纯文本中提取信息
                logger.warning(f"无法从响应中提取 JSON，使用原始文本分析")
                return self._parse_text_response(response_text, code, name)
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}，尝试从文本提取")
            return self._parse_text_response(response_text, code, name)
    
    def _fix_json_string(self, json_str: str) -> str:
        """修复常见的 JSON 格式问题"""
        import re
        
        # 移除注释
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # 修复尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保布尔值是小写
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        
        # fix by json-repair when available
        if callable(repair_json):
            try:
                repaired = repair_json(json_str)
                if isinstance(repaired, str) and repaired.strip():
                    json_str = repaired
            except Exception:
                pass

        return json_str
    
    def _parse_text_response(
        self, 
        response_text: str, 
        code: str, 
        name: str
    ) -> AnalysisResult:
        """从纯文本响应中尽可能提取分析信息"""
        text = response_text or ""
        text_lower = text.lower()

        action_buy_keywords = ['买入', '加仓', '建仓', '试仓', 'buy']
        action_sell_keywords = ['卖出', '减仓', '止盈', '止损', 'sell']
        hold_keywords = ['持有', '观望', '等待', 'wait', 'hold']
        positive_keywords = ['看多', '上涨', '突破', '强势', '利好', 'bullish']
        negative_keywords = ['看空', '下跌', '跌破', '弱势', '利空', 'bearish']

        has_buy_action = any(keyword in text_lower for keyword in action_buy_keywords)
        has_sell_action = any(keyword in text_lower for keyword in action_sell_keywords)
        has_hold_action = any(keyword in text_lower for keyword in hold_keywords)

        explicit_action_signal = ""
        if has_buy_action and not has_sell_action:
            explicit_action_signal = "buy"
        elif has_sell_action and not has_buy_action:
            explicit_action_signal = "sell"
        elif has_hold_action or (has_buy_action and has_sell_action):
            explicit_action_signal = "hold"

        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)

        trend_signal = ""
        if positive_count >= negative_count + 2:
            trend_signal = "buy"
        elif negative_count >= positive_count + 2:
            trend_signal = "sell"
        elif positive_count or negative_count or has_hold_action:
            trend_signal = "hold"

        final_signal, _ = _resolve_consistent_signal(
            [explicit_action_signal, trend_signal]
        )
        hold_bias = "neutral"
        if final_signal == "hold":
            hold_bias = _derive_hold_bias([explicit_action_signal, trend_signal])

        if final_signal == "buy":
            sentiment_score = 62
        elif final_signal == "sell":
            sentiment_score = 38
        else:
            base_score = {
                "bullish": 56,
                "neutral": 50,
                "bearish": 44,
            }.get(hold_bias, 50)
            sentiment_score = _clamp_sentiment_score_to_signal_with_bias(
                base_score,
                "hold",
                hold_bias,
            )

        trend = _canonical_trend_prediction_with_bias(final_signal, hold_bias)
        advice = _canonical_operation_advice_with_bias(final_signal, hold_bias)
        decision_type = final_signal
        
        # 截取前500字符作为摘要
        summary = text[:500] if text else '无分析结果'
        
        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=sentiment_score,
            trend_prediction=trend,
            operation_advice=advice,
            decision_type=decision_type,
            confidence_level='低',
            analysis_summary=summary,
            key_points='JSON解析失败，已降级为保守文本提取结果，仅供参考',
            risk_warning='结构化解析失败，结论已按保守逻辑降级，建议结合原文与其他信息判断',
            raw_response=response_text,
            parse_mode="text_fallback",
            success=True,
        )
    
    def batch_analyze(
        self, 
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0
    ) -> List[AnalysisResult]:
        """
        批量分析多只股票
        
        注意：为避免 API 速率限制，每次分析之间会有延迟
        
        Args:
            contexts: 上下文数据列表
            delay_between: 每次分析之间的延迟（秒）
            
        Returns:
            AnalysisResult 列表
        """
        results = []
        
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                time.sleep(delay_between)
            
            result = self.analyze(context)
            results.append(result)
        
        return results


# 便捷函数
def get_analyzer() -> GeminiAnalyzer:
    """获取 LLM 分析器实例"""
    return GeminiAnalyzer()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 模拟上下文数据
    test_context = {
        'code': '600519',
        'date': '2026-01-09',
        'today': {
            'open': 1800.0,
            'high': 1850.0,
            'low': 1780.0,
            'close': 1820.0,
            'volume': 10000000,
            'amount': 18200000000,
            'pct_chg': 1.5,
            'ma5': 1810.0,
            'ma10': 1800.0,
            'ma20': 1790.0,
            'volume_ratio': 1.2,
        },
        'ma_status': '多头排列 📈',
        'volume_change_ratio': 1.3,
        'price_change_ratio': 1.5,
    }
    
    analyzer = GeminiAnalyzer()
    
    if analyzer.is_available():
        print("=== AI 分析测试 ===")
        result = analyzer.analyze(test_context)
        print(f"分析结果: {result.to_dict()}")
    else:
        print("Gemini API 未配置，跳过测试")

# -*- coding: utf-8 -*-
"""Deterministic trade-plan reference.

The LLM is good at synthesising news and technical context, but it is
inconsistent at turning "trend is bullish" into concrete entry / stop / target
levels.  This module derives a reference plan from data the system already
has (daily K-lines, ATR, moving averages, recent swing high/low) so that:

1. the prompt always carries actionable price levels to anchor the model;
2. the report always answers "when to buy / when to wait / when to sell",
   even if the model omits the trade-plan fields;
3. the backtest receives stop-loss / take-profit levels for every buy verdict.

Everything here is pure computation: no network, no LLM, no database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _round_price(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)


def _empty_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip()
        return normalized in ("", "-", "—", "N/A", "n/a", "NA", "null", "None", "待补充", "无")
    return False


def compute_atr(history: Sequence[Dict[str, Any]], period: int = 14) -> Optional[float]:
    """Average true range over the last *period* bars.

    Returns ``None`` when fewer than two usable bars are available.
    """
    bars: List[Tuple[float, float, float]] = []
    for bar in history or []:
        if not isinstance(bar, dict):
            continue
        high = _to_float(bar.get("high"))
        low = _to_float(bar.get("low"))
        close = _to_float(bar.get("close"))
        if high is None or low is None or close is None:
            continue
        bars.append((high, low, close))

    if len(bars) < 2:
        return None

    true_ranges: List[float] = []
    for (_, prev_low, prev_close), (high, low, close) in zip(bars, bars[1:]):
        true_ranges.append(
            max(high - low, abs(high - prev_close), abs(low - prev_close))
        )

    window = true_ranges[-max(1, period):]
    if not window:
        return None
    return sum(window) / len(window)


def _swing_levels(
    history: Sequence[Dict[str, Any]],
    lookback: int = 20,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (recent swing high, recent swing low)."""
    highs: List[float] = []
    lows: List[float] = []
    for bar in list(history or [])[-max(2, lookback):]:
        if not isinstance(bar, dict):
            continue
        high = _to_float(bar.get("high"))
        low = _to_float(bar.get("low"))
        if high is not None:
            highs.append(high)
        if low is not None:
            lows.append(low)
    swing_high = max(highs) if highs else None
    swing_low = min(lows) if lows else None
    return swing_high, swing_low


def _fallback_atr_pct(market: str) -> float:
    return {"us": 0.025, "hk": 0.02}.get(market, 0.02)


@dataclass
class TradePlan:
    """Reference entry/exit plan derived from price data."""

    ready: bool = False
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    risk_reward: Optional[float] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    buy_trigger: str = ""
    wait_condition: str = ""
    sell_trigger: str = ""
    invalidation: str = ""
    time_stop: str = ""
    basis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_trade_plan(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a reference trade plan from an analysis context.

    The context is the same dict passed to ``GeminiAnalyzer._format_prompt``:
    ``today`` (latest OHLC), ``history`` (previous K-lines) and an optional
    ``trend_analysis`` block with moving averages.
    """
    context = context if isinstance(context, dict) else {}
    today = context.get("today") if isinstance(context.get("today"), dict) else {}
    realtime = context.get("realtime") if isinstance(context.get("realtime"), dict) else {}
    trend = context.get("trend_analysis") if isinstance(context.get("trend_analysis"), dict) else {}
    history = context.get("history") if isinstance(context.get("history"), list) else []

    code = str(context.get("code") or "")
    market = str(context.get("market") or "").strip().lower()
    if market not in ("cn", "hk", "us"):
        try:
            from src.core.trading_calendar import get_market_for_stock

            market = get_market_for_stock(code) or "cn"
        except Exception:
            market = "cn"

    close = _to_float(realtime.get("price")) or _to_float(today.get("close"))
    if close is None or close <= 0:
        return TradePlan(
            basis="缺少最新价格，无法推算交易计划",
        ).to_dict()

    ma5 = _to_float(today.get("ma5")) or _to_float(trend.get("ma5"))
    ma10 = _to_float(today.get("ma10")) or _to_float(trend.get("ma10"))
    ma20 = _to_float(today.get("ma20")) or _to_float(trend.get("ma20"))
    ma50 = _to_float(today.get("ma50")) or _to_float(trend.get("ma50"))

    atr = compute_atr(history, 14)
    if atr is None or atr <= 0:
        atr = close * _fallback_atr_pct(market)
    atr_pct = round(atr / close * 100.0, 2)

    swing_high, swing_low = _swing_levels(history, 20)

    supports = [value for value in (ma10, ma20, ma50, swing_low) if value and value < close * 0.998]
    support = max(supports) if supports else None
    resistance = swing_high if swing_high and swing_high > close * 1.002 else None
    if resistance is None and ma50 and ma50 > close * 1.02:
        resistance = ma50

    trend_status = str(trend.get("trend_status") or "")
    bullish = (
        "多头" in trend_status
        or (ma20 is not None and close >= ma20)
    )
    if not trend_status and ma20 is None:
        bullish = False

    # --- entry zone -----------------------------------------------------
    if bullish and ma20 is not None:
        anchor = ma10 if (ma10 and ma10 < close) else ma20
        entry_low = max(anchor, close - 1.0 * atr)
        entry_high = min(close, anchor + 0.5 * atr)
        if entry_high <= entry_low:
            entry_low = close - 0.8 * atr
            entry_high = close - 0.1 * atr
        breakout_level = resistance if resistance else close + atr
        buy_trigger = (
            f"回踩 {entry_low:.2f}–{entry_high:.2f} 区间止跌（收阳或缩量）后分批买入；"
            f"若放量突破 {breakout_level:.2f}，等回踩不破 {breakout_level:.2f} 再买"
        )
        wait_condition = (
            f"现价高于 {entry_high + 0.8 * atr:.2f}（追高区），"
            f"或回踩跌破 {support:.2f}（若无支撑则跌破 {entry_low - atr:.2f}）未收回 → 继续观望"
            if support
            else f"现价高于 {entry_high + 0.8 * atr:.2f} 时不追，等回踩到买入区间"
        )
    elif ma20 is not None:
        entry_low = ma20 - 0.3 * atr
        entry_high = ma20 + 0.3 * atr
        buy_trigger = (
            f"先不买；等收盘重新站上 MA20（{ma20:.2f}）并维持 2 个交易日，"
            f"回踩 {entry_low:.2f}–{entry_high:.2f} 不破再买"
        )
        wait_condition = (
            f"价格在 MA20（{ma20:.2f}）下方运行时一律观望，"
            f"不接下跌；只有站回 MA20 且量能不放大才重新评估"
        )
    else:
        entry_low = close - 0.8 * atr
        entry_high = close - 0.1 * atr
        buy_trigger = (
            f"均线数据不足；等回踩 {entry_low:.2f}–{entry_high:.2f} 且出现止跌信号后再买，"
            f"或放量突破 {swing_high:.2f} 后回踩确认"
            if swing_high
            else f"均线数据不足；等回踩 {entry_low:.2f}–{entry_high:.2f} 出现止跌信号后再买"
        )
        wait_condition = "趋势结构不清晰，先观望，等价格与均线关系明确"

    entry_mid = (entry_low + entry_high) / 2.0

    # --- stop / targets -------------------------------------------------
    stop_from_support = (support - 0.3 * atr) if support else None
    stop = min(entry_low - 0.8 * atr, stop_from_support) if stop_from_support else entry_low - 0.8 * atr
    if stop >= entry_low:
        stop = entry_low - 0.5 * atr
    # Never let a single stop exceed ~12% of entry: that is a position-sizing
    # problem, not a trading signal.
    if entry_mid > 0 and (entry_mid - stop) / entry_mid > 0.12:
        stop = entry_mid * 0.88

    risk = max(entry_mid - stop, atr * 0.5)
    target_1 = entry_mid + 2.0 * risk
    target_2 = entry_mid + 3.0 * risk
    if resistance and resistance > entry_mid:
        target_1 = min(target_1, max(resistance, entry_mid + 1.2 * risk))
    risk_reward = round((target_1 - entry_mid) / risk, 2) if risk > 0 else None

    ma_reference = ma20 if ma20 is not None else (ma10 or close)
    sell_trigger = (
        f"持仓者：收盘跌破 {stop:.2f} 无条件止损；"
        f"收盘跌破 MA20（{ma_reference:.2f}）且 3 个交易日未收回 → 减仓/离场；"
        f"触及 {target_1:.2f} 减半，{target_2:.2f} 再减"
    )
    invalidation = (
        f"论点失效：放量跌破 {support:.2f}（近期支撑）；财报不及预期或 guidance 下修；"
        f"出现监管调查、做空报告等重大利空"
        if support
        else "论点失效：放量跌破近期低点；财报不及预期或 guidance 下修；出现重大利空"
    )
    time_stop = (
        f"买入后 10 个交易日仍未突破 {target_1:.2f}，且收盘跌破 MA10"
        f"（{ma10:.2f}）→ 退出，不占用资金"
        if ma10
        else f"买入后 10 个交易日仍未突破 {target_1:.2f} → 重新评估，避免长期占用资金"
    )
    basis = f"ATR14={atr:.2f}（{atr_pct:.1f}%）"
    if ma20:
        basis += f" | MA20={ma20:.2f}"
    if support:
        basis += f" | 支撑≈{support:.2f}"
    if resistance:
        basis += f" | 压力≈{resistance:.2f}"

    return TradePlan(
        ready=True,
        entry_low=_round_price(entry_low),
        entry_high=_round_price(entry_high),
        stop_loss=_round_price(stop),
        target_1=_round_price(target_1),
        target_2=_round_price(target_2),
        risk_reward=risk_reward,
        atr=_round_price(atr),
        atr_pct=atr_pct,
        buy_trigger=buy_trigger,
        wait_condition=wait_condition,
        sell_trigger=sell_trigger,
        invalidation=invalidation,
        time_stop=time_stop,
        basis=basis,
    ).to_dict()


def apply_trade_plan_fallback(
    result: Any,
    plan: Optional[Dict[str, Any]],
) -> Any:
    """Fill missing trade-plan / sniper fields from the reference plan.

    Only missing or empty values are filled; anything the model produced is
    preserved verbatim so the report still reflects the model's own judgement.
    """
    if result is None or not isinstance(plan, dict) or not plan.get("ready"):
        return result

    dashboard = getattr(result, "dashboard", None)
    if not isinstance(dashboard, dict):
        dashboard = {}
    battle = dashboard.get("battle_plan")
    if not isinstance(battle, dict):
        battle = {}

    trade_plan = battle.get("trade_plan")
    if not isinstance(trade_plan, dict):
        trade_plan = {}

    for key in ("buy_trigger", "wait_condition", "sell_trigger", "invalidation", "time_stop"):
        if _empty_text(trade_plan.get(key)) and plan.get(key):
            trade_plan[key] = plan[key]
    for key in ("entry_low", "entry_high", "stop_loss", "target_1", "target_2", "risk_reward"):
        if trade_plan.get(key) is None and plan.get(key) is not None:
            trade_plan[key] = plan[key]
    battle["trade_plan"] = trade_plan

    sniper = battle.get("sniper_points")
    if not isinstance(sniper, dict):
        sniper = {}
    if _empty_text(sniper.get("stop_loss")) and plan.get("stop_loss") is not None:
        sniper["stop_loss"] = f"{plan['stop_loss']:.2f}"
    if _empty_text(sniper.get("take_profit")) and plan.get("target_1") is not None:
        sniper["take_profit"] = f"{plan['target_1']:.2f}"

    decision_type = str(getattr(result, "decision_type", "") or "").strip().lower()
    if decision_type == "buy":
        if _empty_text(sniper.get("ideal_buy")) and plan.get("entry_low") is not None:
            sniper["ideal_buy"] = f"{plan['entry_low']:.2f}–{plan['entry_high']:.2f}"
        if _empty_text(sniper.get("secondary_buy")) and plan.get("entry_low") is not None:
            secondary = max(
                float(plan["entry_low"]) - float(plan.get("atr") or 0.0),
                float(plan.get("entry_low") or 0.0) * 0.9,
            )
            sniper["secondary_buy"] = f"{secondary:.2f}"

    battle["sniper_points"] = sniper
    dashboard["battle_plan"] = battle
    result.dashboard = dashboard
    return result


def render_trade_plan_lines(dashboard: Optional[Dict[str, Any]]) -> List[str]:
    """Render the trade plan as markdown lines for reports."""
    if not isinstance(dashboard, dict):
        return []
    battle = dashboard.get("battle_plan")
    plan = battle.get("trade_plan") if isinstance(battle, dict) else None
    if not isinstance(plan, dict):
        return []

    rows = [
        ("🟢 买入", plan.get("buy_trigger")),
        ("🟡 观望", plan.get("wait_condition")),
        ("🔴 卖出", plan.get("sell_trigger")),
        ("⛔ 失效", plan.get("invalidation")),
    ]
    rows = [(label, str(text).strip()) for label, text in rows if not _empty_text(text)]
    if not rows:
        return []

    lines = [
        "**🧭 交易计划（什么时候买 / 卖 / 等）**",
        "",
        "| 场景 | 触发条件 |",
        "|------|----------|",
    ]
    for label, text in rows:
        lines.append(f"| {label} | {text} |")
    lines.append("")

    if not _empty_text(plan.get("time_stop")):
        lines.append(f"- ⏱ **时间止损**: {str(plan['time_stop']).strip()}")
    risk_reward = _to_float(plan.get("risk_reward"))
    if risk_reward is not None:
        lines.append(f"- ⚖️ **盈亏比**: 1 : {risk_reward:g}")
    lines.append("")
    return lines

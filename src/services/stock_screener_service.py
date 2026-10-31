# -*- coding: utf-8 -*-
"""Hot-sector stock screener.

Flow:
1. Take today's leading sector ETFs from :mod:`market_themes_service`.
2. Pull the top constituents of those sectors (Yahoo sector pages, with a
   built-in liquid large-cap fallback when Yahoo rate-limits).
3. Score each stock on 15/30-day momentum, relative strength versus its own
   sector, setup quality (pullback / breakout / extended) and liquidity.
4. Return an actionable watchlist with a rule-derived entry/stop/target.

The screener is deliberately "early positioning" oriented: it rewards leaders
that are consolidating or pulling back on light volume, and penalises stocks
that are already far above MA20.

No paid data source is used.  Everything fails open: if Yahoo is unavailable,
the built-in fallback universe still lets the screener run.
"""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

from src.core.trade_plan import build_trade_plan
from src.services.screener_learning_service import get_screener_learning_service

logger = logging.getLogger(__name__)

# SPDR sector ETF -> Yahoo sector key / display name
SECTOR_KEYS: Dict[str, Tuple[str, str]] = {
    "XLK": ("technology", "科技"),
    "XLC": ("communication-services", "通信服务"),
    "XLY": ("consumer-cyclical", "可选消费"),
    "XLP": ("consumer-defensive", "必需消费"),
    "XLF": ("financial-services", "金融"),
    "XLV": ("healthcare", "医疗"),
    "XLI": ("industrials", "工业"),
    "XLB": ("basic-materials", "原材料"),
    "XLE": ("energy", "能源"),
    "XLU": ("utilities", "公用事业"),
    "XLRE": ("real-estate", "房地产"),
}

# Nasdaq screener sector names (broader universe than Yahoo's top-12 list).
NASDAQ_SECTOR: Dict[str, str] = {
    "XLK": "Technology",
    "XLC": "Telecommunications",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLF": "Finance",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Basic Materials",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
}

EARNINGS_BLACKOUT_DAYS = 3
MIN_MARKET_CAP = 300_000_000
MIN_REVENUE_GROWTH = -0.10
MIN_PROFIT_MARGIN = -0.10
MAX_DEBT_TO_EQUITY = 500.0
MAX_CORRELATION = 0.85
CANDIDATES_PER_SECTOR = 16

# Liquid large-cap fallback.  Used when yfinance's sector endpoint is rate
# limited; keeps the screener deterministic instead of returning nothing.
FALLBACK_UNIVERSE: Dict[str, List[Tuple[str, str]]] = {
    "XLK": [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
        ("AVGO", "Broadcom"), ("ORCL", "Oracle"), ("CRM", "Salesforce"),
        ("AMD", "AMD"), ("ADBE", "Adobe"),
    ],
    "XLC": [
        ("GOOGL", "Alphabet"), ("META", "Meta"), ("NFLX", "Netflix"),
        ("DIS", "Disney"), ("TMUS", "T-Mobile"), ("VZ", "Verizon"),
        ("T", "AT&T"), ("CMCSA", "Comcast"),
    ],
    "XLY": [
        ("AMZN", "Amazon"), ("TSLA", "Tesla"), ("HD", "Home Depot"),
        ("MCD", "McDonald's"), ("NKE", "Nike"), ("SBUX", "Starbucks"),
        ("LOW", "Lowe's"), ("BKNG", "Booking"),
    ],
    "XLP": [
        ("WMT", "Walmart"), ("COST", "Costco"), ("PG", "P&G"),
        ("KO", "Coca-Cola"), ("PEP", "Pepsi"), ("PM", "Philip Morris"),
        ("MO", "Altria"), ("CL", "Colgate"),
    ],
    "XLF": [
        ("BRK-B", "Berkshire"), ("JPM", "JPMorgan"), ("V", "Visa"),
        ("MA", "Mastercard"), ("BAC", "Bank of America"), ("WFC", "Wells Fargo"),
        ("GS", "Goldman Sachs"), ("MS", "Morgan Stanley"),
    ],
    "XLV": [
        ("LLY", "Eli Lilly"), ("UNH", "UnitedHealth"), ("JNJ", "J&J"),
        ("ABBV", "AbbVie"), ("MRK", "Merck"), ("PFE", "Pfizer"),
        ("TMO", "Thermo Fisher"), ("AMGN", "Amgen"),
    ],
    "XLI": [
        ("GE", "GE Aerospace"), ("CAT", "Caterpillar"), ("RTX", "RTX"),
        ("HON", "Honeywell"), ("UPS", "UPS"), ("BA", "Boeing"),
        ("LMT", "Lockheed Martin"), ("DE", "Deere"),
    ],
    "XLB": [
        ("LIN", "Linde"), ("SHW", "Sherwin-Williams"), ("FCX", "Freeport"),
        ("APD", "Air Products"), ("ECL", "Ecolab"), ("NEM", "Newmont"),
        ("NUE", "Nucor"), ("DOW", "Dow"),
    ],
    "XLE": [
        ("XOM", "Exxon Mobil"), ("CVX", "Chevron"), ("COP", "ConocoPhillips"),
        ("SLB", "SLB"), ("EOG", "EOG"), ("MPC", "Marathon Petroleum"),
        ("PSX", "Phillips 66"), ("OXY", "Occidental"),
    ],
    "XLU": [
        ("NEE", "NextEra"), ("SO", "Southern"), ("DUK", "Duke"),
        ("CEG", "Constellation"), ("AEP", "American Electric"), ("SRE", "Sempra"),
        ("D", "Dominion"), ("EXC", "Exelon"),
    ],
    "XLRE": [
        ("PLD", "Prologis"), ("AMT", "American Tower"), ("EQIX", "Equinix"),
        ("WELL", "Welltower"), ("SPG", "Simon Property"), ("O", "Realty Income"),
        ("PSA", "Public Storage"), ("DLR", "Digital Realty"),
    ],
}

_STATE_LABELS = {
    "pullback": "提前布局",
    "breakout": "突破观察",
    "watch": "等待信号",
    "extended": "不追高",
    "weak": "回避",
}


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


def _mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _return_pct(closes: Sequence[float], days: int) -> Optional[float]:
    if len(closes) <= days or days <= 0:
        return None
    base = closes[-1 - days]
    if base <= 0:
        return None
    return round((closes[-1] / base - 1.0) * 100.0, 2)


def _return_pct_at(closes: Sequence[float], index: int, days: int) -> Optional[float]:
    if index < days or index >= len(closes) or days <= 0:
        return None
    base = closes[index - days]
    if base <= 0:
        return None
    return (closes[index] / base - 1.0) * 100.0


def _daily_returns(closes: Sequence[float], window: int = 60) -> List[float]:
    values = [float(value) for value in closes[-(window + 1):] if value]
    if len(values) < 3:
        return []
    return [
        (values[index] / values[index - 1] - 1.0)
        for index in range(1, len(values))
        if values[index - 1]
    ]


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    size = min(len(left), len(right))
    if size < 10:
        return 0.0
    a = list(left[-size:])
    b = list(right[-size:])
    mean_a = sum(a) / size
    mean_b = sum(b) / size
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        # Constant return streams have no defined correlation; treat identical
        # streams as perfectly correlated so duplicate exposure is still caught.
        return 1.0 if a == b else 0.0
    return cov / math.sqrt(var_a * var_b)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _atr_pct(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> Optional[float]:
    if len(closes) < 2:
        return None
    true_ranges = []
    for index in range(1, len(closes)):
        high = highs[index]
        low = lows[index]
        prev_close = closes[index - 1]
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    window = true_ranges[-max(1, period):]
    if not window or closes[-1] <= 0:
        return None
    return round((sum(window) / len(window)) / closes[-1] * 100.0, 2)


@dataclass
class Candidate:
    symbol: str
    name: str
    sector_etf: str
    sector_name: str
    sector_rank: int


def score_metrics(metrics: Dict[str, Any], sector_rank: int, risk_on: bool = True) -> Dict[str, Any]:
    """Score one candidate and classify its setup.

    Pure function so the ranking can be tested without network access.
    """
    rs15 = _to_float(metrics.get("rs_15d"))
    rs30 = _to_float(metrics.get("rs_30d")) or 0.0
    if rs15 is None:
        # Without a sector benchmark the "relative strength" claim is unfounded.
        return {"score": 0.0, "state": "weak", "reason": "缺少板块基准，无法计算相对强弱"}
    dist_ma20 = _to_float(metrics.get("dist_ma20_pct"))
    vol_ratio = _to_float(metrics.get("volume_ratio_5_20")) or 0.0
    proximity = _to_float(metrics.get("high_20d_proximity"))
    close = _to_float(metrics.get("close")) or 0.0
    ma50 = _to_float(metrics.get("ma50"))
    dollar_volume = _to_float(metrics.get("dollar_volume_20d")) or 0.0
    atr_pct = _to_float(metrics.get("atr_pct"))

    if close < 5 or dollar_volume < 5_000_000:
        return {"score": 0.0, "state": "weak", "reason": "价格过低或成交额不足"}
    if atr_pct is not None and (atr_pct < 0.5 or atr_pct > 15):
        return {"score": 0.0, "state": "weak", "reason": f"波动率异常（ATR {atr_pct:.1f}%）"}

    if rs15 < 0 or (ma50 and close < ma50):
        state = "weak"
        reason = "跑输板块或跌破 MA50"
    elif dist_ma20 is not None and dist_ma20 > 8:
        state = "extended"
        reason = f"高于 MA20 {dist_ma20:.1f}%，追高风险"
    elif (
        vol_ratio <= 0.9
        and dist_ma20 is not None
        and -3 <= dist_ma20 <= 5
        and rs15 > 0
    ):
        state = "pullback"
        reason = f"板块领涨 + 个股跑赢 {rs15:+.1f}% + 缩量回踩 MA20"
    elif proximity is not None and proximity >= 0.98 and vol_ratio >= 1.0:
        state = "breakout"
        reason = "接近 20 日高点且放量"
    else:
        state = "watch"
        reason = "相对强势但位置/量能尚未给出信号"

    sector_score = {1: 100.0, 2: 85.0, 3: 70.0}.get(sector_rank, 50.0)
    rs_score = _clamp(50.0 + rs15 * 3.0 + rs30 * 1.5)
    setup_score = {
        "pullback": 95.0,
        "breakout": 85.0,
        "watch": 60.0,
        "extended": 30.0,
        "weak": 15.0,
    }[state]
    if dollar_volume >= 200_000_000:
        liquidity_score = 100.0
    elif dollar_volume >= 50_000_000:
        liquidity_score = 80.0
    elif dollar_volume >= 10_000_000:
        liquidity_score = 60.0
    else:
        liquidity_score = 40.0

    total = (
        sector_score * 0.30
        + rs_score * 0.35
        + setup_score * 0.25
        + liquidity_score * 0.10
    )
    risk_penalty = 0.0
    if not risk_on:
        # In a risk-off tape, breakout/watch signals need more confirmation.
        risk_penalty = 8.0 if state == "pullback" else 15.0
        total -= risk_penalty
        reason += "；大盘 risk-off，降低仓位并等确认"

    return {
        "score": round(max(0.0, total), 1),
        "state": state,
        "state_label": _STATE_LABELS[state],
        "reason": reason,
        "score_parts": {
            "sector": round(sector_score, 1),
            "relative_strength": round(rs_score, 1),
            "setup": setup_score,
            "liquidity": liquidity_score,
            "risk_penalty": risk_penalty,
        },
    }


class StockScreenerService:
    """Rank hot-sector constituents for early positioning."""

    def __init__(
        self,
        themes_service: Any = None,
        candidate_provider: Optional[Callable[[str, str, int], List[Candidate]]] = None,
        history_provider: Optional[Callable[[List[str]], Dict[str, Dict[str, List[float]]]]] = None,
        fundamental_provider: Optional[Callable[[List[str]], Dict[str, Dict[str, Any]]]] = None,
        filing_provider: Any = None,
        learning_service: Any = None,
        cache_ttl_seconds: int = 900,
        earnings_blackout_days: int = EARNINGS_BLACKOUT_DAYS,
    ) -> None:
        self._themes_service = themes_service
        self._candidate_provider = candidate_provider or self._default_candidates
        self._history_provider = history_provider or self._default_history
        self._fundamental_provider = fundamental_provider or self._default_fundamentals
        self._filing_provider = filing_provider
        self._learning = learning_service
        self._cache_ttl = max(0, int(cache_ttl_seconds))
        self._earnings_blackout_days = max(0, int(earnings_blackout_days))
        self._candidate_cache: Dict[str, Dict[str, Any]] = {}
        self._fundamental_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def screen(
        self,
        market: str = "us",
        top_n: int = 10,
        as_of: Optional[str] = None,
        min_score: float = 65.0,
        backtest: bool = False,
    ) -> Dict[str, Any]:
        market = (market or "us").strip().lower()
        if market != "us":
            return {
                "status": "not_supported",
                "market": market,
                "reason": "当前选股器只支持美股；港股需要另建成分股池",
            }

        themes = self._get_themes(market)
        if not isinstance(themes, dict) or themes.get("status") not in ("ok", "partial"):
            return {
                "status": "not_supported",
                "market": market,
                "reason": "市场热点数据不可用，无法确定领涨板块",
            }

        # Rank sectors by a blend of 1-week and 1-month momentum; pure 5d is too
        # noisy and can promote a one-day bounce.
        theme_rows = [
            row for row in (themes.get("all_themes") or themes.get("leaders") or [])
            if isinstance(row, dict) and row.get("symbol") in SECTOR_KEYS
        ]
        theme_rows.sort(
            key=lambda row: (
                (_to_float(row.get("return_5d")) or 0.0) * 0.6
                + (_to_float(row.get("return_20d")) or 0.0) * 0.4
            ),
            reverse=True,
        )
        leaders = theme_rows[:3]
        if not leaders:
            return {
                "status": "not_supported",
                "market": market,
                "reason": "领涨板块为空，无法选股",
            }
        risk_on = bool(themes.get("risk_on"))

        candidates: List[Candidate] = []
        seen = set()
        for rank, row in enumerate(leaders, start=1):
            etf = str(row.get("symbol"))
            _, sector_name = SECTOR_KEYS[etf]
            for item in self._candidate_provider(etf, sector_name, rank):
                if item.symbol in seen:
                    continue
                seen.add(item.symbol)
                candidates.append(item)

        if not candidates:
            return {
                "status": "not_supported",
                "market": market,
                "reason": "领涨板块没有可用成分股",
            }

        symbols = list(dict.fromkeys(
            [item.symbol for item in candidates]
            + [row["symbol"] for row in leaders]
            + list(SECTOR_KEYS)
        ))
        history = self._history_provider(symbols) or {}

        sector_returns: Dict[str, Dict[str, Optional[float]]] = {}
        for row in leaders:
            etf = str(row["symbol"])
            closes = (history.get(etf) or {}).get("close") or []
            sector_returns[etf] = {
                "return_15d": _return_pct(closes, 15),
                "return_30d": _return_pct(closes, 30),
            }

        picks: List[Dict[str, Any]] = []
        for candidate in candidates:
            payload = history.get(candidate.symbol) or {}
            metrics = self._build_metrics(payload)
            if not metrics:
                continue
            sector = sector_returns.get(candidate.sector_etf) or {}
            sector_ret_15 = sector.get("return_15d")
            sector_ret_30 = sector.get("return_30d")
            metrics["rs_15d"] = (
                round(metrics["return_15d"] - sector_ret_15, 2)
                if metrics.get("return_15d") is not None and sector_ret_15 is not None
                else None
            )
            metrics["rs_30d"] = (
                round(metrics["return_30d"] - sector_ret_30, 2)
                if metrics.get("return_30d") is not None and sector_ret_30 is not None
                else None
            )
            if metrics.get("rs_15d") is None:
                # No sector benchmark -> do not pretend the stock is a leader.
                continue
            score = score_metrics(metrics, candidate.sector_rank, risk_on=risk_on)
            if score["score"] <= 0:
                continue
            plan = self._build_plan(metrics, payload)
            picks.append({
                "symbol": candidate.symbol,
                "name": candidate.name or candidate.symbol,
                "sector_etf": candidate.sector_etf,
                "sector_name": candidate.sector_name,
                "sector_rank": candidate.sector_rank,
                **metrics,
                **score,
                "plan": plan,
            })

        picks.sort(key=lambda item: item["score"], reverse=True)
        picks, filter_stats = self._apply_quality_filters(picks, top_n)
        picks, filing_stats = self._apply_filing_risk(picks)
        filter_stats.update(filing_stats)

        learning_stats: Dict[str, Any] = {}
        effective_min_score = min_score
        if self._learning is not None:
            try:
                self._learning.maybe_refresh()
                picks, learning_stats = self._learning.apply_adjustments(picks)
                effective_min_score = min_score + float(learning_stats.get("min_score_delta") or 0)
                picks = [
                    pick for pick in picks
                    if float(pick.get("score") or 0) >= effective_min_score
                ]
            except Exception as exc:
                logger.warning("[选股学习] 应用调整失败，沿用原始评分: %s", exc)
        else:
            picks = [
                pick for pick in picks
                if float(pick.get("score") or 0) >= min_score
            ]

        returns_by_symbol = {
            candidate.symbol: _daily_returns(
                (history.get(candidate.symbol) or {}).get("close") or []
            )
            for candidate in candidates
        }
        qualified = list(picks)
        picks, correlation_skipped = self._select_diversified(
            picks,
            top_n,
            returns_by_symbol,
            max_per_sector=max(2, math.ceil(max(1, int(top_n or 10)) / max(1, len(leaders)))),
        )
        filter_stats["correlation_skipped"] = correlation_skipped
        selected_symbols = {item.get("symbol") for item in picks}
        challengers = [
            item for item in qualified
            if item.get("symbol") not in selected_symbols
        ][: max(10, int(top_n or 10) * 2)]

        if self._learning is not None and picks:
            try:
                for item in challengers:
                    item["is_challenger"] = True
                self._learning.record_picks({
                    "status": "ok",
                    "picks": picks + challengers,
                })
            except Exception as exc:
                logger.debug("[选股学习] 记录 picks 失败: %s", exc)

        validation = None
        if backtest:
            validation = self._walk_forward_validation(
                candidates=candidates,
                history=history,
                top_n=max(3, min(int(top_n or 10), 5)),
                risk_on=risk_on,
            )

        summary = self._build_summary(leaders, picks)
        return {
            "status": "ok" if picks else "partial",
            "market": market,
            "as_of": as_of,
            "risk_on": risk_on,
            "min_score": min_score,
            "effective_min_score": effective_min_score,
            "filters": filter_stats,
            "validation": validation,
            "learning": learning_stats,
            "sectors": [
                {
                    "symbol": row.get("symbol"),
                    "name": row.get("name"),
                    "return_5d": row.get("return_5d"),
                    "return_20d": row.get("return_20d"),
                    "return_15d": sector_returns.get(str(row.get("symbol")), {}).get("return_15d"),
                }
                for row in leaders
            ],
            "picks": picks,
            "challengers": challengers,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Filters / selection / validation
    # ------------------------------------------------------------------
    def _apply_filing_risk(
        self,
        picks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Flag recent dilution / M&A / material-event filings.

        A momentum screen cannot predict fraud, but it can avoid names that
        just filed an actual secondary offering and warn on fresh 8-K / 13D
        events.  Fail-open: if SEC is unavailable, keep the pick and mark the
        risk as unknown.
        """
        stats = {"event_skipped": 0}
        if self._filing_provider is None or not picks:
            return picks, stats

        kept: List[Dict[str, Any]] = []
        today = datetime.now().date()
        for pick in picks:
            symbol = str(pick.get("symbol") or "")
            try:
                response = self._filing_provider.get_recent_filings(symbol, limit=12) or {}
                items = response.get("items") or []
            except Exception as exc:
                logger.debug("[选股] %s SEC 披露检查失败: %s", symbol, exc)
                items = []

            risks: List[str] = []
            blocked = False
            for item in items:
                form = str(item.get("form") or "").upper()
                filed = str(item.get("filed") or "")[:10]
                try:
                    filed_day = datetime.fromisoformat(filed).date()
                except ValueError:
                    continue
                age = (today - filed_day).days
                if form in ("424B5", "424B3") and age <= 30:
                    risks.append(f"{form} 近 {age} 天增发/发行")
                    blocked = True
                elif form in ("S-3", "S-3/A", "F-3", "F-3/A") and age <= 30:
                    risks.append(f"{form} 近 {age} 天注册新证券")
                elif form == "8-K" and age <= 14:
                    risks.append(f"8-K 近 {age} 天重大事项")
                elif form in ("SC 13D", "SC 13D/A") and age <= 60:
                    risks.append(f"{form} 近 {age} 天 activist/并购披露")

            pick["filing_risks"] = risks
            if blocked:
                pick["risk_level"] = "high"
                pick["risk_note"] = "；".join(risks)
                stats["event_skipped"] += 1
                continue
            pick["risk_level"] = "medium" if risks else "low"
            pick["risk_note"] = "；".join(risks) if risks else ""
            if risks:
                pick["score"] = round(max(0.0, float(pick.get("score") or 0) - 5.0), 1)
            kept.append(pick)
        return kept, stats

    def _apply_quality_filters(
        self,
        picks: List[Dict[str, Any]],
        top_n: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Drop earnings-window and low-quality names before final ranking."""
        stats = {"earnings_skipped": 0, "quality_skipped": 0, "quality_unknown": 0}
        prefilter = picks[: max(int(top_n or 10) * 4, 15)]
        if not prefilter:
            return [], stats

        symbols = [item["symbol"] for item in prefilter]
        fundamentals = self._fundamental_provider(symbols) or {}
        filtered: List[Dict[str, Any]] = []
        for pick in prefilter:
            info = fundamentals.get(pick["symbol"]) or {}
            pick.update({
                "earnings_date": info.get("earnings_date"),
                "days_to_earnings": info.get("days_to_earnings"),
                "quality_ok": info.get("quality_ok"),
                "quality_notes": info.get("quality_notes") or [],
                "market_cap": info.get("market_cap"),
                "revenue_growth": info.get("revenue_growth"),
                "profit_margin": info.get("profit_margin"),
                "debt_to_equity": info.get("debt_to_equity"),
                "free_cashflow": info.get("free_cashflow"),
                "current_ratio": info.get("current_ratio"),
                "return_on_equity": info.get("return_on_equity"),
            })

            days = info.get("days_to_earnings")
            if days is not None and 0 <= days <= self._earnings_blackout_days:
                stats["earnings_skipped"] += 1
                continue
            if info.get("quality_ok") is False:
                stats["quality_skipped"] += 1
                continue
            if info.get("quality_ok") is None:
                stats["quality_unknown"] += 1
                pick["score"] = round(max(0.0, float(pick.get("score") or 0) - 3.0), 1)
            penalty = 0.0
            if info.get("free_cashflow") is not None and info.get("free_cashflow") < 0:
                penalty += 2.0
            if info.get("current_ratio") is not None and info.get("current_ratio") < 0.8:
                penalty += 2.0
            if info.get("return_on_equity") is not None and info.get("return_on_equity") < -0.2:
                penalty += 2.0
            if penalty:
                pick["score"] = round(max(0.0, float(pick.get("score") or 0) - penalty), 1)
            filtered.append(pick)
        return filtered, stats

    def _select_diversified(
        self,
        picks: List[Dict[str, Any]],
        top_n: int,
        returns_by_symbol: Dict[str, List[float]],
        max_per_sector: int = 3,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Greedy selection with a per-sector cap and a correlation cap."""
        selected: List[Dict[str, Any]] = []
        sector_counts: Dict[str, int] = {}
        skipped = 0
        target = max(1, int(top_n or 10))
        for pick in picks:
            if len(selected) >= target:
                break
            sector = str(pick.get("sector_name") or pick.get("sector_etf"))
            candidate_returns = returns_by_symbol.get(pick["symbol"]) or []
            replaced = False
            correlated = False
            # If this name is highly correlated with an existing pick but is
            # clearly stronger, replace the weaker one instead of silently
            # dropping the challenger.
            for index, existing in enumerate(selected):
                existing_returns = returns_by_symbol.get(existing["symbol"]) or []
                if _correlation(candidate_returns, existing_returns) <= MAX_CORRELATION:
                    continue
                correlated = True
                if float(pick.get("score") or 0) > float(existing.get("score") or 0) + 2.0:
                    selected[index] = pick
                    replaced = True
                break
            if replaced:
                continue
            if correlated:
                skipped += 1
                continue
            if sector_counts.get(sector, 0) >= max_per_sector:
                # Sector cap reached: keep the strongest, not the first seen.
                weakest = min(
                    (item for item in selected if str(item.get("sector_name") or item.get("sector_etf")) == sector),
                    key=lambda item: float(item.get("score") or 0),
                    default=None,
                )
                if weakest is not None and float(pick.get("score") or 0) > float(weakest.get("score") or 0) + 2.0:
                    selected.remove(weakest)
                    selected.append(pick)
                else:
                    skipped += 1
                continue
            selected.append(pick)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return selected, skipped

    def _walk_forward_validation(
        self,
        candidates: List[Candidate],
        history: Dict[str, Dict[str, List[float]]],
        top_n: int = 5,
        risk_on: bool = True,
    ) -> Dict[str, Any]:
        """Lightweight walk-forward check of the scoring rules.

        Uses the current candidate pool (survivorship-biased) and forward
        10-day excess return vs SPY.  It is a sanity check, not a production
        backtest; transaction costs and earnings filters are excluded.
        """
        spy = (history.get("SPY") or {}).get("close") or []
        if len(spy) < 90:
            return {"status": "insufficient", "reason": "基准历史不足"}
        sector_etfs = [
            etf for etf in SECTOR_KEYS
            if (history.get(etf) or {}).get("close")
        ]
        if not sector_etfs:
            return {"status": "insufficient", "reason": "板块 ETF 历史不足"}

        start = max(70, len(spy) - 90)
        end = len(spy) - 11
        samples: List[float] = []
        for index in range(start, end, 5):
            sector_stats = []
            for etf in sector_etfs:
                closes = history[etf]["close"]
                ret_5 = _return_pct_at(closes, index, 5)
                ret_20 = _return_pct_at(closes, index, 20)
                if ret_5 is None or ret_20 is None:
                    continue
                sector_stats.append((etf, ret_5 * 0.6 + ret_20 * 0.4, ret_5, ret_20))
            sector_stats.sort(key=lambda item: item[1], reverse=True)
            top_sectors = {item[0]: item for item in sector_stats[:3]}
            rank_map = {item[0]: index for index, item in enumerate(sector_stats[:3], start=1)}
            if not top_sectors:
                continue

            date_picks = []
            for candidate in candidates:
                if candidate.sector_etf not in top_sectors:
                    continue
                payload = history.get(candidate.symbol) or {}
                sliced = {
                    key: (values[: index + 1] if isinstance(values, list) else values)
                    for key, values in payload.items()
                }
                metrics = self._build_metrics(sliced)
                if not metrics:
                    continue
                sector = top_sectors[candidate.sector_etf]
                if metrics.get("return_15d") is None:
                    continue
                metrics["rs_15d"] = metrics["return_15d"] - sector[2]
                metrics["rs_30d"] = (
                    metrics["return_30d"] - sector[3]
                    if metrics.get("return_30d") is not None
                    else None
                )
                score = score_metrics(metrics, rank_map.get(candidate.sector_etf, 3), risk_on=risk_on)
                if score["score"] >= 60:
                    date_picks.append((score["score"], candidate.symbol))
            date_picks.sort(reverse=True)
            for _, symbol in date_picks[:top_n]:
                closes = (history.get(symbol) or {}).get("close") or []
                if index + 10 >= len(closes) or closes[index] <= 0:
                    continue
                stock_forward = (closes[index + 10] / closes[index] - 1.0) * 100.0
                spy_forward = (spy[index + 10] / spy[index] - 1.0) * 100.0
                samples.append(stock_forward - spy_forward)

        if not samples:
            return {"status": "insufficient", "reason": "可回测样本不足"}
        wins = sum(1 for value in samples if value > 0)
        return {
            "status": "ok",
            "samples": len(samples),
            "hit_rate_10d": round(wins / len(samples) * 100, 1),
            "avg_excess_10d": round(sum(samples) / len(samples), 2),
            "note": "基于当前成分股池，存在幸存者偏差；未计交易成本与财报过滤",
        }

    # ------------------------------------------------------------------
    # Metrics / plan
    # ------------------------------------------------------------------
    @staticmethod
    def _build_metrics(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        closes = [float(value) for value in (payload.get("close") or []) if _to_float(value)]
        highs = [float(value) for value in (payload.get("high") or []) if _to_float(value)]
        lows = [float(value) for value in (payload.get("low") or []) if _to_float(value)]
        volumes = [float(value) for value in (payload.get("volume") or []) if _to_float(value)]
        if len(closes) < 35 or len(highs) < 35 or len(lows) < 35 or len(volumes) < 35:
            return None

        close = closes[-1]
        ma5 = _mean(closes[-5:])
        ma10 = _mean(closes[-10:])
        ma20 = _mean(closes[-20:])
        ma50 = _mean(closes[-50:]) if len(closes) >= 50 else None
        high_20d = max(highs[-20:])
        vol5 = _mean(volumes[-5:])
        vol20 = _mean(volumes[-20:])
        dollar_volume = _mean(
            [close_value * volume for close_value, volume in zip(closes[-20:], volumes[-20:])]
        )
        if not ma20 or not vol20 or dollar_volume is None:
            return None

        return {
            "close": round(close, 2),
            "return_5d": _return_pct(closes, 5),
            "return_15d": _return_pct(closes, 15),
            "return_30d": _return_pct(closes, 30),
            "ma5": round(ma5, 2) if ma5 else None,
            "ma10": round(ma10, 2) if ma10 else None,
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2) if ma50 else None,
            "dist_ma20_pct": round((close / ma20 - 1.0) * 100.0, 2),
            "high_20d_proximity": round(close / high_20d, 3) if high_20d > 0 else None,
            "volume_ratio_5_20": round(vol5 / vol20, 2) if vol5 and vol20 else None,
            "dollar_volume_20d": round(dollar_volume, 0),
            "atr_pct": _atr_pct(highs, lows, closes),
        }

    @staticmethod
    def _build_plan(metrics: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        closes = [float(value) for value in (payload.get("close") or []) if _to_float(value)]
        highs = [float(value) for value in (payload.get("high") or []) if _to_float(value)]
        lows = [float(value) for value in (payload.get("low") or []) if _to_float(value)]
        history = [
            {"close": close, "high": high, "low": low}
            for close, high, low in zip(closes[:-1], highs[:-1], lows[:-1])
        ]
        context = {
            "code": "",
            "today": {
                "close": metrics.get("close"),
                "ma5": metrics.get("ma5"),
                "ma10": metrics.get("ma10"),
                "ma20": metrics.get("ma20"),
                "ma50": metrics.get("ma50"),
            },
            "history": history,
            "market": "us",
        }
        plan = build_trade_plan(context)
        return {
            "buy_trigger": plan.get("buy_trigger"),
            "entry_low": plan.get("entry_low"),
            "entry_high": plan.get("entry_high"),
            "stop_loss": plan.get("stop_loss"),
            "target_1": plan.get("target_1"),
            "target_2": plan.get("target_2"),
            "risk_reward": plan.get("risk_reward"),
            "time_stop": plan.get("time_stop"),
        }

    # ------------------------------------------------------------------
    # Data providers
    # ------------------------------------------------------------------
    def _get_themes(self, market: str) -> Dict[str, Any]:
        if self._themes_service is not None:
            return self._themes_service.get_snapshot(market)
        from src.services.market_themes_service import get_market_themes_service

        return get_market_themes_service().get_snapshot(market)

    def _default_fundamentals(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch earnings date + quality metrics for the pre-filtered candidates."""
        result: Dict[str, Dict[str, Any]] = {}
        pending: List[str] = []
        now = time.time()
        with self._lock:
            for symbol in symbols:
                cached = self._fundamental_cache.get(symbol)
                if cached and now - float(cached.get("ts", 0)) <= 3600:
                    result[symbol] = cached.get("data") or {}
                else:
                    pending.append(symbol)

        if not pending:
            return result

        def fetch(symbol: str) -> Tuple[str, Dict[str, Any]]:
            try:
                import yfinance as yf

                info = yf.Ticker(symbol).info or {}
            except Exception:
                return symbol, {}

            earnings_ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
            earnings_date = None
            days_to_earnings = None
            try:
                if earnings_ts:
                    earnings_dt = datetime.fromtimestamp(float(earnings_ts))
                    earnings_date = earnings_dt.strftime("%Y-%m-%d")
                    days_to_earnings = (earnings_dt.date() - datetime.now().date()).days
                    if days_to_earnings is not None and days_to_earnings < 0:
                        days_to_earnings = None
            except (TypeError, ValueError, OSError, OverflowError):
                pass

            revenue_growth = _to_float(info.get("revenueGrowth"))
            profit_margin = _to_float(info.get("profitMargins"))
            debt_to_equity = _to_float(info.get("debtToEquity"))
            market_cap = _to_float(info.get("marketCap"))
            free_cashflow = _to_float(info.get("freeCashflow"))
            total_cash = _to_float(info.get("totalCash"))
            total_debt = _to_float(info.get("totalDebt"))
            current_ratio = _to_float(info.get("currentRatio"))
            return_on_equity = _to_float(info.get("returnOnEquity"))
            notes: List[str] = []
            quality_ok: Optional[bool] = True
            if market_cap is not None and market_cap < MIN_MARKET_CAP:
                quality_ok = False
                notes.append("市值过小")
            if revenue_growth is not None and revenue_growth < MIN_REVENUE_GROWTH:
                quality_ok = False
                notes.append("营收下滑")
            if profit_margin is not None and profit_margin < MIN_PROFIT_MARGIN:
                quality_ok = False
                notes.append("净利率为负")
            if debt_to_equity is not None and debt_to_equity > MAX_DEBT_TO_EQUITY:
                quality_ok = False
                notes.append("负债率过高")
            if free_cashflow is not None and free_cashflow < 0:
                notes.append("自由现金流为负")
            if (
                free_cashflow is not None
                and free_cashflow < 0
                and total_debt is not None
                and total_cash is not None
                and total_debt > total_cash
            ):
                quality_ok = False
                notes.append("负FCF且债务高于现金")
            if current_ratio is not None and current_ratio < 0.8:
                notes.append("流动比率偏低")
            if return_on_equity is not None and return_on_equity < -0.2:
                notes.append("ROE 为负")
            if all(value is None for value in (market_cap, revenue_growth, profit_margin, debt_to_equity)):
                quality_ok = None
                notes.append("基本面数据缺失")

            return symbol, {
                "earnings_date": earnings_date,
                "days_to_earnings": days_to_earnings,
                "market_cap": market_cap,
                "revenue_growth": revenue_growth,
                "profit_margin": profit_margin,
                "debt_to_equity": debt_to_equity,
                "free_cashflow": free_cashflow,
                "total_cash": total_cash,
                "total_debt": total_debt,
                "current_ratio": current_ratio,
                "return_on_equity": return_on_equity,
                "quality_ok": quality_ok,
                "quality_notes": notes,
            }

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(fetch, symbol) for symbol in pending]
            for future in as_completed(futures):
                try:
                    symbol, data = future.result()
                except Exception:
                    continue
                result[symbol] = data
                with self._lock:
                    self._fundamental_cache[symbol] = {"ts": now, "data": data}
        return result

    def _default_candidates(self, sector_etf: str, sector_name: str, rank: int) -> List[Candidate]:
        cache_key = sector_etf
        now = time.time()
        with self._lock:
            cached = self._candidate_cache.get(cache_key)
            if cached and now - float(cached.get("ts", 0)) <= 6 * 3600:
                rows = cached.get("rows") or []
                return [
                    Candidate(symbol, name, sector_etf, sector_name, rank)
                    for symbol, name in rows
                ]

        rows = self._fetch_nasdaq_universe(sector_etf)
        if not rows:
            rows = self._fetch_yahoo_top_companies(sector_etf)
        if not rows:
            rows = list(FALLBACK_UNIVERSE.get(sector_etf, []))
        with self._lock:
            self._candidate_cache[cache_key] = {"ts": now, "rows": rows}
        return [Candidate(symbol, name, sector_etf, sector_name, rank) for symbol, name in rows]

    @staticmethod
    def _parse_money(value: Any) -> Optional[float]:
        text = str(value or "").replace("$", "").replace(",", "").strip()
        if not text or text in ("N/A", "--"):
            return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fetch_nasdaq_universe(sector_etf: str) -> List[Tuple[str, str]]:
        """Broader universe from Nasdaq's public screener (large + mid + small).

        Yahoo's sector page only returns ~12 mega caps; Nasdaq lets us take a
        market-cap-sorted slice that also contains mid/small caps.
        """
        sector = NASDAQ_SECTOR.get(sector_etf)
        if not sector:
            return []
        try:
            import requests

            url = (
                "https://api.nasdaq.com/api/screener/stocks"
                "?tableonly=true&limit=80&offset=0"
                f"&sector={quote(sector)}&sortColumn=marketcap&sortOrder=desc"
            )
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=12,
            )
            response.raise_for_status()
            rows = ((response.json() or {}).get("data") or {}).get("table", {}).get("rows") or []
        except Exception as exc:
            logger.debug("[选股] Nasdaq 成分股获取失败 %s: %s", sector_etf, exc)
            return []

        excluded_words = ("warrant", "right", " unit", "preferred", "notes", "fund")
        parsed: List[Tuple[str, str, float]] = []
        for row in rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            name = str(row.get("name") or symbol).strip()
            if not re.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", symbol):
                continue
            lowered = name.lower()
            if any(word in lowered for word in excluded_words):
                continue
            price = StockScreenerService._parse_money(row.get("lastsale"))
            if price is not None and price < 5:
                continue
            market_cap = StockScreenerService._parse_money(row.get("marketCap"))
            if market_cap is None or market_cap < MIN_MARKET_CAP:
                continue
            parsed.append((symbol, name, market_cap))

        large = [item for item in parsed if item[2] >= 10_000_000_000][:8]
        mid = [item for item in parsed if 2_000_000_000 <= item[2] < 10_000_000_000][:4]
        small = [item for item in parsed if MIN_MARKET_CAP <= item[2] < 2_000_000_000][:4]
        selected = large + mid + small
        if len(selected) < CANDIDATES_PER_SECTOR:
            chosen = {item[0] for item in selected}
            for item in parsed:
                if item[0] in chosen:
                    continue
                selected.append(item)
                chosen.add(item[0])
                if len(selected) >= CANDIDATES_PER_SECTOR:
                    break
        return [(symbol, name) for symbol, name, _ in selected[:CANDIDATES_PER_SECTOR]]

    @staticmethod
    def _fetch_yahoo_top_companies(sector_etf: str) -> List[Tuple[str, str]]:
        key_info = SECTOR_KEYS.get(sector_etf)
        if not key_info:
            return []
        key = key_info[0]
        try:
            import yfinance as yf

            frame = yf.Sector(key).top_companies
            if frame is None or getattr(frame, "empty", True):
                return []
            columns = {str(column).lower(): column for column in frame.columns}
            symbol_col = columns.get("symbol") or columns.get("ticker")
            name_col = columns.get("name") or columns.get("company")
            if symbol_col is None:
                return []
            rows: List[Tuple[str, str]] = []
            for _, row in frame.head(12).iterrows():
                symbol = str(row.get(symbol_col) or "").strip().upper()
                if not symbol:
                    continue
                name = str(row.get(name_col) or symbol).strip() if name_col else symbol
                rows.append((symbol, name))
            return rows
        except Exception as exc:
            logger.debug("[选股] Yahoo 板块成分股获取失败 %s: %s", sector_etf, exc)
            return []

    @staticmethod
    def _default_history(symbols: List[str]) -> Dict[str, Dict[str, List[float]]]:
        result = StockScreenerService._download_yfinance_history(symbols)
        if result:
            return result
        return StockScreenerService._download_twelvedata_history(symbols)

    @staticmethod
    def _download_yfinance_history(symbols: List[str]) -> Dict[str, Dict[str, List[float]]]:
        try:
            import yfinance as yf

            frame = yf.download(
                tickers=" ".join(symbols),
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.warning("[选股] 行情批量下载失败: %s", exc)
            return {}
        if frame is None or getattr(frame, "empty", True):
            return {}

        columns = getattr(frame, "columns", None)
        multi = getattr(columns, "nlevels", 1) > 1
        result: Dict[str, Dict[str, List[float]]] = {}
        for symbol in symbols:
            try:
                sub = frame[symbol] if multi else frame
                payload = {}
                for field in ("Close", "High", "Low", "Volume"):
                    series = sub[field].dropna()
                    payload[field.lower()] = [float(item) for item in series.tolist()]
                if len(payload["close"]) >= 2:
                    result[symbol] = payload
            except Exception:
                continue
        return result

    @staticmethod
    def _download_twelvedata_history(symbols: List[str]) -> Dict[str, Dict[str, List[float]]]:
        """Fallback when Yahoo rate-limits the batch download."""
        try:
            from data_provider.twelvedata_fetcher import TwelveDataFetcher

            fetcher = TwelveDataFetcher()
            if not fetcher.is_api_ready():
                return {}
            result: Dict[str, Dict[str, List[float]]] = {}
            for symbol in symbols:
                try:
                    frame, _ = fetcher.get_daily_data(symbol, days=260)
                    if frame is None or getattr(frame, "empty", True):
                        continue
                    payload = {}
                    for field in ("close", "high", "low", "volume"):
                        if field not in frame.columns:
                            payload = {}
                            break
                        payload[field] = [float(item) for item in frame[field].dropna().tolist()]
                    if payload and len(payload["close"]) >= 2:
                        result[symbol] = payload
                except Exception:
                    continue
            if result:
                logger.info("[选股] 已切换到 Twelve Data 兜底行情")
            return result
        except Exception as exc:
            logger.debug("[选股] Twelve Data 兜底失败: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary(leaders: List[Dict[str, Any]], picks: List[Dict[str, Any]]) -> str:
        sector_text = "、".join(
            f"{row.get('name')} {row.get('symbol')}"
            for row in leaders
        )
        if not picks:
            return f"今日主线：{sector_text}；板块内没有满足条件的个股。"
        pick_text = "、".join(
            f"{item['symbol']}（{item['state_label']}）" for item in picks[:5]
        )
        return f"今日主线：{sector_text}；优先观察：{pick_text}。"

    @staticmethod
    def render_markdown(result: Dict[str, Any], top_n: int = 10) -> str:
        if not isinstance(result, dict) or result.get("status") not in ("ok", "partial"):
            reason = (result or {}).get("reason") or "选股数据不可用"
            return f"## 🔎 热门板块选股\n\n> {reason}\n"

        lines = [
            "## 🔎 热门板块选股（提前布局）",
            "",
            f"> {result.get('summary', '')}",
            "",
        ]
        filters = result.get("filters") or {}
        if any(filters.values()):
            lines.append(
                f"> 过滤：财报窗口 {filters.get('earnings_skipped', 0)} 只 / "
                f"质量 {filters.get('quality_skipped', 0)} 只 / "
                f"相关性 {filters.get('correlation_skipped', 0)} 只"
            )
            lines.append("")
        learning = result.get("learning") or {}
        if learning.get("updated_at") and learning.get("samples"):
            lines.append(
                f"> 自动学习：样本 {learning.get('samples', 0)} | "
                f"10日胜率 {learning.get('hit_rate_10d')}% | "
                f"平均超额 {learning.get('avg_excess_10d')}% | "
                f"本次剔除 {learning.get('adjustment_removed', 0)} 只"
            )
            lines.append("")
        lines.extend([
            "| 板块 | 代码 | 名称 | 15日 | 30日 | 相对板块(15日) | 距MA20 | 量比(5/20) | 状态 | 评分 |",
            "|------|------|------|------|------|----------------|--------|------------|------|------|",
        ])
        for item in (result.get("picks") or [])[:top_n]:
            def _pct(value: Any) -> str:
                return "N/A" if value is None else f"{float(value):+.1f}%"

            lines.append(
                f"| {item.get('sector_name')} | {item.get('symbol')} | {item.get('name')} | "
                f"{_pct(item.get('return_15d'))} | {_pct(item.get('return_30d'))} | "
                f"{_pct(item.get('rs_15d'))} | {_pct(item.get('dist_ma20_pct'))} | "
                f"{item.get('volume_ratio_5_20') or 'N/A'} | {item.get('state_label')} | "
                f"{item.get('score')} |"
            )

        validation = result.get("validation")
        if isinstance(validation, dict) and validation.get("status") == "ok":
            lines.extend([
                "",
                "### 历史验证（粗略）",
                f"- 样本 {validation.get('samples')} 个 | "
                f"10日胜率 {validation.get('hit_rate_10d')}% | "
                f"平均超额 {validation.get('avg_excess_10d')}%",
                f"- {validation.get('note')}",
            ])

        lines.extend(["", "### 交易计划", ""])
        for item in (result.get("picks") or [])[:top_n]:
            plan = item.get("plan") or {}
            if not plan.get("buy_trigger"):
                continue
            lines.extend([
                f"**{item.get('symbol')} {item.get('name')}**（{item.get('state_label')}）",
                f"- 为什么：{item.get('reason')}",
                f"- 买入：{plan.get('buy_trigger')}",
                f"- 止损：{plan.get('stop_loss')} | 目标1：{plan.get('target_1')} | "
                f"目标2：{plan.get('target_2')} | 盈亏比：{plan.get('risk_reward')}",
                f"- 时间止损：{plan.get('time_stop')}",
                "",
            ])
        return "\n".join(lines)


def _build_default_screener() -> StockScreenerService:
    try:
        from data_provider.fundamental_adapter import UsSecFundamentalAdapter

        filing_provider = UsSecFundamentalAdapter()
    except Exception:
        filing_provider = None
    return StockScreenerService(
        learning_service=get_screener_learning_service(),
        filing_provider=filing_provider,
    )


_SCREENER = _build_default_screener()


def get_stock_screener_service() -> StockScreenerService:
    return _SCREENER

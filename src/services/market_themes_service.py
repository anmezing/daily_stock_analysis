# -*- coding: utf-8 -*-
"""Market themes / sector rotation snapshot.

The per-stock prompt already had news and fundamentals, but no market-wide
answer to "今天资金在买什么、卖什么".  This service builds that block from
sector ETFs already available on Yahoo Finance, with a short cache because the
same snapshot is reused for every stock in one run.

Fail-open: any network/provider error returns ``status=not_supported`` and the
analysis continues without the block.
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# SPDR sector ETFs: the standard free proxy for US sector rotation.
US_THEMES: Dict[str, str] = {
    "XLK": "科技",
    "XLC": "通信服务",
    "XLY": "可选消费",
    "XLP": "必需消费",
    "XLF": "金融",
    "XLV": "医疗",
    "XLI": "工业",
    "XLB": "原材料",
    "XLE": "能源",
    "XLU": "公用事业",
    "XLRE": "房地产",
}

US_BENCHMARKS: Dict[str, str] = {
    "SPY": "标普500",
    "QQQ": "纳指100",
}

# HK-listed theme/broad ETFs.  Not a full sector grid, but enough to tell
# whether money is in tech, SOEs, or the broad market.
HK_THEMES: Dict[str, str] = {
    "3033.HK": "恒生科技",
    "2828.HK": "国企指数",
    "2822.HK": "A50",
}

HK_BENCHMARKS: Dict[str, str] = {
    "2800.HK": "恒生指数",
}

_CACHE_TTL_SECONDS = 900


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


def _download_closes(symbols: List[str]) -> Dict[str, List[float]]:
    """Download daily closes for *symbols* in one yfinance request."""
    import yfinance as yf

    payload = yf.download(
        tickers=" ".join(symbols),
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    result: Dict[str, List[float]] = {}
    if payload is None or getattr(payload, "empty", True):
        return result

    columns = getattr(payload, "columns", None)
    multi = getattr(columns, "nlevels", 1) > 1
    for symbol in symbols:
        try:
            frame = payload[symbol] if multi else payload
            series = frame["Close"].dropna()
            closes = [float(item) for item in series.tolist() if _to_float(item)]
            if len(closes) >= 2:
                result[symbol] = closes
        except Exception:
            continue
    return result


def _return_pct(closes: List[float], days: int) -> Optional[float]:
    if len(closes) <= days or days <= 0:
        return None
    base = closes[-1 - days]
    if base <= 0:
        return None
    return round((closes[-1] / base - 1.0) * 100.0, 2)


def _above_ma(closes: List[float], window: int) -> Optional[bool]:
    if len(closes) < window or window <= 0:
        return None
    average = sum(closes[-window:]) / window
    return closes[-1] >= average


def _perf_row(symbol: str, name: str, closes: List[float]) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "name": name,
        "return_1d": _return_pct(closes, 1),
        "return_5d": _return_pct(closes, 5),
        "return_20d": _return_pct(closes, 20),
        "above_ma50": _above_ma(closes, 50),
        "above_ma200": _above_ma(closes, 200),
    }


def _sort_key(row: Dict[str, Any]) -> float:
    value = row.get("return_5d")
    if value is None:
        value = row.get("return_1d")
    return float(value) if value is not None else 0.0


def _format_return(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:+.2f}%"


class MarketThemesService:
    """Build a cached market-theme snapshot for US/HK."""

    def __init__(
        self,
        fetcher: Optional[Callable[[List[str]], Dict[str, List[float]]]] = None,
        ttl_seconds: int = _CACHE_TTL_SECONDS,
    ) -> None:
        self._fetcher = fetcher or _download_closes
        self._ttl = max(0, int(ttl_seconds))
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def get_snapshot(self, market: Optional[str], as_of: Optional[str] = None) -> Dict[str, Any]:
        market = str(market or "").strip().lower()
        if market == "cn":
            return {
                "status": "not_supported",
                "market": market,
                "reason": "A 股已有板块涨跌榜，不重复拉取 ETF",
            }
        if market not in ("us", "hk"):
            return {"status": "not_supported", "market": market, "reason": "unknown market"}

        cache_key = f"{market}:{as_of or ''}"
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and now - float(cached.get("ts", 0)) <= self._ttl:
                return cached.get("data", {})

        themes = US_THEMES if market == "us" else HK_THEMES
        benchmarks = US_BENCHMARKS if market == "us" else HK_BENCHMARKS
        symbols = list(dict.fromkeys(list(benchmarks) + list(themes)))

        try:
            closes_by_symbol = self._fetcher(symbols) or {}
        except Exception as exc:
            logger.warning("[市场热点] %s 拉取失败: %s", market, exc)
            closes_by_symbol = {}

        if not closes_by_symbol:
            data = {
                "status": "not_supported",
                "market": market,
                "reason": "yfinance 未返回数据",
            }
            self._store(cache_key, data)
            return data

        benchmark_rows = [
            _perf_row(symbol, name, closes_by_symbol[symbol])
            for symbol, name in benchmarks.items()
            if symbol in closes_by_symbol
        ]
        theme_rows = [
            _perf_row(symbol, name, closes_by_symbol[symbol])
            for symbol, name in themes.items()
            if symbol in closes_by_symbol
        ]

        if not theme_rows:
            data = {
                "status": "not_supported",
                "market": market,
                "reason": "行业 ETF 无有效数据",
            }
            self._store(cache_key, data)
            return data

        theme_rows.sort(key=_sort_key, reverse=True)
        leaders = theme_rows[:3]
        negative_rows = [row for row in theme_rows if (_sort_key(row) or 0) < 0]
        laggards = list(reversed(negative_rows[-3:])) if negative_rows else []

        primary = benchmark_rows[0] if benchmark_rows else {}
        above_50 = primary.get("above_ma50")
        above_200 = primary.get("above_ma200")
        checks = [value for value in (above_50, above_200) if value is not None]
        if checks:
            risk_on = all(checks)
        else:
            risk_on = (primary.get("return_20d") or 0) > 0

        leader_text = "、".join(
            f"{row['name']} {row['symbol']} {_format_return(row.get('return_5d'))}"
            for row in leaders
        )
        laggard_text = "、".join(
            f"{row['name']} {row['symbol']} {_format_return(row.get('return_5d'))}"
            for row in laggards
        ) or "无明显领跌"
        bench_text = "、".join(
            f"{row['name']} {row['symbol']} 5日 {_format_return(row.get('return_5d'))}"
            for row in benchmark_rows
        ) or "基准缺失"
        state_text = "risk-on（指数在 MA50/MA200 上方）" if risk_on else "risk-off / 震荡（指数未同时站上 MA50 与 MA200）"
        summary = (
            f"{bench_text}；{state_text}。"
            f"领涨：{leader_text}；领跌：{laggard_text}。"
        )

        data = {
            "status": "ok" if len(theme_rows) >= 3 else "partial",
            "market": market,
            "as_of": as_of,
            "benchmarks": benchmark_rows,
            "leaders": leaders,
            "laggards": laggards,
            "all_themes": theme_rows,
            "risk_on": risk_on,
            "summary": summary,
        }
        logger.info("[市场热点] %s %s", market, summary)
        self._store(cache_key, data)
        return data

    def _store(self, key: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = {"ts": time.time(), "data": data}


_SERVICE = MarketThemesService()


def get_market_themes_service() -> MarketThemesService:
    return _SERVICE

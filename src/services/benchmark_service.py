# -*- coding: utf-8 -*-
"""Benchmark series for backtest attribution.

Stores index/ETF daily bars in ``stock_daily`` under a canonical code, so a
backtest can answer "did this signal beat the market over the same window?"
without introducing another table.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from data_provider.base import canonical_stock_code, normalize_stock_code

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Fetch/cache benchmark bars and compute window returns."""

    def __init__(self, db: Any, config: Any = None) -> None:
        self.db = db
        if config is None:
            from src.config import get_config

            config = get_config()
        self.config = config
        self._series_cache: Dict[str, List[Tuple[date, Optional[float], Optional[float]]]] = {}
        self._ensured_until: Dict[str, date] = {}

    def code_for(self, market: str) -> Optional[str]:
        attr = {
            "cn": "backtest_benchmark_cn",
            "hk": "backtest_benchmark_hk",
            "us": "backtest_benchmark_us",
        }.get((market or "").strip().lower())
        if not attr:
            return None
        code = str(getattr(self.config, attr, "") or "").strip()
        return code or None

    @staticmethod
    def _normalize(code: str) -> str:
        return canonical_stock_code(normalize_stock_code(code))

    def _load_series(self, market: str) -> List[Tuple[date, Optional[float], Optional[float]]]:
        cached = self._series_cache.get(market)
        if cached is not None:
            return cached
        code = self.code_for(market)
        if not code:
            self._series_cache[market] = []
            return []
        normalized = self._normalize(code)
        rows = self.db.get_data_range(normalized, date(1990, 1, 1), date(2100, 1, 1))
        series = [
            (row.date, getattr(row, "open", None), getattr(row, "close", None))
            for row in rows
            if row.date is not None
        ]
        series.sort(key=lambda item: item[0])
        self._series_cache[market] = series
        return series

    def _download(self, symbol: str, start: date, end: date) -> Optional[Any]:
        try:
            import yfinance as yf
        except Exception:
            return None
        try:
            frame = yf.download(
                symbol,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.debug("基准数据下载失败 %s: %s", symbol, exc)
            return None
        if frame is None or getattr(frame, "empty", True):
            return None
        frame = frame.copy()
        if getattr(frame.columns, "nlevels", 1) > 1:
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        frame = frame.reset_index()
        date_col = "date" if "date" in frame.columns else frame.columns[0]
        frame = frame.rename(columns={date_col: "date"})
        if "close" not in frame.columns:
            return None
        if "open" not in frame.columns:
            frame["open"] = frame["close"]
        return frame[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])

    def ensure(self, market: str, start: date, end: date) -> Optional[str]:
        """Ensure benchmark history exists up to *end*; return its DB code."""
        if not bool(getattr(self.config, "backtest_benchmark_enabled", True)):
            return None
        code = self.code_for(market)
        if not code:
            return None
        normalized = self._normalize(code)
        ensured = self._ensured_until.get(market)
        if ensured is not None and ensured >= end:
            return normalized

        series = self._load_series(market)
        covered = bool(series) and series[-1][0] >= end - timedelta(days=7)
        if not covered:
            fetch_start = min(start, date.today() - timedelta(days=365))
            frame = self._download(code, fetch_start, end)
            if frame is not None and not getattr(frame, "empty", True):
                try:
                    self.db.save_daily_data(frame, code=normalized, data_source="benchmark:yfinance")
                except Exception as exc:
                    logger.debug("基准数据入库失败 %s: %s", normalized, exc)
                self._series_cache.pop(market, None)
                series = self._load_series(market)
                covered = bool(series) and series[-1][0] >= end - timedelta(days=7)

        if covered:
            self._ensured_until[market] = end
            return normalized
        return normalized if series else None

    def return_pct(self, market: str, start: date, end: date) -> Optional[float]:
        """Close-to-close benchmark return over the same window."""
        if not start or not end or end <= start:
            return None
        if self.ensure(market, start, end) is None:
            return None
        series = self._load_series(market)
        if len(series) < 2:
            return None
        entry = next((item for item in series if item[0] >= start), None)
        exit_item = next((item for item in reversed(series) if item[0] <= end), None)
        if entry is None or exit_item is None or exit_item[0] <= entry[0]:
            return None
        entry_price = entry[2] if entry[2] else entry[1]
        exit_price = exit_item[2] if exit_item[2] else exit_item[1]
        try:
            if not entry_price or not exit_price or entry_price <= 0:
                return None
            return round((float(exit_price) - float(entry_price)) / float(entry_price) * 100, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def snapshot(self, market: str, as_of: Optional[date] = None) -> Dict[str, Any]:
        """Return a compact regime snapshot (trend vs MA200, 20d volatility).

        Reads cached benchmark bars only. Population happens during backtests,
        so an analysis never blocks on an index download.
        """
        as_of = as_of or date.today()
        series = [item for item in self._load_series(market) if item[0] <= as_of]
        closes = [float(item[2]) for item in series if item[2] is not None]
        if len(closes) < 20:
            return {}
        last = closes[-1]
        ma200 = sum(closes[-200:]) / len(closes[-200:]) if len(closes) >= 200 else None
        returns = [
            (closes[index] - closes[index - 1]) / closes[index - 1] * 100
            for index in range(max(1, len(closes) - 20), len(closes))
            if closes[index - 1]
        ]
        volatility = (sum(value * value for value in returns) / len(returns)) ** 0.5 if returns else None
        recent_vol = (
            sum(value * value for value in returns[-10:]) / len(returns[-10:])
        ) ** 0.5 if len(returns) >= 10 else None
        return {
            "code": self.code_for(market),
            "close": round(last, 4),
            "ma200": round(ma200, 4) if ma200 else None,
            "above_ma200": bool(ma200 and last >= ma200),
            "volatility_20d_pct": round(volatility, 3) if volatility is not None else None,
            "recent_volatility_pct": round(recent_vol, 3) if recent_vol is not None else None,
            "risk_on": bool(ma200 and last >= ma200),
        }

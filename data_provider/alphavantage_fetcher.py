# -*- coding: utf-8 -*-
"""
AlphaVantageFetcher — 美股行情数据源（API 优先链，次级）

数据源：AlphaVantage REST API
限流：免费版 25 calls/day、5 calls/min
市场：仅美股（不支持 A 股 / 港股 / 台股 / 日韩）

日线使用 TIME_SERIES_DAILY 的 compact 输出（最近 100 个交易日），
因此仅适合日线增量分析，不适合长周期回溯。
"""

import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS
from .realtime_types import UnifiedRealtimeQuote, RealtimeSource
from .us_index_mapping import is_us_stock_code

logger = logging.getLogger(__name__)

_AV_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageFetcher(BaseFetcher):
    """AlphaVantage 美股日线与实时行情适配器。"""

    name = "AlphaVantageFetcher"
    priority = int(os.getenv("ALPHAVANTAGE_PRIORITY", "3"))

    def __init__(self):
        from src.config import get_config

        config = get_config()
        self._api_key = getattr(config, "alphavantage_api_key", None) or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self._api_key:
            logger.debug("[AlphaVantage] 未配置 ALPHAVANTAGE_API_KEY，数据源不参与链路")

    def is_api_ready(self) -> bool:
        """是否已具备可用条件（供 DataFetcherManager 组装美股链路时判断）。"""
        return bool(self._api_key)

    def _is_us_stock(self, stock_code: str) -> bool:
        return is_us_stock_code(stock_code)

    def supports_code(self, stock_code: str) -> bool:
        """供 DataFetcherManager 判断该源是否覆盖此代码，避免港股链路空跑一次。"""
        return self._is_us_stock(stock_code)

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._api_key:
            raise DataFetchError("[AlphaVantage] 未配置 ALPHAVANTAGE_API_KEY")
        if not self._is_us_stock(stock_code):
            raise DataFetchError(f"[AlphaVantage] 不支持的市场代码: {stock_code}")

        symbol = stock_code.strip().upper()
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": self._api_key,
        }

        try:
            self.random_sleep(0.5, 1.5)
            resp = requests.get(_AV_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise DataFetchError(f"[AlphaVantage] 请求失败 symbol={symbol}: {e}") from e

        if "Note" in data:
            raise DataFetchError(f"[AlphaVantage] 触发限流: {data['Note']}")
        if "Error Message" in data:
            raise DataFetchError(f"[AlphaVantage] 接口错误 symbol={symbol}: {data['Error Message']}")

        ts_key = "Time Series (Daily)"
        if ts_key not in data or not data[ts_key]:
            raise DataFetchError(f"[AlphaVantage] 返回无数据 symbol={symbol}")

        rows = []
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        for date_str, values in data[ts_key].items():
            row_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if start <= row_date <= end:
                rows.append(
                    {
                        "date": date_str,
                        "1. open": float(values.get("1. open", 0)),
                        "2. high": float(values.get("2. high", 0)),
                        "3. low": float(values.get("3. low", 0)),
                        "4. close": float(values.get("4. close", 0)),
                        "5. volume": float(values.get("5. volume", 0)),
                    }
                )

        if not rows:
            raise DataFetchError(f"[AlphaVantage] 区间内无数据 symbol={symbol}")

        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"])
        df.index.name = None  # 避免与 _normalize_data 重新添加的 'date' 列冲突
        return df.drop(columns=["date"])

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df.index).date
        df = df.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )
        # AlphaVantage 返回最新在前，先按日期升序再算涨跌幅
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        df["pct_chg"] = df["close"].pct_change() * 100
        df["pct_chg"] = df["pct_chg"].fillna(0).round(2)
        df["amount"] = df["volume"] * df["close"]
        df["code"] = stock_code

        keep = ["code"] + STANDARD_COLUMNS
        df = df[[col for col in keep if col in df.columns]]
        return df

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        if not self._api_key or not self._is_us_stock(stock_code):
            return None

        symbol = stock_code.strip().upper()
        try:
            self.random_sleep(0.5, 1.5)
            resp = requests.get(
                _AV_BASE_URL,
                params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": self._api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"[AlphaVantage] 实时行情获取失败 symbol={symbol}: {e}")
            return None

        gq = data.get("Global Quote", {})
        price_str = gq.get("05. price")
        if not price_str:
            return None

        price = float(price_str)
        prev_close = float(gq.get("08. previous close", 0))
        change_pct_str = str(gq.get("10. change percent", "")).replace("%", "").strip()
        change_pct = float(change_pct_str) if change_pct_str else None

        return UnifiedRealtimeQuote(
            code=symbol,
            source=RealtimeSource.FALLBACK,
            price=price,
            change_pct=round(change_pct, 2) if change_pct is not None else None,
            change_amount=round(float(gq.get("09. change", 0)), 4),
            volume=int(float(gq.get("06. volume", 0))),
            amount=None,
            volume_ratio=None,
            turnover_rate=None,
            amplitude=None,
            open_price=float(gq.get("02. open", 0)),
            high=float(gq.get("03. high", 0)),
            low=float(gq.get("04. low", 0)),
            pre_close=prev_close,
        )

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        if not self._api_key or not self._is_us_stock(stock_code):
            return None

        symbol = stock_code.strip().upper()
        try:
            resp = requests.get(
                _AV_BASE_URL,
                params={"function": "SYMBOL_SEARCH", "keywords": symbol, "apikey": self._api_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug(f"[AlphaVantage] 代码搜索失败 symbol={symbol}: {e}")
            return None

        for match in data.get("bestMatches", []):
            if match.get("1. symbol") == symbol and match.get("2. name"):
                return match["2. name"]
        return None

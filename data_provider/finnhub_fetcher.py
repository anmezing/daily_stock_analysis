# -*- coding: utf-8 -*-
"""
FinnhubFetcher — 美股行情数据源（API 优先链）

数据源：Finnhub.io REST API
限流：免费版 60 calls/min
市场：仅美股（不支持 A 股 / 港股 / 台股 / 日韩）
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .base import BaseFetcher, DataFetchError, DataSourceUnavailableError, STANDARD_COLUMNS
from .realtime_types import UnifiedRealtimeQuote, RealtimeSource
from .us_index_mapping import is_us_stock_code

logger = logging.getLogger(__name__)

_FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class _KeyRateLimited(Exception):
    """内部信号：当前 Key 触发 429，换下一个 Key 继续。"""


class FinnhubFetcher(BaseFetcher):
    """Finnhub 美股日线与实时行情适配器。"""

    name = "FinnhubFetcher"
    priority = int(os.getenv("FINNHUB_PRIORITY", "2"))
    # 免费档 60 calls/min/Key；429 只冷却当前 Key
    RATE_LIMIT_COOLDOWN_SECONDS = 60

    def __init__(self):
        from src.config import get_config

        config = get_config()
        keys = list(getattr(config, "finnhub_api_keys", None) or [])
        if not keys:
            single = getattr(config, "finnhub_api_key", None) or os.getenv("FINNHUB_API_KEY")
            keys = [single] if single else []
        self._api_keys: List[str] = list(dict.fromkeys(key.strip() for key in keys if key and key.strip()))
        self._key_index = 0
        self._key_cooldown: Dict[str, float] = {}
        self._dead_keys: set = set()
        # /stock/candle 属于付费接口：免费档恒返回 403。命中一次后进程内不再尝试，
        # 免得每只美股都要多花一次请求 + 一行告警。
        self._candle_forbidden = False
        if not self._api_keys:
            logger.debug("[Finnhub] 未配置 FINNHUB_API_KEY(S)，数据源不参与链路")

    @property
    def _api_key(self) -> str:
        """兼容旧代码/测试：返回第一个 Key。"""
        return self._api_keys[0] if self._api_keys else ""

    @_api_key.setter
    def _api_key(self, value: str) -> None:
        key = (value or "").strip()
        self._api_keys = [key] if key else []

    def is_api_ready(self) -> bool:
        """是否已具备可用条件（供 DataFetcherManager 组装美股链路时判断）。"""
        return bool(self._api_keys)

    def is_temporarily_unavailable(self) -> bool:
        """所有 Key 都在冷却/失效时才退出链路。"""
        return self._next_available_key(advance=False) is None

    def _is_us_stock(self, stock_code: str) -> bool:
        return is_us_stock_code(stock_code)

    def supports_code(self, stock_code: str) -> bool:
        """供 DataFetcherManager 判断该源是否覆盖此代码，避免港股链路空跑一次。"""
        return self._is_us_stock(stock_code)

    def is_daily_unavailable(self) -> bool:
        """日线接口被判为付费专属后，不应再进入日线链路（实时行情仍可用）。"""
        return self._candle_forbidden

    @staticmethod
    def _mask_key(api_key: str) -> str:
        if not api_key:
            return "***"
        return f"***{api_key[-4:]}" if len(api_key) > 4 else "***"

    def _next_available_key(self, *, advance: bool = True) -> Optional[str]:
        """轮询返回下一个未冷却/未失效的 Key；全部不可用时返回 None。"""
        if not self._api_keys:
            return None
        now = time.time()
        total = len(self._api_keys)
        for offset in range(total):
            index = (self._key_index + offset) % total
            key = self._api_keys[index]
            if key in self._dead_keys:
                continue
            if self._key_cooldown.get(key, 0.0) > now:
                continue
            if advance:
                self._key_index = (index + 1) % total
            return key
        return None

    def _cool_down_key(self, api_key: str) -> None:
        now = time.time()
        if self._key_cooldown.get(api_key, 0.0) > now:
            return
        self._key_cooldown[api_key] = now + self.RATE_LIMIT_COOLDOWN_SECONDS
        remaining = sum(
            1
            for key in self._api_keys
            if key not in self._dead_keys and self._key_cooldown.get(key, 0.0) <= now
        )
        logger.warning(
            "[Finnhub] Key %s 触发限流（429），冷却 %ss；仍有 %d/%d 个 Key 可用",
            self._mask_key(api_key),
            self.RATE_LIMIT_COOLDOWN_SECONDS,
            remaining,
            len(self._api_keys),
        )

    def _request_json(
        self,
        path: str,
        params: Dict[str, Any],
        *,
        timeout: float = 15,
        treat_403_as_premium: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """带多 Key 轮询的 GET；返回解析后的 JSON，限流/鉴权失败时抛可预期异常。"""
        attempts = max(1, len(self._api_keys))
        throttled: List[str] = []
        for _ in range(attempts):
            api_key = self._next_available_key()
            if not api_key:
                break
            try:
                resp = requests.get(
                    f"{_FINNHUB_BASE_URL}/{path}",
                    params={**params, "token": api_key},
                    timeout=timeout,
                )
                if resp.status_code == 429:
                    raise _KeyRateLimited()
                if resp.status_code == 401:
                    self._dead_keys.add(api_key)
                    logger.warning(
                        "[Finnhub] Key %s 无效（HTTP 401），已停用", self._mask_key(api_key)
                    )
                    continue
                if resp.status_code == 403 and treat_403_as_premium:
                    self._candle_forbidden = True
                    logger.info(
                        "[Finnhub] /stock/candle 返回 HTTP 403（免费档不含日线接口），本进程不再尝试"
                    )
                    raise DataSourceUnavailableError(
                        "[Finnhub] 日线接口需要付费档"
                    )
                resp.raise_for_status()
                return resp.json()
            except _KeyRateLimited:
                throttled.append(self._mask_key(api_key))
                self._cool_down_key(api_key)
                continue

        if throttled:
            raise DataSourceUnavailableError(
                "[Finnhub] 所有 Key 均触发 429（%s），%ss 内交给下一个数据源"
                % (", ".join(throttled), self.RATE_LIMIT_COOLDOWN_SECONDS)
            )
        raise DataSourceUnavailableError("[Finnhub] 没有可用的 Key（全部冷却或失效）")

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._api_keys:
            raise DataFetchError("[Finnhub] 未配置 FINNHUB_API_KEY")
        if not self._is_us_stock(stock_code):
            raise DataFetchError(f"[Finnhub] 不支持的市场代码: {stock_code}")
        if self._candle_forbidden:
            raise DataSourceUnavailableError(
                "[Finnhub] /stock/candle 需要付费档（本进程已判定 403），日线改由下一个数据源提供"
            )

        symbol = stock_code.strip().upper()
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

        try:
            self.random_sleep(0.3, 0.8)
            data = self._request_json(
                "stock/candle",
                {
                    "symbol": symbol,
                    "resolution": "D",
                    "from": start_ts,
                    "to": end_ts,
                },
                timeout=15,
                treat_403_as_premium=True,
            )
        except DataSourceUnavailableError:
            raise
        except Exception as e:
            raise DataFetchError(f"[Finnhub] 请求失败 symbol={symbol}: {e}") from e

        if not data or data.get("s") != "ok" or not data.get("c"):
            raise DataFetchError(f"[Finnhub] 返回无数据 symbol={symbol}")

        return pd.DataFrame(
            {
                "c": data["c"],
                "h": data["h"],
                "l": data["l"],
                "o": data["o"],
                "t": data["t"],
                "v": data["v"],
            }
        )

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["t"], unit="s").dt.date
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df["pct_chg"] = df["close"].pct_change() * 100
        df["pct_chg"] = df["pct_chg"].fillna(0).round(2)
        df["amount"] = df["volume"] * df["close"]
        df["code"] = stock_code

        keep = ["code"] + STANDARD_COLUMNS
        df = df[[col for col in keep if col in df.columns]]
        return df

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        if not self._api_keys or not self._is_us_stock(stock_code):
            return None

        symbol = stock_code.strip().upper()
        try:
            self.random_sleep(0.3, 0.8)
            data = self._request_json("quote", {"symbol": symbol}, timeout=15)
        except Exception as e:
            logger.warning(f"[Finnhub] 实时行情获取失败 symbol={symbol}: {e}")
            return None

        price = (data or {}).get("c")
        if not price:
            return None

        prev_close = data.get("pc", 0)
        change_pct = data.get("dp")
        change_amount = data.get("d")
        high = data.get("h")
        low = data.get("l")
        open_price = data.get("o")

        amplitude = None
        if high and low and prev_close and prev_close > 0:
            amplitude = round((high - low) / prev_close * 100, 2)

        return UnifiedRealtimeQuote(
            code=symbol,
            source=RealtimeSource.FALLBACK,
            price=price,
            change_pct=round(change_pct, 2) if change_pct is not None else None,
            change_amount=round(change_amount, 4) if change_amount is not None else None,
            volume=data.get("v"),
            amount=None,
            volume_ratio=None,
            turnover_rate=None,
            amplitude=amplitude,
            open_price=open_price,
            high=high,
            low=low,
            pre_close=prev_close,
        )

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        if not self._api_keys or not self._is_us_stock(stock_code):
            return None

        symbol = stock_code.strip().upper()
        try:
            data = self._request_json("search", {"q": symbol}, timeout=10)
        except Exception as e:
            logger.debug(f"[Finnhub] 代码搜索失败 symbol={symbol}: {e}")
            return None

        if not data:
            return None
        for item in data.get("result", []):
            if item.get("symbol") == symbol and item.get("description"):
                return item["description"]
        return None

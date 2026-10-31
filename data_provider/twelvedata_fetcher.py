# -*- coding: utf-8 -*-
"""
===================================
TwelveDataFetcher - Twelve Data API 数据源 (Priority 2)
===================================

数据来源：Twelve Data API (https://twelvedata.com)
特点：无需开户，API Key 即可访问美股/港股历史日线与最新价格
定位：美股/港股 API 优先链候选数据源
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from .base import (
    BaseFetcher,
    DataFetchError,
    DataSourceUnavailableError,
    STANDARD_COLUMNS,
    normalize_stock_code,
)
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote, safe_float
from .us_index_mapping import is_us_stock_code

logger = logging.getLogger(__name__)

# Regular-session length used to normalise an intraday volume ratio.
_SESSION_MINUTES = {"us": 390, "hk": 390, "cn": 330}


class TwelveDataPlanError(DataSourceUnavailableError):
    """当前套餐无权限访问该接口（401/403 等），重试没有意义。"""


class TwelveDataRateLimited(DataSourceUnavailableError):
    """账号触发限流（429），冷却期内不再请求，直接交给下一个数据源。"""


class _KeyRateLimited(Exception):
    """内部信号：当前 Key 触发 429，换下一个 Key 继续。"""


def _is_retryable_request_error(exc: BaseException) -> bool:
    """只重试网络类错误和临时性失败，套餐/参数/限流类错误不重试。"""
    if isinstance(exc, (TwelveDataPlanError, TwelveDataRateLimited)):
        return False
    return isinstance(exc, (requests.RequestException, DataFetchError))


class TwelveDataFetcher(BaseFetcher):
    """Twelve Data 美股/港股 API 数据源。"""

    name = "TwelveDataFetcher"
    priority = int(os.getenv("TWELVEDATA_PRIORITY", "2"))

    BASE_URL = "https://api.twelvedata.com"
    DEFAULT_TIMEOUT = 10
    _FALSEY_VALUES = {"0", "false", "no", "off"}
    # 某个接口被套餐限制后，多久内不再尝试（进程内）
    UNSUPPORTED_ENDPOINT_TTL_SECONDS = 3600
    # 免费档限流窗口：8 credits/分钟/Key。命中 429 后只冷却该 Key，
    # 有多个 Key 时继续用下一个（多账号的额度是独立的）。
    RATE_LIMIT_COOLDOWN_SECONDS = 60

    def __init__(self):
        self._api_keys: List[str] = []
        self._key_index = 0
        self._key_cooldown: Dict[str, float] = {}
        self._dead_keys: set = set()
        self._available: Optional[bool] = None
        self._timeout_seconds = self.DEFAULT_TIMEOUT
        self._us_hk_enable = True
        self._symbol_cache: Dict[str, str] = {}
        self._stock_name_cache: Dict[str, str] = {}
        self._unsupported_endpoints: Dict[str, float] = {}

    @property
    def _api_key(self) -> str:
        """兼容旧代码/测试：返回当前主 Key。"""
        return self._api_keys[0] if self._api_keys else ""

    @_api_key.setter
    def _api_key(self, value: str) -> None:
        key = (value or "").strip()
        self._api_keys = [key] if key else []

    def _load_runtime_config(self) -> None:
        raw_keys = (os.getenv("TWELVEDATA_API_KEYS") or "").strip()
        single_key = (os.getenv("TWELVEDATA_API_KEY") or "").strip()
        timeout_seconds = int(os.getenv("TWELVEDATA_TIMEOUT_SECONDS", str(self.DEFAULT_TIMEOUT)) or self.DEFAULT_TIMEOUT)
        enabled = (os.getenv("TWELVEDATA_US_HK_ENABLE", "true") or "true").strip().lower() not in self._FALSEY_VALUES

        try:
            from src.config import get_config

            config = get_config()
            config_keys = getattr(config, "twelvedata_api_keys", None) or []
            if config_keys:
                raw_keys = ",".join(config_keys)
            single_key = (getattr(config, "twelvedata_api_key", None) or single_key or "").strip()
            timeout_seconds = int(
                getattr(config, "twelvedata_timeout_seconds", timeout_seconds) or timeout_seconds
            )
            enabled = bool(getattr(config, "twelvedata_us_hk_enable", enabled))
        except Exception:
            logger.debug("[TwelveDataFetcher] 使用环境变量作为运行时配置")

        keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        if not keys and single_key:
            keys = [single_key]

        deduped: List[str] = []
        for key in keys:
            if key not in deduped:
                deduped.append(key)
        self._api_keys = deduped
        self._timeout_seconds = max(1, int(timeout_seconds))
        self._us_hk_enable = enabled

    def _is_available(self) -> bool:
        if self._available is not None:
            return self._available

        self._load_runtime_config()
        self._available = bool(self._api_keys and self._us_hk_enable)
        return self._available

    def is_api_ready(self) -> bool:
        """供 DataFetcherManager 判断是否应纳入 US/HK API 优先链。"""
        return self._is_available()

    def is_temporarily_unavailable(self) -> bool:
        """所有 Key 都在冷却时才退出链路，避免每只股票都产生一条跳过日志。"""
        return self._next_available_key(advance=False) is None

    @staticmethod
    def _is_hk_code(stock_code: str) -> bool:
        normalized = (stock_code or "").strip().upper()
        if normalized.endswith(".HK"):
            base = normalized[:-3]
            return base.isdigit() and 1 <= len(base) <= 5
        if normalized.startswith("HK"):
            digits = normalized[2:]
            return digits.isdigit() and 1 <= len(digits) <= 5
        return normalized.isdigit() and 1 <= len(normalized) <= 5

    @classmethod
    def _is_supported_market(cls, stock_code: str) -> bool:
        normalized = normalize_stock_code(stock_code).strip().upper()
        return is_us_stock_code(normalized) or cls._is_hk_code(normalized)

    @staticmethod
    def _extract_hk_digits(stock_code: str) -> str:
        normalized = normalize_stock_code(stock_code).strip().upper()
        if normalized.endswith(".HK"):
            digits = normalized[:-3]
        elif normalized.startswith("HK"):
            digits = normalized[2:]
        else:
            digits = normalized
        return (digits or "").zfill(5)

    def _candidate_search_terms(self, stock_code: str) -> List[str]:
        normalized = normalize_stock_code(stock_code).strip().upper()
        if is_us_stock_code(normalized):
            return [normalized]

        hk_digits = self._extract_hk_digits(normalized)
        return [
            hk_digits,
            hk_digits.lstrip("0") or "0",
            f"HK{hk_digits}",
            f"{hk_digits}.HK",
        ]

    @staticmethod
    def _country_matches(entry: Dict[str, Any], expected_country: str) -> bool:
        country = str(entry.get("country") or "").strip().lower()
        expected = expected_country.strip().lower()
        if not country:
            return False
        aliases = {
            "united states": {"united states", "us", "usa"},
            "hong kong": {"hong kong", "hk"},
        }
        expected_aliases = aliases.get(expected, {expected})
        return country in expected_aliases or any(alias in country for alias in expected_aliases)

    def _select_symbol_entry(
        self,
        entries: List[Dict[str, Any]],
        stock_code: str,
    ) -> Optional[Dict[str, Any]]:
        normalized = normalize_stock_code(stock_code).strip().upper()
        if not entries:
            return None

        if is_us_stock_code(normalized):
            for entry in entries:
                symbol = str(entry.get("symbol") or "").strip().upper()
                if symbol == normalized and self._country_matches(entry, "united states"):
                    return entry
            for entry in entries:
                symbol = str(entry.get("symbol") or "").strip().upper()
                if symbol == normalized:
                    return entry
            return None

        if self._is_hk_code(normalized):
            target_digits = (self._extract_hk_digits(normalized).lstrip("0") or "0")
            for entry in entries:
                symbol = str(entry.get("symbol") or "").strip().upper()
                if not symbol.endswith(".HK"):
                    continue
                symbol_digits = (symbol.split(".", 1)[0].lstrip("0") or "0")
                if symbol_digits == target_digits:
                    return entry
            for entry in entries:
                symbol = str(entry.get("symbol") or "").strip().upper()
                if symbol.endswith(".HK"):
                    return entry
        return None

    def _find_symbol_entry(self, stock_code: str) -> Optional[Dict[str, Any]]:
        if not self._is_available():
            return None

        for search_term in self._candidate_search_terms(stock_code):
            try:
                payload = self._request(
                    "symbol_search",
                    {
                        "symbol": search_term,
                        "outputsize": 20,
                        "show_plan": "false",
                    },
                )
            except Exception as exc:
                logger.debug(
                    "[TwelveDataFetcher] symbol_search 失败: code=%s search_term=%s reason=%s",
                    stock_code,
                    search_term,
                    exc,
                )
                continue
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or not data:
                continue
            entry = self._select_symbol_entry(data, stock_code)
            if entry is not None:
                return entry
        return None

    def _resolve_symbol(self, stock_code: str) -> str:
        normalized = normalize_stock_code(stock_code).strip().upper()
        cached_symbol = self._symbol_cache.get(normalized)
        if cached_symbol:
            return cached_symbol

        if is_us_stock_code(normalized):
            self._symbol_cache[normalized] = normalized
            return normalized

        entry = self._find_symbol_entry(normalized)
        if entry is not None:
            resolved_symbol = str(entry.get("symbol") or "").strip().upper()
            if resolved_symbol:
                self._symbol_cache[normalized] = resolved_symbol
                instrument_name = str(entry.get("instrument_name") or "").strip()
                if instrument_name:
                    self._stock_name_cache[normalized] = instrument_name
                return resolved_symbol

        fallback_symbol = f"{self._extract_hk_digits(normalized)}.HK"
        self._symbol_cache[normalized] = fallback_symbol
        return fallback_symbol

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=_is_retryable_request_error,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_available():
            raise DataFetchError("[TwelveDataFetcher] 未配置 TWELVEDATA_API_KEY(S) 或已禁用")

        # 轮询可用 Key：某个账号 429 只冷却它自己，换个 Key 继续（多账号额度独立）
        attempts = max(1, len(self._api_keys))
        rate_limited_keys: List[str] = []
        for _ in range(attempts):
            api_key = self._next_available_key()
            if not api_key:
                break
            try:
                return self._send(endpoint, params, api_key)
            except _KeyRateLimited:
                rate_limited_keys.append(self._mask_key(api_key))
                self._cool_down_key(api_key)
                continue
            except TwelveDataPlanError:
                raise

        if rate_limited_keys:
            raise TwelveDataRateLimited(
                "[TwelveDataFetcher] 所有 Key 均触发 429（%s），%ss 内交给下一个数据源"
                % (", ".join(rate_limited_keys), self.RATE_LIMIT_COOLDOWN_SECONDS)
            )
        raise TwelveDataRateLimited(
            "[TwelveDataFetcher] 所有 Key 都在限流冷却中，本轮交给下一个数据源"
        )

    def _send(self, endpoint: str, params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        request_params = {**params, "apikey": api_key}
        try:
            response = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                params=request_params,
                timeout=self._timeout_seconds,
                headers={"Authorization": f"apikey {api_key}"},
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise DataFetchError(f"[TwelveDataFetcher] 请求超时: {exc}") from exc
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and int(status) == 429:
                raise _KeyRateLimited(str(exc)) from exc
            if status is not None and int(status) == 401:
                self._dead_keys.add(api_key)
                raise DataFetchError(
                    f"[TwelveDataFetcher] Key {self._mask_key(api_key)} 无效（HTTP 401），已停用"
                ) from exc
            # 其余 4xx 属于套餐/参数问题，重试只会浪费时间
            if status is not None and 400 <= int(status) < 500:
                raise TwelveDataPlanError(
                    f"[TwelveDataFetcher] 接口不可用 endpoint={endpoint} HTTP {status}"
                ) from exc
            raise DataFetchError(f"[TwelveDataFetcher] 网络请求失败: {exc}") from exc
        except requests.RequestException as exc:
            raise DataFetchError(f"[TwelveDataFetcher] 网络请求失败: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise DataFetchError("[TwelveDataFetcher] 返回非 JSON 响应") from exc

        if isinstance(payload, dict) and payload.get("status") == "error":
            code = payload.get("code")
            message = payload.get("message") or "unknown error"
            if str(code) == "429":
                raise _KeyRateLimited(f"API 错误 code=429: {message}")
            raise DataFetchError(f"[TwelveDataFetcher] API 错误 code={code}: {message}")

        return payload

    @staticmethod
    def _mask_key(api_key: str) -> str:
        """日志里只保留 Key 末 4 位。"""
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
            until = self._key_cooldown.get(key, 0.0)
            if until > now:
                continue
            if advance:
                self._key_index = (index + 1) % total
            return key
        return None

    def _cool_down_key(self, api_key: str) -> None:
        """单个 Key 进入冷却；只在该 Key 首次命中时记一条日志。"""
        now = time.time()
        if self._key_cooldown.get(api_key, 0.0) > now:
            return
        self._key_cooldown[api_key] = now + self.RATE_LIMIT_COOLDOWN_SECONDS
        remaining_keys = sum(
            1
            for key in self._api_keys
            if key not in self._dead_keys and self._key_cooldown.get(key, 0.0) <= now
        )
        logger.warning(
            "[TwelveDataFetcher] Key %s 触发免费档限流（429），冷却 %ss；仍有 %d/%d 个 Key 可用",
            self._mask_key(api_key),
            self.RATE_LIMIT_COOLDOWN_SECONDS,
            remaining_keys,
            len(self._api_keys),
        )

    def _endpoint_unsupported(self, endpoint: str) -> bool:
        """该接口是否在 TTL 内被判定为套餐不支持。"""
        marked_at = self._unsupported_endpoints.get(endpoint)
        if marked_at is None:
            return False
        if time.time() - marked_at >= self.UNSUPPORTED_ENDPOINT_TTL_SECONDS:
            self._unsupported_endpoints.pop(endpoint, None)
            return False
        return True

    def _mark_endpoint_unsupported(self, endpoint: str, reason: str) -> None:
        if self._endpoint_unsupported(endpoint):
            return
        self._unsupported_endpoints[endpoint] = time.time()
        logger.info(
            "[TwelveDataFetcher] 接口 %s 不可用（%s），%ss 内不再尝试",
            endpoint,
            reason,
            self.UNSUPPORTED_ENDPOINT_TTL_SECONDS,
        )

    def get_stock_name(self, stock_code: str) -> str:
        normalized = normalize_stock_code(stock_code).strip().upper()
        cached_name = self._stock_name_cache.get(normalized, "")
        if cached_name:
            return cached_name

        try:
            entry = self._find_symbol_entry(normalized)
        except Exception as exc:
            logger.debug("[TwelveDataFetcher] symbol_search 获取名称失败: code=%s reason=%s", normalized, exc)
            return ""

        if entry is None:
            return ""

        instrument_name = str(entry.get("instrument_name") or "").strip()
        resolved_symbol = str(entry.get("symbol") or "").strip().upper()
        if resolved_symbol:
            self._symbol_cache[normalized] = resolved_symbol
        if instrument_name:
            self._stock_name_cache[normalized] = instrument_name
        return instrument_name

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._is_supported_market(stock_code):
            raise DataFetchError(f"[TwelveDataFetcher] 不支持的市场代码: {stock_code}")

        symbol = self._resolve_symbol(stock_code)
        logger.info("[TwelveDataFetcher] 请求历史日线: symbol=%s code=%s", symbol, stock_code)

        payload = self._request(
            "time_series",
            {
                "symbol": symbol,
                "interval": "1day",
                "start_date": start_date,
                "end_date": end_date,
                "outputsize": 5000,
                "order": "asc",
                "format": "JSON",
            },
        )

        values = payload.get("values") if isinstance(payload, dict) else None
        if not values:
            raise DataFetchError(f"[TwelveDataFetcher] 历史日线为空: symbol={symbol}")

        df = pd.DataFrame(values)
        if df.empty:
            raise DataFetchError(f"[TwelveDataFetcher] 历史日线 DataFrame 为空: symbol={symbol}")
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        normalized = df.copy()
        normalized = normalized.rename(columns={"datetime": "date"})
        if "date" not in normalized.columns:
            raise DataFetchError("[TwelveDataFetcher] 缺少 date/datetime 字段")

        missing_columns = [column for column in ("open", "high", "low", "close", "volume") if column not in normalized.columns]
        if missing_columns:
            raise DataFetchError(f"[TwelveDataFetcher] 历史日线缺少关键字段: {', '.join(missing_columns)}")

        for column in ("open", "high", "low", "close", "volume"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        normalized = normalized.sort_values("date", ascending=True).reset_index(drop=True)
        normalized["amount"] = None
        normalized["pct_chg"] = normalized["close"].pct_change() * 100
        normalized["pct_chg"] = normalized["pct_chg"].fillna(0).round(2)

        for column in STANDARD_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = None
        return normalized[STANDARD_COLUMNS]

    @staticmethod
    def _parse_date_token(value: Any) -> Optional[str]:
        """Extract YYYY-MM-DD from Twelve Data date/datetime fields."""
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace(" ", "T")).date().isoformat()
        except ValueError:
            return raw[:10] if len(raw) >= 10 else None

    @staticmethod
    def _elapsed_session_minutes(
        market: str,
        quote_timestamp: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> Optional[int]:
        """Minutes elapsed in the regular session for the quote's timestamp.

        Returns ``None`` outside regular trading (pre-market / after close /
        stale quote), which tells the caller to fall back to a full-day ratio.
        """
        try:
            from zoneinfo import ZoneInfo

            from src.core.trading_calendar import MARKET_TIMEZONE

            market = (market or "").strip().lower()
            tz_name = MARKET_TIMEZONE.get(market)
            if not tz_name:
                return None
            tz = ZoneInfo(tz_name)
            current = now or datetime.now(tz)
            quote_dt = (
                datetime.fromtimestamp(float(quote_timestamp), tz)
                if quote_timestamp
                else current
            )
            if quote_dt.date() != current.date():
                return None
            open_dt = quote_dt.replace(hour=9, minute=30, second=0, microsecond=0)
            elapsed = int((quote_dt - open_dt).total_seconds() // 60)
            total = _SESSION_MINUTES.get(market, 390)
            if elapsed <= 0 or elapsed >= total:
                return None
            return elapsed
        except Exception:
            return None

    @staticmethod
    def _normalize_volume_ratio(
        current_volume: Optional[float],
        avg_daily_volume: Optional[float],
        elapsed_minutes: Optional[int] = None,
        total_minutes: int = 390,
    ) -> Optional[float]:
        """Volume ratio with intraday time normalisation.

        The raw ``current/5-day-average`` comparison is structurally understated
        during the session: at 10:00 ET only ~30 of 390 minutes have traded, so
        a normal session looks like extreme low volume.  Divide by the elapsed
        fraction first, then sanity-check the result.
        """
        try:
            volume = float(current_volume or 0)
            average = float(avg_daily_volume or 0)
        except (TypeError, ValueError):
            return None
        if volume <= 0 or average <= 0:
            return None

        if elapsed_minutes and 0 < int(elapsed_minutes) < total_minutes:
            expected = average * (float(elapsed_minutes) / float(total_minutes))
        else:
            expected = average
        if expected <= 0:
            return None

        ratio = volume / expected
        # 0.0041 (your US intraday log) is a unit/口径 bug, not a tradable signal.
        if ratio <= 0.02 or ratio >= 20.0:
            return None
        return round(ratio, 4)

    def _compute_volume_ratio(
        self,
        symbol: str,
        current_volume: Optional[int],
        *,
        current_date: Optional[str] = None,
        elapsed_minutes: Optional[int] = None,
    ) -> Optional[float]:
        """Compute volume_ratio as current volume / previous 5 daily average volume.

        Mirrors the repository-wide volume_ratio semantics used for other
        providers, with an intraday time adjustment for US/HK sessions.
        """
        if not current_volume or current_volume <= 0:
            return None

        try:
            payload = self._request(
                "time_series",
                {
                    "symbol": symbol,
                    "interval": "1day",
                    "outputsize": 6,
                    "order": "desc",
                    "format": "JSON",
                },
            )
        except Exception as exc:
            logger.debug("[TwelveDataFetcher] 量比补算失败(symbol=%s): %s", symbol, exc)
            return None

        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, list) or not values:
            return None

        history_rows = values
        if current_date:
            latest_date = self._parse_date_token(values[0].get("datetime")) if isinstance(values[0], dict) else None
            if latest_date == current_date:
                history_rows = values[1:]

        historical_volumes: List[float] = []
        for row in history_rows:
            if not isinstance(row, dict):
                continue
            volume = safe_float(row.get("volume"))
            if volume and volume > 0:
                historical_volumes.append(float(volume))
            if len(historical_volumes) >= 5:
                break

        if not historical_volumes:
            return None

        avg_volume = sum(historical_volumes) / len(historical_volumes)
        if avg_volume <= 0:
            return None
        ratio = self._normalize_volume_ratio(
            current_volume,
            avg_volume,
            elapsed_minutes=elapsed_minutes,
        )
        if ratio is None:
            logger.debug(
                "[TwelveDataFetcher] 量比超出合理范围，已丢弃(symbol=%s, volume=%s, avg=%s, elapsed=%s)",
                symbol,
                current_volume,
                round(avg_volume),
                elapsed_minutes,
            )
        return ratio

    def _compute_turnover_rate(self, symbol: str, current_volume: Optional[int]) -> Optional[float]:
        """Compute turnover_rate from Twelve Data statistics when plan allows it.

        Official Twelve Data statistics responses include `stock_statistics`
        fields such as `float_shares` and `shares_outstanding`.  When the
        endpoint is unavailable for the current plan, we fail open and leave
        turnover_rate as None so downstream supplement logic can still use
        other providers.
        """
        if not current_volume or current_volume <= 0:
            return None
        if self._endpoint_unsupported("statistics"):
            return None

        try:
            payload = self._request("statistics", {"symbol": symbol})
        except TwelveDataPlanError as exc:
            # 套餐不含该接口：本次进程内不再尝试，避免每只股票白等三次重试
            self._mark_endpoint_unsupported("statistics", str(exc))
            return None
        except Exception as exc:
            logger.debug("[TwelveDataFetcher] 换手率补算失败(symbol=%s): %s", symbol, exc)
            return None

        stock_statistics = payload.get("stock_statistics") if isinstance(payload, dict) else None
        if not isinstance(stock_statistics, dict):
            return None

        shares_base = (
            safe_float(stock_statistics.get("float_shares"))
            or safe_float(stock_statistics.get("shares_outstanding"))
        )
        if not shares_base or shares_base <= 0:
            return None
        return round(float(current_volume) / float(shares_base) * 100, 4)

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        if not self._is_available() or not self._is_supported_market(stock_code):
            return None

        normalized = normalize_stock_code(stock_code).strip().upper()
        symbol = self._resolve_symbol(normalized)
        logger.info("[TwelveDataFetcher] 请求实时行情: symbol=%s code=%s", symbol, normalized)

        try:
            payload = self._request("quote", {"symbol": symbol, "dp": 4})
        except Exception as exc:
            logger.warning(
                "[TwelveDataFetcher] 实时行情失败，将交由后续 fetcher fallback: symbol=%s reason=%s",
                symbol,
                exc,
            )
            return None

        price = safe_float(payload.get("close")) if isinstance(payload, dict) else None
        if price is None or price <= 0:
            price = safe_float(payload.get("price")) if isinstance(payload, dict) else None
        if price is None or price <= 0:
            logger.warning(
                "[TwelveDataFetcher] 实时行情缺失关键价格字段，将交由后续 fetcher fallback: symbol=%s",
                symbol,
            )
            return None

        current_volume = None
        raw_volume = safe_float(payload.get("volume")) if isinstance(payload, dict) else None
        if raw_volume is not None and raw_volume > 0:
            current_volume = int(raw_volume)

        previous_close = safe_float(payload.get("previous_close")) if isinstance(payload, dict) else None
        high = safe_float(payload.get("high")) if isinstance(payload, dict) else None
        low = safe_float(payload.get("low")) if isinstance(payload, dict) else None
        amplitude = None
        if previous_close and previous_close > 0 and high is not None and low is not None:
            amplitude = round((high - low) / previous_close * 100, 4)

        current_date = self._parse_date_token(payload.get("datetime")) if isinstance(payload, dict) else None
        market = "us"
        try:
            from src.core.trading_calendar import get_market_for_stock

            market = get_market_for_stock(normalized) or "us"
        except Exception:
            market = "us"
        quote_timestamp = safe_float(payload.get("timestamp")) if isinstance(payload, dict) else None
        elapsed_minutes = self._elapsed_session_minutes(market, quote_timestamp)
        volume_ratio = self._compute_volume_ratio(
            symbol,
            current_volume,
            current_date=current_date,
            elapsed_minutes=elapsed_minutes,
        )
        turnover_rate = self._compute_turnover_rate(symbol, current_volume)

        fifty_two_week = payload.get("fifty_two_week") if isinstance(payload, dict) else None
        high_52w = safe_float(fifty_two_week.get("high")) if isinstance(fifty_two_week, dict) else None
        low_52w = safe_float(fifty_two_week.get("low")) if isinstance(fifty_two_week, dict) else None

        return UnifiedRealtimeQuote(
            code=normalized,
            name=str(payload.get("name") or self._stock_name_cache.get(normalized, "")).strip(),
            source=RealtimeSource.TWELVEDATA,
            price=price,
            change_pct=safe_float(payload.get("percent_change")) if isinstance(payload, dict) else None,
            change_amount=safe_float(payload.get("change")) if isinstance(payload, dict) else None,
            volume=current_volume,
            volume_ratio=volume_ratio,
            turnover_rate=turnover_rate,
            amplitude=amplitude,
            open_price=safe_float(payload.get("open")) if isinstance(payload, dict) else None,
            high=high,
            low=low,
            pre_close=previous_close,
            high_52w=high_52w,
            low_52w=low_52w,
        )

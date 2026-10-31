# -*- coding: utf-8 -*-
"""
AkShare fundamental adapter (fail-open).

This adapter intentionally uses capability probing against multiple AkShare
endpoint candidates. It should never raise to caller; partial data is allowed.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timedelta
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_date(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def _scale_ratio(value: Optional[float]) -> Optional[float]:
    """Convert ratio-like decimals to percentage points.

    yfinance 的营收/利润增速、ROE、毛利率、机构持股比例都是小数口径
    （0.12 = 12%）。此前只在 |value| <= 1 时才乘 100，于是 150% 的增速
    会被显示成 1.5%、120% 的 ROE 被显示成 1.2%，方向没错但量级错两个数量级。
    """
    if value is None:
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    return round(ratio * 100.0, 2)


def _clean_html_text(raw: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_regex_snippets(
    text: str,
    patterns: List[str],
    *,
    radius: int = 90,
    limit: int = 3,
) -> List[str]:
    """Return short snippets around regex hits for explainability."""
    snippets: List[str] = []
    if not text:
        return snippets

    for pattern in patterns:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            continue
        for match in regex.finditer(text):
            start = max(0, match.start() - radius)
            end = min(len(text), match.end() + radius)
            snippet = text[start:end].strip(" -:;,.")
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= limit:
                return snippets
    return snippets


def _normalize_code(raw: Any) -> str:
    s = _safe_str(raw).upper()
    if "." in s:
        s = s.split(".", 1)[0]
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    return s


def _pick_by_keywords(row: pd.Series, keywords: List[str]) -> Optional[Any]:
    """
    Return first non-empty row value whose column name contains any keyword.
    """
    for col in row.index:
        col_s = str(col)
        if any(k in col_s for k in keywords):
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "-", "nan", "None"):
                return val
    return None


def _extract_latest_row(df: pd.DataFrame, stock_code: str) -> Optional[pd.Series]:
    """
    Select the most relevant row for the given stock.
    """
    if df is None or df.empty:
        return None

    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "ts_code", "symbol"))]
    target = _normalize_code(stock_code)
    if code_cols:
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                matched = df[series == target]
                if not matched.empty:
                    return matched.iloc[0]
            except Exception:
                continue
        return None

    # Fallback: use latest row
    return df.iloc[0]


def _classify_fact_period(start: Optional[datetime], end: Optional[datetime]) -> Optional[str]:
    if start is None or end is None:
        return None
    duration_days = max(0, (end - start).days)
    if duration_days >= 300:
        return "annual"
    if 70 <= duration_days <= 120:
        return "quarterly"
    return None


def _pct_change(latest: Optional[float], previous: Optional[float]) -> Optional[float]:
    if latest is None or previous in (None, 0):
        return None
    try:
        return round((float(latest) - float(previous)) / abs(float(previous)) * 100.0, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


class AkshareFundamentalAdapter:
    """AkShare adapter for fundamentals, capital flow and dragon-tiger signals."""

    def _call_df_candidates(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
        errors: List[str] = []
        try:
            import akshare as ak
        except Exception as exc:
            return None, None, [f"import_akshare:{type(exc).__name__}"]

        for func_name, kwargs in candidates:
            fn = getattr(ak, func_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                if isinstance(df, pd.Series):
                    df = df.to_frame().T
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df, func_name, errors
            except Exception as exc:
                errors.append(f"{func_name}:{type(exc).__name__}")
                continue
        return None, None, errors

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """
        Return normalized fundamental blocks from AkShare with partial tolerance.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        # Financial indicators
        fin_df, fin_source, fin_errors = self._call_df_candidates([
            ("stock_financial_abstract", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {}),
        ])
        result["errors"].extend(fin_errors)
        if fin_df is not None:
            row = _extract_latest_row(fin_df, stock_code)
            if row is not None:
                revenue_yoy = _safe_float(_pick_by_keywords(row, ["营业收入同比", "营收同比", "收入同比", "同比增长"]))
                profit_yoy = _safe_float(_pick_by_keywords(row, ["净利润同比", "净利同比", "归母净利润同比"]))
                roe = _safe_float(_pick_by_keywords(row, ["净资产收益率", "ROE", "净资产收益"]))
                gross_margin = _safe_float(_pick_by_keywords(row, ["毛利率"]))
                result["growth"] = {
                    "revenue_yoy": revenue_yoy,
                    "net_profit_yoy": profit_yoy,
                    "roe": roe,
                    "gross_margin": gross_margin,
                }
                result["source_chain"].append(f"growth:{fin_source}")

        # Earnings forecast
        forecast_df, forecast_source, forecast_errors = self._call_df_candidates([
            ("stock_yjyg_em", {"symbol": stock_code}),
            ("stock_yjyg_em", {}),
            ("stock_yjbb_em", {"symbol": stock_code}),
            ("stock_yjbb_em", {}),
        ])
        result["errors"].extend(forecast_errors)
        if forecast_df is not None:
            row = _extract_latest_row(forecast_df, stock_code)
            if row is not None:
                result["earnings"]["forecast_summary"] = _safe_str(
                    _pick_by_keywords(row, ["预告", "业绩变动", "内容", "摘要", "公告"])
                )[:200]
                result["source_chain"].append(f"earnings_forecast:{forecast_source}")

        # Earnings quick report
        quick_df, quick_source, quick_errors = self._call_df_candidates([
            ("stock_yjkb_em", {"symbol": stock_code}),
            ("stock_yjkb_em", {}),
        ])
        result["errors"].extend(quick_errors)
        if quick_df is not None:
            row = _extract_latest_row(quick_df, stock_code)
            if row is not None:
                result["earnings"]["quick_report_summary"] = _safe_str(
                    _pick_by_keywords(row, ["快报", "摘要", "公告", "说明"])
                )[:200]
                result["source_chain"].append(f"earnings_quick:{quick_source}")

        # Institution / top shareholders
        inst_df, inst_source, inst_errors = self._call_df_candidates([
            ("stock_institute_hold", {}),
            ("stock_institute_recommend", {}),
        ])
        result["errors"].extend(inst_errors)
        if inst_df is not None:
            row = _extract_latest_row(inst_df, stock_code)
            if row is not None:
                inst_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "变动", "持股变化"]))
                result["institution"]["institution_holding_change"] = inst_change
                result["source_chain"].append(f"institution:{inst_source}")

        top10_df, top10_source, top10_errors = self._call_df_candidates([
            ("stock_gdfx_top_10_em", {"symbol": stock_code}),
            ("stock_gdfx_top_10_em", {}),
            ("stock_zh_a_gdhs_detail_em", {"symbol": stock_code}),
            ("stock_zh_a_gdhs_detail_em", {}),
        ])
        result["errors"].extend(top10_errors)
        if top10_df is not None:
            row = _extract_latest_row(top10_df, stock_code)
            if row is not None:
                holder_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "持股变化", "变动"]))
                result["institution"]["top10_holder_change"] = holder_change
                result["source_chain"].append(f"top10:{top10_source}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Return stock + sector capital flow.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }

        stock_df, stock_source, stock_errors = self._call_df_candidates([
            ("stock_individual_fund_flow", {"stock": stock_code}),
            ("stock_individual_fund_flow", {"symbol": stock_code}),
            ("stock_individual_fund_flow", {}),
            ("stock_main_fund_flow", {"symbol": stock_code}),
            ("stock_main_fund_flow", {}),
        ])
        result["errors"].extend(stock_errors)
        if stock_df is not None:
            row = _extract_latest_row(stock_df, stock_code)
            if row is not None:
                net_inflow = _safe_float(_pick_by_keywords(row, ["主力净流入", "净流入", "净额"]))
                inflow_5d = _safe_float(_pick_by_keywords(row, ["5日", "五日"]))
                inflow_10d = _safe_float(_pick_by_keywords(row, ["10日", "十日"]))
                result["stock_flow"] = {
                    "main_net_inflow": net_inflow,
                    "inflow_5d": inflow_5d,
                    "inflow_10d": inflow_10d,
                }
                result["source_chain"].append(f"capital_stock:{stock_source}")

        sector_df, sector_source, sector_errors = self._call_df_candidates([
            ("stock_sector_fund_flow_rank", {}),
            ("stock_sector_fund_flow_summary", {}),
        ])
        result["errors"].extend(sector_errors)
        if sector_df is not None:
            name_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("板块", "行业", "名称", "name"))), None)
            flow_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("净流入", "主力", "flow", "净额"))), None)
            if name_col and flow_col:
                work_df = sector_df[[name_col, flow_col]].copy()
                work_df[flow_col] = pd.to_numeric(work_df[flow_col], errors="coerce")
                work_df = work_df.dropna(subset=[flow_col])
                top_df = work_df.nlargest(top_n, flow_col)
                bottom_df = work_df.nsmallest(top_n, flow_col)
                result["sector_rankings"] = {
                    "top": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in top_df.iterrows()],
                    "bottom": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in bottom_df.iterrows()],
                }
                result["source_chain"].append(f"capital_sector:{sector_source}")

        has_content = bool(result["stock_flow"] or result["sector_rankings"]["top"] or result["sector_rankings"]["bottom"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """
        Return dragon-tiger signal in lookback window.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }

        df, source, errors = self._call_df_candidates([
            ("stock_lhb_stock_statistic_em", {}),
            ("stock_lhb_detail_em", {}),
            ("stock_lhb_jgmmtj_em", {}),
        ])
        result["errors"].extend(errors)
        if df is None:
            return result

        # Try code filter
        code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码"))]
        target = _normalize_code(stock_code)
        matched = pd.DataFrame()
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                cur = df[series == target]
                if not cur.empty:
                    matched = cur
                    break
            except Exception:
                continue
        if matched.empty:
            result["source_chain"].append(f"dragon_tiger:{source}")
            result["status"] = "ok" if code_cols else "partial"
            return result

        date_col = next((c for c in matched.columns if any(k in str(c) for k in ("日期", "上榜", "交易日", "time"))), None)
        parsed_dates: List[datetime] = []
        if date_col is not None:
            for val in matched[date_col].astype(str).tolist():
                try:
                    parsed_dates.append(pd.to_datetime(val).to_pydatetime())
                except Exception:
                    continue
        now = datetime.now()
        start = now - timedelta(days=max(1, lookback_days))
        recent_dates = [d for d in parsed_dates if start <= d <= now]

        result["is_on_list"] = bool(recent_dates)
        result["recent_count"] = len(recent_dates) if recent_dates else int(len(matched))
        result["latest_date"] = max(recent_dates).date().isoformat() if recent_dates else (
            max(parsed_dates).date().isoformat() if parsed_dates else None
        )
        result["status"] = "ok"
        result["source_chain"].append(f"dragon_tiger:{source}")
        return result

    def get_northbound_flow(self) -> Dict[str, Any]:
        """Return latest northbound net flow in 亿元 when available."""
        result: Dict[str, Any] = {
            "status": "not_supported",
            "net_inflow": None,
            "source_chain": [],
            "errors": [],
        }

        df, source, errors = self._call_df_candidates([
            ("stock_hsgt_north_net_flow_in_em", {"symbol": "北上"}),
            ("stock_hsgt_north_net_flow_in_em", {}),
        ])
        result["errors"].extend(errors)
        if df is None:
            return result

        latest = df.iloc[-1]
        flow = _safe_float(
            _pick_by_keywords(
                latest,
                ["当日净流入", "净流入", "当日买入成交净额", "资金净流入", "净买额", "净额"],
            )
        )
        if flow is not None:
            if abs(flow) > 1_000_000:
                flow = round(flow / 1e8, 2)
            result["net_inflow"] = flow
            result["status"] = "ok"
        else:
            result["status"] = "partial"

        result["source_chain"].append(f"northbound_flow:{source}")
        return result


class YfinanceFundamentalAdapter:
    """Structured fundamental adapter backed by Yahoo Finance.

    Used as the structured source for HK, and as the enrichment/fallback
    source for US (SEC covers filings and reported growth; yfinance adds
    valuation multiples, margins, analyst targets and the earnings calendar).
    """

    market = "hk"

    def __init__(self, market: Optional[str] = None) -> None:
        if market:
            self.market = str(market).strip().lower()

    @staticmethod
    def _to_hk_symbol(stock_code: str) -> str:
        code = _safe_str(stock_code).upper()
        if code.startswith("HK"):
            code = code[2:]
        if code.endswith(".HK"):
            code = code[:-3]
        digits = re.sub(r"\D", "", code)
        if not digits:
            return ""
        return f"{int(digits):04d}.HK"

    def _to_yf_symbol(self, stock_code: str) -> str:
        if self.market == "us":
            code = _safe_str(stock_code).upper()
            # Yahoo uses BRK-B instead of BRK.B for share classes.
            return code.replace(".", "-") if code else ""
        return self._to_hk_symbol(stock_code)

    @staticmethod
    def _coerce_timestamp(value: Any) -> Optional[str]:
        if value in (None, "", [], (), {}):
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                parsed = YfinanceFundamentalAdapter._coerce_timestamp(item)
                if parsed:
                    return parsed
            return None
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None:
            try:
                return datetime.fromtimestamp(numeric).strftime("%Y-%m-%d")
            except (OSError, OverflowError, ValueError):
                return None
        parsed = _safe_date(value)
        if parsed:
            return parsed.strftime("%Y-%m-%d")
        return None

    @staticmethod
    def _aggregate_insider_transactions(ticker: Any, days: int = 90) -> Dict[str, Any]:
        """Aggregate insider buys/sells (direction + value), not just Form 4 count."""
        try:
            frame = ticker.insider_transactions
        except Exception:
            return {}
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return {}

        try:
            work = frame.copy()
            if "Start Date" in work.columns:
                work["_date"] = pd.to_datetime(work["Start Date"], errors="coerce")
                if getattr(work["_date"].dt, "tz", None) is not None:
                    work["_date"] = work["_date"].dt.tz_localize(None)
                cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=max(1, days))
                work = work[work["_date"].notna() & (work["_date"] >= cutoff)]
            if work.empty:
                return {}

            text = (
                work.get("Transaction", pd.Series([""] * len(work))).astype(str)
                + " "
                + work.get("Text", pd.Series([""] * len(work))).astype(str)
            ).str.lower()
            shares = pd.to_numeric(work.get("Shares"), errors="coerce").fillna(0.0)
            value = pd.to_numeric(work.get("Value"), errors="coerce").fillna(0.0)

            is_buy = text.str.contains("purchase|buy", regex=True)
            is_sell = text.str.contains("sale|sell", regex=True)
            if not bool(is_buy.any() or is_sell.any()):
                return {}

            buy_value = float(value[is_buy].sum())
            sell_value = float(value[is_sell].sum())
            return {
                "insider_buy_count_90d": int(is_buy.sum()),
                "insider_sell_count_90d": int(is_sell.sum()),
                "insider_buy_shares_90d": int(round(float(shares[is_buy].sum()))),
                "insider_sell_shares_90d": int(round(float(shares[is_sell].sum()))),
                "insider_buy_value_90d": round(buy_value, 2),
                "insider_sell_value_90d": round(sell_value, 2),
                "insider_net_value_90d": round(buy_value - sell_value, 2),
            }
        except Exception:
            return {}

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "valuation": {},
            "source_chain": [],
            "errors": [],
        }

        symbol = self._to_yf_symbol(stock_code)
        if not symbol:
            result["errors"].append(f"{self.market}_symbol_invalid")
            return result

        try:
            import yfinance as yf
        except Exception as exc:
            result["errors"].append(f"import_yfinance:{type(exc).__name__}")
            return result

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
        except Exception as exc:
            result["errors"].append(f"{self.market}_yfinance_info:{type(exc).__name__}")
            return result
        if not isinstance(info, dict) or not info:
            result["errors"].append(f"{self.market}_yfinance_empty")
            return result

        growth = {
            "revenue_yoy": _scale_ratio(_safe_float(info.get("revenueGrowth"))),
            "net_profit_yoy": _scale_ratio(
                _safe_float(info.get("earningsGrowth"))
                or _safe_float(info.get("earningsQuarterlyGrowth"))
            ),
            "roe": _scale_ratio(_safe_float(info.get("returnOnEquity"))),
            "gross_margin": _scale_ratio(_safe_float(info.get("grossMargins"))),
            "operating_margin": _scale_ratio(_safe_float(info.get("operatingMargins"))),
            "profit_margin": _scale_ratio(_safe_float(info.get("profitMargins"))),
        }
        growth = {key: value for key, value in growth.items() if value is not None}
        if growth:
            result["growth"] = growth
            result["source_chain"].append("growth:yfinance_info")

        earnings = {
            "earnings_date": self._coerce_timestamp(
                info.get("earningsTimestamp")
                or info.get("earningsTimestampStart")
                or info.get("earningsDate")
            ),
            "ex_dividend_date": self._coerce_timestamp(info.get("exDividendDate")),
            "dividend_rate": _safe_float(info.get("dividendRate")),
            # dividendYield 的口径随 yfinance 版本在"百分数"和"小数"之间变过，
            # 无法可靠归一化；原样输出，不按比率再乘 100。
            "dividend_yield_raw": _safe_float(info.get("dividendYield")),
            "eps_trailing": _safe_float(info.get("trailingEps")),
            "eps_forward": _safe_float(info.get("forwardEps")),
        }
        earnings = {key: value for key, value in earnings.items() if value not in (None, "", [], {})}
        if earnings:
            result["earnings"] = earnings
            result["source_chain"].append("earnings:yfinance_info")

        institution = {
            "institutional_holding_pct": _scale_ratio(_safe_float(info.get("heldPercentInstitutions"))),
            "analyst_coverage_count": _safe_float(info.get("numberOfAnalystOpinions")),
            "target_mean_price": _safe_float(info.get("targetMeanPrice")),
            "recommendation_key": _safe_str(info.get("recommendationKey")) or None,
            "beta": _safe_float(info.get("beta")),
            "week52_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "week52_low": _safe_float(info.get("fiftyTwoWeekLow")),
            # yfinance 的 debtToEquity 已是百分数（78.4 = 78.4%），不再乘 100
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "current_ratio": _safe_float(info.get("currentRatio")),
            "free_cashflow": _safe_float(info.get("freeCashflow")),
            "sector": _safe_str(info.get("sector")) or None,
            "industry": _safe_str(info.get("industry")) or None,
            "float_shares": _safe_float(info.get("floatShares")),
            # 空头结构：shortPercentOfFloat 是小数口径（0.009 = 0.9%）
            "short_percent_float": _scale_ratio(_safe_float(info.get("shortPercentOfFloat"))),
            "short_percent_outstanding": _scale_ratio(
                _safe_float(info.get("shortPercentOfSharesOutstanding"))
            ),
            "short_ratio_days": _safe_float(info.get("shortRatio")),
            "shares_short": _safe_float(info.get("sharesShort")),
            "shares_short_prior_month": _safe_float(info.get("sharesShortPriorMonth")),
            "short_interest_date": self._coerce_timestamp(info.get("dateShortInterest")),
            "short_change_pct": _pct_change(
                _safe_float(info.get("sharesShort")),
                _safe_float(info.get("sharesShortPriorMonth")),
            ),
        }
        institution.update(
            self._aggregate_insider_transactions(ticker) if self.market == "us" else {}
        )
        institution = {key: value for key, value in institution.items() if value not in (None, "", [], {})}
        if institution:
            result["institution"] = institution
            result["source_chain"].append("institution:yfinance_info")

        valuation = {
            "pe_ratio": _safe_float(info.get("trailingPE")),
            "forward_pe": _safe_float(info.get("forwardPE")),
            "pb_ratio": _safe_float(info.get("priceToBook")),
            "ps_ratio": _safe_float(info.get("priceToSalesTrailing12Months")),
            "peg_ratio": _safe_float(info.get("pegRatio") or info.get("trailingPegRatio")),
            "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
            "market_cap": _safe_float(info.get("marketCap")),
        }
        valuation = {key: value for key, value in valuation.items() if value is not None}
        if valuation:
            result["valuation"] = valuation
            result["source_chain"].append("valuation:yfinance_info")

        if result["growth"] or result["earnings"] or result["institution"] or result["valuation"]:
            result["status"] = "partial"

        return result


class HkYfinanceFundamentalAdapter(YfinanceFundamentalAdapter):
    """Backward-compatible HK adapter."""

    market = "hk"


class UsYfinanceFundamentalAdapter(YfinanceFundamentalAdapter):
    """US fallback/enrichment adapter (valuation, margins, analyst targets)."""

    market = "us"


def merge_fundamental_bundles(
    primary: Optional[Dict[str, Any]],
    secondary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge two fundamental bundles without losing either source.

    ``primary`` wins for scalar conflicts (e.g. reported growth from SEC),
    while dict blocks are merged key-by-key so valuation fields from a
    secondary provider still land in the bundle.
    """
    primary = primary if isinstance(primary, dict) else {}
    secondary = secondary if isinstance(secondary, dict) else {}

    has_primary = any(bool(primary.get(block)) for block in ("growth", "earnings", "institution", "valuation"))
    has_secondary = any(bool(secondary.get(block)) for block in ("growth", "earnings", "institution", "valuation"))
    if not has_primary:
        return secondary if has_secondary else primary
    if not has_secondary:
        return primary

    merged: Dict[str, Any] = {
        "status": "partial",
        "growth": {},
        "earnings": {},
        "institution": {},
        "valuation": {},
        "source_chain": [],
        "errors": [],
    }
    for block in ("growth", "earnings", "institution", "valuation"):
        merged[block] = dict(secondary.get(block) or {})
        merged[block].update(primary.get(block) or {})
        merged[block] = {key: value for key, value in merged[block].items() if value is not None}

    for chain in (primary.get("source_chain") or [], secondary.get("source_chain") or []):
        for item in chain:
            if item not in merged["source_chain"]:
                merged["source_chain"].append(item)
    for err in (primary.get("errors") or []) + (secondary.get("errors") or []):
        if err not in merged["errors"]:
            merged["errors"].append(err)

    if not any(merged[block] for block in ("growth", "earnings", "institution", "valuation")):
        merged["status"] = str(primary.get("status") or secondary.get("status") or "not_supported")
    return merged


class UsSecFundamentalAdapter:
    """Official SEC-based adapter for US structured fundamentals."""

    _TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    _SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    _COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    _TICKER_CACHE_TTL_SECONDS = 24 * 60 * 60

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "daily-stock-analysis/1.0 admin@example.com",
                "From": "admin@example.com",
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/plain,*/*",
            }
        )
        self._ticker_cache: Dict[str, Dict[str, Any]] = {}
        self._ticker_cache_ts: float = 0.0

    def _get_json(self, url: str) -> Any:
        response = self._session.get(url, timeout=8)
        response.raise_for_status()
        return response.json()

    def _get_text(self, url: str) -> str:
        response = self._session.get(url, timeout=8)
        response.raise_for_status()
        return response.text

    def _load_ticker_map(self) -> Dict[str, Dict[str, Any]]:
        now = datetime.now().timestamp()
        if self._ticker_cache and now - self._ticker_cache_ts <= self._TICKER_CACHE_TTL_SECONDS:
            return self._ticker_cache

        payload = self._get_json(self._TICKER_MAP_URL)
        records: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                records = payload.get("data", [])
            else:
                records = [item for item in payload.values() if isinstance(item, dict)]
        elif isinstance(payload, list):
            records = payload

        ticker_map: Dict[str, Dict[str, Any]] = {}
        for item in records:
            ticker = _safe_str(item.get("ticker")).upper()
            if not ticker:
                continue
            cik = item.get("cik_str") or item.get("cik")
            cik_value = None
            try:
                cik_value = int(cik)
            except (TypeError, ValueError):
                continue
            ticker_map[ticker] = {
                "ticker": ticker,
                "cik": f"{cik_value:010d}",
                "title": _safe_str(item.get("title") or item.get("name")),
            }

        self._ticker_cache = ticker_map
        self._ticker_cache_ts = now
        return ticker_map

    @staticmethod
    def _extract_recent_filings(submissions: Dict[str, Any]) -> List[Dict[str, Any]]:
        recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        accessions = recent.get("accessionNumber") or []
        primary_docs = recent.get("primaryDocument") or []

        company_cik = _safe_str((submissions or {}).get("cik")).zfill(10)
        archive_cik = str(int(company_cik)) if company_cik.isdigit() else company_cik

        filings: List[Dict[str, Any]] = []
        for form, filed, accession, primary_doc in zip_longest(forms, dates, accessions, primary_docs, fillvalue=""):
            form_value = _safe_str(form)
            filed_value = _safe_str(filed)
            accession_value = _safe_str(accession)
            primary_doc_value = _safe_str(primary_doc)
            if not form_value or not filed_value:
                continue
            accession_compact = accession_value.replace("-", "")
            filing_url = ""
            if archive_cik and accession_compact and primary_doc_value:
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{archive_cik}/{accession_compact}/{primary_doc_value}"
                )
            filings.append(
                {
                    "form": form_value,
                    "filed": filed_value,
                    "filed_dt": _safe_date(filed_value),
                    "url": filing_url,
                }
            )
        return filings

    @staticmethod
    def _extract_company_fact_records(
        companyfacts: Dict[str, Any],
        taxonomy: str,
        concepts: List[str],
    ) -> List[Dict[str, Any]]:
        taxonomy_facts = (((companyfacts or {}).get("facts") or {}).get(taxonomy) or {})
        records: List[Dict[str, Any]] = []
        for concept in concepts:
            concept_payload = taxonomy_facts.get(concept)
            if not isinstance(concept_payload, dict):
                continue
            units = concept_payload.get("units") or {}
            for unit_name, entries in units.items():
                if not isinstance(entries, list):
                    continue
                for item in entries:
                    value = _safe_float(item.get("val"))
                    start = _safe_date(item.get("start"))
                    end = _safe_date(item.get("end"))
                    filed = _safe_date(item.get("filed"))
                    if value is None or filed is None:
                        continue
                    period_type = _classify_fact_period(start, end)
                    records.append(
                        {
                            "concept": concept,
                            "unit": _safe_str(unit_name),
                            "value": value,
                            "form": _safe_str(item.get("form")),
                            "start": start,
                            "end": end,
                            "filed": filed,
                            "period_type": period_type,
                        }
                    )
        records.sort(
            key=lambda item: (
                item.get("filed") or datetime.min,
                item.get("end") or datetime.min,
            ),
            reverse=True,
        )
        return records

    @staticmethod
    def _latest_comparable_pair(records: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        filtered = [item for item in records if item.get("period_type") in {"annual", "quarterly"}]
        if not filtered:
            return None, None

        latest = filtered[0]
        target_period = latest.get("period_type")
        for previous in filtered[1:]:
            if previous.get("period_type") == target_period:
                return latest, previous
        return latest, None

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """Return normalized US fundamental blocks from SEC official filings."""
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        ticker = _safe_str(stock_code).upper()
        if not ticker:
            return result

        try:
            ticker_map = self._load_ticker_map()
        except Exception as exc:
            result["errors"].append(f"sec_ticker_map:{type(exc).__name__}")
            return result

        company = ticker_map.get(ticker)
        if not company:
            result["errors"].append("sec_ticker_not_found")
            return result

        cik = company.get("cik") or ""
        if not cik:
            result["errors"].append("sec_cik_missing")
            return result

        submissions: Dict[str, Any] = {}
        companyfacts: Dict[str, Any] = {}
        try:
            submissions = self._get_json(self._SUBMISSIONS_URL.format(cik=cik))
            result["source_chain"].append("earnings:sec_submissions")
            result["source_chain"].append("institution:sec_submissions")
        except Exception as exc:
            result["errors"].append(f"sec_submissions:{type(exc).__name__}")

        try:
            companyfacts = self._get_json(self._COMPANY_FACTS_URL.format(cik=cik))
            result["source_chain"].append("growth:sec_companyfacts")
        except Exception as exc:
            result["errors"].append(f"sec_companyfacts:{type(exc).__name__}")

        revenue_records = self._extract_company_fact_records(
            companyfacts,
            "us-gaap",
            [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues",
                "SalesRevenueNet",
            ],
        )
        income_records = self._extract_company_fact_records(
            companyfacts,
            "us-gaap",
            ["NetIncomeLoss", "ProfitLoss"],
        )
        equity_records = self._extract_company_fact_records(
            companyfacts,
            "us-gaap",
            [
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ],
        )
        gross_profit_records = self._extract_company_fact_records(
            companyfacts,
            "us-gaap",
            ["GrossProfit"],
        )

        latest_revenue, previous_revenue = self._latest_comparable_pair(revenue_records)
        latest_income, previous_income = self._latest_comparable_pair(income_records)
        latest_gross_profit, _ = self._latest_comparable_pair(gross_profit_records)
        latest_equity = equity_records[0] if equity_records else None

        revenue_yoy = _pct_change(
            latest_revenue.get("value") if latest_revenue else None,
            previous_revenue.get("value") if previous_revenue else None,
        )
        net_income_yoy = _pct_change(
            latest_income.get("value") if latest_income else None,
            previous_income.get("value") if previous_income else None,
        )

        roe = None
        if latest_income and latest_equity and latest_equity.get("value") not in (None, 0):
            income_value = float(latest_income["value"])
            if latest_income.get("period_type") == "quarterly":
                income_value *= 4
            try:
                roe = round(income_value / float(latest_equity["value"]) * 100.0, 2)
            except (TypeError, ValueError, ZeroDivisionError):
                roe = None

        gross_margin = None
        if latest_gross_profit and latest_revenue and latest_revenue.get("value") not in (None, 0):
            try:
                gross_margin = round(
                    float(latest_gross_profit["value"]) / float(latest_revenue["value"]) * 100.0,
                    2,
                )
            except (TypeError, ValueError, ZeroDivisionError):
                gross_margin = None

        result["growth"] = {
            "revenue_yoy": revenue_yoy,
            "net_profit_yoy": net_income_yoy,
            "roe": roe,
            "gross_margin": gross_margin,
            "latest_period_type": latest_revenue.get("period_type") if latest_revenue else None,
        }

        filings = self._extract_recent_filings(submissions)
        financial_forms = [item for item in filings if item["form"] in {"10-Q", "10-Q/A", "10-K", "10-K/A", "20-F", "20-F/A", "6-K"}]
        latest_financial = financial_forms[0] if financial_forms else None
        if latest_financial:
            result["earnings"] = {
                "latest_filing_form": latest_financial["form"],
                "latest_filing_date": latest_financial["filed"],
                "latest_filing_url": latest_financial["url"],
                "recent_financial_forms": [
                    {"form": item["form"], "filed": item["filed"], "url": item["url"]}
                    for item in financial_forms[:5]
                ],
            }

        now = datetime.now()
        insider_forms = [
            item for item in filings
            if item["form"].startswith("4")
            and item.get("filed_dt") is not None
            and (now - item["filed_dt"]).days <= 90
        ]
        ownership_forms = [
            item for item in filings
            if item["form"] in {"13D", "13D/A", "13G", "13G/A"}
            and item.get("filed_dt") is not None
            and (now - item["filed_dt"]).days <= 180
        ]
        if insider_forms or ownership_forms:
            result["institution"] = {
                "insider_form4_count_90d": len(insider_forms),
                "ownership_disclosure_count_180d": len(ownership_forms),
                "latest_insider_filing_date": insider_forms[0]["filed"] if insider_forms else None,
                "latest_ownership_filing_date": ownership_forms[0]["filed"] if ownership_forms else None,
            }

        has_content = any(bool(result[key]) for key in ("growth", "earnings", "institution"))
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_recent_filings(
        self,
        stock_code: str,
        *,
        forms: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Return recent SEC filings for a ticker using the official submissions feed."""
        result: Dict[str, Any] = {
            "status": "not_supported",
            "items": [],
            "source_chain": [],
            "errors": [],
        }

        ticker = _safe_str(stock_code).upper()
        if not ticker:
            return result

        try:
            ticker_map = self._load_ticker_map()
        except Exception as exc:
            result["errors"].append(f"sec_ticker_map:{type(exc).__name__}")
            return result

        company = ticker_map.get(ticker)
        if not company:
            result["errors"].append("sec_ticker_not_found")
            return result

        cik = company.get("cik") or ""
        if not cik:
            result["errors"].append("sec_cik_missing")
            return result

        try:
            submissions = self._get_json(self._SUBMISSIONS_URL.format(cik=cik))
        except Exception as exc:
            result["errors"].append(f"sec_submissions:{type(exc).__name__}")
            return result

        normalized_forms = {str(form).upper() for form in (forms or []) if str(form).strip()}
        filings = self._extract_recent_filings(submissions)
        if normalized_forms:
            filings = [item for item in filings if _safe_str(item.get("form")).upper() in normalized_forms]

        result["items"] = filings[: max(1, int(limit or 1))]
        if result["items"]:
            result["status"] = "ok"
            result["source_chain"].append("sec_submissions")
        return result

    def get_china_exposure_summary(self, stock_code: str) -> Dict[str, Any]:
        """Assess China exposure using official SEC filing text instead of search snippets."""
        result: Dict[str, Any] = {
            "status": "not_supported",
            "level": "unknown",
            "signals": [],
            "evidence": [],
            "filing_form": None,
            "filing_date": None,
            "filing_url": "",
            "source_chain": [],
            "errors": [],
        }

        filings = self.get_recent_filings(
            stock_code,
            forms=["10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "6-K"],
            limit=3,
        )
        result["errors"].extend(filings.get("errors", []))
        if filings.get("status") != "ok":
            return result

        filing = next((item for item in filings.get("items", []) if item.get("url")), None)
        if not filing:
            result["status"] = "partial"
            return result

        filing_url = _safe_str(filing.get("url"))
        result["filing_form"] = filing.get("form")
        result["filing_date"] = filing.get("filed")
        result["filing_url"] = filing_url

        try:
            filing_text = _clean_html_text(self._get_text(filing_url))
            result["source_chain"].append("sec_filing_text")
        except Exception as exc:
            result["errors"].append(f"sec_filing_text:{type(exc).__name__}")
            result["status"] = "partial"
            return result

        signal_patterns = {
            "revenue": [
                r"greater china",
                r"china revenue",
                r"china sales",
                r"net sales.{0,60}china",
                r"china.{0,60}net sales",
                r"mainland china",
                r"prc operations",
            ],
            "supply_chain": [
                r"supply chain.{0,100}china",
                r"china.{0,100}supply chain",
                r"manufactur.{0,80}china",
                r"china.{0,80}manufactur",
                r"assembly.{0,80}china",
                r"china.{0,80}assembly",
                r"suppliers?.{0,80}china",
                r"china.{0,80}suppliers?",
            ],
            "policy": [
                r"export controls?.{0,120}china",
                r"china.{0,120}export controls?",
                r"tariffs?.{0,120}china",
                r"china.{0,120}tariffs?",
                r"prc.{0,80}regulations?",
                r"china.{0,80}regulations?",
                r"sanctions?.{0,120}china",
                r"china.{0,120}sanctions?",
                r"rare earth",
            ],
        }

        evidence_map: Dict[str, List[str]] = {}
        for signal, patterns in signal_patterns.items():
            snippets = _extract_regex_snippets(filing_text, patterns, limit=2)
            if snippets:
                evidence_map[signal] = snippets

        result["signals"] = sorted(evidence_map.keys())
        evidence_lines: List[str] = []
        for signal in result["signals"]:
            label = {
                "revenue": "中国收入/需求",
                "supply_chain": "中国供应链/制造",
                "policy": "中国政策/关税/出口管制",
            }.get(signal, signal)
            for snippet in evidence_map.get(signal, []):
                evidence_lines.append(f"{label}: {snippet[:220]}")
        result["evidence"] = evidence_lines[:4]
        result["status"] = "partial"

        signals = set(result["signals"])
        if "revenue" in signals and ("supply_chain" in signals or "policy" in signals):
            result["level"] = "high"
        elif len(signals) >= 2:
            result["level"] = "medium"
        elif len(signals) == 1:
            result["level"] = "low"
        else:
            result["level"] = "unknown"

        return result

# -*- coding: utf-8 -*-
"""Unit tests for TwelveDataFetcher."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from data_provider.realtime_types import RealtimeSource
from data_provider.twelvedata_fetcher import TwelveDataFetcher


def _make_fetcher() -> TwelveDataFetcher:
    fetcher = TwelveDataFetcher()
    fetcher._available = True
    fetcher._api_key = "demo-key"
    fetcher._timeout_seconds = 10
    fetcher._us_hk_enable = True
    return fetcher


def test_fetch_raw_data_uses_time_series_endpoint_for_hk_stock():
    fetcher = _make_fetcher()

    with patch.object(fetcher, "_resolve_symbol", return_value="00700.HK"), patch.object(
        fetcher,
        "_request",
        return_value={
            "values": [
                {
                    "datetime": "2026-04-07",
                    "open": "500.1",
                    "high": "505.0",
                    "low": "498.0",
                    "close": "503.2",
                    "volume": "1234567",
                }
            ]
        },
    ) as mock_request:
        df = fetcher._fetch_raw_data("0700", "2026-04-01", "2026-04-10")

    assert not df.empty
    mock_request.assert_called_once_with(
        "time_series",
        {
            "symbol": "00700.HK",
            "interval": "1day",
            "start_date": "2026-04-01",
            "end_date": "2026-04-10",
            "outputsize": 5000,
            "order": "asc",
            "format": "JSON",
        },
    )


def test_get_stock_name_uses_symbol_search_result():
    fetcher = _make_fetcher()

    with patch.object(
        fetcher,
        "_find_symbol_entry",
        return_value={"symbol": "00700.HK", "instrument_name": "Tencent Holdings Ltd"},
    ):
        name = fetcher.get_stock_name("HK00700")

    assert name == "Tencent Holdings Ltd"
    assert fetcher._symbol_cache["HK00700"] == "00700.HK"
    assert fetcher._stock_name_cache["HK00700"] == "Tencent Holdings Ltd"


def test_get_realtime_quote_returns_unified_quote():
    fetcher = _make_fetcher()
    fetcher._stock_name_cache["AAPL"] = "Apple Inc."

    with patch.object(fetcher, "_resolve_symbol", return_value="AAPL"), patch.object(
        fetcher,
        "_request",
        side_effect=[
            {
                "name": "Apple Inc.",
                "close": "201.23",
                "open": "199.50",
                "high": "202.00",
                "low": "198.75",
                "previous_close": "198.00",
                "change": "3.23",
                "percent_change": "1.63",
                "volume": "1200000",
                "datetime": "2026-04-08",
                "fifty_two_week": {"high": "220.00", "low": "150.00"},
            },
            {
                "values": [
                    {"datetime": "2026-04-08", "volume": "1100000"},
                    {"datetime": "2026-04-07", "volume": "1000000"},
                    {"datetime": "2026-04-04", "volume": "900000"},
                    {"datetime": "2026-04-03", "volume": "800000"},
                    {"datetime": "2026-04-02", "volume": "700000"},
                    {"datetime": "2026-04-01", "volume": "600000"},
                ]
            },
            {"stock_statistics": {"float_shares": "60000000"}},
        ],
    ) as mock_request:
        quote = fetcher.get_realtime_quote("AAPL")

    assert quote is not None
    assert quote.code == "AAPL"
    assert quote.name == "Apple Inc."
    assert quote.price == 201.23
    assert quote.change_pct == 1.63
    assert quote.change_amount == 3.23
    assert quote.volume == 1200000
    assert quote.volume_ratio == 1.5
    assert quote.turnover_rate == 2.0
    assert quote.open_price == 199.5
    assert quote.high == 202.0
    assert quote.low == 198.75
    assert quote.pre_close == 198.0
    assert quote.amplitude == pytest.approx(1.6414, rel=1e-4)
    assert quote.high_52w == 220.0
    assert quote.low_52w == 150.0
    assert quote.source == RealtimeSource.TWELVEDATA
    assert mock_request.call_args_list == [
        (("quote", {"symbol": "AAPL", "dp": 4}),),
        (
            (
                "time_series",
                {
                    "symbol": "AAPL",
                    "interval": "1day",
                    "outputsize": 6,
                    "order": "desc",
                    "format": "JSON",
                },
            ),
        ),
        (( "statistics", {"symbol": "AAPL"}),),
    ]


def test_get_realtime_quote_leaves_turnover_rate_empty_when_statistics_unavailable():
    fetcher = _make_fetcher()
    fetcher._stock_name_cache["AAPL"] = "Apple Inc."

    with patch.object(fetcher, "_resolve_symbol", return_value="AAPL"), patch.object(
        fetcher,
        "_request",
        side_effect=[
            {
                "name": "Apple Inc.",
                "close": "201.23",
                "volume": "1200000",
                "datetime": "2026-04-08",
            },
            {
                "values": [
                    {"datetime": "2026-04-08", "volume": "1100000"},
                    {"datetime": "2026-04-07", "volume": "1000000"},
                    {"datetime": "2026-04-04", "volume": "900000"},
                    {"datetime": "2026-04-03", "volume": "800000"},
                    {"datetime": "2026-04-02", "volume": "700000"},
                    {"datetime": "2026-04-01", "volume": "600000"},
                ]
            },
            Exception("statistics plan gated"),
        ],
    ):
        quote = fetcher.get_realtime_quote("AAPL")

    assert quote is not None
    assert quote.volume_ratio == 1.5
    assert quote.turnover_rate is None


def test_get_realtime_quote_returns_none_when_quote_has_no_price():
    fetcher = _make_fetcher()

    with patch.object(fetcher, "_resolve_symbol", return_value="AAPL"), patch.object(
        fetcher,
        "_request",
        return_value={"name": "Apple Inc."},
    ):
        quote = fetcher.get_realtime_quote("AAPL")

    assert quote is None


def test_normalize_data_rejects_missing_volume():
    fetcher = _make_fetcher()

    df = pd.DataFrame(
        [
            {
                "datetime": "2026-04-07",
                "open": "100.0",
                "high": "101.0",
                "low": "99.0",
                "close": "100.5",
            }
        ]
    )

    with pytest.raises(Exception, match="缺少关键字段"):
        fetcher._normalize_data(df, "AAPL")

# --- /statistics 403 处理 ---------------------------------------------------


def _forbidden_response(endpoint: str = "statistics"):
    import requests

    response = requests.Response()
    response.status_code = 403
    response.url = f"https://api.twelvedata.com/{endpoint}"
    response._content = b'{"code":403,"message":"Forbidden","status":"error"}'
    return response


def test_statistics_403_is_not_retried_and_marked_unsupported():
    fetcher = _make_fetcher()

    with patch("data_provider.twelvedata_fetcher.requests.get") as mock_get:
        mock_get.return_value = _forbidden_response()

        assert fetcher._compute_turnover_rate("AAPL", 1_000_000) is None
        # 403 属于套餐问题，tenacity 不应重试
        assert mock_get.call_count == 1
        assert fetcher._endpoint_unsupported("statistics") is True

        # 后续标的直接跳过，不再发请求
        assert fetcher._compute_turnover_rate("MSFT", 2_000_000) is None
        assert mock_get.call_count == 1


def test_statistics_success_is_still_used():
    fetcher = _make_fetcher()

    with patch.object(
        fetcher,
        "_request",
        return_value={"stock_statistics": {"float_shares": "1000000000"}},
    ) as mock_request:
        rate = fetcher._compute_turnover_rate("AAPL", 10_000_000)

    assert mock_request.call_count == 1
    assert rate == pytest.approx(1.0)
    assert fetcher._endpoint_unsupported("statistics") is False

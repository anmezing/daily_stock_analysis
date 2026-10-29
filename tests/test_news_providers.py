# -*- coding: utf-8 -*-
"""Finnhub 公司新闻与 Google News RSS 兜底源的离线测试。"""

import sys
from unittest.mock import MagicMock, patch

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()

from src.search_service import FinnhubNewsProvider, RSSNewsProvider  # noqa: E402


class _FakeResponse:
    def __init__(self, *, payload=None, content: bytes = b"", status: int = 200):
        self._payload = payload
        self.content = content
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# --- Finnhub --------------------------------------------------------------


def test_finnhub_symbol_mapping():
    to_symbol = FinnhubNewsProvider._to_finnhub_symbol

    assert to_symbol("AAPL") == "AAPL"
    assert to_symbol("aapl") == "AAPL"
    assert to_symbol("hk00700") == "0700.HK"
    assert to_symbol("HK09988") == "9988.HK"
    assert to_symbol("0700.HK") == "0700.HK"
    assert to_symbol("0700") == "0700.HK"
    assert to_symbol("") is None
    assert to_symbol(None) is None


def test_finnhub_search_maps_company_news():
    provider = FinnhubNewsProvider(["key-1"])
    payload = [
        {
            "category": "company",
            "datetime": 1793000000,
            "headline": "Apple ships new laptop",
            "id": 1,
            "source": "Reuters",
            "summary": "Apple announced a touchscreen MacBook.",
            "url": "https://example.com/a",
        },
        {
            "category": "company",
            "datetime": 1793000001,
            "headline": "Second story",
            "source": "CNBC",
            "summary": "Another one",
            "url": "https://example.com/b",
        },
    ]

    with patch(
        "src.search_service.requests.get", return_value=_FakeResponse(payload=payload)
    ) as mock_get:
        response = provider.search("AAPL latest news", max_results=1, days=3, stock_code="AAPL")

    assert response.success is True
    assert len(response.results) == 1
    first = response.results[0]
    assert first.title == "Apple ships new laptop"
    assert first.source == "Reuters"
    assert first.url == "https://example.com/a"
    assert first.published_date is not None
    assert mock_get.call_args.kwargs["params"]["symbol"] == "AAPL"


def test_finnhub_requires_stock_code():
    provider = FinnhubNewsProvider(["key-1"])

    with patch("src.search_service.requests.get") as mock_get:
        response = provider.search("some query", max_results=3, days=3)

    assert response.success is False
    assert "缺少股票代码" in (response.error_message or "")
    mock_get.assert_not_called()


def test_finnhub_handles_error_payload():
    provider = FinnhubNewsProvider(["key-1"])

    with patch(
        "src.search_service.requests.get",
        return_value=_FakeResponse(payload={"error": "Invalid API key"}),
    ):
        response = provider.search("AAPL", max_results=3, days=3, stock_code="AAPL")

    assert response.success is False
    assert "Invalid API key" in (response.error_message or "")


# --- Google News RSS ------------------------------------------------------


_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Apple stock climbs - CNBC</title>
    <link>https://news.example.com/a</link>
    <pubDate>Wed, 28 Oct 2026 22:09:52 GMT</pubDate>
    <description>&lt;a href="x"&gt;Apple&lt;/a&gt; shares rose today.</description>
    <source url="https://cnbc.com">CNBC</source>
  </item>
  <item>
    <title>Second item</title>
    <link>https://news.example.com/b</link>
    <pubDate>not-a-date</pubDate>
    <description>plain text</description>
  </item>
</channel></rss>
"""


def test_rss_parses_items_and_dates():
    provider = RSSNewsProvider()

    with patch(
        "src.search_service.requests.get", return_value=_FakeResponse(content=_RSS)
    ) as mock_get:
        response = provider.search("AAPL stock latest news", max_results=5, days=3)

    assert response.success is True
    assert len(response.results) == 2

    first = response.results[0]
    assert first.title == "Apple stock climbs - CNBC"
    assert first.source == "CNBC"
    assert first.published_date == "2026-10-28"
    assert "<a" not in first.snippet

    # 非法时间不应抛异常，只是留空
    assert response.results[1].published_date is None
    assert mock_get.call_args.kwargs["params"]["q"] == "AAPL stock latest news"


def test_rss_respects_max_results():
    provider = RSSNewsProvider()

    with patch("src.search_service.requests.get", return_value=_FakeResponse(content=_RSS)):
        response = provider.search("q", max_results=1, days=3)

    assert len(response.results) == 1


def test_rss_reports_http_failure():
    provider = RSSNewsProvider()

    with patch(
        "src.search_service.requests.get", return_value=_FakeResponse(status=503, content=b"")
    ):
        response = provider.search("q", max_results=3, days=3)

    assert response.success is False
    assert "Google News RSS 失败" in (response.error_message or "")


def test_rss_is_always_available_without_key():
    provider = RSSNewsProvider()
    assert provider.is_available is True

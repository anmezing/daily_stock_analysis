# -*- coding: utf-8 -*-
"""
Regression tests for Tavily news-mode date mapping (Issue #782).
"""

import sys
import unittest
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import MagicMock, patch

# Mock newspaper before search_service import (optional dependency)
if "newspaper" not in sys.modules:
    mock_np = MagicMock()
    mock_np.Article = MagicMock()
    mock_np.Config = MagicMock()
    sys.modules["newspaper"] = mock_np

from src.search_service import SearchService, TavilySearchProvider


class _FakeTavilyClient:
    response_payload = {"results": []}
    init_api_keys = []
    search_calls = []

    def __init__(self, api_key=None, **_kwargs):
        type(self).init_api_keys.append(api_key)

    def search(self, **kwargs):
        type(self).search_calls.append(kwargs)
        return type(self).response_payload

    @classmethod
    def reset(cls) -> None:
        cls.response_payload = {"results": []}
        cls.init_api_keys = []
        cls.search_calls = []


def _fake_tavily_module() -> ModuleType:
    module = ModuleType("tavily")
    module.TavilyClient = _FakeTavilyClient
    return module


class TestTavilySearchProvider(unittest.TestCase):
    """Tests for Tavily provider-specific request and mapping behavior."""

    def _patch_tavily(self, payload):
        _FakeTavilyClient.reset()
        _FakeTavilyClient.response_payload = payload
        return patch.dict(sys.modules, {"tavily": _fake_tavily_module()})

    def test_provider_uses_news_topic_when_explicitly_requested(self) -> None:
        published_text = "2026-03-20T09:30:00Z"
        provider = TavilySearchProvider(["dummy_key"])

        with self._patch_tavily(
            {
                "results": [
                    {
                        "title": "Alibaba earnings beat",
                        "url": "https://example.com/alibaba-earnings",
                        "content": "Fresh coverage",
                        "published_date": published_text,
                    },
                ]
            }
        ):
            resp = provider.search("BABA latest news", max_results=5, days=3, topic="news")

        self.assertTrue(resp.success)
        self.assertEqual(_FakeTavilyClient.init_api_keys, ["dummy_key"])
        self.assertEqual(len(_FakeTavilyClient.search_calls), 1)
        self.assertEqual(_FakeTavilyClient.search_calls[0]["topic"], "news")
        self.assertEqual(_FakeTavilyClient.search_calls[0]["days"], 3)
        self.assertEqual(_FakeTavilyClient.search_calls[0]["max_results"], 5)
        self.assertEqual(_FakeTavilyClient.search_calls[0]["search_depth"], "advanced")
        self.assertEqual(len(resp.results), 1)
        self.assertEqual(resp.results[0].published_date, published_text)
        self.assertEqual(resp.results[0].url, "https://example.com/alibaba-earnings")

    def test_provider_supports_publishedDate_variant(self) -> None:
        provider = TavilySearchProvider(["dummy_key"])

        with self._patch_tavily(
            {
                "results": [
                    {
                        "title": "Alibaba guidance update",
                        "url": "https://example.com/alibaba-guidance",
                        "content": "Fresh coverage",
                        "publishedDate": "2026-03-20T11:00:00Z",
                    }
                ]
            }
        ):
            resp = provider.search("BABA latest news", max_results=5, days=7, topic="news")

        self.assertTrue(resp.success)
        self.assertEqual(len(resp.results), 1)
        self.assertEqual(resp.results[0].published_date, "2026-03-20T11:00:00Z")

    def test_non_news_search_paths_do_not_force_news_topic(self) -> None:
        provider = TavilySearchProvider(["dummy_key"])

        with self._patch_tavily(
            {
                "results": [
                    {
                        "title": "Alibaba price action",
                        "url": "https://example.com/alibaba-price",
                        "content": "General search result",
                    }
                ]
            }
        ):
            resp = provider.search("BABA stock price", max_results=3)

        self.assertTrue(resp.success)
        self.assertEqual(len(_FakeTavilyClient.search_calls), 1)
        self.assertNotIn("topic", _FakeTavilyClient.search_calls[0])

    def test_search_stock_news_keeps_tavily_results_with_supported_date_fields(self) -> None:
        published_dt = datetime.now(timezone.utc).replace(microsecond=0)
        published_text = published_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        expected_date = published_dt.astimezone().date().isoformat()

        for field_name in ("published_date", "publishedDate"):
            with self.subTest(field_name=field_name):
                with self._patch_tavily(
                    {
                        "results": [
                            {
                                "title": f"Fresh article via {field_name}",
                                "url": "https://example.com/fresh-article",
                                "content": "Fresh coverage",
                                field_name: published_text,
                            }
                        ]
                    }
                ):
                    service = SearchService(
                        tavily_keys=["dummy_key"],
                        searxng_public_instances_enabled=False,
                        news_max_age_days=3,
                        news_strategy_profile="short",
                    )
                    resp = service.search_stock_news("BABA", "阿里巴巴", max_results=3)

                self.assertTrue(resp.success)
                self.assertEqual(len(resp.results), 1)
                self.assertEqual(resp.results[0].published_date, expected_date)
                self.assertEqual(_FakeTavilyClient.search_calls[0]["topic"], "news")

    def test_search_stock_events_does_not_force_news_topic(self) -> None:
        with self._patch_tavily(
            {
                "results": [
                    {
                        "title": "Alibaba quarterly results",
                        "url": "https://example.com/alibaba-event",
                        "content": "Event coverage",
                    }
                ]
            }
        ):
            service = SearchService(
                tavily_keys=["dummy_key"],
                searxng_public_instances_enabled=False,
            )
            resp = service.search_stock_events("BABA", "阿里巴巴")

        self.assertTrue(resp.success)
        self.assertEqual(len(resp.results), 1)
        self.assertNotIn("topic", _FakeTavilyClient.search_calls[0])

    def test_search_comprehensive_intel_uses_dimension_specific_topic_for_tavily(self) -> None:
        published_dt = datetime.now(timezone.utc).replace(microsecond=0)
        published_text = published_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        with self._patch_tavily(
            {
                "results": [
                    {
                        "title": "Alibaba intel article",
                        "url": "https://example.com/alibaba-intel",
                        "content": "Recent intel",
                        "published_date": published_text,
                    }
                ]
            }
        ):
            service = SearchService(
                tavily_keys=["dummy_key"],
                searxng_public_instances_enabled=False,
                news_max_age_days=3,
                news_strategy_profile="short",
            )
            intel = service.search_comprehensive_intel("BABA", "阿里巴巴", max_searches=2)

        self.assertIn("latest_news", intel)
        self.assertIn("market_analysis", intel)
        self.assertGreaterEqual(len(_FakeTavilyClient.search_calls), 2)
        self.assertEqual(_FakeTavilyClient.search_calls[0]["topic"], "news")
        self.assertNotIn("topic", _FakeTavilyClient.search_calls[1])

    def test_search_comprehensive_intel_etf_risk_check_does_not_force_news_topic(self) -> None:
        published_dt = datetime.now(timezone.utc).replace(microsecond=0)
        published_text = published_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        with self._patch_tavily(
            {
                "results": [
                    {
                        "title": "ETF intel article",
                        "url": "https://example.com/etf-intel",
                        "content": "Recent ETF coverage",
                        "published_date": published_text,
                    }
                ]
            }
        ):
            service = SearchService(
                tavily_keys=["dummy_key"],
                searxng_public_instances_enabled=False,
                news_max_age_days=3,
                news_strategy_profile="short",
            )
            intel = service.search_comprehensive_intel("510300", "沪深300ETF", max_searches=3)

        self.assertIn("latest_news", intel)
        self.assertIn("market_analysis", intel)
        self.assertIn("risk_check", intel)
        self.assertGreaterEqual(len(_FakeTavilyClient.search_calls), 3)
        self.assertEqual(_FakeTavilyClient.search_calls[0]["topic"], "news")
        self.assertNotIn("topic", _FakeTavilyClient.search_calls[1])
        self.assertNotIn("topic", _FakeTavilyClient.search_calls[2])

    def test_search_comprehensive_intel_non_etf_risk_check_stays_in_news_topic(self) -> None:
        published_dt = datetime.now(timezone.utc).replace(microsecond=0)
        published_text = published_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        with self._patch_tavily(
            {
                "results": [
                    {
                        "title": "Moutai intel article",
                        "url": "https://example.com/moutai-intel",
                        "content": "Recent non-ETF coverage",
                        "published_date": published_text,
                    }
                ]
            }
        ):
            service = SearchService(
                tavily_keys=["dummy_key"],
                searxng_public_instances_enabled=False,
                news_max_age_days=3,
                news_strategy_profile="short",
            )
            intel = service.search_comprehensive_intel("600519", "贵州茅台", max_searches=3)

        self.assertIn("latest_news", intel)
        self.assertIn("market_analysis", intel)
        self.assertIn("risk_check", intel)
        self.assertGreaterEqual(len(_FakeTavilyClient.search_calls), 3)
        self.assertEqual(_FakeTavilyClient.search_calls[0]["topic"], "news")
        self.assertNotIn("topic", _FakeTavilyClient.search_calls[1])
        self.assertEqual(_FakeTavilyClient.search_calls[2]["topic"], "news")


if __name__ == "__main__":
    unittest.main()


class TestTavilyCustomEndpoint(unittest.TestCase):
    """配置 TAVILY_BASE_URL 时改走 Tavily 风格 REST 中转。"""

    def _fake_response(self, payload):
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    def test_custom_endpoint_posts_to_base_search(self) -> None:
        provider = TavilySearchProvider(
            ["th-relay-token"], base_url="https://tavily.example.com/api/tavily/"
        )
        payload = {
            "results": [
                {"title": "T", "url": "https://example.com/a", "content": "body"},
            ]
        }

        with patch(
            "src.search_service.requests.post", return_value=self._fake_response(payload)
        ) as mock_post:
            resp = provider.search("BABA news", max_results=3, days=2, topic="news")

        self.assertTrue(resp.success)
        self.assertEqual(len(resp.results), 1)

        call = mock_post.call_args
        self.assertEqual(call.args[0], "https://tavily.example.com/api/tavily/search")
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer th-relay-token")
        body = call.kwargs["json"]
        self.assertEqual(body["query"], "BABA news")
        self.assertEqual(body["topic"], "news")
        self.assertEqual(body["max_results"], 3)
        self.assertEqual(body["days"], 2)
        # 中转无法自定义 Header 时可用 body 里的 api_key
        self.assertEqual(body["api_key"], "th-relay-token")

    def test_custom_endpoint_reports_http_failure(self) -> None:
        provider = TavilySearchProvider(["k"], base_url="https://relay.example.com")
        failing = MagicMock()
        failing.raise_for_status.side_effect = RuntimeError("401 Unauthorized")

        with patch("src.search_service.requests.post", return_value=failing):
            resp = provider.search("q", max_results=3, days=2)

        self.assertFalse(resp.success)
        self.assertIn("401", resp.error_message or "")

    def test_base_url_read_from_env_and_stripped(self) -> None:
        with patch.dict("os.environ", {"TAVILY_BASE_URL": "https://relay.example.com/v1/"}):
            provider = TavilySearchProvider(["k"])
        self.assertEqual(provider._base_url, "https://relay.example.com/v1")
        self.assertTrue(provider.is_custom_endpoint)

    def test_official_endpoint_is_not_custom(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("TAVILY_BASE_URL", None)
            provider = TavilySearchProvider(["k"])
        self.assertEqual(provider._base_url, TavilySearchProvider.DEFAULT_BASE_URL)
        self.assertFalse(provider.is_custom_endpoint)

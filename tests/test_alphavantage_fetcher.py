# -*- coding: utf-8 -*-
"""AlphaVantageFetcher 离线单元测试（不发起真实网络请求）。"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _make_mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _daily_payload():
    return {
        "Meta Data": {"2. Symbol": "AAPL"},
        "Time Series (Daily)": {
            "2024-06-11": {
                "1. open": "151.0",
                "2. high": "153.0",
                "3. low": "150.0",
                "4. close": "152.0",
                "5. volume": "1200000",
            },
            "2024-06-10": {
                "1. open": "149.5",
                "2. high": "151.0",
                "3. low": "149.0",
                "4. close": "150.0",
                "5. volume": "1000000",
            },
        },
    }


class TestAlphaVantageNormalize(unittest.TestCase):
    """校验 TIME_SERIES_DAILY 响应到标准列的映射。"""

    def setUp(self):
        from data_provider.alphavantage_fetcher import AlphaVantageFetcher
        self.fetcher = AlphaVantageFetcher()

    def test_normalize_sorts_ascending(self):
        import pandas as pd
        raw = pd.DataFrame(
            {
                '1. open': [151.0, 149.5],
                '2. high': [153.0, 151.0],
                '3. low': [150.0, 149.0],
                '4. close': [152.0, 150.0],
                '5. volume': [1200000, 1000000],
            },
            index=pd.to_datetime(['2024-06-11', '2024-06-10']),
        )
        result = self.fetcher._normalize_data(raw, 'AAPL')
        self.assertIn('date', result.columns)
        self.assertAlmostEqual(result.iloc[0]['close'], 150.0)
        self.assertAlmostEqual(result.iloc[1]['close'], 152.0)
        self.assertEqual(result.iloc[0]['code'], 'AAPL')

    def test_normalize_pct_chg_uses_ascending_order(self):
        import pandas as pd
        raw = pd.DataFrame(
            {
                '1. open': [151.0, 149.5],
                '2. high': [153.0, 151.0],
                '3. low': [150.0, 149.0],
                '4. close': [153.0, 150.0],
                '5. volume': [1200000, 1000000],
            },
            index=pd.to_datetime(['2024-06-11', '2024-06-10']),
        )
        result = self.fetcher._normalize_data(raw, 'AAPL')
        self.assertAlmostEqual(result.iloc[0]['pct_chg'], 0.0)
        self.assertAlmostEqual(result.iloc[1]['pct_chg'], 2.0)

    def test_normalize_empty_df(self):
        import pandas as pd
        self.assertTrue(self.fetcher._normalize_data(pd.DataFrame(), 'AAPL').empty)


class TestAlphaVantageMarketScope(unittest.TestCase):
    """只有美股走这个数据源。"""

    def setUp(self):
        from data_provider.alphavantage_fetcher import AlphaVantageFetcher
        self.fetcher = AlphaVantageFetcher()
        self.fetcher._api_key = "test_key"

    def test_rejects_non_us_daily(self):
        from data_provider.base import DataFetchError
        for code in ('600519', 'hk00700', '2330.TW', '7203.T', '005930.KS'):
            with self.assertRaises(DataFetchError):
                self.fetcher._fetch_raw_data(code, '2024-06-10', '2024-06-11')

    def test_rejects_non_us_realtime(self):
        for code in ('600519', 'hk00700', '2330.TW'):
            self.assertIsNone(self.fetcher.get_realtime_quote(code))

    def test_rejects_non_us_name(self):
        self.assertIsNone(self.fetcher.get_stock_name('600519'))


class TestAlphaVantageFetchRaw(unittest.TestCase):
    """校验 HTTP 层与限流/错误分支。"""

    def setUp(self):
        from data_provider.alphavantage_fetcher import AlphaVantageFetcher
        self.fetcher = AlphaVantageFetcher()
        self.fetcher._api_key = "test_key"

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_fetch_raw_success_filters_range(self, mock_get):
        mock_get.return_value = _make_mock_response(_daily_payload())
        df = self.fetcher._fetch_raw_data('AAPL', '2024-06-11', '2024-06-11')
        self.assertEqual(len(df), 1)
        self.assertIn('4. close', df.columns)

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_fetch_raw_empty_range(self, mock_get):
        from data_provider.base import DataFetchError
        mock_get.return_value = _make_mock_response(_daily_payload())
        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data('AAPL', '2024-01-01', '2024-01-02')

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_fetch_raw_rate_limited(self, mock_get):
        from data_provider.base import DataFetchError
        mock_get.return_value = _make_mock_response(
            {'Note': 'Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day.'}
        )
        with self.assertRaises(DataFetchError) as ctx:
            self.fetcher._fetch_raw_data('AAPL', '2024-06-10', '2024-06-11')
        self.assertIn('限流', str(ctx.exception))

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_fetch_raw_api_error(self, mock_get):
        from data_provider.base import DataFetchError
        mock_get.return_value = _make_mock_response({'Error Message': 'Invalid API call.'})
        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data('AAPL', '2024-06-10', '2024-06-11')

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_fetch_raw_http_error(self, mock_get):
        from data_provider.base import DataFetchError
        mock_get.side_effect = Exception("connection timeout")
        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data('AAPL', '2024-06-10', '2024-06-11')

    def test_fetch_raw_without_key(self):
        from data_provider.base import DataFetchError
        self.fetcher._api_key = None
        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data('AAPL', '2024-06-10', '2024-06-11')


class TestAlphaVantageRealtimeQuote(unittest.TestCase):
    """校验 GLOBAL_QUOTE 字段映射。"""

    def setUp(self):
        from data_provider.alphavantage_fetcher import AlphaVantageFetcher
        self.fetcher = AlphaVantageFetcher()
        self.fetcher._api_key = "test_key"

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_realtime_quote_us_stock(self, mock_get):
        mock_get.return_value = _make_mock_response({
            "Global Quote": {
                "01. symbol": "AAPL",
                "02. open": "149.0000",
                "03. high": "151.0000",
                "04. low": "148.0000",
                "05. price": "150.0000",
                "06. volume": "5000000",
                "07. latest trading day": "2024-06-11",
                "08. previous close": "148.0000",
                "09. change": "2.0000",
                "10. change percent": "1.3514%",
            }
        })
        quote = self.fetcher.get_realtime_quote('AAPL')
        self.assertIsNotNone(quote)
        self.assertEqual(quote.code, 'AAPL')
        self.assertAlmostEqual(quote.price, 150.0)
        self.assertAlmostEqual(quote.change_pct, 1.35)
        self.assertAlmostEqual(quote.change_amount, 2.0)
        self.assertEqual(quote.volume, 5000000)

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_realtime_quote_empty(self, mock_get):
        mock_get.return_value = _make_mock_response({"Global Quote": {}})
        self.assertIsNone(self.fetcher.get_realtime_quote('AAPL'))

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_realtime_quote_http_failure(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        self.assertIsNone(self.fetcher.get_realtime_quote('AAPL'))


class TestAlphaVantageStockName(unittest.TestCase):
    """校验 SYMBOL_SEARCH。"""

    def setUp(self):
        from data_provider.alphavantage_fetcher import AlphaVantageFetcher
        self.fetcher = AlphaVantageFetcher()
        self.fetcher._api_key = "test_key"

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_get_stock_name_found(self, mock_get):
        mock_get.return_value = _make_mock_response({
            "bestMatches": [
                {"1. symbol": "AAPL", "2. name": "Apple Inc"},
                {"1. symbol": "AAPL.MEX", "2. name": "Apple Inc"},
            ]
        })
        self.assertEqual(self.fetcher.get_stock_name('AAPL'), 'Apple Inc')

    @patch('data_provider.alphavantage_fetcher.requests.get')
    def test_get_stock_name_empty(self, mock_get):
        mock_get.return_value = _make_mock_response({"bestMatches": []})
        self.assertIsNone(self.fetcher.get_stock_name('NOTEXIST'))


class TestAlphaVantageRegistration(unittest.TestCase):
    """校验注册与链路顺序。"""

    @patch('src.config.get_config')
    def test_registered_with_key(self, mock_config):
        mock_config.return_value = MagicMock(
            alphavantage_api_key='av-test',
            finnhub_api_key=None,
            twelvedata_api_key=None,
            tushare_token=None,
            longbridge_app_key=None,
            longbridge_app_secret=None,
            longbridge_access_token=None,
            tickflow_api_key=None,
        )
        from data_provider.base import DataFetcherManager
        mgr = DataFetcherManager()
        self.assertIn('AlphaVantageFetcher', [f.name for f in mgr._fetchers])

    @patch('src.config.get_config')
    def test_not_registered_without_key(self, mock_config):
        mock_config.return_value = MagicMock(
            alphavantage_api_key=None,
            finnhub_api_key=None,
            twelvedata_api_key=None,
            tushare_token=None,
            longbridge_app_key=None,
            longbridge_app_secret=None,
            longbridge_access_token=None,
            tickflow_api_key=None,
        )
        from data_provider.base import DataFetcherManager
        mgr = DataFetcherManager()
        self.assertNotIn('AlphaVantageFetcher', [f.name for f in mgr._fetchers])

    @patch('src.config.get_config')
    def test_us_chain_orders_finnhub_before_alphavantage(self, mock_config):
        mock_config.return_value = MagicMock(
            alphavantage_api_key='av-test',
            finnhub_api_key='sk-test',
            twelvedata_api_key=None,
            tushare_token=None,
            longbridge_app_key=None,
            longbridge_app_secret=None,
            longbridge_access_token=None,
            tickflow_api_key=None,
        )
        from data_provider.base import DataFetcherManager
        mgr = DataFetcherManager()
        chain = [f.name for f in mgr._get_us_hk_priority_chain('AAPL')]
        self.assertIn('FinnhubFetcher', chain)
        self.assertIn('AlphaVantageFetcher', chain)
        self.assertLess(chain.index('FinnhubFetcher'), chain.index('AlphaVantageFetcher'))
        self.assertLess(chain.index('AlphaVantageFetcher'), chain.index('YfinanceFetcher'))

    @patch('src.config.get_config')
    def test_hk_chain_does_not_route_to_alphavantage(self, mock_config):
        mock_config.return_value = MagicMock(
            alphavantage_api_key='av-test',
            finnhub_api_key='sk-test',
            twelvedata_api_key=None,
            tushare_token=None,
            longbridge_app_key=None,
            longbridge_app_secret=None,
            longbridge_access_token=None,
            tickflow_api_key=None,
        )
        from data_provider.base import DataFetcherManager
        mgr = DataFetcherManager()
        chain = [f.name for f in mgr._get_us_hk_priority_chain('hk00700')]
        self.assertNotIn('AlphaVantageFetcher', chain)
        self.assertNotIn('FinnhubFetcher', chain)


if __name__ == '__main__':
    unittest.main()

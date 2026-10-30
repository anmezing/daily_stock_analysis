# -*- coding: utf-8 -*-
"""FinnhubFetcher 离线单元测试（不发起真实网络请求）。"""

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


class TestFinnhubFetcherNormalize(unittest.TestCase):
    """校验 Finnhub 原始 candle 响应到标准列的映射。"""

    def setUp(self):
        from data_provider.finnhub_fetcher import FinnhubFetcher
        self.fetcher = FinnhubFetcher()

    def test_normalize_candle_data(self):
        import pandas as pd
        raw = pd.DataFrame({
            'c': [150.0, 152.0, 148.5],
            'h': [151.0, 153.0, 150.0],
            'l': [149.0, 150.0, 147.0],
            'o': [149.5, 151.0, 149.0],
            't': [1718000000, 1718086400, 1718172800],
            'v': [1000000, 1200000, 900000],
        })
        result = self.fetcher._normalize_data(raw, 'AAPL')
        self.assertIn('date', result.columns)
        self.assertIn('close', result.columns)
        self.assertAlmostEqual(result.iloc[0]['close'], 150.0)
        self.assertEqual(result.iloc[0]['code'], 'AAPL')

    def test_normalize_calculates_pct_chg(self):
        import pandas as pd
        raw = pd.DataFrame({
            'c': [150.0, 153.0],
            'h': [151.0, 154.0],
            'l': [149.0, 152.0],
            'o': [149.5, 152.0],
            't': [1718000000, 1718086400],
            'v': [1000000, 1200000],
        })
        result = self.fetcher._normalize_data(raw, 'AAPL')
        self.assertAlmostEqual(result.iloc[1]['pct_chg'], 2.0)

    def test_normalize_empty_df(self):
        import pandas as pd
        raw = pd.DataFrame()
        result = self.fetcher._normalize_data(raw, 'AAPL')
        self.assertTrue(result.empty)


class TestFinnhubFetcherMarketScope(unittest.TestCase):
    """只有美股走这个数据源，其它市场必须直接放弃。"""

    def setUp(self):
        from data_provider.finnhub_fetcher import FinnhubFetcher
        self.fetcher = FinnhubFetcher()
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


class TestFinnhubFetcherFetchRaw(unittest.TestCase):
    """校验 HTTP 层行为。"""

    def setUp(self):
        from data_provider.finnhub_fetcher import FinnhubFetcher
        self.fetcher = FinnhubFetcher()
        self.fetcher._api_key = "test_key"

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_fetch_raw_success(self, mock_get):
        mock_get.return_value = _make_mock_response({
            'c': [150.0],
            'h': [151.0],
            'l': [149.0],
            'o': [149.5],
            't': [1718000000],
            'v': [1000000],
            's': 'ok',
        })
        df = self.fetcher._fetch_raw_data('AAPL', '2024-06-10', '2024-06-11')
        self.assertFalse(df.empty)
        self.assertIn('c', df.columns)

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_fetch_raw_empty_response(self, mock_get):
        from data_provider.base import DataFetchError
        mock_get.return_value = _make_mock_response({
            'c': [], 'h': [], 'l': [], 'o': [], 't': [], 'v': [], 's': 'no_data',
        })
        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data('INVALIDX', '2024-06-10', '2024-06-11')

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_candle_403_disables_the_endpoint_for_the_process(self, mock_get):
        """免费档不含 /stock/candle：命中 403 后不该每只股票再撞一次。"""
        from data_provider.base import DataFetchError

        mock_get.return_value = _make_mock_response({}, status_code=403)

        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data('AAPL', '2024-06-10', '2024-06-11')

        self.assertTrue(self.fetcher._candle_forbidden)
        self.assertEqual(mock_get.call_count, 1)

        # 后续标的直接跳过，不再发请求
        with self.assertRaises(DataFetchError):
            self.fetcher._fetch_raw_data('MSFT', '2024-06-10', '2024-06-11')
        self.assertEqual(mock_get.call_count, 1)

    @patch('data_provider.finnhub_fetcher.requests.get')
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


class TestFinnhubMultiKey(unittest.TestCase):
    """多 Key 轮询：429 只冷却当前 Key，换下一个继续。"""

    def _fetcher(self, *keys):
        from unittest.mock import MagicMock, patch as _patch

        from data_provider.finnhub_fetcher import FinnhubFetcher

        with _patch('src.config.get_config') as mock_config:
            mock_config.return_value = MagicMock(
                finnhub_api_keys=list(keys),
                finnhub_api_key=keys[0] if keys else None,
            )
            fetcher = FinnhubFetcher()
        fetcher._api_keys = list(keys)
        fetcher._key_index = 0
        fetcher._key_cooldown = {}
        fetcher._dead_keys = set()
        return fetcher

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_second_key_takes_over_after_429(self, mock_get):
        fetcher = self._fetcher('key-aaaa1111', 'key-bbbb2222')
        mock_get.side_effect = [
            _make_mock_response({}, status_code=429),
            _make_mock_response({'c': 250.0}),
        ]

        data = fetcher._request_json('quote', {'symbol': 'AAPL'})

        self.assertEqual(data, {'c': 250.0})
        sent = [call.kwargs['params']['token'] for call in mock_get.call_args_list]
        self.assertEqual(sent, ['key-aaaa1111', 'key-bbbb2222'])
        self.assertGreater(fetcher._key_cooldown['key-aaaa1111'], 0)
        # 只有一个 Key 冷却，整体仍可用
        self.assertFalse(fetcher.is_temporarily_unavailable())

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_all_keys_throttled_marks_source_unavailable(self, mock_get):
        from data_provider.base import DataSourceUnavailableError

        fetcher = self._fetcher('key-aaaa1111', 'key-bbbb2222')
        mock_get.return_value = _make_mock_response({}, status_code=429)

        with self.assertRaises(DataSourceUnavailableError):
            fetcher._request_json('quote', {'symbol': 'AAPL'})

        self.assertEqual(mock_get.call_count, 2)
        self.assertTrue(fetcher.is_temporarily_unavailable())

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_keys_rotate_between_requests(self, mock_get):
        fetcher = self._fetcher('key-aaaa1111', 'key-bbbb2222')
        mock_get.return_value = _make_mock_response({'c': 250.0})

        for _ in range(4):
            fetcher._request_json('quote', {'symbol': 'AAPL'})

        sent = [call.kwargs['params']['token'] for call in mock_get.call_args_list]
        self.assertEqual(
            sent,
            ['key-aaaa1111', 'key-bbbb2222', 'key-aaaa1111', 'key-bbbb2222'],
        )

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_invalid_key_is_disabled(self, mock_get):
        fetcher = self._fetcher('key-bad00000', 'key-good2222')
        mock_get.side_effect = [
            _make_mock_response({}, status_code=401),
            _make_mock_response({'c': 250.0}),
        ]

        data = fetcher._request_json('quote', {'symbol': 'AAPL'})

        self.assertEqual(data, {'c': 250.0})
        self.assertIn('key-bad00000', fetcher._dead_keys)

    def test_masked_key_never_leaks_the_value(self):
        fetcher = self._fetcher('abcdef0123456789')

        masked = fetcher._mask_key('abcdef0123456789')

        self.assertEqual(masked, '***6789')


class TestFinnhubFetcherRealtimeQuote(unittest.TestCase):
    """校验实时行情字段映射。"""

    def setUp(self):
        from data_provider.finnhub_fetcher import FinnhubFetcher
        self.fetcher = FinnhubFetcher()
        self.fetcher._api_key = "test_key"

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_realtime_quote_us_stock(self, mock_get):
        mock_get.return_value = _make_mock_response({
            'c': 150.0,
            'd': 2.0,
            'dp': 1.35,
            'h': 151.0,
            'l': 148.0,
            'o': 149.0,
            'pc': 148.0,
            't': 1718172800,
            'v': 5000000,
        })
        quote = self.fetcher.get_realtime_quote('AAPL')
        self.assertIsNotNone(quote)
        self.assertEqual(quote.code, 'AAPL')
        self.assertAlmostEqual(quote.price, 150.0)
        self.assertAlmostEqual(quote.change_pct, 1.35)
        self.assertAlmostEqual(quote.amplitude, 2.03)

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_realtime_quote_http_failure(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        self.assertIsNone(self.fetcher.get_realtime_quote('AAPL'))


class TestFinnhubFetcherStockName(unittest.TestCase):
    """校验代码搜索。"""

    def setUp(self):
        from data_provider.finnhub_fetcher import FinnhubFetcher
        self.fetcher = FinnhubFetcher()
        self.fetcher._api_key = "test_key"

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_get_stock_name_found(self, mock_get):
        mock_get.return_value = _make_mock_response({
            'result': [{'description': 'APPLE INC', 'symbol': 'AAPL'}],
            'count': 1,
        })
        self.assertEqual(self.fetcher.get_stock_name('AAPL'), 'APPLE INC')

    @patch('data_provider.finnhub_fetcher.requests.get')
    def test_get_stock_name_empty(self, mock_get):
        mock_get.return_value = _make_mock_response({'result': [], 'count': 0})
        self.assertIsNone(self.fetcher.get_stock_name('NOTEXIST'))


class TestFinnhubFetcherRegistration(unittest.TestCase):
    """校验 DataFetcherManager 的注册与美股链路接入。"""

    @patch('src.config.get_config')
    def test_registered_with_key(self, mock_config):
        mock_config.return_value = MagicMock(
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
        names = [f.name for f in mgr._fetchers]
        self.assertIn('FinnhubFetcher', names)

    @patch('src.config.get_config')
    def test_not_registered_without_key(self, mock_config):
        mock_config.return_value = MagicMock(
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
        names = [f.name for f in mgr._fetchers]
        self.assertNotIn('FinnhubFetcher', names)

    @patch('src.config.get_config')
    def test_us_chain_includes_finnhub_before_yfinance(self, mock_config):
        mock_config.return_value = MagicMock(
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
        self.assertLess(chain.index('FinnhubFetcher'), chain.index('YfinanceFetcher'))

    @patch('src.config.get_config')
    def test_hk_chain_does_not_route_to_finnhub(self, mock_config):
        mock_config.return_value = MagicMock(
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
        self.assertNotIn('FinnhubFetcher', chain)


if __name__ == '__main__':
    unittest.main()

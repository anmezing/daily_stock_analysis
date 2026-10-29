# -*- coding: utf-8 -*-
"""港股基本面：yfinance 比率字段的量级不能因数值大小而变。"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from data_provider.fundamental_adapter import HkYfinanceFundamentalAdapter


class HkFundamentalScalingTestCase(unittest.TestCase):
    def test_ratios_above_one_are_still_converted_to_percent(self) -> None:
        """150% 的增速是 1.5 的比率，不能再被当成已经是百分数。"""
        adapter = HkYfinanceFundamentalAdapter()
        info = {
            "revenueGrowth": 1.5,
            "earningsGrowth": 2.2,
            "returnOnEquity": 1.2,
            "grossMargins": 0.75,
            "heldPercentInstitutions": 0.62,
        }
        fake_yf = MagicMock()
        fake_yf.Ticker.return_value.info = info

        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            result = adapter.get_fundamental_bundle("00700")

        self.assertEqual(result["growth"]["revenue_yoy"], 150.0)
        self.assertEqual(result["growth"]["net_profit_yoy"], 220.0)
        self.assertEqual(result["growth"]["roe"], 120.0)
        self.assertEqual(result["growth"]["gross_margin"], 75.0)
        self.assertEqual(result["institution"]["institutional_holding_pct"], 62.0)

    def test_dividend_yield_is_not_rescaled(self) -> None:
        """dividendYield 的口径在各版本不一致，原样保留，避免再乘 100。"""
        adapter = HkYfinanceFundamentalAdapter()
        fake_yf = MagicMock()
        fake_yf.Ticker.return_value.info = {"dividendYield": 0.55, "dividendRate": 1.2}

        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            result = adapter.get_fundamental_bundle("00700")

        self.assertEqual(result["earnings"]["dividend_yield_raw"], 0.55)


if __name__ == "__main__":
    unittest.main()

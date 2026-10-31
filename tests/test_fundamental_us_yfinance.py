# -*- coding: utf-8 -*-
"""Tests for the US yfinance enrichment adapter and bundle merge."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.fundamental_adapter import (
    UsYfinanceFundamentalAdapter,
    merge_fundamental_bundles,
)


class UsYfinanceFundamentalAdapterTest(unittest.TestCase):
    def test_bundle_includes_valuation_and_margins(self) -> None:
        adapter = UsYfinanceFundamentalAdapter()
        info = {
            "trailingPE": 39.5,
            "forwardPE": 36.0,
            "priceToBook": 45.2,
            "priceToSalesTrailing12Months": 10.7,
            "pegRatio": 2.8,
            "enterpriseToEbitda": 30.0,
            "marketCap": 5.0e12,
            "revenueGrowth": 0.164,
            "earningsGrowth": 0.287,
            "returnOnEquity": 1.48,
            "grossMargins": 0.5,
            "operatingMargins": 0.32,
            "profitMargins": 0.27,
            "heldPercentInstitutions": 0.66,
            "targetMeanPrice": 332.3,
            "recommendationKey": "buy",
            "beta": 1.06,
            "fiftyTwoWeekHigh": 351.7,
            "fiftyTwoWeekLow": 243.4,
            "debtToEquity": 78.4,
            "currentRatio": 1.0,
            "freeCashflow": 1.07e11,
            "trailingEps": 8.72,
            "forwardEps": 9.55,
            "earningsTimestamp": 1793649600,
        }
        fake_yf = MagicMock()
        fake_yf.Ticker.return_value.info = info

        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            result = adapter.get_fundamental_bundle("AAPL")

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["valuation"]["pe_ratio"], 39.5)
        self.assertEqual(result["valuation"]["forward_pe"], 36.0)
        self.assertEqual(result["growth"]["revenue_yoy"], 16.4)
        self.assertEqual(result["growth"]["operating_margin"], 32.0)
        self.assertEqual(result["institution"]["target_mean_price"], 332.3)
        self.assertEqual(result["institution"]["debt_to_equity"], 78.4)
        self.assertEqual(result["earnings"]["eps_forward"], 9.55)
        self.assertIn("valuation:yfinance_info", result["source_chain"])

    def test_class_share_symbol_uses_dash(self) -> None:
        adapter = UsYfinanceFundamentalAdapter()
        self.assertEqual(adapter._to_yf_symbol("BRK.B"), "BRK-B")

    def test_bundle_includes_short_interest_and_insider_direction(self) -> None:
        adapter = UsYfinanceFundamentalAdapter()
        today = pd.Timestamp.now().normalize()
        info = {
            "sharesShort": 131_591_865,
            "sharesShortPriorMonth": 128_753_092,
            "shortRatio": 3.48,
            "shortPercentOfFloat": 0.009,
            "shortPercentOfSharesOutstanding": 0.008,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "floatShares": 14_569_078_010,
        }
        insider = pd.DataFrame(
            [
                {
                    "Start Date": (today - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                    "Transaction": "Purchase at price 100 per share.",
                    "Shares": 1000,
                    "Value": 100_000,
                },
                {
                    "Start Date": (today - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                    "Transaction": "Sale at price 150 per share.",
                    "Shares": 2000,
                    "Value": 300_000,
                },
                {
                    "Start Date": (today - pd.Timedelta(days=200)).strftime("%Y-%m-%d"),
                    "Transaction": "Sale at price 90 per share.",
                    "Shares": 9999,
                    "Value": 899_910,
                },
            ]
        )
        fake_yf = MagicMock()
        fake_yf.Ticker.return_value.info = info
        fake_yf.Ticker.return_value.insider_transactions = insider

        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            result = adapter.get_fundamental_bundle("AAPL")

        institution = result["institution"]
        self.assertEqual(institution["short_percent_float"], 0.9)
        self.assertEqual(institution["short_ratio_days"], 3.48)
        self.assertGreater(institution["short_change_pct"], 0)
        self.assertEqual(institution["insider_buy_count_90d"], 1)
        self.assertEqual(institution["insider_sell_count_90d"], 1)
        self.assertEqual(institution["insider_net_value_90d"], -200_000.0)
        self.assertEqual(institution["sector"], "Technology")


class MergeFundamentalBundlesTest(unittest.TestCase):
    def test_primary_growth_wins_and_secondary_valuation_is_kept(self) -> None:
        sec = {
            "status": "partial",
            "growth": {"revenue_yoy": 16.0},
            "earnings": {"latest_filing_form": "10-Q"},
            "institution": {"insider_form4_count_90d": 25},
            "source_chain": ["growth:sec_companyfacts"],
            "errors": [],
        }
        yf = {
            "status": "partial",
            "growth": {"revenue_yoy": 99.0, "roe": 110.0},
            "earnings": {"eps_forward": 9.5},
            "institution": {"target_mean_price": 332.0},
            "valuation": {"pe_ratio": 39.5},
            "source_chain": ["valuation:yfinance_info"],
            "errors": [],
        }

        merged = merge_fundamental_bundles(sec, yf)

        self.assertEqual(merged["growth"]["revenue_yoy"], 16.0)
        self.assertEqual(merged["growth"]["roe"], 110.0)
        self.assertEqual(merged["valuation"]["pe_ratio"], 39.5)
        self.assertEqual(merged["institution"]["insider_form4_count_90d"], 25)
        self.assertEqual(merged["institution"]["target_mean_price"], 332.0)
        self.assertEqual(merged["earnings"]["latest_filing_form"], "10-Q")
        self.assertIn("valuation:yfinance_info", merged["source_chain"])

    def test_missing_primary_returns_secondary(self) -> None:
        yf = {"status": "partial", "valuation": {"pe_ratio": 39.5}}
        self.assertEqual(merge_fundamental_bundles({}, yf), yf)
        self.assertEqual(merge_fundamental_bundles(None, yf), yf)


if __name__ == "__main__":
    unittest.main()

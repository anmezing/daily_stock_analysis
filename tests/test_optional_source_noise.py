# -*- coding: utf-8 -*-
""""没配置的可选数据源"不该在启动日志里刷 WARNING。

服务器只跑美股/港股，Tushare（仅 A 股）、IBKR（需本地 TWS + 可选依赖）
本来就不会参与链路，每次启动都提示"未配置/未安装"只会让人以为配置出错。
"""

import logging
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.config import Config


class TushareValidationHintTest(unittest.TestCase):
    @patch("src.config.setup_env")
    @patch.object(Config, "_parse_litellm_yaml", return_value=[])
    def _load(self, env: dict, _mock_parse_yaml, _mock_setup_env) -> Config:
        with patch.dict(os.environ, env, clear=True):
            return Config._load_from_env()

    def _tushare_messages(self, config: Config):
        return [message for message in config.validate() if "Tushare" in message]

    def test_us_only_pool_does_not_mention_tushare(self) -> None:
        config = self._load({"STOCK_LIST": "META,TSM", "MARKET_REVIEW_REGION": "us"})

        self.assertEqual(self._tushare_messages(config), [])

    def test_hk_us_pool_does_not_mention_tushare(self) -> None:
        config = self._load({"STOCK_LIST": "hk00700,AAPL", "MARKET_REVIEW_REGION": "hk+us"})

        self.assertEqual(self._tushare_messages(config), [])

    def test_a_share_pool_still_gets_the_hint(self) -> None:
        config = self._load({"STOCK_LIST": "600519", "MARKET_REVIEW_REGION": "cn"})

        self.assertEqual(len(self._tushare_messages(config)), 1)

    def test_cn_market_review_region_still_gets_the_hint(self) -> None:
        config = self._load({"STOCK_LIST": "AAPL", "MARKET_REVIEW_REGION": "cn+us"})

        self.assertEqual(len(self._tushare_messages(config)), 1)


class AgentSkillModeHintTest(unittest.TestCase):
    """AGENT_SKILLS 非空会自动启用 Agent 链，必须提示。"""

    @patch("src.config.setup_env")
    @patch.object(Config, "_parse_litellm_yaml", return_value=[])
    def _load(self, env: dict, _mock_parse_yaml, _mock_setup_env) -> Config:
        with patch.dict(os.environ, env, clear=True):
            return Config._load_from_env()

    def _agent_messages(self, config: Config):
        return [message for message in config.validate() if "AGENT_SKILLS" in message]

    def test_skills_without_agent_mode_is_reported(self) -> None:
        config = self._load({
            "STOCK_LIST": "AAPL",
            "AGENT_MODE": "false",
            "AGENT_SKILLS": "bull_trend,ma_golden_cross",
        })

        self.assertEqual(len(self._agent_messages(config)), 1)

    def test_empty_skills_does_not_warn(self) -> None:
        config = self._load({"STOCK_LIST": "AAPL", "AGENT_MODE": "false", "AGENT_SKILLS": ""})

        self.assertEqual(self._agent_messages(config), [])

    def test_agent_mode_does_not_warn(self) -> None:
        config = self._load({
            "STOCK_LIST": "AAPL",
            "AGENT_MODE": "true",
            "AGENT_SKILLS": "bull_trend",
        })

        self.assertEqual(self._agent_messages(config), [])


class OptionalFetcherInitLoggingTest(unittest.TestCase):
    def test_missing_tushare_token_logs_at_debug(self) -> None:
        from data_provider.tushare_fetcher import TushareFetcher

        with patch(
            "data_provider.tushare_fetcher.get_config",
            return_value=SimpleNamespace(tushare_token=None),
        ):
            with self.assertLogs("data_provider.tushare_fetcher", level="DEBUG") as captured:
                TushareFetcher()

        noisy = [
            record for record in captured.records
            if record.levelno >= logging.WARNING and "Tushare" in record.getMessage()
        ]
        self.assertEqual(noisy, [])

    def test_missing_ib_insync_logs_at_debug(self) -> None:
        import importlib.util

        if importlib.util.find_spec("ib_insync") is not None:
            self.skipTest("ib_insync installed; warning path not reachable")

        from data_provider.ibkr_fetcher import IBKRFetcher

        with self.assertLogs("data_provider.ibkr_fetcher", level="DEBUG") as captured:
            IBKRFetcher()

        noisy = [
            record for record in captured.records
            if record.levelno >= logging.WARNING and "ib_insync" in record.getMessage()
        ]
        self.assertEqual(noisy, [])


if __name__ == "__main__":
    unittest.main()

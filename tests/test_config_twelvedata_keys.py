# -*- coding: utf-8 -*-
"""Twelve Data 多 Key 配置解析测试。

免费档是 8 credits/分钟/账号，多账号可以并行用：
``TWELVEDATA_API_KEYS=key1,key2`` 优先，旧的单数 ``TWELVEDATA_API_KEY`` 仍兼容。
"""

import os
import unittest
from unittest.mock import patch

from src.config import Config


class TwelveDataKeysTest(unittest.TestCase):
    @patch("src.config.setup_env")
    @patch.object(Config, "_parse_litellm_yaml", return_value=[])
    def _load(self, env: dict, _mock_parse_yaml, _mock_setup_env) -> Config:
        with patch.dict(os.environ, env, clear=True):
            return Config._load_from_env()

    def test_plural_keys_are_parsed_and_deduplicated(self) -> None:
        config = self._load({"TWELVEDATA_API_KEYS": "key-a, key-b ,key-a,,key-c"})

        self.assertEqual(config.twelvedata_api_keys, ["key-a", "key-b", "key-c"])
        self.assertEqual(config.twelvedata_api_key, "key-a")

    def test_plural_takes_precedence_over_singular(self) -> None:
        config = self._load({
            "TWELVEDATA_API_KEYS": "key-a,key-b",
            "TWELVEDATA_API_KEY": "legacy-key",
        })

        self.assertEqual(config.twelvedata_api_keys, ["key-a", "key-b"])

    def test_singular_falls_back_to_one_key(self) -> None:
        config = self._load({"TWELVEDATA_API_KEY": "legacy-key"})

        self.assertEqual(config.twelvedata_api_keys, ["legacy-key"])
        self.assertEqual(config.twelvedata_api_key, "legacy-key")

    def test_no_key_configured(self) -> None:
        config = self._load({})

        self.assertEqual(config.twelvedata_api_keys, [])
        self.assertIsNone(config.twelvedata_api_key)


class FinnhubKeysTest(unittest.TestCase):
    """Finnhub 同样支持多 Key（免费档 60 calls/min/账号）。"""

    @patch("src.config.setup_env")
    @patch.object(Config, "_parse_litellm_yaml", return_value=[])
    def _load(self, env: dict, _mock_parse_yaml, _mock_setup_env) -> Config:
        with patch.dict(os.environ, env, clear=True):
            return Config._load_from_env()

    def test_plural_keys_are_parsed_and_deduplicated(self) -> None:
        config = self._load({"FINNHUB_API_KEYS": "k1, k2 ,k1"})

        self.assertEqual(config.finnhub_api_keys, ["k1", "k2"])
        self.assertEqual(config.finnhub_api_key, "k1")

    def test_singular_falls_back_to_one_key(self) -> None:
        config = self._load({"FINNHUB_API_KEY": "legacy"})

        self.assertEqual(config.finnhub_api_keys, ["legacy"])
        self.assertEqual(config.finnhub_api_key, "legacy")

    def test_plural_takes_precedence(self) -> None:
        config = self._load({"FINNHUB_API_KEYS": "k1,k2", "FINNHUB_API_KEY": "legacy"})

        self.assertEqual(config.finnhub_api_keys, ["k1", "k2"])


if __name__ == "__main__":
    unittest.main()

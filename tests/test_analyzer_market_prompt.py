# -*- coding: utf-8 -*-
"""Tests for market-aware analyzer prompts."""

import sys
import unittest
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from src.analyzer import AnalysisResult, GeminiAnalyzer


class TestAnalyzerMarketPrompt(unittest.TestCase):
    def _make_analyzer(self) -> GeminiAnalyzer:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer()
        analyzer._litellm_available = False
        return analyzer

    def test_system_prompt_is_assembled_per_market(self) -> None:
        """系统提示按市场拼装：公共约束一致，市场规则互不混入。"""
        analyzer = self._make_analyzer()

        prompts = {
            market: analyzer.build_system_prompt(market)
            for market in ("cn", "hk", "us")
        }

        for market, prompt in prompts.items():
            self.assertIn("China exposure", prompt, market)
            self.assertIn("震荡偏多/震荡/震荡偏空", prompt, market)
            self.assertIn("null", prompt, market)
            self.assertIn("只用给定数据", prompt, market)

        # 市场专属规则
        self.assertIn("涨跌停、T+1、做空受限", prompts["cn"])
        self.assertIn("不宜重仓追价", prompts["hk"])
        self.assertIn("不能仅因离均线较远就把强趋势股直接判成看空", prompts["us"])
        self.assertIn("不强制 MA5 / MA10 模板", prompts["us"])

        # 不把其它市场的规则串进来
        self.assertNotIn("HKEX", prompts["us"])
        self.assertNotIn("T+1、做空受限", prompts["us"])
        self.assertNotIn("earnings gap hold", prompts["cn"])

    def test_unknown_market_falls_back_to_us_rules(self) -> None:
        analyzer = self._make_analyzer()
        self.assertIn(
            "不能仅因离均线较远就把强趋势股直接判成看空",
            analyzer.build_system_prompt("jp"),
        )

    def test_us_prompt_includes_china_exposure_guidance(self) -> None:
        analyzer = self._make_analyzer()
        context = {
            "code": "AAPL",
            "stock_name": "Apple",
            "date": "2026-03-23",
            "today": {"close": 180.0, "ma5": 178.0, "ma10": 175.0, "ma20": 170.0},
            "trend_analysis": {
                "trend_status": "强势多头",
                "ma_alignment": "多头排列",
                "trend_strength": 88,
                "bias_ma5": 1.1,
                "bias_ma10": 2.0,
                "volume_status": "正常",
                "volume_trend": "量能正常",
                "buy_signal": "买入",
                "signal_score": 76,
            },
            "market_context": {
                "market": "us",
                "market_label": "美股",
                "official_source_priority": "SEC 披露、财报电话会、公司指引",
                "analysis_focus": "先看 SEC/财报/指引，再判断政策变量。",
                "policy_scope": "中国政策只有在存在 China exposure 时才提高权重。",
                "china_exposure": {
                    "level_label": "高",
                    "policy_weight_label": "高权重",
                    "reasoning": "同时存在中国收入和供应链暴露。",
                },
            },
        }

        prompt = analyzer._format_prompt(context, "Apple", news_context="news")

        self.assertIn("当前市场：美股", prompt)
        self.assertIn("China exposure：高", prompt)
        self.assertIn("中国政策权重：高权重", prompt)
        self.assertIn("SEC/财报/公司指引", prompt)
        self.assertIn("SEC/财报/公司指引是否出现超预期或下修", prompt)
        self.assertIn("breakout retest", prompt)
        self.assertIn("不要机械套用 A股 MA5/MA10 模式", prompt)
        self.assertIn("优先看趋势延续/突破确认，再结合财报、市场环境和估值", prompt)

    def test_cn_prompt_keeps_a_share_policy_focus(self) -> None:
        analyzer = self._make_analyzer()
        context = {
            "code": "600519",
            "stock_name": "贵州茅台",
            "date": "2026-03-23",
            "today": {"close": 1500.0, "ma5": 1490.0, "ma10": 1480.0, "ma20": 1460.0},
            "trend_analysis": {
                "trend_status": "多头排列",
                "ma_alignment": "MA5>MA10>MA20",
                "trend_strength": 82,
                "bias_ma5": 1.6,
                "bias_ma10": 2.4,
                "volume_status": "缩量回调",
                "volume_trend": "量能配合",
                "buy_signal": "买入",
                "signal_score": 74,
            },
            "market_context": {
                "market": "cn",
                "market_label": "A股",
                "official_source_priority": "巨潮/沪深交易所公告、业绩预告、监管函",
                "analysis_focus": "政策、监管、业绩预告与资金流本身就是 A 股核心变量。",
                "policy_scope": "中国政策与产业监管属于核心主因。",
            },
        }

        prompt = analyzer._format_prompt(context, "贵州茅台", news_context=None)

        self.assertIn("当前市场：A股", prompt)
        self.assertIn("中国政策与产业监管属于核心主因", prompt)
        self.assertIn("是否满足 MA5>MA10>MA20 多头排列", prompt)
        self.assertIn("涨停/交易约束/T+1", prompt)
        self.assertIn("MA5>MA10>MA20 为买入级硬条件", prompt)

    def test_prompt_keeps_markdown_tables_but_removes_prompt_emojis(self) -> None:
        analyzer = self._make_analyzer()
        context = {
            "code": "AAPL",
            "stock_name": "Apple",
            "date": "2026-03-23",
            "today": {"close": 180.0, "ma5": 178.0, "ma10": 175.0, "ma20": 170.0},
            "market_context": {
                "market": "us",
                "market_label": "美股",
                "official_source_priority": "SEC 披露、财报电话会、公司指引",
                "analysis_focus": "先看 SEC/财报/指引，再判断政策变量。",
                "policy_scope": "中国政策只有在存在 China exposure 时才提高权重。",
            },
        }

        prompt = analyzer._format_prompt(context, "Apple", news_context="news")

        self.assertIn("| 项目 | 数据 |", prompt)
        self.assertIn("## 股票基础信息", prompt)
        self.assertNotIn("## 📊 股票基础信息", prompt)
        self.assertNotIn("## 📰 舆情情报", prompt)
        self.assertNotIn("❓", prompt)
        self.assertNotIn("✅", prompt)
        self.assertNotIn("⚠️", prompt)
        self.assertNotIn("⚪", prompt)
        self.assertNotIn("❌", prompt)

    def test_prompt_formats_prices_with_two_decimals(self) -> None:
        analyzer = self._make_analyzer()
        context = {
            "code": "META",
            "stock_name": "Meta Platforms, Inc.",
            "date": "2026-04-14",
            "today": {
                "close": 649.339999,
                "open": 643.0123,
                "high": 652.45123,
                "low": 639.36999,
                "pct_chg": 1.756,
                "ma5": 630.9080024414063,
                "ma10": 602.8430026855469,
                "ma20": 593.4319963378906,
            },
            "history": [
                {
                    "date": "2026-01-14",
                    "open": 625.9640568168747,
                    "high": 627.9124008742662,
                    "low": 614.2940558608916,
                    "close": 614.9934692382812,
                    "pct_chg": -2.47,
                }
            ],
            "market_context": {
                "market": "us",
                "market_label": "美股",
                "official_source_priority": "SEC 披露、财报电话会、公司指引",
                "analysis_focus": "先看 SEC/财报/指引，再判断政策变量。",
                "policy_scope": "中国政策只有在存在 China exposure 时才提高权重。",
            },
        }

        prompt = analyzer._format_prompt(context, "Meta Platforms, Inc.", news_context=None)

        self.assertIn("| 收盘价 | 649.34 美元 |", prompt)
        self.assertIn("| 开盘价 | 643.01 美元 |", prompt)
        self.assertIn("| MA5 | 630.91 |", prompt)
        self.assertIn("| MA10 | 602.84 |", prompt)
        self.assertIn("| MA20 | 593.43 |", prompt)
        self.assertIn("| 2026-01-14 | 625.96 | 627.91 | 614.29 | 614.99 | -2.47% |", prompt)
        self.assertNotIn("630.9080024414063", prompt)
        self.assertNotIn("625.9640568168747", prompt)

    def test_prompt_distinguishes_recent_news_from_background_intel(self) -> None:
        analyzer = self._make_analyzer()
        context = {
            "code": "AVGO",
            "stock_name": "Broadcom Inc.",
            "date": "2026-04-16",
            "today": {"close": 396.72, "ma5": 385.10, "ma10": 359.40, "ma20": 335.46},
            "market_context": {
                "market": "us",
                "market_label": "美股",
                "official_source_priority": "SEC 披露、财报电话会、公司指引",
                "analysis_focus": "先看 SEC/财报/指引，再判断政策变量。",
                "policy_scope": "中国政策只有在存在 China exposure 时才提高权重。",
            },
        }
        news_context = """【Broadcom Inc. 情报搜索结果】
注：`行业分析` 维度可能包含百科、公司介绍或历史财务等背景资料，只能作背景参考，不能直接当作近7日新闻、最新催化或当前业绩展望。
"""

        prompt = analyzer._format_prompt(context, "Broadcom Inc.", news_context=news_context)

        self.assertIn("近7日检索到的情报", prompt)
        self.assertIn("请严格区分“近期事件”和“背景资料”", prompt)
        self.assertIn("不得写成“最新消息”", prompt)
        self.assertIn("旧财年数据只能作背景参考", prompt)

    def test_analyze_logs_the_actual_model_used_after_fallback(self) -> None:
        analyzer = self._make_analyzer()
        analyzer._litellm_available = True

        context = {
            "code": "META",
            "stock_name": "Meta Platforms, Inc.",
            "date": "2026-04-14",
            "today": {"close": 649.34},
        }
        parsed = AnalysisResult(
            code="META",
            name="Meta Platforms, Inc.",
            sentiment_score=80,
            trend_prediction="看多",
            operation_advice="买入",
        )

        with patch("src.analyzer.get_config") as mock_cfg, \
             patch.object(analyzer, "_format_prompt", return_value="prompt"), \
             patch.object(
                 analyzer,
                 "_call_litellm",
                 return_value=("response", "gemini/gemini-2.5-flash", {}),
             ), \
             patch.object(analyzer, "_parse_response", return_value=parsed), \
             patch.object(analyzer, "_build_market_snapshot", return_value={}), \
             patch("src.analyzer.persist_llm_usage"), \
             patch("src.analyzer.logger.info") as mock_logger_info, \
             patch("src.analyzer.logger.debug") as mock_logger_debug:
            cfg = MagicMock()
            cfg.gemini_request_delay = 0
            cfg.report_integrity_enabled = False
            cfg.report_integrity_retry = 0
            cfg.llm_temperature = 0.7
            cfg.litellm_model = "gemini/gemini-2.5-pro"
            mock_cfg.return_value = cfg

            result = analyzer.analyze(context, news_context="news")

        self.assertEqual(result.model_used, "gemini/gemini-2.5-flash")
        info_messages = [str(call.args[0]) for call in mock_logger_info.call_args_list if call.args]
        debug_messages = [str(call.args[0]) for call in mock_logger_debug.call_args_list if call.args]
        self.assertTrue(
            any("[LLM返回] gemini/gemini-2.5-flash 响应成功" in message for message in info_messages)
        )
        self.assertFalse(any("[LLM Prompt 预览]" in message for message in info_messages))
        self.assertFalse(any("[LLM返回 预览]" in message for message in info_messages))
        self.assertTrue(any("[LLM Prompt 预览]" in message for message in debug_messages))
        self.assertTrue(any("[LLM返回 预览]" in message for message in debug_messages))


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""免费文本列不受 VARCHAR 长度限制的回归测试。

线上问题：``analysis_history.name`` / ``news_intel.name`` 当初是 ``VARCHAR(50)``，
美股长公司名（"Taiwan Semiconductor Manufacturing Company Ltd. ADR"，51 字符）
超长，PostgreSQL 抛 ``StringDataRightTruncation``，整条分析记录被丢弃。

修法：凡值来自行情源 / 搜索源 / 模型的文本列一律用 ``TEXT``，从根上避免"多长"
这个问题；仍保留 VARCHAR 的列在写入前按【模型声明的真实列宽】截断。
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from sqlalchemy import Text

from src.analyzer import AnalysisResult
from src.config import Config
from src.storage import (
    AnalysisHistory,
    DatabaseManager,
    NewsIntel,
    _clip_to_column,
    _column_text_limit,
)


LONG_US_NAME = "Taiwan Semiconductor Manufacturing Company Ltd. ADR"


class ClipToColumnTest(unittest.TestCase):
    def test_text_columns_never_truncate(self) -> None:
        long_value = "长" * 5000

        self.assertEqual(_clip_to_column(AnalysisHistory, "name", long_value), long_value)
        self.assertEqual(_clip_to_column(NewsIntel, "title", long_value), long_value)
        self.assertEqual(_clip_to_column(NewsIntel, "query", long_value), long_value)

    def test_bounded_columns_use_the_declared_width(self) -> None:
        self.assertEqual(_column_text_limit(NewsIntel, "url"), 1000)

        clipped = _clip_to_column(NewsIntel, "url", "https://example.com/" + "a" * 2000)
        self.assertEqual(len(clipped), 1000)

    def test_missing_column_is_not_truncated(self) -> None:
        self.assertIsNone(_column_text_limit(NewsIntel, "not_a_column"))
        self.assertEqual(_clip_to_column(NewsIntel, "not_a_column", "x" * 10), "x" * 10)

    def test_none_stays_none(self) -> None:
        self.assertIsNone(_clip_to_column(AnalysisHistory, "name", None))


class TextColumnStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = os.path.join(self._temp_dir.name, "test_text_columns.db")
        self._old_env = {
            key: os.environ.get(key)
            for key in ("ENV_FILE", "DATABASE_URL", "DATABASE_PATH")
        }
        os.environ["ENV_FILE"] = os.path.join(self._temp_dir.name, "empty.env")
        os.environ.pop("DATABASE_URL", None)
        os.environ["DATABASE_PATH"] = self._db_path

        Config._instance = None
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config._instance = None
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._temp_dir.cleanup()

    def _result(self, name: str) -> AnalysisResult:
        return AnalysisResult(
            code="TSM",
            name=name,
            sentiment_score=59,
            trend_prediction="震荡偏多",
            operation_advice="持有",
            analysis_summary="测试",
        )

    def test_very_long_stock_name_is_saved_intact(self) -> None:
        long_name = LONG_US_NAME + " Ordinary Shares Class A Depositary Receipt"

        saved = self.db.save_analysis_history(
            result=self._result(long_name),
            query_id="q_long_name",
            report_type="simple",
            news_content=None,
            context_snapshot=None,
            save_snapshot=False,
        )

        self.assertEqual(saved, 1)
        with self.db.get_session() as session:
            row = session.query(AnalysisHistory).filter(
                AnalysisHistory.query_id == "q_long_name"
            ).first()
        self.assertEqual(row.name, long_name)

    def test_long_model_text_is_saved_intact(self) -> None:
        result = self._result("TSM")
        result.trend_prediction = "震" * 300
        result.operation_advice = "强烈买入（建议分三批建仓，首仓不超过总仓位三分之一）"

        saved = self.db.save_analysis_history(
            result=result,
            query_id="q_long_text",
            report_type="simple",
            news_content=None,
            context_snapshot=None,
            save_snapshot=False,
        )

        self.assertEqual(saved, 1)
        with self.db.get_session() as session:
            row = session.query(AnalysisHistory).filter(
                AnalysisHistory.query_id == "q_long_text"
            ).first()
        self.assertEqual(row.trend_prediction, "震" * 300)
        self.assertEqual(row.operation_advice, result.operation_advice)

    def test_news_intel_long_name_and_title_are_saved(self) -> None:
        long_title = "TSM files 6-K with SEC — " + "very long headline " * 40
        response = SimpleNamespace(
            provider="Tavily",
            results=[
                SimpleNamespace(
                    title=long_title,
                    url="https://example.com/tsm-6k",
                    source="sec.gov",
                    snippet="filing",
                    published_date="2026-10-29",
                )
            ],
        )

        saved = self.db.save_news_intel(
            code="TSM",
            name=LONG_US_NAME,
            dimension="official_filings",
            query="TSM 6-K",
            response=response,
        )

        self.assertEqual(saved, 1)
        with self.db.get_session() as session:
            row = session.query(NewsIntel).filter(NewsIntel.url == "https://example.com/tsm-6k").first()
        self.assertEqual(row.name, LONG_US_NAME)
        self.assertEqual(row.title, long_title.strip())

    def test_overlong_url_is_clipped_to_column_width(self) -> None:
        response = SimpleNamespace(
            provider="Tavily",
            results=[
                SimpleNamespace(
                    title="TSM",
                    url="https://example.com/" + "a" * 1500,
                    source="sec.gov",
                    snippet="filing",
                    published_date="2026-10-29",
                )
            ],
        )

        saved = self.db.save_news_intel(
            code="TSM",
            name="TSM",
            dimension="latest_news",
            query="TSM",
            response=response,
        )

        self.assertEqual(saved, 1)
        with self.db.get_session() as session:
            row = session.query(NewsIntel).first()
        self.assertEqual(len(row.url), 1000)


class RelaxColumnToTextMigrationTest(unittest.TestCase):
    def _manager(self, dialect: str) -> DatabaseManager:
        manager = object.__new__(DatabaseManager)
        manager._engine = SimpleNamespace(dialect=SimpleNamespace(name=dialect))
        return manager

    def test_varchar_column_is_relaxed_to_text(self) -> None:
        manager = self._manager("postgresql")
        conn = MagicMock()
        inspector = MagicMock()
        inspector.get_columns.return_value = [
            {"name": "name", "type": SimpleNamespace(length=50), "nullable": True}
        ]

        manager._relax_column_to_text_if_needed(conn, inspector, "analysis_history", "name")

        conn.exec_driver_sql.assert_called_once_with(
            "ALTER TABLE analysis_history ALTER COLUMN name TYPE TEXT"
        )

    def test_text_column_is_left_alone(self) -> None:
        manager = self._manager("postgresql")
        conn = MagicMock()
        inspector = MagicMock()
        inspector.get_columns.return_value = [
            {"name": "name", "type": Text(), "nullable": True}
        ]

        manager._relax_column_to_text_if_needed(conn, inspector, "analysis_history", "name")

        conn.exec_driver_sql.assert_not_called()


class SearchComprehensiveIntelToolTest(unittest.TestCase):
    def test_tool_calls_format_intel_report_with_supported_arguments(self) -> None:
        from unittest.mock import patch

        from src.agent.tools import search_tools

        service = MagicMock()
        service.search_comprehensive_intel.return_value = {"latest_news": MagicMock(success=True, results=[])}
        service.format_intel_report.return_value = "TSM 情报报告"
        service.build_market_intel_summary.return_value = {"market": "us"}

        with patch.object(search_tools, "_get_search_service", return_value=service):
            payload = search_tools._handle_search_comprehensive_intel("TSM", LONG_US_NAME)

        self.assertNotIn("error", payload)
        self.assertEqual(payload["report"], "TSM 情报报告")
        service.format_intel_report.assert_called_once_with(
            service.search_comprehensive_intel.return_value,
            LONG_US_NAME,
        )


if __name__ == "__main__":
    unittest.main()

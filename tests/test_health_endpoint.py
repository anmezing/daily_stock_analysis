# -*- coding: utf-8 -*-
"""健康检查与数据库单例重试的回归测试。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from api.app import create_app
from src.config import Config
from src.storage import DatabaseManager, check_database_connection


class HealthEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.env_path = self.data_dir / ".env"
        self.db_path = self.data_dir / "health_test.db"
        self.env_path.write_text(
            "\n".join(
                [
                    "STOCK_LIST=600519",
                    "GEMINI_API_KEY=test",
                    "ADMIN_AUTH_ENABLED=false",
                    f"DATABASE_PATH={self.db_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        os.environ["ENV_FILE"] = str(self.env_path)
        os.environ["DATABASE_PATH"] = str(self.db_path)
        Config.reset_instance()
        DatabaseManager.reset_instance()
        self.client = TestClient(create_app(static_dir=self.data_dir / "empty-static"))

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        os.environ.pop("ENV_FILE", None)
        os.environ.pop("DATABASE_PATH", None)
        self.temp_dir.cleanup()

    def test_health_reports_database_ok(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["database"], "ok")

    def test_health_returns_503_when_database_is_down(self) -> None:
        with patch("api.app.check_database_connection", return_value=(False, "OperationalError: down")):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertIn("down", payload["database"])

    def test_v1_health_returns_503_when_database_is_down(self) -> None:
        with patch(
            "api.v1.endpoints.health.check_database_connection",
            return_value=(False, "OperationalError: down"),
        ):
            response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 503)

    def test_check_database_connection_uses_real_session(self) -> None:
        ok, error = check_database_connection()

        self.assertTrue(ok, error)
        self.assertEqual(error, "")


class DatabaseManagerRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.env_path = self.data_dir / ".env"
        self.db_path = self.data_dir / "retry_test.db"
        self.env_path.write_text(f"DATABASE_PATH={self.db_path}\n", encoding="utf-8")
        os.environ["ENV_FILE"] = str(self.env_path)
        os.environ["DATABASE_PATH"] = str(self.db_path)
        Config.reset_instance()
        DatabaseManager.reset_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        os.environ.pop("ENV_FILE", None)
        os.environ.pop("DATABASE_PATH", None)
        self.temp_dir.cleanup()

    def test_get_instance_retries_after_initialization_failure(self) -> None:
        with patch("src.storage.Base.metadata.create_all", side_effect=RuntimeError("database is starting")):
            with self.assertRaises(RuntimeError):
                DatabaseManager.get_instance()

        # 数据库恢复后，下一次取用应当重新初始化，而不是永久返回坏实例
        db = DatabaseManager.get_instance()

        self.assertTrue(getattr(db, "_initialized", False))
        with db.get_session() as session:
            self.assertIsNotNone(session)


if __name__ == "__main__":
    unittest.main()

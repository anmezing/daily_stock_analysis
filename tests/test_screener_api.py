# -*- coding: utf-8 -*-
"""Tests for the read-only screener API helpers."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.v1.endpoints import screener


class ScreenerApiTest(unittest.TestCase):
    def test_latest_returns_saved_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            data_dir.mkdir()
            payload = {
                "screen_date": "2026-10-31",
                "generated_at": "2026-10-31T08:31:00",
                "result": {"status": "ok", "picks": [{"symbol": "AAA"}]},
                "markdown": "## test",
            }
            (data_dir / "screener_latest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with patch.object(screener, "_DATA_DIR", data_dir):
                result = screener.get_latest_screener()
            self.assertEqual(result["screen_date"], "2026-10-31")
            self.assertEqual(result["result"]["picks"][0]["symbol"], "AAA")

    def test_latest_missing_returns_404(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(screener, "_DATA_DIR", Path(temp_dir)):
                with self.assertRaises(HTTPException) as context:
                    screener.get_latest_screener()
            self.assertEqual(context.exception.status_code, 404)

    def test_history_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports = Path(temp_dir)
            (reports / "screener_20261030.md").write_text("old", encoding="utf-8")
            (reports / "screener_20261031.md").write_text("new", encoding="utf-8")
            with patch.object(screener, "_REPORTS_DIR", reports):
                history = screener.get_screener_history()
                report = screener.get_screener_report("2026-10-31")
            self.assertEqual(history["items"][0]["date"], "2026-10-31")
            self.assertEqual(report["markdown"], "new")

    def test_pool_and_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "screener_pool.json").write_text(
                json.dumps({"items": {"AAA": {"stage": "ready"}}, "summary": {}}),
                encoding="utf-8",
            )
            (data_dir / "screener_alerts.json").write_text(
                json.dumps({"alerts": [{"symbol": "AAA", "kind": "stop_breach"}]}),
                encoding="utf-8",
            )
            with patch.object(screener, "_DATA_DIR", data_dir):
                pool = screener.get_screener_pool()
                alerts = screener.get_screener_alerts()
            self.assertEqual(pool["items"]["AAA"]["stage"], "ready")
            self.assertEqual(alerts["alerts"][0]["kind"], "stop_breach")


if __name__ == "__main__":
    unittest.main()

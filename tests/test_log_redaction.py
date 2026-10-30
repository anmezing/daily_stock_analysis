# -*- coding: utf-8 -*-
"""日志脱敏测试：任何带密钥的日志都必须被替换成 ***。"""

import io
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logging_config import RelativePathFormatter, setup_logging
from src.utils.log_redact import mask_database_url, redact_secrets

TD_KEY = "e6f0974f1d24e5b7a374e0359c3638"
DSN_PASSWORD = "60ff0f3956ddc8417eec0ba4af297db7"


class RedactSecretsTest(unittest.TestCase):
    def test_database_dsn_password_is_masked(self) -> None:
        text = f"postgresql+psycopg2://dsa:{DSN_PASSWORD}@db:5432/dsa"

        masked = redact_secrets(text)

        self.assertNotIn(DSN_PASSWORD, masked)
        self.assertEqual(masked, "postgresql+psycopg2://dsa:***@db:5432/dsa")

    def test_mask_database_url_keeps_user_and_host(self) -> None:
        masked = mask_database_url(f"postgresql://dsa:{DSN_PASSWORD}@db:5432/dsa")

        self.assertIn("dsa", masked)
        self.assertIn("@db:5432/dsa", masked)
        self.assertNotIn(DSN_PASSWORD, masked)

    def test_query_params_are_masked(self) -> None:
        for param in ("apikey", "api_key", "token", "access_token", "secret", "password", "key"):
            text = f"https://api.example.com/q?symbol=AAPL&{param}={TD_KEY}"

            self.assertNotIn(TD_KEY, redact_secrets(text), param)

    def test_telegram_bot_url_is_masked(self) -> None:
        text = "https://api.telegram.org/bot8123456789:AAH-abcdefghijklmnop1234/sendMessage"

        masked = redact_secrets(text)

        self.assertNotIn("AAH-abcdefghijklmnop1234", masked)
        self.assertIn("/bot***/sendMessage", masked)

    def test_authorization_header_is_masked(self) -> None:
        masked = redact_secrets("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")

        self.assertNotIn("eyJhbGciOiJIUzI1NiJ9", masked)

    def test_known_key_prefixes_are_masked(self) -> None:
        samples = [
            "AIzaSyD-1234567890abcdefghijklmnopqrstuv",
            "sk-abcdefghijklmnopqrstuvwx",
            "xoxb-123456789012-abcdefghij",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AKIAIOSFODNN7EXAMPLE",
        ]
        for sample in samples:
            masked = redact_secrets(f"key={sample} in log")

            self.assertNotIn(sample, masked, sample)

    def test_plain_text_is_untouched(self) -> None:
        text = "美股 AAPL 使用链路: TwelveDataFetcher(P2) -> YfinanceFetcher(P4), rows=303"

        self.assertEqual(redact_secrets(text), text)


class FormatterRedactionTest(unittest.TestCase):
    def _format(self, message: str) -> str:
        formatter = RelativePathFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord(
            name="x", level=logging.ERROR, pathname=__file__, lineno=1,
            msg=message, args=(), exc_info=None,
        )
        return formatter.format(record)

    def test_formatter_masks_dsn_even_with_percent_formatting(self) -> None:
        output = self._format(f"数据库初始化完成: postgresql://dsa:{DSN_PASSWORD}@db:5432/dsa")

        self.assertNotIn(DSN_PASSWORD, output)

    def test_formatter_masks_query_secrets(self) -> None:
        output = self._format(f"GET https://api.twelvedata.com/time_series?symbol=MU&apikey={TD_KEY}")

        self.assertNotIn(TD_KEY, output)

    def test_formatter_masks_secrets_inside_traceback(self) -> None:
        formatter = RelativePathFormatter("%(levelname)s %(message)s")
        try:
            raise RuntimeError(f"call failed: https://finnhub.io/api/v1/quote?token={TD_KEY}")
        except RuntimeError:
            record = logging.LogRecord(
                name="x", level=logging.ERROR, pathname=__file__, lineno=1,
                msg="request failed", args=(), exc_info=sys.exc_info(),
            )
            output = formatter.format(record)

        self.assertNotIn(TD_KEY, output)
        self.assertIn("***", output)


class SetupLoggingRedactionTest(unittest.TestCase):
    def test_log_file_does_not_contain_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                setup_logging(log_prefix="redact", log_dir=os.path.join(temp_dir, "logs"))
                logging.getLogger("test.secret").error(
                    "db=postgresql://dsa:%s@db:5432/dsa key=%s", DSN_PASSWORD, TD_KEY
                )
                for handler in logging.getLogger().handlers:
                    handler.flush()

                content = ""
                for name in os.listdir(os.path.join(temp_dir, "logs")):
                    if name.startswith("redact_") and name.endswith(".log") and "debug" not in name:
                        with io.open(os.path.join(temp_dir, "logs", name), encoding="utf-8") as handle:
                            content += handle.read()
            finally:
                for handler in list(logging.getLogger().handlers):
                    logging.getLogger().removeHandler(handler)
                    try:
                        handler.close()
                    except Exception:
                        pass
                os.chdir(cwd)

        self.assertIn("***", content)
        self.assertNotIn(DSN_PASSWORD, content)
        self.assertNotIn(TD_KEY, content)


if __name__ == "__main__":
    unittest.main()

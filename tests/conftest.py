# -*- coding: utf-8 -*-
"""Shared lightweight test stubs for optional dependencies.

Also isolates tests from any developer-configured external database:
``.env`` may point ``DATABASE_URL`` at a real PostgreSQL/Supabase instance,
and ``Config.get_db_url()`` prefers it over ``DATABASE_PATH``. Without this
guard, tests that only override ``DATABASE_PATH`` would still write to that
remote database. Set ``DSA_ALLOW_EXTERNAL_DB_IN_TESTS=1`` to opt out when a
test really must use the configured external database.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


def _isolate_test_environment() -> None:
    """Point config at an empty env file and a throwaway SQLite database."""
    if os.environ.get("DSA_ALLOW_EXTERNAL_DB_IN_TESTS") == "1":
        return

    workdir = Path(tempfile.mkdtemp(prefix="dsa-test-env-"))
    empty_env = workdir / "empty.env"
    empty_env.write_text("", encoding="utf-8")

    # Loading the developer's .env would re-populate DATABASE_URL, so switch
    # ENV_FILE instead of only clearing the variable. Some test modules also
    # call load_dotenv(".env") at import time with override=False; an empty
    # (but present) DATABASE_URL keeps those calls from restoring the value.
    os.environ["ENV_FILE"] = str(empty_env)
    os.environ["DATABASE_URL"] = ""
    os.environ.setdefault("DATABASE_PATH", str(workdir / "dsa_test.db"))
    # 测试不应真的访问新闻源；需要测 RSS 的用例自行显式开启
    os.environ.setdefault("NEWS_RSS_ENABLED", "false")


_isolate_test_environment()


if "fake_useragent" not in sys.modules:
    ua_instance = MagicMock()
    ua_instance.random = "Mozilla/5.0"
    ua_instance.chrome = "Mozilla/5.0"

    fake_useragent_module = MagicMock()
    fake_useragent_module.UserAgent = MagicMock(return_value=ua_instance)
    sys.modules["fake_useragent"] = fake_useragent_module

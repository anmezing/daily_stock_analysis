# -*- coding: utf-8 -*-
"""数据库连接建立时的只读会话纠正（Supabase pooler 等场景）。"""

import sys
from unittest.mock import MagicMock

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()

from src.storage import _install_postgres_read_write_handler  # noqa: E402


class _FakeCursor:
    def __init__(self, show_value):
        self._show_value = show_value
        self.executed = []
        self.closed = False

    def execute(self, sql):
        self.executed.append(sql)

    def fetchone(self):
        return (self._show_value,)

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self, show_value):
        self._cursor = _FakeCursor(show_value)
        self.autocommit = False

    def cursor(self):
        return self._cursor


def _install_and_capture(monkeypatch, dialeck_name="postgresql"):
    captured = {}

    def fake_listens_for(target, identifier):
        def decorator(fn):
            captured[identifier] = fn
            return fn

        return decorator

    monkeypatch.setattr("src.storage.event.listens_for", fake_listens_for)

    engine = MagicMock()
    engine.dialect.name = dialeck_name
    _install_postgres_read_write_handler(engine)
    return captured


def test_enables_writes_when_session_defaults_to_read_only(monkeypatch):
    captured = _install_and_capture(monkeypatch)
    conn = _FakeConnection("on")

    captured["connect"](conn, None)

    assert any("SET default_transaction_read_only = off" in sql for sql in conn._cursor.executed)
    assert conn.autocommit is False, "autocommit 必须恢复原值"
    assert conn._cursor.closed


def test_leaves_writable_sessions_untouched(monkeypatch):
    captured = _install_and_capture(monkeypatch)
    conn = _FakeConnection("off")

    captured["connect"](conn, None)

    assert not any("SET default_transaction_read_only" in sql for sql in conn._cursor.executed)


def test_ignores_non_postgres_engines(monkeypatch):
    called = {"n": 0}

    def fake_listens_for(target, identifier):
        called["n"] += 1
        return lambda fn: fn

    monkeypatch.setattr("src.storage.event.listens_for", fake_listens_for)

    engine = MagicMock()
    engine.dialect.name = "sqlite"
    _install_postgres_read_write_handler(engine)

    assert called["n"] == 0

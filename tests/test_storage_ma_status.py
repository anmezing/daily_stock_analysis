# -*- coding: utf-8 -*-
"""均线形态：缺失数据不能伪装成“震荡整理”。"""

from types import SimpleNamespace

from src.storage import DatabaseManager


def _bar(**kwargs):
    base = {"close": None, "ma5": None, "ma10": None, "ma20": None}
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_missing_moving_average_is_reported_as_unknown():
    status = DatabaseManager._analyze_ma_status(None, _bar(close=100.0))

    assert "数据不足" in status
    assert "震荡" not in status


def test_bull_alignment_is_detected():
    status = DatabaseManager._analyze_ma_status(
        None, _bar(close=110.0, ma5=105.0, ma10=100.0, ma20=95.0)
    )

    assert "多头" in status


def test_bear_alignment_is_detected():
    status = DatabaseManager._analyze_ma_status(
        None, _bar(close=90.0, ma5=95.0, ma10=100.0, ma20=105.0)
    )

    assert "空头" in status

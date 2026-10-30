# -*- coding: utf-8 -*-
"""LLM 请求体的 JSON 清洗。

背景：
    httpx 用 ``json.dumps(..., allow_nan=False)`` 序列化请求体，而
    ``float('nan')`` / ``float('inf')`` 是"合法"的 Python 浮点数。指标计算
    （均线、乖离率、涨跌幅）只要有一处把 NaN 带进 messages / tools /
    temperature，调用就会在**发出请求之前**失败，报：

        ValueError: Out of range float values are not JSON compliant

    这类错误被 LiteLLM 包成 ``APIConnectionError``，看起来像网络问题，
    实际是本地数据脏了，很难排查。

这里在调用 LiteLLM 之前做一次递归清洗：非有限浮点数替换为 ``None``，
numpy 浮点数统一转成内置 ``float``，并回传被替换的字段路径用于告警。
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple

try:  # numpy 是分析链的硬依赖，但清洗模块要能独立导入
    import numpy as _np
except Exception:  # pragma: no cover - numpy 缺失时退化为纯 float 判断
    _np = None


def _is_numpy_float(value: Any) -> bool:
    return _np is not None and isinstance(value, _np.floating)


def _to_builtin(value: Any) -> Any:
    """numpy 标量在 json 里不可序列化，统一转成内置类型。"""
    if _np is None:
        return value
    if isinstance(value, _np.integer):
        return int(value)
    if isinstance(value, _np.bool_):
        return bool(value)
    return value


def _is_non_finite_float(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if _is_numpy_float(value):
        return not math.isfinite(float(value))
    return False


def _sanitize(value: Any, path: str, replaced: List[str]) -> Any:
    if _is_non_finite_float(value):
        replaced.append(path or "<root>")
        return None
    if _is_numpy_float(value):
        return float(value)
    value = _to_builtin(value)
    if isinstance(value, dict):
        return {
            key: _sanitize(item, f"{path}.{key}" if path else str(key), replaced)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item, f"{path}[{index}]", replaced) for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_sanitize(item, f"{path}[{index}]", replaced) for index, item in enumerate(value))
    return value


def sanitize_non_finite(payload: Any) -> Tuple[Any, List[str]]:
    """递归替换 ``nan`` / ``inf``，返回 ``(清洗后的对象, 被替换的路径列表)``。

    只处理容器与数值，字符串内容原样保留（文本里出现 "nan" 不影响 JSON）。
    """
    replaced: List[str] = []
    cleaned = _sanitize(payload, "", replaced)
    return cleaned, replaced

# -*- coding: utf-8 -*-
"""买卖建议的统一语义表。

`analyzer` 负责把 LLM 输出规范化成信号，`backtest_engine` 负责把同一条建议
翻译成方向/仓位做评估。过去两边各写一套关键词，交集外的词（清仓、止盈、
止损、增持、建仓、add、reduce）会出现"上游识别成卖出、下游算成没方向"这类错位。

这里集中定义一次，两边都从这里取，避免再次漂移。

方向语义：
- ``up``       预期上涨
- ``down``     预期下跌（明确的看空）
- ``not_down`` 允许波动，但不该显著下跌（持有）
- ``not_up``   允许波动，但不该显著上涨（减仓/止盈/止损/观望——仓位管理而非看空）
- ``flat``     无法归类时的兜底
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class AdviceSemantics:
    signal: str  # buy / hold / sell
    direction: str  # up / down / not_down / not_up / flat
    position: str  # long / cash


# 注意：键统一用小写；中文不受影响，英文别名覆盖 buy/add/hold/wait/sell/reduce。
ADVICE_TABLE: Dict[str, AdviceSemantics] = {
    # 看多
    "强烈买入": AdviceSemantics("buy", "up", "long"),
    "买入": AdviceSemantics("buy", "up", "long"),
    "加仓": AdviceSemantics("buy", "up", "long"),
    "增持": AdviceSemantics("buy", "up", "long"),
    "建仓": AdviceSemantics("buy", "up", "long"),
    "做多": AdviceSemantics("buy", "up", "long"),
    "strong buy": AdviceSemantics("buy", "up", "long"),
    "buy": AdviceSemantics("buy", "up", "long"),
    "add": AdviceSemantics("buy", "up", "long"),
    # 持有：已有仓位，允许波动但不应显著下跌
    "持有": AdviceSemantics("hold", "not_down", "long"),
    "hold": AdviceSemantics("hold", "not_down", "long"),
    # 离场：仓位管理，不应要求"必须下跌"才算对
    "观望": AdviceSemantics("hold", "not_up", "cash"),
    "等待": AdviceSemantics("hold", "not_up", "cash"),
    "wait": AdviceSemantics("hold", "not_up", "cash"),
    "watch": AdviceSemantics("hold", "not_up", "cash"),
    # 看空/减仓
    "强烈卖出": AdviceSemantics("sell", "down", "cash"),
    "卖出": AdviceSemantics("sell", "down", "cash"),
    "清仓": AdviceSemantics("sell", "down", "cash"),
    "strong sell": AdviceSemantics("sell", "down", "cash"),
    "sell": AdviceSemantics("sell", "down", "cash"),
    "减仓": AdviceSemantics("sell", "not_up", "cash"),
    "止盈": AdviceSemantics("sell", "not_up", "cash"),
    "止损": AdviceSemantics("sell", "not_up", "cash"),
    "reduce": AdviceSemantics("sell", "not_up", "cash"),
}

# 否定词：出现在关键词之前时，该关键词不生效（"不要卖出" / "do not sell"）
_NEGATION_PATTERNS = (
    "not", "don't", "do not", "no", "never", "avoid",
    "不要", "不", "别", "勿", "没有",
)


def _is_negated(prefix: str) -> bool:
    stripped = prefix.rstrip()
    return any(stripped.endswith(neg) for neg in _NEGATION_PATTERNS)


def resolve_advice(value: Optional[str]) -> Optional[AdviceSemantics]:
    """把建议文本解析为统一语义；无法识别返回 None。

    匹配规则（确定性，避免"持有观望"这类复合词两边解释不同）：
    1. 完整相等优先；
    2. 否则取**最早出现**的关键词；
    3. 同一位置取更长的关键词。
    """
    text = str(value or "").strip().lower()
    if not text:
        return None

    exact = ADVICE_TABLE.get(text)
    if exact is not None:
        return exact

    best_key = None
    best_semantics: Optional[AdviceSemantics] = None
    for keyword, semantics in ADVICE_TABLE.items():
        index = text.find(keyword)
        if index == -1:
            continue
        if _is_negated(text[:index]):
            continue
        key = (index, -len(keyword))
        if best_key is None or key < best_key:
            best_key = key
            best_semantics = semantics
    return best_semantics


def infer_signal(value: Optional[str]) -> str:
    """buy / hold / sell，无法识别时返回空串。"""
    semantics = resolve_advice(value)
    return semantics.signal if semantics else ""


def infer_direction(value: Optional[str]) -> str:
    """up / down / not_down / not_up，无法识别时按 flat 处理。"""
    semantics = resolve_advice(value)
    return semantics.direction if semantics else "flat"


def infer_position(value: Optional[str]) -> str:
    """long / cash，无法识别时按 cash 处理（宁可空仓）。"""
    semantics = resolve_advice(value)
    return semantics.position if semantics else "cash"

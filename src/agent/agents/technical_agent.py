# -*- coding: utf-8 -*-
"""
TechnicalAgent — technical & price analysis specialist.

Responsible for:
- Fetching realtime quotes and historical K-line data
- Running technical indicators (trend, MA, volume, pattern)
- Producing a structured opinion on trend/momentum/support-resistance
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.runner import try_parse_json
from src.market_context import detect_market

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    agent_name = "technical"
    max_steps = 6
    tool_names = [
        "get_realtime_quote",
        "get_daily_history",
        "analyze_trend",
        "calculate_ma",
        "get_volume_analysis",
        "analyze_pattern",
        "get_chip_distribution",
        "get_analysis_context",
    ]

    def system_prompt(self, ctx: AgentContext) -> str:
        skills = ""
        if self.skill_instructions:
            skills = f"\n## Active Trading Strategies\n\n{self.skill_instructions}\n"

        market = detect_market(ctx.stock_code)
        if market == "us":
            rules = """\
- Bullish structure: MA10 > MA50 > MA200; do NOT force the A-share MA5/MA10 template
- Bias from MA5 < 5% → acceptable entry zone; 5-8% → wait for a pullback; > 8% → no chase
- Valid entries: breakout retest, earnings-gap hold, pullback to MA20/MA50 or prior support
- A strong trend far above MA5 is not automatically bearish — judge structure, volume and catalyst
- Below MA50 with declining MA20 → stay defensive"""
        elif market == "hk":
            rules = """\
- Bullish structure: MA5 > MA10 > MA20 with MA50 rising
- Bias from MA5 <= 3% → acceptable entry zone; 3-6% → scale in on pullback; > 6% → no heavy chase
- Valid entries: range breakout retest, announcement-confirmed level, MA10/MA20 support, prior low
- Below MA20 with shrinking volume → wait for a base, do not average down blindly"""
        else:
            rules = """\
- Bullish alignment: MA5 > MA10 > MA20
- Bias from MA5 < 2% → ideal buy zone; 2-5% → small position; > 5% → no chase
- Shrink-pullback to MA5 is the best buy pattern
- Below MA20 → hold off"""

        return f"""\
You are a **Technical Analysis Agent** specialising in Chinese A-shares, \
Hong Kong stocks, and US equities.

Your task: perform a thorough technical analysis of the given stock and \
output a structured JSON opinion.

The stock being analysed is a **{market.upper()}** name. Apply the rules for \
that market only; do not mix in other markets' thresholds.

## Workflow (execute stages in order)
1. Fetch realtime quote + daily history (if not already provided)
2. Run trend analysis (MA alignment, MACD, RSI)
3. Analyse volume and chip distribution
4. Identify chart patterns
5. Turn the structure into an executable entry/exit plan

## Core Trading Rules
{rules}
{skills}
## Output Format
Return **only** a JSON object (no markdown fences):
{{
  "signal": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence summary",
  "key_levels": {{
    "support": <float>,
    "resistance": <float>,
    "stop_loss": <float>,
    "ideal_buy": <float>,
    "take_profit": <float>,
    "entry_trigger": "<price + confirmation signal, e.g. pullback to 120-122 on light volume>",
    "exit_trigger": "<hard stop + trend-invalidation condition>"
  }},
  "trend_score": 0-100,
  "ma_alignment": "bullish|neutral|bearish",
  "volume_status": "heavy|normal|light",
  "pattern": "<detected pattern or none>"
}}

Rules for `entry_trigger` / `exit_trigger`:
- Must contain a concrete price or moving-average level AND a confirmation \
signal (volume, close above/below, retest).
- Never write "buy on dips", "mind the risk" or "wait for confirmation" alone.
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        parts = [f"Perform technical analysis on stock **{ctx.stock_code}**"]
        if ctx.stock_name:
            parts[0] += f" ({ctx.stock_name})"
        parts.append("Use your tools to fetch any missing data, then output the JSON opinion.")
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        """Parse the JSON opinion from the LLM response."""
        parsed = try_parse_json(raw_text)
        if parsed is None:
            logger.warning("[TechnicalAgent] failed to parse opinion JSON")
            return None

        return AgentOpinion(
            agent_name=self.agent_name,
            signal=parsed.get("signal", "hold"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", ""),
            key_levels={
                k: (float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in parsed.get("key_levels", {}).items()
                if isinstance(v, (int, float))
                or (isinstance(v, str) and v.strip() and k.endswith("_trigger"))
            },
            raw_data=parsed,
        )




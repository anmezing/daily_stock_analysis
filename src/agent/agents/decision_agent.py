# -*- coding: utf-8 -*-
"""
DecisionAgent — final synthesis and decision-making specialist.

Responsible for:
- Aggregating opinions from technical + intel + risk + strategy agents
- Producing the final Decision Dashboard JSON
- Generating actionable buy/hold/sell recommendations with price levels
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion, normalize_decision_signal

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """Synthesise prior agent opinions into the final dashboard."""

    agent_name = "decision"
    max_steps = 3  # pure synthesis, should not need many tool calls
    tool_names: Optional[List[str]] = []  # no tool access — works from context only

    @staticmethod
    def _is_chat_mode(ctx: AgentContext) -> bool:
        return ctx.meta.get("response_mode") == "chat"

    def system_prompt(self, ctx: AgentContext) -> str:
        if self._is_chat_mode(ctx):
            return """\
You are a **Decision Synthesis Agent** replying directly to the user's latest
stock-analysis question.

You will receive structured opinions from the technical, intelligence, risk,
and strategy stages. Synthesize them into a concise, natural-language answer.

Requirements:
- Answer the user's actual question directly
- Use Markdown when helpful
- Keep the response practical and specific
- Highlight the main signal, key reasoning, and major risks
- Do NOT output JSON or code fences unless the user explicitly asks for them
"""

        skills = ""
        if self.skill_instructions:
            skills = f"\n## Active Trading Strategies\n\n{self.skill_instructions}\n"

        return f"""\
You are a **Decision Synthesis Agent** that produces the final investment \
Decision Dashboard.

You will receive:
1. Structured opinions from a Technical Agent and an Intel Agent
2. Any risk flags raised by a Risk Agent
3. Strategy evaluation results (if applicable)

Your task: synthesise all inputs into a single, actionable Decision Dashboard.
{skills}
## Core Principles
1. **Core conclusion first** — one sentence, ≤30 chars
2. **Split advice** — different for no-position vs has-position
3. **Precise sniper levels** — concrete price numbers, no hedging
4. **Checklist visual** — ✅⚠️❌ for each checkpoint
5. **Risk priority** — risk alerts must be prominent. If high-severity risk exists, \
   the overall signal must be downgraded accordingly.

## Signal Weighting Guidelines
- Technical opinion weight: ~40%
- Intel / sentiment weight: ~30%
- Risk flags weight: ~30% (negative override: any high-severity risk caps signal at "hold")
- If a strategy opinion is present, blend it at 20% weight (reducing others proportionally)

## Scoring
- 80-100: buy (all conditions met, high conviction)
- 60-79: buy (mostly positive, minor caveats)
- 40-59: hold (mixed signals, or risk present)
- 20-39: sell (negative trend + risk)
- 0-19: sell (major risk + bearish)

## Trade Plan — must answer when to buy / wait / sell

The user adds a stock because they want an actionable answer, not a rating.
`dashboard.battle_plan.trade_plan` must contain all five fields:

- `buy_trigger`: price zone **and** confirmation signal (e.g. "pull back to
  120–122 on shrinking volume and close green")
- `wait_condition`: what must stay true to keep waiting / not chase
- `sell_trigger`: for holders — hard stop, trend-invalidation exit, scale-out targets
- `invalidation`: the event/price that proves the thesis wrong
- `time_stop`: how many trading days without progress before exiting

Rules:
- Every field must reference a concrete price, moving average, volume condition
  or event. "Buy on dips", "mind the risk", "depends on the market" are forbidden.
- `buy` needs entry zone + stop + target; `hold` still needs both the sell
  condition and the condition that would turn it into a buy; `sell` needs the
  exit condition and the re-entry condition.
- If a reference plan is supplied in the pre-fetched data, keep its levels
  unless the news/fundamentals justify a different one — then say why.
- `sniper_points.stop_loss` and `sniper_points.take_profit` must be consistent
  with the trade plan (same currency, sane ordering: stop < entry < target).
- Use the pre-fetched `fundamental_context` (valuation multiples, growth,
  margins, analyst target, earnings date) when judging whether the move is
  supported by fundamentals. If a field is missing, say it is missing instead
  of guessing a number.
- Use `market_themes` (leading/lagging sectors) and `relative_strength`
  (20d/60d excess return) to time entries: a pullback in a leading sector is
  more actionable than a breakout in a lagging one; a new high with negative
  excess return is a divergence warning, not a buy signal.
- Use short interest and insider flow when present: crowded shorts can fuel a
  squeeze but also signal risk; sustained insider selling reduces conviction.

## Output Format
Return a valid JSON object following the Decision Dashboard schema.  The JSON \
must include at minimum these top-level keys:
  stock_name, sentiment_score, trend_prediction, operation_advice,
  decision_type, confidence_level, dashboard, analysis_summary,
  key_points, risk_warning

Important: ``decision_type`` must stay within the existing enum
``buy|hold|sell``. Express stronger conviction via ``confidence_level``,
``sentiment_score``, and the natural-language fields instead of inventing
new decision_type values.
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        if self._is_chat_mode(ctx):
            parts = [
                "# User Question",
                ctx.query,
                "",
                f"Stock: {ctx.stock_code} ({ctx.stock_name})" if ctx.stock_name else f"Stock: {ctx.stock_code}",
                "",
            ]
        else:
            parts = [
                f"# Synthesis Request for {ctx.stock_code}",
                f"Stock: {ctx.stock_code} ({ctx.stock_name})" if ctx.stock_name else f"Stock: {ctx.stock_code}",
                "",
            ]

        # Feed prior opinions
        if ctx.opinions:
            parts.append("## Agent Opinions")
            for op in ctx.opinions:
                parts.append(f"\n### {op.agent_name}")
                parts.append(f"Signal: {op.signal} | Confidence: {op.confidence:.2f}")
                parts.append(f"Reasoning: {op.reasoning}")
                if op.key_levels:
                    parts.append(f"Key levels: {json.dumps(op.key_levels)}")
                if op.raw_data:
                    extra_keys = {k: v for k, v in op.raw_data.items()
                                  if k not in ("signal", "confidence", "reasoning", "key_levels")}
                    if extra_keys:
                        parts.append(f"Extra data: {json.dumps(extra_keys, ensure_ascii=False, default=str)}")
                parts.append("")

        # Feed risk flags
        if ctx.risk_flags:
            parts.append("## Risk Flags")
            for rf in ctx.risk_flags:
                parts.append(f"- [{rf.get('severity', 'medium')}] {rf.get('category', '')}: {rf.get('description', '')}")
            parts.append("")

        # Strategy meta
        if ctx.meta.get("strategies_requested"):
            parts.append(f"## Strategies: {', '.join(ctx.meta['strategies_requested'])}")
            parts.append("")

        if self._is_chat_mode(ctx):
            parts.append(
                "Answer the user in natural language using the evidence above. "
                "Do not output JSON unless the user explicitly requests structured data."
            )
        else:
            parts.append("Synthesise the above into the Decision Dashboard JSON.")
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        """Store the parsed dashboard in ctx.meta; also return an opinion."""
        if self._is_chat_mode(ctx):
            text = (raw_text or "").strip()
            if not text:
                return None

            ctx.set_data("final_response_text", text)
            prior = next((op for op in reversed(ctx.opinions) if op.agent_name != self.agent_name), None)
            return AgentOpinion(
                agent_name=self.agent_name,
                signal=prior.signal if prior is not None else "hold",
                confidence=prior.confidence if prior is not None else 0.5,
                reasoning=text,
                raw_data={"response_mode": "chat"},
            )

        from src.agent.runner import parse_dashboard_json

        dashboard = parse_dashboard_json(raw_text)
        if dashboard:
            dashboard["decision_type"] = normalize_decision_signal(
                dashboard.get("decision_type", "hold")
            )
            ctx.set_data("final_dashboard", dashboard)
            try:
                _raw_score = dashboard.get("sentiment_score", 50) or 50
                _score = float(_raw_score)
            except (TypeError, ValueError):
                _score = 50.0
            return AgentOpinion(
                agent_name=self.agent_name,
                signal=dashboard.get("decision_type", "hold"),
                confidence=min(1.0, _score / 100.0),
                reasoning=dashboard.get("analysis_summary", ""),
                raw_data=dashboard,
            )
        else:
            # Even if JSON parsing fails, store the raw text for downstream use
            ctx.set_data("final_dashboard_raw", raw_text)
            logger.warning("[DecisionAgent] failed to parse dashboard JSON")
            return None

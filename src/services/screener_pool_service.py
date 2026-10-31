# -*- coding: utf-8 -*-
"""Multi-day candidate pool for the screener.

The screener produces a fresh list every day.  A human workflow is different:
a name is interesting only if it *stays* in the hot sector for several days,
its relative strength holds, and its fundamentals do not deteriorate.  This
service turns daily picks into that persistent watchlist.

Stages:
- ``new``       : first day in the pool
- ``watching``  : 2 consecutive days, or temporarily missing
- ``ready``     : >=3 consecutive days + sector still hot + RS positive
- ``trigger``   : ready and currently in a pullback/entry setup
- ``dropped``   : missing for >=2 runs or removed after 30 days
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_POOL_PATH = Path(__file__).resolve().parents[2] / "data" / "screener_pool.json"
_MAX_PER_SECTOR = 2
_MAX_HISTORY = 10
_READY_DAYS = 3
_MAX_ABSENT_DAYS = 10


def _project_path() -> Path:
    return _POOL_PATH


def _fundamental_score(pick: Dict[str, Any]) -> Optional[float]:
    """Compact 0-100 quality score from the fields captured at selection."""
    values: List[float] = []

    growth = pick.get("revenue_growth")
    if growth is not None:
        values.append(max(0.0, min(100.0, 50 + float(growth) * 200)))
    margin = pick.get("profit_margin")
    if margin is not None:
        values.append(max(0.0, min(100.0, 50 + float(margin) * 250)))
    roe = pick.get("return_on_equity")
    if roe is not None:
        values.append(max(0.0, min(100.0, 50 + float(roe) * 150)))
    fcf = pick.get("free_cashflow")
    if fcf is not None:
        values.append(100.0 if float(fcf) > 0 else 30.0)
    debt = pick.get("debt_to_equity")
    if debt is not None:
        if float(debt) < 50:
            values.append(100.0)
        elif float(debt) < 150:
            values.append(75.0)
        elif float(debt) < 300:
            values.append(50.0)
        else:
            values.append(25.0)
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def _trend(history: List[float]) -> str:
    if len(history) < 2:
        return "unknown"
    recent = history[-3:]
    if recent[-1] > recent[0] + 2:
        return "improving"
    if recent[-1] < recent[0] - 2:
        return "worsening"
    return "stable"


class ScreenerPoolService:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else _project_path()

    def load(self) -> Dict[str, Any]:
        try:
            if self._path.exists():
                payload = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
        except Exception as exc:
            logger.debug("[候选池] 读取失败: %s", exc)
        return {"updated_at": None, "items": {}, "summary": {}}

    def update(self, result: Dict[str, Any], today: Optional[str] = None) -> Dict[str, Any]:
        pool = self.load()
        items: Dict[str, Any] = pool.get("items") or {}
        run_day = (today or result.get("as_of") or date.today().isoformat())[:10]
        picks = [
            item for item in ((result.get("picks") or []) + (result.get("challengers") or []))
            if item.get("symbol")
        ]
        today_symbols = {str(item["symbol"]).upper() for item in picks}
        sectors = {str(row.get("symbol")) for row in (result.get("sectors") or [])}
        for item in items.values():
            item["today_selected"] = False

        for pick in picks:
            symbol = str(pick["symbol"]).upper()
            existing = items.get(symbol) or {}
            previous_day = str(existing.get("last_seen") or "")[:10]
            same_day = previous_day == run_day
            if same_day:
                consecutive = int(existing.get("consecutive_days") or 1)
            elif previous_day:
                try:
                    gap = (date.fromisoformat(run_day) - date.fromisoformat(previous_day)).days
                except ValueError:
                    gap = 99
                consecutive = int(existing.get("consecutive_days") or 0) + 1 if gap <= 4 else 1
            else:
                consecutive = 1

            fund_score = _fundamental_score(pick)
            fund_history = list(existing.get("fundamental_history") or [])
            if fund_score is not None:
                fund_history.append(fund_score)
                fund_history = fund_history[-_MAX_HISTORY:]

            score_history = list(existing.get("score_history") or [])
            score_history.append(float(pick.get("score") or 0))
            score_history = score_history[-_MAX_HISTORY:]

            fund_trend = _trend(fund_history)
            sector_ok = str(pick.get("sector_etf")) in sectors
            rs_ok = float(pick.get("rs_15d") or 0) > 0
            risk = str(pick.get("risk_level") or "low")
            previous_stage = str(existing.get("stage") or "")
            ready = (
                consecutive >= _READY_DAYS
                and sector_ok
                and rs_ok
                and fund_trend != "worsening"
                and risk != "high"
            )
            if ready and str(pick.get("state")) == "pullback":
                stage = "trigger"
            elif ready:
                stage = "ready"
            elif consecutive >= 2:
                stage = "watching"
            else:
                stage = "new"
            promoted_today = stage in ("ready", "trigger") and previous_stage not in ("ready", "trigger")

            items[symbol] = {
                "symbol": symbol,
                "name": pick.get("name"),
                "sector_etf": pick.get("sector_etf"),
                "sector_name": pick.get("sector_name"),
                "first_seen": existing.get("first_seen") or run_day,
                "last_seen": run_day,
                "days_seen": int(existing.get("days_seen") or 0) + (0 if same_day else 1),
                "consecutive_days": consecutive,
                "missing_days": 0,
                "today_selected": True,
                "stage": stage,
                "stage_changed": stage != previous_stage,
                "promoted_today": promoted_today,
                "last_score": pick.get("score"),
                "score_history": score_history,
                "fundamental_score": fund_score if fund_score is not None else existing.get("fundamental_score"),
                "fundamental_history": fund_history,
                "fundamental_trend": fund_trend,
                "last_state": pick.get("state"),
                "last_state_label": pick.get("state_label"),
                "risk_level": risk,
                "risk_note": pick.get("risk_note"),
                "plan": pick.get("plan"),
            }

        # Same-sector competition: a new stronger name must not be blocked by
        # an older pool member.  Keep the best two per sector; if a challenger
        # is clearly stronger, the weaker incumbent is marked replaced.
        sector_items: Dict[str, List[Dict[str, Any]]] = {}
        for symbol, item in items.items():
            if item.get("stage") == "dropped":
                continue
            sector = str(item.get("sector_etf") or "")
            sector_items.setdefault(sector, []).append(item)
        for sector, rows in sector_items.items():
            rows.sort(key=lambda item: float(item.get("last_score") or 0), reverse=True)
            for weak in rows[_MAX_PER_SECTOR:]:
                weak["stage"] = "dropped"
                weak["drop_reason"] = f"同板块竞争落选（保留前 {_MAX_PER_SECTOR}）"

        # Names that disappeared from today's list are NOT dropped immediately:
        # the pool is cumulative, and a good name often pauses for a few days.
        # Keep it until stop/risk or 10 consecutive absences.
        for symbol, item in list(items.items()):
            if symbol in today_symbols:
                continue
            last_seen = str(item.get("last_seen") or "")[:10]
            if last_seen == run_day:
                continue
            missing = int(item.get("missing_days") or 0) + 1
            item["missing_days"] = missing
            item["today_selected"] = False
            last_stage = item.get("stage")
            if missing >= _MAX_ABSENT_DAYS:
                item["stage"] = "dropped"
                item["drop_reason"] = f"连续 {missing} 个交易日未入选"
            elif item.get("stage") == "new":
                item["stage"] = "watching"
            item["stage_changed"] = item.get("stage") != last_stage
            item["promoted_today"] = False
            try:
                age = (date.fromisoformat(run_day) - date.fromisoformat(last_seen)).days
            except ValueError:
                age = 99
            if age > 30:
                items.pop(symbol, None)

        summary = {
            "new": sum(1 for item in items.values() if item.get("stage") == "new"),
            "watching": sum(1 for item in items.values() if item.get("stage") == "watching"),
            "ready": sum(1 for item in items.values() if item.get("stage") == "ready"),
            "trigger": sum(1 for item in items.values() if item.get("stage") == "trigger"),
            "dropped": sum(1 for item in items.values() if item.get("stage") == "dropped"),
        }
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "trade_date": run_day,
            "items": items,
            "summary": summary,
        }
        self._write(payload)
        return payload

    def apply_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mark pool items that hit a hard exit condition."""
        if not alerts:
            return self.load()
        pool = self.load()
        items = pool.get("items") or {}
        for alert in alerts:
            symbol = str(alert.get("symbol") or "").upper()
            item = items.get(symbol)
            if not item:
                continue
            kind = str(alert.get("kind") or "")
            if kind in ("stop_breach", "dilution"):
                item["stage"] = "dropped"
                item["drop_reason"] = alert.get("message")
            elif kind == "target_hit":
                item["stage"] = "dropped"
                item["drop_reason"] = "目标已达成"
            elif kind in ("material_event", "activist_stake"):
                item["risk_level"] = "medium"
                item["risk_note"] = alert.get("message")
            elif kind == "earnings_soon":
                item["risk_note"] = alert.get("message")
        pool["items"] = items
        self._write(pool)
        return pool

    def build_notification(self, pool: Dict[str, Any]) -> str:
        """Only notify promotions into ready/trigger, never day-1 picks."""
        items = list((pool.get("items") or {}).values())
        promoted = [
            item for item in items
            if item.get("promoted_today") and item.get("stage") in ("ready", "trigger")
        ]
        promoted.sort(
            key=lambda item: (
                1 if item.get("stage") == "trigger" else 0,
                int(item.get("consecutive_days") or 0),
            ),
            reverse=True,
        )
        if not promoted:
            return ""
        lines = ["## 🧺 股池新晋关注", ""]
        for item in promoted[:10]:
            label = "触发买入" if item.get("stage") == "trigger" else "买入候选"
            lines.append(
                f"- **{item.get('symbol')}** {item.get('name') or ''}："
                f"连续 {item.get('consecutive_days')} 天在池，{label}，"
                f"基本面{item.get('fundamental_trend') or '未知'}，评分 {item.get('last_score')}"
            )
        lines.append("")
        lines.append("> 只推送连续在池且晋级的标的；每日新入选不推送。")
        return "\n".join(lines)

    def render_markdown(self, pool: Dict[str, Any], limit: int = 15) -> str:
        items = [
            item for item in (pool.get("items") or {}).values()
            if item.get("stage") != "dropped"
        ]
        items.sort(
            key=lambda item: (
                {"trigger": 4, "ready": 3, "watching": 2}.get(str(item.get("stage")), 1),
                int(item.get("consecutive_days") or 0),
            ),
            reverse=True,
        )
        if not items:
            return ""
        lines = [
            "### 🧺 股池（持续跟踪，不按天替换）",
            "",
            "| 代码 | 名称 | 连续 | 今日入选 | 基本面趋势 | 状态 | 板块 |",
            "|------|------|------|----------|------------|------|------|",
        ]
        labels = {
            "trigger": "触发买入",
            "ready": "买入候选",
            "watching": "继续观察",
            "new": "观察第1天",
        }
        trend_labels = {
            "improving": "改善",
            "stable": "持平",
            "worsening": "转弱",
            "unknown": "数据不足",
        }
        for item in items[:limit]:
            lines.append(
                f"| {item.get('symbol')} | {item.get('name') or ''} | "
                f"{item.get('consecutive_days')} | "
                f"{'是' if item.get('today_selected') else '否'} | "
                f"{trend_labels.get(str(item.get('fundamental_trend')), '--')} | "
                f"{labels.get(str(item.get('stage')), item.get('stage'))} | "
                f"{item.get('sector_name') or ''} |"
            )
        return "\n".join(lines)

    def _write(self, payload: Dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._path.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self._path)
        except Exception as exc:
            logger.debug("[候选池] 写入失败: %s", exc)


_POOL = ScreenerPoolService()


def get_screener_pool_service() -> ScreenerPoolService:
    return _POOL

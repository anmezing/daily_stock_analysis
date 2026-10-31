# -*- coding: utf-8 -*-
"""Post-selection monitor for screener picks.

The screener cannot predict a fraud or an overnight guidance cut.  What it can
do is watch every selected name after the fact and raise an alert when:

- price breaks the planned stop or reaches target 1
- earnings are within 3 days
- a new 8-K / secondary offering / 13D appears after the pick date
- the pick has been open for 30 days without resolution

Alerts are deduplicated per symbol/date and pushed through the normal
notification channels.  This runs inside the screener line only.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScreenerMonitorService:
    def __init__(
        self,
        learning_service: Any,
        filing_provider: Any = None,
        notify: bool = True,
        active_days: int = 30,
    ) -> None:
        self._learning = learning_service
        self._filing_provider = filing_provider
        self._notify = notify
        self._active_days = max(5, int(active_days))

    def check(self) -> Dict[str, Any]:
        records = self._learning._read_ledger()
        if not records:
            return {"status": "empty", "alerts": []}

        today = date.today()
        active: List[Dict[str, Any]] = []
        for record in records:
            if record.get("closed"):
                continue
            raw_day = record.get("screen_date")
            try:
                pick_day = date.fromisoformat(str(raw_day)[:10])
            except ValueError:
                continue
            if (today - pick_day).days > self._active_days:
                record["closed"] = True
                record["closed_reason"] = "time_stop"
                continue
            active.append(record)

        if not active:
            self._learning._write_ledger(records)
            return {"status": "ok", "alerts": []}

        symbols = sorted({str(item.get("symbol") or "").upper() for item in active})
        history = self._learning._default_history(symbols) or {}
        alerts: List[Dict[str, Any]] = []

        for record in active:
            symbol = str(record.get("symbol") or "").upper()
            rows = history.get(symbol) or []
            if not rows:
                continue
            latest = rows[-1]
            plan = record.get("plan") or {}

            def _number(key: str) -> Optional[float]:
                try:
                    return float(plan.get(key))
                except (TypeError, ValueError):
                    return None

            stop = _number("stop_loss")
            target = _number("target_1")
            sent = record.get("alerts_sent") or {}
            day_key = today.isoformat()

            def emit(kind: str, message: str) -> None:
                key = f"{kind}:{day_key}"
                if sent.get(key):
                    return
                sent[key] = day_key
                alerts.append({
                    "symbol": symbol,
                    "name": record.get("name"),
                    "kind": kind,
                    "message": message,
                    "date": day_key,
                })

            if stop is not None and latest.close <= stop:
                emit("stop_breach", f"现价 {latest.close:.2f} 跌破止损 {stop:.2f}")
                record["closed"] = True
                record["closed_reason"] = "stop_loss"
            elif target is not None and latest.close >= target:
                emit("target_hit", f"现价 {latest.close:.2f} 触及目标1 {target:.2f}")
                record["closed"] = True
                record["closed_reason"] = "target_hit"

            earnings_date = record.get("earnings_date")
            if earnings_date:
                try:
                    earnings_day = date.fromisoformat(str(earnings_date)[:10])
                    days = (earnings_day - today).days
                    if 0 <= days <= 3:
                        emit("earnings_soon", f"距财报 {days} 天，注意事件风险")
                except ValueError:
                    pass

            if self._filing_provider is not None:
                try:
                    response = self._filing_provider.get_recent_filings(symbol, limit=10) or {}
                    items = response.get("items") or []
                except Exception:
                    items = []
                for item in items:
                    form = str(item.get("form") or "").upper()
                    filed = str(item.get("filed") or "")[:10]
                    try:
                        filed_day = date.fromisoformat(filed)
                    except ValueError:
                        continue
                    if filed_day < pick_day:
                        continue
                    if form in ("424B5", "424B3"):
                        emit("dilution", f"{form} {filed} 新增发行/增发")
                    elif form == "8-K":
                        emit("material_event", f"8-K {filed} 重大事项")
                    elif form in ("SC 13D", "SC 13D/A"):
                        emit("activist_stake", f"{form} {filed} activist/并购披露")

            record["alerts_sent"] = sent

        self._learning._write_ledger(records)
        self._write_alerts(alerts)
        if alerts and self._notify:
            self._send(alerts)
        return {"status": "ok", "alerts": alerts}

    @staticmethod
    def _write_alerts(alerts: List[Dict[str, Any]]) -> None:
        try:
            path = Path(__file__).resolve().parents[2] / "data" / "screener_alerts.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "alerts": alerts,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("[选股监控] 告警写入失败: %s", exc)

    @staticmethod
    def _send(alerts: List[Dict[str, Any]]) -> None:
        lines = ["## 🚨 选股持仓告警", ""]
        for item in alerts[:20]:
            lines.append(
                f"- **{item.get('symbol')}** {item.get('name') or ''}：{item.get('message')}"
            )
        try:
            from src.notification import NotificationService

            NotificationService().send("\n".join(lines))
        except Exception as exc:
            logger.warning("[选股监控] 告警推送失败: %s", exc)

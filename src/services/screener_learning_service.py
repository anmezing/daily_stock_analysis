# -*- coding: utf-8 -*-
"""Self-learning loop for the hot-sector screener.

It does three things, all with hard guardrails against overfitting:

1. **Record** every pick together with the features that produced it.
2. **Evaluate** matured picks at T+5 / T+10 / T+20 against SPY and the sector.
3. **Adjust** the next run: small score bonuses/penalties per setup/sector,
   disable rules that are consistently negative, and nudge the minimum score.

Adjustments only kick in after a minimum sample count and are capped, so a
handful of lucky/unlucky picks cannot rewrite the model.  State is kept in
JSON files under ``data/`` so it survives container restarts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MIN_SAMPLES = 20
_MAX_ADJUSTMENT = 8.0
_STATE_TTL_HOURS = 24
_HORIZONS = (5, 10, 20)


def _project_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _pick_date(record: Dict[str, Any]) -> Optional[date]:
    raw = record.get("screen_date")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _excess(record: Dict[str, Any], horizon: int) -> Optional[float]:
    outcome = (record.get("outcomes") or {}).get(f"d{horizon}") or {}
    return _safe_float(outcome.get("excess_spy"))


def _hit_rate(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(1 for value in values if value > 0) / len(values) * 100, 1)


def _group_stats(
    records: Sequence[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], str],
    min_samples: int,
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[float]] = {}
    for record in records:
        value = _excess(record, 10)
        if value is None:
            continue
        key = key_fn(record)
        if not key:
            continue
        groups.setdefault(key, []).append(value)

    result: Dict[str, Dict[str, Any]] = {}
    for key, values in groups.items():
        if len(values) < min_samples:
            continue
        average = sum(values) / len(values)
        result[key] = {
            "samples": len(values),
            "hit_rate_10d": _hit_rate(values),
            "avg_excess_10d": round(average, 2),
        }
    return result


def compute_adjustments(
    records: Sequence[Dict[str, Any]],
    min_samples: int = _DEFAULT_MIN_SAMPLES,
) -> Dict[str, Any]:
    """Derive small, sample-gated adjustments from evaluated picks."""
    records = [record for record in records if _excess(record, 10) is not None]
    overall_values = [
        value for value in (_excess(record, 10) for record in records)
        if value is not None
    ]
    overall = {
        "samples": len(overall_values),
        "hit_rate_10d": _hit_rate(overall_values),
        "avg_excess_10d": round(sum(overall_values) / len(overall_values), 2) if overall_values else None,
    }

    state_stats = _group_stats(
        records,
        lambda record: str(record.get("state") or ""),
        min_samples,
    )
    sector_stats = _group_stats(
        records,
        lambda record: str(record.get("sector_etf") or ""),
        min_samples,
    )

    state_bonus: Dict[str, float] = {}
    sector_bonus: Dict[str, float] = {}
    disabled_states: List[str] = []
    disabled_sectors: List[str] = []

    def apply_group(
        stats: Dict[str, Dict[str, Any]],
        bonus: Dict[str, float],
        disabled: List[str],
    ) -> None:
        for key, item in stats.items():
            if item["samples"] < min_samples:
                continue
            avg = item["avg_excess_10d"]
            hit = item["hit_rate_10d"]
            if avg <= -2.0 and hit <= 40 and item["samples"] >= min_samples * 1.5:
                disabled.append(key)
            elif avg >= 1.0 and hit >= 55:
                bonus[key] = 3.0
            elif avg <= -1.0 and hit <= 45:
                bonus[key] = -5.0

    apply_group(state_stats, state_bonus, disabled_states)
    apply_group(sector_stats, sector_bonus, disabled_sectors)

    min_score_delta = 0.0
    if overall_values and len(overall_values) >= min_samples:
        if overall["hit_rate_10d"] < 45:
            min_score_delta = 5.0
        elif overall["hit_rate_10d"] > 60 and (overall["avg_excess_10d"] or 0) > 0:
            min_score_delta = -3.0

    return {
        "version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "min_samples": min_samples,
        "overall": overall,
        "state_stats": state_stats,
        "sector_stats": sector_stats,
        "state_bonus": state_bonus,
        "sector_bonus": sector_bonus,
        "disabled_states": sorted(set(disabled_states)),
        "disabled_sectors": sorted(set(disabled_sectors)),
        "min_score_delta": min_score_delta,
    }


@dataclass
class _HistoryRow:
    day: date
    close: float


class ScreenerLearningService:
    """JSON-backed ledger + evaluator + adjustment store."""

    def __init__(
        self,
        ledger_path: Optional[str] = None,
        state_path: Optional[str] = None,
        history_provider: Optional[Callable[[List[str]], Dict[str, List[_HistoryRow]]]] = None,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        refresh_hours: int = _STATE_TTL_HOURS,
    ) -> None:
        data_dir = _project_data_dir()
        self._ledger_path = Path(ledger_path) if ledger_path else data_dir / "screener_picks.jsonl"
        self._state_path = Path(state_path) if state_path else data_dir / "screener_learning.json"
        self._history_provider = history_provider or self._default_history
        self._min_samples = max(5, int(min_samples))
        self._refresh_hours = max(1, int(refresh_hours))
        self._lock = RLock()
        self._cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_adjustments(self) -> Dict[str, Any]:
        with self._lock:
            if self._cache is not None:
                return self._cache
            try:
                if self._state_path.exists():
                    self._cache = json.loads(self._state_path.read_text(encoding="utf-8"))
                else:
                    self._cache = {}
            except Exception as exc:
                logger.debug("[选股学习] 读取调整状态失败: %s", exc)
                self._cache = {}
            return self._cache

    def maybe_refresh(self) -> Dict[str, Any]:
        """Evaluate matured picks and refresh adjustments at most once a day."""
        adjustments = self.load_adjustments()
        pending = self._has_pending()
        stale = True
        updated_at = adjustments.get("updated_at")
        if updated_at:
            try:
                age = datetime.now() - datetime.fromisoformat(updated_at)
                stale = age >= timedelta(hours=self._refresh_hours)
            except ValueError:
                stale = True

        if pending or stale:
            evaluated = self.evaluate_pending()
            if evaluated or stale:
                adjustments = self.refresh_adjustments()
        return adjustments

    def record_picks(
        self,
        result: Dict[str, Any],
        screen_date: Optional[str] = None,
    ) -> int:
        """Append today's picks, replacing duplicates for the same day/symbol."""
        picks = result.get("picks") or []
        if not picks or result.get("status") not in ("ok", "partial"):
            return 0
        day = (screen_date or date.today().isoformat())[:10]
        records = self._read_ledger()
        index = {(item.get("screen_date"), item.get("symbol")): item for item in records}
        written = 0
        for pick in picks:
            symbol = str(pick.get("symbol") or "").upper()
            if not symbol:
                continue
            record = {
                "screen_date": day,
                "symbol": symbol,
                "name": pick.get("name"),
                "sector_etf": pick.get("sector_etf"),
                "sector_name": pick.get("sector_name"),
                "state": pick.get("state"),
                "score": pick.get("score"),
                "close": pick.get("close"),
                "return_15d": pick.get("return_15d"),
                "return_30d": pick.get("return_30d"),
                "rs_15d": pick.get("rs_15d"),
                "rs_30d": pick.get("rs_30d"),
                "dist_ma20_pct": pick.get("dist_ma20_pct"),
                "volume_ratio_5_20": pick.get("volume_ratio_5_20"),
                "plan": pick.get("plan") or {},
                "earnings_date": pick.get("earnings_date"),
                "days_to_earnings": pick.get("days_to_earnings"),
                "risk_level": pick.get("risk_level"),
                "filing_risks": pick.get("filing_risks") or [],
                "is_challenger": bool(pick.get("is_challenger")),
                "outcomes": index.get((day, symbol), {}).get("outcomes", {}),
            }
            index[(day, symbol)] = record
            written += 1
        self._write_ledger(list(index.values()))
        return written

    def evaluate_pending(self) -> int:
        """Fill T+5/T+10/T+20 outcomes for picks old enough to evaluate."""
        records = self._read_ledger()
        today = date.today()
        pending: List[Tuple[int, Dict[str, Any], date]] = []
        symbols = set()
        for position, record in enumerate(records):
            day = _pick_date(record)
            if day is None:
                continue
            if (today - day).days < 7:
                continue
            outcomes = record.get("outcomes") or {}
            missing = [horizon for horizon in _HORIZONS if f"d{horizon}" not in outcomes]
            if not missing:
                continue
            pending.append((position, record, day))
            symbols.add(str(record.get("symbol") or "").upper())
            sector = str(record.get("sector_etf") or "").upper()
            if sector:
                symbols.add(sector)
        symbols.add("SPY")
        if not pending:
            return 0

        history = self._history_provider(sorted(symbols)) or {}
        updated = 0
        for position, record, day in pending:
            symbol = str(record.get("symbol") or "").upper()
            sector = str(record.get("sector_etf") or "").upper()
            stock_rows = history.get(symbol) or []
            spy_rows = history.get("SPY") or []
            sector_rows = history.get(sector) or [] if sector else []
            start = next((index for index, row in enumerate(stock_rows) if row.day >= day), None)
            if start is None:
                continue
            outcomes = record.get("outcomes") or {}
            for horizon in _HORIZONS:
                key = f"d{horizon}"
                if key in outcomes:
                    continue
                target = start + horizon
                if target >= len(stock_rows):
                    continue
                base = stock_rows[start].close
                if base <= 0:
                    continue
                stock_return = (stock_rows[target].close / base - 1.0) * 100.0
                spy_return = None
                if target < len(spy_rows) and spy_rows[start].close > 0:
                    spy_return = (spy_rows[target].close / spy_rows[start].close - 1.0) * 100.0
                sector_return = None
                if sector_rows and target < len(sector_rows) and sector_rows[start].close > 0:
                    sector_return = (sector_rows[target].close / sector_rows[start].close - 1.0) * 100.0
                outcomes[key] = {
                    "return": round(stock_return, 2),
                    "excess_spy": round(stock_return - spy_return, 2) if spy_return is not None else None,
                    "excess_sector": round(stock_return - sector_return, 2) if sector_return is not None else None,
                }
            record["outcomes"] = outcomes
            record["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
            updated += 1

        if updated:
            self._write_ledger(records)
        return updated

    def refresh_adjustments(self) -> Dict[str, Any]:
        records = self._read_ledger()
        adjustments = compute_adjustments(records, min_samples=self._min_samples)
        adjustments["ledger_size"] = len(records)
        with self._lock:
            self._cache = adjustments
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(adjustments, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp, self._state_path)
        logger.info(
            "[选股学习] 调整已更新: samples=%s hit=%s%% avg_excess=%s%% disabled_states=%s",
            adjustments.get("overall", {}).get("samples"),
            adjustments.get("overall", {}).get("hit_rate_10d"),
            adjustments.get("overall", {}).get("avg_excess_10d"),
            adjustments.get("disabled_states"),
        )
        return adjustments

    def apply_adjustments(
        self,
        picks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        adjustments = self.load_adjustments() or {}
        state_bonus = adjustments.get("state_bonus") or {}
        sector_bonus = adjustments.get("sector_bonus") or {}
        disabled_states = set(adjustments.get("disabled_states") or [])
        disabled_sectors = set(adjustments.get("disabled_sectors") or [])

        filtered: List[Dict[str, Any]] = []
        removed = 0
        for pick in picks:
            state = str(pick.get("state") or "")
            sector = str(pick.get("sector_etf") or "")
            if state in disabled_states or sector in disabled_sectors:
                removed += 1
                continue
            adjustment = float(state_bonus.get(state, 0.0)) + float(sector_bonus.get(sector, 0.0))
            adjustment = max(-_MAX_ADJUSTMENT, min(_MAX_ADJUSTMENT, adjustment))
            if adjustment:
                pick["score"] = round(float(pick.get("score") or 0) + adjustment, 1)
                pick["learning_adjustment"] = adjustment
                notes = []
                if state_bonus.get(state):
                    notes.append(f"{state} {state_bonus[state]:+.0f}")
                if sector_bonus.get(sector):
                    notes.append(f"{sector} {sector_bonus[sector]:+.0f}")
                pick["learning_note"] = "；".join(notes)
            filtered.append(pick)

        filtered.sort(key=lambda item: item.get("score") or 0, reverse=True)
        overall = adjustments.get("overall") or {}
        return filtered, {
            "adjustment_removed": removed,
            "min_score_delta": float(adjustments.get("min_score_delta") or 0),
            "updated_at": adjustments.get("updated_at"),
            "samples": overall.get("samples", 0),
            "hit_rate_10d": overall.get("hit_rate_10d"),
            "avg_excess_10d": overall.get("avg_excess_10d"),
            "disabled_states": adjustments.get("disabled_states") or [],
        }

    # ------------------------------------------------------------------
    # Ledger helpers
    # ------------------------------------------------------------------
    def _has_pending(self) -> bool:
        today = date.today()
        for record in self._read_ledger():
            day = _pick_date(record)
            if day is None or (today - day).days < 7:
                continue
            outcomes = record.get("outcomes") or {}
            if any(f"d{horizon}" not in outcomes for horizon in _HORIZONS):
                return True
        return False

    def _read_ledger(self) -> List[Dict[str, Any]]:
        if not self._ledger_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        try:
            with self._ledger_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        records.append(item)
        except Exception as exc:
            logger.debug("[选股学习] 读取台账失败: %s", exc)
        return records

    def _write_ledger(self, records: Sequence[Dict[str, Any]]) -> None:
        with self._lock:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._ledger_path.with_suffix(".tmp")
            with temp.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(temp, self._ledger_path)

    @staticmethod
    def _default_history(symbols: List[str]) -> Dict[str, List[_HistoryRow]]:
        try:
            import yfinance as yf

            frame = yf.download(
                tickers=" ".join(symbols),
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.warning("[选股学习] 行情下载失败: %s", exc)
            return ScreenerLearningService._twelvedata_history(symbols)
        if frame is None or getattr(frame, "empty", True):
            return ScreenerLearningService._twelvedata_history(symbols)

        columns = getattr(frame, "columns", None)
        multi = getattr(columns, "nlevels", 1) > 1
        result: Dict[str, List[_HistoryRow]] = {}
        for symbol in symbols:
            try:
                sub = frame[symbol] if multi else frame
                series = sub["Close"].dropna()
                rows = [
                    _HistoryRow(day=index.date(), close=float(value))
                    for index, value in series.items()
                ]
                if rows:
                    result[symbol] = rows
            except Exception:
                continue
        return result or ScreenerLearningService._twelvedata_history(symbols)

    @staticmethod
    def _twelvedata_history(symbols: List[str]) -> Dict[str, List[_HistoryRow]]:
        try:
            from data_provider.twelvedata_fetcher import TwelveDataFetcher

            fetcher = TwelveDataFetcher()
            if not fetcher.is_api_ready():
                return {}
            result: Dict[str, List[_HistoryRow]] = {}
            for symbol in symbols:
                try:
                    frame, _ = fetcher.get_daily_data(symbol, days=300)
                    if frame is None or getattr(frame, "empty", True):
                        continue
                    date_column = "date" if "date" in frame.columns else frame.columns[0]
                    rows = []
                    for _, row in frame.iterrows():
                        try:
                            day = datetime.fromisoformat(str(row[date_column])[:10]).date()
                            close = float(row["close"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        rows.append(_HistoryRow(day=day, close=close))
                    if rows:
                        result[symbol] = rows
                except Exception:
                    continue
            if result:
                logger.info("[选股学习] 已切换到 Twelve Data 兜底行情")
            return result
        except Exception as exc:
            logger.debug("[选股学习] Twelve Data 兜底失败: %s", exc)
            return {}


_SERVICE = ScreenerLearningService()


def get_screener_learning_service() -> ScreenerLearningService:
    return _SERVICE

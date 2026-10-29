# -*- coding: utf-8 -*-
"""Portfolio-level backtest: cross-sectional ranking + capital allocation.

The per-signal backtest answers "was this call right?". This service answers
"what would an account have earned?" by ranking same-day signals, sizing
positions, holding for a fixed horizon and marking equity to market daily.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from data_provider.base import canonical_stock_code
from src.core.backtest_engine import BacktestEngine
from src.core.trading_calendar import get_market_for_stock
from src.repositories.backtest_repo import BacktestRepository
from src.services.benchmark_service import BenchmarkService

logger = logging.getLogger(__name__)


@dataclass
class PortfolioBacktestConfig:
    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    max_positions: int = 5
    holding_days: int = 10
    min_score: int = 0
    commission_bps: float = 0.0
    slippage_bps: float = 5.0
    benchmark_market: Optional[str] = None


@dataclass
class _Signal:
    code: str
    signal_date: date
    decision: str
    score: int


@dataclass
class _Position:
    code: str
    shares: float
    entry_price: float
    entry_date: date
    exit_date: date
    entry_cost: float


@dataclass
class PortfolioBacktestResult:
    metrics: Dict[str, Any] = field(default_factory=dict)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)


class PortfolioBacktestService:
    """Long-only daily portfolio simulation over stored analyses."""

    def __init__(self, db: Any, config: Any = None) -> None:
        self.db = db
        if config is None:
            from src.config import get_config

            config = get_config()
        self.config = config
        self.repo = BacktestRepository(db)
        self.benchmark = BenchmarkService(db, config=config)

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------
    def _load_signals(self, cfg: PortfolioBacktestConfig) -> List[_Signal]:
        from src.storage import AnalysisHistory

        with self.db.get_session() as session:
            rows = (
                session.query(AnalysisHistory)
                .filter(AnalysisHistory.created_at >= cfg.start_date - timedelta(days=30))
                .filter(AnalysisHistory.created_at < cfg.end_date + timedelta(days=1))
                .order_by(AnalysisHistory.created_at.desc())
                .all()
            )

        signals: List[_Signal] = []
        seen: set = set()
        for row in rows:
            if not BacktestRepository._is_evaluable(row):
                continue
            payload = self._parse_payload(row.raw_result)
            if payload.get("numeric_consistency") is False:
                continue
            signal_date = (
                BacktestRepository.parse_analysis_date_from_snapshot(row.context_snapshot)
                or (row.created_at.date() if row.created_at else None)
            )
            if signal_date is None or not (cfg.start_date <= signal_date <= cfg.end_date):
                continue
            decision = str(payload.get("decision_type") or "").strip().lower()
            if decision not in {"buy", "hold", "sell"}:
                decision = BacktestEngine.infer_direction_expected(row.operation_advice)
                decision = {"up": "buy", "down": "sell"}.get(decision, "hold")
            key = (canonical_stock_code(row.code), signal_date)
            if key in seen:
                continue
            seen.add(key)
            signals.append(
                _Signal(
                    code=canonical_stock_code(row.code),
                    signal_date=signal_date,
                    decision=decision,
                    score=int(payload.get("sentiment_score") or row.sentiment_score or 50),
                )
            )
        signals.sort(key=lambda item: (item.signal_date, -item.score, item.code))
        return signals

    @staticmethod
    def _parse_payload(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (TypeError, ValueError):
                return {}
        return {}

    def _load_bars(self, codes: List[str], start: date, end: date) -> Dict[str, Dict[date, Any]]:
        buffer_start = start - timedelta(days=10)
        buffer_end = end + timedelta(days=30)
        result: Dict[str, Dict[date, Any]] = {}
        for code in codes:
            rows = self.db.get_data_range(code, buffer_start, buffer_end)
            result[code] = {row.date: row for row in rows if row.date is not None}
        return result

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def run(self, cfg: Optional[PortfolioBacktestConfig] = None) -> PortfolioBacktestResult:
        cfg = cfg or PortfolioBacktestConfig(start_date=date.today(), end_date=date.today())
        signals = self._load_signals(cfg)
        if not signals:
            return PortfolioBacktestResult(metrics={"trades": 0, "reason": "no_signals"})

        codes = sorted({signal.code for signal in signals})
        bars = self._load_bars(codes, cfg.start_date, cfg.end_date)
        calendar = sorted(
            {
                day
                for code_bars in bars.values()
                for day in code_bars
                if cfg.start_date <= day <= cfg.end_date
            }
        )
        if len(calendar) < 2:
            return PortfolioBacktestResult(metrics={"trades": 0, "reason": "insufficient_bars"})

        cost_rate = (float(cfg.commission_bps) + float(cfg.slippage_bps)) / 10_000.0
        calendar_index = {day: index for index, day in enumerate(calendar)}
        # 同一天信号按分数排序，选 Top N；买入在下一个可用交易日开盘执行
        entries_by_day: Dict[date, List[_Signal]] = {}
        exits_by_day: Dict[date, List[str]] = {}
        for signal in signals:
            code_bars = bars.get(signal.code, {})
            exec_day = next((day for day in sorted(code_bars) if day > signal.signal_date), None)
            if exec_day is None:
                continue
            if signal.decision == "buy" and signal.score >= cfg.min_score:
                entries_by_day.setdefault(exec_day, []).append(signal)
            elif signal.decision == "sell":
                exits_by_day.setdefault(exec_day, []).append(signal.code)

        cash = float(cfg.initial_capital)
        positions: Dict[str, _Position] = {}
        trades: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []
        peak = cash
        max_drawdown = 0.0
        daily_returns: List[float] = []
        previous_equity = cash
        previous_day: Optional[date] = None

        for day in calendar:
            sizing_equity = (
                cash + self._market_value(positions, bars, previous_day)
                if previous_day is not None
                else cash
            )
            # 1) 卖出信号（开盘）
            for code in exits_by_day.get(day, []):
                position = positions.get(code)
                bar = bars.get(code, {}).get(day)
                if position and bar and bar.open:
                    cash += self._close_position(position, float(bar.open), day, cost_rate, trades)
                    positions.pop(code, None)

            # 2) 到期平仓（开盘）
            for code in list(positions):
                position = positions[code]
                if day >= position.exit_date:
                    bar = bars.get(code, {}).get(day)
                    if bar and bar.open:
                        cash += self._close_position(position, float(bar.open), day, cost_rate, trades)
                        positions.pop(code, None)

            # 3) 新开仓（开盘），先排序再按剩余仓位分配
            for signal in sorted(entries_by_day.get(day, []), key=lambda item: -item.score):
                if len(positions) >= cfg.max_positions or signal.code in positions:
                    continue
                bar = bars.get(signal.code, {}).get(day)
                if not bar or not bar.open or bar.open <= 0:
                    continue
                remaining_slots = max(1, cfg.max_positions - len(positions))
                slot = min(
                    max(0.0, sizing_equity / max(1, cfg.max_positions)),
                    cash / remaining_slots,
                )
                price = float(bar.open) * (1 + cost_rate)
                shares = slot / price if price > 0 else 0.0
                if shares <= 0 or slot > cash:
                    continue
                exit_index = calendar_index[day] + max(1, cfg.holding_days)
                exit_date = calendar[exit_index] if exit_index < len(calendar) else calendar[-1]
                position = _Position(
                    code=signal.code,
                    shares=shares,
                    entry_price=price,
                    entry_date=day,
                    exit_date=exit_date,
                    entry_cost=slot,
                )
                cash -= slot
                positions[signal.code] = position

            # 4) 收盘估值
            market_value = self._market_value(positions, bars, day)
            equity = cash + market_value
            if previous_equity > 0:
                daily_returns.append((equity - previous_equity) / previous_equity)
            previous_equity = equity
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = min(max_drawdown, (equity - peak) / peak * 100)
            equity_curve.append(
                {
                    "date": day.isoformat(),
                    "equity": round(equity, 2),
                    "positions": len(positions),
                    "cash": round(cash, 2),
                }
            )
            previous_day = day

        # 收盘平掉剩余仓位
        last_day = calendar[-1]
        for code in list(positions):
            bar = bars.get(code, {}).get(last_day)
            price = float(bar.close) if bar and bar.close else positions[code].entry_price
            cash += self._close_position(positions[code], price, last_day, cost_rate, trades)
            positions.pop(code, None)

        final_equity = cash
        total_return_pct = (final_equity / cfg.initial_capital - 1) * 100 if cfg.initial_capital else 0.0
        metrics = self._build_metrics(
            cfg=cfg,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            daily_returns=daily_returns,
            max_drawdown=max_drawdown,
            trades=trades,
            equity_curve=equity_curve,
            calendar=calendar,
        )
        return PortfolioBacktestResult(metrics=metrics, equity_curve=equity_curve, trades=trades)

    @staticmethod
    def _market_value(
        positions: Dict[str, _Position], bars: Dict[str, Dict[date, Any]], day: date
    ) -> float:
        total = 0.0
        for code, position in positions.items():
            bar = bars.get(code, {}).get(day)
            price = float(bar.close) if bar and bar.close else position.entry_price
            total += position.shares * price
        return total

    @staticmethod
    def _close_position(
        position: _Position,
        price: float,
        day: date,
        cost_rate: float,
        trades: List[Dict[str, Any]],
    ) -> float:
        net_price = price * (1 - cost_rate)
        proceeds = position.shares * net_price
        return_pct = (net_price - position.entry_price) / position.entry_price * 100
        trades.append(
            {
                "code": position.code,
                "entry_date": position.entry_date.isoformat(),
                "exit_date": day.isoformat(),
                "entry_price": round(position.entry_price, 4),
                "exit_price": round(net_price, 4),
                "return_pct": round(return_pct, 2),
            }
        )
        return proceeds

    def _build_metrics(
        self,
        *,
        cfg: PortfolioBacktestConfig,
        final_equity: float,
        total_return_pct: float,
        daily_returns: List[float],
        max_drawdown: float,
        trades: List[Dict[str, Any]],
        equity_curve: List[Dict[str, Any]],
        calendar: List[date],
    ) -> Dict[str, Any]:
        win_trades = [trade for trade in trades if trade["return_pct"] > 0]
        avg_trade = sum(trade["return_pct"] for trade in trades) / len(trades) if trades else None
        mean = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
        variance = (
            sum((value - mean) ** 2 for value in daily_returns) / len(daily_returns)
            if len(daily_returns) > 1
            else 0.0
        )
        std = math.sqrt(variance)
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else None
        downside = [value for value in daily_returns if value < 0]
        downside_std = (
            math.sqrt(sum(value * value for value in downside) / len(downside))
            if downside
            else 0.0
        )
        sortino = (mean / downside_std * math.sqrt(252)) if downside_std > 0 else None
        days = max(1, (calendar[-1] - calendar[0]).days)
        cagr = ((final_equity / cfg.initial_capital) ** (365.0 / days) - 1) * 100 if cfg.initial_capital > 0 else None
        exposure = (
            sum(row["positions"] for row in equity_curve) / len(equity_curve) / max(1, cfg.max_positions)
            if equity_curve
            else 0.0
        )

        market = cfg.benchmark_market
        if not market:
            markets = [get_market_for_stock(code) for code in {trade["code"] for trade in trades}]
            market = next((item for item in markets if item), None)
        benchmark_return = (
            self.benchmark.return_pct(market, calendar[0], calendar[-1]) if market else None
        )
        excess = (
            round(total_return_pct - benchmark_return, 2)
            if benchmark_return is not None
            else None
        )

        return {
            "start_date": cfg.start_date.isoformat(),
            "end_date": cfg.end_date.isoformat(),
            "initial_capital": cfg.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2) if cagr is not None else None,
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe": round(sharpe, 2) if sharpe is not None else None,
            "sortino": round(sortino, 2) if sortino is not None else None,
            "trades": len(trades),
            "win_rate_pct": round(len(win_trades) / len(trades) * 100, 2) if trades else None,
            "avg_trade_return_pct": round(avg_trade, 2) if avg_trade is not None else None,
            "avg_exposure_pct": round(exposure * 100, 2),
            "max_positions": cfg.max_positions,
            "holding_days": cfg.holding_days,
            "benchmark_return_pct": benchmark_return,
            "excess_return_pct": excess,
        }

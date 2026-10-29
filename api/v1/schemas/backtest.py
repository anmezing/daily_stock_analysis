# -*- coding: utf-8 -*-
"""Backtest API schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BacktestRunRequest(BaseModel):
    code: Optional[str] = Field(None, description="仅回测指定股票")
    force: bool = Field(False, description="强制重新计算")
    eval_window_days: Optional[int] = Field(None, ge=1, le=120, description="评估窗口（交易日数）")
    min_age_days: Optional[int] = Field(None, ge=0, le=365, description="分析记录最小天龄（0=不限）")
    limit: int = Field(200, ge=1, le=2000, description="最多处理的分析记录数")


class BacktestRunResponse(BaseModel):
    processed: int = Field(..., description="候选记录数")
    saved: int = Field(..., description="写入回测结果数")
    completed: int = Field(..., description="完成回测数")
    insufficient: int = Field(..., description="数据不足数")
    errors: int = Field(..., description="错误数")
    data_quality: int = Field(0, description="疑似拆股/并股而跳过的窗口数")
    windows: List[int] = Field(default_factory=list, description="本次评估的窗口（交易日）")
    per_window: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class PortfolioBacktestRequest(BaseModel):
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD")
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD")
    initial_capital: float = Field(100000.0, gt=0)
    max_positions: int = Field(5, ge=1, le=50)
    holding_days: int = Field(10, ge=1, le=120)
    min_score: int = Field(0, ge=0, le=100)
    benchmark_market: Optional[str] = Field(None, description="cn/hk/us，默认按持仓推断")


class PortfolioBacktestResponse(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    trades: List[Dict[str, Any]] = Field(default_factory=list)


class BacktestResultItem(BaseModel):
    analysis_history_id: int
    code: str
    analysis_date: Optional[str] = None
    eval_window_days: int
    engine_version: str
    eval_status: str
    evaluated_at: Optional[str] = None
    operation_advice: Optional[str] = None
    position_recommendation: Optional[str] = None
    raw_operation_advice: Optional[str] = Field(None, description="校准前的原始建议，用于归因对比")
    raw_direction_expected: Optional[str] = None
    raw_direction_correct: Optional[bool] = None
    calibration_arm_operation_advice: Optional[str] = Field(
        None, description="校准器给出的建议（暂停时也记录，作为影子对照）"
    )
    calibration_arm_direction_expected: Optional[str] = None
    calibration_arm_direction_correct: Optional[bool] = None
    start_price: Optional[float] = None
    end_close: Optional[float] = None
    max_high: Optional[float] = None
    min_low: Optional[float] = None
    stock_return_pct: Optional[float] = None
    direction_expected: Optional[str] = None
    direction_correct: Optional[bool] = None
    outcome: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    hit_stop_loss: Optional[bool] = None
    hit_take_profit: Optional[bool] = None
    first_hit: Optional[str] = None
    first_hit_date: Optional[str] = None
    first_hit_trading_days: Optional[int] = None
    simulated_entry_price: Optional[float] = None
    simulated_exit_price: Optional[float] = None
    simulated_exit_reason: Optional[str] = None
    simulated_return_pct: Optional[float] = None
    simulated_return_pct_gross: Optional[float] = None
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    days_to_mfe: Optional[int] = None
    days_to_mae: Optional[int] = None
    benchmark_return_pct: Optional[float] = None
    excess_return_pct: Optional[float] = None
    strategy_id: Optional[str] = None


class BacktestResultsResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: List[BacktestResultItem] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    scope: str
    code: Optional[str] = None
    eval_window_days: int
    engine_version: str
    computed_at: Optional[str] = None

    total_evaluations: int
    completed_count: int
    insufficient_count: int
    long_count: int
    cash_count: int
    win_count: int
    loss_count: int
    neutral_count: int

    direction_accuracy_pct: Optional[float] = Field(
        None, description="方向准确率：全部已完成记录，按窗口末涨跌判定"
    )
    raw_direction_accuracy_pct: Optional[float] = Field(
        None, description="校准前原始判断的方向准确率"
    )
    calibration_lift_pct: Optional[float] = Field(
        None, description="校准提升（百分点）= 校准建议准确率 − 原始判断准确率；<=0 时自动暂停应用校准"
    )
    calibration_sample_count: int = Field(
        0, description="校准器会改变方向结论、且前后都有回测结论的样本数（暂停期间也记录影子对照）"
    )
    win_rate_pct: Optional[float] = Field(
        None, description="执行胜率：仅统计 long 仓位的模拟成交结果，不含现金信号与中性"
    )
    neutral_rate_pct: Optional[float] = Field(
        None, description="中性占比：分母为 long 仓位模拟成交记录"
    )
    avg_stock_return_pct: Optional[float] = Field(
        None, description="平均股票收益率：全部已完成记录"
    )
    avg_simulated_return_pct: Optional[float] = Field(
        None, description="平均模拟执行收益率：仅统计 long 仓位，含止盈止损与缺口成交"
    )
    avg_simulated_return_pct_gross: Optional[float] = Field(
        None, description="扣佣金/滑点前的平均模拟收益率"
    )
    avg_mfe_pct: Optional[float] = Field(None, description="窗口内平均最大浮盈")
    avg_mae_pct: Optional[float] = Field(None, description="窗口内平均最大浮亏（负数）")
    avg_benchmark_return_pct: Optional[float] = Field(
        None, description="同窗口基准（SPY/2800.HK/沪深300）收益"
    )
    avg_excess_return_pct: Optional[float] = Field(
        None, description="平均超额收益 = 个股收益 − 基准收益"
    )
    excess_win_rate_pct: Optional[float] = Field(
        None, description="跑赢基准的 long 交易占比"
    )

    stop_loss_trigger_rate: Optional[float] = None
    take_profit_trigger_rate: Optional[float] = None
    ambiguous_rate: Optional[float] = None
    avg_days_to_first_hit: Optional[float] = None

    advice_breakdown: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    learning_model: Dict[str, Any] = Field(default_factory=dict)

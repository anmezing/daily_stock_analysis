/**
 * Backtest API type definitions
 * Mirrors api/v1/schemas/backtest.py
 */

// ============ Request / Response ============

export interface BacktestRunRequest {
  code?: string;
  force?: boolean;
  evalWindowDays?: number;
  minAgeDays?: number;
  limit?: number;
}

export interface BacktestRunResponse {
  processed: number;
  saved: number;
  completed: number;
  insufficient: number;
  errors: number;
  dataQuality?: number;
}

// ============ Result Item ============

export interface BacktestResultItem {
  analysisHistoryId: number;
  code: string;
  analysisDate?: string;
  evalWindowDays: number;
  engineVersion: string;
  evalStatus: string;
  evaluatedAt?: string;
  operationAdvice?: string;
  positionRecommendation?: string;
  startPrice?: number;
  endClose?: number;
  maxHigh?: number;
  minLow?: number;
  stockReturnPct?: number;
  directionExpected?: string;
  directionCorrect?: boolean;
  outcome?: string;
  stopLoss?: number;
  takeProfit?: number;
  hitStopLoss?: boolean;
  hitTakeProfit?: boolean;
  firstHit?: string;
  firstHitDate?: string;
  firstHitTradingDays?: number;
  simulatedEntryPrice?: number;
  simulatedExitPrice?: number;
  simulatedExitReason?: string;
  simulatedReturnPct?: number;
  simulatedReturnPctGross?: number;
  mfePct?: number;
  maePct?: number;
  daysToMfe?: number;
  daysToMae?: number;
  benchmarkReturnPct?: number;
  excessReturnPct?: number;
  rawOperationAdvice?: string;
  rawDirectionExpected?: string;
  rawDirectionCorrect?: boolean;
  calibrationArmOperationAdvice?: string;
  calibrationArmDirectionExpected?: string;
  calibrationArmDirectionCorrect?: boolean;
  strategyId?: string;
}

export interface BacktestResultsResponse {
  total: number;
  page: number;
  limit: number;
  items: BacktestResultItem[];
}

// ============ Performance Metrics ============

export interface PerformanceMetrics {  scope: string;
  code?: string;
  evalWindowDays: number;
  engineVersion: string;
  computedAt?: string;

  totalEvaluations: number;
  completedCount: number;
  insufficientCount: number;
  longCount: number;
  cashCount: number;
  winCount: number;
  lossCount: number;
  neutralCount: number;

  directionAccuracyPct?: number;
  rawDirectionAccuracyPct?: number;
  calibrationLiftPct?: number;
  calibrationSampleCount?: number;
  winRatePct?: number;
  neutralRatePct?: number;
  avgStockReturnPct?: number;
  avgSimulatedReturnPct?: number;
  avgSimulatedReturnPctGross?: number;
  avgMfePct?: number;
  avgMaePct?: number;
  avgBenchmarkReturnPct?: number;
  avgExcessReturnPct?: number;
  excessWinRatePct?: number;

  stopLossTriggerRate?: number;
  takeProfitTriggerRate?: number;
  ambiguousRate?: number;
  avgDaysToFirstHit?: number;

  adviceBreakdown: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
}

// ============ Portfolio Backtest ============

export interface PortfolioBacktestRequest {
  startDate: string;
  endDate: string;
  initialCapital?: number;
  maxPositions?: number;
  holdingDays?: number;
  minScore?: number;
  benchmarkMarket?: string;
}

export interface PortfolioBacktestPoint {
  date: string;
  equity: number;
  positions: number;
  cash: number;
}

export interface PortfolioTrade {
  code: string;
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  returnPct: number;
}

export interface PortfolioBacktestMetrics {
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalEquity: number;
  totalReturnPct: number;
  cagrPct?: number | null;
  maxDrawdownPct: number;
  sharpe?: number | null;
  sortino?: number | null;
  trades: number;
  winRatePct?: number | null;
  avgTradeReturnPct?: number | null;
  avgExposurePct: number;
  benchmarkReturnPct?: number | null;
  excessReturnPct?: number | null;
}

export interface PortfolioBacktestResponse {
  metrics: PortfolioBacktestMetrics;
  equityCurve: PortfolioBacktestPoint[];
  trades: PortfolioTrade[];
}

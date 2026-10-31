export interface ScreenerPlan {
  buyTrigger?: string | null;
  entryLow?: number | null;
  entryHigh?: number | null;
  stopLoss?: number | null;
  target1?: number | null;
  target2?: number | null;
  riskReward?: number | null;
  timeStop?: string | null;
}

export interface ScreenerPick {
  symbol: string;
  name?: string;
  sectorEtf?: string;
  sectorName?: string;
  state?: string;
  stateLabel?: string;
  score?: number;
  return15d?: number | null;
  return30d?: number | null;
  rs15d?: number | null;
  rs30d?: number | null;
  distMa20Pct?: number | null;
  volumeRatio520?: number | null;
  earningsDate?: string | null;
  daysToEarnings?: number | null;
  qualityOk?: boolean | null;
  qualityNotes?: string[];
  learningAdjustment?: number | null;
  learningNote?: string | null;
  plan?: ScreenerPlan;
}

export interface ScreenerSector {
  symbol: string;
  name?: string;
  return5d?: number | null;
  return15d?: number | null;
  return20d?: number | null;
}

export interface ScreenerFilters {
  earningsSkipped?: number;
  qualitySkipped?: number;
  qualityUnknown?: number;
  correlationSkipped?: number;
}

export interface ScreenerValidation {
  status?: string;
  samples?: number;
  hitRate10d?: number;
  avgExcess10d?: number;
  note?: string;
}

export interface ScreenerLearning {
  samples?: number;
  hitRate10d?: number | null;
  avgExcess10d?: number | null;
  adjustmentRemoved?: number;
  minScoreDelta?: number;
  updatedAt?: string | null;
  disabledStates?: string[];
}

export interface ScreenerResult {
  status: string;
  market?: string;
  asOf?: string | null;
  riskOn?: boolean;
  minScore?: number;
  effectiveMinScore?: number;
  summary?: string;
  sectors?: ScreenerSector[];
  picks?: ScreenerPick[];
  filters?: ScreenerFilters;
  validation?: ScreenerValidation | null;
  learning?: ScreenerLearning;
  reason?: string;
}

export interface ScreenerLatest {
  screenDate: string;
  generatedAt: string;
  result: ScreenerResult;
  markdown?: string;
}

export interface ScreenerHistoryItem {
  date: string;
  filename: string;
}

export interface ScreenerReport {
  date: string;
  filename: string;
  markdown: string;
}

export interface ScreenerAlert {
  symbol: string;
  name?: string;
  kind: string;
  message: string;
  date: string;
}

export interface ScreenerAlerts {
  status?: string;
  generatedAt?: string;
  alerts?: ScreenerAlert[];
}

export interface ScreenerPoolItem {
  symbol: string;
  name?: string;
  sectorName?: string;
  firstSeen?: string;
  lastSeen?: string;
  daysSeen?: number;
  consecutiveDays?: number;
  missingDays?: number;
  todaySelected?: boolean;
  stage?: string;
  lastScore?: number;
  fundamentalScore?: number | null;
  fundamentalTrend?: string;
  riskLevel?: string;
  riskNote?: string;
  plan?: ScreenerPlan;
}

export interface ScreenerPool {
  status?: string;
  updatedAt?: string;
  tradeDate?: string;
  items?: Record<string, ScreenerPoolItem>;
  summary?: Record<string, number>;
}

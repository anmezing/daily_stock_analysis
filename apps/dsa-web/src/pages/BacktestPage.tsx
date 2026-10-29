import type React from 'react';
import { useState, useEffect, useCallback } from 'react';
import { Area, AreaChart, ResponsiveContainer, Tooltip, YAxis } from 'recharts';
import { backtestApi } from '../api/backtest';
import type { ParsedApiError } from '../api/error';
import { getParsedApiError } from '../api/error';
import { ApiErrorAlert, Card, Badge, Pagination } from '../components/common';
import type {
  BacktestResultItem,
  BacktestRunResponse,
  PerformanceMetrics,
  PortfolioBacktestResponse,
} from '../types/backtest';

// ============ Helpers ============

function pct(value?: number | null): string {
  if (value == null) return '--';
  return `${value.toFixed(1)}%`;
}

function outcomeBadge(outcome?: string) {
  if (!outcome) return <Badge variant="default">--</Badge>;
  switch (outcome) {
    case 'win':
      return <Badge variant="success" glow>WIN</Badge>;
    case 'loss':
      return <Badge variant="danger" glow>LOSS</Badge>;
    case 'neutral':
      return <Badge variant="warning">NEUTRAL</Badge>;
    default:
      return <Badge variant="default">{outcome}</Badge>;
  }
}

function statusBadge(status: string) {
  switch (status) {
    case 'completed':
      return <Badge variant="success">completed</Badge>;
    case 'insufficient':
      return <Badge variant="warning">insufficient</Badge>;
    case 'data_quality':
      return <Badge variant="warning">data quality</Badge>;
    case 'error':
      return <Badge variant="danger">error</Badge>;
    default:
      return <Badge variant="default">{status}</Badge>;
  }
}

function boolIcon(value?: boolean | null) {
  if (value === true) return <span className="text-emerald-400">&#10003;</span>;
  if (value === false) return <span className="text-red-400">&#10007;</span>;
  return <span className="text-muted-text">--</span>;
}

// ============ Metric Row ============

const MetricRow: React.FC<{ label: string; value: string; accent?: boolean }> = ({ label, value, accent }) => (
  <div className="flex items-center justify-between py-1.5 border-b border-white/5 last:border-0">
    <span className="text-xs text-secondary-text">{label}</span>
    <span className={`text-sm font-mono font-semibold ${accent ? 'text-cyan' : 'text-white'}`}>{value}</span>
  </div>
);

// ============ Performance Card ============

const PerformanceCard: React.FC<{ metrics: PerformanceMetrics; title: string }> = ({ metrics, title }) => (
  <Card variant="gradient" padding="md" className="animate-fade-in">
    <div className="mb-3">
      <span className="label-uppercase">{title}</span>
    </div>
    <MetricRow label="Direction Accuracy" value={pct(metrics.directionAccuracyPct)} accent />
    {metrics.rawDirectionAccuracyPct != null && (
      <MetricRow label="Raw Accuracy" value={pct(metrics.rawDirectionAccuracyPct)} />
    )}
    {metrics.calibrationLiftPct != null && (
      <MetricRow
        label={`Calibration Lift${metrics.calibrationSampleCount ? ` (n=${metrics.calibrationSampleCount})` : ''}`}
        value={pct(metrics.calibrationLiftPct)}
        accent
      />
    )}
    <MetricRow label="Win Rate" value={pct(metrics.winRatePct)} accent />
    <MetricRow label="Avg Sim. Return" value={pct(metrics.avgSimulatedReturnPct)} />
    <MetricRow label="Avg Stock Return" value={pct(metrics.avgStockReturnPct)} />
    {metrics.avgBenchmarkReturnPct != null && (
      <MetricRow label="Benchmark" value={pct(metrics.avgBenchmarkReturnPct)} />
    )}
    {metrics.avgExcessReturnPct != null && (
      <MetricRow label="Avg Excess" value={pct(metrics.avgExcessReturnPct)} accent />
    )}
    {metrics.avgMfePct != null && (
      <MetricRow label="Avg MFE / MAE" value={`${pct(metrics.avgMfePct)} / ${pct(metrics.avgMaePct)}`} />
    )}
    <MetricRow label="SL Trigger Rate" value={pct(metrics.stopLossTriggerRate)} />
    <MetricRow label="TP Trigger Rate" value={pct(metrics.takeProfitTriggerRate)} />
    <MetricRow label="Avg Days to Hit" value={metrics.avgDaysToFirstHit != null ? metrics.avgDaysToFirstHit.toFixed(1) : '--'} />
    <div className="mt-3 pt-2 border-t border-white/5 flex items-center justify-between">
      <span className="text-xs text-muted-text">Evaluations</span>
      <span className="text-xs text-secondary-text font-mono">
        {Number(metrics.completedCount)} / {Number(metrics.totalEvaluations)}
      </span>
    </div>
    <div className="flex items-center justify-between">
      <span className="text-xs text-muted-text">W / L / N</span>
      <span className="text-xs font-mono">
        <span className="text-emerald-400">{metrics.winCount}</span>
        {' / '}
        <span className="text-red-400">{metrics.lossCount}</span>
        {' / '}
        <span className="text-amber-400">{metrics.neutralCount}</span>
      </span>
    </div>
  </Card>
);

// ============ Run Summary ============

const RunSummary: React.FC<{ data: BacktestRunResponse }> = ({ data }) => (
  <div className="flex items-center gap-4 px-3 py-2 rounded-lg bg-elevated border border-white/5 text-xs font-mono animate-fade-in">
    <span className="text-secondary-text">Processed: <span className="text-white">{data.processed}</span></span>
    <span className="text-secondary-text">Saved: <span className="text-cyan">{data.saved}</span></span>
    <span className="text-secondary-text">Completed: <span className="text-emerald-400">{data.completed}</span></span>
    <span className="text-secondary-text">Insufficient: <span className="text-amber-400">{data.insufficient}</span></span>
    {Boolean(data.dataQuality) && (
      <span className="text-secondary-text">Data Quality: <span className="text-amber-400">{data.dataQuality}</span></span>
    )}
    {data.errors > 0 && (
      <span className="text-secondary-text">Errors: <span className="text-red-400">{data.errors}</span></span>
    )}
  </div>
);

// ============ Portfolio Backtest ============

function isoDaysAgo(days: number): string {
  const value = new Date();
  value.setDate(value.getDate() - days);
  return value.toISOString().slice(0, 10);
}

const PortfolioCard: React.FC = () => {
  const [startDate, setStartDate] = useState(() => isoDaysAgo(180));
  const [endDate, setEndDate] = useState(() => isoDaysAgo(0));
  const [capital, setCapital] = useState('100000');
  const [maxPositions, setMaxPositions] = useState('5');
  const [holdingDays, setHoldingDays] = useState('10');
  const [minScore, setMinScore] = useState('0');
  const [market, setMarket] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<PortfolioBacktestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const response = await backtestApi.runPortfolio({
        startDate,
        endDate,
        initialCapital: Number(capital) || 100000,
        maxPositions: Number(maxPositions) || 5,
        holdingDays: Number(holdingDays) || 10,
        minScore: Number(minScore) || 0,
        benchmarkMarket: market || undefined,
      });
      setResult(response);
    } catch (err) {
      setError(getParsedApiError(err).message);
    } finally {
      setIsRunning(false);
    }
  };

  const metrics = result?.metrics;
  const chartData = (result?.equityCurve || []).map((point) => ({
    date: point.date,
    equity: point.equity,
  }));

  return (
    <Card variant="gradient" padding="md" className="animate-fade-in">
      <div className="mb-2 flex items-center justify-between">
        <span className="label-uppercase">Portfolio</span>
        <span className="text-[10px] text-muted-text">Top-N by score</span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <label className="text-[10px] text-muted-text">
          Start
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="input-terminal w-full text-xs py-1.5" />
        </label>
        <label className="text-[10px] text-muted-text">
          End
          <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="input-terminal w-full text-xs py-1.5" />
        </label>
        <label className="text-[10px] text-muted-text">
          Capital
          <input type="number" value={capital} onChange={(e) => setCapital(e.target.value)} className="input-terminal w-full text-xs py-1.5" />
        </label>
        <label className="text-[10px] text-muted-text">
          Max Pos.
          <input type="number" min={1} max={50} value={maxPositions} onChange={(e) => setMaxPositions(e.target.value)} className="input-terminal w-full text-xs py-1.5" />
        </label>
        <label className="text-[10px] text-muted-text">
          Hold Days
          <input type="number" min={1} max={120} value={holdingDays} onChange={(e) => setHoldingDays(e.target.value)} className="input-terminal w-full text-xs py-1.5" />
        </label>
        <label className="text-[10px] text-muted-text">
          Min Score
          <input type="number" min={0} max={100} value={minScore} onChange={(e) => setMinScore(e.target.value)} className="input-terminal w-full text-xs py-1.5" />
        </label>
      </div>

      <div className="mt-2 flex items-center gap-2">
        <select value={market} onChange={(e) => setMarket(e.target.value)} className="input-terminal text-xs py-1.5 flex-1">
          <option value="">Auto market</option>
          <option value="us">US</option>
          <option value="hk">HK</option>
          <option value="cn">CN</option>
        </select>
        <button type="button" onClick={handleRun} disabled={isRunning} className="btn-secondary text-xs whitespace-nowrap">
          {isRunning ? 'Running...' : 'Run'}
        </button>
      </div>

      {error && <p className="mt-2 text-[11px] text-red-400">{error}</p>}

      {metrics && (
        <div className="mt-3">
          <MetricRow label="Total Return" value={pct(metrics.totalReturnPct)} accent />
          <MetricRow label="CAGR" value={pct(metrics.cagrPct)} />
          <MetricRow label="Max Drawdown" value={pct(metrics.maxDrawdownPct)} />
          <MetricRow label="Sharpe" value={metrics.sharpe != null ? metrics.sharpe.toFixed(2) : '--'} />
          <MetricRow label="Win Rate" value={pct(metrics.winRatePct)} />
          <MetricRow label="Trades" value={String(metrics.trades)} />
          <MetricRow label="Exposure" value={pct(metrics.avgExposurePct)} />
          <MetricRow label="Benchmark" value={pct(metrics.benchmarkReturnPct)} />
          <MetricRow label="Excess" value={pct(metrics.excessReturnPct)} accent />

          {chartData.length > 1 && (
            <div className="mt-3 h-28">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id="portfolioEquity" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <YAxis hide domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', fontSize: 11 }}
                    labelStyle={{ color: '#94a3b8' }}
                  />
                  <Area type="monotone" dataKey="equity" stroke="#22d3ee" strokeWidth={1.5} fill="url(#portfolioEquity)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </Card>
  );
};

// ============ Main Page ============

const BacktestPage: React.FC = () => {
  // Input state
  const [codeFilter, setCodeFilter] = useState('');
  const [evalDays, setEvalDays] = useState('');
  const [forceRerun, setForceRerun] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [runResult, setRunResult] = useState<BacktestRunResponse | null>(null);
  const [runError, setRunError] = useState<ParsedApiError | null>(null);
  const [pageError, setPageError] = useState<ParsedApiError | null>(null);

  // Results state
  const [results, setResults] = useState<BacktestResultItem[]>([]);
  const [totalResults, setTotalResults] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const pageSize = 20;

  // Performance state
  const [overallPerf, setOverallPerf] = useState<PerformanceMetrics | null>(null);
  const [stockPerf, setStockPerf] = useState<PerformanceMetrics | null>(null);
  const [isLoadingPerf, setIsLoadingPerf] = useState(false);

  // Fetch results
  const fetchResults = useCallback(async (page = 1, code?: string, windowDays?: number) => {
    setIsLoadingResults(true);
    try {
      const response = await backtestApi.getResults({ code: code || undefined, evalWindowDays: windowDays, page, limit: pageSize });
      setResults(response.items);
      setTotalResults(response.total);
      setCurrentPage(response.page);
      setPageError(null);
    } catch (err) {
      console.error('Failed to fetch backtest results:', err);
      setPageError(getParsedApiError(err));
    } finally {
      setIsLoadingResults(false);
    }
  }, []);

  // Fetch performance
  const fetchPerformance = useCallback(async (code?: string, windowDays?: number) => {
    setIsLoadingPerf(true);
    try {
      const overall = await backtestApi.getOverallPerformance(windowDays);
      setOverallPerf(overall);

      if (code) {
        const stock = await backtestApi.getStockPerformance(code, windowDays);
        setStockPerf(stock);
      } else {
        setStockPerf(null);
      }
      setPageError(null);
    } catch (err) {
      console.error('Failed to fetch performance:', err);
      setPageError(getParsedApiError(err));
    } finally {
      setIsLoadingPerf(false);
    }
  }, []);

  // Initial load — fetch performance first, then filter results by its window
  useEffect(() => {
    const init = async () => {
      // Get latest performance (unfiltered returns most recent summary)
      const overall = await backtestApi.getOverallPerformance();
      setOverallPerf(overall);
      // Use the summary's eval_window_days to filter results consistently
      const windowDays = overall?.evalWindowDays;
      if (windowDays && !evalDays) {
        setEvalDays(String(windowDays));
      }
      fetchResults(1, undefined, windowDays);
    };
    init();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Run backtest
  const handleRun = async () => {
    setIsRunning(true);
    setRunResult(null);
    setRunError(null);
    try {
      const code = codeFilter.trim() || undefined;
      const evalWindowDays = evalDays ? parseInt(evalDays, 10) : undefined;
      const response = await backtestApi.run({
        code,
        force: forceRerun || undefined,
        minAgeDays: forceRerun ? 0 : undefined,
        evalWindowDays,
      });
      setRunResult(response);
      // Refresh data with same eval_window_days
      fetchResults(1, codeFilter.trim() || undefined, evalWindowDays);
      fetchPerformance(codeFilter.trim() || undefined, evalWindowDays);
    } catch (err) {
      setRunError(getParsedApiError(err));
    } finally {
      setIsRunning(false);
    }
  };

  // Filter by code
  const handleFilter = () => {
    const code = codeFilter.trim() || undefined;
    const windowDays = evalDays ? parseInt(evalDays, 10) : undefined;
    setCurrentPage(1);
    fetchResults(1, code, windowDays);
    fetchPerformance(code, windowDays);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleFilter();
    }
  };

  // Pagination
  const totalPages = Math.ceil(totalResults / pageSize);
  const handlePageChange = (page: number) => {
    const windowDays = evalDays ? parseInt(evalDays, 10) : undefined;
    fetchResults(page, codeFilter.trim() || undefined, windowDays);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="flex-shrink-0 px-4 py-3 border-b border-white/5">
        <div className="flex items-center gap-2 max-w-4xl">
          <div className="flex-1 relative">
            <input
              type="text"
              value={codeFilter}
              onChange={(e) => setCodeFilter(e.target.value.toUpperCase())}
              onKeyDown={handleKeyDown}
              placeholder="Filter by stock code (leave empty for all)"
              disabled={isRunning}
              className="input-terminal w-full"
            />
          </div>
          <button
            type="button"
            onClick={handleFilter}
            disabled={isLoadingResults}
            className="btn-secondary flex items-center gap-1.5 whitespace-nowrap"
          >
            Filter
          </button>
          <div className="flex items-center gap-1 whitespace-nowrap">
            <span className="text-xs text-muted-text">Window</span>
            <input
              type="number"
              min={1}
              max={120}
              value={evalDays}
              onChange={(e) => setEvalDays(e.target.value)}
              placeholder="10"
              disabled={isRunning}
              className="input-terminal w-14 text-center text-xs py-2"
            />
          </div>
          <button
            type="button"
            onClick={() => setForceRerun(!forceRerun)}
            disabled={isRunning}
            className={`
              flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium
              transition-all duration-200 whitespace-nowrap border cursor-pointer
              ${forceRerun
                ? 'border-cyan/40 bg-cyan/10 text-cyan shadow-[0_0_8px_rgba(0,212,255,0.15)]'
                : 'border-white/10 bg-transparent text-muted-text hover:border-white/20 hover:text-secondary-text'
              }
              disabled:opacity-50 disabled:cursor-not-allowed
            `}
          >
            <span className={`
              inline-block w-1.5 h-1.5 rounded-full transition-colors duration-200
              ${forceRerun ? 'bg-cyan shadow-[0_0_4px_rgba(0,212,255,0.6)]' : 'bg-white/20'}
            `} />
            Force
          </button>
          <button
            type="button"
            onClick={handleRun}
            disabled={isRunning}
            className="btn-primary flex items-center gap-1.5 whitespace-nowrap"
          >
            {isRunning ? (
              <>
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Running...
              </>
            ) : (
              'Run Backtest'
            )}
          </button>
        </div>
        {runResult && (
          <div className="mt-2 max-w-4xl">
            <RunSummary data={runResult} />
          </div>
        )}
        {runError && (
          <ApiErrorAlert error={runError} className="mt-2 max-w-4xl" />
        )}
      </header>

      {/* Main content */}
      <main className="flex-1 flex overflow-hidden p-3 gap-3">
        {/* Left sidebar - Performance */}
        <div className="flex flex-col gap-3 w-64 flex-shrink-0 overflow-y-auto">
          {isLoadingPerf ? (
            <div className="flex items-center justify-center py-8">
              <div className="w-8 h-8 border-2 border-cyan/20 border-t-cyan rounded-full animate-spin" />
            </div>
          ) : overallPerf ? (
            <PerformanceCard metrics={overallPerf} title="Overall Performance" />
          ) : (
            <Card padding="md">
              <p className="text-xs text-muted-text text-center py-4">
                No backtest data yet. Run a backtest to see performance metrics.
              </p>
            </Card>
          )}

          {stockPerf && (
            <PerformanceCard metrics={stockPerf} title={`${stockPerf.code || codeFilter}`} />
          )}

          <PortfolioCard />
        </div>

        {/* Right content - Results table */}
        <section className="flex-1 overflow-y-auto">
          {pageError ? (
            <ApiErrorAlert error={pageError} className="mb-3" />
          ) : null}
          {isLoadingResults ? (
            <div className="flex flex-col items-center justify-center h-64">
              <div className="w-10 h-10 border-3 border-cyan/20 border-t-cyan rounded-full animate-spin" />
              <p className="mt-3 text-secondary-text text-sm">Loading results...</p>
            </div>
          ) : results.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <div className="w-12 h-12 mb-3 rounded-xl bg-elevated flex items-center justify-center">
                <svg className="w-6 h-6 text-muted-text" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-white mb-1.5">No Results</h3>
              <p className="text-xs text-muted-text max-w-xs">
                Run a backtest to evaluate historical analysis accuracy
              </p>
            </div>
          ) : (
            <div className="animate-fade-in">
              <div className="overflow-x-auto rounded-xl border border-white/5">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-elevated text-left">
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider">Code</th>
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider">Date</th>
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider">Advice</th>
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider">Dir.</th>
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider">Outcome</th>
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider text-right">Return%</th>
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider text-center">SL</th>
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider text-center">TP</th>
                      <th className="px-3 py-2.5 text-xs font-medium text-secondary-text uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((row) => (
                      <tr
                        key={row.analysisHistoryId}
                        className="border-t border-white/5 hover:bg-hover transition-colors"
                      >
                        <td className="px-3 py-2 font-mono text-cyan text-xs">{row.code}</td>
                        <td className="px-3 py-2 text-xs text-secondary-text">{row.analysisDate || '--'}</td>
                        <td className="px-3 py-2 text-xs text-white truncate max-w-[140px]" title={row.operationAdvice || ''}>
                          {row.operationAdvice || '--'}
                        </td>
                        <td className="px-3 py-2 text-xs">
                          <span className="flex items-center gap-1">
                            {boolIcon(row.directionCorrect)}
                            <span className="text-muted-text">{row.directionExpected || ''}</span>
                          </span>
                        </td>
                        <td className="px-3 py-2">{outcomeBadge(row.outcome)}</td>
                        <td className="px-3 py-2 text-xs font-mono text-right">
                          <span className={
                            row.simulatedReturnPct != null
                              ? row.simulatedReturnPct > 0 ? 'text-emerald-400' : row.simulatedReturnPct < 0 ? 'text-red-400' : 'text-secondary-text'
                              : 'text-muted-text'
                          }>
                            {pct(row.simulatedReturnPct)}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-center">{boolIcon(row.hitStopLoss)}</td>
                        <td className="px-3 py-2 text-center">{boolIcon(row.hitTakeProfit)}</td>
                        <td className="px-3 py-2">{statusBadge(row.evalStatus)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="mt-4">
                <Pagination
                  currentPage={currentPage}
                  totalPages={totalPages}
                  onPageChange={handlePageChange}
                />
              </div>

              <p className="text-xs text-muted-text text-center mt-2">
                {totalResults} result{totalResults !== 1 ? 's' : ''} total
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default BacktestPage;

import type React from 'react';
import { useCallback, useEffect, useState } from 'react';
import { screenerApi } from '../api/screener';
import type { ParsedApiError } from '../api/error';
import { getParsedApiError } from '../api/error';
import {
  ApiErrorAlert,
  AppPage,
  Badge,
  Card,
  EmptyState,
  Loading,
  PageHeader,
} from '../components/common';
import type {
  ScreenerAlert,
  ScreenerLatest,
  ScreenerPick,
  ScreenerPool,
} from '../types/screener';

const pct = (value?: number | null): string => {
  if (value == null) return '--';
  return `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
};

const stateVariant = (state?: string): 'success' | 'info' | 'warning' | 'danger' | 'default' => {
  switch (state) {
    case 'pullback':
      return 'success';
    case 'breakout':
      return 'info';
    case 'watch':
      return 'warning';
    case 'extended':
    case 'weak':
      return 'danger';
    default:
      return 'default';
  }
};

const PlanCard: React.FC<{ pick: ScreenerPick }> = ({ pick }) => {
  const plan = pick.plan;
  if (!plan?.buyTrigger) return null;
  return (
    <Card variant="bordered" padding="sm" className="animate-fade-in">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-base font-semibold text-white">{pick.symbol}</span>
        <span className="text-sm text-secondary-text">{pick.name}</span>
        <Badge variant={stateVariant(pick.state)}>{pick.stateLabel || pick.state}</Badge>
        <span className="ml-auto font-mono text-sm text-cyan">{pick.score?.toFixed(1)}</span>
      </div>
      <p className="mt-2 text-sm text-secondary-text">{plan.buyTrigger}</p>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-text">
        <span>止损 {plan.stopLoss ?? '--'}</span>
        <span>目标1 {plan.target1 ?? '--'}</span>
        <span>目标2 {plan.target2 ?? '--'}</span>
        <span>盈亏比 {plan.riskReward ?? '--'}</span>
      </div>
      {plan.timeStop ? (
        <p className="mt-1 text-xs text-muted-text">时间止损：{plan.timeStop}</p>
      ) : null}
      {pick.learningNote ? (
        <p className="mt-1 text-xs text-cyan">学习调整：{pick.learningNote}</p>
      ) : null}
    </Card>
  );
};

const ScreenerPage: React.FC = () => {
  const [latest, setLatest] = useState<ScreenerLatest | null>(null);
  const [alerts, setAlerts] = useState<ScreenerAlert[]>([]);
  const [pool, setPool] = useState<ScreenerPool | null>(null);
  const [loading, setLoading] = useState(true);
  const [empty, setEmpty] = useState(false);
  const [error, setError] = useState<ParsedApiError | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setEmpty(false);
    try {
      const data = await screenerApi.getLatest();
      setLatest(data);
    } catch (err: unknown) {
      const parsed = getParsedApiError(err);
      if (parsed.status === 404) {
        setLatest(null);
        setEmpty(true);
      } else {
        setError(parsed);
      }
    }
    try {
      const alertData = await screenerApi.getAlerts();
      setAlerts(alertData.alerts || []);
    } catch {
      setAlerts([]);
    }
    try {
      const poolData = await screenerApi.getPool();
      setPool(poolData);
    } catch {
      setPool(null);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  const result = latest?.result;
  const picks = result?.picks || [];
  const poolItems = Object.values(pool?.items || {})
    .filter((item) => item.stage !== 'dropped')
    .sort(
      (a, b) =>
        (b.consecutiveDays || 0) - (a.consecutiveDays || 0)
        || (b.lastScore || 0) - (a.lastScore || 0),
    );

  return (
    <AppPage className="space-y-5">
      <PageHeader
        eyebrow="HOT SECTOR SCREENER"
        title="热门板块选股"
        description="从领涨板块中筛选相对强势、缩量回踩或突破的个股；选股与每日分析完全独立。"
        actions={
          <button type="button" className="btn-secondary" onClick={() => void load()} disabled={loading}>
            刷新
          </button>
        }
      />

      {loading ? <Loading label="正在读取选股结果" /> : null}
      {error ? <ApiErrorAlert error={error} /> : null}

      {empty ? (
        <EmptyState
          title="还没有选股结果"
          description="运行 python main.py --screen，或在 screener 容器启动后等待每日 08:30 ET 自动选股。"
        />
      ) : null}

      {latest && result ? (
        <>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            <Card variant="gradient" padding="sm">
              <span className="label-uppercase">选股日期</span>
              <p className="mt-1 text-lg font-semibold text-white">{latest.screenDate}</p>
              <p className="mt-1 text-xs text-muted-text">{latest.generatedAt}</p>
            </Card>
            <Card variant="gradient" padding="sm">
              <span className="label-uppercase">市场状态</span>
              <p className={`mt-1 text-lg font-semibold ${result.riskOn ? 'text-success' : 'text-warning'}`}>
                {result.riskOn ? 'Risk-on' : 'Risk-off / 震荡'}
              </p>
              <p className="mt-1 text-xs text-muted-text">最低分 {result.effectiveMinScore ?? result.minScore ?? '--'}</p>
            </Card>
            <Card variant="gradient" padding="sm">
              <span className="label-uppercase">入选数量</span>
              <p className="mt-1 text-lg font-semibold text-cyan">{picks.length}</p>
              <p className="mt-1 text-xs text-muted-text">
                过滤：财报 {result.filters?.earningsSkipped ?? 0} / 质量 {result.filters?.qualitySkipped ?? 0} / 相关性 {result.filters?.correlationSkipped ?? 0}
              </p>
            </Card>
            <Card variant="gradient" padding="sm">
              <span className="label-uppercase">自动学习</span>
              <p className="mt-1 text-lg font-semibold text-purple">
                {result.learning?.samples ? `${result.learning.samples} 样本` : '积累中'}
              </p>
              <p className="mt-1 text-xs text-muted-text">
                {result.learning?.hitRate10d != null
                  ? `10日胜率 ${result.learning.hitRate10d}% / 超额 ${pct(result.learning.avgExcess10d)}`
                  : '样本不足，暂不调整'}
              </p>
            </Card>
          </div>

          {result.summary ? (
            <Card variant="bordered" padding="sm">
              <p className="text-sm text-secondary-text">{result.summary}</p>
            </Card>
          ) : null}

          {alerts.length ? (
            <Card variant="bordered" padding="sm" className="border-danger/30 bg-danger/5">
              <span className="label-uppercase text-danger">选后告警</span>
              <div className="mt-2 space-y-1">
                {alerts.slice(0, 8).map((alert, index) => (
                  <p key={`${alert.symbol}-${alert.kind}-${index}`} className="text-sm text-secondary-text">
                    <span className="font-mono text-danger">{alert.symbol}</span>
                    {' '}{alert.name || ''}：{alert.message}
                  </p>
                ))}
              </div>
            </Card>
          ) : null}

          {poolItems.length ? (
            <Card padding="none" className="overflow-hidden">
              <div className="border-b border-white/8 px-4 py-3">
                <span className="label-uppercase">🧺 股池（持续跟踪）</span>
                <p className="mt-1 text-xs text-muted-text">
                  连续在榜、相对强势、基本面不转弱才保留；停止满足条件会自动移出。
                </p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-left text-sm">
                  <thead className="border-b border-white/8 bg-elevated/40 text-xs uppercase tracking-wide text-muted-text">
                    <tr>
                      <th className="px-3 py-2">代码</th>
                      <th className="px-3 py-2">名称</th>
                      <th className="px-3 py-2 text-right">连续</th>
                      <th className="px-3 py-2 text-right">累计</th>
                      <th className="px-3 py-2">今日入选</th>
                      <th className="px-3 py-2">基本面趋势</th>
                      <th className="px-3 py-2">阶段</th>
                      <th className="px-3 py-2">板块</th>
                    </tr>
                  </thead>
                  <tbody>
                    {poolItems.map((item) => (
                      <tr key={`pool-${item.symbol}`} className="border-b border-white/5 last:border-0">
                        <td className="px-3 py-2 font-mono text-white">{item.symbol}</td>
                        <td className="px-3 py-2 text-secondary-text">{item.name}</td>
                        <td className="px-3 py-2 text-right font-mono text-cyan">{item.consecutiveDays}</td>
                        <td className="px-3 py-2 text-right font-mono text-secondary-text">{item.daysSeen}</td>
                        <td className="px-3 py-2 text-center">
                          {item.todaySelected ? <span className="text-success">是</span> : <span className="text-muted-text">否</span>}
                        </td>
                        <td className="px-3 py-2">
                          <Badge
                            variant={item.fundamentalTrend === 'improving' ? 'success' : item.fundamentalTrend === 'worsening' ? 'danger' : 'default'}
                          >
                            {item.fundamentalTrend === 'improving'
                              ? '改善'
                              : item.fundamentalTrend === 'worsening'
                                ? '转弱'
                                : item.fundamentalTrend === 'stable'
                                  ? '持平'
                                  : '数据不足'}
                          </Badge>
                        </td>
                        <td className="px-3 py-2">
                          <Badge
                            variant={item.stage === 'trigger' ? 'success' : item.stage === 'ready' ? 'info' : 'warning'}
                          >
                            {item.stage === 'trigger'
                              ? (item.strong ? '强确认/触发' : '触发买入')
                              : item.stage === 'ready'
                                ? (item.strong ? '强确认/候选' : '买入候选')
                                : '继续观察'}
                          </Badge>
                        </td>
                        <td className="px-3 py-2 text-secondary-text">{item.sectorName}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ) : null}

          <Card padding="none" className="overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[980px] text-left text-sm">
                <thead className="border-b border-white/8 bg-elevated/40 text-xs uppercase tracking-wide text-muted-text">
                  <tr>
                    <th className="px-3 py-2">板块</th>
                    <th className="px-3 py-2">代码</th>
                    <th className="px-3 py-2">名称</th>
                    <th className="px-3 py-2 text-right">15日</th>
                    <th className="px-3 py-2 text-right">30日</th>
                    <th className="px-3 py-2 text-right">相对板块</th>
                    <th className="px-3 py-2 text-right">距MA20</th>
                    <th className="px-3 py-2 text-right">量比5/20</th>
                    <th className="px-3 py-2">状态</th>
                    <th className="px-3 py-2 text-right">评分</th>
                  </tr>
                </thead>
                <tbody>
                  {picks.map((pick) => (
                    <tr key={pick.symbol} className="border-b border-white/5 last:border-0 hover:bg-white/[0.02]">
                      <td className="px-3 py-2 text-secondary-text">{pick.sectorName}</td>
                      <td className="px-3 py-2 font-mono text-white">{pick.symbol}</td>
                      <td className="px-3 py-2 text-secondary-text">{pick.name}</td>
                      <td className={`px-3 py-2 text-right font-mono ${(pick.return15d || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                        {pct(pick.return15d)}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${(pick.return30d || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                        {pct(pick.return30d)}
                      </td>
                      <td className={`px-3 py-2 text-right font-mono ${(pick.rs15d || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                        {pct(pick.rs15d)}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-secondary-text">{pct(pick.distMa20Pct)}</td>
                      <td className="px-3 py-2 text-right font-mono text-secondary-text">
                        {pick.volumeRatio520 ?? '--'}
                      </td>
                      <td className="px-3 py-2">
                        <Badge variant={stateVariant(pick.state)}>{pick.stateLabel || pick.state}</Badge>
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-cyan">{pick.score?.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {picks.length ? (
            <section className="space-y-3">
              <h2 className="text-lg font-semibold text-white">交易计划</h2>
              <div className="grid gap-3 lg:grid-cols-2">
                {picks.map((pick) => (
                  <PlanCard key={`plan-${pick.symbol}`} pick={pick} />
                ))}
              </div>
            </section>
          ) : null}

          {result.validation?.status === 'ok' ? (
            <Card variant="bordered" padding="sm">
              <span className="label-uppercase">粗略历史验证</span>
              <p className="mt-1 text-sm text-secondary-text">
                样本 {result.validation.samples} | 10日胜率 {result.validation.hitRate10d}% | 平均超额 {pct(result.validation.avgExcess10d)}
              </p>
              <p className="mt-1 text-xs text-muted-text">{result.validation.note}</p>
            </Card>
          ) : null}
        </>
      ) : null}
    </AppPage>
  );
};

export default ScreenerPage;

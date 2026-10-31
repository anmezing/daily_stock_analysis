# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 核心分析流水线
===================================

职责：
1. 管理整个分析流程
2. 协调数据获取、存储、搜索、分析、通知等模块
3. 实现并发控制和异常处理
4. 提供股票分析的核心功能
"""

import logging
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from src.config import get_config, Config
from src.storage import get_db
from data_provider import DataFetcherManager
from data_provider.realtime_types import ChipDistribution
from src.analyzer import (
    GeminiAnalyzer,
    AnalysisResult,
    fill_chip_structure_if_needed,
    fill_price_position_if_needed,
    normalize_analysis_result_signals,
)
from src.data.stock_mapping import STOCK_NAME_MAP
from src.notification import NotificationService, NotificationChannel
from src.search_service import SearchService
from src.services.analysis_calibration_service import AnalysisCalibrationService
from src.services.benchmark_service import BenchmarkService
from src.enums import ReportType
from src.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult
from src.core.relative_strength import compute_relative_strength
from src.core.trade_plan import apply_trade_plan_fallback, build_trade_plan
from src.core.trading_calendar import get_market_for_stock, get_market_now, get_market_today, is_market_open, get_market_session, get_session_label
from src.services.market_themes_service import get_market_themes_service
from bot.models import BotMessage


logger = logging.getLogger(__name__)


class StockAnalysisPipeline:
    """
    股票分析主流程调度器
    
    职责：
    1. 管理整个分析流程
    2. 协调数据获取、存储、搜索、分析、通知等模块
    3. 实现并发控制和异常处理
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        max_workers: Optional[int] = None,
        source_message: Optional[BotMessage] = None,
        query_id: Optional[str] = None,
        query_source: Optional[str] = None,
        save_context_snapshot: Optional[bool] = None
    ):
        """
        初始化调度器
        
        Args:
            config: 配置对象（可选，默认使用全局配置）
            max_workers: 最大并发线程数（可选，默认从配置读取）
        """
        self.config = config or get_config()
        self.max_workers = max_workers or self.config.max_workers
        self.source_message = source_message
        self.query_id = query_id
        self.query_source = self._resolve_query_source(query_source)
        self.save_context_snapshot = (
            self.config.save_context_snapshot if save_context_snapshot is None else save_context_snapshot
        )
        
        # 初始化各模块
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        # 不再单独创建 akshare_fetcher，统一使用 fetcher_manager 获取增强数据
        self.trend_analyzer = StockTrendAnalyzer()  # 趋势分析器
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService(source_message=source_message)
        self.calibration_service = AnalysisCalibrationService(db_manager=self.db, config=self.config)
        self.benchmark_service = BenchmarkService(self.db, config=self.config)
        self._learning_refresh_lock = Lock()
        self._single_stock_notify_lock = Lock()
        
        # 初始化搜索服务
        self.search_service = None
        try:
            self.search_service = SearchService(
                bocha_keys=self.config.bocha_api_keys,
                tavily_keys=self.config.tavily_api_keys,
                anspire_keys=self.config.anspire_api_keys,
                brave_keys=self.config.brave_api_keys,
                serpapi_keys=self.config.serpapi_keys,
                minimax_keys=self.config.minimax_api_keys,
                xai_keys=self.config.xai_api_keys,
                xai_search_model=self.config.xai_search_model,
                searxng_base_urls=self.config.searxng_base_urls,
                searxng_public_instances_enabled=self.config.searxng_public_instances_enabled,
                news_max_age_days=self.config.news_max_age_days,
                news_strategy_profile=self.config.news_strategy_profile,
            )
        except Exception as e:
            logger.warning("搜索服务初始化失败，后续将按 fail-open 跳过情报搜索: %s", e)
        
        logger.info(f"调度器初始化完成，最大并发数: {self.max_workers}")
        logger.info("已启用趋势分析器 (MA5>MA10>MA20 多头判断)")
        # 打印实时行情/筹码配置状态
        if self.config.enable_realtime_quote:
            logger.info(f"实时行情已启用 (优先级: {self.config.realtime_source_priority})")
        else:
            logger.info("实时行情已禁用，将使用历史收盘价")
        if self.config.enable_chip_distribution:
            logger.info("筹码分布分析已启用")
        else:
            logger.info("筹码分布分析已禁用")
        if self.search_service and self.search_service.is_available:
            provider_names = [p.name for p in self.search_service._providers]
            if self.search_service._x_signal_provider:
                provider_names.append("xAI")
            logger.info(f"搜索服务已启用 ({', '.join(provider_names) or 'N/A'})")
        else:
            logger.warning("搜索服务未启用（未配置 API Key）")

    def _maybe_refresh_learning_loop(self) -> None:
        """Periodically refresh incremental backtests for the evolution loop."""
        if not getattr(self.calibration_service, "enabled", False):
            return

        interval_seconds = getattr(self.calibration_service, "refresh_interval_seconds", 0)
        now = time.time()
        last_refresh = getattr(self.calibration_service, "_last_refresh_ts", 0.0)
        if last_refresh and interval_seconds > 0 and now - last_refresh < interval_seconds:
            return

        with self._learning_refresh_lock:
            last_refresh = getattr(self.calibration_service, "_last_refresh_ts", 0.0)
            now = time.time()
            if last_refresh and interval_seconds > 0 and now - last_refresh < interval_seconds:
                if hasattr(self.calibration_service, "maybe_refresh_model"):
                    self.calibration_service.maybe_refresh_model()
                return
            self.calibration_service.maybe_refresh_backtests()
            if hasattr(self.calibration_service, "maybe_refresh_model"):
                self.calibration_service.maybe_refresh_model()

    def _send_single_stock_notification(
        self,
        result: AnalysisResult,
        report_type: ReportType,
    ) -> None:
        """Send one stock notification in a serialized path to avoid shared-notifier races."""
        if not result or not self.notifier.is_available():
            return

        lock = getattr(self, '_single_stock_notify_lock', None)
        if lock is None:
            lock = Lock()
            self._single_stock_notify_lock = lock

        with lock:
            if report_type == ReportType.FULL:
                report_content = self.notifier.generate_dashboard_report([result])
                logger.info("[%s] 使用完整报告格式", result.code)
            elif report_type == ReportType.BRIEF:
                report_content = self.notifier.generate_brief_report([result])
                logger.info("[%s] 使用简洁报告格式", result.code)
            else:
                report_content = self.notifier.generate_single_stock_report(result)
                logger.info("[%s] 使用精简报告格式", result.code)

            if self.notifier.send(report_content, email_stock_codes=[result.code]):
                logger.info("[%s] 单股推送成功", result.code)
            else:
                logger.warning("[%s] 单股推送失败", result.code)

    def _apply_result_calibration(
        self,
        result: Optional[AnalysisResult],
        *,
        stock_name: str,
        code: str,
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnalysisResult]:
        """Apply history-driven calibration without breaking the main pipeline."""
        if result is None:
            return None
        # 没拿到模型结论时不要做校准：这时的信号是占位值（观望/50），
        # 校准它只会产出一条假的"校准建议"，也会污染影子对照样本。
        if not getattr(result, "success", True):
            return result

        try:
            calibrated = self.calibration_service.calibrate_result(result, context_snapshot=context_snapshot)
            info = getattr(calibrated, "calibration_info", None) or {}
            if info.get("applied"):
                logger.info(
                    "%s(%s) 回测校准已应用: %s -> %s, scope=%s, samples=%s",
                    stock_name,
                    code,
                    info.get("current_signal"),
                    info.get("suggested_signal"),
                    info.get("source_scope"),
                    info.get("samples"),
                )
            return calibrated
        except Exception as e:
            logger.warning(f"{stock_name}({code}) 应用回测校准失败: {e}")
            return result

    def fetch_and_save_stock_data(
        self, 
        code: str,
        force_refresh: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        获取并保存单只股票数据
        
        断点续传逻辑：
        1. 检查数据库是否已有今日数据
        2. 如果有且不强制刷新，则跳过网络请求
        3. 否则从数据源获取并保存
        
        Args:
            code: 股票代码
            force_refresh: 是否强制刷新（忽略本地缓存）
            
        Returns:
            Tuple[是否成功, 错误信息]
        """
        stock_name = code
        try:
            # 首先获取股票名称
            stock_name = self.fetcher_manager.get_stock_name(code, allow_realtime=False)

            market = get_market_for_stock(code)
            today = get_market_today(market)
            # 注意：这里用自然日 date.today() 做“断点续传”判断。
            # 若在周末/节假日/非交易日运行，或机器时区不在中国，可能出现：
            # - 数据库已有最新交易日数据但仍会重复拉取（has_today_data 返回 False）
            # - 或在跨日/时区偏移时误判“今日已有数据”
            # 这里改为“按股票所属市场本地日期”判断，降低美股/港股跨时区误判。
            
            # 断点续传检查：如果今日数据已存在，跳过
            if not force_refresh and self.db.has_today_data(code, today):
                logger.info(f"{stock_name}({code}) 今日数据已存在，跳过获取（断点续传）")
                return True, None

            # 从数据源获取数据（拉取足够天数以支持 MA200 + 历史分析窗口）
            logger.info(f"{stock_name}({code}) 开始从数据源获取数据...")
            df, source_name = self.fetcher_manager.get_daily_data(code, days=220)

            if df is None or df.empty:
                return False, "获取数据为空"

            # 保存到数据库
            saved_count = self.db.save_daily_data(df, code, source_name)
            logger.info(f"{stock_name}({code}) 数据保存成功（来源: {source_name}，新增 {saved_count} 条）")

            return True, None

        except Exception as e:
            error_msg = f"获取/保存数据失败: {str(e)}"
            logger.error(f"{stock_name}({code}) {error_msg}")
            return False, error_msg
    
    def analyze_stock(self, code: str, report_type: ReportType, query_id: str) -> Optional[AnalysisResult]:
        """
        分析单只股票（增强版：含量比、换手率、筹码分析、多维度情报）
        
        流程：
        1. 获取实时行情（量比、换手率）- 通过 DataFetcherManager 自动故障切换
        2. 获取筹码分布 - 通过 DataFetcherManager 带熔断保护
        3. 进行趋势分析（基于交易理念）
        4. 多维度情报搜索（最新消息+风险排查+业绩预期）
        5. 从数据库获取分析上下文
        6. 调用 AI 进行综合分析
        
        Args:
            query_id: 查询链路关联 id
            code: 股票代码
            report_type: 报告类型
            
        Returns:
            AnalysisResult 或 None（如果分析失败）
        """
        stock_name = code
        try:
            self._maybe_refresh_learning_loop()

            # 获取股票名称（优先从实时行情获取真实名称）
            stock_name = self.fetcher_manager.get_stock_name(code, allow_realtime=False)

            # Step 1: 获取实时行情（量比、换手率等）- 使用统一入口，自动故障切换
            realtime_quote = None
            market = get_market_for_stock(code) or "cn"
            market_session = get_market_session(market)
            try:
                realtime_quote = self.fetcher_manager.get_realtime_quote(code)
                if realtime_quote:
                    # 使用实时行情返回的真实股票名称
                    if realtime_quote.name:
                        stock_name = realtime_quote.name
                    # 兼容不同数据源的字段（有些数据源可能没有 volume_ratio）
                    volume_ratio = getattr(realtime_quote, 'volume_ratio', None)
                    turnover_rate = getattr(realtime_quote, 'turnover_rate', None)
                    session = market_session
                    session_tag = {"pre_market": "[盘前]", "trading": "[盘中]", "post_market": "[盘后]", "closed": "[休市]"}.get(session, "")
                    _price_display = round(realtime_quote.price, 4) if realtime_quote.price is not None else None
                    logger.info(f"{stock_name}({code}) 实时行情{session_tag}: 价格={_price_display}, "
                              f"量比={volume_ratio}, 换手率={turnover_rate}% "
                              f"(来源: {realtime_quote.source.value if hasattr(realtime_quote, 'source') else 'unknown'})")
                else:
                    logger.info(f"{stock_name}({code}) 实时行情获取失败或已禁用，将使用历史数据进行分析")
            except Exception as e:
                logger.warning(f"{stock_name}({code}) 获取实时行情失败: {e}")

            # 如果还是没有名称，使用代码作为名称
            if not stock_name:
                stock_name = f'股票{code}'

            # Step 2: 获取筹码分布 - 使用统一入口，带熔断保护
            chip_data = None
            try:
                chip_data = self.fetcher_manager.get_chip_distribution(code)
                if chip_data:
                    logger.info(f"{stock_name}({code}) 筹码分布: 获利比例={chip_data.profit_ratio:.1%}, "
                              f"90%集中度={chip_data.concentration_90:.2%}")
                else:
                    logger.debug(f"{stock_name}({code}) 筹码分布获取失败或已禁用")
            except Exception as e:
                logger.warning(f"{stock_name}({code}) 获取筹码分布失败: {e}")

            # If agent mode is explicitly enabled, or specific agent skills are configured, use the Agent analysis pipeline.
            # NOTE: use config.agent_mode (explicit opt-in) instead of
            # config.is_agent_available() so that users who only configured an
            # API Key for the traditional analysis path are not silently
            # switched to Agent mode (which is slower and more expensive).
            use_agent = getattr(self.config, 'agent_mode', False)
            if not use_agent:
                # Auto-enable agent mode when specific skills are configured (e.g., scheduled task with strategy)
                configured_skills = getattr(self.config, 'agent_skills', [])
                if configured_skills and configured_skills != ['all']:
                    use_agent = True
                    logger.info(f"{stock_name}({code}) Auto-enabled agent mode due to configured skills: {configured_skills}")

            # Step 2.5: 基本面能力聚合（统一入口，异常降级）
            # - 失败时返回 partial/failed，不影响既有技术面/新闻链路
            # - 关闭开关时仍返回 not_supported 结构
            fundamental_context = None
            try:
                fundamental_context = self.fetcher_manager.get_fundamental_context(
                    code,
                    budget_seconds=getattr(self.config, 'fundamental_stage_timeout_seconds', 1.5),
                )
            except Exception as e:
                logger.warning(f"{stock_name}({code}) 基本面聚合失败: {e}")
                fundamental_context = self.fetcher_manager.build_failed_fundamental_context(code, str(e))

            # P0: write-only snapshot, fail-open, no read dependency on this table.
            try:
                self.db.save_fundamental_snapshot(
                    query_id=query_id,
                    code=code,
                    payload=fundamental_context,
                    source_chain=fundamental_context.get("source_chain", []),
                    coverage=fundamental_context.get("coverage", {}),
                )
            except Exception as e:
                logger.debug(f"{stock_name}({code}) 基本面快照写入失败: {e}")

            # Step 2.6: 市场热点 / 板块轮动（按市场缓存，失败不影响主流程）
            try:
                market_themes = get_market_themes_service().get_snapshot(market)
            except Exception as e:
                logger.debug(f"{stock_name}({code}) 市场热点获取失败: {e}")
                market_themes = {"status": "not_supported", "market": market, "reason": str(e)}

            # 基准状态在 Agent / 标准链之前算一次，供两条路径与相对强弱共用
            analysis_date = get_market_today(market)
            market_regime: Dict[str, Any] = {}
            try:
                if market in ("us", "hk"):
                    self.benchmark_service.ensure(
                        market,
                        analysis_date - timedelta(days=120),
                        analysis_date,
                    )
            except Exception as exc:
                logger.debug("[%s] 基准数据补齐失败: %s", code, exc)
            try:
                market_regime = self.benchmark_service.snapshot(market, analysis_date)
            except Exception as exc:
                logger.debug("[%s] 基准状态获取失败: %s", code, exc)

            # Step 3: 趋势分析（基于交易理念）— 在 Agent 分支之前执行，供两条路径共用
            trend_result: Optional[TrendAnalysisResult] = None
            try:
                end_date = get_market_today(get_market_for_stock(code))
                # 480 个自然日 ≈ 320 个交易日，保证 MA200 可计算
                start_date = end_date - timedelta(days=480)
                historical_bars = self.db.get_data_range(code, start_date, end_date)
                if historical_bars:
                    df = pd.DataFrame([bar.to_dict() for bar in historical_bars])
                    # Issue #234: Augment with realtime for intraday MA calculation
                    if self.config.enable_realtime_quote and realtime_quote:
                        df = self._augment_historical_with_realtime(df, realtime_quote, code)
                    intraday_bar = self._is_intraday_bar(df, code, market_session)
                    trend_result = self.trend_analyzer.analyze(
                        df, code, stock_name, intraday=intraday_bar
                    )
                    logger.info(f"{stock_name}({code}) 趋势分析: {trend_result.trend_status.value}, "
                              f"买入信号={trend_result.buy_signal.value}, 评分={trend_result.signal_score}")
            except Exception as e:
                logger.warning(f"{stock_name}({code}) 趋势分析失败: {e}", exc_info=True)

            market_intel_summary: Dict[str, Any] = {}
            try:
                if self.search_service:
                    market_intel_summary = self.search_service.build_market_intel_summary(
                        code,
                        stock_name,
                        {},
                    )
            except Exception as e:
                logger.debug(f"{stock_name}({code}) 构建市场摘要失败，使用兜底标签: {e}")
            if not isinstance(market_intel_summary, dict):
                market_tag = get_market_for_stock(code) or "cn"
                market_intel_summary = {
                    "market": market_tag,
                    "market_label": {
                        "cn": "A股",
                        "hk": "港股",
                        "us": "美股",
                    }.get(market_tag, market_tag),
                }

            # Step 4: 多维度情报搜索（最新消息+风险排查+业绩预期）
            news_context = None
            intel_results: Dict[str, Any] = {}
            if self.search_service and self.search_service.can_run_comprehensive_intel(code):
                logger.info(f"{stock_name}({code}) 开始多维度情报搜索...")
                try:
                    intel_results = self.search_service.search_comprehensive_intel(
                        stock_code=code,
                        stock_name=stock_name,
                        max_searches=9
                    )

                    if isinstance(intel_results, dict) and intel_results:
                        market_intel_summary = self.search_service.build_market_intel_summary(
                            code,
                            stock_name,
                            intel_results,
                        )
                        news_context = self.search_service.format_intel_report(
                            intel_results,
                            stock_name,
                        )
                        total_results = sum(
                            len(r.results) for r in intel_results.values() if r.success
                        )
                        logger.info(f"{stock_name}({code}) 情报搜索完成: 共 {total_results} 条结果")
                        logger.debug(f"{stock_name}({code}) 情报搜索结果:\n{news_context}")

                        try:
                            query_context = self._build_query_context(query_id=query_id)
                            for dim_name, response in intel_results.items():
                                if response and response.success and response.results:
                                    self.db.save_news_intel(
                                        code=code,
                                        name=stock_name,
                                        dimension=dim_name,
                                        query=response.query,
                                        response=response,
                                        query_context=query_context
                                    )
                        except Exception as e:
                            logger.warning(f"{stock_name}({code}) 保存新闻情报失败: {e}")
                    else:
                        intel_results = {}
                except Exception as e:
                    logger.warning(f"{stock_name}({code}) 多维度情报搜索失败: {e}")
            else:
                logger.info(f"{stock_name}({code}) 无可用情报源，跳过情报搜索")

            if use_agent:
                logger.info(f"{stock_name}({code}) 启用 Agent 模式进行分析")
                return self._analyze_with_agent(
                    code,
                    report_type,
                    query_id,
                    stock_name,
                    realtime_quote,
                    chip_data,
                    fundamental_context,
                    trend_result,
                    market_context=market_intel_summary,
                    news_context=news_context,
                    intel_results=intel_results,
                    market_regime=market_regime,
                    market_themes=market_themes,
                )

            # Step 5: 获取分析上下文（技术面数据 + 历史K线）
            history_days = getattr(self.config, 'analysis_history_days', 60)
            context = self.db.get_analysis_context(code, target_date=analysis_date, history_days=history_days)
            # 分析日期是“这次判断发生在哪一天”；context['date'] 是最后一根 K 线的日期。
            # 两者必须分开，否则数据一过期，报告就会把旧 K 线日期写成分析日期。
            if context is not None:
                context['analysis_date'] = analysis_date.isoformat()

            if context is None:
                logger.warning(f"{stock_name}({code}) 无法获取历史行情数据，将仅基于新闻和实时行情分析")
                market_today = get_market_today(get_market_for_stock(code))
                context = {
                    'code': code,
                    'stock_name': stock_name,
                    'date': market_today.isoformat(),
                    'analysis_date': market_today.isoformat(),
                    'data_missing': True,
                    'today': {},
                    'yesterday': {}
                }
            
            # Step 6: 增强上下文数据（添加实时行情、筹码、趋势分析结果、股票名称）
            enhanced_context = self._enhance_context(
                context, 
                realtime_quote, 
                chip_data,
                trend_result,
                stock_name,  # 传入股票名称
                fundamental_context,
                market_context=market_intel_summary,
                market_regime=market_regime,
                market_themes=market_themes,
            )
            
            # Step 7: 调用 AI 分析（传入增强的上下文和新闻）
            result = self.analyzer.analyze(enhanced_context, news_context=news_context)
            if result:
                result.market_themes = market_themes

            # Step 7.5: 填充分析时的价格信息到 result
            if result:
                result.query_id = query_id
                result.analysis_date = analysis_date.isoformat()
                result.data_date = enhanced_context.get('date') or context.get('date')
                realtime_data = enhanced_context.get('realtime', {})
                result.current_price = realtime_data.get('price')
                result.change_pct = realtime_data.get('change_pct')

            # Step 7.6: chip_structure fallback (Issue #589)
            if result and chip_data:
                fill_chip_structure_if_needed(result, chip_data)

            # Step 7.7: price_position fallback
            if result:
                fill_price_position_if_needed(result, trend_result, realtime_quote)

            context_snapshot = self._build_context_snapshot(
                enhanced_context=enhanced_context,
                news_content=news_context,
                realtime_quote=realtime_quote,
                chip_data=chip_data,
            )

            # Step 7.8: 历史回测校准（普通分析链）
            result = self._apply_result_calibration(
                result,
                stock_name=stock_name,
                code=code,
                context_snapshot=context_snapshot,
            )

            # Step 8: 保存分析历史记录
            # 与 Agent 链保持一致：LLM 失败时 result 只是占位默认值
            # （50 / 持有 / 成功后无 model_used），不该进 Web 历史与回测样本。
            if result and not result.success:
                logger.error(
                    "%s(%s) 分析未成功（%s），本轮不写入分析历史",
                    stock_name,
                    code,
                    result.error_message or "未知原因",
                )
            if result and result.success:
                try:
                    saved = self.db.save_analysis_history(
                        result=result,
                        query_id=query_id,
                        report_type=report_type.value,
                        news_content=news_context,
                        context_snapshot=context_snapshot,
                        save_snapshot=self.save_context_snapshot,
                        strategy_id=None,
                    )
                    if not saved:
                        logger.error(
                            "%s(%s) 分析结果未写入数据库：本次结果不会进入回测/学习",
                            stock_name,
                            code,
                        )
                        note = "本次分析结果未写入数据库，无法参与回测校准。"
                        result.risk_warning = f"{result.risk_warning} {note}".strip()
                except Exception as e:
                    logger.warning(f"{stock_name}({code}) 保存分析历史失败: {e}")

            return result

        except Exception as e:
            logger.error(f"{stock_name}({code}) 分析失败: {e}")
            logger.exception(f"{stock_name}({code}) 详细错误信息:")
            return None
    
    def _enhance_context(
        self,
        context: Dict[str, Any],
        realtime_quote,
        chip_data: Optional[ChipDistribution],
        trend_result: Optional[TrendAnalysisResult],
        stock_name: str = "",
        fundamental_context: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        market_regime: Optional[Dict[str, Any]] = None,
        market_themes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        增强分析上下文
        
        将实时行情、筹码分布、趋势分析结果、股票名称添加到上下文中
        
        Args:
            context: 原始上下文
            realtime_quote: 实时行情数据（UnifiedRealtimeQuote 或 None）
            chip_data: 筹码分布数据
            trend_result: 趋势分析结果
            stock_name: 股票名称
            
        Returns:
            增强后的上下文
        """
        enhanced = context.copy()
        
        # 添加股票名称
        if stock_name:
            enhanced['stock_name'] = stock_name
        elif realtime_quote and getattr(realtime_quote, 'name', None):
            enhanced['stock_name'] = realtime_quote.name
        
        # 添加实时行情（兼容不同数据源的字段差异）
        if realtime_quote:
            # 使用 getattr 安全获取字段，缺失字段返回 None 或默认值
            volume_ratio = getattr(realtime_quote, 'volume_ratio', None)
            enhanced['realtime'] = {
                'name': getattr(realtime_quote, 'name', ''),
                'price': getattr(realtime_quote, 'price', None),
                'change_pct': getattr(realtime_quote, 'change_pct', None),
                'volume_ratio': volume_ratio,
                'volume_ratio_desc': self._describe_volume_ratio(volume_ratio) if volume_ratio else '无数据',
                'turnover_rate': getattr(realtime_quote, 'turnover_rate', None),
                'pe_ratio': getattr(realtime_quote, 'pe_ratio', None),
                'pb_ratio': getattr(realtime_quote, 'pb_ratio', None),
                'total_mv': getattr(realtime_quote, 'total_mv', None),
                'circ_mv': getattr(realtime_quote, 'circ_mv', None),
                'change_60d': getattr(realtime_quote, 'change_60d', None),
                'source': getattr(realtime_quote, 'source', None),
            }
            # 移除 None 值以减少上下文大小
            enhanced['realtime'] = {k: v for k, v in enhanced['realtime'].items() if v is not None}
        
        # 添加筹码分布
        if chip_data:
            current_price = getattr(realtime_quote, 'price', 0) if realtime_quote else 0
            enhanced['chip'] = {
                'profit_ratio': chip_data.profit_ratio,
                'avg_cost': chip_data.avg_cost,
                'concentration_90': chip_data.concentration_90,
                'concentration_70': chip_data.concentration_70,
                'chip_status': chip_data.get_chip_status(current_price or 0),
            }
        
        # 添加趋势分析结果
        if trend_result:
            enhanced['trend_analysis'] = {
                'trend_status': trend_result.trend_status.value,
                'ma_alignment': trend_result.ma_alignment,
                'trend_strength': trend_result.trend_strength,
                'bias_ma5': trend_result.bias_ma5,
                'bias_ma10': trend_result.bias_ma10,
                'volume_status': trend_result.volume_status.value,
                'volume_ratio_5d': trend_result.volume_ratio_5d,
                'volume_trend': trend_result.volume_trend,
                'buy_signal': trend_result.buy_signal.value,
                'signal_score': trend_result.signal_score,
                'signal_reasons': trend_result.signal_reasons,
                'risk_factors': trend_result.risk_factors,
            }

        # Issue #234: Override today with realtime OHLC + trend MA for intraday analysis
        # Guard: trend_result.ma5 > 0 ensures MA calculation succeeded (data sufficient)
        if realtime_quote and trend_result and trend_result.ma5 > 0:
            price = getattr(realtime_quote, 'price', None)
            if price is not None and price > 0:
                yesterday_close = None
                if enhanced.get('yesterday') and isinstance(enhanced['yesterday'], dict):
                    yesterday_close = enhanced['yesterday'].get('close')
                orig_today = enhanced.get('today') or {}
                open_p = getattr(realtime_quote, 'open_price', None) or getattr(
                    realtime_quote, 'pre_close', None
                ) or yesterday_close or orig_today.get('open') or price
                high_p = getattr(realtime_quote, 'high', None) or price
                low_p = getattr(realtime_quote, 'low', None) or price
                vol = getattr(realtime_quote, 'volume', None)
                amt = getattr(realtime_quote, 'amount', None)
                pct = getattr(realtime_quote, 'change_pct', None)
                realtime_today = {
                    'close': price,
                    'open': open_p,
                    'high': high_p,
                    'low': low_p,
                    'ma5': trend_result.ma5,
                    'ma10': trend_result.ma10,
                    'ma20': trend_result.ma20,
                    'ma50': trend_result.ma50,
                    'ma60': trend_result.ma60,
                    'ma200': trend_result.ma200,
                }
                if vol is not None:
                    realtime_today['volume'] = vol
                if amt is not None:
                    realtime_today['amount'] = amt
                if pct is not None:
                    realtime_today['pct_chg'] = pct
                for k, v in orig_today.items():
                    if k not in realtime_today and v is not None:
                        realtime_today[k] = v
                enhanced['today'] = realtime_today
                enhanced['quote_injected'] = True
                enhanced['ma_status'] = self._compute_ma_status(
                    price, trend_result.ma5, trend_result.ma10, trend_result.ma20
                )
                enhanced['date'] = get_market_now(get_market_for_stock(context.get('code', ''))).date().isoformat()
                if yesterday_close is not None:
                    try:
                        yc = float(yesterday_close)
                        if yc > 0:
                            enhanced['price_change_ratio'] = round(
                                (price - yc) / yc * 100, 2
                            )
                    except (TypeError, ValueError):
                        pass
                if vol is not None and enhanced.get('yesterday'):
                    yest_vol = enhanced['yesterday'].get('volume') if isinstance(
                        enhanced['yesterday'], dict
                    ) else None
                    if yest_vol is not None:
                        try:
                            yv = float(yest_vol)
                            if yv > 0:
                                enhanced['volume_change_ratio'] = round(
                                    float(vol) / yv, 2
                                )
                        except (TypeError, ValueError):
                            pass

        # 趋势结果里的均线是本地重算的，比数据库里可能为空的 ma5/ma10/ma20 更可靠
        if trend_result and trend_result.ma5 > 0:
            today_block = enhanced.get('today')
            if isinstance(today_block, dict):
                for key, value in (
                    ('ma5', trend_result.ma5),
                    ('ma10', trend_result.ma10),
                    ('ma20', trend_result.ma20),
                    ('ma50', trend_result.ma50),
                    ('ma60', trend_result.ma60),
                    ('ma200', trend_result.ma200),
                ):
                    if today_block.get(key) is None and value:
                        today_block[key] = value

        # ETF/index flag for analyzer prompt (Fixes #274)
        enhanced['is_index_etf'] = SearchService.is_index_or_etf(
            context.get('code', ''), enhanced.get('stock_name', stock_name)
        )

        # Expose news search window so downstream consumers can align date ranges
        if self.search_service is not None:
            enhanced['news_window_days'] = getattr(self.search_service, 'news_window_days', 3)

        market_tag = get_market_for_stock(context.get('code', '')) or "cn"
        fallback_market_context = {
            "market": market_tag,
            "market_label": {
                "cn": "A股",
                "hk": "港股",
                "us": "美股",
            }.get(market_tag, market_tag),
        }
        enhanced["market_context"] = (
            market_context
            if isinstance(market_context, dict) and market_context.get("market")
            else fallback_market_context
        )
        enhanced["market"] = enhanced["market_context"].get("market", market_tag)

        # 基准指数状态（SPY/2800.HK）：风险偏好与中期趋势
        enhanced["market_regime"] = market_regime if isinstance(market_regime, dict) else {}

        # Improvement 3+4: inject market session label so LLM knows data timeliness
        enhanced["session_label"] = get_session_label(market_tag)
        enhanced["is_intraday"] = get_market_session(market_tag) == "trading"

        # P0: append unified fundamental block; keep as additional context only
        enhanced["fundamental_context"] = (
            fundamental_context
            if isinstance(fundamental_context, dict)
            else self.fetcher_manager.build_failed_fundamental_context(
                context.get("code", ""),
                "invalid fundamental context",
            )
        )

        # 市场热点与相对强弱：把"个股 vs 市场"的信息一并喂给模型
        enhanced["market_themes"] = market_themes if isinstance(market_themes, dict) else {}
        enhanced["relative_strength"] = self._compute_relative_strength(
            context, realtime_quote, market_regime
        )

        return enhanced

    def _ensure_agent_history(self, code: str, min_days: int = 320) -> None:
        """Ensure at least *min_days* of K-line history is in DB for agent tools."""
        from src.services.history_loader import get_frozen_target_date

        target = get_frozen_target_date()
        if target is None:
            target = get_market_today(get_market_for_stock(code))
        start = target - timedelta(days=int(min_days * 1.8))
        bars = self.db.get_data_range(code, start, target)
        if bars and len(bars) >= min(min_days, 260):
            logger.debug("[%s] Agent history: %d bars in DB, sufficient", code, len(bars))
            return
        try:
            df, source = self.fetcher_manager.get_daily_data(code, days=min_days)
            if df is not None and not df.empty:
                self.db.save_daily_data(df, code, source)
                logger.info("[%s] Prefetched %d rows of history for agent (source: %s)", code, len(df), source)
        except Exception as e:
            logger.warning("[%s] Agent history prefetch failed: %s", code, e)

    def _load_agent_analysis_context(self, code: str) -> Dict[str, Any]:
        """Load the 60-day K-line context for the Agent path (fail-open)."""
        try:
            from src.services.history_loader import get_frozen_target_date

            target_date = get_frozen_target_date()
            if target_date is None:
                target_date = get_market_today(get_market_for_stock(code))
            raw_context = self.db.get_analysis_context(
                code,
                target_date=target_date,
                history_days=60,
            )
            if isinstance(raw_context, dict):
                return raw_context
        except Exception as exc:
            logger.debug("[%s] Agent 分析上下文获取失败: %s", code, exc)
        return {"code": code}

    def _compute_relative_strength(
        self,
        context: Optional[Dict[str, Any]],
        realtime_quote: Any,
        market_regime: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute 20d/60d excess return versus the market benchmark."""
        context = context if isinstance(context, dict) else {}
        regime = market_regime if isinstance(market_regime, dict) else {}
        today = context.get("today") if isinstance(context.get("today"), dict) else {}
        close = getattr(realtime_quote, "price", None) if realtime_quote else None
        if close is None:
            close = today.get("close")
        return compute_relative_strength(
            context.get("history") or [],
            close,
            benchmark=str(regime.get("code") or ("SPY" if (get_market_for_stock(context.get("code", "")) == "us") else "benchmark")),
            benchmark_return_20d=regime.get("return_20d"),
            benchmark_return_60d=regime.get("return_60d"),
        )

    def _build_trade_plan_reference(
        self,
        code: str,
        trend_result: Optional[TrendAnalysisResult] = None,
        realtime_quote: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the rule-derived trade plan used by the Agent path.

        The Agent chain skips ``get_analysis_context`` in the main flow, so we
        fetch the same 60-day K-line window here to keep entry/stop levels
        grounded in real data instead of letting the model invent them.
        """
        context = dict(context) if isinstance(context, dict) else self._load_agent_analysis_context(code)
        context.setdefault("code", code)
        if trend_result is not None:
            context["trend_analysis"] = {
                "trend_status": trend_result.trend_status.value,
                "ma5": trend_result.ma5,
                "ma10": trend_result.ma10,
                "ma20": trend_result.ma20,
                "ma50": trend_result.ma50,
            }
        price = getattr(realtime_quote, "price", None) if realtime_quote else None
        if price:
            context.setdefault("realtime", {})
            if isinstance(context["realtime"], dict):
                context["realtime"]["price"] = price
        return build_trade_plan(context)

    def _analyze_with_agent(
        self, 
        code: str, 
        report_type: ReportType, 
        query_id: str,
        stock_name: str,
        realtime_quote: Any,
        chip_data: Optional[ChipDistribution],
        fundamental_context: Optional[Dict[str, Any]] = None,
        trend_result: Optional[TrendAnalysisResult] = None,
        market_context: Optional[Dict[str, Any]] = None,
        news_context: Optional[str] = None,
        intel_results: Optional[Dict[str, Any]] = None,
        market_regime: Optional[Dict[str, Any]] = None,
        market_themes: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnalysisResult]:
        """
        使用 Agent 模式分析单只股票。
        """
        try:
            from src.agent.factory import build_agent_executor

            # Build executor from shared factory (ToolRegistry and SkillManager prototype are cached)
            executor = build_agent_executor(self.config, getattr(self.config, 'agent_skills', None) or None)

            # Build initial context to avoid redundant tool calls
            initial_context = {
                "stock_code": code,
                "stock_name": stock_name,
                "report_type": report_type.value,
                "report_language": getattr(self.config, "report_language", "zh"),
                # 回测要按"做出判断的那一天"取前向行情；缺了它只能退回 created_at，
                # 而 created_at 是服务器本地时间，美股夜盘/跨时区会错一天。
                "analysis_date": get_market_today(get_market_for_stock(code)).isoformat(),
                "fundamental_context": fundamental_context,
            }
            agent_analysis_context = self._load_agent_analysis_context(code)
            trade_plan_reference = self._build_trade_plan_reference(
                code, trend_result, realtime_quote, context=agent_analysis_context
            )
            initial_context["trade_plan_reference"] = trade_plan_reference
            initial_context["relative_strength"] = self._compute_relative_strength(
                agent_analysis_context, realtime_quote, market_regime
            )
            if isinstance(market_themes, dict) and market_themes:
                initial_context["market_themes"] = market_themes
            if isinstance(market_regime, dict) and market_regime:
                initial_context["market_regime"] = market_regime
            if isinstance(market_context, dict) and market_context:
                initial_context["market_context"] = market_context
            if news_context:
                initial_context["news_context"] = news_context
                initial_context["intel_report"] = news_context
            if isinstance(intel_results, dict) and intel_results:
                initial_context["intel_dimensions"] = {
                    dim_name: {
                        "query": getattr(response, "query", ""),
                        "provider": getattr(response, "provider", ""),
                        "success": getattr(response, "success", False),
                        "error_message": getattr(response, "error_message", None),
                        "results": [
                            {
                                "title": getattr(item, "title", ""),
                                "snippet": getattr(item, "snippet", ""),
                                "url": getattr(item, "url", ""),
                                "source": getattr(item, "source", ""),
                                "published_date": getattr(item, "published_date", None),
                            }
                            for item in getattr(response, "results", [])[:4]
                        ],
                        "metadata": getattr(response, "metadata", {}),
                    }
                    for dim_name, response in intel_results.items()
                }
            
            if realtime_quote:
                initial_context["realtime_quote"] = self._safe_to_dict(realtime_quote)
            if chip_data:
                initial_context["chip_distribution"] = self._safe_to_dict(chip_data)
            if trend_result:
                initial_context["trend_result"] = self._safe_to_dict(trend_result)

            # Issue #1066: ensure deep history is in DB before agent tools run
            self._ensure_agent_history(code)

            # 运行 Agent
            message = f"请分析股票 {code} ({stock_name})，并生成决策仪表盘报告。"
            agent_result = executor.run(message, context=initial_context)

            # 转换为 AnalysisResult
            result = self._agent_result_to_analysis_result(agent_result, code, stock_name, report_type, query_id)
            if result:
                result.query_id = query_id
                result.market_themes = market_themes if isinstance(market_themes, dict) else None
                # 模型漏掉的入场/止损/卖出条件用规则计划补齐，保证报告和回测都有可执行点位
                if result.success:
                    apply_trade_plan_fallback(result, trade_plan_reference)
            # Agent weak integrity: placeholder fill only, no LLM retry
            # 只有拿到模型结论时才做完整性补全：失败结果本来就不入库，
            # 再补一遍占位只会让日志出现"必填字段缺失"的假象。
            if result and result.success and getattr(self.config, "report_integrity_enabled", False):
                from src.analyzer import check_content_integrity, apply_placeholder_fill

                pass_integrity, missing = check_content_integrity(result)
                if not pass_integrity:
                    apply_placeholder_fill(result, missing)
                    logger.info(
                        "[LLM完整性] integrity_mode=agent_weak 必填字段缺失 %s，已占位补全",
                        missing,
                    )
            # chip_structure fallback (Issue #589), before save_analysis_history
            if result and chip_data:
                fill_chip_structure_if_needed(result, chip_data)

            # price_position fallback (same as non-agent path Step 7.7)
            if result:
                fill_price_position_if_needed(result, trend_result, realtime_quote)

            resolved_stock_name = result.name if result and result.name else stock_name

            # 历史回测校准（Agent 链复用同一套公共逻辑）
            result = self._apply_result_calibration(
                result,
                stock_name=resolved_stock_name,
                code=code,
                context_snapshot=initial_context,
            )
            resolved_stock_name = result.name if result and result.name else resolved_stock_name

            # 兼容旧路径：若本轮未预取情报，则至少持久化一轮新闻搜索结果。
            if not intel_results and self.search_service and self.search_service.is_available:
                try:
                    news_response = self.search_service.search_stock_news(
                        stock_code=code,
                        stock_name=resolved_stock_name,
                        max_results=5
                    )
                    if news_response.success and news_response.results:
                        query_context = self._build_query_context(query_id=query_id)
                        self.db.save_news_intel(
                            code=code,
                            name=resolved_stock_name,
                            dimension="latest_news",
                            query=news_response.query,
                            response=news_response,
                            query_context=query_context
                        )
                        logger.info(f"[{code}] Agent 模式: 新闻情报已保存 {len(news_response.results)} 条")
                except Exception as e:
                    logger.warning(f"[{code}] Agent 模式保存新闻情报失败: {e}")

            # 保存分析历史记录
            # 只有真正拿到模型结论才入库：LLM 全部失败时 result 仍是占位默认值
            # （评分 50 / 观望 / 未知），入库后会污染 Web 历史与回测样本。
            if result and not result.success:
                logger.error(
                    "[%s] Agent 未产出有效结论（%s），本轮不写入分析历史",
                    code,
                    result.error_message or "未知原因",
                )
            if result and result.success:
                try:
                    initial_context["stock_name"] = resolved_stock_name
                    self.db.save_analysis_history(
                        result=result,
                        query_id=query_id,
                        report_type=report_type.value,
                        news_content=None,
                        context_snapshot=initial_context,
                        save_snapshot=self.save_context_snapshot,
                        strategy_id=result.strategy_id,
                    )
                except Exception as e:
                    logger.warning(f"[{code}] 保存 Agent 分析历史失败: {e}")

            return result

        except Exception as e:
            logger.error(f"[{code}] Agent 分析失败: {e}")
            logger.exception(f"[{code}] Agent 详细错误信息:")
            return None

    def _agent_result_to_analysis_result(
        self, agent_result, code: str, stock_name: str, report_type: ReportType, query_id: str
    ) -> AnalysisResult:
        """
        将 AgentResult 转换为 AnalysisResult。
        """
        result = AnalysisResult(
            code=code,
            name=stock_name,
            sentiment_score=50,
            trend_prediction="未知",
            operation_advice="观望",
            success=agent_result.success,
            error_message=agent_result.error or None,
            data_sources=f"agent:{agent_result.provider}",
            model_used=agent_result.model or None,
            strategy_id=getattr(agent_result, 'strategy_id', None),
        )

        if agent_result.success and agent_result.dashboard:
            dash = agent_result.dashboard
            ai_stock_name = str(dash.get("stock_name", "")).strip()
            if ai_stock_name and self._is_placeholder_stock_name(stock_name, code):
                result.name = ai_stock_name
            result.sentiment_score = self._safe_int(dash.get("sentiment_score"), 50)
            result.trend_prediction = dash.get("trend_prediction", "未知")
            raw_advice = dash.get("operation_advice", "观望")
            if isinstance(raw_advice, dict):
                # LLM may return {"no_position": "...", "has_position": "..."}
                # Derive a short string from decision_type for the scalar field
                _signal_to_advice = {
                    "buy": "买入", "sell": "卖出", "hold": "持有",
                    "strong_buy": "强烈买入", "strong_sell": "强烈卖出",
                }
                # Normalize decision_type (strip/lower) before lookup so
                # variants like "BUY" or " Buy " map correctly.
                raw_dt = str(dash.get("decision_type") or "hold").strip().lower()
                result.operation_advice = _signal_to_advice.get(raw_dt, "观望")
            else:
                result.operation_advice = str(raw_advice) if raw_advice else "观望"
            from src.agent.protocols import normalize_decision_signal

            result.decision_type = normalize_decision_signal(
                dash.get("decision_type", "hold")
            )
            result.analysis_summary = dash.get("analysis_summary", "")
            # The AI returns a top-level dict that contains a nested 'dashboard' sub-key
            # with core_conclusion / battle_plan / intelligence.  AnalysisResult's helper
            # methods (get_sniper_points, get_core_conclusion, etc.) expect that inner
            # structure, so we unwrap it here.
            result.dashboard = dash.get("dashboard") or dash
            result = normalize_analysis_result_signals(result)
        else:
            result.sentiment_score = 50
            result.operation_advice = "观望"
            if not result.error_message:
                result.error_message = "Agent 未能生成有效的决策仪表盘"

        return result

    @staticmethod
    def _is_placeholder_stock_name(name: str, code: str) -> bool:
        """Return True when the stock name is missing or placeholder-like."""
        if not name:
            return True
        normalized = str(name).strip()
        if not normalized:
            return True
        if normalized == code:
            return True
        if normalized.startswith("股票"):
            return True
        if "Unknown" in normalized:
            return True
        return False

    @staticmethod
    def _safe_int(value: Any, default: int = 50) -> int:
        """安全地将值转换为整数。"""
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            import re
            match = re.search(r'-?\d+', value)
            if match:
                return int(match.group())
        return default
    
    def _describe_volume_ratio(self, volume_ratio: float) -> str:
        """
        量比描述
        
        量比 = 当前成交量 / 过去5日平均成交量
        """
        if volume_ratio < 0.5:
            return "极度萎缩"
        elif volume_ratio < 0.8:
            return "明显萎缩"
        elif volume_ratio < 1.2:
            return "正常"
        elif volume_ratio < 2.0:
            return "温和放量"
        elif volume_ratio < 3.0:
            return "明显放量"
        else:
            return "巨量"

    @staticmethod
    def _compute_ma_status(close: float, ma5: float, ma10: float, ma20: float) -> str:
        """
        Compute MA alignment status from price and MA values.
        Logic mirrors storage._analyze_ma_status (Issue #234).
        """
        close = close or 0
        ma5 = ma5 or 0
        ma10 = ma10 or 0
        ma20 = ma20 or 0
        if close > ma5 > ma10 > ma20 > 0:
            return "多头排列 📈"
        elif close < ma5 < ma10 < ma20 and ma20 > 0:
            return "空头排列 📉"
        elif close > ma5 and ma5 > ma10:
            return "短期向好 🔼"
        elif close < ma5 and ma5 < ma10:
            return "短期走弱 🔽"
        else:
            return "震荡整理 ↔️"

    def _is_intraday_bar(self, df: pd.DataFrame, code: str, market_session: str) -> bool:
        """最后一根 K 线是否仍是"当日未收盘"的那根。

        只在交易时段内成立：此时成交量只是盘中累计值，不能按全天量能解读。
        收盘后（post_market/closed）拿到的是完整 K 线，可以正常判断。
        """
        if market_session != "trading":
            return False
        if df is None or df.empty or "date" not in df.columns:
            return False
        try:
            last_value = df["date"].iloc[-1]
            last_date = (
                last_value.date() if hasattr(last_value, "date") else pd.Timestamp(last_value).date()
            )
        except (TypeError, ValueError):
            return False
        return last_date == get_market_today(get_market_for_stock(code))

    def _augment_historical_with_realtime(
        self, df: pd.DataFrame, realtime_quote: Any, code: str
    ) -> pd.DataFrame:
        """
        Augment historical OHLCV with today's realtime quote for intraday MA calculation.
        Issue #234: Use realtime price instead of yesterday's close for technical indicators.
        """
        if df is None or df.empty or 'close' not in df.columns:
            return df
        if realtime_quote is None:
            return df
        price = getattr(realtime_quote, 'price', None)
        if price is None or not (isinstance(price, (int, float)) and price > 0):
            return df

        # Optional: skip augmentation on non-trading days (fail-open)
        enable_realtime_tech = getattr(
            self.config, 'enable_realtime_technical_indicators', True
        )
        if not enable_realtime_tech:
            return df
        market = get_market_for_stock(code)
        market_today = get_market_now(market).date()
        if market and not is_market_open(market, market_today):
            return df

        last_val = df['date'].max()
        last_date = (
            last_val.date() if hasattr(last_val, 'date') else
            (last_val if isinstance(last_val, date) else pd.Timestamp(last_val).date())
        )
        yesterday_close = float(df.iloc[-1]['close']) if len(df) > 0 else price
        open_p = getattr(realtime_quote, 'open_price', None) or getattr(
            realtime_quote, 'pre_close', None
        ) or yesterday_close
        high_p = getattr(realtime_quote, 'high', None) or price
        low_p = getattr(realtime_quote, 'low', None) or price
        vol = getattr(realtime_quote, 'volume', None) or 0
        amt = getattr(realtime_quote, 'amount', None)
        pct = getattr(realtime_quote, 'change_pct', None)

        if last_date >= market_today:
            # Update last row with realtime close (copy to avoid mutating caller's df)
            df = df.copy()
            idx = df.index[-1]
            df.loc[idx, 'close'] = price
            if open_p is not None:
                df.loc[idx, 'open'] = open_p
            if high_p is not None:
                df.loc[idx, 'high'] = high_p
            if low_p is not None:
                df.loc[idx, 'low'] = low_p
            if vol:
                df.loc[idx, 'volume'] = vol
            if amt is not None:
                df.loc[idx, 'amount'] = amt
            if pct is not None:
                df.loc[idx, 'pct_chg'] = pct
        else:
            # Append virtual today row
            new_row = {
                'code': code,
                'date': market_today,
                'open': open_p,
                'high': high_p,
                'low': low_p,
                'close': price,
                'volume': vol,
                'amount': amt if amt is not None else 0,
                'pct_chg': pct if pct is not None else 0,
            }
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)

        # Deduplicate by date (keep last) to avoid double-counting if
        # historical data already includes today's row
        if 'date' in df.columns:
            df['_dedup_date'] = pd.to_datetime(df['date']).dt.date
            df = df.drop_duplicates(subset=['_dedup_date'], keep='last').drop(columns=['_dedup_date'])
            df = df.reset_index(drop=True)

        return df

    def _build_context_snapshot(
        self,
        enhanced_context: Dict[str, Any],
        news_content: Optional[str],
        realtime_quote: Any,
        chip_data: Optional[ChipDistribution]
    ) -> Dict[str, Any]:
        """
        构建分析上下文快照
        """
        return {
            "enhanced_context": enhanced_context,
            "news_content": news_content,
            "realtime_quote_raw": self._safe_to_dict(realtime_quote),
            "chip_distribution_raw": self._safe_to_dict(chip_data),
        }

    @staticmethod
    def _safe_to_dict(value: Any) -> Optional[Dict[str, Any]]:
        """
        安全转换为字典
        """
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                return None
        if hasattr(value, "__dict__"):
            try:
                return dict(value.__dict__)
            except Exception:
                return None
        return None

    def _resolve_query_source(self, query_source: Optional[str]) -> str:
        """
        解析请求来源。

        优先级（从高到低）：
        1. 显式传入的 query_source：调用方明确指定时优先使用，便于覆盖推断结果或兼容未来 source_message 来自非 bot 的场景
        2. 存在 source_message 时推断为 "bot"：当前约定为机器人会话上下文
        3. 存在 query_id 时推断为 "web"：Web 触发的请求会带上 query_id
        4. 默认 "system"：定时任务或 CLI 等无上述上下文时

        Args:
            query_source: 调用方显式指定的来源，如 "bot" / "web" / "cli" / "system"

        Returns:
            归一化后的来源标识字符串，如 "bot" / "web" / "cli" / "system"
        """
        if query_source:
            return query_source
        if self.source_message:
            return "bot"
        if self.query_id:
            return "web"
        return "system"

    def _build_query_context(self, query_id: Optional[str] = None) -> Dict[str, str]:
        """
        生成用户查询关联信息
        """
        effective_query_id = query_id or self.query_id or ""

        context: Dict[str, str] = {
            "query_id": effective_query_id,
            "query_source": self.query_source or "",
        }

        if self.source_message:
            context.update({
                "requester_platform": self.source_message.platform or "",
                "requester_user_id": self.source_message.user_id or "",
                "requester_user_name": self.source_message.user_name or "",
                "requester_chat_id": self.source_message.chat_id or "",
                "requester_message_id": self.source_message.message_id or "",
                "requester_query": self.source_message.content or "",
            })

        return context

    @staticmethod
    def _filter_supported_stock_codes(stock_codes: List[str]) -> List[str]:
        """Drop syntactically unsupported stock codes before entering the pipeline."""
        filtered_codes: List[str] = []
        for code in stock_codes:
            market = get_market_for_stock(code)
            if market is None:
                logger.warning("无法识别股票代码，已跳过: %s", code)
                continue
            filtered_codes.append(code)
        return filtered_codes
    
    def _latest_bar_date(self, code: str):
        db = getattr(self, "db", None)
        if db is None or not hasattr(db, "get_latest_data"):
            return None
        rows = db.get_latest_data(code, 1)
        return rows[0].date if rows else None

    def _ensure_data_fresh(self, code: str) -> Tuple[bool, Any, Any]:
        """Refresh once when the latest daily bar is too old.

        Returns (fresh, latest_date, target_date). Stale data must never be used
        to produce a trading call: a manual run in pre-market would otherwise
        mix months-old technicals with today's realtime price.
        """
        market = get_market_for_stock(code)
        target = get_market_today(market)
        config = getattr(self, "config", None)
        if config is None or getattr(self, "db", None) is None:
            # Test doubles / partial pipelines: fail open, the LLM call still has
            # its own data-missing warnings.
            return True, None, target
        max_gap = int(getattr(config, "analysis_max_staleness_days", 7))
        latest = self._latest_bar_date(code)
        if latest is not None and (target - latest).days <= max_gap:
            return True, latest, target

        logger.warning(
            "[%s] 行情数据过期（最新 %s，目标 %s），尝试强制刷新",
            code,
            latest,
            target,
        )
        self.fetch_and_save_stock_data(code, force_refresh=True)
        latest = self._latest_bar_date(code)
        fresh = latest is not None and (target - latest).days <= max_gap
        if not fresh:
            logger.error(
                "[%s] 强制刷新后仍然过期（最新 %s，目标 %s，阈值 %s 天）",
                code,
                latest,
                target,
                max_gap,
            )
        return fresh, latest, target

    def process_single_stock(
        self,
        code: str,
        skip_analysis: bool = False,
        single_stock_notify: bool = False,
        report_type: ReportType = ReportType.SIMPLE,
        analysis_query_id: Optional[str] = None,
    ) -> Optional[AnalysisResult]:
        """
        处理单只股票的完整流程

        包括：
        1. 获取数据
        2. 保存数据
        3. AI 分析
        4. 单股推送（可选，#55）

        此方法会被线程池调用，需要处理好异常

        Args:
            analysis_query_id: 查询链路关联 id
            code: 股票代码
            skip_analysis: 是否跳过 AI 分析
            single_stock_notify: 是否启用单股推送模式（每分析完一只立即推送）
            report_type: 报告类型枚举（从配置读取，Issue #119）

        Returns:
            AnalysisResult 或 None
        """
        logger.info(f"========== 开始处理 {code} ==========")

        if not self._filter_supported_stock_codes([code]):
            return None

        from src.services.history_loader import set_frozen_target_date, reset_frozen_target_date
        frozen_td = get_market_today(get_market_for_stock(code))
        token = set_frozen_target_date(frozen_td)

        try:
            # Step 1: 获取并保存数据
            success, error = self.fetch_and_save_stock_data(code)
            
            if not success:
                logger.warning(f"[{code}] 数据获取失败: {error}")
                # 即使获取失败，也尝试用已有数据分析

            # 过期保护：数据落后超过阈值时不生成交易建议，避免用旧 K 线判断
            fresh, latest_bar, target_date = self._ensure_data_fresh(code)
            if not fresh:
                max_gap = int(getattr(self.config, "analysis_max_staleness_days", 7))
                message = (
                    f"行情数据过期：最新 {latest_bar or '无'}，目标 {target_date}"
                    f"（阈值 {max_gap} 天），已停止分析"
                )
                logger.error("[%s] %s", code, message)
                display_name = STOCK_NAME_MAP.get(code) or code
                return AnalysisResult(
                    code=code,
                    name=display_name,
                    sentiment_score=50,
                    trend_prediction="未知",
                    operation_advice="观望",
                    confidence_level="低",
                    analysis_summary=message,
                    risk_warning="行情数据过期，已停止分析；请检查数据源或网络后重试。",
                    success=False,
                    error_message=message,
                    data_sources="stale-data-guard",
                )
            
            # Step 2: AI 分析
            if skip_analysis:
                logger.info(f"[{code}] 跳过 AI 分析（dry-run 模式）")
                if not success:
                    return None

                display_name = STOCK_NAME_MAP.get(code)
                if not display_name:
                    try:
                        display_name = self.fetcher_manager.get_stock_name(code, allow_realtime=False)
                    except Exception:
                        display_name = ""

                return AnalysisResult(
                    code=code,
                    name=display_name or code,
                    sentiment_score=0,
                    trend_prediction="未分析",
                    operation_advice="仅拉取数据",
                    analysis_summary="dry-run 模式，仅验证数据链路。",
                    data_sources="dry-run",
                    success=True,
                )
            
            effective_query_id = analysis_query_id or self.query_id or uuid.uuid4().hex
            result = self.analyze_stock(code, report_type, query_id=effective_query_id)
            
            if result:
                if not result.success:
                    logger.warning(
                        f"[{code}] 分析未成功: {result.error_message or '未知错误'}"
                    )
                else:
                    logger.info(
                        f"[{code}] 分析完成: {result.operation_advice}, "
                        f"评分 {result.sentiment_score}"
                    )

                # Issue #55: single-stock notify when called directly (not via run())
                if single_stock_notify and result.success:
                    try:
                        self._send_single_stock_notification(result, report_type)
                    except Exception as notify_err:
                        logger.error("[%s] 单股推送异常: %s", code, notify_err)
                
            return result

        except Exception as e:
            # 捕获所有异常，确保单股失败不影响整体
            logger.exception(f"[{code}] 处理过程发生未知异常: {e}")
            return None
        finally:
            reset_frozen_target_date(token)

    def run(
        self,
        stock_codes: Optional[List[str]] = None,
        dry_run: bool = False,
        send_notification: bool = True,
        merge_notification: bool = False
    ) -> List[AnalysisResult]:
        """
        运行完整的分析流程

        流程：
        1. 获取待分析的股票列表
        2. 使用线程池并发处理
        3. 收集分析结果
        4. 发送通知

        Args:
            stock_codes: 股票代码列表（可选，默认使用配置中的自选股）
            dry_run: 是否仅获取数据不分析
            send_notification: 是否发送推送通知
            merge_notification: 是否合并推送（跳过本次推送，由 main 层合并个股+大盘后统一发送，Issue #190）

        Returns:
            分析结果列表
        """
        start_time = time.time()
        
        # 使用配置中的股票列表
        if stock_codes is None:
            self.config.refresh_stock_list()
            stock_codes = self.config.stock_list
        
        if not stock_codes:
            logger.error("未配置自选股列表，请在 .env 文件中设置 STOCK_LIST")
            return []

        stock_codes = self._filter_supported_stock_codes(list(stock_codes))
        if not stock_codes:
            logger.error("未识别到有效股票代码，请检查 STOCK_LIST 或命令行传入的股票列表")
            return []
        
        logger.info(f"===== 开始分析 {len(stock_codes)} 只股票 =====")
        logger.info(f"股票列表: {', '.join(stock_codes)}")
        logger.info(f"并发数: {self.max_workers}, 模式: {'仅获取数据' if dry_run else '完整分析'}")
        
        # === 批量预取实时行情（优化：按市场分组，避免跨市场缓存污染）===
        # 只有股票数量 >= 5 时才进行预取，少量股票直接逐个查询更高效
        if len(stock_codes) >= 5:
            # Group by market so that A-share prefetch only triggers CN sources
            from collections import defaultdict
            market_groups: dict = defaultdict(list)
            for sc in stock_codes:
                mkt = get_market_for_stock(sc) or "cn"
                market_groups[mkt].append(sc)
            total_prefetch = 0
            for mkt, codes_in_market in market_groups.items():
                if len(codes_in_market) >= 2:
                    count = self.fetcher_manager.prefetch_realtime_quotes(codes_in_market)
                    total_prefetch += count
                    if count > 0:
                        logger.info(f"[{mkt}] 批量预取 {len(codes_in_market)} 只股票行情 → 缓存 {count} 条")
            if total_prefetch > 0:
                logger.info(f"已启用按市场分组预取架构：{len(stock_codes)} 只股票，{len(market_groups)} 个市场")

        # Issue #455: 预取股票名称，避免并发分析时显示「股票xxxxx」
        # dry_run 仅做数据拉取，不需要名称预取，避免额外网络开销
        if not dry_run:
            self.fetcher_manager.prefetch_stock_names(stock_codes, use_bulk=False)

        # 单股推送模式（#55）：从配置读取
        single_stock_notify = getattr(self.config, 'single_stock_notify', False)
        # Issue #119: 从配置读取报告类型
        report_type_str = getattr(self.config, 'report_type', 'simple').lower()
        if report_type_str == 'brief':
            report_type = ReportType.BRIEF
        elif report_type_str == 'full':
            report_type = ReportType.FULL
        else:
            report_type = ReportType.SIMPLE
        # Issue #128: 从配置读取分析间隔
        analysis_delay = getattr(self.config, 'analysis_delay', 0)
        llm_min_interval = max(0.0, float(getattr(self.config, 'llm_min_interval', 0.0) or 0.0))
        effective_submit_delay = max(0.0, float(analysis_delay or 0.0) - llm_min_interval)

        if single_stock_notify:
            logger.info(f"已启用单股推送模式：每分析完一只股票立即推送（报告类型: {report_type_str}）")
        
        results: List[AnalysisResult] = []
        
        # 使用线程池并发处理
        # 注意：max_workers 设置较低（默认3）以避免触发反爬
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_code = {}
            for idx, code in enumerate(stock_codes):
                future = executor.submit(
                    self.process_single_stock,
                    code,
                    skip_analysis=dry_run,
                    single_stock_notify=False,
                    report_type=report_type,  # Issue #119: 传递报告类型
                    analysis_query_id=uuid.uuid4().hex,
                )
                future_to_code[future] = code
                if idx < len(stock_codes) - 1 and effective_submit_delay > 0:
                    logger.debug(
                        f"等待 {effective_submit_delay} 秒后提交下一只股票 "
                        f"(analysis_delay={analysis_delay}, llm_min_interval={llm_min_interval})..."
                    )
                    time.sleep(effective_submit_delay)
            
            # 收集结果
            for future in as_completed(future_to_code):
                code = future_to_code[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if single_stock_notify and send_notification and not dry_run:
                            try:
                                self._send_single_stock_notification(result, report_type)
                            except Exception as e:
                                logger.error("[%s] 单股推送异常: %s", code, e)

                except Exception as e:
                    logger.error(f"[{code}] 任务执行失败: {e}")
        
        # 统计
        elapsed_time = time.time() - start_time
        
        # dry-run 模式下，数据获取成功即视为成功
        if dry_run:
            success_count = sum(1 for result in results if getattr(result, "success", False))
            fail_count = len(stock_codes) - success_count
        else:
            success_count = sum(1 for result in results if getattr(result, "success", True))
            fail_count = len(stock_codes) - success_count
        
        logger.info("===== 分析完成 =====")
        logger.info(f"成功: {success_count}, 失败: {fail_count}, 耗时: {elapsed_time:.2f} 秒")
        
        # 保存报告到本地文件（无论是否推送通知都保存）
        if results and not dry_run:
            self._save_local_report(results, report_type)

        # 发送通知（单股推送模式下跳过汇总推送，避免重复）
        if results and send_notification and not dry_run:
            if single_stock_notify:
                # 单股推送模式：只保存汇总报告，不再重复推送
                logger.info("单股推送模式：跳过汇总推送，仅保存报告到本地")
                self._send_notifications(results, report_type, skip_push=True)
            elif merge_notification:
                # 合并模式（Issue #190）：仅保存，不推送，由 main 层合并个股+大盘后统一发送
                logger.info("合并推送模式：跳过本次推送，将在个股+大盘复盘后统一发送")
                self._send_notifications(results, report_type, skip_push=True)
            else:
                self._send_notifications(results, report_type)
        
        return results
    
    def _save_local_report(
        self,
        results: List[AnalysisResult],
        report_type: ReportType = ReportType.SIMPLE,
    ) -> None:
        """保存分析报告到本地文件（与通知推送解耦）"""
        try:
            report = self._generate_aggregate_report(results, report_type)
            filepath = self.notifier.save_report_to_file(report)
            logger.info(f"决策仪表盘日报已保存: {filepath}")
        except Exception as e:
            logger.error(f"保存本地报告失败: {e}")

    def _send_notifications(
        self,
        results: List[AnalysisResult],
        report_type: ReportType = ReportType.SIMPLE,
        skip_push: bool = False,
    ) -> None:
        """
        发送分析结果通知
        
        生成决策仪表盘格式的报告
        
        Args:
            results: 分析结果列表
            skip_push: 是否跳过推送（仅保存到本地，用于单股推送模式）
        """
        try:
            logger.info("生成决策仪表盘日报...")
            report = self._generate_aggregate_report(results, report_type)
            
            # 跳过推送（单股推送模式 / 合并模式：报告已由 _save_local_report 保存）
            if skip_push:
                return
            
            # 推送通知
            if self.notifier.is_available():
                channels = self.notifier.get_available_channels()
                context_success = self.notifier.send_to_context(report)

                # Issue #455: Markdown 转图片（与 notification.send 逻辑一致）
                from src.md2img import markdown_to_image

                channels_needing_image = {
                    ch for ch in channels
                    if ch.value in self.notifier._markdown_to_image_channels
                }
                non_wechat_channels_needing_image = {
                    ch for ch in channels_needing_image if ch != NotificationChannel.WECHAT
                }

                def _get_md2img_hint() -> str:
                    try:
                        engine = getattr(get_config(), "md2img_engine", "wkhtmltoimage")
                    except Exception:
                        engine = "wkhtmltoimage"
                    return (
                        "npm i -g markdown-to-file" if engine == "markdown-to-file"
                        else "wkhtmltopdf (apt install wkhtmltopdf / brew install wkhtmltopdf)"
                    )

                image_bytes = None
                if non_wechat_channels_needing_image:
                    image_bytes = markdown_to_image(
                        report, max_chars=self.notifier._markdown_to_image_max_chars
                    )
                    if image_bytes:
                        logger.info(
                            "Markdown 已转换为图片，将向 %s 发送图片",
                            [ch.value for ch in non_wechat_channels_needing_image],
                        )
                    else:
                        logger.warning(
                            "Markdown 转图片失败，将回退为文本发送。请检查 MARKDOWN_TO_IMAGE_CHANNELS 配置并安装 %s",
                            _get_md2img_hint(),
                        )

                # 企业微信：只发精简版（平台限制）
                wechat_success = False
                if NotificationChannel.WECHAT in channels:
                    if report_type == ReportType.BRIEF:
                        dashboard_content = self.notifier.generate_brief_report(results)
                    else:
                        dashboard_content = self.notifier.generate_wechat_dashboard(results)
                    logger.info(f"企业微信仪表盘长度: {len(dashboard_content)} 字符")
                    logger.debug(f"企业微信推送内容:\n{dashboard_content}")
                    wechat_image_bytes = None
                    if NotificationChannel.WECHAT in channels_needing_image:
                        wechat_image_bytes = markdown_to_image(
                            dashboard_content,
                            max_chars=self.notifier._markdown_to_image_max_chars,
                        )
                        if wechat_image_bytes is None:
                            logger.warning(
                                "企业微信 Markdown 转图片失败，将回退为文本发送。请检查 MARKDOWN_TO_IMAGE_CHANNELS 配置并安装 %s",
                                _get_md2img_hint(),
                            )
                    use_image = self.notifier._should_use_image_for_channel(
                        NotificationChannel.WECHAT, wechat_image_bytes
                    )
                    if use_image:
                        wechat_success = self.notifier._send_wechat_image(wechat_image_bytes)
                    else:
                        wechat_success = self.notifier.send_to_wechat(dashboard_content)

                # 其他渠道：发完整报告（避免自定义 Webhook 被 wechat 截断逻辑污染）
                non_wechat_success = False
                stock_email_groups = getattr(self.config, 'stock_email_groups', []) or []
                for channel in channels:
                    if channel == NotificationChannel.WECHAT:
                        continue
                    if channel == NotificationChannel.FEISHU:
                        non_wechat_success = self.notifier.send_to_feishu(report) or non_wechat_success
                    elif channel == NotificationChannel.TELEGRAM:
                        use_image = self.notifier._should_use_image_for_channel(
                            channel, image_bytes
                        )
                        if use_image:
                            result = self.notifier._send_telegram_photo(image_bytes)
                        else:
                            result = self.notifier.send_to_telegram(report)
                        non_wechat_success = result or non_wechat_success
                    elif channel == NotificationChannel.EMAIL:
                        if stock_email_groups:
                            code_to_emails: Dict[str, Optional[List[str]]] = {}
                            for r in results:
                                if r.code not in code_to_emails:
                                    emails = []
                                    for stocks, emails_list in stock_email_groups:
                                        if r.code in stocks:
                                            emails.extend(emails_list)
                                    code_to_emails[r.code] = list(dict.fromkeys(emails)) if emails else None
                            emails_to_results: Dict[Optional[Tuple], List] = defaultdict(list)
                            for r in results:
                                recs = code_to_emails.get(r.code)
                                key = tuple(recs) if recs else None
                                emails_to_results[key].append(r)
                            for key, group_results in emails_to_results.items():
                                grp_report = self._generate_aggregate_report(group_results, report_type)
                                grp_image_bytes = None
                                if channel.value in self.notifier._markdown_to_image_channels:
                                    grp_image_bytes = markdown_to_image(
                                        grp_report,
                                        max_chars=self.notifier._markdown_to_image_max_chars,
                                    )
                                use_image = self.notifier._should_use_image_for_channel(
                                    channel, grp_image_bytes
                                )
                                receivers = list(key) if key is not None else None
                                if use_image:
                                    result = self.notifier._send_email_with_inline_image(
                                        grp_image_bytes, receivers=receivers
                                    )
                                else:
                                    result = self.notifier.send_to_email(
                                        grp_report, receivers=receivers
                                    )
                                non_wechat_success = result or non_wechat_success
                        else:
                            use_image = self.notifier._should_use_image_for_channel(
                                channel, image_bytes
                            )
                            if use_image:
                                result = self.notifier._send_email_with_inline_image(image_bytes)
                            else:
                                result = self.notifier.send_to_email(report)
                            non_wechat_success = result or non_wechat_success
                    elif channel == NotificationChannel.CUSTOM:
                        use_image = self.notifier._should_use_image_for_channel(
                            channel, image_bytes
                        )
                        if use_image:
                            result = self.notifier._send_custom_webhook_image(
                                image_bytes, fallback_content=report
                            )
                        else:
                            result = self.notifier.send_to_custom(report)
                        non_wechat_success = result or non_wechat_success
                    elif channel == NotificationChannel.PUSHPLUS:
                        non_wechat_success = self.notifier.send_to_pushplus(report) or non_wechat_success
                    elif channel == NotificationChannel.SERVERCHAN3:
                        non_wechat_success = self.notifier.send_to_serverchan3(report) or non_wechat_success
                    elif channel == NotificationChannel.DISCORD:
                        non_wechat_success = self.notifier.send_to_discord(report) or non_wechat_success
                    elif channel == NotificationChannel.PUSHOVER:
                        non_wechat_success = self.notifier.send_to_pushover(report) or non_wechat_success
                    elif channel == NotificationChannel.ASTRBOT:
                        non_wechat_success = self.notifier.send_to_astrbot(report) or non_wechat_success
                    else:
                        logger.warning(f"未知通知渠道: {channel}")

                success = wechat_success or non_wechat_success or context_success
                if success:
                    logger.info("决策仪表盘推送成功")
                else:
                    logger.warning("决策仪表盘推送失败")
            else:
                logger.info("通知渠道未配置，跳过推送")
                
        except Exception as e:
            import traceback
            logger.error(f"发送通知失败: {e}\n{traceback.format_exc()}")

    def _generate_aggregate_report(
        self,
        results: List[AnalysisResult],
        report_type: ReportType,
    ) -> str:
        """Generate aggregate report with backward-compatible notifier fallback."""
        generator = getattr(self.notifier, "generate_aggregate_report", None)
        if callable(generator):
            return generator(results, report_type)
        if report_type == ReportType.BRIEF and hasattr(self.notifier, "generate_brief_report"):
            return self.notifier.generate_brief_report(results)
        return self.notifier.generate_dashboard_report(results)

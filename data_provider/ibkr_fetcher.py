# -*- coding: utf-8 -*-
"""
===================================
IBKRFetcher - Interactive Brokers 数据源
===================================

数据来源：Interactive Brokers TWS API / IB Gateway
特点：券商级别数据质量，支持全球市场
定位：美股/港股/A股实时行情数据源

依赖：
    pip install ib_insync

配置：
    IBKR_HOST=127.0.0.1          # TWS/Gateway 地址
    IBKR_PORT=7497               # TWS=7497, Gateway=4001
    IBKR_CLIENT_ID=1             # 客户端ID
    IBKR_TIMEOUT=10              # 连接超时（秒）
    IBKR_PRIORITY=2              # 数据源优先级
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from threading import Lock

import pandas as pd

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, normalize_stock_code
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote, safe_float

logger = logging.getLogger(__name__)


class IBKRFetcher(BaseFetcher):
    """Interactive Brokers 数据源"""

    name = "IBKRFetcher"
    priority = int(os.getenv("IBKR_PRIORITY", "2"))

    def __init__(self):
        self._host = os.getenv("IBKR_HOST", "127.0.0.1")
        self._port = int(os.getenv("IBKR_PORT", "7497"))  # TWS=7497, Gateway=4001
        self._client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))
        self._timeout = int(os.getenv("IBKR_TIMEOUT", "10"))
        
        self._ib = None
        self._connected = False
        self._connection_lock = Lock()
        self._contract_cache: Dict[str, Any] = {}
        
        # 尝试导入 ib_insync
        try:
            from ib_insync import IB, Stock, util
            self._IB = IB
            self._Stock = Stock
            self._util = util
            self._available = True
        except ImportError:
            # IBKR 需要本地 TWS/Gateway + 可选依赖，未安装属于正常情况
            logger.debug("[IBKRFetcher] ib_insync 未安装，IBKR 数据源不参与链路")
            self._available = False

    def _ensure_connection(self) -> bool:
        """确保连接到 TWS/Gateway"""
        if not self._available:
            return False
        
        with self._connection_lock:
            if self._connected and self._ib and self._ib.isConnected():
                return True
            
            try:
                if self._ib is None:
                    self._ib = self._IB()
                
                if not self._ib.isConnected():
                    logger.info(
                        f"[IBKRFetcher] 连接到 IBKR: {self._host}:{self._port} (ClientID={self._client_id})"
                    )
                    self._ib.connect(
                        self._host,
                        self._port,
                        clientId=self._client_id,
                        timeout=self._timeout
                    )
                    self._connected = True
                    logger.info("[IBKRFetcher] ✅ 连接成功")
                
                return True
                
            except Exception as e:
                logger.error(f"[IBKRFetcher] 连接失败: {e}")
                self._connected = False
                return False

    def _disconnect(self):
        """断开连接"""
        with self._connection_lock:
            if self._ib and self._ib.isConnected():
                try:
                    self._ib.disconnect()
                    logger.info("[IBKRFetcher] 已断开连接")
                except Exception as e:
                    logger.debug(f"[IBKRFetcher] 断开连接时出错: {e}")
                finally:
                    self._connected = False

    @staticmethod
    def _parse_stock_code(stock_code: str) -> tuple:
        """
        解析股票代码，返回 (symbol, exchange, currency)
        
        示例：
            AAPL -> (AAPL, SMART, USD)
            0700.HK -> (0700, SEHK, HKD)
            600519 -> (600519, SEHKNTL, CNY)  # A股通过港股通
        """
        normalized = normalize_stock_code(stock_code).strip().upper()
        
        # 港股
        if normalized.endswith(".HK") or normalized.startswith("HK"):
            if normalized.endswith(".HK"):
                symbol = normalized[:-3]
            elif normalized.startswith("HK"):
                symbol = normalized[2:]
            else:
                symbol = normalized
            
            # 补齐5位数字
            symbol = symbol.zfill(5)
            return (symbol, "SEHK", "HKD")
        
        # A股（通过港股通或者直接访问）
        if normalized.startswith(("6", "0", "3")) and len(normalized) == 6:
            # A股代码，尝试通过 SEHKNTL (沪港通/深港通)
            return (normalized, "SEHKNTL", "CNY")
        
        # 美股（默认）
        return (normalized, "SMART", "USD")

    def _create_contract(self, stock_code: str):
        """创建 IBKR 合约对象"""
        if stock_code in self._contract_cache:
            return self._contract_cache[stock_code]
        
        symbol, exchange, currency = self._parse_stock_code(stock_code)
        
        contract = self._Stock(symbol, exchange, currency)
        self._contract_cache[stock_code] = contract
        
        logger.debug(
            f"[IBKRFetcher] 创建合约: {stock_code} -> {symbol} @ {exchange} ({currency})"
        )
        
        return contract

    def _compute_volume_ratio(
        self,
        contract,
        current_volume: Optional[int]
    ) -> Optional[float]:
        """
        计算量比：当前成交量 / 过去5日平均成交量
        """
        if not current_volume or current_volume <= 0:
            return None
        
        try:
            # 获取过去6天的日线数据（排除今天）
            end_date = datetime.now() - timedelta(days=1)
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr='6 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars or len(bars) < 2:
                return None
            
            # 取最近5天的成交量
            volumes = [bar.volume for bar in bars[-5:] if bar.volume > 0]
            
            if not volumes:
                return None
            
            avg_volume = sum(volumes) / len(volumes)
            
            if avg_volume <= 0:
                return None
            
            volume_ratio = round(float(current_volume) / avg_volume, 4)
            
            logger.debug(
                f"[IBKRFetcher] 量比计算: 当前={current_volume:,}, "
                f"5日均={avg_volume:,.0f}, 量比={volume_ratio}"
            )
            
            return volume_ratio
            
        except Exception as e:
            logger.debug(f"[IBKRFetcher] 量比计算失败: {e}")
            return None

    def _compute_turnover_rate(
        self,
        contract,
        current_volume: Optional[int]
    ) -> Optional[float]:
        """
        计算换手率：成交量 / 流通股本 × 100%
        """
        if not current_volume or current_volume <= 0:
            return None
        
        try:
            # 获取基本面数据
            self._ib.reqFundamentalData(contract, 'ReportSnapshot')
            self._ib.sleep(1)  # 等待数据返回
            
            # 尝试获取流通股本
            details = self._ib.reqContractDetails(contract)
            
            if not details:
                return None
            
            # 从合约详情中获取股本信息
            # 注意：IBKR 的股本数据可能不完整
            for detail in details:
                # 尝试从 fundamentalData 获取
                if hasattr(detail, 'fundamentalData'):
                    # 这里需要解析 XML 数据
                    # 简化处理：返回 None，让其他数据源补充
                    pass
            
            return None
            
        except Exception as e:
            logger.debug(f"[IBKRFetcher] 换手率计算失败: {e}")
            return None

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """获取实时行情"""
        if not self._ensure_connection():
            return None
        
        try:
            normalized = normalize_stock_code(stock_code).strip().upper()
            contract = self._create_contract(normalized)
            
            # 请求市场数据
            self._ib.qualifyContracts(contract)
            ticker = self._ib.reqMktData(contract, '', False, False)
            
            # 等待数据返回
            self._ib.sleep(2)
            
            # 取消订阅
            self._ib.cancelMktData(contract)
            
            # 检查数据有效性
            if not ticker or ticker.last <= 0:
                logger.warning(f"[IBKRFetcher] {stock_code} 未获取到有效价格")
                return None
            
            # 提取数据
            price = safe_float(ticker.last)
            volume = int(ticker.volume) if ticker.volume and ticker.volume > 0 else None
            open_price = safe_float(ticker.open)
            high = safe_float(ticker.high)
            low = safe_float(ticker.low)
            pre_close = safe_float(ticker.close)
            
            # 计算涨跌幅
            change_amount = None
            change_pct = None
            amplitude = None
            
            if pre_close and pre_close > 0 and price:
                change_amount = price - pre_close
                change_pct = (change_amount / pre_close) * 100
                
                if high and low:
                    amplitude = ((high - low) / pre_close) * 100
            
            # 计算量比
            volume_ratio = self._compute_volume_ratio(contract, volume)
            
            # 计算换手率（IBKR 数据可能不完整，返回 None 让其他源补充）
            turnover_rate = self._compute_turnover_rate(contract, volume)
            
            quote = UnifiedRealtimeQuote(
                code=normalized,
                name=ticker.contract.localSymbol or normalized,
                source=RealtimeSource.IBKR,
                price=price,
                change_pct=round(change_pct, 2) if change_pct else None,
                change_amount=round(change_amount, 4) if change_amount else None,
                volume=volume,
                amount=None,  # IBKR 不直接提供成交额
                volume_ratio=volume_ratio,
                turnover_rate=turnover_rate,
                amplitude=round(amplitude, 2) if amplitude else None,
                open_price=open_price,
                high=high,
                low=low,
                pre_close=pre_close,
            )
            
            logger.info(
                f"[IBKRFetcher] {stock_code} 实时行情: 价格={price}, "
                f"成交量={volume:,}, 量比={volume_ratio}"
            )
            
            return quote
            
        except Exception as e:
            logger.warning(f"[IBKRFetcher] {stock_code} 获取实时行情失败: {e}")
            return None

    def get_stock_name(self, stock_code: str) -> str:
        """获取股票名称"""
        if not self._ensure_connection():
            return ""
        
        try:
            contract = self._create_contract(stock_code)
            self._ib.qualifyContracts(contract)
            
            if contract.localSymbol:
                return contract.localSymbol
            
            return ""
            
        except Exception as e:
            logger.debug(f"[IBKRFetcher] 获取股票名称失败: {e}")
            return ""

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取历史日线数据"""
        if not self._ensure_connection():
            raise DataFetchError("[IBKRFetcher] 未连接到 IBKR")
        
        try:
            contract = self._create_contract(stock_code)
            self._ib.qualifyContracts(contract)
            
            # 计算时间跨度
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days + 1
            
            # 请求历史数据
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr=f'{days} D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                raise DataFetchError(f"[IBKRFetcher] {stock_code} 未获取到历史数据")
            
            # 转换为 DataFrame
            df = self._util.df(bars)
            
            if df.empty:
                raise DataFetchError(f"[IBKRFetcher] {stock_code} 历史数据为空")
            
            logger.info(f"[IBKRFetcher] {stock_code} 获取到 {len(df)} 条历史数据")
            
            return df
            
        except Exception as e:
            raise DataFetchError(f"[IBKRFetcher] {stock_code} 获取历史数据失败: {e}")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """标准化数据格式"""
        normalized = df.copy()
        
        # 重命名列
        normalized = normalized.rename(columns={
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        })
        
        # 确保日期格式
        if 'date' in normalized.columns:
            normalized['date'] = pd.to_datetime(normalized['date']).dt.strftime('%Y-%m-%d')
        
        # 计算涨跌幅
        normalized['pct_chg'] = normalized['close'].pct_change() * 100
        normalized['pct_chg'] = normalized['pct_chg'].fillna(0).round(2)
        
        # 成交额（IBKR 不提供，设为 None）
        normalized['amount'] = None
        
        # 确保所有标准列存在
        for column in STANDARD_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = None
        
        return normalized[STANDARD_COLUMNS]

    def __del__(self):
        """析构函数，断开连接"""
        self._disconnect()

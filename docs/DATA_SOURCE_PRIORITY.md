# 数据源优先级与降级策略

本文档详细说明股票智能分析系统的数据源优先级机制和故障降级策略。

## 目录

- [数据源优先级](#数据源优先级)
- [降级策略](#降级策略)
- [实时行情链路](#实时行情链路)
- [日线数据链路](#日线数据链路)
- [字段补充机制](#字段补充机制)
- [配置说明](#配置说明)

---

## 数据源优先级

系统在初始化时会按照以下优先级（Priority）对数据源进行排序，数字越小优先级越高：

| 优先级 | 数据源 | 说明 | 适用市场 |
|--------|--------|------|----------|
| **0** | EfinanceFetcher | 东方财富数据源（默认最高优先级） | A股 |
| **0** | TushareFetcher | Tushare Pro（仅当配置了有效 Token 时） | A股 |
| **1** | AkshareFetcher | AKShare 数据源 | A股 |
| **2** | PytdxFetcher | 通达信数据源（可配置服务器地址） | A股 |
| **2** | TushareFetcher | Tushare（未配置 Token 时） | A股 |
| **2** | IBKRFetcher | Interactive Brokers（需本地 TWS/Gateway） | 美股/港股/A股 |
| **3** | BaostockFetcher | Baostock 数据源 | A股 |
| **4** | YfinanceFetcher | Yahoo Finance（免费，全球市场） | 美股/港股/A股 |
| **-** | TwelveDataFetcher | Twelve Data API（需配置 API Key） | 美股/港股 |
| **-** | LongbridgeFetcher | 长桥证券 API（需配置 API Key） | 美股/港股 |
| **-** | FinnhubFetcher | Finnhub API（需配置 API Key，免费版 60 calls/min） | 美股 |
| **-** | AlphaVantageFetcher | AlphaVantage API（需配置 API Key，免费版 25 calls/day） | 美股 |

**注意**：
- TwelveData、Longbridge、Finnhub、AlphaVantage 不参与通用优先级排序，仅在美股/港股 API 优先链中使用
- Finnhub / AlphaVantage 是纯美股数据源；港股链路会通过 `supports_code()` 把它们过滤掉，不会空跑
- AlphaVantage 免费档日线只返回最近 100 个交易日且限 25 calls/day，定位是 Finnhub 之后的次级兜底
- IBKR 虽然优先级为 2，但需要本地运行 TWS 或 Gateway，不适用于 GitHub Actions 等云环境

---

## 降级策略

系统采用**自动故障切换（Failover）**机制，当某个数据源失败时，自动尝试下一个数据源。

### 核心原则

1. **按优先级顺序尝试**：从最高优先级开始，逐个尝试数据源
2. **异常捕获与记录**：捕获每个数据源的失败原因，记录详细日志
3. **自动切换**：单个数据源失败不影响整体流程，自动切换到下一个
4. **降级兜底**：所有数据源都失败时，返回 `None` 或抛出 `DataFetchError`
5. **市场专用链路**：美股/港股/美股指数使用专门的数据源链路

### 降级流程示例

```
尝试 EfinanceFetcher (P0)
  ↓ 失败：网络超时
尝试 AkshareFetcher (P1)
  ↓ 失败：数据格式错误
尝试 PytdxFetcher (P2)
  ↓ 成功 ✓
返回数据
```

如果所有数据源都失败：

```
尝试 EfinanceFetcher (P0) → 失败
尝试 AkshareFetcher (P1) → 失败
尝试 PytdxFetcher (P2) → 失败
尝试 TushareFetcher (P2) → 失败
尝试 BaostockFetcher (P3) → 失败
尝试 YfinanceFetcher (P4) → 失败
  ↓
返回 None 或抛出 DataFetchError
```

---

## 实时行情链路

实时行情数据的获取策略根据市场类型有所不同。

### A股实时行情

**配置项**：`REALTIME_SOURCE_PRIORITY`（默认：`efinance,akshare_em,akshare_sina,tencent`）

**降级顺序**（按配置）：

```
1. efinance        → EfinanceFetcher.get_realtime_quote()
2. akshare_em      → AkshareFetcher.get_realtime_quote(source="em")
3. akshare_sina    → AkshareFetcher.get_realtime_quote(source="sina")
4. tencent         → AkshareFetcher.get_realtime_quote(source="tencent")
5. tushare         → TushareFetcher.get_realtime_quote()（需 Token）
```

**特点**：
- 支持量比、换手率等高级字段
- 支持字段补充机制（见下文）
- 单个数据源失败不影响整体流程

### 美股实时行情

**降级顺序**：

```
1. TwelveDataFetcher   → 如果配置了 TWELVEDATA_API_KEY
2. LongbridgeFetcher   → 如果配置了 LONGBRIDGE_APP_KEY
3. FinnhubFetcher      → 如果配置了 FINNHUB_API_KEY
4. AlphaVantageFetcher → 如果配置了 ALPHAVANTAGE_API_KEY
5. IBKRFetcher         → 如果本地运行了 TWS/Gateway
6. YfinanceFetcher     → 永远作为最后兜底
```

**特点**：
- API 数据源（TwelveData/Longbridge/Finnhub/AlphaVantage/IBKR）优先
- YFinance 作为免费兜底方案
- 支持字段补充机制

### 港股实时行情

**降级顺序**：与美股相同，但 Finnhub / AlphaVantage 不参与（纯美股源，链路会按 `supports_code()` 自动跳过）

```
1. TwelveDataFetcher   → 如果配置了 TWELVEDATA_API_KEY
2. LongbridgeFetcher   → 如果配置了 LONGBRIDGE_APP_KEY
3. IBKRFetcher         → 如果本地运行了 TWS/Gateway
4. YfinanceFetcher     → 永远作为最后兜底
```

**港股实时缓存**：`AkshareFetcher` 的 `ak.stock_hk_spot_em()` 返回的是全市场快照，单次成本高。现在按进程内缓存复用（默认 20 分钟 TTL，失败结果 30 秒内不重试，并发冷启动只会真正拉取一次），因此同一次分析里多只港股不会重复拉全市场。

### 美股指数实时行情

**降级顺序**：

```
1. YfinanceFetcher     → 优先使用（免费且稳定）
2. LongbridgeFetcher   → 如果配置了 LONGBRIDGE_APP_KEY
```

**特点**：
- 美股指数（如 ^GSPC、^DJI、^IXIC）使用专门链路
- YFinance 对指数数据支持良好，优先使用

---

## 日线数据链路

日线数据的获取策略也根据市场类型有所不同。

### A股日线数据

**降级顺序**（按优先级）：

```
1. EfinanceFetcher (P0)
2. AkshareFetcher (P1)
3. PytdxFetcher (P2)
4. TushareFetcher (P2)
5. BaostockFetcher (P3)
6. YfinanceFetcher (P4)
```

**特点**：
- 排除 TwelveData 和 Longbridge（仅用于美股/港股）
- 按优先级顺序逐个尝试
- 所有数据源失败时抛出 `DataFetchError`

### 美股日线数据

**降级顺序**：

```
1. TwelveDataFetcher   → 如果配置了 TWELVEDATA_API_KEY
2. LongbridgeFetcher   → 如果配置了 LONGBRIDGE_APP_KEY
3. IBKRFetcher         → 如果本地运行了 TWS/Gateway
4. YfinanceFetcher     → 永远作为最后兜底
```

**特点**：
- API 数据源优先
- YFinance 作为免费兜底方案

### 港股日线数据

**降级顺序**：与美股相同

```
1. TwelveDataFetcher   → 如果配置了 TWELVEDATA_API_KEY
2. LongbridgeFetcher   → 如果配置了 LONGBRIDGE_APP_KEY
3. IBKRFetcher         → 如果本地运行了 TWS/Gateway
4. YfinanceFetcher     → 永远作为最后兜底
```

### 美股指数日线数据

**降级顺序**：

```
1. YfinanceFetcher     → 优先使用（免费且稳定）
2. LongbridgeFetcher   → 如果配置了 LONGBRIDGE_APP_KEY
```

---

## 字段补充机制

当第一个成功的数据源返回的数据中某些字段为空时，系统会尝试从后续数据源补充这些字段。

### 可补充字段

按重要性排序：

1. `volume_ratio` - 量比
2. `turnover_rate` - 换手率
3. `pe_ratio` - 市盈率
4. `pb_ratio` - 市净率
5. `total_mv` - 总市值
6. `circ_mv` - 流通市值
7. `amplitude` - 振幅

### 补充策略

1. **主数据源**：第一个成功返回基本价格字段的数据源成为主数据源
2. **检查缺失**：检查主数据源是否缺少上述补充字段
3. **尝试补充**：如果有缺失，继续尝试下一个数据源
4. **字段合并**：将后续数据源的非空字段合并到主数据源
5. **补充上限**：最多尝试 1 次补充（避免过多请求）
6. **提前返回**：如果所有字段都已填充，立即返回

### 补充流程示例

```
股票：600519（贵州茅台）

1. EfinanceFetcher 成功返回：
   - price: 1650.00 ✓
   - volume: 12345 ✓
   - volume_ratio: None ✗
   - turnover_rate: None ✗

2. 检测到缺失字段，继续尝试 AkshareFetcher

3. AkshareFetcher 返回：
   - price: 1651.00（忽略，使用主数据源）
   - volume: 12350（忽略，使用主数据源）
   - volume_ratio: 1.25 ✓
   - turnover_rate: 0.98 ✓

4. 合并字段：
   - price: 1650.00（来自 EfinanceFetcher）
   - volume: 12345（来自 EfinanceFetcher）
   - volume_ratio: 1.25（来自 AkshareFetcher）
   - turnover_rate: 0.98（来自 AkshareFetcher）

5. 所有字段已填充，返回合并后的数据
```

**日志示例**：

```
[实时行情] A股 600519 成功获取 (来源: EfinanceFetcher)
[实时行情] A股 600519 部分字段缺失，尝试从后续数据源补充
[实时行情] A股 600519 从 akshare_em 补充了缺失字段: ['volume_ratio', 'turnover_rate']
```

---

## 配置说明

### 实时行情配置

```bash
# 启用/禁用实时行情功能
ENABLE_REALTIME_QUOTE=true

# A股实时行情数据源优先级（逗号分隔）
# 有效值：efinance, akshare_em, akshare_sina, tencent, akshare_qq, tushare
REALTIME_SOURCE_PRIORITY=efinance,akshare_em,akshare_sina,tencent

# 启用/禁用批量预取（仅对全量数据源有效）
PREFETCH_REALTIME_QUOTES=true
```

### 美股/港股 API 配置

```bash
# Twelve Data API Key（美股/港股）
TWELVEDATA_API_KEY=your_api_key_here

# 长桥证券 API（美股/港股）
LONGBRIDGE_APP_KEY=your_app_key_here
LONGBRIDGE_APP_SECRET=your_app_secret_here
LONGBRIDGE_ACCESS_TOKEN=your_access_token_here

# Interactive Brokers（需本地 TWS/Gateway）
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
```

### Tushare 配置

```bash
# Tushare Pro Token（配置后优先级提升至 0）
TUSHARE_TOKEN=your_token_here
```

### 通达信配置

```bash
# 通达信服务器地址（可选）
PYTDX_HOST=119.147.212.81
PYTDX_PORT=7709
```

---

## 最佳实践

### 本地开发环境

**推荐配置**：

```bash
# A股：使用免费数据源
REALTIME_SOURCE_PRIORITY=efinance,akshare_em,akshare_sina

# 美股/港股：如果有 IBKR 账户
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# 或者使用免费的 YFinance（自动兜底）
# 无需额外配置
```

**优点**：
- 完全免费
- 支持量比、换手率等高级字段
- IBKR 提供实时数据（如果本地运行）

### GitHub Actions 环境

**推荐配置**：

```bash
# A股：使用免费数据源
REALTIME_SOURCE_PRIORITY=efinance,akshare_em,akshare_sina

# 美股/港股：使用 YFinance 兜底（无需配置）
# 注意：YFinance 不支持量比、换手率
```

**限制**：
- 不能使用 IBKR（需要本地 TWS/Gateway）
- 不能使用 TwelveData/Longbridge（需要付费 API Key）
- 美股/港股仅支持基础字段（价格、成交量等）

详见：[GitHub Actions 配置指南](./GITHUB_ACTIONS_SETUP.md)

### 生产服务器环境

**推荐配置**：

```bash
# A股：使用 Tushare Pro（如果有积分）
TUSHARE_TOKEN=your_token_here
REALTIME_SOURCE_PRIORITY=tushare,efinance,akshare_em

# 美股/港股：使用付费 API 或 IBKR
TWELVEDATA_API_KEY=your_api_key_here
# 或
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
```

**优点**：
- 数据质量更高
- 更稳定的服务
- 支持更多高级字段

---

## 故障排查

### 问题：所有数据源都失败

**可能原因**：
1. 网络连接问题
2. 数据源服务器维护
3. API 配置错误
4. 股票代码格式错误

**解决方法**：
1. 检查网络连接
2. 查看日志中的详细错误信息
3. 验证 API Key 配置
4. 确认股票代码格式正确（如：600519、hk00700、AAPL）

### 问题：缺少量比、换手率字段

**可能原因**：
1. 使用的数据源不支持这些字段（如 YFinance）
2. 字段补充机制未生效

**解决方法**：
1. 对于 A股：确保使用 EfinanceFetcher 或 AkshareFetcher
2. 对于美股/港股：使用 IBKR（需本地运行）或接受字段缺失
3. 查看日志确认是否尝试了字段补充

### 问题：IBKR 连接失败

**可能原因**：
1. TWS 或 Gateway 未运行
2. 端口配置错误
3. API 权限未启用

**解决方法**：
1. 确保 TWS 或 Gateway 正在运行
2. 检查 `IBKR_PORT` 配置（TWS: 7497, Gateway: 4002）
3. 在 TWS/Gateway 中启用 API 连接
4. 运行测试脚本：`python tests/test_ibkr_connection.py`

详见：[IBKR 配置指南](./IBKR_SETUP.md)

---

## 相关文档

- [数据源对比](./DATA_SOURCE_COMPARISON.md) - 各数据源功能对比
- [IBKR 配置指南](./IBKR_SETUP.md) - IBKR 详细配置说明
- [GitHub Actions 配置](./GITHUB_ACTIONS_SETUP.md) - 云环境配置指南
- [主文档](../README.md) - 项目总览

---

## 更新日志

- 2024-01-XX：初始版本，详细说明数据源优先级和降级策略
- 2024-01-XX：添加 IBKR 数据源说明
- 2024-01-XX：添加字段补充机制说明

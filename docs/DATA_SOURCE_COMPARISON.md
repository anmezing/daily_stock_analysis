# 数据源对比与选择指南

## 📊 完整对比表

| 数据源 | 市场 | 成交量 | 量比 | 换手率 | 费用 | 需要注册 | 配置难度 | 部署场景 |
|--------|------|--------|------|--------|------|----------|----------|----------|
| **EfinanceFetcher** | A股 | ✅ | ✅ | ✅ | 免费 | ❌ | ⭐ 简单 | 全部 |
| **AkshareFetcher** | A股 | ✅ | ✅ | ✅ | 免费 | ❌ | ⭐ 简单 | 全部 |
| **IBKRFetcher** | 全球 | ✅ | ✅ | ⚠️ | 免费 | ✅ | ⭐⭐⭐ 中等 | 本地/服务器 |
| **LongbridgeFetcher** | 美港 | ✅ | ✅ | ✅ | 免费 | ✅ | ⭐⭐ 一般 | 全部 |
| **TwelveDataFetcher** | 美港 | ⚠️ | ⚠️ | ❌ | 付费 | ✅ | ⭐ 简单 | 全部 |
| **YfinanceFetcher** | 全球 | ✅ | ❌ | ❌ | 免费 | ❌ | ⭐ 简单 | 全部 |
| **FinnhubFetcher** | 美股 | ✅ | ❌ | ❌ | 免费（60 calls/min） | ✅ | ⭐ 简单 | 全部 |
| **AlphaVantageFetcher** | 美股 | ✅ | ❌ | ❌ | 免费（25 calls/day） | ✅ | ⭐ 简单 | 全部 |
| **TushareFetcher** | A股 | ✅ | ⚠️ | ✅ | 付费 | ✅ | ⭐⭐ 一般 | 全部 |

**图例**：
- ✅ 完整支持
- ⚠️ 部分支持或需要额外配置
- ❌ 不支持

**部署场景说明**：
- **全部**：适用于本地、服务器、GitHub Actions、云函数等所有场景
- **本地/服务器**：仅适用于本地开发环境或自有服务器，不适用于 GitHub Actions

---

## 🎯 推荐方案

### 方案 1: 纯 A股用户（最简单）⭐⭐⭐⭐⭐
**适用场景**：所有部署环境（本地/服务器/GitHub Actions）

```bash
# .env 配置
ENABLE_REALTIME_QUOTE=true
REALTIME_SOURCE_PRIORITY=efinance,akshare_em,akshare_sina
```

**特点**：
- ✅ 完全免费
- ✅ 无需注册
- ✅ 量比、换手率完整支持
- ✅ 开箱即用
- ✅ 适用于所有部署场景

---

### 方案 2: A股 + 美股/港股（本地/服务器，有 IBKR 账户）⭐⭐⭐⭐⭐
**适用场景**：本地开发环境或自有服务器

```bash
# .env 配置
ENABLE_REALTIME_QUOTE=true

# A股使用免费数据源
REALTIME_SOURCE_PRIORITY=efinance,akshare_em

# 美股/港股使用 IBKR
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_PRIORITY=2
```

**特点**：
- ✅ 完全免费
- ✅ A股：量比+换手率
- ✅ 美股/港股：量比（换手率可能不可用）
- ✅ 券商级别数据质量
- ⚠️ 需要本地运行 TWS/Gateway
- ❌ 不适用于 GitHub Actions

**配置指南**：[docs/IBKR_SETUP.md](./IBKR_SETUP.md)

---

### 方案 3: A股 + 美股/港股（GitHub Actions/云函数）⭐⭐⭐⭐
**适用场景**：GitHub Actions、云函数、无状态环境

```bash
# .env 或 GitHub Secrets
ENABLE_REALTIME_QUOTE=true

# A股使用免费数据源（完美支持）
REALTIME_SOURCE_PRIORITY=efinance,akshare_em

# 美股/港股使用 YFinance（仅基础数据）
YFINANCE_PRIORITY=4
```

**特点**：
- ✅ 完全免费
- ✅ A股：量比+换手率
- ⚠️ 美股/港股：仅价格和成交量
- ❌ 美股/港股：无量比和换手率
- ✅ 适用于所有部署场景

---

### 方案 4: A股 + 美股/港股（GitHub Actions，需要量比）⭐⭐⭐
**适用场景**：GitHub Actions，需要美股/港股量比和换手率

```bash
# GitHub Secrets 配置
ENABLE_REALTIME_QUOTE=true
REALTIME_SOURCE_PRIORITY=efinance,akshare_em

# 使用 Twelve Data（付费）
TWELVEDATA_API_KEY=your_api_key
TWELVEDATA_PRIORITY=2
```

**特点**：
- ⚠️ 需要付费（$7.99/月起）
- ✅ A股：量比+换手率
- ✅ 美股/港股：量比（换手率需要 Pro 套餐）
- ✅ 适用于所有部署场景

---

## 📋 详细说明

### A股数据源

#### 1. EfinanceFetcher（推荐）
**优点**：
- 数据最全面（量比、换手率、市盈率、市值）
- 响应速度快
- 完全免费

**缺点**：
- 全量拉取容易被限流（系统已优化）

**配置**：
```bash
REALTIME_SOURCE_PRIORITY=efinance,...
```

---

#### 2. AkshareFetcher
**优点**：
- 多个数据源可选（东财/新浪/腾讯）
- 稳定性好
- 完全免费

**缺点**：
- 部分数据源可能被限流

**配置**：
```bash
REALTIME_SOURCE_PRIORITY=akshare_em,akshare_sina,tencent
```

---

### 美股/港股数据源

#### 1. IBKRFetcher（推荐，有 IBKR 账户）
**优点**：
- 券商级别数据质量
- 支持全球市场
- 自动计算量比
- 完全免费

**缺点**：
- 需要 IBKR 账户
- 需要本地运行 TWS/Gateway
- 配置相对复杂
- 换手率可能不可用

**配置**：
```bash
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
```

**详细指南**：[docs/IBKR_SETUP.md](./IBKR_SETUP.md)

---

#### 2. LongbridgeFetcher
**优点**：
- 数据质量高
- 支持量比和换手率
- 免费（需要注册）

**缺点**：
- ❌ **目前无法开户**
- 需要申请 OpenAPI 权限

**状态**：暂时不可用

---

#### 3. TwelveDataFetcher
**优点**：
- API 简单易用
- 支持 WebSocket

**缺点**：
- 免费套餐不支持成交量
- 需要付费套餐（$7.99/月起）

**配置**：
```bash
TWELVEDATA_API_KEY=your_key
```

---

#### 4. YfinanceFetcher（兜底）
**优点**：
- 完全免费
- 无需注册
- 全球市场支持

**缺点**：
- 不支持量比
- 不支持换手率
- 数据更新可能延迟

**配置**：无需配置，自动兜底

---

## 🚀 快速配置

### 配置 1: 最简单（仅 A股）

```bash
# .env
ENABLE_REALTIME_QUOTE=true
REALTIME_SOURCE_PRIORITY=efinance,akshare_em
```

### 配置 2: 完整支持（A股 + IBKR）

```bash
# .env
ENABLE_REALTIME_QUOTE=true
REALTIME_SOURCE_PRIORITY=efinance,akshare_em

# IBKR 配置
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
IBKR_PRIORITY=2
```

### 配置 3: 基础支持（A股 + YFinance）

```bash
# .env
ENABLE_REALTIME_QUOTE=true
REALTIME_SOURCE_PRIORITY=efinance,akshare_em
YFINANCE_PRIORITY=4
```

---

## 🧪 测试数据源

### 测试 A股
```bash
python -c "
from data_provider.base import DataFetcherManager
manager = DataFetcherManager()
quote = manager.get_realtime_quote('600519')
print(f'价格: {quote.price}')
print(f'量比: {quote.volume_ratio}')
print(f'换手率: {quote.turnover_rate}%')
"
```

### 测试 IBKR
```bash
python tests/test_ibkr_connection.py
```

---

## 💡 常见问题

### Q1: 我应该选择哪个数据源？

**A股用户**：直接使用 EfinanceFetcher 或 AkshareFetcher，无需额外配置。

**美股/港股用户**：
- 有 IBKR 账户 → 使用 IBKRFetcher
- 无 IBKR 账户 → 使用 YfinanceFetcher（仅基础数据）

### Q2: IBKR 配置复杂吗？

相对复杂，需要：
1. 安装 `ib_insync`
2. 运行 TWS 或 IB Gateway
3. 配置 API 权限

但配置一次后可以长期使用。

### Q3: 为什么 IBKR 的换手率是 None？

IBKR 的流通股本数据可能不完整，系统会返回 None，让其他数据源补充。

### Q4: 可以同时使用多个数据源吗？

可以！系统会自动 fallback：
1. 优先使用高优先级数据源
2. 失败后自动切换到下一个
3. 支持字段补充（如果第一个源缺少量比，会从第二个源补充）

---

## 📚 相关文档

- [IBKR 配置指南](./IBKR_SETUP.md)
- [数据源架构说明](../data_provider/README.md)
- [实时行情配置](../README.md#实时行情配置)

---

## 🎉 总结

**最佳实践**：
1. **A股**：使用 EfinanceFetcher（免费、完整）
2. **美股/港股**：
   - 有 IBKR → 使用 IBKRFetcher（免费、高质量）
   - 无 IBKR → 使用 YfinanceFetcher（免费、基础）
3. **多重 fallback**：配置多个数据源，提高稳定性

**不推荐**：
- ❌ 付费订阅 Twelve Data（除非有特殊需求）
- ❌ 等待长桥开户（目前不可用）

需要帮助？查看文档或提交 Issue！

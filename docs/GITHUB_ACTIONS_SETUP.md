# GitHub Actions 部署配置指南

## 📋 概述

本指南专门针对在 GitHub Actions 上运行股票分析系统的用户。

## ⚠️ 重要说明

### 不适用于 GitHub Actions 的数据源
- ❌ **IBKRFetcher**：需要本地运行 TWS/Gateway
- ❌ **LongbridgeFetcher**：目前无法开户

### ✅ 适用于 GitHub Actions 的数据源
- ✅ **EfinanceFetcher**：A股，完美支持量比和换手率
- ✅ **AkshareFetcher**：A股，完美支持量比和换手率
- ✅ **YfinanceFetcher**：全球市场，仅基础数据
- ✅ **TwelveDataFetcher**：美股/港股，需要付费

---

## 🚀 推荐配置

### 配置 1: 纯 A股用户（推荐）⭐⭐⭐⭐⭐

**无需任何额外配置！** 系统默认已完美支持。

#### GitHub Secrets 配置
```
# 无需添加任何 Secrets
# 系统会自动使用免费的 A股数据源
```

#### Workflow 环境变量（可选）
```yaml
# .github/workflows/daily_analysis.yml
env:
  ENABLE_REALTIME_QUOTE: true
  REALTIME_SOURCE_PRIORITY: efinance,akshare_em,akshare_sina
```

#### 支持的数据
- ✅ 成交量
- ✅ 量比
- ✅ 换手率
- ✅ 市盈率、市值

---

### 配置 2: A股 + 美股/港股（基础数据）⭐⭐⭐⭐

**适用于**：需要美股/港股价格和成交量，但不需要量比和换手率

#### GitHub Secrets 配置
```
# 无需添加任何 Secrets
```

#### Workflow 环境变量
```yaml
# .github/workflows/daily_analysis.yml
env:
  ENABLE_REALTIME_QUOTE: true
  REALTIME_SOURCE_PRIORITY: efinance,akshare_em
  YFINANCE_PRIORITY: 4
```

#### 支持的数据
**A股**：
- ✅ 成交量
- ✅ 量比
- ✅ 换手率

**美股/港股**：
- ✅ 成交量
- ❌ 量比
- ❌ 换手率

---

### 配置 3: A股 + 美股/港股（完整数据，付费）⭐⭐⭐

**适用于**：需要美股/港股的量比和换手率

#### GitHub Secrets 配置
```
TWELVEDATA_API_KEY = your_api_key_here
```

**获取 API Key**：
1. 注册 Twelve Data：https://twelvedata.com/register
2. 订阅套餐：https://twelvedata.com/pricing
   - Basic ($7.99/月)：支持成交量和量比
   - Pro ($49.99/月)：支持成交量、量比、换手率
3. 获取 API Key：https://twelvedata.com/account/api-keys

#### Workflow 环境变量
```yaml
# .github/workflows/daily_analysis.yml
env:
  ENABLE_REALTIME_QUOTE: true
  REALTIME_SOURCE_PRIORITY: efinance,akshare_em
  TWELVEDATA_API_KEY: ${{ secrets.TWELVEDATA_API_KEY }}
  TWELVEDATA_PRIORITY: 2
```

#### 支持的数据
**A股**：
- ✅ 成交量
- ✅ 量比
- ✅ 换手率

**美股/港股**：
- ✅ 成交量
- ✅ 量比（Basic 及以上）
- ⚠️ 换手率（Pro 及以上）

---

## 📝 完整的 Workflow 示例

### 示例 1: 纯 A股（推荐）

```yaml
name: Daily Stock Analysis

on:
  schedule:
    # 同时覆盖美东 10:00（开盘后 30 分钟）的夏令时 / 冬令时 UTC 映射
    # 工作流会按计划 cron 与美东夏令/冬令状态判断，避免备用 cron 重复运行。
    - cron: '0 14 * * 1-5'  # 夏令时：北京时间 22:00（美东 10:00）
    - cron: '0 15 * * 1-5'  # 冬令时：北京时间 23:00（美东 10:00）
  workflow_dispatch:

jobs:
  analyze:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run analysis
        env:
          # A股实时行情（完美支持）
          ENABLE_REALTIME_QUOTE: true
          REALTIME_SOURCE_PRIORITY: efinance,akshare_em,akshare_sina
          
          # 其他配置
          STOCK_LIST: 600519,000001,000858
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          
        run: |
          python main.py
```

---

### 示例 2: A股 + 美股/港股（使用 Twelve Data）

```yaml
name: Daily Stock Analysis

on:
  schedule:
    # 同时覆盖美东 10:00（开盘后 30 分钟）的夏令时 / 冬令时 UTC 映射
    - cron: '0 14 * * 1-5'
    - cron: '0 15 * * 1-5'
  workflow_dispatch:

jobs:
  analyze:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run analysis
        env:
          # 实时行情配置
          ENABLE_REALTIME_QUOTE: true
          REALTIME_SOURCE_PRIORITY: efinance,akshare_em
          
          # Twelve Data（美股/港股）
          TWELVEDATA_API_KEY: ${{ secrets.TWELVEDATA_API_KEY }}
          TWELVEDATA_PRIORITY: 2
          
          # 股票列表（A股 + 美股）
          STOCK_LIST: 600519,000001,AAPL,TSLA
          
          # AI 配置
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          
        run: |
          python main.py
```

---

## 🔧 GitHub Secrets 配置

### 必需的 Secrets
```
GEMINI_API_KEY = your_gemini_api_key
```

### 可选的 Secrets（根据需求）
```
# Twelve Data（美股/港股量比和换手率）
TWELVEDATA_API_KEY = your_twelvedata_key

# Tushare Pro（A股高级数据）
TUSHARE_TOKEN = your_tushare_token

# 通知配置
WECHAT_WEBHOOK_URL = your_wechat_webhook
FEISHU_WEBHOOK_URL = your_feishu_webhook
TELEGRAM_BOT_TOKEN = your_telegram_token
TELEGRAM_CHAT_ID = your_chat_id
```

### 配置方法
1. 进入仓库 → Settings → Secrets and variables → Actions
2. 点击 "New repository secret"
3. 添加 Name 和 Value
4. 点击 "Add secret"

---

## 📊 数据支持对比

| 市场 | 数据源 | 成交量 | 量比 | 换手率 | 费用 |
|------|--------|--------|------|--------|------|
| A股 | Efinance/Akshare | ✅ | ✅ | ✅ | 免费 |
| 美股/港股 | YFinance | ✅ | ❌ | ❌ | 免费 |
| 美股/港股 | Twelve Data Basic | ✅ | ✅ | ❌ | $7.99/月 |
| 美股/港股 | Twelve Data Pro | ✅ | ✅ | ✅ | $49.99/月 |

---

## 💡 常见问题

### Q1: 为什么不能使用 IBKR？
**A**: IBKR 需要本地运行 TWS 或 IB Gateway，GitHub Actions 是无状态的云环境，无法运行这些应用。

### Q2: 我只交易 A股，需要配置什么？
**A**: 无需任何配置！系统默认已完美支持 A股的量比和换手率。

### Q3: 美股/港股的量比和换手率重要吗？
**A**: 取决于你的交易策略：
- 如果主要看价格和趋势 → 免费的 YFinance 足够
- 如果需要量价分析 → 考虑订阅 Twelve Data

### Q4: Twelve Data 值得订阅吗？
**A**: 
- **Basic ($7.99/月)**：如果你需要美股/港股的量比，值得
- **Pro ($49.99/月)**：如果你需要完整的量价分析，值得
- **免费套餐**：不支持成交量，不推荐

### Q5: 可以在本地用 IBKR，GitHub Actions 用 YFinance 吗？
**A**: 可以！系统会自动 fallback：
- 本地：IBKR 可用时优先使用
- GitHub Actions：自动使用 YFinance

---

## 🎯 推荐配置总结

### 纯 A股用户
```yaml
env:
  ENABLE_REALTIME_QUOTE: true
  REALTIME_SOURCE_PRIORITY: efinance,akshare_em
```
**费用**：免费
**数据**：完整（量比+换手率）

### A股 + 美股/港股（基础）
```yaml
env:
  ENABLE_REALTIME_QUOTE: true
  REALTIME_SOURCE_PRIORITY: efinance,akshare_em
  YFINANCE_PRIORITY: 4
```
**费用**：免费
**数据**：A股完整，美股/港股基础

### A股 + 美股/港股（完整）
```yaml
env:
  ENABLE_REALTIME_QUOTE: true
  REALTIME_SOURCE_PRIORITY: efinance,akshare_em
  TWELVEDATA_API_KEY: ${{ secrets.TWELVEDATA_API_KEY }}
  TWELVEDATA_PRIORITY: 2
```
**费用**：$7.99/月起
**数据**：全部完整

---

## 📚 相关文档

- [数据源对比指南](./DATA_SOURCE_COMPARISON.md)
- [IBKR 配置指南](./IBKR_SETUP.md)（仅适用于本地/服务器）
- [Twelve Data 账户检查](../scripts/check_twelvedata_account.py)

---

## 🎉 快速开始

1. **Fork 本仓库**
2. **配置 GitHub Secrets**（至少添加 `GEMINI_API_KEY`）
3. **启用 GitHub Actions**
4. **运行 Workflow** 或等待定时触发

就这么简单！系统会自动使用最佳的数据源配置。

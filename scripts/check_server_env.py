#!/usr/bin/env python3
"""校验服务器 `.env` 是否能直接用于 `docker/docker-compose.server.yml`。

用途：服务器部署前跑一遍，确认「模板里的键都填了、填错了、还缺什么」，
避免启动容器后才发现连不上数据库或缺 LLM Key。

用法：
    python scripts/check_server_env.py                     # 读 ./.env，按 compose 内置 db 检查
    python scripts/check_server_env.py --env /opt/stock-analysis/.env
    python scripts/check_server_env.py --host              # 应用直接跑在宿主机（非 Docker）时用这个
    python scripts/check_server_env.py --strict            # 把 warning 也当作失败

退出码：0 无 error；1 存在 error（或 --strict 下存在 warning）。

注意：脚本只读，不会修改 `.env`；输出不会打印任何密钥原文。
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent

# 判断占位符：模板复制后没改的值
PLACEHOLDER_TOKENS = (
    "CHANGE_ME",
    "CHANGEME",
    "YOUR_KEY",
    "YOUR-KEY",
    "YOUR_TOKEN",
    "REPLACE_ME",
    "PLACEHOLDER",
    "<",
)

SECRET_KEY_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "WEBHOOK", "URL", "DSN")

# 通知渠道：(渠道名, 必填键)
# 只列真正会推送日报的渠道。钉钉 AppKey/AppSecret 只用于 Stream 模式回复对话，
# 不会推送日报，所以不在这里（见 check_notifications 的单独提示）。
NOTIFY_CHANNELS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("企业微信", ("WECHAT_WEBHOOK_URL",)),
    ("飞书", ("FEISHU_WEBHOOK_URL",)),
    ("Telegram", ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")),
    ("邮件", ("EMAIL_SENDER", "EMAIL_PASSWORD", "EMAIL_RECEIVERS")),
    ("自定义 Webhook", ("CUSTOM_WEBHOOK_URLS",)),
    ("Slack", ("SLACK_WEBHOOK_URL",)),
    ("Discord", ("DISCORD_WEBHOOK_URL",)),
    ("PushPlus", ("PUSHPLUS_TOKEN",)),
)

# DingTalk 应用凭证（仅 Stream 回复，不推送日报）
DINGTALK_STREAM_KEYS: Tuple[str, ...] = ("DINGTALK_APP_KEY", "DINGTALK_APP_SECRET")

LLM_KEY_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("GEMINI_API_KEYS", "GEMINI_API_KEY"),
    ("OPENAI_API_KEYS", "OPENAI_API_KEY", "AIHUBMIX_KEY"),
    ("ANTHROPIC_API_KEYS", "ANTHROPIC_API_KEY"),
    ("DEEPSEEK_API_KEYS", "DEEPSEEK_API_KEY"),
)

A_SHARE_PATTERN = re.compile(r"^(?:sh|sz|bj)?\d{6}$", re.IGNORECASE)


class Report:
    def __init__(self) -> None:
        self.entries: List[Tuple[str, str, str, str]] = []

    def add(self, level: str, key: str, message: str, hint: str = "") -> None:
        self.entries.append((level, key, message, hint))

    def by_level(self, level: str) -> List[Tuple[str, str, str, str]]:
        return [e for e in self.entries if e[0] == level]

    def render(self) -> None:
        order = ("ERROR", "WARN", "INFO")
        markers = {"ERROR": "[错误]", "WARN": "[警告]", "INFO": "[提示]"}
        for level in order:
            for _, key, message, hint in self.by_level(level):
                label = f"{key}: " if key else ""
                print(f"{markers[level]} {label}{message}")
                if hint:
                    print(f"         -> {hint}")
        counts = ", ".join(
            f"{name} {len(self.by_level(level))}" for level, name in (("ERROR", "错误"), ("WARN", "警告"), ("INFO", "提示"))
        )
        print(f"\n汇总：{counts}")


def parse_env_file(path: Path) -> Tuple[Dict[str, str], Dict[str, int], List[str], List[str]]:
    """返回 (values, line_numbers, duplicate_keys, inline_comment_keys)。"""
    values: Dict[str, str] = {}
    line_numbers: Dict[str, int] = {}
    duplicates: List[str] = []
    inline_comments: List[str] = []

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for index, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif re.search(r"\s#", value):
            # Docker Compose 不会剥离未加引号值里的行内注释，容易把注释带进配置
            inline_comments.append(key)
            value = re.split(r"\s#", value, maxsplit=1)[0].rstrip()
        if key in values:
            duplicates.append(key)
        values[key] = value
        line_numbers[key] = index
    return values, line_numbers, duplicates, inline_comments


def parse_template_keys(path: Path) -> Tuple[set[str], set[str]]:
    """返回 (模板中生效的键, 模板中出现过的所有键（含注释）)。"""
    if not path.exists():
        return set(), set()
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    active = {m.group(1) for m in re.finditer(r"^([A-Z][A-Z0-9_]+)=", text, flags=re.M)}
    mentioned = set(active) | {m.group(1) for m in re.finditer(r"^#\s*([A-Z][A-Z0-9_]+)=", text, flags=re.M)}
    return active, mentioned


def known_env_keys() -> set[str]:
    """仓库里被代码或文档实际读取的键，用于发现拼写错误。"""
    keys: set[str] = set()
    config_path = REPO_ROOT / "src" / "config.py"
    if config_path.exists():
        keys |= set(
            re.findall(
                r"os\.getenv\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]",
                config_path.read_text(encoding="utf-8", errors="ignore"),
            )
        )
    example_path = REPO_ROOT / ".env.example"
    if example_path.exists():
        keys |= {
            m.group(1)
            for m in re.finditer(r"^\s*#?\s*([A-Z][A-Z0-9_]+)=", example_path.read_text(encoding="utf-8", errors="ignore"), flags=re.M)
        }
    # Docker Compose / 运行时使用、但不在 config.py 里的键
    keys |= {"TZ", "API_PORT", "ENV_FILE", "COMPOSE_PROJECT_NAME", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "LLM_MIN_INTERVAL", "ENV"}
    return keys


def is_placeholder(value: str) -> bool:
    upper = value.strip().upper()
    if not upper:
        return False
    return any(token.upper() in upper for token in PLACEHOLDER_TOKENS)


def is_set(values: Dict[str, str], key: str) -> bool:
    return bool(values.get(key, "").strip())


def describe(value: str) -> str:
    """只暴露长度，避免把密钥打到终端或日志里。"""
    return f"已填（{len(value)} 字符）" if value else "空"


def split_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_db_url(url: str) -> Tuple[str, str, str]:
    """从 SQLAlchemy URL 里取 (host, password, database)，取不到就返回空串。"""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "", "", ""
    host = (parts.hostname or "").lower()
    password = unquote(parts.password) if parts.password else ""
    database = (parts.path or "").lstrip("/")
    return host, password, database


def check_database(values: Dict[str, str], report: Report, docker: bool) -> None:
    url = values.get("DATABASE_URL", "").strip()
    compose_db = is_set(values, "POSTGRES_PASSWORD")
    pg_password = values.get("POSTGRES_PASSWORD", "").strip()
    pg_db = values.get("POSTGRES_DB", "").strip() or "dsa"
    expected_url = f"postgresql+psycopg2://{values.get('POSTGRES_USER', '').strip() or 'dsa'}:<POSTGRES_PASSWORD>@db:5432/{pg_db}"

    if compose_db and is_placeholder(pg_password):
        report.add(
            "ERROR",
            "POSTGRES_PASSWORD",
            "还是模板占位符",
            "生成一个：openssl rand -hex 16，然后同步改 DATABASE_URL 里的密码。",
        )

    if not url:
        if compose_db:
            report.add(
                "ERROR",
                "DATABASE_URL",
                "未配置（compose 里的 db 服务已启用）",
                f"填 {expected_url}；不填会退回本地 SQLite，db 容器就白起了。",
            )
        else:
            report.add(
                "ERROR",
                "DATABASE_URL",
                "未配置",
                "容器部署请填 postgresql+psycopg2://user:pass@db:5432/dbname；不填会退回本地 SQLite。",
            )
        return

    lower = url.lower()
    if lower.startswith("postgres://"):
        report.add("ERROR", "DATABASE_URL", "使用了 SQLAlchemy 不支持的 postgres:// 前缀", "改成 postgresql+psycopg2:// 开头。")
    elif not lower.startswith(("postgresql", "mysql")):
        report.add("ERROR", "DATABASE_URL", "仅支持 postgresql:// 或 mysql://", "检查是否漏了协议前缀。")
    if "changeme" in lower or "change_me" in lower or "密码" in url:
        report.add("ERROR", "DATABASE_URL", "密码还是模板占位符", "替换成真实数据库密码。")
    if "supabase" in lower or "pooler.supabase" in lower:
        report.add(
            "WARN",
            "DATABASE_URL",
            "仍然指向 Supabase",
            "迁移到服务器后应改成服务器本机 PostgreSQL；Supabase 数据需要单独导入。",
        )

    host, password, database = parse_db_url(url)

    if compose_db:
        if host and host != "db":
            report.add(
                "WARN",
                "DATABASE_URL",
                f"主机名是 {host}，compose 里的 db 服务没被用上",
                "改成 @db:5432 使用容器内 PostgreSQL；确实要用宿主机库时，把 compose 的 db 服务删掉或停掉。",
            )
        if pg_password and password and not is_placeholder(pg_password) and password != pg_password:
            report.add(
                "ERROR",
                "DATABASE_URL",
                "密码与 POSTGRES_PASSWORD 不一致",
                "db 容器按 POSTGRES_PASSWORD 初始化，两处必须完全一致。",
            )
        if database and database != pg_db:
            report.add(
                "WARN",
                "DATABASE_URL",
                f"库名 {database} 与 POSTGRES_DB={pg_db} 不一致",
                "首次初始化只会创建 POSTGRES_DB 指定的库。",
            )
        if host == "db":
            report.add(
                "INFO",
                "DATABASE_URL",
                "使用独立 db 容器（dsa-net 网络别名 db）",
                "先 `docker network create dsa-net` 再起 docker/docker-compose.db.yml；备份见 docs/DEPLOY.md。",
            )
        return

    if host == "db":
        report.add(
            "ERROR",
            "DATABASE_URL",
            "主机名 db 只在 compose 网络内可解析，且未设置 POSTGRES_PASSWORD",
            "改用 compose 内置 db（补 POSTGRES_PASSWORD），或换成 host.docker.internal / 127.0.0.1。",
        )
        return

    if docker:
        if re.search(r"@(?:127\.0\.0\.1|localhost)[:/]", url):
            report.add(
                "ERROR",
                "DATABASE_URL",
                "容器内用 127.0.0.1 连不到宿主机 PostgreSQL",
                "改成 host.docker.internal（compose 已配置 extra_hosts）。",
            )
        if "host.docker.internal" not in lower:
            report.add(
                "INFO",
                "DATABASE_URL",
                "未使用 host.docker.internal",
                "若数据库在另一台机器，确认安全组只放行你的服务器 IP。",
            )
    else:
        if "host.docker.internal" in lower:
            report.add(
                "ERROR",
                "DATABASE_URL",
                "宿主机直跑不能用 host.docker.internal",
                "改成 127.0.0.1；或加 --docker 参数按容器环境检查。",
            )


def check_llm(values: Dict[str, str], report: Report) -> None:
    has_llm_key = False
    for group in LLM_KEY_GROUPS:
        filled = [key for key in group if is_set(values, key)]
        if not filled:
            continue
        has_llm_key = True
        for key in filled:
            if is_placeholder(values[key]):
                report.add("ERROR", key, "还是模板占位符", "填入真实 Key，或把该行删掉。")
    if not has_llm_key:
        report.add(
            "ERROR",
            "LLM",
            "没有任何 LLM Key",
            "至少配置 GEMINI_API_KEY / GEMINI_API_KEYS，否则报告会退化成模板。",
        )

    model = values.get("LITELLM_MODEL", "").strip()
    if model and "/" not in model:
        report.add(
            "WARN",
            "LITELLM_MODEL",
            f"缺少 provider 前缀：{model}",
            "写成 gemini/gemini-3-flash-preview 这种带 provider 的形式。",
        )
    if model and is_set(values, "GEMINI_MODEL"):
        report.add(
            "WARN",
            "GEMINI_MODEL",
            "同时设置了 GEMINI_MODEL 和 LITELLM_MODEL",
            "LITELLM_MODEL 会覆盖 GEMINI_MODEL；建议只保留 LITELLM_MODEL，避免两套配置打架。",
        )
    fallback = values.get("LITELLM_FALLBACK_MODELS", "").strip()
    if fallback and model and fallback == model:
        report.add("INFO", "LITELLM_FALLBACK_MODELS", "与主模型相同，等于没有降级", "填一个不同的备用模型。")
    for key, model_value in (("LITELLM_MODEL", model), ("LITELLM_FALLBACK_MODELS", fallback)):
        if model_value.startswith("gemini/") and "preview" in model_value and "FLASH" in model_value.upper():
            report.add(
                "INFO",
                key,
                f"使用 preview 模型：{model_value}",
                "preview 模型可能下线或限流更严；稳定后可换回正式版本。",
            )

    delay_raw = values.get("GEMINI_REQUEST_DELAY", "").strip()
    if delay_raw:
        try:
            delay = float(delay_raw)
        except ValueError:
            report.add("ERROR", "GEMINI_REQUEST_DELAY", f"不是数字：{delay_raw}", "例如 7。")
        else:
            if delay < 4:
                report.add(
                    "WARN",
                    "GEMINI_REQUEST_DELAY",
                    f"间隔 {delay:g} 秒偏短",
                    "Gemini 免费档约 10 RPM，建议 6–8 秒，否则会频繁 429。",
                )


def check_stocks(values: Dict[str, str], report: Report) -> None:
    raw = values.get("STOCK_LIST", "").strip()
    if not raw:
        report.add("ERROR", "STOCK_LIST", "未配置自选股", "例如 AAPL,MSFT,NVDA,0700。")
        return
    codes = split_list(raw)
    a_share = [code for code in codes if A_SHARE_PATTERN.fullmatch(code)]
    if a_share:
        report.add(
            "WARN",
            "STOCK_LIST",
            f"混入疑似 A 股代码：{', '.join(a_share)}",
            "只做美股/港股时用 AAPL/MSFT/0700 这类代码；A 股会走不同数据源。",
        )
    if len(codes) > 12:
        report.add(
            "INFO",
            "STOCK_LIST",
            f"共 {len(codes)} 只",
            "MAX_WORKERS=1 时单轮耗时随数量线性增长，首轮建议 3–5 只。",
        )


def normalize_region(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "")
    for separator in ("/", ",", ";"):
        normalized = normalized.replace(separator, "+")
    return normalized.strip("+")


def is_valid_region(value: str) -> bool:
    normalized = normalize_region(value)
    if normalized in {"cn", "hk", "us", "both", "all"}:
        return True
    parts = {part for part in normalized.split("+") if part}
    return bool(parts) and parts <= {"cn", "hk", "us"}


def check_market_review(values: Dict[str, str], report: Report) -> None:
    region = values.get("MARKET_REVIEW_REGION", "").strip().lower()
    stocks = split_list(values.get("STOCK_LIST", ""))
    has_a_share = any(A_SHARE_PATTERN.fullmatch(code) for code in stocks)
    has_us_hk = any(not A_SHARE_PATTERN.fullmatch(code) for code in stocks)

    if region and not is_valid_region(region):
        report.add(
            "ERROR",
            "MARKET_REVIEW_REGION",
            f"非法值：{region}",
            "可选 cn / hk / us / both / all，或组合 cn+hk / cn+us / hk+us（也接受 us/hk 这类写法）。",
        )
        return

    if not is_set(values, "MARKET_REVIEW_REGION") and is_set(values, "MARKET_REVIEW_ENABLED"):
        report.add("INFO", "MARKET_REVIEW_REGION", "未设置，默认 cn", "只做美股/港股时显式设为 us / hk / hk+us。")
        return

    if has_us_hk and not has_a_share:
        if normalize_region(region) == "cn":
            report.add(
                "WARN",
                "MARKET_REVIEW_REGION",
                "自选股只有美股/港股，但大盘复盘是 cn",
                "改成 us / hk / hk+us / all，或先 MARKET_REVIEW_ENABLED=false 缩短首轮时间。",
            )
        elif region:
            report.add("INFO", "MARKET_REVIEW_REGION", f"复盘范围 {region}，与自选股一致", "")


def check_schedule(values: Dict[str, str], report: Report) -> None:
    if values.get("SCHEDULE_ENABLED", "").strip().lower() == "true":
        report.add(
            "ERROR",
            "SCHEDULE_ENABLED",
            "分容器部署必须为 false",
            "server 容器用 --serve-only、analyzer 容器用 --schedule；设成 true 会让 server 进入定时模式，网页打不开。",
        )
    schedule_time = values.get("SCHEDULE_TIME", "").strip()
    if schedule_time and not re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", schedule_time):
        report.add("ERROR", "SCHEDULE_TIME", f"格式不是 HH:MM：{schedule_time}", "例如 10:00。")
    for key in ("SCHEDULE_RUN_IMMEDIATELY", "RUN_IMMEDIATELY"):
        if values.get(key, "").strip().lower() == "true":
            report.add(
                "WARN",
                key,
                "启动即跑一轮，容器每次重启都会重跑全量分析",
                "正式运行建议 false；需要立即跑一次时用 docker compose run --rm analyzer python main.py --stocks AAPL,MSFT。",
            )


def check_runtime(values: Dict[str, str], report: Report) -> None:
    delay_raw = values.get("ANALYSIS_DELAY", "").strip()
    if delay_raw:
        try:
            delay = float(delay_raw)
        except ValueError:
            report.add("ERROR", "ANALYSIS_DELAY", f"不是数字：{delay_raw}", "例如 0 或 2。")
        else:
            if delay > 5:
                report.add(
                    "WARN",
                    "ANALYSIS_DELAY",
                    f"每题额外等待 {delay:g} 秒",
                    "20 只股票就是 5 分钟以上纯等待，服务器建议 0–2。",
                )

    workers_raw = values.get("MAX_WORKERS", "").strip()
    if workers_raw:
        try:
            workers = int(workers_raw)
        except ValueError:
            report.add("ERROR", "MAX_WORKERS", f"不是整数：{workers_raw}", "建议 1。")
        else:
            if workers > 2:
                report.add(
                    "WARN",
                    "MAX_WORKERS",
                    f"并发 {workers}",
                    "Gemini 免费档建议 1，并发会放大 429 与重试耗时。",
                )
            if workers < 1:
                report.add("ERROR", "MAX_WORKERS", "必须 >= 1", "建议 1。")

    if not is_set(values, "TZ"):
        report.add("WARN", "TZ", "未设置时区", "日志与调度时间会按容器 UTC 解释，建议 TZ=Asia/Shanghai。")

    api_port = values.get("API_PORT", "").strip()
    if api_port:
        if not api_port.isdigit():
            report.add("ERROR", "API_PORT", f"不是整数：{api_port}", "例如 8000。")
        elif not (1 <= int(api_port) <= 65535):
            report.add("ERROR", "API_PORT", f"端口越界：{api_port}", "取 1024–65535。")
    webui_port = values.get("WEBUI_PORT", "").strip()
    if webui_port and webui_port != "8000":
        report.add(
            "WARN",
            "WEBUI_PORT",
            f"设为 {webui_port}，但 server 容器固定监听 8000",
            "容器内端口由 compose command 指定；映射用 API_PORT，WEBUI_PORT 保持 8000 避免误解。",
        )
    if values.get("ADMIN_AUTH_ENABLED", "").strip().lower() != "true":
        report.add(
            "WARN",
            "ADMIN_AUTH_ENABLED",
            "未开启登录认证",
            "域名对外访问必须 ADMIN_AUTH_ENABLED=true，否则任何人都能看到持仓与报告。",
        )
    if values.get("AGENT_MODE", "").strip().lower() == "true":
        report.add(
            "WARN",
            "AGENT_MODE",
            "开启后走 Agent 编排，与标准分析链是两套提示词路径",
            "近期日期/regime/基本面/数值校验修复主要在标准链；先用 false 验证输出。",
        )
        if values.get("AGENT_ARCH", "").strip().lower() == "multi":
            report.add("INFO", "AGENT_ARCH", "multi 需要更长时间与更多 LLM 调用", "首轮用 single 更容易定位瓶颈。")

    windows = values.get("BACKTEST_EXTRA_WINDOWS", "").strip()
    if windows:
        bad = [item for item in split_list(windows) if not item.isdigit() or int(item) <= 0]
        if bad:
            report.add("ERROR", "BACKTEST_EXTRA_WINDOWS", f"非法窗口：{', '.join(bad)}", "逗号分隔的正整数，例如 1,5,20。")
        elif len(split_list(windows)) > 3:
            report.add(
                "INFO",
                "BACKTEST_EXTRA_WINDOWS",
                f"评估 {len(split_list(windows))} 个窗口",
                "首轮建议只留 10，稳定后再开 1,5,20。",
            )

    timeout_raw = values.get("FUNDAMENTAL_STAGE_TIMEOUT_SECONDS", "").strip()
    if timeout_raw:
        try:
            timeout = float(timeout_raw)
        except ValueError:
            report.add("ERROR", "FUNDAMENTAL_STAGE_TIMEOUT_SECONDS", f"不是数字：{timeout_raw}", "美股建议 5。")
        else:
            if timeout < 3:
                report.add(
                    "WARN",
                    "FUNDAMENTAL_STAGE_TIMEOUT_SECONDS",
                    f"仅 {timeout:g} 秒",
                    "美股基本面走 SEC，容易超时；建议 5。",
                )


def check_notifications(values: Dict[str, str], report: Report) -> None:
    ready: List[str] = []
    partial: List[str] = []
    for name, keys in NOTIFY_CHANNELS:
        filled = [key for key in keys if is_set(values, key)]
        if len(filled) == len(keys):
            ready.append(name)
        elif filled:
            partial.append(f"{name}（缺 {', '.join(k for k in keys if k not in filled)}）")
    if ready:
        report.add("INFO", "通知", f"已启用：{', '.join(ready)}", "")
        return

    has_dingtalk = any(is_set(values, key) for key in DINGTALK_STREAM_KEYS)
    if has_dingtalk:
        report.add(
            "ERROR",
            "通知",
            "只配置了钉钉 AppKey/AppSecret，它不会推送每日报告",
            "钉钉应用凭证只用于 Stream 模式回复对话。要用钉钉收日报，把群机器人 webhook 填到 "
            "CUSTOM_WEBHOOK_URLS=https://oapi.dingtalk.com/robot/send?access_token=...；"
            "或改配 WECHAT_WEBHOOK_URL / FEISHU_WEBHOOK_URL / TELEGRAM_* / 邮件。",
        )
        return

    report.add(
        "WARN",
        "通知",
        "没有任何完整配置的通知渠道" + (f"；不完整：{'; '.join(partial)}" if partial else ""),
        "至少配一个，否则分析跑完也收不到报告。",
    )


def check_search_sources(values: Dict[str, str], report: Report) -> None:
    search_keys = ("TAVILY_API_KEYS", "BOCHA_API_KEYS", "BRAVE_API_KEYS", "SERPAPI_API_KEYS", "XAI_API_KEYS", "ANSPIRE_API_KEYS")
    filled = [key for key in search_keys if is_set(values, key)]
    placeholder = [key for key in filled if is_placeholder(values[key])]
    for key in placeholder:
        report.add("ERROR", key, "还是模板占位符", "填入真实 Key，或把该行删掉。")
    if not [key for key in filled if key not in placeholder]:
        if values.get("NEWS_RSS_ENABLED", "true").strip().lower() == "false":
            report.add(
                "ERROR",
                "搜索源",
                "没有任何搜索源：所有搜索 Key 为空且 NEWS_RSS_ENABLED=false",
                "至少启用 NEWS_RSS_ENABLED=true，或配置一个有效 Key。",
            )
        else:
            report.add(
                "INFO",
                "搜索源",
                "只用免费 RSS / 新闻源",
                "够用但覆盖面窄；确认某个 Key 有效后再逐个加，失效 Key 会拖慢整轮。",
            )
    if is_set(values, "TICKFLOW_API_KEY"):
        stocks = split_list(values.get("STOCK_LIST", ""))
        if stocks and not any(A_SHARE_PATTERN.fullmatch(code) for code in stocks):
            report.add(
                "INFO",
                "TICKFLOW_API_KEY",
                "已配置，但自选股里没有 A 股",
                "TickFlow 只服务 A 股大盘复盘/指数，纯美股/港股时留着不影响但没用。",
            )


def check_hygiene(path: Path, values: Dict[str, str], line_numbers: Dict[str, int], report: Report,
                  duplicates: List[str], inline_comments: List[str]) -> None:
    for key in duplicates:
        report.add("WARN", key, f"重复定义（最后一次在第 {line_numbers[key]} 行生效）", "删掉多余那行，避免改了不生效。")
    if inline_comments:
        report.add(
            "WARN",
            "行内注释",
            f"这些键的值后面跟了行内注释：{', '.join(inline_comments)}",
            "Compose 可能把注释当成值的一部分；把注释单独放一行。",
        )
    if not path.exists() or os.name == "nt":
        return
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if hasattr(stat, "S_ISREG") and stat.S_ISREG(mode) and mode & 0o077:
        report.add("WARN", ".env", "文件权限对其他用户可读", "执行 chmod 600 .env，里面是数据库密码和 API Key。")
    if path.suffix == "" and "ENV_FILE" not in values:
        report.add("INFO", ".env", "未设置 ENV_FILE，按默认路径读取", "")


def check_unknown_keys(values: Dict[str, str], template_mentioned: set[str], report: Report) -> None:
    known = known_env_keys() | template_mentioned
    unknown = sorted(key for key in values if key not in known)
    if unknown:
        shown = ", ".join(unknown[:12])
        more = f" 等 {len(unknown)} 个" if len(unknown) > 12 else ""
        report.add(
            "WARN",
            "未知键",
            f"代码与模板都没读取：{shown}{more}",
            "多半是拼写错误或旧名字，确认后删除，避免“填了但不生效”。",
        )


def check_template_coverage(values: Dict[str, str], active_template_keys: set[str], report: Report) -> None:
    missing = sorted(active_template_keys - set(values))
    if missing:
        shown = ", ".join(missing[:12])
        suffix = f" 等 {len(missing)} 个" if len(missing) > 12 else ""
        report.add(
            "INFO",
            "模板未填",
            f"{len(missing)} 个模板键不在 .env 里：{shown}{suffix}",
            "这些键有代码默认值，缺失通常不影响启动；但 Key、超时、调度相关建议显式写上。",
        )
    empty_required = [
        key
        for key in ("DATABASE_URL", "STOCK_LIST", "TZ", "LITELLM_MODEL", "GEMINI_REQUEST_DELAY", "SCHEDULE_TIME")
        if key in values and not values[key].strip()
    ]
    for key in empty_required:
        report.add("WARN", key, "键存在但值为空", "留空等于没配，确认是否故意如此。")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="校验服务器 .env 是否可用于 docker/docker-compose.server.yml")
    parser.add_argument("--env", default=".env", help="要校验的 .env 路径（默认 ./.env）")
    parser.add_argument("--template", default=str(REPO_ROOT / "docker" / "env.server.example"), help="服务器模板路径")
    parser.add_argument("--host", action="store_true", help="按宿主机直跑检查（默认按容器访问宿主机检查）")
    parser.add_argument("--strict", action="store_true", help="把 warning 也当作失败")
    args = parser.parse_args(argv)

    path = Path(args.env).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    print("== 服务器 .env 检查 ==")
    print(f"文件：{path}")
    print(f"场景：{'宿主机直跑' if args.host else 'Docker Compose（db + server + analyzer）'}")
    if not path.exists():
        print(f"\n[错误] 文件不存在：{path}")
        print("         -> 先执行：cp docker/env.server.example .env")
        return 1

    values, line_numbers, duplicates, inline_comments = parse_env_file(path)
    active_template_keys, template_mentioned = parse_template_keys(Path(args.template))
    print(f"解析到 {len(values)} 个生效键\n")

    report = Report()
    check_database(values, report, docker=not args.host)
    check_llm(values, report)
    check_stocks(values, report)
    check_market_review(values, report)
    check_schedule(values, report)
    check_runtime(values, report)
    check_notifications(values, report)
    check_search_sources(values, report)
    check_hygiene(path, values, line_numbers, report, duplicates, inline_comments)
    check_unknown_keys(values, template_mentioned, report)
    check_template_coverage(values, active_template_keys, report)

    # 不打印任何值，只打印“有没有填”
    for key in (
        "DATABASE_URL",
        "POSTGRES_PASSWORD",
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "FINNHUB_API_KEY",
        "FINNHUB_API_KEYS",
        "TWELVEDATA_API_KEY",
        "TWELVEDATA_API_KEYS",
        "TAVILY_API_KEYS",
    ):
        if key in values:
            print(f"  {key} = {describe(values[key])}")
    print()

    report.render()

    errors = len(report.by_level("ERROR"))
    warnings = len(report.by_level("WARN"))
    if errors:
        print("\n结论：存在错误，先修完再启动容器。")
        return 1
    if warnings and args.strict:
        print("\n结论：--strict 模式下警告也算失败。")
        return 1
    if warnings:
        print("\n结论：可以启动，但建议先看上面的警告。")
        return 0
    print("\n结论：配置检查通过，可以启动容器。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

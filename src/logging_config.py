# -*- coding: utf-8 -*-
"""
===================================
日志配置模块 - 统一的日志系统初始化
===================================

职责：
1. 提供统一的日志格式和配置常量
2. 支持控制台 + 文件（常规/调试）三层日志输出
3. 自动降低第三方库日志级别
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

from src.utils.log_redact import redact_secrets


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(pathname)s:%(lineno)d | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RelativePathFormatter(logging.Formatter):
    """自定义 Formatter，输出相对路径而非绝对路径，并对密钥统一脱敏。"""

    def __init__(self, fmt=None, datefmt=None, relative_to=None):
        super().__init__(fmt, datefmt)
        self.relative_to = Path(relative_to) if relative_to else Path.cwd()

    def format(self, record):
        # 将绝对路径转为相对路径
        try:
            record.pathname = str(Path(record.pathname).relative_to(self.relative_to))
        except ValueError:
            # 如果无法转换为相对路径，保持原样
            pass
        # 统一脱敏：DSN 密码、URL 的 apikey/token、鉴权头、Telegram bot token 等。
        # 放在 Formatter 这一层，任何日志（含第三方库与 traceback）都逃不掉。
        return redact_secrets(super().format(record))


# 默认需要降低日志级别的第三方库
DEFAULT_QUIET_LOGGERS = [
    'urllib3',
    'sqlalchemy',
    'google',
    'httpx',
    'httpcore',
    'yfinance',
    'peewee',
    'charset_normalizer',
    'asyncio',
    'python_multipart',
    'apscheduler',
]

# 这些库的 INFO/DEBUG 只在调试日志里保留，控制台和常规日志只放 ERROR
NOISY_THIRD_PARTY_PREFIXES = (
    'litellm',
    'openai',
    'anthropic',
    'httpx',
    'httpcore',
    'urllib3',
    'yfinance',
    'peewee',
    'pandas',
)


class ThirdPartyNoiseFilter(logging.Filter):
    """把第三方库（LiteLLM / HTTP 客户端 / 行情库）的 INFO 噪音挡在控制台外。

    调试日志 handler 不挂这个过滤器，需要排查时仍然能看到完整输出。
    """

    def __init__(self, prefixes=NOISY_THIRD_PARTY_PREFIXES):
        super().__init__()
        self.prefixes = tuple(prefix.lower() for prefix in prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name.lower()
        if any(name.startswith(prefix) for prefix in self.prefixes):
            return record.levelno >= logging.ERROR
        return True


def _silence_existing_litellm_loggers() -> None:
    """LiteLLM 会创建 ``LiteLLM``、``LiteLLM Router`` 等独立 logger。

    它们不是 ``litellm`` 的子 logger，逐个按名字前缀压到 ERROR，
    避免 ``LITELLM_LOG=ERROR`` 不生效时控制台被路由/重试日志刷屏。
    """
    for name in ('litellm', 'LiteLLM', 'LiteLLM Router', 'LiteLLM Proxy', 'LiteLLM Fallback'):
        logging.getLogger(name).setLevel(logging.ERROR)
    for name in list(logging.root.manager.loggerDict):
        lowered = name.lower()
        if lowered.startswith('litellm') or lowered.startswith('openai') or lowered.startswith('anthropic'):
            logging.getLogger(name).setLevel(logging.ERROR)


IGNORED_WARNING_MARKERS = (
    "inplace keyword",
    "PDEP-8",
    # LiteLLM Router 自己的 async 回调 bug，与我们无关，且会在每次冷却时重复刷屏
    "was never awaited",
)


def _should_ignore_warning(message: object) -> bool:
    """行情库的 pandas inplace 弃用警告，对我们没有可操作性。"""
    text = str(message)
    return any(marker in text for marker in IGNORED_WARNING_MARKERS)


def _configure_third_party_warnings() -> None:
    """压掉行情库触发的 pandas 弃用警告。

    yfinance 仍在使用 ``DataFrame.sort_index(inplace=True)`` / ``rename(inplace=True)``
    这类写法，每次拉行情都会刷屏。分层处理：

    1. ``filterwarnings`` 按模块/消息忽略（cost 最低）；
    2. 必须再接管 ``warnings.showwarning``：某些库（pandas/yfinance）在运行期
       调用 ``simplefilter("always")``，会插到过滤器链最前面并**压过**我们注册的
       规则，此时只有接管 showwarning 才能拦住；
    3. ``logging.captureWarnings`` 让警告走日志系统，便于统一格式。
    """
    import warnings

    warnings.filterwarnings('ignore', module=r'yfinance(\..*)?')
    warnings.filterwarnings('ignore', message=r'.*inplace keyword.*')
    warnings.filterwarnings('ignore', message=r'.*Pandas4Warning.*')

    logging.captureWarnings(True)
    previous_showwarning = warnings.showwarning

    def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
        if _should_ignore_warning(message):
            return
        previous_showwarning(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _filtered_showwarning


def setup_logging(
    log_prefix: str = "app",
    log_dir: str = "./logs",
    console_level: Optional[int] = None,
    debug: bool = False,
    extra_quiet_loggers: Optional[List[str]] = None,
) -> None:
    """
    统一的日志系统初始化

    配置三层日志输出：
    1. 控制台：根据 debug 参数或 console_level 设置级别
    2. 常规日志文件：INFO 级别，10MB 轮转，保留 5 个备份
    3. 调试日志文件：DEBUG 级别，50MB 轮转，保留 3 个备份

    Args:
        log_prefix: 日志文件名前缀（如 "api_server" -> api_server_20240101.log）
        log_dir: 日志文件目录，默认 ./logs
        console_level: 控制台日志级别（可选，优先于 debug 参数）
        debug: 是否启用调试模式（控制台输出 DEBUG 级别）
        extra_quiet_loggers: 额外需要降低日志级别的第三方库列表
    """
    # 确定控制台日志级别
    if console_level is not None:
        level = console_level
    else:
        level = logging.DEBUG if debug else logging.INFO

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 日志文件路径（按日期分文件）
    today_str = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"{log_prefix}_{today_str}.log"
    debug_log_file = log_path / f"{log_prefix}_debug_{today_str}.log"

    # 配置根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 根 logger 设为 DEBUG，由 handler 控制输出级别

    # 清除已有 handler，避免重复添加
    if root_logger.handlers:
        root_logger.handlers.clear()
    # 创建相对路径 Formatter（相对于项目根目录）
    project_root = Path.cwd()
    rel_formatter = RelativePathFormatter(
        LOG_FORMAT, LOG_DATE_FORMAT, relative_to=project_root
    )
    # Handler 1: 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(rel_formatter)
    root_logger.addHandler(console_handler)
    # Handler 2: 常规日志文件（INFO 级别，10MB 轮转）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(rel_formatter)
    root_logger.addHandler(file_handler)

    # Handler 3: 调试日志文件（DEBUG 级别，包含所有详细信息）
    debug_handler = RotatingFileHandler(
        debug_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=3,
        encoding='utf-8'
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(rel_formatter)
    root_logger.addHandler(debug_handler)

    # 降低第三方库的日志级别
    quiet_loggers = DEFAULT_QUIET_LOGGERS.copy()
    if extra_quiet_loggers:
        quiet_loggers.extend(extra_quiet_loggers)

    for logger_name in quiet_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # 控制台与常规日志只保留第三方库的 ERROR，调试日志仍保留全部
    noise_filter = ThirdPartyNoiseFilter()
    console_handler.addFilter(noise_filter)
    file_handler.addFilter(noise_filter)
    _silence_existing_litellm_loggers()
    _configure_third_party_warnings()

    # 输出初始化完成信息（使用相对路径）
    try:
        rel_log_path = log_path.resolve().relative_to(project_root)
    except ValueError:
        rel_log_path = log_path

    try:
        rel_log_file = log_file.resolve().relative_to(project_root)
    except ValueError:
        rel_log_file = log_file

    try:
        rel_debug_log_file = debug_log_file.resolve().relative_to(project_root)
    except ValueError:
        rel_debug_log_file = debug_log_file

    logging.info(f"日志系统初始化完成，日志目录: {rel_log_path}")
    logging.info(f"常规日志: {rel_log_file}")
    logging.info(f"调试日志: {rel_debug_log_file}")

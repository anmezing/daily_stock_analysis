# -*- coding: utf-8 -*-
"""日志脱敏工具。

第三方库（requests / httpx / LiteLLM / Telegram）会把完整 URL、DSN、鉴权头
写进异常消息，一旦落到日志文件就等于把密钥泄露。这里提供统一的脱敏函数，
由日志 Formatter 统一调用，保证任何一条日志（含 traceback）都不会带出密钥。

覆盖：
- 连接串：``postgresql://user:password@host`` -> ``postgresql://user:***@host``
- 查询参数：``?apikey=xxx`` / ``&token=xxx`` / ``password=xxx`` ...
- 鉴权头：``Authorization: Bearer xxx`` / ``apikey xxx``
- Telegram bot URL：``/bot<id>:<token>/``
- 常见 Key 前缀：Google ``AIza...``、OpenAI ``sk-...``、Slack ``xox*-``、GitHub ``ghp_``、AWS ``AKIA``
"""

import re
from typing import Optional

# 连接串里的密码段：scheme://user:password@host
_DSN_PATTERN = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^:/@\s]+):([^@\s/]+)@")

# URL / 表单查询参数，以及日志里常见的 key=value 写法
_SECRET_PARAMS = (
    "apikey|api_key|api-key|access_token|refresh_token|id_token|token|"
    "secret|client_secret|app_secret|password|passwd|pwd|sign|signature|auth|key"
)
_SECRET_PARAM_PATTERN = re.compile(
    rf"(?i)\b({_SECRET_PARAMS})\s*=\s*(?!\*\*\*)([^\s&,'\"<>]+)"
)

# 鉴权头 / inline 头：Authorization: Bearer xxx、apikey xxx、token=xxx 之外的写法
_AUTH_HEADER_PATTERN = re.compile(
    r"(?i)((?:authorization|proxy-authorization)\s*[:=]\s*)(?:bearer|basic|apikey|token)?\s*([^\s'\"，,]+)"
)
_INLINE_APIKEY_PATTERN = re.compile(r"(?i)\b(apikey|api-key|api_key)\s+([A-Za-z0-9_\-]{8,})")

# Telegram: https://api.telegram.org/bot<digits>:<token>/sendMessage
_TELEGRAM_BOT_PATTERN = re.compile(r"(?i)(/bot)(\d+:[A-Za-z0-9_\-]{20,})")

# 常见供应商 Key 前缀（长度阈值避免误伤普通单词）
_KNOWN_KEY_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{35,}"),         # Google
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),          # OpenAI / LiteLLM 兼容
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),   # Slack
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),      # GitHub
    re.compile(r"AKIA[0-9A-Z]{16}"),                # AWS
)

_REDACTED = "***"


def redact_secrets(text: Optional[str]) -> str:
    """把文本里的密钥、密码、令牌替换成 ***；None 返回空串。"""
    if text is None:
        return ""
    value = str(text)
    if not value:
        return value

    value = _DSN_PATTERN.sub(r"\1:***@", value)
    value = _TELEGRAM_BOT_PATTERN.sub(r"\1***", value)
    value = _SECRET_PARAM_PATTERN.sub(r"\1=***", value)
    value = _AUTH_HEADER_PATTERN.sub(r"\1***", value)
    value = _INLINE_APIKEY_PATTERN.sub(r"\1 ***", value)
    for pattern in _KNOWN_KEY_PATTERNS:
        value = pattern.sub(_REDACTED, value)
    return value


def mask_database_url(url: Optional[str]) -> str:
    """只保留数据库 DSN 的用户名与主机，隐藏密码。"""
    if not url:
        return ""
    return redact_secrets(str(url))


if __name__ == "__main__":  # pragma: no cover - 手动排查用
    import sys

    print(redact_secrets(" ".join(sys.argv[1:])))

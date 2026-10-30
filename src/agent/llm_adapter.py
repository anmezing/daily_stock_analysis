# -*- coding: utf-8 -*-
"""
Multi-provider LLM Tool-Calling Adapter.

Normalizes function-calling / tool-use across all providers into a unified
interface consumed by the AgentExecutor, via LiteLLM.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import litellm
    from litellm import Router
except Exception:  # pragma: no cover - optional dependency
    litellm = None
    Router = None

from src.config import (
    extra_litellm_params,
    get_api_keys_for_model,
    get_config,
    get_configured_llm_models,
    get_effective_agent_models_to_try,
    get_effective_agent_primary_model,
)
from src.core.llm_rate_limiter import get_llm_rate_limiter, get_rate_limit_bucket
from src.utils.json_sanitize import sanitize_non_finite

logger = logging.getLogger(__name__)


# ============================================================
# Unified response types
# ============================================================

@dataclass
class ToolCall:
    """A single tool call requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]
    thought_signature: Optional[str] = None


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: Optional[str] = None          # text response (final answer)
    tool_calls: List[ToolCall] = field(default_factory=list)  # tool calls to execute
    reasoning_content: Optional[str] = None  # Chain-of-thought (CoT) from DeepSeek thinking mode; must be passed back in multi-turn assistant messages; None for other providers
    usage: Dict[str, Any] = field(default_factory=dict)       # token usage info
    provider: str = ""                     # which provider handled this call
    model: str = ""                        # full model name used (e.g. gemini/gemini-2.0-flash), for report meta
    raw: Any = None                        # raw provider response for debugging


# Models that auto-return reasoning_content; do NOT send extra_body (may cause 400).
_AUTO_THINKING_MODELS: List[str] = ["deepseek-reasoner", "deepseek-r1", "qwq"]

# Models that need explicit opt-in via extra_body; payload decoupled from model name.
_OPT_IN_THINKING_MODELS: Dict[str, dict] = {
    "deepseek-chat": {"thinking": {"type": "enabled"}},
    "deepseek-v4-pro": {"thinking": {"type": "enabled"}},
}

# Models that require a top-level reasoning_effort parameter.
_REASONING_EFFORT_MODELS: Dict[str, str] = {
    "deepseek-v4-pro": "high",
}

# Custom model pricing for models not in LiteLLM's built-in price list
# Official MiniMax pricing: https://platform.minimax.io/docs/guides/pricing-paygo
# - MiniMax-M2.7 / M2.5: $0.3/M input tokens, $1.2/M output tokens
_CUSTOM_MODEL_PRICING: Dict[str, dict] = {
    "MiniMax-M2.7": {
        "supports_function_calling": True,
        "supports_vision": False,
        "supports_audio_input": False,
        "supports_audio_output": False,
        "context_window": 100000,
        "max_tokens": 10000,
        "input_cost_per_token": 0.0000003,   # $0.3 / 1M tokens
        "output_cost_per_token": 0.0000012,   # $1.2 / 1M tokens
    },
    "MiniMax-M2.5": {
        "supports_function_calling": True,
        "supports_vision": False,
        "supports_audio_input": False,
        "supports_audio_output": False,
        "context_window": 100000,
        "max_tokens": 10000,
        "input_cost_per_token": 0.0000003,   # $0.3 / 1M tokens
        "output_cost_per_token": 0.0000012,   # $1.2 / 1M tokens
    },
}


def _safe_litellm_exc_type(name: str):
    """Return a litellm exception type when available; otherwise None."""
    if litellm is None:
        return None
    exc_type = getattr(litellm, name, None)
    return exc_type if isinstance(exc_type, type) and issubclass(exc_type, BaseException) else None


def _model_matches(model: str, entries: List[str]) -> bool:
    """Check if model name matches any entry (exact or prefix with version suffix)."""
    if not model:
        return False
    m = model.lower().strip()
    for e in entries:
        if m == e or m.startswith(e + "-"):
            return True
    return False


def _get_opt_in_payload(model: str, opt_in: Dict[str, dict]) -> Optional[dict]:
    """Return extra_body payload for opt-in thinking models, or None."""
    if not model:
        return None
    m = model.lower().strip()
    for key, payload in opt_in.items():
        if m == key or m.startswith(key + "-"):
            return payload
    return None


def get_thinking_extra_body(model: str) -> Optional[dict]:
    """Return extra_body for thinking mode, or None.

    - Auto-thinking models (_AUTO_THINKING_MODELS: deepseek-reasoner, deepseek-r1, qwq):
      These models automatically return reasoning_content in API responses; sending
      extra_body would cause 400 because the API already enables thinking by default.
      Return None to avoid duplicate activation.
    - Opt-in models (_OPT_IN_THINKING_MODELS: deepseek-chat, deepseek-v4-pro): Return
      the activation payload to explicitly enable thinking mode.
    - Some opt-in models (e.g. deepseek-v4-pro) also require a top-level
      reasoning_effort; use get_reasoning_effort() for that.
    - All other models: Return None (no thinking mode).
    """
    if _model_matches(model, _AUTO_THINKING_MODELS):
        return None
    return _get_opt_in_payload(model, _OPT_IN_THINKING_MODELS)


def get_reasoning_effort(model: str) -> Optional[str]:
    """Return top-level reasoning_effort for models that require it."""
    if not model:
        return None
    m = model.lower().strip()
    for key, effort in _REASONING_EFFORT_MODELS.items():
        if m == key or m.startswith(key + "-"):
            return effort
    return None


# ============================================================
# LLM Tool Adapter
# ============================================================

class LLMToolAdapter:
    """Unified adapter for tool-calling via LiteLLM.

    Supports all providers (Gemini, Anthropic, OpenAI, DeepSeek, etc.) through
    a single litellm.completion() interface with optional Router for multi-key
    load balancing.
    """

    def __init__(self, config=None):
        config = config or get_config()
        self._config = config
        self._router = None          # litellm Router (multi-key primary model)
        self._litellm_available = False
        self._config_signature = None  # Issue #152: detect config changes for hot-reload
        self._register_custom_model_pricing()
        self._init_litellm()

    @staticmethod
    def _register_custom_model_pricing() -> None:
        """Register custom model pricing for models not in LiteLLM's built-in price list.

        This prevents cost calculation errors for MiniMax-M2.7 and similar models.
        """
        for model_name, pricing in _CUSTOM_MODEL_PRICING.items():
            try:
                litellm.register_model(
                    {
                        model_name: pricing
                    }
                )
                logger.debug(f"Registered custom pricing for {model_name}")
            except Exception as e:
                logger.debug(f"Model {model_name} may already be registered or pricing error: {e}")

    def _has_channel_config(self) -> bool:
        """Check if multi-channel config (channels / YAML) is active."""
        return bool(self._config.llm_model_list) and not all(
            e.get('model_name', '').startswith('__legacy_') for e in self._config.llm_model_list
        )

    def _init_litellm(self) -> None:
        """Initialize litellm Router from channels / YAML / legacy keys."""
        if litellm is None or Router is None:
            logger.warning("Agent LLM: litellm not installed")
            return
        config = self._config
        litellm_model = get_effective_agent_primary_model(config)
        if not litellm_model:
            logger.warning("Agent LLM: no effective primary model configured")
            return

        self._litellm_available = True

        # --- Channel / YAML path ---
        if self._has_channel_config():
            model_list = config.llm_model_list
            self._router = Router(
                model_list=model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            unique_models = list(dict.fromkeys(
                e['litellm_params']['model'] for e in model_list
            ))
            logger.info(
                f"Agent LLM: Router initialized from channels/YAML — "
                f"{len(model_list)} deployment(s), models: {unique_models}"
            )
            return

        # --- Legacy path ---
        keys = get_api_keys_for_model(litellm_model, config)
        if not keys:
            logger.info(
                f"Agent LLM: litellm initialized (model={litellm_model}, "
                f"API key from environment)"
            )
            return

        if len(keys) > 1:
            ep = extra_litellm_params(litellm_model, config)
            legacy_model_list = [
                {
                    "model_name": litellm_model,
                    "litellm_params": {
                        "model": litellm_model,
                        "api_key": k,
                        **ep,
                    },
                }
                for k in keys
            ]
            self._router = Router(
                model_list=legacy_model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            logger.info(
                f"Agent LLM: Legacy Router initialized with {len(keys)} keys "
                f"for {litellm_model}"
            )
        else:
            logger.info(f"Agent LLM: litellm initialized (model={litellm_model})")

        # Issue #152: record config signature for hot-reload detection
        self._config_signature = self._compute_config_signature(self._config)

    @staticmethod
    def _compute_config_signature(config) -> tuple:
        """Compute a lightweight signature of LLM-relevant config (Issue #152).

        Used to detect runtime config changes (e.g. model switch via WebUI)
        so the Router can be re-initialized without restarting the process.
        """
        model_list = getattr(config, 'llm_model_list', None) or []
        params_tuple = tuple(
            json.dumps(e.get('litellm_params', {}), sort_keys=True, default=str)
            for e in model_list
        )
        return (getattr(config, 'litellm_model', '') or '', params_tuple)

    def _maybe_reinit_router(self) -> None:
        """Re-initialize Router if runtime config has changed (Issue #152).

        When users switch models via WebUI, Config.reset_instance() is called
        and get_config() returns the new config. However, the Router created
        during __init__ still holds the old model_list. This method detects
        the change and rebuilds the Router lazily on the next LLM call.
        """
        fresh_config = get_config()
        current_sig = self._compute_config_signature(fresh_config)
        previous_sig = getattr(self, "_config_signature", None)
        if previous_sig is not None and current_sig != previous_sig:
            logger.info("[Issue #152] Agent LLM config changed, re-initializing Router...")
            self._config = fresh_config
            self._router = None
            self._litellm_available = False
            self._init_litellm()
            # _init_litellm updates _config_signature internally
        elif previous_sig is not None:
            # Even if signature unchanged, refresh cached config reference
            self._config = fresh_config

    @property
    def is_available(self) -> bool:
        """True if litellm is configured and at least one API key is present."""
        return self._router is not None or self._litellm_available

    @property
    def primary_provider(self) -> str:
        """Provider name extracted from litellm_model prefix."""
        model = get_effective_agent_primary_model(self._config)
        if "/" in model:
            return model.split("/")[0]
        return model or "none"

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Return True when the exception looks like a transient rate-limit error."""
        rate_limit_type = _safe_litellm_exc_type("RateLimitError")
        if rate_limit_type and isinstance(exc, rate_limit_type):
            return True
        message = str(exc).lower()
        markers = (
            "rate limit",
            "resource_exhausted",
            "429",
            "too many requests",
        )
        return any(marker in message for marker in markers)

    def _is_context_window_error(self, exc: Exception) -> bool:
        """Return True when the exception is caused by prompt/context length overflow."""
        context_type = _safe_litellm_exc_type("ContextWindowExceededError")
        if context_type and isinstance(exc, context_type):
            return True
        message = str(exc).lower()
        markers = (
            "context window",
            "maximum context length",
            "prompt is too long",
            "context length exceeded",
        )
        return any(marker in message for marker in markers)

    # ============================================================
    # Unified call
    # ============================================================

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[dict],
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Send messages + tool declarations to LLM, return normalized response.

        Args:
            messages: Conversation history in provider-neutral format:
                      [{"role": "system"/"user"/"assistant"/"tool", "content": ...}, ...]
            tools: OpenAI-format tool declarations; litellm converts to each provider's format.
            provider: Ignored (kept for backward compatibility).
            timeout: Optional timeout in seconds for the completion call.

        Returns:
            LLMResponse with either content (final answer) or tool_calls.
        """
        return self.call_completion(messages, tools=tools, provider=provider, timeout=timeout)

    def call_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Send a text-only completion through the shared routing stack."""
        return self.call_completion(
            messages,
            tools=None,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def call_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[dict]] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Shared completion path for both tool and text-only calls."""
        self._maybe_reinit_router()  # Issue #152: hot-reload Router on config change
        config = self._config
        models_to_try = get_effective_agent_models_to_try(config)
        started_at = time.time()
        providers = [self._get_model_provider(m) for m in models_to_try]

        last_error = None
        hit_rate_limit = False
        for idx, model in enumerate(models_to_try):
            remaining_timeout = timeout
            if timeout is not None and timeout > 0:
                remaining_timeout = max(0.0, float(timeout) - (time.time() - started_at))
                if remaining_timeout <= 0:
                    last_error = TimeoutError(
                        f"LLM completion timed out before trying fallback model {model}"
                    )
                    break
            try:
                return self._call_litellm_model(
                    messages,
                    tools or [],
                    model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=remaining_timeout,
                )
            except Exception as e:
                if self._is_context_window_error(e):
                    logger.warning("Agent LLM context window exceeded on %s: %s", model, e)
                elif self._is_rate_limit_error(e):
                    logger.warning("Agent LLM rate-limited on %s: %s", model, e)
                    hit_rate_limit = True
                    # Avoid blind backoff across different providers; cross-provider
                    # fallback usually means different accounts/rate-limit buckets.
                    should_backoff = (
                        idx + 1 < len(models_to_try)
                        and providers[idx] == providers[idx + 1]
                    )
                    if should_backoff:
                        backoff_sleep = min(2.0, (time.time() - started_at) * 0.1 + 0.5)
                        if timeout is not None and timeout > 0:
                            remaining = max(0.0, float(timeout) - (time.time() - started_at))
                            if remaining > 0:
                                time.sleep(min(backoff_sleep, remaining))
                        else:
                            time.sleep(backoff_sleep)
                else:
                    logger.warning("Agent LLM call failed with %s: %s", model, e)
                last_error = e
                continue

        suffix = " (rate-limit encountered during fallback)" if hit_rate_limit else ""
        error_msg = f"All LLM models failed{suffix}. Last error: {last_error}"
        logger.error(error_msg)
        return LLMResponse(content=error_msg, provider="error")

    @staticmethod
    def _get_model_provider(model: str) -> str:
        """Return LiteLLM provider namespace for model fallback grouping."""
        if "/" in model:
            return model.split("/", 1)[0]
        return "openai"

    def _call_litellm_model(
        self,
        messages: List[Dict[str, Any]],
        tools: List[dict],
        model: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Call a specific litellm model with OpenAI-format messages and tools."""
        openai_messages = self._convert_messages(messages)

        # Use short model name (without provider prefix) for thinking model lookup
        model_short = model.split("/")[-1] if "/" in model else model

        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "temperature": self._get_temperature(model) if temperature is None else temperature,
        }
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            call_kwargs["timeout"] = timeout

        extra = get_thinking_extra_body(model_short)
        if extra:
            call_kwargs["extra_body"] = extra
        reasoning_effort = get_reasoning_effort(model_short)
        if reasoning_effort:
            call_kwargs["reasoning_effort"] = reasoning_effort

        if tools:
            call_kwargs["tools"] = tools

        # messages / tools 里只要有一个 nan/inf，httpx 在发请求前就会抛
        # "Out of range float values are not JSON compliant"，被 LiteLLM 包成
        # APIConnectionError，看起来像网络错误。发送前统一清洗。
        call_kwargs, replaced_paths = sanitize_non_finite(call_kwargs)
        if replaced_paths:
            logger.warning(
                "LLM 请求体包含 %d 处非法数值（nan/inf），已替换为 null：%s",
                len(replaced_paths),
                ", ".join(replaced_paths[:10]),
            )

        get_llm_rate_limiter().acquire(
            bucket=get_rate_limit_bucket(model, default=self._get_model_provider(model)),
            min_interval_seconds=getattr(self._config, "llm_min_interval", 0.0),
        )

        # Use Router for primary model (multi-key), direct litellm for others
        use_channel_router = self._has_channel_config()
        _router_model_names = set(get_configured_llm_models(self._config.llm_model_list))
        if use_channel_router and self._router and model in _router_model_names:
            # Channel / YAML path: Router manages all models in its model_list
            response = self._router.completion(**call_kwargs)
        elif self._router and model == self._config.litellm_model and not use_channel_router:
            # Legacy path: Router for primary model multi-key
            response = self._router.completion(**call_kwargs)
        else:
            # Legacy/direct-env path: direct call (also handles direct-env
            # providers like groq/ or bedrock/ that are not in the Router
            # model_list even when channel mode is active)
            keys = get_api_keys_for_model(model, self._config)
            if keys:
                call_kwargs["api_key"] = keys[0]
            call_kwargs.update(extra_litellm_params(model, self._config))
            response = litellm.completion(**call_kwargs)

        return self._parse_litellm_response(response, model)

    def _get_temperature(self, model: str) -> float:
        """Return unified temperature from config."""
        return self._config.llm_temperature

    @staticmethod
    def _json_safe_text(value: Any) -> str:
        """把工具参数/结果转成 JSON 文本，且保证``NaN``/numpy 不会漏进请求体。"""
        if isinstance(value, str):
            return value
        cleaned, _ = sanitize_non_finite(value)
        try:
            return json.dumps(cleaned, ensure_ascii=False, default=str, allow_nan=False)
        except (TypeError, ValueError):
            return str(cleaned)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal message format to OpenAI-compatible format for litellm."""
        openai_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    # 工具返回值直接来自 DataFrame，常带 numpy 类型和 NaN；
                    # json.dumps 默认允许 NaN，会写出 `NaN` 字面量，LiteLLM 在
                    # 转成 Gemini functionResponse 时又会解析回 float，最终在
                    # httpx 序列化阶段报 "Out of range float values"。
                    "content": self._json_safe_text(msg["content"]),
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                openai_tc = []
                for tc in msg["tool_calls"]:
                    tc_dict: Dict[str, Any] = {
                        "id": tc.get("id", str(uuid.uuid4())[:8]),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": self._json_safe_text(tc["arguments"]),
                        },
                    }
                    sig = tc.get("thought_signature")
                    if sig is not None:
                        tc_dict["provider_specific_fields"] = {"thought_signature": sig}
                    openai_tc.append(tc_dict)
                openai_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": openai_tc,
                }
                if msg.get("reasoning_content") is not None:
                    openai_msg["reasoning_content"] = msg["reasoning_content"]
                openai_messages.append(openai_msg)
            else:
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        return openai_messages

    def _parse_litellm_response(self, response: Any, model: str) -> LLMResponse:
        """Parse litellm OpenAI-compatible response into LLMResponse."""
        choice = response.choices[0]
        tool_calls: List[ToolCall] = []

        # Handle MiniMax-specific content_blocks format
        # MiniMax-M2.7 may return content_blocks at choice level or inside message
        # Check both possible locations for content_blocks to ensure consistency
        # Concatenate ALL text blocks to avoid truncating multi-block responses
        text_content = choice.message.content
        if text_content is None:
            content_blocks = None
            if hasattr(choice, "content_blocks"):
                content_blocks = choice.content_blocks
            elif hasattr(choice.message, "content_blocks"):
                content_blocks = choice.message.content_blocks

            if content_blocks:
                # MiniMax response format: content_blocks[].text
                # Concatenate ALL text blocks to preserve complete response
                text_parts = []
                for block in content_blocks:
                    if getattr(block, "type", None) == "text":
                        text = getattr(block, "text", "") or ""
                        if text:
                            text_parts.append(text)
                    elif hasattr(block, "content") and block.content:
                        text_parts.append(block.content)
                text_content = "".join(text_parts).strip()

        # DeepSeek/Qwen thinking mode; not in standard OpenAI type, accessed via getattr
        reasoning_content = getattr(choice.message, "reasoning_content", None)

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args: Dict[str, Any] = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}

                # Extract thought_signature: stored in provider_specific_fields (Gemini 3 via LiteLLM proxy)
                psf = getattr(tc, "provider_specific_fields", None)
                if psf is not None:
                    sig = psf.get("thought_signature") if isinstance(psf, dict) else getattr(psf, "thought_signature", None)
                else:
                    func_psf = getattr(tc.function, "provider_specific_fields", None)
                    if func_psf is not None:
                        sig = func_psf.get("thought_signature") if isinstance(func_psf, dict) else getattr(func_psf, "thought_signature", None)
                    else:
                        sig = getattr(tc, "thought_signature", None)

                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                    thought_signature=sig,
                ))

        usage: Dict[str, Any] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        provider_name = model.split("/")[0] if "/" in model else model
        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            usage=usage,
            provider=provider_name,
            model=model,
            raw=response,
        )

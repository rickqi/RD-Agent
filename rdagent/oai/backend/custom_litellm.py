"""
Custom LiteLLM backend with multi-model rotation and separate embedding provider.

Features:
1. Round-robin rotation between N chat models (DeepSeek V4 Pro + ZhiPu GLM + DeepSeek V4 Flash)
2. Automatic thinking support for DeepSeek reasoning models (V4 Pro, V4 Flash, R1)
3. Separate embedding provider support (ZhiPu embedding-3)
4. Per-model API credentials with smart fallback

Configuration (.env):
    BACKEND=rdagent.oai.backend.custom_litellm.CustomLiteLLMAPIBackend
    # Chat Model 1 (DeepSeek V4 Pro) - primary
    DEEPSEEK_API_KEY=sk-xxx
    CHAT_MODEL=deepseek/deepseek-v4-pro
    # Chat Model 2 (ZhiPu GLM) - rotation partner
    CHAT_MODEL_2=openai/glm-5.1
    CHAT_MODEL_2_API_KEY=<zhipu_key>
    CHAT_MODEL_2_API_BASE=https://open.bigmodel.cn/api/paas/v4
    # Chat Model 3 (DeepSeek V4 Flash) - rotation partner (same DEEPSEEK_API_KEY)
    CHAT_MODEL_3=deepseek/deepseek-v4-flash
    # Embedding (ZhiPu)
    EMBEDDING_MODEL=openai/embedding-3
    EMBEDDING_API_KEY=<zhipu_key>
    EMBEDDING_API_BASE=https://open.bigmodel.cn/api/paas/v4
    # Reasoning model support
    REASONING_THINK_RM=True
    REASONING_EFFORT=high
"""

from __future__ import annotations

import os
import threading
from typing import Any, Literal, Optional, Type, Union, cast

import litellm as _litellm_module
from litellm import embedding as litellm_embedding
from pydantic import BaseModel

from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.litellm import LITELLM_SETTINGS, LiteLLMAPIBackend

# Models that support DeepSeek thinking mode (reasoning_content + content)
_DEEPSEEK_THINKING_MODELS = ("v4", "r1", "reasoner")

# Models that support reasoning_effort parameter
_REASONING_EFFORT_MODELS = ("deepseek-v4", "deepseek-r1", "o1", "o3")


class CustomLiteLLMAPIBackend(LiteLLMAPIBackend):
    """
    LiteLLMAPIBackend with multi-model rotation and separate embedding provider.

    Scans CHAT_MODEL (primary), CHAT_MODEL_2, CHAT_MODEL_3, ... from environment.
    Alternates between all available models on each call for maximum diversity.
    Auto-enables thinking mode for DeepSeek reasoning models (V4 Pro, V4 Flash, R1).
    """

    _call_counter: int = 0
    _counter_lock: threading.Lock = threading.Lock()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    def _get_next_index(cls) -> int:
        with cls._counter_lock:
            idx = cls._call_counter
            cls._call_counter += 1
            return idx

    @staticmethod
    def _needs_thinking(model: str) -> bool:
        """Check if a model requires thinking mode enabled."""
        m = model.lower()
        return "deepseek" in m and any(tag in m for tag in _DEEPSEEK_THINKING_MODELS)

    @staticmethod
    def _supports_reasoning_effort(model: str) -> bool:
        """Check if a model supports reasoning_effort parameter."""
        m = model.lower()
        return any(tag in m for tag in _REASONING_EFFORT_MODELS)

    def _get_rotation_models(self) -> list[dict[str, Any]]:
        """Build list of available models with their API credentials.

        Scans CHAT_MODEL (primary), CHAT_MODEL_2, CHAT_MODEL_3, ...
        For models without explicit API key, falls back to DEEPSEEK_API_KEY / OPENAI_API_KEY.
        """
        models = []

        # Primary model (from CHAT_MODEL / LITELLM_CHAT_MODEL)
        primary_model = LITELLM_SETTINGS.chat_model
        models.append({
            "model": primary_model,
            "api_key": os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            "api_base": os.getenv("OPENAI_API_BASE", ""),
            "temperature": LITELLM_SETTINGS.chat_temperature,
            "max_tokens": LITELLM_SETTINGS.chat_max_tokens,
        })

        # Additional models: CHAT_MODEL_2, CHAT_MODEL_3, ... (auto-detect count)
        n = 2
        while True:
            model_n = os.getenv(f"CHAT_MODEL_{n}", "")
            if not model_n:
                break
            # API key: explicit > DEEPSEEK (for deepseek/ models) > OPENAI fallback
            api_key = os.getenv(f"CHAT_MODEL_{n}_API_KEY", "")
            if not api_key:
                if model_n.startswith("deepseek/"):
                    api_key = os.getenv("DEEPSEEK_API_KEY", "")
                else:
                    api_key = os.getenv("OPENAI_API_KEY", "")
            models.append({
                "model": model_n,
                "api_key": api_key,
                "api_base": os.getenv(f"CHAT_MODEL_{n}_API_BASE", ""),
                "temperature": float(os.getenv(f"CHAT_MODEL_{n}_TEMPERATURE", str(LITELLM_SETTINGS.chat_temperature))),
                "max_tokens": int(os.getenv(f"CHAT_MODEL_{n}_MAX_TOKENS", "0")) or None,
            })
            n += 1

        return models

    def get_complete_kwargs(self) -> LiteLLMAPIBackend.CompleteKwargs:
        """Override to implement round-robin model rotation."""
        # First check chat_model_map tag-based routing (preserves existing behavior)
        if LITELLM_SETTINGS.chat_model_map:
            for t, mc in LITELLM_SETTINGS.chat_model_map.items():
                if t in logger._tag:
                    return LiteLLMAPIBackend.CompleteKwargs(
                        model=mc["model"],
                        temperature=float(mc.get("temperature", LITELLM_SETTINGS.chat_temperature)),
                        max_tokens=int(mc["max_tokens"]) if "max_tokens" in mc else LITELLM_SETTINGS.chat_max_tokens,
                        reasoning_effort=cast(Literal["low", "medium", "high"], mc["reasoning_effort"]) if "reasoning_effort" in mc and mc["reasoning_effort"] in ["low", "medium", "high"] else None,
                    )

        # Round-robin rotation
        models = self._get_rotation_models()
        if len(models) <= 1:
            return super().get_complete_kwargs()

        idx = self._get_next_index() % len(models)
        selected = models[idx]
        model_name = selected["model"]

        # Only set reasoning_effort for models that support it
        reasoning_effort = None
        if self._supports_reasoning_effort(model_name) and LITELLM_SETTINGS.reasoning_effort:
            reasoning_effort = LITELLM_SETTINGS.reasoning_effort

        logger.info(
            f"{LogColors.GREEN}Rotation: using model [{idx}/{len(models)}] {model_name}"
            f"{' (thinking)' if self._needs_thinking(model_name) else ''}"
            f"{' (effort=' + str(reasoning_effort) + ')' if reasoning_effort else ''}"
            f"{LogColors.END}",
            tag="model_rotation",
        )
        return LiteLLMAPIBackend.CompleteKwargs(
            model=model_name,
            temperature=selected["temperature"],
            max_tokens=selected["max_tokens"],
            reasoning_effort=reasoning_effort,
        )

    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """Override to inject per-model API credentials and thinking support."""
        complete_kwargs = self.get_complete_kwargs()
        model = complete_kwargs["model"]

        # Get rotation models to find the right API key/base
        models = self._get_rotation_models()
        selected_model_config = None
        for m in models:
            if m["model"] == model:
                selected_model_config = m
                break

        # Build completion call kwargs — convert TypedDict to plain dict for merging
        completion_kwargs: dict[str, Any] = dict(complete_kwargs)
        if selected_model_config and selected_model_config.get("api_key"):
            completion_kwargs["api_key"] = selected_model_config["api_key"]
        if selected_model_config and selected_model_config.get("api_base"):
            completion_kwargs["api_base"] = selected_model_config["api_base"]

        # Enable thinking for DeepSeek reasoning models (V4 Pro, V4 Flash, R1)
        extra_call_kwargs: dict[str, Any] = {}
        if self._needs_thinking(model):
            extra_call_kwargs["thinking"] = {"type": "enabled"}

        # Handle response_format support check
        from litellm import supports_response_schema
        if response_format and not supports_response_schema(model=model):
            logger.warning(
                f"{LogColors.YELLOW}Model {model} does not support response schema, ignoring response_format.{LogColors.END}",
                tag="llm_messages",
            )
            response_format = None

        if response_format:
            kwargs["response_format"] = response_format

        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(self._build_log_messages(messages), tag="llm_messages")

        response = _litellm_module.completion(
            messages=messages,
            stream=LITELLM_SETTINGS.chat_stream,
            max_retries=0,
            **completion_kwargs,
            **extra_call_kwargs,
            **kwargs,
        )
        if LITELLM_SETTINGS.log_llm_chat_content:
            logger.info(f"{LogColors.GREEN}Using chat model{LogColors.END} {model}", tag="llm_messages")

        if LITELLM_SETTINGS.chat_stream:
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(f"{LogColors.BLUE}assistant:{LogColors.END}", tag="llm_messages")
            content = ""
            finish_reason = None
            for message in response:
                if message["choices"][0]["finish_reason"]:
                    finish_reason = message["choices"][0]["finish_reason"]
                # Only capture 'content', ignore 'reasoning_content' from thinking models
                if "content" in message["choices"][0]["delta"]:
                    chunk = message["choices"][0]["delta"]["content"] or ""
                    content += chunk
                    if LITELLM_SETTINGS.log_llm_chat_content:
                        logger.info(LogColors.CYAN + chunk + LogColors.END, raw=True, tag="llm_messages")
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info("\n", raw=True, tag="llm_messages")
        else:
            content = str(response.choices[0].message.content)
            finish_reason = response.choices[0].finish_reason
            finish_reason_str = (
                f"({LogColors.RED}Finish reason: {finish_reason}{LogColors.END})"
                if finish_reason and finish_reason != "stop"
                else ""
            )
            if LITELLM_SETTINGS.log_llm_chat_content:
                logger.info(
                    f"{LogColors.BLUE}assistant:{LogColors.END} {finish_reason_str}\n{content}", tag="llm_messages"
                )

        # Cost tracking — reference parent module's ACC_COST
        from rdagent.oai.backend import litellm as _litellm_backend_mod
        try:
            cost = _litellm_module.completion_cost(model=model, messages=messages, completion=content)
            _litellm_backend_mod.ACC_COST += cost
        except Exception as e:
            logger.warning(f"Cost calculation failed for model {model}: {e}. Skip cost statistics.")

        try:
            prompt_tokens = _litellm_module.token_counter(model=model, messages=messages)
            completion_tokens = _litellm_module.token_counter(model=model, text=content)
        except Exception as e:
            logger.warning(f"Token counting failed for model {model}: {e}. Skip token statistics.")
            prompt_tokens = 0
            completion_tokens = 0

        logger.log_object(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "accumulated_cost": _litellm_backend_mod.ACC_COST,
            },
            tag="token_cost",
        )
        return content, finish_reason

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        model_name = LITELLM_SETTINGS.embedding_model
        embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
        embedding_api_base = os.getenv("EMBEDDING_API_BASE", "")

        if embedding_api_key and embedding_api_base:
            logger.info(f"Custom embedding: model={model_name}, base={embedding_api_base}")
            response = litellm_embedding(
                model=model_name,
                input=input_content_list,
                api_key=embedding_api_key,
                api_base=embedding_api_base,
            )
        else:
            response = litellm_embedding(
                model=model_name,
                input=input_content_list,
            )

        response_list = [data["embedding"] for data in response.data]
        return response_list

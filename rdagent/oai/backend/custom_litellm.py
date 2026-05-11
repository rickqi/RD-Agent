"""
Custom LiteLLM backend that overrides embedding to use ZhiPu BigModel.

This module extends LiteLLMAPIBackend to support a separate embedding provider
(ZhiPu embedding-3) while keeping the chat model on DeepSeek.

Configuration (.env):
    BACKEND=rdagent.oai.backend.custom_litellm.CustomLiteLLMAPIBackend
    # Chat (DeepSeek)
    OPENAI_API_KEY=...
    OPENAI_API_BASE=https://api.deepseek.com/v1
    CHAT_MODEL=deepseek-chat
    # Embedding (ZhiPu)
    EMBEDDING_MODEL=openai/embedding-3
    EMBEDDING_API_KEY=<zhipu_key>
    EMBEDDING_API_BASE=https://open.bigmodel.cn/api/paas/v4
"""

from __future__ import annotations

from typing import Any

from rdagent.oai.backend.litellm import LiteLLMAPIBackend
from rdagent.oai.llm_conf import LITELLM_SETTINGS

import os
from litellm import embedding as litellm_embedding
from rdagent.log import rdagent_logger as logger


class CustomLiteLLMAPIBackend(LiteLLMAPIBackend):
    """
    LiteLLMAPIBackend with separate embedding provider support.

    Reads EMBEDDING_API_KEY and EMBEDDING_API_BASE from environment variables
    to route embedding calls to a different provider than the chat model.
    """

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
            # Fallback to default behavior (uses OPENAI_API_KEY/BASE)
            response = litellm_embedding(
                model=model_name,
                input=input_content_list,
            )

        response_list = [data["embedding"] for data in response.data]
        return response_list

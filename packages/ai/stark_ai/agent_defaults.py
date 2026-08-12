"""Model + endpoint configuration for the stark-ai processors.

Everything is read from the environment, so nothing about your setup is baked
into the package. Point it at any OpenAI-compatible backend:

  STARK_AI_MODEL             pydantic-ai model string   (default: openai:gpt-4o-mini)
  STARK_AI_EMBEDDINGS_MODEL  embeddings model           (default: text-embedding-3-small)
  OPENAI_BASE_URL            OpenAI-compatible endpoint  (optional; unset = OpenAI)
  OPENAI_API_KEY             API key
  PROMPTED_OUTPUT            "1" = prompt-based JSON extraction (local models without
                             reliable function-calling); "0" = function-calling (default)

Examples:
  OpenAI:            STARK_AI_MODEL=openai:gpt-4o-mini    OPENAI_API_KEY=sk-...
  Anthropic:         STARK_AI_MODEL=anthropic:claude-...  ANTHROPIC_API_KEY=sk-ant-...
  llama.cpp/Ollama:  STARK_AI_MODEL=openai:<model>  OPENAI_BASE_URL=http://127.0.0.1:8080/v1  PROMPTED_OUTPUT=1
"""
import os
from typing import Any

from pydantic_ai import Embedder, PromptedOutput
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

MODEL_NAME = os.environ.get("STARK_AI_MODEL", "openai:gpt-4o-mini")
MODEL_EMBEDDINGS = os.environ.get("STARK_AI_EMBEDDINGS_MODEL", "text-embedding-3-small")

_BASE_URL = os.environ.get("OPENAI_BASE_URL")
_API_KEY = os.environ.get("OPENAI_API_KEY")

# True  → PromptedOutput (system-prompt JSON extraction) — for local models
# False → pydantic-ai default (function-calling) — for OpenAI / Anthropic
PROMPTED_OUTPUT: bool = os.environ.get("PROMPTED_OUTPUT", "0") == "1"


def output_type[T](base_type: type[T]) -> Any:
    """PromptedOutput when PROMPTED_OUTPUT is set, the plain type otherwise."""
    return PromptedOutput(base_type) if PROMPTED_OUTPUT else base_type


def embedder(model: str = MODEL_EMBEDDINGS) -> Embedder:
    """Embedder pointed at the configured OpenAI-compatible endpoint."""
    return Embedder(
        OpenAIEmbeddingModel(model, provider=OpenAIProvider(base_url=_BASE_URL, api_key=_API_KEY))
    )

"""
Central config for all LLM/embeddings processors.

Pick a preset below (MODEL_TINY / MODEL_SMALL / MODEL_MEDIUM / MODEL_FLAGSHIP),
assign it to MODEL_NAME, then set the matching env vars for your backend:

  OPENAI_BASE_URL  — OpenAI-compatible endpoint  (default: http://127.0.0.1:8080/v1)
  OPENAI_API_KEY   — API key                      (default: 1234)
  LLM_PROMPTED     — "1" = prompt-based JSON extraction instead of function-calling;
                     use for llama.cpp / Ollama, set "0" for OpenAI / Anthropic (default: 1)

# llama.cpp / Ollama openai-compat:  OPENAI_BASE_URL=http://127.0.0.1:8080/v1  OPENAI_API_KEY=1234
# Ollama native:                      OLLAMA_BASE_URL=http://127.0.0.1:11434
# OpenAI:                             OPENAI_API_KEY=sk-...          LLM_PROMPTED=0
# Anthropic:                          ANTHROPIC_API_KEY=sk-ant-...   LLM_PROMPTED=0
"""

# ---------------------------------------------------------------------------------

# local
MODEL_NANO = "openai:qwen3-0.6b"
MODEL_TINY = "openai:qwen3-1.7b"
MODEL_SMALL = "openai:qwen3-4b"
MODEL_MEDIUM = "openai:qwen3-8b"
MODEL_LARGE = "openai:qwen3-14b"
# remote
MODEL_HUGE = ""

MODEL_NAME = MODEL_NANO  # change this to MODEL_TINY, MODEL_MEDIUM, MODEL_LARGE or as needed
MODEL_EMBEDDINGS = "qwen3-embedding-0.6b"  # embeddings need a separate model

# ---------------------------------------------------------------------------------
import os

os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
os.environ.setdefault("OPENAI_API_KEY", "1234")
os.environ.setdefault("MODEL_NAME", MODEL_NAME)
_BASE_URL = os.environ.get("OPENAI_BASE_URL")
_API_KEY = os.environ.get("OPENAI_API_KEY")

# True  → PromptedOutput (system-prompt JSON extraction) — for llama.cpp / Ollama
# False → pydantic-ai default (function-calling) — for OpenAI / Anthropic
PROMPTED_OUTPUT: bool = os.environ.get("PROMPTED_OUTPUT", "0") == "1"

# ---------------------------------------------------------------------------------

# Concrete JSON examples for PromptedOutput — small models follow an example far
# more reliably than an abstract JSON schema (they tend to echo "properties" keys).
# Braces doubled to escape str.format() used internally by pydantic-ai.
# _EXAMPLES: dict[str, str] = {
#     "_AgentDecision": (
#         '{{"response": "your answer or null", "response_type": "immediate|background|none", "reasoning": "short reason", "confidence": 0.95}}'
#     ),
# }

# _PROMPTED_TEMPLATE = (
#     "Always respond with a JSON object with exactly these fields — no wrapping, no extra keys:\n\n"
#     "{example}\n\n"
#     "Don't include any text or Markdown fencing before or after."
# )

# ---------------------------------------------------------------------------------


from typing import Any

from pydantic_ai import Embedder, PromptedOutput
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider


def output_type[T](base_type: type[T], *, example_key: str | None = None) -> Any:
    """PromptedOutput when PROMPTED, plain type otherwise."""
    if not PROMPTED_OUTPUT:
        return base_type
    example = _EXAMPLES.get(example_key or base_type.__name__, "{schema}")
    return PromptedOutput(base_type, template=_PROMPTED_TEMPLATE.format(example=example))


def embedder(model: str = MODEL_EMBEDDINGS) -> Embedder:
    """Embedder pointed at the configured OpenAI-compatible endpoint."""
    return Embedder(OpenAIEmbeddingModel(model, provider=OpenAIProvider(base_url=_BASE_URL, api_key=_API_KEY)))

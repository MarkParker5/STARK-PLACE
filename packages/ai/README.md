# stark-ai

LLM-powered processors for the [S.T.A.R.K. engine](https://github.com/MarkParker5/STARK).
Each is a `CommandsContextProcessor` you drop into the engine's processing chain to
add LLM understanding, NER, or agentic behaviour.

```bash
pip install stark-ai --find-links https://markparker5.github.io/STARK-PLACE/
```

Import as `stark_ai`. Depends on `stark-engine` (>=4.5,<5) plus `pydantic-ai`.

## Quick start

**1. Point it at a model** (env only — nothing is baked into the package):

```bash
export STARK_AI_MODEL=openai:gpt-4o-mini
export OPENAI_API_KEY=sk-...
```

**2. Drop a processor into the engine's chain.** Processors are passed to
`CommandsContext(..., processors=[...])`; put an AI processor after the engine's
default `SearchProcessor` so it only runs when normal matching finds nothing:

```python
from stark.core.processors import SearchProcessor
from stark_ai.fallback_agent_processor import FallbackAgentProcessor

context = CommandsContext(
    task_group,
    commands_manager,
    processors=[SearchProcessor(), FallbackAgentProcessor()],
)
await context.process_string("what's the capital of France?")
```

The engine owns the runtime around this (the `task_group`, registering commands on
`CommandsManager`, and delivering `Response`s via the context delegate) — see the
[STARK docs](https://stark.markparker.me). This package only supplies the processors
and their model config.

## Exposed processors

| Import | What it does |
|---|---|
| `stark_ai.ner_processor_embeddings.EmbeddingsNERProcessor` | NER via embedding similarity (no generation). |
| `stark_ai.ner_processor_llm.LLMNERProcessor` | Named-entity recognition via the LLM. |
| `stark_ai.one_step_search_parsing_processor_llm.OneStepLLMProcessor` | One LLM call that both **picks** the command and **parses** its parameters. |
| `stark_ai.two_step_search_parsing_processor_llm.TwoStepLLMProcessor` | Two calls: **search** (choose command) then **parse** (extract parameters) — more robust on larger command sets. |
| `stark_ai.structured_llm_processor.StructuredLLMProcessor` | Single-shot structured extraction — LLM fills a command's parameters directly. |
| `stark_ai.fallback_agent_processor.FallbackAgentProcessor` | Fallback agent invoked when normal command search finds nothing. |
| `stark_ai.agentic_loop_processor.AgenticLoopProcessor` | Multi-step agentic loop with a supervisor and `ProgressUpdate` streaming. |

## Configuration

All processors read their model/endpoint from the environment via
`stark_ai.agent_defaults`:

| Env var | Purpose | Default |
|---|---|---|
| `STARK_AI_MODEL` | pydantic-ai model string | `openai:gpt-4o-mini` |
| `STARK_AI_EMBEDDINGS_MODEL` | embeddings model (for `EmbeddingsNERProcessor`) | `text-embedding-3-small` |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint | unset → OpenAI |
| `OPENAI_API_KEY` | API key | — |
| `PROMPTED_OUTPUT` | `1` = prompt-based JSON extraction for local models without reliable function-calling; `0` = function-calling | `0` |

```bash
# OpenAI            STARK_AI_MODEL=openai:gpt-4o-mini     OPENAI_API_KEY=sk-...
# Anthropic         STARK_AI_MODEL=anthropic:claude-...   ANTHROPIC_API_KEY=sk-ant-...
# llama.cpp/Ollama  STARK_AI_MODEL=openai:<model>  OPENAI_BASE_URL=http://127.0.0.1:8080/v1  PROMPTED_OUTPUT=1
```

# stark-ai

LLM-powered processors for the [S.T.A.R.K. engine](https://github.com/MarkParker5/STARK).
They plug into the engine's command-processing pipeline (subclasses of
`CommandsContextProcessor`) to add LLM understanding, NER, and agentic behaviour.

```bash
pip install stark-ai --find-links https://markparker5.github.io/STARK-PLACE/
```

Import as `stark_ai`. Depends on `stark-engine` (>=4.5,<5) plus `pydantic-ai`.

## Exposed processors

| Import | What it does |
|---|---|
| `stark_ai.structured_llm_processor.StructuredLLMProcessor` | Single-shot structured extraction — LLM fills a command's parameters directly. |
| `stark_ai.one_step_search_parsing_processor_llm.OneStepLLMProcessor` | One LLM call that both **picks** the command and **parses** its parameters. |
| `stark_ai.two_step_search_parsing_processor_llm.TwoStepLLMProcessor` | Two calls: **search** (choose command) then **parse** (extract parameters) — more robust on larger command sets. |
| `stark_ai.agentic_loop_processor.AgenticLoopProcessor` | Multi-step agentic loop with a supervisor and `ProgressUpdate` streaming. |
| `stark_ai.fallback_agent_processor.FallbackAgentProcessor` | Fallback agent invoked when normal command search finds nothing. |
| `stark_ai.ner_processor_llm.LLMNERProcessor` | Named-entity recognition via the LLM. |
| `stark_ai.ner_processor_embeddings.EmbeddingsNERProcessor` | NER via embedding similarity (no generation). |

## Configuration

All processors read model/endpoint config from `stark_ai.agent_defaults`.
Pick a preset and point the env vars at your backend:

| Preset | Model |
|---|---|
| `MODEL_NANO` / `MODEL_TINY` / `MODEL_SMALL` / `MODEL_MEDIUM` / `MODEL_LARGE` | `qwen3` 0.6b … 14b |
| `MODEL_EMBEDDINGS` | `qwen3-embedding-0.6b` |

```bash
# llama.cpp / Ollama (OpenAI-compatible)
OPENAI_BASE_URL=http://127.0.0.1:8080/v1  OPENAI_API_KEY=1234  LLM_PROMPTED=1
# OpenAI / Anthropic
OPENAI_API_KEY=sk-...        LLM_PROMPTED=0
ANTHROPIC_API_KEY=sk-ant-... LLM_PROMPTED=0
```

`LLM_PROMPTED=1` uses prompt-based JSON extraction (for local models without
reliable function-calling); set `0` for OpenAI/Anthropic function-calling.

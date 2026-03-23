# LLM Playground

Experimental LLM and embeddings-based processors for [STARK](https://github.com/MarkParker5/STARK).
All processors are drop-in `CommandsContextProcessor` implementations — no core changes required.
Implemented with different approaches, operating at different pipeline levels, with different LLM requirements and degrees of freedom — from a tiny embedding model to a flagship reasoning agent.
See each file's docstring for details and trade-offs.

---

## Structure

```
llm-playground/
├── ready/          # Complete, usable processors
├── draft/          # Blocked by open design questions — see draft/DESIGN.md
├── todo/           # Notes and future work
├── tests/          # (to be populated)
├── CORNERSTONES.md # Pipeline invariants every processor must uphold
└── README.md
```

### `ready/`

| File | Role | Model | Summary |
|------|------|-------|----------------------|
| `ner_processor_embeddings.py` | NER pre-processor via sliding-window embeddings | tiny embedding | fast after type-embedding cache is warm; fastest NER option in this playground |
| `ner_processor_llm.py` | NER pre-processor via LLM | small | one call per request; slower than embeddings NER |
| `two_step_search_parsing_processor_llm.py` | Command search + parameter parsing | small–medium | two sequential structured calls; simple tasks; might be slower than one-step; smaller context per call; deterministic-ish |
| `one_step_search_parsing_processor_llm.py` | Command search + parameter parsing | medium–large | one call; might be faster and smarter than two-step, but also needs a more capable model; see file's docstring |
| `fallback_agent_processor.py` | Last-resort fallback to a 3rd-party agent with unconstrained answer | large–flagship for reasonable output | pre-flight + optional second call for larger tasks; still no agentic loops or tools or system access provided (at least not by stark) |
| `agentic_loop_processor.py` | Full agentic command loop | large–flagship | pydantic-ai agent with full access to all STARK commands as tools; LLM serialisation via lock (per `ModelRequestNode` only); message injection folded in-place between tool steps; conversation history persisted across requests |

> **Model size notes** — not yet benchmarked. Expected ranges: tiny ≈ embedding-only models, small ≈ 1–4B, medium ≈ 7–14B, large ≈ 30–70B, flagship = best available. Details and trade-offs in each file's module docstring.

### `draft/` — blocked

See `draft/DESIGN.md` and each file's docstring for proposed solutions.

---

## Environment

All processors default to a local Ollama instance:

```
OLLAMA_BASE_URL   # default: http://127.0.0.1:8080/v1
OLLAMA_API_KEY    # default: 1234
```

Model names are hardcoded at the top of each file — swap them in the `Agent(...)` / `Embedder(...)` call.

---

## Recommended chain

Processors are tried in order; the **first to return a non-empty result short-circuits the rest** — later processors in the same section are never called for that request. NER pre-processors are the exception: they always return empty and never short-circuit.

Each processor can be used standalone or combined with others. They can replace the native `SearchProcessor` entirely, or sit alongside it as an addition.

The chain splits into three natural sections; order within each section is your choice:

```
# 1. Pre-processing — always run, never short-circuit
[EmbeddingsNERProcessor]   ← fast, good for open-ended entity types
[LLMNERProcessor]          ← slower, better contextual understanding

# 2. Command search — first match wins
[SearchProcessor]                        ← native pattern matching; fast and precise
[TwoStepLLMProcessor]                    ← LLM fallback; smaller model, two calls
[OneStepLLMProcessor]                    ← LLM fallback; one call, better context understanding, needs stronger model

# 3. Fallbacks — reached only if nothing above matched
[FallbackAgentProcessor]    ← ask LLM for a free-text response as a last resort
[AgenticLoopProcessor]      ← full agentic loop with STARK commands as tools; dead last
```

Pipeline invariants every processor must respect: `CORNERSTONES.md`.

---

## Tests

```sh
# Unit tests only (no model required)
poetry run pytest tests/ -m "not benchmark"

# Benchmarks only (Ollama must be running with the right models)
poetry run pytest tests/ -m benchmark

# Everything
poetry run pytest tests/
```

### Intermediate Results

#### Embeddings

Even a small embeddings model performs good in finding the closest command name to the request. However, embeddings are unable to neither parse parameters, nor perform NER. Trigger words often match sloser to the parameter type then the parameter. For example, in "play bohemian rapsody song" words "play" and "song" match closer to "Song" class then "bohemian rapsody". Embeddings-powered intent suggestions might work, but NER is defenitely will stay in drafts.

#### LLM

with `openai:qwen3-14b` all current tests are passed except these:

```sh
FAILED tests/test_one_step_search_parsing_processor_llm.py::test_multi_command_both_returned - AssertionError: assert 'lamp_brightness' in {'lamp_on'}
FAILED tests/test_one_step_search_parsing_processor_llm.py::test_song_and_artist_parsed - AssertionError: assert None is not None
FAILED tests/test_two_step_search_parsing_processor_llm.py::test_song_and_artist_parsed - AssertionError: assert None is not None
```

It takes from 30s to 1m per call on macbook air m2 16GB.

Smaller models tend to fail structured output and tool calls, smaller - worse. Interestingly, 0.6b somehow performs better then 1.4b and 4b in structuring output.

Results might differ for each run, smaller models are more random.

Next: 
- search other instruct and code tuned models
- optimise processors with structured llm output for smaller models
- alternative impl: minimize prompt complexity and overhead by ditching function call and schemas in favor of prompt with examples
- refactor prompts
- create a better benchmark with bigger dataset

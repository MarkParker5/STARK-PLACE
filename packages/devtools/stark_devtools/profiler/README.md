# STARK Profiler + Visualizer

A **dev-only** tool that traces the whole STARK engine at runtime and visualizes it. It lives
**outside** the `stark/` package — nothing here ships with the engine.

## How it works

Capture is **interpreter-level** via `sys.monitoring` (PEP 669): CPython fires a callback on every
function call/return, so there is **no method wrapping, no registry, and no edit to any stark source
file**. The single decision about *what* gets profiled is a path filter — is the running code object
inside the stark package directory. Everything else (deps, libs, stdlib) is excluded for free. This
is "swizzle all of STARK", but the interpreter does the swizzling.

`sys.monitoring` (not `sys.setprofile`/`cProfile`) because it is interpreter-**global** — it also
sees the worker threads `asyncer.asyncify` spins up for sync commands — and it costs **nothing when
no tool is registered**, so normal runtime is untouched.

```
engine calls ──sys.monitoring──▶ Capture ──emit──▶ ProfilerBus (thread-safe) ──drain──▶ Sinks
                                                                                          ├─ MemorySink
                                                                                          ├─ JSONLSink (recording)
                                                                                          └─ CallbackSink (stream)
```

## Python API

```python
from profiler import profile, JSONLSink

with profile(JSONLSink("trace.jsonl")) as session:      # zero overhead when not active
    await context.process_string("play Metallica and turn off the kitchen lights")

for event in session.events:                            # in-memory by default too
    print(event.symbol, event.dur_ns, event.data)
```

`ProfileEvent` (stable schema, `SCHEMA_VERSION`): `trace_id, seq, t_ns, phase, symbol, module, depth,
thread, dur_ns, data`. Input payload on `call`, output on `return` — structured by a STARK-type-aware
serializer that never consumes generators.

## Modules

| Module | Role |
|--------|------|
| `poc.py` | self-contained single-file reference (the minimal idea) |
| `schema.py` | `ProfileEvent` + `SCHEMA_VERSION` |
| `capture.py` | the `sys.monitoring` swizzle (per-thread depth, frame-id durations, guards) |
| `bus.py` | thread-safe runtime bus (SimpleQueue + daemon drain, batches off the hot path) |
| `sinks.py` | `MemorySink` / `JSONLSink` / `CallbackSink` |
| `extractors.py` | structured input/output payloads per meaningful frame |
| `curate.py` | collapse the raw call graph into ~14 semantic **steps** |
| `structure.py` | AST introspection → the auto-graph (weights from calls + public/private methods + OOP relations) |
| `replay.py` | load / print a recorded trace |
| `server.py` | dependency-free stdlib server (serves the visualizer + `/api`) |

## CLI

```bash
python -m profiler                    # trace the demo utterance, pretty-print input+output
python -m profiler --jsonl t.jsonl    # record
python -m profiler.replay t.jsonl     # curated semantic steps  (--full for the whole graph)
python -m profiler.structure t.jsonl  # auto-graph node weights
python -m profiler.server             # http://localhost:8765  (serves visualizer/dist + /api)
```

## Visualizer

`../visualizer/` — a self-contained Vite + React + TypeScript + SVG app (decoupled from STARK; it only
depends on the event schema, so it can move to STARK-PLACE or its own repo later).

```bash
cd visualizer && npm install && npm run build   # served by `python -m profiler.server`
# or, for development:
npm run dev                                       # proxies /api to :8765
```

Three views, one step-based replay (events are sub-second → replay by step, not seconds):

- **Dashboard** — the curated semantic steps as banded I/O cards (design `12a`).
- **Wiring** — a small semantic graph derived from the steps (no hardcoded coordinates).
- **Brain (auto)** — a **purely generic** force graph of *every* stark class. Nothing hardcoded:
  node weight = `calls + 3·public methods + private methods + 2·OOP relations`; heavier nodes are
  pulled to the centre, more-linked nodes pulled together; color = module group; trace-active nodes
  light up during replay. Per-group opacity + on/off toggles; edit mode drags + persists positions.

Type an utterance in the header and hit **run** to profile it live.

## Tests

```bash
pytest tests/test_profiling      # capture safety, worker-thread, bus load, schema, curate, structure
```

## Known simplifications

- `depth` is per-thread and approximate across interleaved coroutines (durations stay accurate via
  frame-id matching; step pairing is depth-independent).
- The server profiles one utterance at a time (`sys.monitoring` has a single profiler slot).
- Corrections/NER/multilingual steps appear when those processors + their deps (espeak/spacy) are
  configured; the default demo is dependency-free.

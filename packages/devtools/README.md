# stark-devtools

Dev tooling for the [S.T.A.R.K. engine](https://github.com/MarkParker5/STARK):
a zero-overhead **profiler** (interpreter-level capture via `sys.monitoring`) and
a **web visualizer** (replay, dashboard, wiring, and brain views).

```bash
pip install stark-devtools --find-links https://markparker5.github.io/STARK-PLACE/
```

Import as `stark_devtools`. Requires **Python ≥ 3.12** (PEP 669 `sys.monitoring`)
and `stark-engine`.

## Quick start

**Trace the built-in demo utterance** (captures a real run, pretty-prints it):

```bash
python -m stark_devtools.profiler
```

**Launch the web visualizer** (serves the bundled UI + trace API):

```bash
python -m stark_devtools.profiler.server        # http://localhost:8765
```

**Profile your own code** — wrap any STARK run in a `profile()` session:

```python
from stark_devtools.profiler import profile, JSONLSink

with profile(JSONLSink("trace.jsonl")):
    await context.process_string("play Metallica and turn off the kitchen lights")
# -> trace.jsonl, replayable in the visualizer
```

Sinks: `MemorySink` (in-process), `JSONLSink` (file), `CallbackSink` (stream).
Capture is filtered to code inside the `stark` package and costs nothing when no
`profile()` session is active.

## Rebuilding the visualizer (optional)

The wheel ships a prebuilt `visualizer/dist`, so `server` works out of the box.
To hack on the UI:

```bash
cd stark_devtools/visualizer
npm install
npm run dev      # live dev server
npm run build    # refresh the bundled dist/ the profiler server serves
```

The visualizer's only coupling to STARK is `src/schema.ts` (a mirror of
`profiler/schema.py`); everything else consumes the trace over HTTP/JSON.

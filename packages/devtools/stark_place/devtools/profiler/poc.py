"""Minimal STARK profiler PoC — interpreter-level tracing via sys.monitoring (PEP 669).

Profiles ALL of STARK's own code and nothing else. The single decision about "what gets
profiled" is a path filter: if a called code object's source file lives inside the stark
package directory, we record it; every dependency, library and stdlib call is excluded for
free. This is the "swizzle all of STARK" idea (cf. Bryce Bostwick, *Swizzle All of UIKit*) —
except the CPython interpreter does the swizzling for us: `sys.monitoring` fires a callback on
every function call/return, so there is no method wrapping, no registry, and no edit to any
stark source file.

Why sys.monitoring and not sys.setprofile/cProfile:
  * it is interpreter-*global*, so it also sees the worker threads that `asyncer.asyncify`
    spins up for sync commands (sys.setprofile is per-thread and would miss them);
  * it has zero cost when no tool is registered — so normal runtime is untouched;
  * per-event callbacks (PY_START / PY_RETURN / ...) give us args and return values directly.

Payloads (input + output):
  * on PY_START we read the just-started frame's `f_locals` — the call's INPUT arguments;
  * on PY_RETURN we read the return value — the call's OUTPUT.
  A small `EXTRACTORS` table gives clean, structured payloads for the meaningful engine frames
  (process_string, the processors, PatternParser.match, run_command, respond, Response, ...);
  everything else falls back to a generic structured capture. A `_ser` serializer understands
  STARK's own types (MatchResult, SearchResult, Response, Correction, Object, ...).

Safety:
  * a thread-local RE-ENTRANCY GUARD makes the callbacks ignore any stark frames triggered by
    our own payload serialization (e.g. an object's __repr__), so the serializer never pollutes
    the trace or recurses;
  * a GENERATOR GUARD means we never iterate a lazy return value (e.g. Dictionary
    .search_in_sentence yields a generator) — consuming it would corrupt the engine run.

This is a dev tool. It deliberately lives OUTSIDE the `stark/` package so nothing here ever
ships with the engine.

Run it:
    python -m profiler.poc                     # depth-indented trace w/ input+output payloads
    python -m profiler.poc --only-core         # narrow the filter to stark/core
    python -m profiler.poc --jsonl out.jsonl   # also write a replayable recording

Known PoC simplifications (the real version fixes these): the event list / `depth` / `seq`
counters are not thread-synchronised (fine for a single utterance, but asyncify runs commands on
worker threads); async interleaving makes `depth` approximate.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import types
from typing import Any, Callable

import stark

# The one decision: is this code object part of stark's own implementation?
_STARK_DIR = os.path.dirname(os.path.abspath(stark.__file__))
_TID = sys.monitoring.PROFILER_ID

# Re-entrancy guard: while a callback is building a payload, ignore the stark frames that our own
# str()/repr()/attribute access may trigger. Thread-local so it is correct across asyncify workers.
_guard = threading.local()


# --------------------------------------------------------------------------------------------------
# Serialization: turn a STARK value into a compact, JSON-friendly structure for the trace payload.
# Defensive (never raises), depth/'breadth'-limited, and NEVER consumes generators/iterators.
# --------------------------------------------------------------------------------------------------

def _short(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _ser(v: Any, _depth: int = 0) -> Any:
    try:
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, str):
            return _short(v, 120)
        if isinstance(v, type):  # a class passed as an argument (e.g. object_type=Word)
            return v.__name__
        # never consume a lazy result — that would change engine behaviour
        if isinstance(v, (types.GeneratorType, map, filter, zip)) or (
            hasattr(v, "__next__") and not isinstance(v, (list, tuple, set, dict))
        ):
            return f"<{type(v).__name__} (not consumed)>"
        if _depth >= 4:
            return _short(str(v))
        if isinstance(v, dict):
            return {str(k): _ser(val, _depth + 1) for k, val in list(v.items())[:8]}
        if isinstance(v, (list, tuple, set)):
            return [_ser(x, _depth + 1) for x in list(v)[:8]]

        name = type(v).__name__

        if name == "MatchResult":
            return {
                "substring": str(v.substring),
                "start": v.start,
                "end": v.end,
                "parameters": {k: _ser(getattr(o, "value", o), _depth + 1) for k, o in v.parameters.items()},
                "corrections": [f"{c.correction.variant}→{c.correction.keyword}" for c in getattr(v, "corrections", [])],
                "corrected_string": getattr(v, "corrected_string", ""),
            }
        if name == "SearchResult":
            return {"command": v.command.name, "index": v.index, "match": _ser(v.match_result, _depth + 1)}
        if name == "ParseResult":
            return {"obj": _ser(v.obj, _depth + 1), "substring": str(v.substring)}
        if name == "ParameterMatch":
            return {"name": v.name, "value": _ser(getattr(v.parsed_obj, "value", v.parsed_obj), _depth + 1)}
        if name == "Response":
            return {
                "text": str(v.text),
                "voice": str(v.voice),
                "status": getattr(v.status, "name", str(v.status)),
                "commands": [c.name for c in v.commands],
                "parameters": _ser(v.parameters, _depth + 1),
            }
        if name == "Correction":
            return f"{v.variant}→{v.keyword}"
        if name == "CorrectionMatch":
            return f"{v.correction.variant}→{v.correction.keyword}"
        if name == "LookupResult":
            return {"span": [v.span.start, v.span.end], "item": v.item.name}
        if name == "DictionaryItem":
            return {"name": v.name, "simple_phonetic": v.simple_phonetic}
        if name == "CommandsContextLayer":
            return {"commands": [c.name for c in v.commands], "parameters": _ser(v.parameters, _depth + 1)}
        if name == "RecognizedEntity":
            return {"substring": v.substring, "type": getattr(v.type, "__name__", str(v.type))}
        if name == "Command":
            return v.name
        if name == "Pattern":
            return getattr(v, "_origin", str(v))

        mod = getattr(type(v), "__module__", "") or ""
        if mod.startswith("stark") and hasattr(v, "value"):  # an Object subclass (Word, String, ...)
            return {"type": name, "value": _ser(getattr(v, "value"), _depth + 1)}

        return _short(str(v))
    except Exception as exc:  # serialization must never break a trace
        return f"<unser {type(v).__name__}: {exc}>"


# --------------------------------------------------------------------------------------------------
# Extractors: clean INPUT (from frame locals) and OUTPUT (from return value) for meaningful frames.
# Keyed by qualname. Each value may have "in": f(locals)->dict and/or "out": f(retval)->dict.
# Frames without an entry fall back to a generic structured capture.
# --------------------------------------------------------------------------------------------------

def _cmd_names(commands) -> list[str]:
    return [c.name for c in commands or []]


_proc_in = lambda L: {"string": str(L.get("string")), "entities": _ser(L.get("recognized_entities"))}
_proc_out = lambda r: (
    {"results": _ser(r[0]), "pops": r[1]} if isinstance(r, tuple) else {"results": _ser(r)}
)

EXTRACTORS: dict[str, dict[str, Callable[[Any], dict]]] = {
    "CommandsContext.process_string": {
        "in": lambda L: {"string": str(L.get("string"))},
        "out": lambda r: {"results": _ser(r)},
    },
    "CommandsContextProcessor.process_string": {"in": _proc_in, "out": _proc_out},
    "CorrectionsProcessor.process_string": {"in": _proc_in, "out": _proc_out},
    "SpacyNERProcessor.process_string": {"in": _proc_in, "out": _proc_out},
    "SearchProcessor.process_context_layer": {
        "in": lambda L: {"commands": _cmd_names(getattr(L.get("context_layer"), "commands", None))},
        "out": lambda r: {"results": _ser(r)},
    },
    "SearchProcessor.search": {
        "in": lambda L: {"string": str(L.get("string")), "commands": _cmd_names(L.get("commands"))},
        "out": lambda r: {"results": _ser(r)},
    },
    "SearchProcessor._match_commands": {
        "in": lambda L: {"string": str(L.get("string")), "lang": L.get("language_code")},
        "out": lambda r: {"results": _ser(r)},
    },
    "PatternParser.match": {
        "in": lambda L: {"pattern": getattr(L.get("pattern"), "_origin", str(L.get("pattern"))), "string": str(L.get("string"))},
        "out": lambda r: {"matches": _ser(r)},
    },
    "PatternParser.parse_object": {
        "in": lambda L: {"type": getattr(L.get("object_type"), "__name__", None), "string": str(L.get("from_string"))},
        "out": lambda r: {"result": _ser(r)},
    },
    "PatternParser._parse_single_parameter": {
        "in": lambda L: {"name": L.get("parameter_name"), "substring": str(L.get("raw_param_substr"))},
        "out": lambda r: {"param": _ser(r)},
    },
    "Dictionary.search_in_sentence": {  # returns a generator — INPUT only, never consume the output
        "in": lambda L: {"sentence": str(L.get("sentence")), "mode": str(L.get("mode"))},
    },
    "CommandsContext.run_command": {
        "in": lambda L: {"command": getattr(L.get("command"), "name", None), "parameters": _ser(L.get("parameters"))},
    },
    "CommandsContext.respond": {"in": lambda L: {"response": _ser(L.get("response"))}},
    "CommandsContext._process_response": {"in": lambda L: {"response": _ser(L.get("response"))}},
    "CommandsContext.add_context": {"in": lambda L: {"context": _ser(L.get("context"))}},
    "Response.__init__": {
        "in": lambda L: {"args": _ser(L.get("args")), "kwargs": _ser(L.get("kwargs"))},
    },
    "VoiceAssistant.speech_recognizer_did_receive_final_result": {
        "in": lambda L: {"result": str(L.get("result"))},
    },
    "VoiceAssistant._play_response": {"in": lambda L: {"response": _ser(L.get("response"))}},
}


def _generic_in(frame) -> dict:
    return {k: _ser(v) for k, v in frame.f_locals.items() if k != "self"}


# --------------------------------------------------------------------------------------------------
# The tracer.
# --------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def trace(root: str = _STARK_DIR, jsonl: str | None = None):
    """Record every stark call/return inside the `with` block, with input+output payloads.

    Yields the growing list of event dicts:
        {seq, t_ns, phase("→"/"←"), symbol(qualname), module(rel path), depth, data}
    """
    events: list[dict] = []
    seq = [0]
    depth = [0]
    mon = sys.monitoring

    def want(code) -> bool:
        return code.co_filename.startswith(root)

    def rel(code) -> str:
        return os.path.relpath(code.co_filename, _STARK_DIR)

    def on_start(code, instruction_offset):
        if not want(code) or getattr(_guard, "on", False):
            return
        _guard.on = True
        try:
            qual = code.co_qualname
            frame = sys._getframe(1)
            ex = EXTRACTORS.get(qual)
            if ex and "in" in ex:
                try:
                    payload = ex["in"](frame.f_locals)
                except Exception as exc:
                    payload = {"_err": str(exc)}
            else:
                payload = _generic_in(frame)
            depth[0] += 1
            seq[0] += 1
            events.append({
                "seq": seq[0], "t_ns": time.perf_counter_ns(), "phase": "→",
                "symbol": qual, "module": rel(code), "depth": depth[0], "data": payload,
            })
        finally:
            _guard.on = False

    def on_return(code, instruction_offset, retval):
        if not want(code) or getattr(_guard, "on", False):
            return
        _guard.on = True
        try:
            qual = code.co_qualname
            ex = EXTRACTORS.get(qual)
            if ex and "out" in ex:
                try:
                    payload = ex["out"](retval)
                except Exception as exc:
                    payload = {"_err": str(exc)}
            else:
                payload = {"return": _ser(retval)}
            seq[0] += 1
            events.append({
                "seq": seq[0], "t_ns": time.perf_counter_ns(), "phase": "←",
                "symbol": qual, "module": rel(code), "depth": depth[0], "data": payload,
            })
            depth[0] -= 1
        finally:
            _guard.on = False

    mon.use_tool_id(_TID, "stark-poc")
    try:
        mon.register_callback(_TID, mon.events.PY_START, on_start)
        mon.register_callback(_TID, mon.events.PY_RETURN, on_return)
        mon.set_events(_TID, mon.events.PY_START | mon.events.PY_RETURN)
        try:
            yield events
        finally:
            mon.set_events(_TID, 0)
            mon.register_callback(_TID, mon.events.PY_START, None)
            mon.register_callback(_TID, mon.events.PY_RETURN, None)
    finally:
        mon.free_tool_id(_TID)

    if jsonl:
        with open(jsonl, "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in events)


# --------------------------------------------------------------------------------------------------
# Demo / CLI.
# --------------------------------------------------------------------------------------------------

def _fmt(data: dict) -> str:
    return ", ".join(f"{k}={data[k]}" for k in data)


def _print_trace(events: list[dict]) -> None:
    for e in events:
        indent = "  " * (e["depth"] - 1)
        if e["phase"] == "→":
            print(f"{indent}→ \x1b[1m{e['symbol']}\x1b[0m  \x1b[36m{_fmt(e['data'])}\x1b[0m  \x1b[2m{e['module']}\x1b[0m")
        else:
            print(f"{indent}← \x1b[2m{e['symbol']}\x1b[0m  \x1b[33m{_fmt(e['data'])}\x1b[0m")


async def _demo(root: str, jsonl: str | None) -> None:
    import asyncer

    from stark.core import CommandsContext, CommandsManager, Response

    manager = CommandsManager()

    @manager.new("play $band:Word")
    async def play_music(band):
        return Response(f"Playing {band}.")

    @manager.new("turn off the ** lights")
    async def lights_off():
        return Response("Lights off.")

    utterance = "play Metallica and turn off the kitchen lights"

    async with asyncer.create_task_group() as task_group:
        # build the engine outside the trace so it focuses on process_string, not construction
        context = CommandsContext(task_group=task_group, commands_manager=manager)
        with trace(root=root, jsonl=jsonl) as events:
            await context.process_string(utterance)

    starts = sum(1 for e in events if e["phase"] == "→")
    scope = os.path.relpath(root, _STARK_DIR) or "stark/"
    print(f"\n\x1b[1m=== {starts} stark calls for \"{utterance}\"  (filter: {scope}) ===\x1b[0m\n")
    _print_trace(events)
    if jsonl:
        print(f"\n\x1b[2mrecording written to {jsonl}\x1b[0m")


def main() -> None:
    import anyio

    root = _STARK_DIR
    jsonl = None
    argv = sys.argv[1:]
    if "--only-core" in argv:
        root = os.path.join(_STARK_DIR, "core")
    if "--jsonl" in argv:
        jsonl = argv[argv.index("--jsonl") + 1]

    anyio.run(_demo, root, jsonl)


if __name__ == "__main__":
    main()

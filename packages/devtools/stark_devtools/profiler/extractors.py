"""Payload extraction for the STARK profiler.

`serialize` turns any STARK value into a compact, JSON-friendly structure — it understands the
engine's own types (MatchResult, SearchResult, Response, Correction, Object subclasses, ...) and is
defensive (never raises), depth/breadth-limited, and NEVER consumes a generator/iterator (doing so
would change engine behaviour, e.g. Dictionary.search_in_sentence yields lazily).

`EXTRACTORS` maps a qualname to clean INPUT (from frame locals) and/or OUTPUT (from the return value)
extractors for the meaningful engine frames. Frames without an entry fall back to `generic_input` /
a generic serialize of the return value.
"""

from __future__ import annotations

import types
from typing import Any, Callable


def _short(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def serialize(v: Any, _depth: int = 0) -> Any:
    """Best-effort, side-effect-free serialization of a STARK value."""
    try:
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, str):
            return _short(v)
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
            return {str(k): serialize(val, _depth + 1) for k, val in list(v.items())[:8]}
        if isinstance(v, (list, tuple, set)):
            return [serialize(x, _depth + 1) for x in list(v)[:8]]

        name = type(v).__name__

        if name == "MatchResult":
            return {
                "substring": str(v.substring),
                "start": v.start,
                "end": v.end,
                "parameters": {k: serialize(getattr(o, "value", o), _depth + 1) for k, o in v.parameters.items()},
                "corrections": [f"{c.correction.variant}→{c.correction.keyword}" for c in getattr(v, "corrections", [])],
                "corrected_string": getattr(v, "corrected_string", ""),
            }
        if name == "SearchResult":
            return {"command": v.command.name, "index": v.index, "match": serialize(v.match_result, _depth + 1)}
        if name == "ParseResult":
            return {"obj": serialize(v.obj, _depth + 1), "substring": str(v.substring)}
        if name == "ParameterMatch":
            return {"name": v.name, "value": serialize(getattr(v.parsed_obj, "value", v.parsed_obj), _depth + 1)}
        if name == "Response":
            return {
                "text": str(v.text),
                "voice": str(v.voice),
                "status": getattr(v.status, "name", str(v.status)),
                "commands": [c.name for c in v.commands],
                "parameters": serialize(v.parameters, _depth + 1),
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
            return {"commands": [c.name for c in v.commands], "parameters": serialize(v.parameters, _depth + 1)}
        if name == "RecognizedEntity":
            return {"substring": v.substring, "type": getattr(v.type, "__name__", str(v.type))}
        if name == "Command":
            return v.name
        if name == "Pattern":
            return getattr(v, "_origin", str(v))

        mod = getattr(type(v), "__module__", "") or ""
        if mod.startswith("stark") and hasattr(v, "value"):  # an Object subclass (Word, String, ...)
            return {"type": name, "value": serialize(getattr(v, "value"), _depth + 1)}

        return _short(str(v))
    except Exception as exc:  # serialization must never break a trace
        return f"<unser {type(v).__name__}: {exc}>"


def _cmd_names(commands) -> list[str]:
    return [c.name for c in commands or []]


_proc_in = lambda L: {"string": str(L.get("string")), "entities": serialize(L.get("recognized_entities"))}
_proc_out = lambda r, L: (
    {"results": serialize(r[0]), "pops": r[1]} if isinstance(r, tuple) else {"results": serialize(r)}
)

# walk up the call stack from CommandsContext.respond to the command_task closure that yielded
# this response, so each response is attributed to the command that produced it (handles bg
# commands that emit several responses over time).
def _emitting_command(frame):
    f = getattr(frame, "f_back", None)
    depth = 0
    while f is not None and depth < 10:
        cmd = f.f_locals.get("command")
        name = getattr(cmd, "name", None)
        if name:
            return name
        f = f.f_back
        depth += 1
    return None


# qualname -> {"in": f(frame_locals)->dict, "out": f(retval)->dict}
EXTRACTORS: dict[str, dict[str, Callable[[Any], dict]]] = {
    "CommandsContext.process_string": {
        "in": lambda L: {"string": str(L.get("string"))},
        "out": lambda r, L: {"results": serialize(r)},
    },
    "CommandsContextProcessor.process_string": {"in": _proc_in, "out": _proc_out},
    # corrections/NER return ([],0) but mutate shared state — read the produced results from locals
    "CorrectionsProcessor.process_string": {
        "in": lambda L: {"string": str(L.get("string"))},
        "out": lambda r, L: {"corrections": [f"{c.variant}→{c.keyword}" for c in getattr(L.get("string"), "corrections", [])] or "none"},
    },
    "SpacyNERProcessor.process_string": {
        "in": lambda L: {"string": str(L.get("string"))},
        "out": lambda r, L: {"entities": [f"{e.substring}:{getattr(e.type, '__name__', e.type)}" for e in (L.get("recognized_entities") or [])] or "none"},
    },
    "SearchProcessor.process_context_layer": {
        "in": lambda L: {"commands": _cmd_names(getattr(L.get("context_layer"), "commands", None))},
        "out": lambda r, L: {"results": serialize(r)},
    },
    "SearchProcessor.search": {
        "in": lambda L: {"string": str(L.get("string")), "commands": _cmd_names(L.get("commands"))},
        "out": lambda r, L: {"results": serialize(r)},
    },
    "SearchProcessor._match_commands": {
        "in": lambda L: {"string": str(L.get("string")), "lang": L.get("language_code")},
        "out": lambda r, L: {"results": serialize(r)},
    },
    "PatternParser.match": {
        "in": lambda L: {"pattern": getattr(L.get("pattern"), "_origin", str(L.get("pattern"))), "string": str(L.get("string"))},
        "out": lambda r, L: {"matches": serialize(r)},
    },
    "PatternParser.parse_object": {
        "in": lambda L: {"type": getattr(L.get("object_type"), "__name__", None), "string": str(L.get("from_string"))},
        "out": lambda r, L: {"result": serialize(r)},
    },
    "PatternParser._parse_single_parameter": {
        "in": lambda L: {"name": L.get("parameter_name"), "substring": str(L.get("raw_param_substr"))},
        "out": lambda r, L: {"param": serialize(r)},
    },
    "Dictionary.search_in_sentence": {  # returns a generator — INPUT only, never consume the output
        "in": lambda L: {"sentence": str(L.get("sentence")), "mode": str(L.get("mode"))},
    },
    "CommandsContext.run_command": {
        "in": lambda L: {"command": getattr(L.get("command"), "name", None), "parameters": serialize(L.get("parameters"))},
    },
    "Command.run": {
        "in": lambda L: {"command": getattr(L.get("self"), "name", None), "parameters": serialize(L.get("parameters_dict") or L.get("kwparameters"))},
    },
    "CommandsContext.respond": {"wants_frame": True, "in": lambda L, F: {"response": serialize(L.get("response")), "command": _emitting_command(F)}},
    "CommandsContext._process_response": {"in": lambda L: {"response": serialize(L.get("response"))}},
    "CommandsContext.add_context": {"in": lambda L: {"context": serialize(L.get("context"))}},
    "Response.__init__": {"in": lambda L: {"args": serialize(L.get("args")), "kwargs": serialize(L.get("kwargs"))}},
    "VoiceAssistant.speech_recognizer_did_receive_final_result": {"in": lambda L: {"result": str(L.get("result"))}},
    "VoiceAssistant._play_response": {"in": lambda L: {"response": serialize(L.get("response"))}},
}


def generic_input(frame) -> dict:
    return {k: serialize(v) for k, v in frame.f_locals.items() if k != "self"}

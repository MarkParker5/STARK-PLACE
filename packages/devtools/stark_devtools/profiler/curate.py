"""Curation: collapse a raw trace into the handful of semantic steps a human cares about.

The capture layer records *everything* stark does (hundreds of calls per utterance) — that is the
right default (nothing hardcoded, the viewer decides what to show). Curation is the opposite end:
a small, declarative set of milestone frames, paired call→return into `Step`s with input + output +
duration. The dashboard renders these steps; the full graph stays available underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import CALL, RETURN, ProfileEvent

# The semantic milestones — the story of one utterance. Order here is the canonical step order.
MILESTONES: list[str] = [
    "VoiceAssistant.speech_recognizer_did_receive_final_result",
    "CommandsContext.process_string",
    "CorrectionsProcessor.process_string",
    "DemoCorrectionsProcessor.process_string",
    "SpacyNERProcessor.process_string",
    "FallbackProcessor.process_string",
    "Dictionary.search_in_sentence",
    "SearchProcessor.search",
    "PatternParser.match",
    "CommandsContextProcessor.process_string",
    "CommandsContext.run_command",
    "Command.run",
    "CommandsContext.respond",
    "CommandsContext._process_response",
    "CommandsContext.add_context",
    "VoiceAssistant._play_response",
]

# A friendlier label + a group for each milestone (groups drive dashboard bands / colors).
STEP_META: dict[str, dict[str, str]] = {
    "VoiceAssistant.speech_recognizer_did_receive_final_result": {"label": "heard", "group": "io_in"},
    "CommandsContext.process_string": {"label": "process", "group": "engine"},
    "CorrectionsProcessor.process_string": {"label": "corrections", "group": "processors"},
    "SpacyNERProcessor.process_string": {"label": "NER", "group": "processors"},
    "FallbackProcessor.process_string": {"label": "fallback", "group": "processors"},
    "Dictionary.search_in_sentence": {"label": "dictionary lookup", "group": "phonetics"},
    "SearchProcessor.search": {"label": "search", "group": "processors"},
    "PatternParser.match": {"label": "pattern match", "group": "matching"},
    "CommandsContextProcessor.process_string": {"label": "processor pass", "group": "processors"},
    "CommandsContext.run_command": {"label": "dispatch", "group": "execution"},
    "Command.run": {"label": "command", "group": "execution"},
    "CommandsContext.respond": {"label": "respond", "group": "execution"},
    "CommandsContext._process_response": {"label": "handle response", "group": "engine"},
    "CommandsContext.add_context": {"label": "push context", "group": "engine"},
    "VoiceAssistant._play_response": {"label": "speak", "group": "io_out"},
}


@dataclass
class Step:
    seq: int                       # the CALL seq (order)
    trace_id: str
    symbol: str
    label: str
    group: str
    depth: int
    input: dict[str, Any]
    output: dict[str, Any] = field(default_factory=dict)
    dur_ns: int | None = None
    error: str | None = None
    t_ns: int = 0                  # CALL timestamp
    end_ns: int = 0                # RETURN timestamp (t_ns + dur, for parallel detection)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "trace_id": self.trace_id, "symbol": self.symbol,
            "label": self.label, "group": self.group, "depth": self.depth,
            "input": self.input, "output": self.output, "dur_ns": self.dur_ns, "error": self.error,
            "t_ns": self.t_ns, "end_ns": self.end_ns,
        }


def curate(events: list[ProfileEvent], milestones: list[str] | None = None) -> list[Step]:
    """Pair milestone CALL/RETURN events into ordered `Step`s.

    Matching is by (symbol, thread, depth) using a per-key stack, so nested/repeated milestones
    (e.g. PatternParser.match called per command) each get their own step.
    """
    keep = set(milestones or MILESTONES)
    by_id: dict[int, ProfileEvent] = {}                       # call_id -> its CALL event (exact pairing)
    open_calls: dict[tuple, list[ProfileEvent]] = {}          # fallback stack for pre-call_id traces
    steps: list[Step] = []

    for e in events:
        if e.symbol not in keep:
            continue
        # Pair a RETURN with its OWN call by id(frame) — correct even when coroutines of the same
        # symbol interleave on one thread (a LIFO stack would swap their inputs/outputs). Fall back
        # to the (symbol, thread) stack only for legacy traces that predate call_id.
        key = (e.symbol, e.thread)
        if e.phase == CALL:
            if e.call_id:
                by_id[e.call_id] = e
            open_calls.setdefault(key, []).append(e)
        elif e.phase in (RETURN, "error"):
            call = None
            if e.call_id and e.call_id in by_id:
                call = by_id.pop(e.call_id)
                stack = open_calls.get(key)
                if stack and call in stack:
                    stack.remove(call)
            else:
                stack = open_calls.get(key)
                if not stack:
                    continue
                call = stack.pop()
            meta = STEP_META.get(e.symbol, {"label": e.symbol, "group": "other"})
            steps.append(Step(
                seq=call.seq,
                trace_id=call.trace_id,
                symbol=call.symbol,
                label=meta["label"],
                group=meta["group"],
                depth=call.depth,
                input=call.data,
                output=e.data if e.phase == RETURN else {},
                dur_ns=e.dur_ns,
                error=(e.data.get("exception") if e.phase == "error" else None),
                t_ns=call.t_ns,
                end_ns=e.t_ns,
            ))

    steps.sort(key=lambda s: s.seq)
    return steps

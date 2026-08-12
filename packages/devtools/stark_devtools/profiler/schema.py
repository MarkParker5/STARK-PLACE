"""Stable event schema for the STARK profiler.

One `ProfileEvent` is emitted per traced call boundary (a stark function starting or returning).
The schema is intentionally small and flat so it survives JSONL recording, WebSocket streaming and
the visualizer's replay model unchanged. Bump `SCHEMA_VERSION` on any breaking change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 1

# phase values
CALL = "call"
RETURN = "return"
ERROR = "error"


@dataclass(slots=True)
class ProfileEvent:
    trace_id: str          # one utterance / top-level engine entry
    seq: int               # globally monotonic, total order across threads
    t_ns: int              # time.perf_counter_ns() at the boundary
    phase: str             # CALL | RETURN | ERROR
    symbol: str            # code.co_qualname, e.g. "PatternParser.match"
    module: str            # stark-relative path, e.g. "core/parsing.py"
    depth: int             # per-thread call nesting (1 = top)
    thread: int            # thread id the boundary happened on
    dur_ns: int | None     # RETURN/ERROR only: wall time since the matching CALL
    data: dict[str, Any]   # structured payload (input on CALL, output on RETURN)
    call_id: int = 0       # id(frame): identical on a CALL and its own RETURN — exact pairing
                           # even when coroutines of the same symbol interleave on one thread

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "seq": self.seq,
            "t_ns": self.t_ns,
            "phase": self.phase,
            "symbol": self.symbol,
            "module": self.module,
            "depth": self.depth,
            "thread": self.thread,
            "dur_ns": self.dur_ns,
            "data": self.data,
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProfileEvent:
        return cls(
            trace_id=d["trace_id"],
            seq=d["seq"],
            t_ns=d["t_ns"],
            phase=d["phase"],
            symbol=d["symbol"],
            module=d["module"],
            depth=d["depth"],
            thread=d["thread"],
            dur_ns=d.get("dur_ns"),
            data=d.get("data", {}),
            call_id=d.get("call_id", 0),
        )

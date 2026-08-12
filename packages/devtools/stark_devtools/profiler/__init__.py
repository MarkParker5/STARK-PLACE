"""STARK profiler — a dev-only tool that traces the whole engine at runtime.

It profiles ALL of STARK's own code via `sys.monitoring` (PEP 669) — no method wrapping, no registry,
no edits to any stark source file, and zero overhead when not active. Events flow over a thread-safe
runtime bus to swappable sinks.

    from profiler import profile, JSONLSink

    with profile(JSONLSink("trace.jsonl")) as session:
        await context.process_string("play Metallica and turn off the kitchen lights")

    for event in session.events:      # in-memory too, by default
        print(event.symbol, event.data)

This package lives OUTSIDE the `stark/` package so nothing here ships with the engine.
"""

from .capture import STARK_DIR, Capture
from .bus import ProfilerBus
from .schema import CALL, ERROR, RETURN, SCHEMA_VERSION, ProfileEvent
from .session import Session, profile
from .sinks import CallbackSink, JSONLSink, MemorySink, Sink

__all__ = [
    "profile",
    "Session",
    "ProfileEvent",
    "SCHEMA_VERSION",
    "CALL",
    "RETURN",
    "ERROR",
    "Sink",
    "MemorySink",
    "JSONLSink",
    "CallbackSink",
    "ProfilerBus",
    "Capture",
    "STARK_DIR",
]

"""Session wiring: capture → bus → sinks, exposed as a single `profile()` context manager."""

from __future__ import annotations

import contextlib
from typing import Iterator

from .bus import ProfilerBus
from .capture import STARK_DIR, Capture
from .schema import ProfileEvent
from .sinks import MemorySink, Sink


class Session:
    """A live profiling session. Holds the attached sinks and offers convenience accessors."""

    def __init__(self, sinks: list[Sink], bus: ProfilerBus, capture: Capture) -> None:
        self.sinks = sinks
        self._bus = bus
        self._capture = capture

    @property
    def memory(self) -> MemorySink | None:
        for sink in self.sinks:
            if isinstance(sink, MemorySink):
                return sink
        return None

    @property
    def events(self) -> list[ProfileEvent]:
        """Events captured so far (requires a MemorySink; present by default). Only safe to read
        after the session's `with` block has exited, when the drain thread has flushed."""
        mem = self.memory
        return mem.events if mem else []


@contextlib.contextmanager
def profile(*sinks: Sink, root: str = STARK_DIR, memory: bool = True) -> Iterator[Session]:
    """Profile all of STARK for the duration of the `with` block.

        with profile() as session:
            await context.process_string("...")
        for event in session.events:
            ...

    Pass sinks to fan out (e.g. `profile(JSONLSink("trace.jsonl"))`). A `MemorySink` is added by
    default so `session.events` works; pass `memory=False` to opt out. `root` narrows the capture
    filter (e.g. to `stark/core`); module paths stay relative to the package root regardless.
    """
    sink_list: list[Sink] = list(sinks)
    if memory and not any(isinstance(s, MemorySink) for s in sink_list):
        sink_list.insert(0, MemorySink())

    bus = ProfilerBus()
    capture = Capture(bus.emit, root=root)

    bus.start(sink_list)          # start the drain before capturing so nothing is lost
    capture.enable()
    try:
        yield Session(sink_list, bus, capture)
    finally:
        capture.disable()         # stop generating events first
        bus.stop()                # then flush everything still queued and close the sinks

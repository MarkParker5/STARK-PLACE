"""Swappable sinks for the STARK profiler.

A sink receives batches of `ProfileEvent`s off the hot path (from the bus drain thread) and does
something with them: keep them in memory, append to a JSONL recording, hand each to a callback for
live streaming, etc. Sinks are plain and synchronous; anything async (e.g. a WebSocket broadcaster)
adapts at its own edge.

`write_batch` and `close` are always called from the single bus drain thread, so implementations do
not need their own locking.
"""

from __future__ import annotations

import json
from typing import Callable, Protocol, runtime_checkable

from .schema import ProfileEvent


@runtime_checkable
class Sink(Protocol):
    def write_batch(self, events: list[ProfileEvent]) -> None: ...
    def close(self) -> None: ...


class MemorySink:
    """Keep every event in a list. Handy for tests and the CLI pretty-printer."""

    def __init__(self) -> None:
        self.events: list[ProfileEvent] = []

    def write_batch(self, events: list[ProfileEvent]) -> None:
        self.events.extend(events)

    def close(self) -> None:
        pass


class JSONLSink:
    """Append one JSON object per line — the replayable recording the visualizer consumes."""

    def __init__(self, path: str) -> None:
        self._file = open(path, "w")

    def write_batch(self, events: list[ProfileEvent]) -> None:
        for event in events:
            self._file.write(json.dumps(event.to_dict()) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class CallbackSink:
    """Forward each event to a callable — the seam for live WebSocket streaming."""

    def __init__(self, callback: Callable[[ProfileEvent], None]) -> None:
        self._callback = callback

    def write_batch(self, events: list[ProfileEvent]) -> None:
        for event in events:
            self._callback(event)

    def close(self) -> None:
        pass

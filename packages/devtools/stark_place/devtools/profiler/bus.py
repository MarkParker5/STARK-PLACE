"""Thread-safe runtime event bus for the STARK profiler.

The engine emits many quick, sub-second events, and emission happens on whichever thread the code
runs on — the main event-loop thread AND the worker threads `asyncer.asyncify` spins up for sync
commands. So the emit path must be:

  * cheap and non-blocking — a single `queue.SimpleQueue.put`, which is a C-level, thread-safe,
    lock-free-enough operation that never blocks the caller;
  * decoupled from the event loop — a dedicated daemon drain thread pulls batches off the queue and
    hands them to the sinks, so no sink I/O (file writes, sockets) ever runs on the hot path or
    blocks the async loop.

`emit` is the only thing on the hot path. Draining, batching and sink dispatch all happen on the
drain thread. When there is no active session the capture layer never calls `emit`, so a non-running
bus costs nothing.
"""

from __future__ import annotations

import queue
import threading

from .schema import ProfileEvent
from .sinks import Sink


class ProfilerBus:
    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[ProfileEvent] = queue.SimpleQueue()
        self._sinks: list[Sink] = []
        self._thread: threading.Thread | None = None
        self._running = False

    # hot path — any thread
    def emit(self, event: ProfileEvent) -> None:
        self._queue.put(event)

    def start(self, sinks: list[Sink]) -> None:
        self._sinks = sinks
        self._running = True
        self._thread = threading.Thread(target=self._drain, name="stark-profiler-drain", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting drains, flush everything still queued, then close the sinks."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        for sink in self._sinks:
            sink.close()

    def _drain(self) -> None:
        # keep draining while running, and finish whatever is still queued after stop()
        while self._running or not self._queue.empty():
            try:
                batch = [self._queue.get(timeout=0.05)]
            except queue.Empty:
                continue
            while True:  # coalesce the current burst into one batch
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            for sink in self._sinks:
                sink.write_batch(batch)

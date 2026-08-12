"""Demo CLI for the STARK profiler package.

    python -m profiler                    # trace the demo utterance, pretty-print (input+output)
    python -m profiler --only-core        # narrow the capture filter to stark/core
    python -m profiler --jsonl out.jsonl  # also record a replayable JSONL file

This runs the real pipeline: capture (sys.monitoring) → thread-safe bus → sinks.
"""

from __future__ import annotations

import os
import sys

from .capture import STARK_DIR
from .schema import CALL, ERROR
from .session import profile
from .sinks import JSONLSink


def _fmt(data: dict) -> str:
    return ", ".join(f"{k}={data[k]}" for k in data)


def _print(events) -> None:
    for e in events:
        indent = "  " * max(0, e.depth - 1)
        if e.phase == CALL:
            print(f"{indent}→ \x1b[1m{e.symbol}\x1b[0m  \x1b[36m{_fmt(e.data)}\x1b[0m  \x1b[2m{e.module}\x1b[0m")
        elif e.phase == ERROR:
            print(f"{indent}✗ \x1b[31m{e.symbol}  {_fmt(e.data)}\x1b[0m")
        else:
            dur = f"  \x1b[2m{e.dur_ns/1000:.0f}µs\x1b[0m" if e.dur_ns else ""
            print(f"{indent}← \x1b[2m{e.symbol}\x1b[0m  \x1b[33m{_fmt(e.data)}\x1b[0m{dur}")


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
    extra_sinks = [JSONLSink(jsonl)] if jsonl else []

    async with asyncer.create_task_group() as task_group:
        context = CommandsContext(task_group=task_group, commands_manager=manager)
        with profile(*extra_sinks, root=root) as session:
            await context.process_string(utterance)

    events = session.events
    starts = sum(1 for e in events if e.phase == CALL)
    scope = os.path.relpath(root, STARK_DIR) or "stark/"
    print(f"\n\x1b[1m=== {starts} stark calls for \"{utterance}\"  (filter: {scope}) ===\x1b[0m\n")
    _print(events)
    total = events[-1].t_ns - events[0].t_ns if events else 0
    print(f"\n\x1b[2m{len(events)} events over {total/1000:.0f}µs on the bus"
          + (f"; recording written to {jsonl}" if jsonl else "") + "\x1b[0m")


def main() -> None:
    import anyio

    argv = sys.argv[1:]
    root = os.path.join(STARK_DIR, "core") if "--only-core" in argv else STARK_DIR
    jsonl = argv[argv.index("--jsonl") + 1] if "--jsonl" in argv else None
    anyio.run(_demo, root, jsonl)


if __name__ == "__main__":
    main()

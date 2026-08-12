"""Load and inspect recorded traces.

    python -m profiler.replay trace.jsonl            # curated semantic steps
    python -m profiler.replay trace.jsonl --full     # the whole call graph
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from .curate import curate
from .schema import CALL, ERROR, ProfileEvent


def load(path: str) -> list[ProfileEvent]:
    with open(path) as f:
        return [ProfileEvent.from_dict(json.loads(line)) for line in f if line.strip()]


def group_by_trace(events: list[ProfileEvent]) -> dict[str, list[ProfileEvent]]:
    grouped: dict[str, list[ProfileEvent]] = defaultdict(list)
    for e in events:
        grouped[e.trace_id].append(e)
    return dict(grouped)


def _fmt(data: dict) -> str:
    return ", ".join(f"{k}={data[k]}" for k in data)


def _print_full(events: list[ProfileEvent]) -> None:
    for e in events:
        indent = "  " * max(0, e.depth - 1)
        if e.phase == CALL:
            print(f"{indent}→ {e.symbol}  {_fmt(e.data)}  [{e.module}]")
        elif e.phase == ERROR:
            print(f"{indent}✗ {e.symbol}  {_fmt(e.data)}")
        else:
            dur = f"  {e.dur_ns/1000:.0f}µs" if e.dur_ns else ""
            print(f"{indent}← {e.symbol}  {_fmt(e.data)}{dur}")


def _print_steps(events: list[ProfileEvent]) -> None:
    for tid, evs in group_by_trace(events).items():
        print(f"\n=== trace {tid} ===")
        for i, step in enumerate(curate(evs), 1):
            dur = f"  {step.dur_ns/1000:.0f}µs" if step.dur_ns else ""
            print(f"{i:2}. [{step.group}] {step.label} ({step.symbol}){dur}")
            if step.input:
                print(f"      in  {_fmt(step.input)}")
            if step.output:
                print(f"      out {_fmt(step.output)}")


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print("usage: python -m profiler.replay <trace.jsonl> [--full]")
        raise SystemExit(2)
    path = argv[0]
    events = load(path)
    print(f"{len(events)} events, {len(group_by_trace(events))} trace(s)")
    if "--full" in argv:
        _print_full(events)
    else:
        _print_steps(events)


if __name__ == "__main__":
    main()

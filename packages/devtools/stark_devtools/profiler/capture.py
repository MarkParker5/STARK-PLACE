"""Interpreter-level capture for the STARK profiler ("swizzle all of STARK").

Uses `sys.monitoring` (PEP 669): the CPython interpreter itself fires a callback on every function
call/return, so there is no method wrapping, no registry and no edit to any stark source file. The
single decision about what is profiled is a path filter — is the code object's file inside the stark
package directory. Everything else (deps, libs, stdlib) is excluded for free.

`sys.monitoring` is chosen over `sys.setprofile`/`cProfile` because it is interpreter-*global* (it
also fires on the worker threads `asyncer.asyncify` spins up), it has zero cost when no tool is
registered, and its per-event callbacks hand us args and return values directly.

Each boundary becomes a `ProfileEvent` handed to a `emit` callback (the bus). Per-thread state (kept
in a `threading.local`) tracks call nesting and open frames so that:
  * `depth` is correct per thread;
  * `dur_ns` matches a RETURN to its CALL by `id(frame)` — correct even when coroutines interleave
    on one thread (a single LIFO stack would mismatch there).

Guards: a re-entrancy flag makes the callbacks ignore stark frames triggered by our own payload
serialization; the serializer never consumes generators. Frames left open by an exception unwind are
reconciled via PY_UNWIND so the open-frames map does not leak.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from typing import Callable

import stark

from . import extractors
from .schema import CALL, ERROR, RETURN, ProfileEvent

_MON = sys.monitoring
_TOOL_ID = _MON.PROFILER_ID

# The stark package root. The filter `root` may be narrower (e.g. stark/core), but module paths in
# events are always reported relative to the package root so they stay stable.
STARK_DIR = os.path.dirname(os.path.abspath(stark.__file__))


class Capture:
    def __init__(self, emit: Callable[[ProfileEvent], None], root: str, tool_name: str = "stark-profiler") -> None:
        self._emit = emit
        self._root = root
        self._tool_name = tool_name
        self._seq = itertools.count(1)  # next() is atomic in CPython → total order across threads
        self._local = threading.local()
        self._trace_lock = threading.Lock()
        self._trace_counter = 0
        self._trace_id = "t0"
        self._enabled = False

    # per-thread state -----------------------------------------------------------------------------

    def _frames(self) -> dict[int, int]:
        frames = getattr(self._local, "frames", None)
        if frames is None:
            frames = self._local.frames = {}
        return frames

    # filter / helpers -----------------------------------------------------------------------------

    def _want(self, code) -> bool:
        return code.co_filename.startswith(self._root)

    def _rel(self, code) -> str:
        return os.path.relpath(code.co_filename, STARK_DIR)

    def _new_trace_if_top(self, frames: dict) -> None:
        # A top-level stark entry (no open stark frames on this thread) on the main thread starts a
        # new trace = one utterance. Worker-thread entries inherit the current trace id.
        if not frames and threading.current_thread() is threading.main_thread():
            with self._trace_lock:
                self._trace_counter += 1
                self._trace_id = f"t{self._trace_counter}"

    # callbacks ------------------------------------------------------------------------------------

    def _on_start(self, code, instruction_offset):
        if not self._want(code) or getattr(self._local, "busy", False):
            return
        self._local.busy = True
        try:
            frames = self._frames()
            self._new_trace_if_top(frames)
            qual = code.co_qualname
            frame = sys._getframe(1)
            ex = extractors.EXTRACTORS.get(qual)
            if ex and "in" in ex:
                try:
                    # a few extractors need the live frame (to read a caller's locals, e.g. which
                    # command is emitting a response); the rest just get frame.f_locals.
                    payload = ex["in"](frame.f_locals, frame) if ex.get("wants_frame") else ex["in"](frame.f_locals)
                except Exception as exc:
                    payload = {"_err": str(exc)}
            else:
                payload = extractors.generic_input(frame)

            now = time.perf_counter_ns()
            frames[id(frame)] = now
            self._emit(ProfileEvent(
                trace_id=self._trace_id,
                seq=next(self._seq),
                t_ns=now,
                phase=CALL,
                symbol=qual,
                module=self._rel(code),
                depth=len(frames),
                thread=threading.get_ident(),
                dur_ns=None,
                data=payload,
                call_id=id(frame),
            ))
        finally:
            self._local.busy = False

    def _on_return(self, code, instruction_offset, retval):
        if not self._want(code) or getattr(self._local, "busy", False):
            return
        self._local.busy = True
        try:
            frames = self._frames()
            qual = code.co_qualname
            frame = sys._getframe(1)
            ex = extractors.EXTRACTORS.get(qual)
            if ex and "out" in ex:
                try:
                    # out-extractors receive (retval, frame_locals) so they can read state a call
                    # mutated in place (e.g. string.corrections, recognized_entities).
                    payload = ex["out"](retval, frame.f_locals)
                except Exception as exc:
                    payload = {"_err": str(exc)}
            else:
                payload = {"return": extractors.serialize(retval)}

            now = time.perf_counter_ns()
            depth = len(frames)
            start = frames.pop(id(frame), None)
            self._emit(ProfileEvent(
                trace_id=self._trace_id,
                seq=next(self._seq),
                t_ns=now,
                phase=RETURN,
                symbol=qual,
                module=self._rel(code),
                depth=depth,
                thread=threading.get_ident(),
                dur_ns=(now - start) if start is not None else None,
                data=payload,
                call_id=id(frame),
            ))
        finally:
            self._local.busy = False

    def _on_unwind(self, code, instruction_offset, exception):
        # An exception is leaving this frame — emit an error boundary and reconcile open frames.
        if not self._want(code) or getattr(self._local, "busy", False):
            return
        # GeneratorExit / StopIteration are normal control flow (a consumed/closed generator),
        # not errors — reconcile the open frame silently instead of emitting a noisy ERROR event.
        normal = isinstance(exception, (GeneratorExit, StopIteration, StopAsyncIteration))
        self._local.busy = True
        try:
            frames = self._frames()
            frame = sys._getframe(1)
            if normal:
                frames.pop(id(frame), None)
                return
            now = time.perf_counter_ns()
            depth = len(frames)
            start = frames.pop(id(frame), None)
            self._emit(ProfileEvent(
                trace_id=self._trace_id,
                seq=next(self._seq),
                t_ns=now,
                phase=ERROR,
                symbol=code.co_qualname,
                module=self._rel(code),
                depth=depth,
                thread=threading.get_ident(),
                dur_ns=(now - start) if start is not None else None,
                data={"exception": f"{type(exception).__name__}: {exception}"},
                call_id=id(frame),
            ))
        finally:
            self._local.busy = False

    # lifecycle ------------------------------------------------------------------------------------

    def enable(self) -> None:
        if self._enabled:
            raise RuntimeError("Capture already enabled")
        if _MON.get_tool(_TOOL_ID) is not None:
            raise RuntimeError(
                f"sys.monitoring PROFILER_ID is already in use by {_MON.get_tool(_TOOL_ID)!r}"
            )
        _MON.use_tool_id(_TOOL_ID, self._tool_name)
        _MON.register_callback(_TOOL_ID, _MON.events.PY_START, self._on_start)
        _MON.register_callback(_TOOL_ID, _MON.events.PY_RETURN, self._on_return)
        _MON.register_callback(_TOOL_ID, _MON.events.PY_UNWIND, self._on_unwind)
        _MON.set_events(_TOOL_ID, _MON.events.PY_START | _MON.events.PY_RETURN | _MON.events.PY_UNWIND)
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        _MON.set_events(_TOOL_ID, 0)
        for event in (_MON.events.PY_START, _MON.events.PY_RETURN, _MON.events.PY_UNWIND):
            _MON.register_callback(_TOOL_ID, event, None)
        _MON.free_tool_id(_TOOL_ID)
        self._enabled = False

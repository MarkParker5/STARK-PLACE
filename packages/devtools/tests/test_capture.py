import sys
import threading

from stark_devtools.profiler import MemorySink, ProfilerBus
from stark_devtools.profiler.capture import STARK_DIR, Capture
from stark_devtools.profiler.schema import CALL, RETURN

_TOOL = sys.monitoring.PROFILER_ID


def test_disabled_by_default():
    # zero-overhead invariant: with no active session, the interpreter has no profiler tool.
    assert sys.monitoring.get_tool(_TOOL) is None


def test_enable_disable_restores_interpreter():
    cap = Capture(lambda e: None, root=STARK_DIR)
    assert sys.monitoring.get_tool(_TOOL) is None
    cap.enable()
    try:
        assert sys.monitoring.get_tool(_TOOL) == "stark-profiler"
    finally:
        cap.disable()
    assert sys.monitoring.get_tool(_TOOL) is None


def test_double_enable_raises():
    cap = Capture(lambda e: None, root=STARK_DIR)
    cap.enable()
    try:
        cap2 = Capture(lambda e: None, root=STARK_DIR)
        raised = False
        try:
            cap2.enable()
        except RuntimeError:
            raised = True
        assert raised
    finally:
        cap.disable()


async def test_captures_engine_calls(run_utterance):
    events = await run_utterance("play Metallica")
    symbols = {e.symbol for e in events}
    assert "CommandsContext.process_string" in symbols
    assert "PatternParser.match" in symbols
    assert "SearchProcessor.search" in symbols
    # every event is stark code and well-formed
    assert all(e.module.endswith(".py") for e in events)
    assert all(e.phase in (CALL, RETURN, "error") for e in events)


async def test_seq_is_total_order(run_utterance):
    events = await run_utterance("play Metallica")
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)  # unique


async def test_durations_present_on_returns(run_utterance):
    events = await run_utterance("play Metallica")
    returns = [e for e in events if e.phase == RETURN]
    assert returns
    assert any(e.dur_ns is not None and e.dur_ns >= 0 for e in returns)


async def test_worker_thread_capture(run_utterance, make_manager):
    # a sync command runs via asyncify on a worker thread; sys.monitoring must still see it.
    events = await run_utterance("play Metallica", manager=make_manager(sync_play=True))
    main_id = threading.main_thread().ident
    worker_events = [e for e in events if e.thread != main_id]
    assert worker_events, "expected at least one event on an asyncify worker thread"
    # the Response built inside the sync command is one of them
    assert any(e.symbol == "Response.__init__" for e in worker_events)


def test_bus_no_events_without_capture():
    # the bus alone never fabricates events
    sink = MemorySink()
    bus = ProfilerBus()
    bus.start([sink])
    bus.stop()
    assert sink.events == []

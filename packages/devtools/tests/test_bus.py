import os
import threading

from stark_place.devtools.profiler import JSONLSink, MemorySink, ProfileEvent, ProfilerBus
from stark_place.devtools.profiler.schema import CALL


def _event(seq: int, thread: int) -> ProfileEvent:
    return ProfileEvent(
        trace_id="t1", seq=seq, t_ns=seq, phase=CALL, symbol="x", module="m",
        depth=1, thread=thread, dur_ns=None, data={},
    )


def test_bus_drains_all_from_many_threads():
    sink = MemorySink()
    bus = ProfilerBus()
    bus.start([sink])

    per_thread = 500
    n_threads = 8

    def producer(base):
        for i in range(per_thread):
            bus.emit(_event(base + i, threading.get_ident()))

    threads = [threading.Thread(target=producer, args=(t * per_thread,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    bus.stop()

    # nothing dropped
    assert len(sink.events) == per_thread * n_threads
    # every produced seq is present exactly once
    assert {e.seq for e in sink.events} == set(range(per_thread * n_threads))


def test_jsonl_sink_writes_and_closes(tmp_path):
    path = os.path.join(tmp_path, "trace.jsonl")
    sink = JSONLSink(path)
    bus = ProfilerBus()
    bus.start([sink])
    for i in range(10):
        bus.emit(_event(i, 1))
    bus.stop()

    with open(path) as f:
        lines = [ProfileEvent.from_dict(__import__("json").loads(l)) for l in f]
    assert [e.seq for e in lines] == list(range(10))


def test_stop_flushes_pending():
    sink = MemorySink()
    bus = ProfilerBus()
    bus.start([sink])
    for i in range(1000):
        bus.emit(_event(i, 1))
    bus.stop()  # must flush everything still queued before closing
    assert len(sink.events) == 1000

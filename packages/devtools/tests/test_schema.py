import json

from stark_devtools.profiler import ProfileEvent, SCHEMA_VERSION
from stark_devtools.profiler.schema import CALL


def test_schema_version():
    assert SCHEMA_VERSION == 1


def test_event_roundtrip():
    e = ProfileEvent(
        trace_id="t1", seq=3, t_ns=123, phase=CALL, symbol="PatternParser.match",
        module="core/parsing.py", depth=2, thread=42, dur_ns=None,
        data={"pattern": "play $band:Word", "string": "play Metallica"},
        call_id=140234,
    )
    d = e.to_dict()
    assert set(d) == {"trace_id", "seq", "t_ns", "phase", "symbol", "module", "depth", "thread", "dur_ns", "data", "call_id"}
    # survives a JSON round-trip unchanged
    back = ProfileEvent.from_dict(json.loads(json.dumps(d)))
    assert back == e

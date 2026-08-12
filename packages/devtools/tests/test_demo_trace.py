"""Golden end-to-end trace: the real engine, profiled, asserted at the semantic level."""

from stark_place.devtools.profiler.schema import CALL, RETURN


def _find(events, symbol, phase):
    return [e for e in events if e.symbol == symbol and e.phase == phase]


async def test_demo_trace_semantics(run_utterance):
    events = await run_utterance("play Metallica and turn off the kitchen lights")

    # input payload of the top-level call
    ps_call = _find(events, "CommandsContext.process_string", CALL)
    assert ps_call and ps_call[0].data["string"] == "play Metallica and turn off the kitchen lights"

    # output payload: two SearchResults for the two commands
    ps_ret = _find(events, "CommandsContext.process_string", RETURN)
    assert ps_ret
    results = ps_ret[0].data["results"]
    commands = {r["command"] for r in results}
    assert commands == {"CommandsManager.play_music", "CommandsManager.lights_off"}

    # the band parameter was parsed out of the utterance
    play = next(r for r in results if r["command"] == "CommandsManager.play_music")
    assert play["match"]["parameters"]["band"] == "Metallica" or "Metallica" in str(play["match"]["parameters"]["band"])

    # both commands were dispatched with their params
    run_calls = _find(events, "CommandsContext.run_command", CALL)
    dispatched = {e.data["command"] for e in run_calls}
    assert dispatched == {"CommandsManager.play_music", "CommandsManager.lights_off"}

    # PatternParser.match produced a MatchResult with span + corrected string
    matches = _find(events, "PatternParser.match", RETURN)
    assert any(
        isinstance(m.data.get("matches"), list) and m.data["matches"]
        and "substring" in m.data["matches"][0]
        for m in matches
    )


async def test_no_generator_was_consumed(run_utterance):
    # Dictionary.search_in_sentence isn't used here, but the invariant is that any generator return
    # is reported as not-consumed rather than materialized.
    events = await run_utterance("play Metallica")
    for e in events:
        ret = e.data.get("return")
        if isinstance(ret, str):
            assert "not consumed" in ret or "generator" not in ret.lower() or "not consumed" in ret

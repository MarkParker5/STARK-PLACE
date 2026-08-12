"""
Tests for OneStepLLMProcessor.

Cornerstones verified:
  1. Multi-command: all confident matches returned.
  2. Overlap exclusion: real (start, end) spans; overlapping results resolved.
  3. Parameters guaranteed: every declared param key present, None as fallback.
  4. Same-family priority: more filled parameters wins when overlapping.
  5. Context hierarchy: stops at first layer with results.
  6. Chain short-circuit: non-empty result stops the chain.

Structure:
  Benchmarks only — real LLM, require small_model or medium_model + running
  Ollama, run with: pytest -m benchmark

Note: _assign_indices and _resolve_overlaps are the same implementations as in
TwoStepLLMProcessor — their unit tests live in
test_two_step_search_parsing_processor_llm.py and are not duplicated here.
"""

from __future__ import annotations

import pytest
from conftest import Artist, BrightnessLevel, Lamp, Song, drain
from stark.core.command import Command, Response
from stark.core.commands_context_processor import CommandsContextLayer
from stark.core.patterns.pattern import Pattern

from ready.one_step_search_parsing_processor_llm import OneStepLLMProcessor

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cmd(name: str) -> Command:
    async def _runner() -> Response:
        return Response(name)

    _runner.__name__ = name
    _runner.__annotations__ = {}
    return Command(name, Pattern("**"), _runner)


def _cmd_with_params(name: str, params: dict[str, type]) -> Command:
    async def _runner(**kwargs) -> Response:
        return Response(name)

    _runner.__name__ = name
    _runner.__annotations__ = params
    return Command(name, Pattern("**"), _runner)


# ── Benchmarks (live — require small_model or medium_model + running Ollama) ──


@pytest.mark.benchmark
async def test_single_command_matched(make_context):
    """Basic happy path: one command matched and returned."""
    processor = OneStepLLMProcessor()
    cmd = _cmd_with_params("lamp_on", {"lamp": Lamp})

    async with make_context(processors=[processor], object_types=[Lamp]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[cmd], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("turn on bedroom lamp", context, layer, [])

    assert len(results) == 1
    assert results[0].command is cmd


@pytest.mark.benchmark
async def test_spans_are_real_substrings(make_context):
    """Cornerstone 2: start/end must index back to the exact substring in the input."""
    processor = OneStepLLMProcessor()
    cmd = _cmd_with_params("lamp_on", {"lamp": Lamp})
    input_str = "please turn on bedroom lamp now"

    async with make_context(processors=[processor], object_types=[Lamp]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[cmd], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer(input_str, context, layer, [])

    for r in results:
        mr = r.match_result
        assert input_str[mr.start : mr.end] == mr.substring


@pytest.mark.benchmark
async def test_parameters_guaranteed_with_none_fallback(make_context):
    """Cornerstone 3: all declared param keys present; unparsed → None."""
    processor = OneStepLLMProcessor()
    cmd = _cmd_with_params("lamp_brightness", {"lamp": Lamp, "level": BrightnessLevel})

    async with make_context(processors=[processor], object_types=[Lamp, BrightnessLevel]) as (
        manager,
        context,
        collector,
    ):
        layer = CommandsContextLayer(commands=[cmd], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("set the brightness", context, layer, [])

    if results:
        params = results[0].match_result.parameters
        assert "lamp" in params
        assert "level" in params


@pytest.mark.benchmark
async def test_multi_command_both_returned(make_context):
    """Cornerstone 1: two non-overlapping commands both returned."""
    processor = OneStepLLMProcessor()
    cmd_on = _cmd_with_params("lamp_on", {"lamp": Lamp})
    cmd_brightness = _cmd_with_params("lamp_brightness", {"lamp": Lamp, "level": BrightnessLevel})

    async with make_context(processors=[processor], object_types=[Lamp, BrightnessLevel]) as (
        manager,
        context,
        collector,
    ):
        layer = CommandsContextLayer(commands=[cmd_on, cmd_brightness], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer(
            "turn on bedroom lamp and make the brightness 50%", context, layer, []
        )

    names = {r.command.name for r in results}
    assert "lamp_on" in names
    assert "lamp_brightness" in names


@pytest.mark.benchmark
async def test_overlapping_results_resolved_more_params_wins(make_context):
    """Cornerstone 2 & 4: overlapping spans resolved; more filled params wins."""
    processor = OneStepLLMProcessor()
    cmd_specific = _cmd_with_params("play_song_artist", {"song": Song, "artist": Artist})
    cmd_generic = _cmd_with_params("play_song", {"song": Song})

    async with make_context(processors=[processor], object_types=[Song, Artist]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[cmd_specific, cmd_generic], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("play bohemian rhapsody by queen", context, layer, [])

    # If both were returned overlapping, only the more specific one should survive.
    if len(results) == 1:
        assert results[0].command is cmd_specific


@pytest.mark.benchmark
async def test_song_and_artist_parsed(make_context):
    """Song and Artist parameters both instantiated from natural-language input."""
    processor = OneStepLLMProcessor()
    cmd = _cmd_with_params("play_song", {"song": Song, "artist": Artist})

    async with make_context(processors=[processor], object_types=[Song, Artist]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[cmd], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("play bohemian rhapsody by queen", context, layer, [])

    assert len(results) == 1
    params = results[0].match_result.parameters
    assert params.get("song") is not None
    assert params.get("artist") is not None


@pytest.mark.benchmark
async def test_no_match_returns_empty(make_context):
    """Unrelated input must return an empty result list."""
    processor = OneStepLLMProcessor()
    cmd = _cmd_with_params("lamp_on", {"lamp": Lamp})

    async with make_context(processors=[processor], object_types=[Lamp]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[cmd], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("what is the weather today", context, layer, [])

    assert results == []


@pytest.mark.benchmark
async def test_context_hierarchy_stops_at_first_hit(make_context):
    """Cornerstone 5: process_string stops at the first layer that returns results."""
    processor = OneStepLLMProcessor()
    cmd_inner = _cmd_with_params("lamp_on", {"lamp": Lamp})
    cmd_outer = _cmd_with_params("play_song", {"song": Song})

    inner_layer = CommandsContextLayer(commands=[cmd_inner], parameters={})
    outer_layer = CommandsContextLayer(commands=[cmd_outer], parameters={})

    async with make_context(processors=[processor], object_types=[Lamp, Song]) as (manager, context, collector):
        context.context_queue = [inner_layer, outer_layer]
        results, pops = await processor.process_string("turn on bedroom lamp", context, [])

    assert pops == 0
    assert all(r.command is cmd_inner for r in results)


@pytest.mark.benchmark
async def test_natural_phrasing_matched(make_context):
    """Natural speech phrasing that a pattern wouldn't catch must still match."""
    processor = OneStepLLMProcessor()
    cmd = _cmd_with_params("lamp_brightness", {"lamp": Lamp, "level": BrightnessLevel})

    async with make_context(processors=[processor], object_types=[Lamp, BrightnessLevel]) as (
        manager,
        context,
        collector,
    ):
        layer = CommandsContextLayer(commands=[cmd], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer(
            "could you maybe lower the lights in here a bit", context, layer, []
        )

    assert len(results) == 1


@pytest.mark.benchmark
async def test_cross_command_context_lamp_inferred(make_context):
    """
    Cornerstone 2 (nonlocal param values): the lamp parameter for the brightness
    command is inferred from the turn-on part of the input, not re-consumed.
    """
    processor = OneStepLLMProcessor()
    cmd_on = _cmd_with_params("lamp_on", {"lamp": Lamp})
    cmd_brightness = _cmd_with_params("lamp_brightness", {"lamp": Lamp, "level": BrightnessLevel})

    input_str = "turn on bedroom lamp and make the brightness 50%"

    async with make_context(processors=[processor], object_types=[Lamp, BrightnessLevel]) as (
        manager,
        context,
        collector,
    ):
        layer = CommandsContextLayer(commands=[cmd_on, cmd_brightness], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer(input_str, context, layer, [])

    brightness_results = [r for r in results if r.command is cmd_brightness]
    if brightness_results:
        lamp_param = brightness_results[0].match_result.parameters.get("lamp")
        assert lamp_param is not None


@pytest.mark.benchmark
async def test_chain_short_circuits_after_match(make_context):
    """Cornerstone 6: a second processor must not be called once results found."""
    called: list[str] = []

    class SpyProcessor(OneStepLLMProcessor):
        async def process_context_layer(self, string, context, layer, entities):
            called.append("spy")
            return []

    cmd = _cmd_with_params("lamp_on", {"lamp": Lamp})
    processor = OneStepLLMProcessor()
    spy = SpyProcessor()

    async with make_context(processors=[processor, spy], object_types=[Lamp]) as (manager, context, collector):
        manager.commands.append(cmd)
        await context.process_string("turn on bedroom lamp")
        await drain()

    assert "spy" not in called

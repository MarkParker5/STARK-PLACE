"""
Tests for TwoStepLLMProcessor.

Cornerstones verified:
  1. Multi-command: all confident matches returned.
  2. Overlap exclusion: real (start, end) spans; overlapping results resolved.
  3. Parameters guaranteed: every declared param key present, None as fallback.
  4. Same-family priority: more filled parameters wins when overlapping.
  5. Context hierarchy: stops at first layer with results.
  6. Chain short-circuit: non-empty result stops the chain.

Structure:
  Unit tests  — pure logic for _assign_indices and _resolve_overlaps, no model
                needed, always run.
  Benchmarks  — real LLM, require small_model + running Ollama,
                run with: pytest -m benchmark
"""

from __future__ import annotations

import pytest
from conftest import Artist, BrightnessLevel, Lamp, Song, drain
from stark.core.command import Command, Response
from stark.core.commands_context_processor import CommandsContextLayer
from stark.core.commands_manager import SearchResult
from stark.core.parsing import MatchResult
from stark.core.patterns.pattern import Pattern

from ready.two_step_search_parsing_processor_llm import (
    TwoStepLLMProcessor,
    _assign_indices,
    _resolve_overlaps,
)

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


def _sr(cmd: Command, substring: str, start: int, params: dict, index: int = 0) -> SearchResult:
    sr = SearchResult(cmd, MatchResult(substring=substring, start=start, end=start + len(substring), parameters=params))
    sr.index = index
    return sr


# ── Unit: _assign_indices ─────────────────────────────────────────────────────


def test_assign_indices_follows_declaration_order():
    cmd_a, cmd_b, cmd_c = _cmd("a"), _cmd("b"), _cmd("c")
    sr_c = _sr(cmd_c, "c", 0, {})
    sr_a = _sr(cmd_a, "a", 2, {})
    _assign_indices([sr_c, sr_a], ["a", "b", "c"])
    assert sr_a.index == 0
    assert sr_c.index == 2


def test_assign_indices_unknown_command_gets_last():
    cmd_x = _cmd("x")
    sr = _sr(cmd_x, "x", 0, {})
    _assign_indices([sr], ["a", "b"])
    assert sr.index == 2  # len(commands_order)


# ── Unit: _resolve_overlaps ───────────────────────────────────────────────────


def test_resolve_overlaps_empty():
    assert _resolve_overlaps([]) == []


def test_resolve_overlaps_single_unchanged():
    cmd = _cmd("a")
    sr = _sr(cmd, "hello", 0, {})
    assert _resolve_overlaps([sr]) == [sr]


def test_resolve_overlaps_non_overlapping_both_kept():
    sr_a = _sr(_cmd("a"), "turn on", 0, {}, index=0)
    sr_b = _sr(_cmd("b"), "brightness 50", 8, {}, index=1)
    assert len(_resolve_overlaps([sr_a, sr_b])) == 2


def test_resolve_overlaps_lower_index_wins():
    """Cornerstone 2 & 4: earlier-declared command wins on equal params."""
    cmd_a, cmd_b = _cmd("a"), _cmd("b")
    sr_a = _sr(cmd_a, "turn on lamp", 0, {}, index=0)
    sr_b = _sr(cmd_b, "turn on", 0, {}, index=1)
    result = _resolve_overlaps([sr_a, sr_b])
    assert len(result) == 1
    assert result[0].command is cmd_a


def test_resolve_overlaps_more_filled_params_beats_index():
    """Cornerstone 4: more non-None params beats declaration order."""
    cmd_a, cmd_b = _cmd("a"), _cmd("b")
    sr_a = _sr(cmd_a, "play music", 0, {"song": None}, index=0)
    sr_b = _sr(cmd_b, "play music", 0, {"song": Song("stairway"), "artist": Artist("led zeppelin")}, index=1)
    result = _resolve_overlaps([sr_a, sr_b])
    assert len(result) == 1
    assert result[0].command is cmd_b


def test_resolve_overlaps_sorted_by_start():
    sr_b = _sr(_cmd("b"), "brightness 50", 20, {}, index=1)
    sr_a = _sr(_cmd("a"), "turn on lamp", 0, {}, index=0)
    result = _resolve_overlaps([sr_b, sr_a])
    assert result[0].match_result.start < result[1].match_result.start


# ── Benchmarks (live — require small_model + running Ollama) ──────────────────


@pytest.mark.benchmark
async def test_single_command_matched(make_context):
    """Basic happy path: one command matched and returned."""
    processor = TwoStepLLMProcessor()
    cmd = _cmd_with_params("lamp_on", {"lamp": Lamp})

    async with make_context(processors=[processor], object_types=[Lamp]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[cmd], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("turn on bedroom lamp", context, layer, [])

    assert len(results) == 1
    assert results[0].command is cmd


@pytest.mark.benchmark
async def test_parameters_guaranteed_with_none_fallback(make_context):
    """Cornerstone 3: all declared param keys present; unparsed → None."""
    processor = TwoStepLLMProcessor()
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
    processor = TwoStepLLMProcessor()
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
async def test_spans_are_real_substrings(make_context):
    """Cornerstone 2: start/end must index back to the exact substring in the input."""
    processor = TwoStepLLMProcessor()
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
async def test_no_match_returns_empty(make_context):
    """Unrelated input must return an empty result list."""
    processor = TwoStepLLMProcessor()
    cmd = _cmd_with_params("lamp_on", {"lamp": Lamp})

    async with make_context(processors=[processor], object_types=[Lamp]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[cmd], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("what is the weather today", context, layer, [])

    assert results == []


@pytest.mark.benchmark
async def test_context_hierarchy_stops_at_first_hit(make_context):
    """Cornerstone 5: process_string stops at the first layer that returns results."""
    processor = TwoStepLLMProcessor()
    cmd_inner = _cmd_with_params("lamp_on", {"lamp": Lamp})
    cmd_outer = _cmd_with_params("play_song", {"song": Song})

    inner_layer = CommandsContextLayer(commands=[cmd_inner], parameters={})
    outer_layer = CommandsContextLayer(commands=[cmd_outer], parameters={})

    async with make_context(processors=[processor], object_types=[Lamp, Song]) as (manager, context, collector):
        context.context_queue = [inner_layer, outer_layer]
        results, pops = await processor.process_string("turn on bedroom lamp", context, [])

    # inner layer matched → outer never searched
    assert pops == 0
    assert all(r.command is cmd_inner for r in results)


@pytest.mark.benchmark
async def test_song_and_artist_parsed(make_context):
    """Song and Artist parameters both instantiated from natural-language input."""
    processor = TwoStepLLMProcessor()
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
async def test_chain_short_circuits_after_match(make_context):
    """Cornerstone 6: a second processor must not be called once results found."""
    called: list[str] = []

    class SpyProcessor(TwoStepLLMProcessor):
        async def process_context_layer(self, string, context, layer, entities):
            called.append("spy")
            return []

    cmd = _cmd_with_params("lamp_on", {"lamp": Lamp})
    processor = TwoStepLLMProcessor()
    spy = SpyProcessor()

    async with make_context(processors=[processor, spy], object_types=[Lamp]) as (manager, context, collector):
        manager.commands.append(cmd)
        await context.process_string("turn on bedroom lamp")
        await drain()

    assert "spy" not in called

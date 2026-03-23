"""
Tests for LLMNERProcessor.

Cornerstones verified:
  5. NER processors return ([], 0) — never short-circuit the chain.
  7. Populated recognized_entities feed downstream processors.

Structure:
  Unit tests  — guard logic only (no-types early exit, unknown type name,
                substring not in input), no model needed, always run.
  Benchmarks  — real LLM, require small_model + running Ollama,
                run with: pytest -m benchmark
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from conftest import Artist, BrightnessLevel, Lamp, Song
from stark.core.commands_context_processor import RecognizedEntity

from ready.ner_processor_llm import LLMNERProcessor, RecognizedEntityMatch

# ── Unit: guard / early-exit paths (no LLM call) ─────────────────────────────


async def test_no_types_returns_immediately(make_context):
    """Cornerstone 5: with no types registered, must return ([], 0) without calling the LLM."""
    processor = LLMNERProcessor(types=[])

    with patch("ready.ner_processor_llm._ner_agent") as mock_agent:
        mock_agent.run = AsyncMock(side_effect=AssertionError("must not be called"))
        async with make_context(processors=[processor]) as (manager, context, collector):
            results, pops = await processor.process_string("turn on the lamp", context, [])

    assert results == []
    assert pops == 0


async def test_unknown_type_name_skipped(make_context):
    """LLM returning a type name not in the registry must be silently dropped."""
    from unittest.mock import MagicMock

    processor = LLMNERProcessor(types=[Lamp])

    mock_result = MagicMock()
    mock_result.output = [
        RecognizedEntityMatch(type_name="HallucinatedType", substring="bedroom lamp", confidence=0.9),
    ]

    with patch("ready.ner_processor_llm._ner_agent") as mock_agent:
        mock_agent.run = AsyncMock(return_value=mock_result)
        async with make_context(processors=[processor]) as (manager, context, collector):
            recognized: list[RecognizedEntity] = []
            results, pops = await processor.process_string("turn on bedroom lamp", context, recognized)

    assert recognized == []
    assert results == []
    assert pops == 0


async def test_substring_not_in_input_skipped(make_context):
    """LLM returning a substring absent from the input must be silently dropped."""
    from unittest.mock import MagicMock

    processor = LLMNERProcessor(types=[Lamp])

    mock_result = MagicMock()
    mock_result.output = [
        RecognizedEntityMatch(type_name="Lamp", substring="ceiling fan", confidence=0.88),
    ]

    with patch("ready.ner_processor_llm._ner_agent") as mock_agent:
        mock_agent.run = AsyncMock(return_value=mock_result)
        async with make_context(processors=[processor]) as (manager, context, collector):
            recognized: list[RecognizedEntity] = []
            await processor.process_string("turn on bedroom lamp", context, recognized)

    assert recognized == []


async def test_llm_exception_swallowed(make_context):
    """A failing LLM call must not propagate — processor returns ([], 0) gracefully."""
    processor = LLMNERProcessor(types=[Lamp])

    with patch("ready.ner_processor_llm._ner_agent") as mock_agent:
        mock_agent.run = AsyncMock(side_effect=RuntimeError("connection refused"))
        async with make_context(processors=[processor]) as (manager, context, collector):
            results, pops = await processor.process_string("turn on the lamp", context, [])

    assert results == []
    assert pops == 0


# ── Benchmarks (live — require small_model + running Ollama) ──────────────────


@pytest.mark.benchmark
async def test_always_returns_empty_results_and_zero_pops(make_context):
    """Cornerstone 5: live call must still return ([], 0)."""
    processor = LLMNERProcessor(types=[Lamp])
    async with make_context(processors=[processor]) as (manager, context, collector):
        results, pops = await processor.process_string("turn on bedroom lamp", context, [])

    assert results == []
    assert pops == 0


@pytest.mark.benchmark
async def test_lamp_entity_matched(make_context):
    """Lamp name extracted and appended to recognized_entities."""
    processor = LLMNERProcessor(types=[Lamp])
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string("turn on the bedroom lamp", context, recognized)

    assert any(e.type is Lamp for e in recognized)


@pytest.mark.benchmark
async def test_song_and_artist_matched_separately(make_context):
    """Both Song and Artist extracted as independent entities."""
    processor = LLMNERProcessor(types=[Song, Artist])
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string("play bohemian rhapsody by queen", context, recognized)

    types_found = {e.type for e in recognized}
    assert Song in types_found
    assert Artist in types_found


@pytest.mark.benchmark
async def test_lamp_and_brightness_matched(make_context):
    """Lamp name and brightness level both extracted from the same input."""
    processor = LLMNERProcessor(types=[Lamp, BrightnessLevel])
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string("set bedroom lamp to 50%", context, recognized)

    types_found = {e.type for e in recognized}
    assert Lamp in types_found
    assert BrightnessLevel in types_found


@pytest.mark.benchmark
async def test_no_entities_on_unrelated_input(make_context):
    """Input with no named entities should produce an empty recognized_entities list."""
    processor = LLMNERProcessor(types=[Lamp, Song])
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string("what time is it", context, recognized)

    assert recognized == []


@pytest.mark.benchmark
async def test_matched_substring_is_in_input(make_context):
    """Cornerstone 2 spirit: every matched substring must be a real substring of the input."""
    processor = LLMNERProcessor(types=[Song, Artist])
    input_str = "play bohemian rhapsody by queen"
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string(input_str, context, recognized)

    for entity in recognized:
        assert entity.substring in input_str

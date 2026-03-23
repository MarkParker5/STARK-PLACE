"""
Tests for EmbeddingsNERProcessor.

Cornerstones verified:
  5. NER processors return ([], 0) — never short-circuit the chain.
  7. Populated recognized_entities feed downstream processors.

Structure:
  Unit tests  — pure logic, no model, always run.
  Benchmarks  — real embedder, require tiny_model + running Ollama,
                run with: pytest -m benchmark
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from conftest import Artist, BrightnessLevel, Lamp, Song
from stark.core.commands_context_processor import RecognizedEntity

from ready.ner_processor_embeddings import EmbeddingsNERProcessor, _sliding_window_candidates

# ── Unit: _sliding_window_candidates ─────────────────────────────────────────


def test_sliding_window_single_word():
    assert _sliding_window_candidates("hello", max_words=3) == ["hello"]


def test_sliding_window_order():
    # left-to-right, size-ascending
    assert _sliding_window_candidates("a b c", max_words=3) == ["a", "b", "c", "a b", "b c", "a b c"]


def test_sliding_window_no_duplicates():
    result = _sliding_window_candidates("a b c", max_words=4)
    assert len(result) == len(set(result))


def test_sliding_window_respects_max_words():
    result = _sliding_window_candidates("one two three four", max_words=2)
    assert all(len(c.split()) <= 2 for c in result)


def test_sliding_window_empty_string():
    assert _sliding_window_candidates("", max_words=4) == []


# ── Unit: guard / early-exit paths (no embedder call) ────────────────────────


async def test_no_types_returns_immediately(make_context):
    """Cornerstone 5: with no types registered, must return ([], 0) without calling the embedder."""
    processor = EmbeddingsNERProcessor(types=[])

    with patch("ready.ner_processor_embeddings._embedder") as mock_embedder:
        mock_embedder.embed_documents = AsyncMock(side_effect=AssertionError("must not be called"))
        async with make_context(processors=[processor]) as (manager, context, collector):
            results, pops = await processor.process_string("turn on the lamp", context, [])

    assert results == []
    assert pops == 0


async def test_empty_string_returns_immediately(make_context):
    """Empty input produces no candidates; embedder must not be called."""
    processor = EmbeddingsNERProcessor(types=[Lamp])

    with patch("ready.ner_processor_embeddings._embedder") as mock_embedder:
        mock_embedder.embed_documents = AsyncMock(side_effect=AssertionError("must not be called"))
        async with make_context(processors=[processor]) as (manager, context, collector):
            results, pops = await processor.process_string("", context, [])

    assert results == []
    assert pops == 0


async def test_embedder_exception_swallowed(make_context):
    """A failing embedder must not propagate — processor returns ([], 0) gracefully."""
    processor = EmbeddingsNERProcessor(types=[Lamp])

    with patch("ready.ner_processor_embeddings._embedder") as mock_embedder:
        mock_embedder.embed_documents = AsyncMock(side_effect=RuntimeError("network error"))
        async with make_context(processors=[processor]) as (manager, context, collector):
            results, pops = await processor.process_string("turn on the lamp", context, [])

    assert results == []
    assert pops == 0


# ── Benchmarks (live — require tiny_model + running Ollama) ──────────────────


@pytest.mark.benchmark
async def test_song_entity_matched(make_context):
    """Song name extracted; 'play' is not included in the matched substring."""
    processor = EmbeddingsNERProcessor(types=[Song])
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        results, pops = await processor.process_string("play bohemian rhapsody", context, recognized)

    assert results == []
    assert pops == 0
    assert any(e.type is Song and "bohemian rhapsody" in e.substring for e in recognized), recognized


@pytest.mark.benchmark
async def test_song_and_artist_matched_separately(make_context):
    """Both Song and Artist extracted as independent entities from the same input."""
    processor = EmbeddingsNERProcessor(types=[Song, Artist])
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string("stairway to heaven by led zeppelin", context, recognized)

    types_found = {e.type for e in recognized}
    assert Song in types_found
    assert Artist in types_found


@pytest.mark.benchmark
async def test_lamp_and_brightness_matched(make_context):
    """Lamp name and brightness level extracted from the same input."""
    processor = EmbeddingsNERProcessor(types=[Lamp, BrightnessLevel])
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string("set bedroom lamp to 50%", context, recognized)

    types_found = {e.type for e in recognized}
    assert Lamp in types_found
    assert BrightnessLevel in types_found


@pytest.mark.benchmark
async def test_always_returns_empty_results_and_zero_pops(make_context):
    """Cornerstone 5: live call must still return ([], 0)."""
    processor = EmbeddingsNERProcessor(types=[Lamp])
    async with make_context(processors=[processor]) as (manager, context, collector):
        results, pops = await processor.process_string("turn on bedroom lamp", context, [])

    assert results == []
    assert pops == 0


@pytest.mark.benchmark
async def test_matched_substring_is_in_input(make_context):
    """Cornerstone 2 spirit: every matched substring must be a real substring of the input."""
    processor = EmbeddingsNERProcessor(types=[Song, Lamp])
    input_str = "play stairway to heaven and turn on the desk lamp"
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string(input_str, context, recognized)

    for entity in recognized:
        assert entity.substring in input_str


@pytest.mark.benchmark
async def test_type_embedding_cache(make_context):
    """Type embeddings must be computed once and reused across calls on the same instance."""
    processor = EmbeddingsNERProcessor(types=[Lamp])
    async with make_context(processors=[processor]) as (manager, context, collector):
        await processor.process_string("turn on the lamp", context, [])
        assert processor._type_embeddings is not None
        snapshot = processor._type_embeddings.copy()

        await processor.process_string("dim the lamp", context, [])
        assert (processor._type_embeddings == snapshot).all()


@pytest.mark.benchmark
async def test_no_match_on_unrelated_input(make_context):
    """Input with no entity-like content should produce no confident matches."""
    processor = EmbeddingsNERProcessor(types=[Song], similarity_threshold=0.8)
    async with make_context(processors=[processor]) as (manager, context, collector):
        recognized: list[RecognizedEntity] = []
        await processor.process_string("the the the and or but", context, recognized)

    # Not a hard assert — embeddings are fuzzy. Just check no Song was hallucinated
    # with high confidence from pure stopwords.
    song_matches = [e for e in recognized if e.type is Song]
    assert len(song_matches) == 0

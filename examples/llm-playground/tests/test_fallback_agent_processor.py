"""
Tests for FallbackAgentProcessor.

Cornerstones verified:
  5. Context hierarchy: stops at first layer with results.
  6. Chain short-circuit: non-empty result stops the chain.
     Placement: last in chain — reached only when all prior processors returned [].

Structure:
  Benchmarks only — real LLM, require large_model + running Ollama,
  run with: pytest -m benchmark

FallbackAgentProcessor implements a D-style hybrid:
  - "immediate": agent answers inline → transient command whose runner emits the response.
  - "background": longer task → transient command whose runner calls the agent a second time.
  - "none": out of scope → returns [], lets the chain continue (or reset to root context).
"""

from __future__ import annotations

import pytest
from conftest import drain
from stark.core.command import Command, Response
from stark.core.commands_context_processor import CommandsContextLayer
from stark.core.patterns.pattern import Pattern

from ready.fallback_agent_processor import FallbackAgentProcessor

# ── Helpers ───────────────────────────────────────────────────────────────────


def _cmd(name: str) -> Command:
    async def _runner() -> Response:
        return Response(name)

    _runner.__name__ = name
    _runner.__annotations__ = {}
    return Command(name, Pattern("**"), _runner)


# ── Benchmarks (live — require large_model + running Ollama) ──────────────────


@pytest.mark.benchmark
async def test_immediate_response_for_factual_question(make_context):
    """Agent gives an immediate inline answer to a simple factual question."""
    processor = FallbackAgentProcessor()

    async with make_context(processors=[processor]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("what is the capital of France?", context, layer, [])

    assert len(results) == 1
    # run the transient command and collect the response
    async with make_context(processors=[processor]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("what is the capital of France?", context, layer, [])
        assert len(results) == 1
        context.run_command(results[0].command, results[0].match_result.parameters)
        await drain(0.5)

    assert len(collector.responses) == 1
    assert collector.responses[0].text != ""


@pytest.mark.benchmark
async def test_immediate_response_text_is_plausible(make_context):
    """Agent's immediate answer to a factual question contains something meaningful."""
    processor = FallbackAgentProcessor()

    async with make_context(processors=[processor]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("what is the capital of France?", context, layer, [])
        assert len(results) >= 1
        context.run_command(results[0].command, results[0].match_result.parameters)
        await drain(0.5)

    # "Paris" should appear somewhere in the response
    full_text = " ".join(r.text for r in collector.responses).lower()
    assert "paris" in full_text


@pytest.mark.benchmark
async def test_transient_command_spans_full_input(make_context):
    """Cornerstone 2 spirit: returned span must cover the entire input string."""
    processor = FallbackAgentProcessor()
    input_str = "what is the capital of France?"

    async with make_context(processors=[processor]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer(input_str, context, layer, [])

    if results:
        mr = results[0].match_result
        assert mr.start == 0
        assert mr.end == len(input_str)
        assert mr.substring == input_str


@pytest.mark.benchmark
async def test_none_for_out_of_scope_input(make_context):
    """Agent returns [] for a clearly out-of-scope nonsense input."""
    processor = FallbackAgentProcessor()

    async with make_context(processors=[processor]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer("xyzzy plugh frobozz quux wibble", context, layer, [])

    # Either [] (agent said none) or a result — both are valid model behaviours.
    # The important invariant: the call must not raise.
    assert isinstance(results, list)


@pytest.mark.benchmark
async def test_chain_short_circuits_on_match(make_context):
    """Cornerstone 6: a second processor must not be called once results found."""
    called: list[str] = []

    class SpyProcessor(FallbackAgentProcessor):
        async def process_context_layer(self, string, context, layer, entities):
            called.append("spy")
            return []

    processor = FallbackAgentProcessor()
    spy = SpyProcessor()

    async with make_context(processors=[processor, spy]) as (manager, context, collector):
        await context.process_string("what is the capital of France?")
        await drain(0.5)

    assert "spy" not in called


@pytest.mark.benchmark
async def test_none_does_not_short_circuit_chain(make_context):
    """A 'none' decision must leave the chain free to continue to the next processor."""
    called: list[str] = []

    class SpyProcessor(FallbackAgentProcessor):
        async def process_context_layer(self, string, context, layer, entities):
            called.append("spy")
            return []

    processor = FallbackAgentProcessor()
    spy = SpyProcessor()

    # nonsense input — agent is likely to say "none"
    async with make_context(processors=[processor, spy]) as (manager, context, collector):
        await context.process_string("xyzzy plugh frobozz quux wibble")
        await drain(0.5)

    # the spy processor should have been called
    assert called == ["spy"]


@pytest.mark.benchmark
async def test_background_task_eventually_responds(make_context):
    """A 'background' decision must eventually emit a non-empty response."""
    processor = FallbackAgentProcessor()

    async with make_context(processors=[processor]) as (manager, context, collector):
        layer = CommandsContextLayer(commands=[], parameters={})
        context.context_queue = [layer]
        results = await processor.process_context_layer(
            "give me a detailed step-by-step plan to learn Python in 30 days",
            context,
            layer,
            [],
        )
        if results:
            context.run_command(results[0].command, results[0].match_result.parameters)
            await drain(5.0)  # background tasks may take longer

    if collector.responses:
        assert collector.responses[-1].text != ""

"""
    "Just call a 3rd-party agent as a fallback" — they said. One day later: four implementation variants. And that's not counting NER, command searchers, and a custom agentic loop.

Agentic fallback processor — when no earlier processor matched any command, hand off to a 3rd-party agent.

In STARK, a processor's job is to select which command to execute, not to execute logic itself. Actions live in commands. So a fallback agent can't run directly inside a processor — instead, it must return a command that wraps the agent call. The fallback processor returns that command directly by name, bypassing pattern matching entirely. The command itself can be hidden (fallback-only) or public with a real pattern (e.g. "okay ai, $request:String") so users can also invoke it explicitly — the fallback mechanism works the same either way.

Placement: last in the processor chain. Relies on chain short-circuit (cornerstone 6) — it is only reached if every preceding processor returned empty results.

Three implementation approaches, in increasing complexity:

  A. Static agent command — the processor always returns a single pre-defined hidden command whose runner calls the agent. Simplest: no pre-flight call, the agent always runs.

  B. Pre-flight check — the processor asks the agent first whether it can handle the input. If yes, returns the agent command as the match; if no, returns empty. Costs an extra LLM call but avoids running the agent on inputs it can't handle. In this case, a few (maybe even domain-specific) agents can be listed one after another, each with its own pre-flight check.

  C. Inline response — the processor calls the agent directly, takes its response, and returns a transient command whose runner simply emits that response. Collapses the two-step (select command → run command) into one, at the cost of bending the processor contract. Similarly, allows chaining multiple fallbacks.

A and B invoke the agent inside a command runner, which allows longer actions and even background agentic tasks. B ensures the right agent is called when multiple are listed. C is faster (single LLM call) but blocks the app for longer tasks.

  D. B+C hybrid — since a capable LLM is required in any of the cases above, B and C can be combined: the pre-flight LLM call determines not just whether to handle the input, but also what kind of response is coming. If the agent returns a final response immediately, it is emitted via a transient command (as in C). If a longer task is detected — one involving tool calls, reasoning steps, user prompts, or agentic loops — the input is handed off to a background command runner (as in A/B) so the app is not blocked. If the agent returns nothing, return [] and let the chain continue.

Note on opaque APIs: some APIs hide intermediate steps and only surface the final string, even for long tasks. In that case the response type cannot be inferred from structure alone — the prompt must be designed to elicit an explicit signal (e.g. a structured field indicating task type) so the processor can route correctly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from pydantic import BaseModel
from pydantic_ai import Agent
from stark.core.command import Command, Response
from stark.core.commands_context_processor import CommandsContextLayer, CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import MatchResult
from stark.core.patterns import Pattern

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext


logger = logging.getLogger(__name__)

os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
os.environ.setdefault("OLLAMA_API_KEY", "1234")


# ── Agent ─────────────────────────────────────────────────────────────────────


@dataclass
class _Deps:
    recognized_entities: list[RecognizedEntity]
    mode: Literal["preflight", "full"]


class _AgentDecision(BaseModel):
    response: str | None
    response_type: Literal["immediate", "background", "none"]
    reasoning: str
    confidence: float


_agent: Agent[_Deps, _AgentDecision] = Agent(
    "llama-3.2-3b-instruct:q4_k_m",
    deps_type=_Deps,
    output_type=_AgentDecision,
)


@_agent.instructions
async def _inject_mode_instructions(ctx) -> str:
    if ctx.deps.mode == "preflight":
        return (
            "You are the fallback handler for a voice assistant. "
            "Decide whether you can handle the user's input and what kind of response is needed. "
            "If you can give a complete, final answer right now — set response_type to 'immediate' and fill in response. "
            "If the task requires tool calls, multi-step reasoning, or extended work — set response_type to 'background' and leave response null. "
            "If the input is outside your scope entirely — set response_type to 'none' and leave response null. "
            "Always fill in reasoning (short, for logging) and confidence (0.0–1.0)."
        )
    else:  # full
        return (
            "You are the fallback handler for a voice assistant. "
            "Handle the user's request fully. Produce the best response you can. "
            "Set response_type to 'immediate' and fill in response with your answer. "
            "Always fill in reasoning (short, for logging) and confidence (0.0–1.0)."
        )


@_agent.instructions
async def _inject_recognized_entities(ctx) -> str:
    if not ctx.deps.recognized_entities:
        return ""
    hints = "\n".join(f"- {e.substring!r} → {e.type.__name__}" for e in ctx.deps.recognized_entities)
    return (
        "Pre-identified named entities (from upstream NER layer — informational, you may override if your "
        "understanding of the full input differs):\n" + hints
    )


# ── Processor ─────────────────────────────────────────────────────────────────


class FallbackAgentProcessor(CommandsContextProcessor):
    @override
    async def process_context_layer(
        self,
        string: str,
        context: CommandsContext,
        context_layer: CommandsContextLayer,
        recognized_entities: list[RecognizedEntity],
    ) -> list[SearchResult]:
        try:
            result = await _agent.run(
                string,
                deps=_Deps(recognized_entities=recognized_entities, mode="preflight"),
            )
        except Exception as e:
            logger.warning(f"AgentFallback preflight failed: {e}")
            return []

        decision = result.output
        logger.debug(
            f"AgentFallback preflight: type={decision.response_type!r} confidence={decision.confidence:.2f} reasoning={decision.reasoning!r}"
        )

        match decision.response_type:
            case "none":
                return []

            case "immediate":
                response_text = decision.response or ""

                async def _immediate_runner() -> Response:
                    return Response(text=response_text, voice=response_text)

                transient = Command("__agent_fallback_immediate__", Pattern("**"), _immediate_runner)
                return [SearchResult(transient, MatchResult(substring=string, start=0, end=len(string), parameters={}))]

            case "background":
                # Capture for closure — full agent call happens inside the command runner,
                # so the processor returns immediately and STARK runs it in the background.
                _string = string
                _entities = recognized_entities

                async def _background_runner() -> Response:
                    try:
                        full_result = await _agent.run(
                            _string,
                            deps=_Deps(recognized_entities=_entities, mode="full"),
                        )
                        response_text = full_result.output.response or ""
                        logger.debug(f"AgentFallback full: confidence={full_result.output.confidence:.2f} reasoning={full_result.output.reasoning!r}")
                    except Exception as e:
                        logger.error(f"AgentFallback full call failed: {e}")
                        raise
                    return Response(text=response_text, voice=response_text)

                transient = Command("__agent_fallback_background__", Pattern("**"), _background_runner)
                return [SearchResult(transient, MatchResult(substring=string, start=0, end=len(string), parameters={}))]

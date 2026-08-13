"""
Experimental two-step search + parsing processor.

Splits the search processor task into two sequential LLM calls:
  1. Search — given the user input and available commands (via CommandInfo), identify which command(s) match and extract the trigger substring.
  2. Parse  — given the matched command and its substring, extract and instantiate each typed parameter (via TypeInfo), bypassing PatternParser entirely.

Theoretically, simplifying each step may allow using smaller, faster, and less capable LLMs per step,
or improve accuracy compared to a single one-shot call that must do both simultaneously.
Trade-offs:
    - two sequential generation requests will generally have higher total latency than a single structured-output one-shot call.
    - separation of command search and parameter parsing removes command validation by parameters.

Pipeline role — three valid placements (always after NER pre-processors, before fallbacks):
  - Before SearchProcessor: LLM runs first; if it returns results, SearchProcessor is skipped (first
    non-empty result wins in the processor chain).
  - Instead of SearchProcessor: pure LLM pipeline, no pattern matching at all. Simpler, but loses
    pattern-based validation and overlap resolution.
  - After SearchProcessor: patterns run first for inputs they can handle; if they fail — e.g. due to
    natural speech variations or phrasing the pattern didn't anticipate — the LLM gets a chance.
    This is often the most practical setup: fast and precise patterns where they work, LLM as a capable fallback where they don't.

A one-shot alternative is also available — definitely worth comparing, as both approaches have trade-offs.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from stark.core.command import Command
from stark.core.commands_context_processor import CommandsContextLayer, CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import MatchResult
from stark.general.json_encoder import CommandInfo, TypeInfo

from stark_ai import agent_defaults

from .dev_raise import dev_raise

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext


logger = logging.getLogger(__name__)


# ── Step 1: Search ────────────────────────────────────────────────────────────


@dataclass
class _SearchDeps:
    command_infos: list[CommandInfo]


class _CommandMatch(BaseModel):
    command_name: str = Field(description="Exact name of the matched command from the available commands list")
    substring: str = Field(description="The uninterrupted substring of the user's input that triggered this command match")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 for this match")


type _CommandMatches = list[_CommandMatch]

_search_agent: Agent[_SearchDeps, _CommandMatches] = Agent(
    model=agent_defaults.MODEL_NAME,
    deps_type=_SearchDeps,
    output_type=_CommandMatches,
    instructions=(
        "You are the command search component of a natural language voice assistant. "
        "Given the user's input, find the most relevant command(s) — none, one, or several. "
        "Return only the command name and the exact substring of the input that triggered it."
        "Matched substrings must not overlap across commands. "
        "Each match must be an uninterrupted substring of the original input."
        "Only return matches you are confident about."
        "Your are only allowed to output valid JSON tool calls. Whenever you want to present a final answer use one of the final_result tools available to you, never answer with plain text."
    ),
)


@_search_agent.instructions
async def _inject_commands(ctx) -> str:
    lines = [f"- {info.as_text()}" for info in ctx.deps.command_infos]
    return "Available commands:\n" + "\n".join(lines)


# ── Step 2: Parse ─────────────────────────────────────────────────────────────


@dataclass
class _ParseDeps:
    command_info: CommandInfo
    type_infos: list[TypeInfo]
    matched_substring: str
    recognized_entities: list[RecognizedEntity]


class ParsedParameter(BaseModel):
    name: str = Field(description="Parameter name as declared in the command signature")
    value: str = Field(
        description="Extracted value as a clean, code-friendly string — not a raw NL phrase. E.g. for a song name: 'Bohemian Rhapsody', not 'play bohemian rhapsody by queen'"
    )


type _ParsedParameters = list[ParsedParameter]

_parse_agent: Agent[_ParseDeps, _ParsedParameters] = Agent(
    model=agent_defaults.MODEL_NAME,
    deps_type=_ParseDeps,
    output_type=_ParsedParameters,
    instructions=(
        "You are the parameter extraction component of a natural language voice assistant. "
        "Extract each parameter of the matched command as a clean, code-friendly value. "
        "You are given the full user input — use the whole sentence as context, the way a human would. "
        "Natural speech often omits repeated information (e.g. a device named earlier applies to a "
        "later command too). Infer parameter values from the full context, not just the matched part. "
        "E.g. for a song name: 'Bohemian Rhapsody', not 'play bohemian rhapsody by queen'."
    ),
)


@_parse_agent.instructions
async def _inject_command_and_types(ctx) -> str:
    parts = [
        f"Matched command: {ctx.deps.command_info.as_text()}",
        f'Matched part of the input: "{ctx.deps.matched_substring}"',
    ]
    if ctx.deps.type_infos:
        parts.append("Parameter types:\n" + "\n".join(f"- {t.as_text()}" for t in ctx.deps.type_infos))
    if ctx.deps.recognized_entities:
        hints = "\n".join(f"- {e.substring!r} → {e.type.__name__}" for e in ctx.deps.recognized_entities)
        parts.append(
            f"Pre-identified named entities (from upstream NER layer — informational, you may override if your understanding of the full input differs):\n{hints}"
        )
    return "\n\n".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _collect_type_infos(commands: list[Command]) -> list[TypeInfo]:
    """Collect TypeInfo for all unique NLObject subclass parameter types across commands."""
    from stark.core.types import NLObject

    seen: set[str] = set()
    result: list[TypeInfo] = []
    for cmd in commands:
        for _, param_type in cmd._runner.__annotations__.items():
            if inspect.isclass(param_type) and issubclass(param_type, NLObject) and param_type.__name__ not in seen:
                seen.add(param_type.__name__)
                result.append(TypeInfo.from_type(param_type))
    return result


def _instantiate_parameters(cmd: Command, parsed_params: list[ParsedParameter]) -> dict[str, object]:
    """
    Instantiate NLObject subclasses from LLM-extracted string values.
    Bypasses PatternParser / ObjectParser / did_parse — the LLM already produced clean values.
    Every declared NLObject parameter is present in the result; None if not parsed.
    """
    from stark.core.types import NLObject

    parsed_by_name = {p.name: p.value for p in parsed_params}
    result: dict[str, object] = {}
    for param_name, param_type in cmd._runner.__annotations__.items():
        if not (inspect.isclass(param_type) and issubclass(param_type, NLObject)):
            continue
        if param_name not in parsed_by_name:
            result[param_name] = None
            continue
        try:
            result[param_name] = param_type(parsed_by_name[param_name])
        except Exception as e:
            dev_raise(f"Failed to instantiate {param_type.__name__} for param '{param_name}'", e)
            result[param_name] = None
    return result


def _assign_indices(results: list[SearchResult], commands_order: list[str]) -> None:
    """Set SearchResult.index from declaration order — lower index = higher priority."""
    order = {name: i for i, name in enumerate(commands_order)}
    for result in results:
        result.index = order.get(result.command.name, len(commands_order))


def _resolve_overlaps(results: list[SearchResult]) -> list[SearchResult]:
    """
    Drop lower-priority overlapping results. No pattern to re-match against, so cutting
    a result shorter is not possible — drop is the only resolution.

    Priority (higher = wins):
      1. More non-None parameters (more specific command variant).
      2. Lower index (earlier-declared command).
      3. Higher confidence score (stored as result._confidence by the processor).
    """
    if len(results) <= 1:
        return results

    def _priority(r: SearchResult) -> tuple[int, int, float]:
        filled = sum(1 for v in r.match_result.parameters.values() if v is not None)
        confidence: float = getattr(r, "_confidence", 0.0)
        return (filled, -r.index, confidence)

    def _overlaps(a: SearchResult, b: SearchResult) -> bool:
        return a.match_result.start < b.match_result.end and b.match_result.start < a.match_result.end

    sorted_results = sorted(results, key=lambda r: r.match_result.start)
    kept: list[SearchResult] = []

    for candidate in sorted_results:
        dominated = False
        to_remove: list[SearchResult] = []
        for existing in kept:
            if not _overlaps(existing, candidate):
                continue
            if _priority(candidate) > _priority(existing):
                to_remove.append(existing)
            else:
                dominated = True
                break
        if dominated:
            continue
        for r in to_remove:
            kept.remove(r)
        kept.append(candidate)

    return sorted(kept, key=lambda r: r.match_result.start)


# ── Processor ─────────────────────────────────────────────────────────────────


class TwoStepLLMProcessor(CommandsContextProcessor):
    @override
    async def process_context_layer(
        self,
        string: str,
        context: CommandsContext,
        context_layer: CommandsContextLayer,
        recognized_entities: list[RecognizedEntity],
    ) -> list[SearchResult]:
        commands = context_layer.commands
        if not commands:
            return []

        cmd_by_name = {cmd.name: cmd for cmd in commands}
        command_infos = [CommandInfo.from_command(cmd) for cmd in commands]

        # Step 1: search
        try:
            search_response = await _search_agent.run(
                string,
                deps=_SearchDeps(command_infos=command_infos),
            )
        except Exception as e:
            dev_raise(f"TwoStep LLM search failed: {e}")
            return []

        results: list[SearchResult] = []

        for match in search_response.output:
            cmd = cmd_by_name.get(match.command_name)
            if cmd is None:
                dev_raise(f"LLM search returned unknown command: {match.command_name!r}")
                continue
            if match.substring not in string:
                dev_raise(f"LLM search returned substring not in input: {match.substring!r}")
                continue

            # Step 2: parse against the full input — LLM uses whole-sentence context
            parameters = await self._parse_parameters(cmd, string, match.substring, recognized_entities)
            start = string.find(match.substring)

            result = SearchResult(
                cmd,
                MatchResult(
                    substring=match.substring,
                    start=start,
                    end=start + len(match.substring),
                    parameters=parameters,
                ),
            )
            result._confidence = match.confidence  # type: ignore[attr-defined]
            results.append(result)
            logger.debug(f"TwoStep matched '{match.command_name}' substring={match.substring!r} params={parameters}")

        _assign_indices(results, [cmd.name for cmd in commands])
        return _resolve_overlaps(results)

    async def _parse_parameters(
        self,
        cmd: Command,
        full_input: str,
        matched_substring: str,
        recognized_entities: list[RecognizedEntity],
    ) -> dict[str, object]:
        type_infos = _collect_type_infos([cmd])
        if not type_infos:
            return {}

        logger.debug(f"LLM TwoStep: string={full_input!r}")
        try:
            response = await _parse_agent.run(
                full_input,
                deps=_ParseDeps(
                    command_info=CommandInfo.from_command(cmd),
                    type_infos=type_infos,
                    matched_substring=matched_substring,
                    recognized_entities=recognized_entities,
                ),
            )
        except Exception as e:
            dev_raise(f"TwoStep LLM parse failed for '{cmd.name}': {e}")
            return {}
        return _instantiate_parameters(cmd, response.output)

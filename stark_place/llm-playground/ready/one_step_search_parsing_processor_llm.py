'''
Combined one-shot search + parsing processor.

Identifies the matching command(s) and extracts all typed parameters in a single LLM call.
Almost an agent at this point — but no open tasks or thinking loops yet.

Advantages:
    - command and parameters are matched together in a single LLM context, so the model can use
      parameter fit to confirm command identity — partially recovering the validation that the
      two-step approach loses by splitting search and parsing.
    - still lighter than the full agent.
    - produces a more deterministic outcome with lower LLM quality expectations than the agent
      since it only chooses from available commands and uses structured output.
    - single LLM call — may have lower total latency than two sequential calls in some cases.

Trade-offs:
    - requires a somewhat smarter model with better attention and a larger context window to reason
      over all commands and all parameter types simultaneously, compared to the split two-step option.

Pipeline role — same three placements as the two-step option apply here; see its docstring for details.
A two-step alternative is also available — definitely worth comparing, as both approaches have trade-offs.
'''
from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from stark.core.command import Command
from stark.core.commands_context_processor import CommandsContextLayer, CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import MatchResult
from stark.general.json_encoder import CommandInfo, TypeInfo

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext


logger = logging.getLogger(__name__)

os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
os.environ.setdefault("OLLAMA_API_KEY", "1234")


# ── Agent ─────────────────────────────────────────────────────────────────────

@dataclass
class _Deps:
    command_infos: list[CommandInfo]
    type_infos: list[TypeInfo]
    recognized_entities: list[RecognizedEntity]


class ParsedParameter(BaseModel):
    name: str = Field(description="Parameter name as declared in the command signature")
    value: str = Field(description="Extracted value as a clean, code-friendly string — not a raw NL phrase. E.g. for a song name: 'Bohemian Rhapsody', not 'play bohemian rhapsody by queen'")


class _ParsedCommand(BaseModel):
    command_name: str = Field(description="Exact name of the matched command from the available commands list")
    substring: str = Field(description="The uninterrupted substring of the user's input that corresponds to this command")
    parameters: list[ParsedParameter] = Field(default_factory=list, description="All parameters extracted from the input for this command")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


type _ParsedCommands = list[_ParsedCommand]

_agent: Agent[_Deps, _ParsedCommands] = Agent(
    "llama-3.2-3b-instruct:q4_k_m",
    deps_type=_Deps,
    output_type=_ParsedCommands,
    instructions=(
        "You are the core processor of a natural language voice assistant. "
        "Given the user's input, identify which command(s) match and extract all their typed parameters in one shot. "
        "Use the whole sentence as context the way a human would — natural speech often omits repeated information "
        "(e.g. a device named earlier in the sentence applies to a later command too). "
        "Matched substrings must not overlap across commands. "
        "Parameter values must be clean and code-friendly — extract the semantic entity, not the raw phrase. "
        "Only return matches you are confident about."
    ),
)


@_agent.instructions
async def _inject_commands(ctx) -> str:
    lines = [f"- {info.as_text()}" for info in ctx.deps.command_infos]
    return "Available commands:\n" + "\n".join(lines)


@_agent.instructions
async def _inject_types(ctx) -> str:
    if not ctx.deps.type_infos:
        return ""
    lines = [f"- {t.as_text()}" for t in ctx.deps.type_infos]
    return "Parameter types:\n" + "\n".join(lines)


@_agent.instructions
async def _inject_recognized_entities(ctx) -> str:
    if not ctx.deps.recognized_entities:
        return ""
    hints = "\n".join(f"- {e.substring!r} → {e.type.__name__}" for e in ctx.deps.recognized_entities)
    return f"Pre-identified named entities (from upstream NER layer — informational, you may override if your understanding of the full input differs):\n{hints}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_type_infos(commands: list[Command]) -> list[TypeInfo]:
    """Collect TypeInfo for all unique Object subclass parameter types across commands."""
    from stark.core.types import Object
    seen: set[str] = set()
    result: list[TypeInfo] = []
    for cmd in commands:
        for _, param_type in cmd._runner.__annotations__.items():
            if inspect.isclass(param_type) and issubclass(param_type, Object) and param_type.__name__ not in seen:
                seen.add(param_type.__name__)
                result.append(TypeInfo.from_type(param_type))
    return result


def _instantiate_parameters(cmd: Command, parsed_params: list[ParsedParameter]) -> dict[str, object]:
    """
    Instantiate Object subclasses from LLM-extracted string values.
    Bypasses PatternParser / ObjectParser / did_parse — the LLM already produced clean values.
    Every declared Object parameter is present in the result; None if not parsed.
    """
    from stark.core.types import Object
    parsed_by_name = {p.name: p.value for p in parsed_params}
    result: dict[str, object] = {}
    for param_name, param_type in cmd._runner.__annotations__.items():
        if not (inspect.isclass(param_type) and issubclass(param_type, Object)):
            continue
        if param_name not in parsed_by_name:
            result[param_name] = None
            continue
        try:
            result[param_name] = param_type(parsed_by_name[param_name])
        except Exception as e:
            logger.warning(f"Failed to instantiate {param_type.__name__} for param '{param_name}': {e}")
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

class OneShotLLMProcessor(CommandsContextProcessor):

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
        type_infos = _collect_type_infos(commands)

        try:
            response = await _agent.run(
                string,
                deps=_Deps(command_infos=command_infos, type_infos=type_infos, recognized_entities=recognized_entities),
            )
        except Exception as e:
            logger.warning(f"OneShot LLM failed: {e}")
            return []

        results: list[SearchResult] = []

        for parsed in response.output:
            cmd = cmd_by_name.get(parsed.command_name)
            if cmd is None:
                logger.warning(f"LLM returned unknown command: {parsed.command_name!r}")
                continue
            if parsed.substring not in string:
                logger.warning(f"LLM returned substring not in input: {parsed.substring!r}")
                continue

            parameters = _instantiate_parameters(cmd, parsed.parameters)
            start = string.find(parsed.substring)

            result = SearchResult(
                cmd,
                MatchResult(
                    substring=parsed.substring,
                    start=start,
                    end=start + len(parsed.substring),
                    parameters=parameters,
                ),
            )
            result._confidence = parsed.confidence  # type: ignore[attr-defined]
            results.append(result)
            logger.debug(f"OneShot matched '{parsed.command_name}' substring={parsed.substring!r} params={parameters}")

        _assign_indices(results, [cmd.name for cmd in commands])
        return _resolve_overlaps(results)

'''
LLM-powered search among existing commands with no parameter parsing
Note: parameters dict is empty — a separate parsing step is needed after match.
See draft/DESIGN.md for the required core changes.
See ready solutions for alternatives.
'''
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from stark.core.commands_context_processor import CommandsContextLayer, CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import MatchResult
from stark.general.json_encoder import CommandInfo

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext


@dataclass
class LLMCommandSearchDeps:
    current_context_commands: list[CommandInfo]
    # context vars
    # history

class LLMCommandMatch(BaseModel):
    command_name: str = Field(description="Exact name of the matched command from the available commands list")
    substring: str = Field(description="The uninterrupted substring of the user's input that triggered this command match")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 for this match")
    # parameters: dict[str, str]

type LLMCommandMatches = list[LLMCommandMatch]

os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
os.environ.setdefault("OLLAMA_API_KEY", "1234")
search_agent = Agent(
    'llama-3.2-3b-instruct:q4_k_m',
    deps_type=LLMCommandSearchDeps,
    output_type=LLMCommandMatches,
    instructions=(
        "You're the core processor of user request in a natural language personal assistant framework. "
        "Your task is to understand user's request and find the most relevant among available command(s) - none, one, or a few."
    ),
)


@search_agent.instructions
async def _inject_commands(ctx) -> str:
    lines = [f"- {info.as_text()}" for info in ctx.deps.current_context_commands]
    return "Available commands:\n" + "\n".join(lines)

logger = logging.getLogger(__name__)

class LLMCommandSearchProcessor(CommandsContextProcessor):

    # CommandsContextProcessor Implementation

    @override
    async def process_context_layer(
        self,
        string: str,
        context: CommandsContext,
        context_layer: CommandsContextLayer,
        recognized_entities: list[RecognizedEntity],
    ) -> list[SearchResult]:
        return await self._search_command(string, context_layer)

    # Private

    async def _search_command(self, string: str, context_layer: CommandsContextLayer) -> list[SearchResult]:
        try:
            response = await search_agent.run(string, deps=LLMCommandSearchDeps(
                current_context_commands=[CommandInfo.from_command(cmd) for cmd in context_layer.commands]
            ))
        except Exception as e:
            logger.warning(f"LLM command search failed: {e}")
            return []

        matches: LLMCommandMatches = response.output

        cmd_name_to_cmd = {cmd.name: cmd for cmd in context_layer.commands}

        # command names -> search results

        results = []
        for match in matches:
            if match.command_name not in cmd_name_to_cmd:
                logger.warning(f"LLM returned unknown command name: {match.command_name!r}")
                continue

            start = string.find(match.substring)
            results.append(SearchResult(
                cmd_name_to_cmd[match.command_name],
                MatchResult(
                    match.substring,
                    start,
                    start + len(match.substring),
                    await self._recognize_parameters(match.substring),
                ),
            ))
        return results

    async def _recognize_parameters(self, substring: str) -> dict:
        raise NotImplementedError # see the docstring

'''
LLM-only-powered full parameter parsing (as an evolution of LLM-powered NER) — given a pre-matched command and substring, parse and instantiate typed parameters with code-friendly values
Note: this is a draft. Pure parse-only requires the caller to pass a pre-matched (command, substring) pair.
That split is currently not supported by the processor pipeline — search and parsing are fused in SearchProcessor.
See draft/DESIGN.md for the required core changes.
See ready solutions for alternatives.
'''

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from stark.core.command import Command

from stark.general.json_encoder import CommandInfo, TypeInfo


logger = logging.getLogger(__name__)

os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
os.environ.setdefault("OLLAMA_API_KEY", "1234")


@dataclass
class _ParseDeps:
    command_info: CommandInfo
    type_infos: list[TypeInfo]


class ParsedParameter(BaseModel):
    name: str = Field(description="Parameter name as declared in the command signature")
    value: str = Field(description="Extracted value as a clean, code-friendly string — not a raw NL phrase. E.g. for a song name: 'Bohemian Rhapsody', not 'play bohemian rhapsody by queen'")


_parse_agent: Agent[_ParseDeps, list[ParsedParameter]] = Agent(
    "llama-3.2-3b-instruct:q4_k_m",
    deps_type=_ParseDeps,
    output_type=list[ParsedParameter],
    instructions=(
        "You are the parameter extraction component of a natural language voice assistant. "
        "Given a user's input substring and a matched command, extract each parameter as a clean, "
        "code-friendly value — not a raw NL phrase. E.g. for a song name: 'Bohemian Rhapsody', "
        "not 'play bohemian rhapsody by queen'."
    ),
)


@_parse_agent.instructions
async def _inject_command_and_types(ctx) -> str:
    parts = [f"Matched command: {ctx.deps.command_info.as_text()}"]
    if ctx.deps.type_infos:
        parts.append("Parameter types:\n" + "\n".join(f"- {t.as_text()}" for t in ctx.deps.type_infos))
    return "\n\n".join(parts)


async def parse_parameters(cmd: Command, substring: str) -> dict[str, object]:
    """Parse parameters for a pre-matched command from its trigger substring using an LLM."""
    type_infos = collect_type_infos([cmd])
    if not type_infos:
        return {}
    try:
        response = await _parse_agent.run(
            substring,
            deps=_ParseDeps(
                command_info=CommandInfo.from_command(cmd),
                type_infos=type_infos,
            ),
        )
    except Exception as e:
        logger.warning(f"LLM parameter parsing failed for '{cmd.name}': {e}")
        return {}
    return instantiate_parameters(cmd, response.output)


def collect_type_infos(commands: list[Command]) -> list[TypeInfo]:
    """Collect TypeInfo for all unique Object subclass parameter types across commands."""
    from stark.core.types import Object
    seen: set[str] = set()
    type_infos: list[TypeInfo] = []
    for cmd in commands:
        for param_name, param_type in cmd._runner.__annotations__.items():
            if inspect.isclass(param_type) and issubclass(param_type, Object) and param_type.__name__ not in seen:
                seen.add(param_type.__name__)
                type_infos.append(TypeInfo.from_type(param_type))
    return type_infos


def instantiate_parameters(cmd: Command, parsed_params: list[ParsedParameter]) -> dict[str, object]:
    """
    Directly instantiate Object subclasses from LLM-extracted string values.
    Bypasses PatternParser / ObjectParser / did_parse entirely — the LLM already gave us clean values.
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
            obj = param_type(parsed_by_name[param_name])
            result[param_name] = obj
            logger.debug(f"Instantiated {param_type.__name__}({parsed_by_name[param_name]!r}) for param '{param_name}'")
        except Exception as e:
            logger.warning(f"Failed to instantiate {param_type.__name__} for param '{param_name}': {e}")
            result[param_name] = None

    return result

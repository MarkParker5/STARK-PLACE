"""
LLM-powered NER markup — finds substrings in input corresponding to registered Object types.
Experimental.
The use of STARK's built-in options for NER, or any other NER-first ML model is preferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from stark.core.commands_context_processor import CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import ObjectType
from stark.general.json_encoder import TypeInfo

from stark_place.ai import agent_defaults

from .dev_raise import dev_raise

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext


logger = logging.getLogger(__name__)


@dataclass
class LLMNERDeps:
    type_infos: list[TypeInfo]


class RecognizedEntityMatch(BaseModel):
    type_name: str = Field(description="Exact name of the matched type from the available types list")
    substring: str = Field(description="The uninterrupted substring of the user's input that corresponds to this entity")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 for this match")


type RecognizedEntityMatches = list[RecognizedEntityMatch]

_ner_agent: Agent[LLMNERDeps, RecognizedEntityMatches] = Agent(
    model=agent_defaults.MODEL_NAME,
    deps_type=LLMNERDeps,
    output_type=RecognizedEntityMatches,
    instructions=(
        "You are an NER (Named Entity Recognition) system for a voice assistant framework. "
        "Identify and extract all substrings of the user's input that correspond to any of the provided entity types, so that they can be passed later as arguments to a command call. "
        "Only return matches you are confident about. "
        "Each match must be an uninterrupted substring of the original input."
        "Your are only allowed to output valid JSON tool calls. Whenever you want to present a final answer use one of the final_result tools available to you, never answer with plain text."
    ),
)


@_ner_agent.instructions
async def _inject_types(ctx) -> str:
    type_infos: list[TypeInfo] = ctx.deps.type_infos
    lines = "\n".join(f"- {info.as_text()}" for info in type_infos)
    return f"Available entity types:\n{lines}"


# @_ner_agent.output_validator # called too late, after pydantic has already raised the validation error
# def fix_result(ctx: RunContext, result):
#     logger.debug("Called custom output validator, result=%s", result)
#     # Case 1: model returned tool call and pydantic forgot to unwrap it
#     if isinstance(result, dict) and result.get("name") == "final_result" and "arguments" in result:
#         logger.debug("Final result tool call detected, arguments=%s", result["arguments"])
#         return result["arguments"]
#     return result


class LLMNERProcessor(CommandsContextProcessor):
    def __init__(self, types: list[ObjectType]):
        self._types = types
        self._type_infos = [TypeInfo.from_type(t) for t in types]
        self._type_by_name = {t.__name__: t for t in types}

    # CommandsContextProcessor Impl

    @override
    async def process_string(
        self,
        string: str,
        context: CommandsContext,
        recognized_entities: list[RecognizedEntity],
    ) -> tuple[list[SearchResult], int]:
        if not self._types:
            return [], 0

        logger.debug(f"LLM NER: string={string!r}")

        try:
            response = await _ner_agent.run(
                string,
                deps=LLMNERDeps(type_infos=self._type_infos),
            )
        except Exception as e:
            dev_raise(e)
            return [], 0

        for match in response.output:
            entity_type = self._type_by_name.get(match.type_name)
            if entity_type is None:
                dev_raise(f"LLM NER returned unknown type name: {match.type_name!r}")
                continue
            if match.substring not in string:
                dev_raise(f"LLM NER returned substring not found in input: {match.substring!r}")
                continue
            logger.debug(f"LLM NER found '{match.substring}' -> {match.type_name} (confidence={match.confidence:.2f})")
            recognized_entities.append(RecognizedEntity(match.substring, entity_type))

        return [], 0

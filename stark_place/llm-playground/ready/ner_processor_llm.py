'''
LLM-powered NER markup — finds substrings in input corresponding to registered Object types.
Experimental.
The use of STARK's built-in options for NER, or any other NER-first ML model is preferred.
'''
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from stark.core.commands_context_processor import CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import ObjectType
from stark.general.json_encoder import TypeInfo

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext


logger = logging.getLogger(__name__)

os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
os.environ.setdefault("OLLAMA_API_KEY", "1234")


@dataclass
class LLMNERDeps:
    type_infos: list[TypeInfo]


class RecognizedEntityMatch(BaseModel):
    type_name: str = Field(description="Exact name of the matched type from the available types list")
    substring: str = Field(description="The uninterrupted substring of the user's input that corresponds to this entity")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 for this match")


type RecognizedEntityMatches = list[RecognizedEntityMatch]

_ner_agent: Agent[LLMNERDeps, RecognizedEntityMatches] = Agent(
    "llama-3.2-3b-instruct:q4_k_m",
    deps_type=LLMNERDeps,
    output_type=RecognizedEntityMatches,
    instructions=(
        "You are an NER (Named Entity Recognition) system for a voice assistant framework. "
        "Identify and extract all substrings of the user's input that correspond to any of the provided entity types. "
        "Only return matches you are confident about. "
        "Each match must be an uninterrupted substring of the original input."
    ),
)


@_ner_agent.instructions
async def _inject_types(ctx) -> str:
    type_infos: list[TypeInfo] = ctx.deps.type_infos
    lines = "\n".join(f"- {info.as_text()}" for info in type_infos)
    return f"Available entity types:\n{lines}"


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

        try:
            response = await _ner_agent.run(
                string,
                deps=LLMNERDeps(type_infos=self._type_infos),
            )
        except Exception as e:
            logger.warning(f"LLM NER failed: {e}")
            return [], 0

        for match in response.output:
            entity_type = self._type_by_name.get(match.type_name)
            if entity_type is None:
                logger.warning(f"LLM NER returned unknown type name: {match.type_name!r}")
                continue
            if match.substring not in string:
                logger.warning(f"LLM NER returned substring not found in input: {match.substring!r}")
                continue
            logger.debug(f"LLM NER found '{match.substring}' -> {match.type_name} (confidence={match.confidence:.2f})")
            recognized_entities.append(RecognizedEntity(match.substring, entity_type))

        return [], 0

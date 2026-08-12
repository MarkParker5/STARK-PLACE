"""
Embeddings-powered search among existing commands (no parameter parsing)
Note: returns whole input string as match substring — no span localization.
See draft/DESIGN.md for the required core changes.
See ready solutions for alternatives.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, override

import numpy as np
from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider
from stark.core.commands_context_processor import CommandsContextLayer, CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import MatchResult
from stark.general.json_encoder import CommandInfo

from .dev_raise import dev_raise

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext

logger = logging.getLogger(__name__)

_embedder = Embedder(
    OpenAIEmbeddingModel(
        "nomic-embed-text",
        provider=OpenAIProvider(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:8080/v1"),
            api_key=os.environ.get("OLLAMA_API_KEY", "1234"),
        ),
    )
)

_SIMILARITY_THRESHOLD = 0.5
_CACHE_MAX_BYTES = 1 * 1024 * 1024  # 1MB

# LRU cache: tuple[str, ...] -> np.ndarray of shape (n_commands, dims)
_embedding_cache: OrderedDict[tuple[str, ...], np.ndarray] = OrderedDict()
_embedding_cache_bytes: int = 0


def _cache_entry_size(embeddings: np.ndarray) -> int:
    return embeddings.nbytes


def _cache_get(key: tuple[str, ...]) -> np.ndarray | None:
    if key not in _embedding_cache:
        return None
    _embedding_cache.move_to_end(key)
    return _embedding_cache[key]


def _cache_put(key: tuple[str, ...], embeddings: np.ndarray) -> None:
    global _embedding_cache_bytes
    entry_size = _cache_entry_size(embeddings)
    if key in _embedding_cache:
        _embedding_cache_bytes -= _cache_entry_size(_embedding_cache[key])
        del _embedding_cache[key]
    while _embedding_cache and _embedding_cache_bytes + entry_size > _CACHE_MAX_BYTES:
        _, evicted = _embedding_cache.popitem(last=False)
        _embedding_cache_bytes -= _cache_entry_size(evicted)
    _embedding_cache[key] = embeddings
    _embedding_cache_bytes += entry_size


class EmbeddingCommandSearchProcessor(CommandsContextProcessor):
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
        commands = context_layer.commands
        if not commands:
            return []

        command_infos = [CommandInfo.from_command(cmd) for cmd in commands]
        command_texts = [info.as_text() for info in command_infos]
        cache_key = tuple(command_texts)

        try:
            query_vec = np.array((await _embedder.embed_query(string)).embeddings[0])
            cached = _cache_get(cache_key)
            if cached is None:
                cached = np.array((await _embedder.embed_documents(command_texts)).embeddings)
                _cache_put(cache_key, cached)
        except Exception as e:
            dev_raise(e)
            return []

        # cosine similarity: (n_commands,)
        sims = (cached / np.linalg.norm(cached, axis=1, keepdims=True)) @ (query_vec / np.linalg.norm(query_vec))

        results: list[tuple[float, SearchResult]] = []
        for cmd, sim in zip(commands, sims):
            if sim < _SIMILARITY_THRESHOLD:
                continue
            results.append(
                (
                    float(sim),
                    SearchResult(
                        cmd,
                        MatchResult(
                            string,
                            start := 0,
                            start + len(string),
                            {},
                        ),
                    ),
                )
            )

        results.sort(key=lambda x: x[0], reverse=True)
        return [sr for _, sr in results]

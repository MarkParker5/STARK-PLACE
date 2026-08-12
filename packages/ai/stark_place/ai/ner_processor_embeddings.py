"""
Experimental.
Embeddings-powered NER markup — finds substrings in input corresponding to registered Object types. Uses sliding window to find substrings which might affect performance, but it should be compensated by the speed of generating embeddings relative to LLM output.
The use of STARK's built-in options for NER, or any other NER-first ML model is preferred.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

import numpy as np
from stark.core.commands_context_processor import CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import ObjectType
from stark.general.json_encoder import TypeInfo

from stark_place.ai import agent_defaults

from .dev_raise import dev_raise

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext


logger = logging.getLogger(__name__)

_embedder = agent_defaults.embedder()

_SIMILARITY_THRESHOLD = 0.5
# NER targets open-ended ** types (song names, artist names, etc.) — pattern shape doesn't help.
# Use a flat word cap and embed all windows once, compare against all types in one matrix multiply.
_MAX_WINDOW_WORDS = 8


def _sliding_window_candidates(string: str, max_words: int) -> list[str]:
    """All unique substrings of 1..max_words words from string, left to right."""
    words = string.split()
    n = len(words)
    seen: set[str] = set()
    candidates: list[str] = []
    for size in range(1, min(max_words, n) + 1):
        for i in range(n - size + 1):
            candidate = " ".join(words[i : i + size])
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


class EmbeddingsNERProcessor(CommandsContextProcessor):
    """
    NER processor for open-ended entity types (song names, artists, addresses, etc.)
    that can't be resolved by pattern matching alone.

    Embeds all sliding-window candidates once, then finds the best-matching type
    for each candidate via a single cosine similarity matrix multiply.
    Only the highest-scoring candidate per type is kept, above the threshold.

    Type embeddings are cached per instance (types are fixed at init time).
    """

    def __init__(
        self,
        types: list[ObjectType],
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        max_window_words: int = _MAX_WINDOW_WORDS,
    ):
        self._types = types
        self._type_infos = [TypeInfo.from_type(t) for t in types]
        self._similarity_threshold = similarity_threshold
        self._max_window_words = max_window_words
        # cached type embeddings matrix: shape (n_types, dims), set on first call
        self._type_embeddings: np.ndarray | None = None

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

        candidates = _sliding_window_candidates(string, self._max_window_words)
        if not candidates:
            return [], 0

        logger.debug(f"Embeddings NER: {len(candidates)} candidates, {len(self._types)} types")

        try:
            type_vecs = await self._get_type_embeddings()  # (n_types, dims)
            candidate_vecs = np.array((await _embedder.embed_documents(candidates)).embeddings)  # (n_candidates, dims)
        except Exception as e:
            dev_raise(e)
            return [], 0

        # cosine similarity matrix: (n_candidates, n_types)
        candidate_norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        type_norms = np.linalg.norm(type_vecs, axis=1, keepdims=True)
        sims = (candidate_vecs / candidate_norms) @ (type_vecs / type_norms).T

        # for each type, pick the single best candidate above threshold
        best_candidate_indices = np.argmax(sims, axis=0)  # (n_types,)
        best_similarities = sims[best_candidate_indices, np.arange(len(self._types))]

        for obj_type, best_idx, best_sim in zip(self._types, best_candidate_indices, best_similarities):
            if best_sim < self._similarity_threshold:
                continue
            best_candidate = candidates[int(best_idx)]
            logger.debug(f"Embeddings NER matched '{best_candidate}' -> {obj_type.__name__} (similarity={float(best_sim):.3f})")
            recognized_entities.append(RecognizedEntity(best_candidate, obj_type))

        return [], 0

    # Private

    async def _get_type_embeddings(self) -> np.ndarray:
        if self._type_embeddings is not None:
            return self._type_embeddings

        type_texts = [info.as_text() for info in self._type_infos]
        self._type_embeddings = np.array((await _embedder.embed_documents(type_texts)).embeddings)
        logger.debug(f"Cached embeddings for {len(self._types)} types.")
        return self._type_embeddings

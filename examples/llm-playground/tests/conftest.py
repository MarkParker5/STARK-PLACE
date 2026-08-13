"""
Shared fixtures for llm-playground tests.

Model configuration lives in ready/agent_defaults.py.
Set STARK_LLM_BASE_URL, STARK_LLM_API_KEY, and STARK_LLM_MODEL_* env vars to point
all processors at your local or remote endpoint. See agent_defaults.py for presets.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import AsyncGenerator

import anyio
import asyncer
import pytest

os.environ["ENV"] = "TEST"

# Configure logging

logging.getLogger("ready").setLevel(logging.DEBUG)  # always log DEBUG for the current package (llm-playground)

# too noisy or not relevant
logging.getLogger("stark").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ------


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Pin all anyio-backed async tests to asyncio (pydantic-ai is not trio-compatible)."""
    return "asyncio"


from stark.core.command import Response
from stark.core.commands_context import CommandsContext
from stark.core.commands_context_processor import CommandsContextLayer
from stark.core.commands_manager import CommandsManager
from stark.core.parsing import ObjectType
from stark.core.patterns.pattern import Pattern
from stark.core.types.object import NLObject
from stark.general.classproperty import classproperty
from stark.general.dependencies import DependencyManager

# ── Shared NLObject types for tests ─────────────────────────────────────────────


class Lamp(NLObject[str]):
    """A smart lamp or light fixture that can be controlled by name."""

    @classproperty
    def pattern(cls) -> Pattern:
        return Pattern("**")


class Song(NLObject[str]):
    """A music track identified by its title."""

    @classproperty
    def pattern(cls) -> Pattern:
        return Pattern("**")


class Artist(NLObject[str]):
    """A music artist or band name."""

    @classproperty
    def pattern(cls) -> Pattern:
        return Pattern("**")


class BrightnessLevel(NLObject[str]):
    """A brightness percentage or named level (e.g. '50%', 'max', 'low')."""

    @classproperty
    def pattern(cls) -> Pattern:
        return Pattern("**")


# ── Response collector ────────────────────────────────────────────────────────


class ResponseCollector:
    """Minimal CommandsContextDelegate that collects responses for assertions."""

    def __init__(self) -> None:
        self.responses: list[Response] = []

    async def commands_context_did_receive_response(self, response: Response) -> None:
        self.responses.append(response)

    async def remove_response(self, response: Response) -> None:
        if response in self.responses:
            self.responses.remove(response)


# ── Context factory fixture ───────────────────────────────────────────────────


@pytest.fixture
def make_context():
    """
    Factory fixture: returns an async context manager that yields
    (manager, context, collector) wired up and running.

    Usage::

        async with make_context(processors=[MyProcessor()]) as (manager, context, collector):
            @manager.new("turn on $lamp:Lamp")
            def lamp_on(lamp: Lamp) -> Response: ...

            await context.process_string("turn on the bedroom lamp")
            await anyio.sleep(0.05)
            assert collector.responses[0].text == "..."
    """

    @contextlib.asynccontextmanager
    async def _factory(
        processors: list | None = None,
        object_types: list[ObjectType] | None = None,
    ) -> AsyncGenerator[tuple[CommandsManager, CommandsContext, ResponseCollector], None]:
        async with asyncer.create_task_group() as tg:
            manager = CommandsManager()
            deps = DependencyManager()
            context = CommandsContext(tg, manager, deps, processors=processors or [])
            for ot in object_types or []:
                context.pattern_parser.register_parameter_type(ot)
            collector = ResponseCollector()
            context.delegate = collector
            tg.soonify(context.handle_responses)()
            await anyio.sleep(0)  # let handle_responses start and set is_stopped=False before yielding
            yield manager, context, collector
            context.stop()

    return _factory


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_layer(commands: list, parameters: dict | None = None) -> CommandsContextLayer:
    return CommandsContextLayer(commands=commands, parameters=parameters or {})


async def drain(seconds: float = 0.05) -> None:
    """Yield control long enough for soonify'd command tasks to complete."""
    await anyio.sleep(seconds)

import asyncer
import pytest

from stark_devtools.profiler import profile
from stark_devtools.profiler.capture import STARK_DIR
from stark.core import CommandsContext, CommandsManager, Response


def make_demo_manager(sync_play: bool = False) -> CommandsManager:
    manager = CommandsManager()

    if sync_play:
        @manager.new("play $band:Word")
        def play_music(band):  # sync -> runs via asyncify in a worker thread
            return Response(f"Playing {band}.")
    else:
        @manager.new("play $band:Word")
        async def play_music(band):
            return Response(f"Playing {band}.")

    @manager.new("turn off the ** lights")
    async def lights_off():
        return Response("Lights off.")

    return manager


@pytest.fixture
def make_manager():
    return make_demo_manager


@pytest.fixture
def run_utterance():
    async def _run(utterance, manager=None, root=STARK_DIR):
        manager = manager or make_demo_manager()
        # profile() wraps the whole task group so scheduled command tasks (and their responses,
        # possibly on worker threads) run while capture is active.
        with profile(root=root) as session:
            async with asyncer.create_task_group() as task_group:
                context = CommandsContext(task_group=task_group, commands_manager=manager)
                await context.process_string(utterance)
        return session.events

    return _run

"""
The most powerful processor in the chain — an agentic loop that runs inside a single STARK command,
with full access to the command space and conversation context in one shot.

Unlike all other processors, this one does not return SearchResults to the pipeline directly.
Instead it matches and executes a single hidden "agentic" command, and the real work — calling other
commands, asking the user, reasoning over history — happens inside that command's runner via an LLM
agent loop. From STARK core's perspective: one command was matched, one runner ran.

─── Context layers and message history ───────────────────────────────────────────────────────────────

STARK's context_queue was designed to simulate contextual understanding without real language
comprehension — a layered stack of local commands and parameters that lets the core resolve short
follow-up inputs ("brightness 50" after "turn on light 4") and drive multi-step menus and confirmations
(Response.commands + Response.parameters push a new layer; matched commands pop it).

An LLM agent does not need this simulation. It handles contextual continuity natively through:
  - full conversation history (all prior exchanges, not just the current layer),
  - all loaded context layers (their commands and parameters) passed as a single flattened input,
  - the full root command registry simultaneously visible in the same prompt.

Context layers are still passed to the agent — not for resolution logic, but because they carry
meaningful signal: active layer commands represent what the application currently considers in scope,
and layer parameters carry values (e.g. a selected device) the agent should treat as ambient context.

─── Tool taxonomy ────────────────────────────────────────────────────────────────────────────────────

There is no split between "STARK command tools" and "capability tools" — all tools are registered at
the same level and the agent decides freely which to call:

  STARK commands (dynamic, from all context layers + root):
    Each command is a tool. The agent calls them to perform actions. Calling a command tool is an await —
    the agent waits for the result before continuing. Commands are already background by design (run via
    task_group.soonify); the tool call itself completes when the command runner finishes or yields.

  respond (special built-in tool):
    The agent can emit intermediate or final Response objects at any point in the loop — for partial
    feedback, follow-up questions (needs_user_input=True), or status updates. Maps directly to
    CommandsContext.respond(). This is how the agent drives the same push-new-context / pop-context
    mechanism that native commands use (Response.commands, Response.parameters).

  Capability tools — extension, to be implemented after the main agentic loop is working:
    - cli            : run a shell command; returns stdout/stderr.
    - http           : make an HTTP request; returns status + body.
    - web            : browser automation or search; returns structured results.
    - screen_kb      : screenshot -> vision context; emit keyboard/mouse events.
    - sandboxed_code : run code in an isolated environment.
    - system_code    : run trusted host-level code (gated).

─── Mid-run interruption and concurrency ─────────────────────────────────────────────────────────────

Multiple things can be "in flight" simultaneously: one active LLM generation, plus any number of
background command runners already dispatched. Concurrency is at the LLM call level, not at the
command level — commands run freely in parallel; only LLM generations are serialised.

A supervisor (observer + LLM call queue) coordinates this. Requirements:

  LLM call queue:
    - Only one LLM generation runs at a time. Any code path that needs an LLM call enqueues and
      waits until the current generation completes — this is the serialisation point.
    - Background command runners are not queued; they run concurrently and are never blocked by it.

  Message injection:
    - Between tool calls, external inputs can be injected into the running loop as additional messages.
      Intended for:
        * Adjusting or refining the current task mid-execution.
        * Answering a question the agent asked the user (needs_user_input=True).
        * Requesting a progress report on the current task state.
    - Injection never triggers a restart. Messages are folded directly into the in-progress run by
      appending UserPromptParts to the next ModelRequestNode's request before it is sent to the model.
      There are two fold points:
        * Before a ModelRequestNode executes: any pending injections are appended to that node's
          request parts. The LLM sees them alongside whatever context is already in the request.
        * After a CallToolsNode completes: agent_run.next(call_tools_node) returns the ModelRequestNode
          that CallToolsNode built from tool return parts. Injections are appended there, merging
          injected messages with tool results in a single ModelRequest — no history gaps, no lost work.
    - Injection never happens mid-generation (while the LLM lock is held). Both fold points are at
      node boundaries, between lock acquire and lock release.

  Progress tracking:
    - Tasks report their own progress to the supervisor — the supervisor does not poll.
    - The supervisor aggregates emitted updates and exposes the current task state to outside observers
      (UI, other commands, etc.). Encapsulation: the task knows its state; the supervisor only relays.

─── Cornerstones compliance ──────────────────────────────────────────────────────────────────────────

The agentic processor operates outside the pattern-native cornerstones by design:

  Superseded (not applicable in agentic mode):
    - Real spans: span-based overlap resolution is a pattern-matching concern; the agent resolves
      intent semantically from full input and history, not by substring boundaries.
    - Multi-command via SearchResults: the agent handles multi-intent inputs internally by calling
      multiple command tools in sequence — STARK core sees a single matched command (the agentic
      runner itself). No overlap resolution is needed at the processor level.
    - Context hierarchy traversal / context_pops: the agent receives all layers at once; it does not
      iterate them one at a time. process_string always returns (results, 0).

  Still applicable:
    - All params guaranteed (cornerstone 3): command tools receive typed arguments from the agent;
      every declared Object parameter must be present in the call (None if not provided). The tool
      wrapper enforces this before invoking the command runner.
    - Chain short-circuit (cornerstone 6): this processor returns [] if the agent decides the input
      is out of scope, allowing any downstream processor to handle it.

─── Pipeline placement ───────────────────────────────────────────────────────────────────────────────

Dead last in the processor chain. Reached only when every preceding processor returned empty.
The agentic runner itself is a hidden command (not in any CommandsManager); returned as a transient
SearchResult pointing to the full input string (start=0, end=len(string)), consistent with the
fallback processor pattern.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override
from uuid import UUID, uuid4

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_graph import End
from stark.core.command import Command, Response
from stark.core.commands_context_processor import CommandsContextLayer, CommandsContextProcessor, RecognizedEntity
from stark.core.commands_manager import SearchResult
from stark.core.parsing import MatchResult
from stark.core.patterns import Pattern
from stark.general.json_encoder import CommandInfo, TypeInfo

from stark_ai import agent_defaults

from .dev_raise import dev_raise

if TYPE_CHECKING:
    from stark.core.commands_context import CommandsContext


logger = logging.getLogger(__name__)


# ── Supervisor ────────────────────────────────────────────────────────────────


@dataclass
class ProgressUpdate:
    run_id: UUID
    command_name: str
    message: str


@dataclass
class AgenticLoopSupervisor:
    """Observer + LLM call queue, one per CommandsContext instance.

    LLM generations are serialised via asyncio.Lock — only one runs at a time.
    Background command runners are never blocked.

    Each call to process_string produces a unique run_id. Injections and
    progress observers are scoped to that run_id so concurrent runs never
    bleed into each other.

    Lifecycle (called by AgenticLoopProcessor):
        run_id = supervisor.register_run()     # allocate per-run state
        ...
        supervisor.unregister_run(run_id)      # release per-run state

    External callers (UI, other commands):
        supervisor.inject(run_id, message)
        supervisor.set_progress_observer(run_id, cb)
    """

    _llm_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _injection_queues: dict[UUID, asyncio.Queue[str]] = field(default_factory=dict)
    _progress_observers: dict[UUID, Callable[[ProgressUpdate], None]] = field(default_factory=dict)

    # ── Run lifecycle ─────────────────────────────────────────────────────────

    def register_run(self) -> UUID:
        """Allocate per-run state and return the new run_id."""
        run_id = uuid4()
        self._injection_queues[run_id] = asyncio.Queue()
        return run_id

    def unregister_run(self, run_id: UUID) -> None:
        """Release per-run state. Safe to call even if run_id is unknown."""
        self._injection_queues.pop(run_id, None)
        self._progress_observers.pop(run_id, None)

    # ── Per-run API ───────────────────────────────────────────────────────────

    def set_progress_observer(self, run_id: UUID, cb: Callable[[ProgressUpdate], None]) -> None:
        """Register a progress callback for a specific run."""
        self._progress_observers[run_id] = cb

    def report_progress(self, run_id: UUID, command_name: str, message: str) -> None:
        """Called by command tool wrappers to report execution progress."""
        cb = self._progress_observers.get(run_id)
        if cb:
            cb(ProgressUpdate(run_id=run_id, command_name=command_name, message=message))

    def inject(self, run_id: UUID, message: str) -> None:
        """Inject a message into a specific running loop (between tool calls)."""
        queue = self._injection_queues.get(run_id)
        if queue is None:
            dev_raise(f"AgenticLoopSupervisor.inject: unknown run_id {run_id}")
            return
        queue.put_nowait(message)

    def drain_injections(self, run_id: UUID) -> list[str]:
        """Drain all pending injected messages for a run. Called at each iteration boundary."""
        queue = self._injection_queues.get(run_id)
        if queue is None:
            return []
        messages: list[str] = []
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    @property
    def llm_lock(self) -> asyncio.Lock:
        return self._llm_lock


# ── Deps ──────────────────────────────────────────────────────────────────────


@dataclass
class _Deps:
    context: CommandsContext
    supervisor: AgenticLoopSupervisor
    all_commands: list[Command]
    all_layers: list[CommandsContextLayer]
    command_infos: list[CommandInfo]
    type_infos: list[TypeInfo]
    recognized_entities: list[RecognizedEntity]


# ── Agent ─────────────────────────────────────────────────────────────────────

_agent: Agent[_Deps, str] = Agent(
    model=agent_defaults.MODEL_NAME,
    deps_type=_Deps,
    output_type=str,
    instructions=(
        "You are the agentic core of a natural language voice assistant. "
        "You have full access to the available commands and can call them as tools. "
        "Use the full conversation history and all context layers to understand the user's intent. "
        "Call the 'respond' tool to give intermediate feedback, ask follow-up questions, or confirm actions. "
        "Call command tools to perform actions. You may call multiple commands if the input requires it. "
        "Return an empty string as your final output if you have already responded via the 'respond' tool."
        "Only return results you are confident about."
        "Your are only allowed to output valid JSON tool calls. Whenever you want to present a final answer use one of the final_result tools available to you, never answer with plain text."
    ),
)


@_agent.instructions
async def _inject_commands(ctx: RunContext[_Deps]) -> str:
    if not ctx.deps.command_infos:
        return ""
    lines = [f"- {info.as_text()}" for info in ctx.deps.command_infos]
    return "Available commands (callable as tools):\n" + "\n".join(lines)


@_agent.instructions
async def _inject_types(ctx: RunContext[_Deps]) -> str:
    if not ctx.deps.type_infos:
        return ""
    lines = [f"- {t.as_text()}" for t in ctx.deps.type_infos]
    return "Parameter types:\n" + "\n".join(lines)


@_agent.instructions
async def _inject_layer_context(ctx: RunContext[_Deps]) -> str:
    parts: list[str] = []
    for i, layer in enumerate(ctx.deps.all_layers):
        label = "innermost active context" if i == 0 else f"context layer {i}"
        if layer.parameters:
            param_str = ", ".join(f"{k}={v}" for k, v in layer.parameters.items())
            parts.append(f"[{label}] ambient parameters: {param_str}")
    if not parts:
        return ""
    return "Active context layers (innermost first):\n" + "\n".join(parts)


@_agent.instructions
async def _inject_entities(ctx: RunContext[_Deps]) -> str:
    if not ctx.deps.recognized_entities:
        return ""
    hints = "\n".join(f"- {e.substring!r} → {e.type.__name__}" for e in ctx.deps.recognized_entities)
    return "Pre-identified named entities (from upstream NER — informational, override if needed):\n" + hints


# ── respond tool ──────────────────────────────────────────────────────────────


@_agent.tool
async def respond(
    ctx: RunContext[_Deps],
    text: str,
    needs_user_input: bool = False,
) -> str:
    """Emit a response to the user. Use for feedback, questions, confirmations, or final answers.
    Set needs_user_input=True when asking the user a question that requires a reply.
    """
    response = Response(text, voice=text, needs_user_input=needs_user_input)
    await ctx.deps.context.respond(response)
    logger.debug(f"AgenticLoop respond: {text!r} needs_user_input={needs_user_input}")
    return "response emitted"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _collect_type_infos(commands: list[Command]) -> list[TypeInfo]:
    from stark.core.types import Object

    seen: set[str] = set()
    result: list[TypeInfo] = []
    for cmd in commands:
        # inspect.get_annotations is PEP 563-safe (handles `from __future__ import annotations`)
        for _, param_type in inspect.get_annotations(cmd._runner, eval_str=True).items():
            if inspect.isclass(param_type) and issubclass(param_type, Object) and param_type.__name__ not in seen:
                seen.add(param_type.__name__)
                result.append(TypeInfo.from_type(param_type))
    return result


def _instantiate_parameters(cmd: Command, raw: dict[str, Any]) -> dict[str, object]:
    """Instantiate Object subclass parameters from LLM-provided string values.
    Every declared Object parameter is present in the result; None if missing or unparseable.
    """
    from stark.core.types import Object

    result: dict[str, object] = {}
    for param_name, param_type in inspect.get_annotations(cmd._runner, eval_str=True).items():
        if not (inspect.isclass(param_type) and issubclass(param_type, Object)):
            continue
        raw_value = raw.get(param_name)
        if raw_value is None:
            result[param_name] = None
            continue
        try:
            result[param_name] = param_type(str(raw_value))
        except Exception as e:
            dev_raise(f"Failed to instantiate {param_type.__name__} for '{param_name}'", e)
            result[param_name] = None
    return result


def _build_command_toolset(
    commands: list[Command],
    context: CommandsContext,
    supervisor: AgenticLoopSupervisor,
    run_id: UUID,
) -> FunctionToolset[_Deps]:
    """Build a FunctionToolset with one tool per command. Rebuilt per run — ephemeral."""
    toolset: FunctionToolset[_Deps] = FunctionToolset()

    for cmd in commands:
        # Capture sig/params outside the closure so each iteration binds correctly.
        sig = inspect.signature(cmd._runner)
        params = {name: param for name, param in sig.parameters.items() if name != "self"}

        def _make_tool_fn(bound_cmd: Command, bound_params: dict) -> Callable:
            async def _tool_fn(ctx: RunContext[_Deps], **kwargs: Any) -> str:
                from types import AsyncGeneratorType, GeneratorType

                parameters = _instantiate_parameters(bound_cmd, kwargs)
                logger.debug(f"AgenticLoop calling command {bound_cmd.name!r} with params={parameters}")
                supervisor.report_progress(run_id, bound_cmd.name, "started")
                try:
                    result = await bound_cmd(parameters)
                    # Command runners may produce output in four ways (see Command.run / commands_context.run_command):
                    #   return Response        — single response object
                    #   return AsyncGenerator  — stream of Response objects
                    #   return Generator       — sync stream (runs on thread to avoid blocking)
                    #   return None            — runner called context.respond() directly, or no output
                    # We handle all cases here since we are calling the command outside of run_command().
                    # TODO: review runner, resolve duplication, check parameters and context
                    if isinstance(result, Response):
                        await ctx.deps.context.respond(result)
                    elif isinstance(result, AsyncGeneratorType):
                        async for r in result:
                            if r is not None:
                                await ctx.deps.context.respond(r)
                    elif isinstance(result, GeneratorType):
                        for r in result:
                            if r is not None:
                                await ctx.deps.context.respond(r)
                    supervisor.report_progress(run_id, bound_cmd.name, "done")
                    return f"command '{bound_cmd.name}' completed"
                except Exception as e:
                    supervisor.report_progress(run_id, bound_cmd.name, f"failed: {e}")
                    dev_raise(f"AgenticLoop command {bound_cmd.name!r} failed", e)
                    return f"command '{bound_cmd.name}' failed: {e}"

            _tool_fn.__name__ = bound_cmd.name
            _tool_fn.__doc__ = inspect.getdoc(bound_cmd._runner) or f"Execute command: {bound_cmd.name}"
            _tool_fn.__annotations__ = {
                "ctx": RunContext[_Deps],
                **{
                    name: (p.annotation if p.annotation is not inspect.Parameter.empty else str)
                    for name, p in bound_params.items()
                },
                "return": str,
            }
            return _tool_fn

        toolset.add_function(_make_tool_fn(cmd, params), takes_ctx=True, name=cmd.name)

    return toolset


# ── Agentic runner ────────────────────────────────────────────────────────────


def _fold_injections(supervisor: AgenticLoopSupervisor, run_id: UUID, node: Any) -> None:
    """Drain pending injected messages for run_id and append them as UserPromptParts to node.request.parts.

    node must be a ModelRequestNode — its .request.parts list is mutated in place.
    No-op if the injection queue is empty.
    """
    injected = supervisor.drain_injections(run_id)
    if not injected:
        return
    combined = "\n".join(injected)
    node.request.parts.append(UserPromptPart(combined))
    logger.debug(f"AgenticLoop folded {len(injected)} injection(s) into node.request.parts")


async def _run_agentic_loop(
    string: str,
    context: CommandsContext,
    supervisor: AgenticLoopSupervisor,
    run_id: UUID,
    all_layers: list[CommandsContextLayer],
    recognized_entities: list[RecognizedEntity],
    message_history: list[ModelMessage],
) -> list[ModelMessage]:
    """Run the agent loop inside the hidden command's runner.

    LLM serialisation: the supervisor lock is acquired only around ModelRequestNode steps
    (the actual LLM generation). It is released before CallToolsNode executes, so tool
    calls (command runners) never block other coroutines waiting on the LLM lock.

    Injection: injected messages are folded directly into the in-progress run — no restart.
    There are two injection points:

      1. Before a ModelRequestNode runs: drain injections and append UserPromptParts to
         node.request.parts. The LLM call that follows sees both the existing context and
         the injected messages in one shot.

      2. After a CallToolsNode completes: await agent_run.next(call_tools_node) returns
         the next ModelRequestNode (which already contains all tool return parts built by
         CallToolsNode._handle_tool_calls). Drain injections and append UserPromptParts to
         that node's request.parts before passing it to the next LLM step. Tool results
         and injected messages are merged into a single ModelRequest — no history loss,
         no abandoned work, no restart.

    Returns the updated message history for the caller to persist.
    """
    all_commands: list[Command] = []
    seen_names: set[str] = set()
    for layer in all_layers:
        for cmd in layer.commands:
            if cmd.name not in seen_names:
                all_commands.append(cmd)
                seen_names.add(cmd.name)

    command_infos = [CommandInfo.from_command(cmd) for cmd in all_commands]
    type_infos = _collect_type_infos(all_commands)

    deps = _Deps(
        context=context,
        supervisor=supervisor,
        all_commands=all_commands,
        all_layers=all_layers,
        command_infos=command_infos,
        type_infos=type_infos,
        recognized_entities=recognized_entities,
    )

    command_toolset = _build_command_toolset(all_commands, context, supervisor, run_id)

    current_history = list(message_history)

    logger.debug(f"AgenticLoop preflight: string={string!r}")

    async with _agent.iter(
        string,
        deps=deps,
        message_history=current_history if current_history else None,
        toolsets=[command_toolset],
    ) as agent_run:
        node = agent_run.next_node

        while not isinstance(node, End):
            if Agent.is_model_request_node(node):
                # Injection point 1: before the LLM call.
                # Append any pending messages directly to this node's request parts.
                # The model will see them alongside whatever context is already in the request.
                _fold_injections(supervisor, run_id, node)

                # Acquire the LLM lock only for the duration of the model call.
                # Released before tool execution so command runners are never blocked.
                async with supervisor.llm_lock:
                    node = await agent_run.next(node)

            elif Agent.is_call_tools_node(node):
                # Execute all tool calls — no lock held.
                node = await agent_run.next(node)

                # Injection point 2: after tools finish, before the next LLM call.
                # agent_run.next(call_tools_node) returns the ModelRequestNode that
                # CallToolsNode built from tool return parts. Appending UserPromptParts
                # here merges injected messages with tool results in a single ModelRequest —
                # the LLM sees both together with no history gaps and no restart.
                if Agent.is_model_request_node(node):
                    _fold_injections(supervisor, run_id, node)

            else:
                # UserPromptNode or any future node type — just advance.
                node = await agent_run.next(node)

        # Run completed naturally.
        final = agent_run.result
        if final:
            # Emit final text output if agent didn't already respond via tool.
            if final.output and final.output.strip():
                await context.respond(Response(final.output, voice=final.output))
            return agent_run.all_messages()
        return agent_run.all_messages()


# ── Processor ─────────────────────────────────────────────────────────────────


class AgenticLoopProcessor(CommandsContextProcessor):
    """Dead-last processor. Wraps the agentic loop in a single transient hidden command."""

    def __init__(self) -> None:
        self._supervisor = AgenticLoopSupervisor()
        self._message_history: list[ModelMessage] = []

    @property
    def supervisor(self) -> AgenticLoopSupervisor:
        """Expose supervisor for external injection and progress observation."""
        return self._supervisor

    @override
    async def process_string(
        self,
        string: str,
        context: CommandsContext,
        recognized_entities: list[RecognizedEntity],
    ) -> tuple[list[SearchResult], int]:
        all_layers = list(context.context_queue)
        if not all_layers:
            return [], 0

        _string = string
        _context = context
        _supervisor = self._supervisor
        _layers = all_layers
        _entities = recognized_entities
        _history = list(self._message_history)

        run_id = self._supervisor.register_run()

        async def _runner() -> None:
            try:
                updated = await _run_agentic_loop(_string, _context, _supervisor, run_id, _layers, _entities, _history)
                self._message_history = updated
            except Exception as e:
                dev_raise(e)
            finally:
                self._supervisor.unregister_run(run_id)

        transient = Command("__agentic_loop__", Pattern("**"), _runner)
        result = SearchResult(
            transient,
            MatchResult(substring=string, start=0, end=len(string), parameters={}),
        )
        # Always context_pops=0 — agent receives all layers at once, no layer iteration # TODO: review
        return [result], 0

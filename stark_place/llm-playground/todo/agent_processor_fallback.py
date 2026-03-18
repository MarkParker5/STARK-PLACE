'''
    "Just call a 3rd-party agent as a fallback" — they said. One day later: four implementation variants. And that's not counting NER, command searchers, and a custom agentic loop.

Agentic fallback processor — when no earlier processor matched any command, hand off to a 3rd-party agent.

In STARK, a processor's job is to select which command to execute, not to execute logic itself. Actions live in commands. So a fallback agent can't run directly inside a processor — instead, it must return a command that wraps the agent call. The fallback processor returns that command directly by name, bypassing pattern matching entirely. The command itself can be hidden (fallback-only) or public with a real pattern (e.g. "okay ai, $request:String") so users can also invoke it explicitly — the fallback mechanism works the same either way.

Placement: last in the processor chain. Relies on chain short-circuit (cornerstone 6) — it is only reached if every preceding processor returned empty results.

Three implementation approaches, in increasing complexity:

  A. Static agent command — the processor always returns a single pre-defined hidden command whose runner calls the agent. Simplest: no pre-flight call, the agent always runs.

  B. Pre-flight check — the processor asks the agent first whether it can handle the input. If yes, returns the agent command as the match; if no, returns empty. Costs an extra LLM call but avoids running the agent on inputs it can't handle. In this case, a few (maybe even domain-specific) agents can be listed one after another, each with its own pre-flight check.

  C. Inline response — the processor calls the agent directly, takes its response, and returns a transient command whose runner simply emits that response. Collapses the two-step (select command → run command) into one, at the cost of bending the processor contract. Similarly, allows chaining multiple fallbacks.

A and B invoke the agent inside a command runner, which allows longer actions and even background agentic tasks. B ensures the right agent is called when multiple are listed. C is faster (single LLM call) but blocks the app for longer tasks.

  D. B+C hybrid — since a capable LLM is required in any of the cases above, B and C can be combined: the pre-flight LLM call determines not just whether to handle the input, but also what kind of response is coming. If the agent returns a final response immediately, it is emitted via a transient command (as in C). If a longer task is detected — one involving tool calls, reasoning steps, user prompts, or agentic loops — the input is handed off to a background command runner (as in A/B) so the app is not blocked. If the agent returns nothing, return [] and let the chain continue.

Note on opaque APIs: some APIs hide intermediate steps and only surface the final string, even for long tasks. In that case the response type cannot be inferred from structure alone — the prompt must be designed to elicit an explicit signal (e.g. a structured field indicating task type) so the processor can route correctly.
'''

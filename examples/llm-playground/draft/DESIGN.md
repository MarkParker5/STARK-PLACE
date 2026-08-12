# Draft Processors — Required Core Design Changes

The processors in this folder are drafts because they depend on design changes to STARK core that have not been made yet. The code here is a starting point — helpers, data models, and ideas — not a working implementation.

---

## Problem 1: Search and Parameter Parsing Are Fused

### What the problem is

In the current pipeline, command search and parameter parsing are a single inseparable step. `SearchProcessor` calls `PatternParser.match()`, which simultaneously runs the regex to locate the command substring *and* parses every parameter inline. There is no hook between "found a command candidate" and "parsed its parameters".

This means:
- You cannot run LLM-based parameter parsing on a command that was already found by pattern search.
- You cannot run LLM-based search and then hand off to pattern-based parameter parsing.
- A `parsing_processor_llm.py` that only does parsing has nothing to receive — there is no intermediate result type in the pipeline that represents "matched command, parameters not yet parsed".

### Why it's designed this way

Parameter parsing is also parameter *validation*. If a parameter can't be parsed to its declared type, the command match is rejected. So parsing is what confirms a command identity — search and parse are co-dependent.

### Current ideas for possible solutions

**Option A — Allow splitting `MatchResult` into two phases**
Introduce an intermediate `CommandCandidate(command, substring)` type. Processors can return candidates instead of full results. A second pass (by a separate processor or by `CommandsContext` itself) then runs parameter parsing on each candidate. This requires adding `CommandCandidate` to `parsing.py`, adjusting `CommandsContextProcessor` to optionally return candidates, and updating `CommandsContext.process_string` to handle the two-phase flow.

**Option B — `PatternParser` protocol + DI** - allows alternative implementations of root type parsers, for example LLM-only parser, instead of the native solution based on pattern + `did_parse` hook.

Extract a `PatternParser` protocol (interface) with the two methods the framework actually uses: `register_parameter_type` and `match`. Allow injecting a custom implementation into `CommandsContext`. An `LLMPatternParser` could then implement `match()` to return a fully-populated `MatchResult` using an LLM call — search and parse in one shot — while `SearchProcessor` and the rest of the pipeline stay unchanged. This is the least invasive change to core.

---

## Problem 2: Embedding Search Has No Substring Localization

### What the problem is

`search_processor_embeddings.py` matches commands by cosine similarity between the full input and command descriptions. It returns the entire input string as the match substring. This breaks multi-command search: STARK's overlap resolution logic in `SearchProcessor` depends on each result having a real `(start, end)` span so it can cut overlapping matches.

### Ideas for possible solutions

**Option A — Sliding window over the input**
Generate word-window substrings of the input, embed each, and pick the window most similar to the matched command's embedding. Adds latency (more embed calls) but gives real spans.

**Option B — LLM substring extraction as a second step**
After similarity identifies the command(s), ask the LLM to locate the exact substring. One small extra call, but precise. 

Both of the options could benefit from implementing the `CommandCandidate`.

---

## Contributing

If you want any of the ideas above to be implemented, open an issue explaining your use case and which option you think fits best. Even better — implement it and open a PR. The helpers in `parsing_processor_llm.py` (`collect_type_infos`, `instantiate_parameters`, `ParsedParameter`) are reusable building blocks for whichever approach gets chosen.

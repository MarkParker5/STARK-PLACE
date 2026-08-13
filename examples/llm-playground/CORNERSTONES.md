# STARK Request Processing — Cornerstones

Invariants that every processor in the pipeline must uphold (or be explicitly aware of violating).

---

## 1. Multi-command matching

A single user input can trigger **multiple commands simultaneously**.
`CommandsContext.process_string` iterates every `SearchResult` returned and executes each as an independent command.
Processors must return *all* plausible matches from an input, not just the first one.

---

## 2. Overlap exclusion (non-reuse of substrings)

The **same substring of the input cannot be consumed by two different commands**.
`SearchProcessor` enforces this by comparing `(start, end)` spans after collecting all matches:

- If two results overlap, the lower-priority one is re-matched against the non-overlapping remainder.
- If the lower-priority result survives the cut, both are kept with adjusted spans.
- If neither can survive the cut, the lower-priority result is dropped entirely.

Priority is determined by `SearchResult.index` — **earlier-declared commands win**.

This rule is universal — pattern-based and LLM-based processors alike must return non-overlapping spans.

### LLM context awareness

LLMs understand the full input as a whole, so they can infer a command's parameter values from parts
of the input that belong to a *different* command's substring — without needing to re-consume those
characters. This is just how language models use context, not a relaxation of the overlap rule.

Example — input: `"turn on bedroom lamp and make the brightness 50%"`

Two non-overlapping commands are matched:
- `lamp on` → substring `"turn on bedroom lamp"`, `lamp` parameter resolved from within it.
- `lamp brightness` → substring `"make the brightness 50%"`, `level` parameter resolved from within it.
  The `lamp` parameter (`bedroom lamp`) is stated in the other command's span — the LLM infers it from
  context without that substring being re-assigned. No overlap occurs.

Consequence for LLM/embedding processors:
- Every result must carry a real `(start, end)` span pointing to a specific, non-empty substring of
  the original input. Returning the whole input as every match's span defeats overlap resolution.
- The LLM should be given the **full input** when extracting parameters, so it can use cross-command
  context to fill values that natural speech omits from the immediate phrase (pronoun resolution,
  ellipsis, shared subjects). This is not overlap — the span boundaries don't change.

### Nonlocal parameter values

A parameter value inferred from another command's substring (e.g. `lamp` resolved from `"turn on
bedroom lamp"` while parsing `lamp brightness`) is already handled correctly by the current data model.

`MatchResult.parameters` stores instantiated `NLObject` instances — semantic values, not source spans.
The command runner receives `Lamp("bedroom lamp")` and has no knowledge of where in the input that
value came from. Nothing in the core pipeline tracks or requires per-parameter source spans.

Overlap resolution operates only on `MatchResult.start/end` (the command's own substring span).
Individual parameter locality is invisible to it and irrelevant. No special handling is needed.

---

## 3. Parameters are guaranteed on every result

`MatchResult.parameters` is **always fully populated** with every parameter name declared by the command,
even if some could not be parsed. Unparsed or optional parameters are stored as `None`.

The core (`CommandsContext.process_string`) blindly merges `match_result.parameters` into the kwargs it
passes to the command runner. A missing key causes a `KeyError` at call time. A `None` value is safe —
the command signature is expected to declare those parameters as optional.

Consequence for LLM/embedding processors: after resolving parameter values, ensure the returned dict
contains every key from the command's declared parameter set, with `None` as the fallback.

---

## 4. Same-family priority: more filled parameters wins

When **multiple variants of the same command family** match the same input (e.g. `play $song:Song` and
`play $song:Song by $artist:Artist`), the variant with **more successfully parsed parameters** takes
priority.

In the native `SearchProcessor + PatternParser` path this emerges naturally:
- `PatternParser.match` sorts its results longest-substring-first (more text → more parameters filled).
- `_filter_overlapping_matches` keeps the longer match when two overlap at the same start position.

LLM/embedding processors must replicate this intent: if two results for the same command family overlap,
prefer the one with more non-`None` parameters. If confidence scores are available, they are a reasonable
proxy when parameter counts are equal.

---

## 5. Context hierarchy traversal

Commands are organised in a layered context queue (a stack with the most recent context at index 0).
Processors are asked to match against each layer in order, innermost first.

- If a layer yields results, traversal stops and those results are used.
- The number of layers traversed without a hit is returned as `context_pops` so
  `CommandsContext.process_string` can remove the now-irrelevant inner contexts.
- If no layer matches at all, the queue is reset to the root context.

Processors that do not implement per-layer search (e.g. NER pre-processors) **must return `([], 0)`**
so the context queue is left untouched for the next processor in the chain.

---

## 6. Processor chain short-circuit

Processors are tried **in order**; the first one to return a non-empty result list ends the search.
Subsequent processors in the chain are never called for that request.

Typical orderings:

| Position | Processor kind | Rationale |
|----------|---------------|-----------|
| First | NER pre-processors | Annotate `recognized_entities`; always return `([], 0)` |
| Middle | Fast/pattern search | Precise, low-latency; handles well-formed inputs |
| Last | LLM/embedding search | Capable fallback for natural-phrasing or unknown inputs |

---

## 7. NER pre-processing feeds parameter parsing

`recognized_entities` is a mutable list passed through **every** processor in order.
NER processors populate it; search/parse processors read it.

`PatternParser._parse_single_parameter` checks `recognized_entities` before invoking `ObjectParser`:
if an entity's substring is found inside the raw parameter candidate and its type matches, the parser
constrains the candidate to that substring before parsing. This lets upstream NER processors guide
pattern-based parameter extraction with higher-quality entity boundaries.

LLM search+parse processors that bypass `PatternParser` entirely may choose to read `recognized_entities`
as hints for parameter values, but they are not required to — they already extract values directly from
the LLM response.

---

## Processor compliance checklist

| Cornerstone | NER processors | Pattern `SearchProcessor` | LLM/embedding search processors |
|-------------|---------------|--------------------------|----------------------------------|
| 1. Multi-command | N/A (no results) | ✓ all matches returned | must return all confident matches |
| 2. Overlap exclusion | N/A | ✓ enforced internally | must return real `(start, end)` spans |
| 3. Parameters guaranteed | N/A | ✓ PatternParser fills `None` | must populate all declared param keys |
| 4. Same-family priority | N/A | ✓ longest match wins | prefer more-filled match when overlapping |
| 5. Context hierarchy | ✓ return `([], 0)` | ✓ per-layer iteration | override `process_context_layer` or `process_string` correctly |
| 6. Chain short-circuit | ✓ return `([], 0)` | implicit (non-empty stops chain) | implicit (non-empty stops chain) |
| 7. NER entities | populate list | read list | optionally read list |

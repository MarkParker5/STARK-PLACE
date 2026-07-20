# The escalation ladder — LLM freedom of action, step by step

Each rung is a distinct, separately-testable way to let an LLM act in STARK, ordered by
how much freedom the model is given. Every rung is measured on the controlled corpus
(`../benchmark/conversational/`) and its result snapshot + `manifest.json` is archived here.

Governing principle (measured, not assumed): **the model is a transformer over retrieved
context and tool output — never a knowledge oracle.** It may reason *over* what's in the
context; it must not answer *from* its compressed weights. Facts come from tools/RAG or the
model abstains.

| Rung | Freedom given | Trusted? (edge, ≤4B) | Key finding |
|---|---|---|---|
| **L0 transform** | none — rephrase context already provided | ✅ yes | small models faithfully rephrase provided data; this is their safe zone |
| **L1 rag** | read local state (home_state), answer from it | ⚠️ only if retrieval is **forced** | elected: they skip the tool and guess; forced: faithful |
| **L2 webtool** | call a read tool (web_search), summarize | ⚠️ only if **forced** | same — plus stubborn parametric leakage on "known" facts |
| **L3 action** | execute one command, observe output, confirm | ⚠️ | output-handling is fine; deciding *when* to act is the risk |
| **L4 multi-step** | chain reads + actions in one turn | 🔬 next | compounding latency + error |
| **L5 multi-turn** | conversation state, clarification, slot-filling | 🔬 next | — |
| **L6 agentic loop** | plan→act→observe (existing processor) | flagship only | out of edge scope |

## The rule the ladder produced

**Retrieve first, then transform — never let a small model decide whether to fetch.**
Model-*elected* fetch fails on small models (they fabricate); *forced* retrieval (the pipeline
runs the tool and injects the result) makes even a 1.5B model faithful and offline-safe.
The recommended implementation is `forced_retrieval_processor.py`.

## Archive layout
```
levels/
  README.md                       # this file
  forced_retrieval_processor.py   # the recommended Tier-2 pattern (mandatory RAG + transform)
  L0_transform/manifest.json      # per-rung finding + frozen metrics
  L1_rag/manifest.json
  L2_webtool/manifest.json
  L3_action/manifest.json
```
Metrics in each manifest are the mean over the conversational dataset at that rung; see
`../benchmark/conversational/report_conv/` for the full report and charts.

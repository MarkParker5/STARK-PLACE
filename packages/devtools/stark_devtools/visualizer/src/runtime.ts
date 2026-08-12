// The Brain as a RUNTIME OBJECT graph, built from the trace. Nodes are the objects/actors that
// carried the request's impulses; pure structural noise is kept low-weight rather than hidden, so
// the picture has mass without clutter. Key modelling choices:
//   * commands hang off a shared relay (CommandsManager owns them) instead of wiring straight to the
//     hub — far fewer hub spokes;
//   * edges are typed: `own` (a owns b, single direction), `peer` (mutual, same league, e.g. ctx↔va),
//     `flow` (data passes a→b). Weight is computed RECURSIVELY over ownership (a node inherits the
//     weight of everything it owns) and peers share a league — so genuinely central objects win;
//   * a node's ring is its graph depth from the hub, which the layout uses to reduce edge crossings.
import type { Step } from "./schema";

export interface RNode { id: string; label: string; kind: string; weight: number; depth: number; touched: boolean }
export type EdgeKind = "own" | "peer" | "flow";
export interface REdge { source: string; target: string; kind: EdgeKind }
export interface RFrame { nodes: string[]; edges: [string, string][]; label: string; stepIdxs: number[] }
export interface Runtime { nodes: RNode[]; edges: REdge[]; frames: RFrame[] }

const KIND_COLOR: Record<string, string> = {
  hub: "#f6b45a", processor: "#6cc08a", tool: "#e07aa6", matching: "#8f86c9",
  command: "#8fd6a8", response: "#7fd3c4", io: "#4a90d9", state: "#c9b06a", parse: "#a79ce0",
  data: "#7fd3c4", infra: "#586472",
};
export function kindColor(k: string): string { return KIND_COLOR[k] ?? "#7fd3c4"; }

function shorten(t: string, n = 20): string { return t.length <= n ? t : t.slice(0, n - 1) + "…"; }

export function buildRuntime(steps: Step[]): Runtime {
  const has = (label: string) => steps.some((s) => s.label === label);
  const nodes: RNode[] = [];
  const edges: REdge[] = [];
  const add = (id: string, label: string, kind: string) => { if (!nodes.find((n) => n.id === id)) nodes.push({ id, label, kind, weight: 1, depth: 0, touched: false }); };
  const has_ = (id: string) => nodes.some((n) => n.id === id);
  const E = (a: string, b: string, kind: EdgeKind) => { if (a !== b && has_(a) && has_(b) && !edges.find((e) => e.source === a && e.target === b && e.kind === kind)) edges.push({ source: a, target: b, kind }); };
  const own = (a: string, b: string) => E(a, b, "own");
  const flow = (a: string, b: string) => E(a, b, "flow");
  const peer = (a: string, b: string) => E(a, b, "peer");

  const corr = has("corrections"), ner = has("NER"), dict = has("dictionary lookup"), search = has("search") || has("processor pass");

  // ── nodes ──────────────────────────────────────────────────────────────
  // The full ARCHITECTURE is always present (so the brain shows the whole engine, untouched
  // parts merely muted). Only request-specific DATA objects (products, commands, responses) are
  // conditional on what this utterance actually produced.
  add("ctx", "CommandsContext", "hub");
  add("va", "VoiceAssistant", "io"); add("stt", "STT · Vosk", "io"); add("tts", "TTS · Silero", "io");
  add("speaker", "Speaker", "io"); add("queue", "Response queue", "io"); add("mode", "VA Mode", "io");
  add("ctxtree", "Context tree", "state");
  add("ctxlayer", "ContextLayer", "state"); // the actual pushed/popped runtime layer of the tree
  add("localizer", "Localizer", "tool");
  add("corr", "Corrections", "processor");
  add("ner", "SpacyNER", "processor");
  // dictionaries: the fuzzy corrections dict + the slot/vocabulary dict backing NLDictionaryName
  add("dict", "Dictionary·fuzzy", "tool"); add("vocabdict", "Dictionary·vocab", "tool");
  add("lev", "Levenshtein", "tool"); add("sp", "simplephone", "tool"); add("ipa", "IPA·espeak", "tool");
  add("dictitem", "DictionaryItem", "parse"); add("lookup", "LookupResult", "parse");
  add("search", "SearchProcessor", "processor");
  add("mgr", "CommandsManager", "matching"); add("pp", "PatternParser", "matching");
  add("word", "Word", "parse"); add("string", "String", "parse"); add("slots", "SlotsParser", "parse");
  add("union", "Union", "parse"); add("nldict", "NLDictionaryName", "parse"); add("pattern", "Pattern", "parse"); add("rules", "pattern rules", "parse");
  add("expanded", "ExpandedString", "parse"); // Pattern's vocabulary expansion — the literal input to matching
  add("objparser", "ObjectParser", "matching");
  add("matchresult", "MatchResult", "data"); add("searchresult", "SearchResult[]", "data");
  const cmdNames: string[] = [];
  steps.forEach((s) => { if (s.label === "dispatch") { const c = String(s.input?.command || "").split(".").pop(); if (c && !cmdNames.includes(c)) cmdNames.push(c); } });
  // responses now carry the command that emitted them (bg commands emit several over time)
  const respObjs = steps.filter((s) => s.label === "respond").map((s) => ({ text: (s.input?.response as any)?.text || "response", command: String(s.input?.command || "").split(".").pop() || "" }));
  const respTexts = respObjs.map((r) => r.text);
  cmdNames.forEach((c) => add("cmd_" + c, c, "command"));
  respTexts.forEach((t, i) => add("resp_" + i, "“" + shorten(t) + "”", "response"));
  // a command is "background" if it emitted more than one response
  const respCountByCmd: Record<string, number> = {};
  respObjs.forEach((r) => { if (r.command) respCountByCmd[r.command] = (respCountByCmd[r.command] || 0) + 1; });

  // ── the smart core: the actual NL objects the request produced (real values) ──
  const arr = (v: any) => (Array.isArray(v) ? v.filter((x) => typeof x === "string") : []);
  const corrections: string[] = [];
  steps.forEach((s) => { if (s.label === "corrections") corrections.push(...arr(s.output?.corrections)); });
  const entities: string[] = [];
  steps.forEach((s) => { if (s.label === "NER") entities.push(...arr(s.output?.entities)); });
  const params: string[] = [];
  const resStep = steps.find((s) => s.label === "search") || steps.find((s) => s.label === "process");
  const results = (resStep?.output as any)?.results;
  if (Array.isArray(results)) results.forEach((r: any) => { const p = r?.match?.parameters || {}; Object.entries(p).forEach(([k, v]) => { if (v != null && v !== "None" && String(v) !== "null") params.push(`${k}=${v}`); }); });
  corrections.forEach((c, i) => add("corrn_" + i, c, "data"));
  entities.forEach((e, i) => add("ent_" + i, e, "data"));
  params.forEach((p, i) => add("param_" + i, p, "data"));

  // ── infra tier (de-emphasized satellites — background, not the smart part) ──
  add("dm", "DependencyManager", "infra"); add("health", "health_check", "infra"); add("blockage", "BlockageDetector", "infra");
  add("mic", "Microphone", "infra"); add("relay", "RecognizerRelay", "infra");

  // ── edges (typed) — the full architecture, always wired ───────────────────
  peer("ctx", "va");            // hub ↔ VA — same league
  own("va", "stt"); own("va", "tts"); own("va", "mode"); flow("tts", "speaker"); flow("queue", "va"); own("ctx", "ctxtree"); own("ctxtree", "ctxlayer");
  own("ctx", "mgr");            // hub owns the registry relay
  flow("localizer", "pp"); flow("localizer", "corr");
  own("ctx", "corr"); own("corr", "dict");
  ["lev", "sp", "ipa", "dictitem", "lookup"].forEach((t) => own("dict", t));
  flow("corr", "ner");
  flow("corr", "search"); flow("ner", "search"); flow("ctxtree", "search");
  flow("search", "mgr"); flow("search", "pp"); flow("search", "ctx");
  ["word", "string", "slots", "union", "nldict", "pattern", "rules", "expanded"].forEach((t) => own("pp", t));
  flow("pattern", "expanded"); flow("expanded", "objparser"); // Pattern → its vocabulary expansion → parser
  own("slots", "vocabdict"); flow("nldict", "vocabdict"); // slots/NL-dict names resolve via the vocab dictionary
  own("pp", "objparser"); own("pp", "matchresult"); own("search", "searchresult");
  flow("objparser", "matchresult"); flow("matchresult", "searchresult"); flow("searchresult", "ctx");
  cmdNames.forEach((c) => own("mgr", "cmd_" + c));                 // commands hang off the relay
  respObjs.forEach((r, i) => { if (r.command) own("cmd_" + r.command, "resp_" + i); flow("resp_" + i, "queue"); });

  // core products — processors/parsers OWN their products (so the core carries the weight),
  // and those products flow into the next stage.
  corrections.forEach((_, i) => { own("corr", "corrn_" + i); flow("corrn_" + i, "pp"); });
  entities.forEach((_, i) => { own("ner", "ent_" + i); flow("ent_" + i, "pp"); });
  params.forEach((_, i) => own("objparser", "param_" + i));
  // infra links are `flow` only, so infra stays low-weight (background satellites)
  flow("dm", "ctx"); flow("health", "mgr"); flow("blockage", "ctx"); flow("mic", "stt"); flow("relay", "va");

  // ── recursive + directional weight ───────────────────────────────────────
  const owns: Record<string, string[]> = {}, peers: Record<string, string[]> = {};
  edges.forEach((e) => {
    if (e.kind === "own") (owns[e.source] ??= []).push(e.target);
    if (e.kind === "peer") { (peers[e.source] ??= []).push(e.target); (peers[e.target] ??= []).push(e.source); }
  });
  const w: Record<string, number> = {}; nodes.forEach((n) => (w[n.id] = 1));
  for (let it = 0; it < 16; it++) {
    const nw: Record<string, number> = {};
    nodes.forEach((n) => {
      let s = 1;
      (owns[n.id] || []).forEach((c) => (s += 0.6 * w[c]));       // inherit weight of owned subtree
      (peers[n.id] || []).forEach((p) => (s += 0.4 * w[p]));      // peers share a league
      nw[n.id] = s;
    });
    Object.assign(w, nw);
  }
  nodes.forEach((n) => (n.weight = Math.round(w[n.id] * 10) / 10));

  // ── depth from the hub (drives the radial rings in the layout) ────────────
  const adj: Record<string, string[]> = {};
  edges.forEach((e) => { (adj[e.source] ??= []).push(e.target); (adj[e.target] ??= []).push(e.source); });
  const depth: Record<string, number> = { ctx: 0 };
  let q = ["ctx"];
  while (q.length) { const u = q.shift()!; (adj[u] || []).forEach((v) => { if (depth[v] === undefined) { depth[v] = depth[u] + 1; q.push(v); } }); }
  nodes.forEach((n) => (n.depth = depth[n.id] ?? 3));

  // ── impulse frames · PULSE PHYSICS ──────────────────────────────────────
  // A frame is ONE hop of the travelling pulse. The pulse can never jump over a
  // node: every edge in a frame starts at a node the pulse already reached (or is
  // freshly BORN at an IO node / injected tool). Parallel branches from the same
  // frontier ride together (split); several edges converging on one node glue back
  // together. Returns are explicit reverse hops (bounce), so a chain A→B→C is at
  // least two forward steps and its result bounces C→B→A. No teleport, no jumps.
  const frames: RFrame[] = [];
  const idxOf = (label: string) => { const i = steps.findIndex((s) => s.label === label); return i >= 0 ? [i] : []; };
  // one hop: edges = [from,to][]; nodes = every endpoint. Dropped if nothing survives the filter.
  const hop = (es: [string, string][], label: string, stepIdxs: number[] = []) => {
    const e = es.filter(([a, b]) => has_(a) && has_(b));
    if (!e.length) return;
    frames.push({ nodes: [...new Set(e.flat())], edges: e, label, stepIdxs });
  };
  const fan = (from: string, tos: string[]): [string, string][] => tos.map((t) => [from, t] as [string, string]);
  const gather = (froms: string[], to: string): [string, string][] => froms.map((f) => [f, to] as [string, string]);

  const corrN = corrections.map((_, i) => "corrn_" + i);
  const entN = entities.map((_, i) => "ent_" + i);
  const paramN = params.map((_, i) => "param_" + i);
  const cmdN = cmdNames.map((c) => "cmd_" + c);
  const respN = respTexts.map((_, i) => "resp_" + i);
  const hasCtxPush = steps.some((s) => s.label === "context push") || respTexts.length !== cmdNames.length;

  // 1 · listen — the pulse is BORN at the microphone (an IO node)
  hop([["mic", "stt"]], "listen · Microphone → STT");
  hop([["stt", "va"]], "transcription · STT → VoiceAssistant", idxOf("process"));
  hop([["va", "ctx"]], "route · VoiceAssistant → hub");
  // 2 · hub consults its context tree (down a layer, then BOUNCE back so the pulse
  //     never teleports from a dead-end leaf to the next real step)
  hop([["ctx", "ctxtree"]], "read context · hub → tree");
  hop([["ctxtree", "ctxlayer"]], "active layer · tree → ContextLayer");
  hop([["ctxlayer", "ctxtree"]], "return · ContextLayer → tree (bounce)");
  hop([["ctxtree", "ctx"]], "return · tree → hub (bounce)");
  // 3 · corrections excursion (down to the dictionary, bounce back, emit objects)
  if (corr) {
    hop([["ctx", "corr"]], "correct · hub → CorrectionsProcessor", idxOf("corrections"));
    if (dict) {
      hop([["corr", "dict"]], "lookup · Corrections → Dictionary", idxOf("dictionary lookup"));
      hop(fan("dict", ["lev", "sp", "ipa", "dictitem"]), "phonetics · Dictionary fans to tools");
      hop([["ipa", "dict"], ["sp", "dict"], ["lev", "dict"], ["dict", "lookup"]], "resolve · tools → Dictionary → LookupResult");
      hop([["dict", "corr"]], "return · Dictionary → Corrections (bounce)");
    }
    if (corrN.length) hop(fan("corr", corrN), "produce · Correction objects");
  }
  // 4 · NER excursion
  if (ner) {
    hop([[corr ? "corr" : "ctx", "ner"]], "entities · → SpacyNER", idxOf("NER"));
    if (entN.length) hop(fan("ner", entN), "produce · RecognizedEntity objects");
  }
  // 5 · enriched input + context GLUE into the search processor (convergence)
  if (search) {
    const srcs = [corr ? "corr" : null, ner ? "ner" : null, "ctxtree", !corr && !ner ? "ctx" : null].filter(Boolean) as string[];
    hop(gather(srcs, "search"), "search input · enriched + context (glue)", idxOf("search"));
    // registry first: ask the manager for the command list, it returns (bounce), THEN parse
    hop([["search", "mgr"]], "registry · SearchProcessor → CommandsManager");
    hop([["mgr", "search"]], "commands list · CommandsManager → SearchProcessor (return)");
    hop([["search", "pp"]], "match · SearchProcessor → PatternParser");
    hop([...fan("pp", ["word", "string", "slots", "union", "nldict", "pattern", "rules"]), ["localizer", "pp"]], "expand · PatternParser → Object types (+localize)");
    hop([["pattern", "expanded"]], "vocabulary · Pattern → ExpandedString");
    hop([["expanded", "objparser"], ["pp", "objparser"]], "parse · ExpandedString → ObjectParser");
    if (paramN.length) hop(fan("objparser", paramN), "extract · parameter values");
    hop([["objparser", "matchresult"]], "match · parameters → MatchResult");
    hop([["matchresult", "searchresult"]], "collect · MatchResult → SearchResult[]");
    hop([["searchresult", "ctx"]], "return · SearchResult[] → hub (bounce)");
  }
  // 6 · dispatch — hub → relay → commands (two hops), commands answer in parallel
  if (cmdN.length) {
    hop([["ctx", "mgr"]], "dispatch · hub → CommandsManager", steps.map((s, i) => (s.label === "dispatch" ? i : -1)).filter((i) => i >= 0));
    hop(fan("mgr", cmdN), "relay · CommandsManager → commands (parallel)");
    // pair every response to its command; the FIRST per command is the immediate wave, the rest
    // are the background command's delayed responses (separate later steps).
    const seen = new Set<string>();
    const mainPairs: [string, string][] = [];
    const bgPairs: { edge: [string, string]; text: string }[] = [];
    respObjs.forEach((r, i) => {
      if (!r.command) return;
      const edge: [string, string] = ["cmd_" + r.command, "resp_" + i];
      if (seen.has(r.command)) bgPairs.push({ edge, text: r.text });
      else { seen.add(r.command); mainPairs.push(edge); }
    });
    if (mainPairs.length) hop(mainPairs, "respond · each command → its Response (parallel)", steps.map((s, i) => (s.label === "respond" ? i : -1)).filter((i) => i >= 0));
    // side-effects of the responding command: push a context layer + flip the VA mode
    if (hasCtxPush) {
      const pusher = respObjs.find((r) => r.command && respCountByCmd[r.command] <= 1)?.command || cmdNames[0];
      hop([["cmd_" + pusher, "ctxlayer"]], "context push · command → ContextLayer");
      hop([["cmd_" + pusher, "mode"]], "mode change · command → VA Mode");
    }
    // 6b · the immediate responses bounce out to speech
    if (mainPairs.length) hop(mainPairs.map(([, r]) => [r, "queue"] as [string, string]), "enqueue · Responses → queue (parallel)");
    hop([["queue", "va"]], "reply · queue → VoiceAssistant (bounce)");
    hop([["va", "tts"]], "speak · VoiceAssistant → TTS");
    hop([["tts", "speaker"]], "play · TTS → Speaker");
    // 6c · background command keeps running — each delayed Response is its OWN later step
    bgPairs.forEach(({ edge, text }, k) => {
      hop([edge], `bg tick ${k + 1} · ${edge[0].replace("cmd_", "")} → “${shorten(text, 16)}” (delayed)`);
      hop([[edge[1], "queue"]], "enqueue · delayed Response → queue");
      hop([["queue", "va"]], "reply · queue → VoiceAssistant");
      hop([["va", "tts"]], "speak · VoiceAssistant → TTS");
      hop([["tts", "speaker"]], "play · TTS → Speaker");
    });
  }

  // mark which nodes the pulse actually visited this request (the rest are muted architecture)
  const touchedIds = new Set<string>(); frames.forEach((f) => f.nodes.forEach((n) => touchedIds.add(n)));
  nodes.forEach((n) => (n.touched = touchedIds.has(n.id)));

  return { nodes, edges, frames };
}

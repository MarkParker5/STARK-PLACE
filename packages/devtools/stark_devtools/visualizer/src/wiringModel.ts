// Faithful port of the design's data-rich wiring (C12R / "12c") — coords, group hulls (G12),
// per-node in/out socket FIELDS, the edge/impulse model, and the HEART_ACTIONS replay. Real trace
// values are injected into the socket fields + data chips at render time.
import type { Step } from "./schema";

export const ACC: Record<string, string> = {
  IO: "#4a90d9", ENGINE: "#c9923f", PROC: "#6cc08a", PHON: "#e07aa6",
  MATCH: "#8f86c9", PROD: "#7fd3c4", CMD: "#8fd6a8", RESP: "#7fd3c4",
};

// [name, pattern, matched(0/1), color]
export const CMDL: [string, string, number, string][] = [
  ["play_music", "play $band:NLBandName", 1, "#8fd6a8"],
  ["lights_off", "turn off the ** lights", 1, "#8ec4ee"],
  ["play_movie", "play a movie", 0, "#586472"],
  ["room_ctl", "$room:(kitchen|bedroom)", 0, "#586472"],
  ["set_timer", "set a timer for $dur", 0, "#586472"],
  ["weather", "weather in $city:NLCity", 0, "#586472"],
  ["hello", "hello $name:Word", 0, "#586472"],
  ["clock", "what time is it", 0, "#586472"],
  ["volume", "volume $level percent", 0, "#586472"],
  ["confirm", "yes | no", 0, "#586472"],
  ["seek", "next | previous", 0, "#586472"],
  ["stop", "stop", 0, "#586472"],
];

export interface WNode {
  id: string; x: number; y: number; w: number; h: number;
  label: string; sub: string; grp: string; main: boolean; kind: string;
}

// C12R base nodes (viewBox ~1900x720)
const BASE: [string, number, number, number, number, string, string, string, number, string][] = [
  ["ctx", 430, 410, 214, 152, "CommandsContext", "hub · sockets", "ENGINE", 1, "plain"],
  ["ctxtree", 430, 584, 214, 74, "Context tree", "root ▸ music", "ENGINE", 1, "tree"],
  ["corr", 676, 150, 232, 116, "Corrections", "", "PROC", 1, "plain"],
  ["ner", 932, 150, 224, 92, "SpacyNER", "", "PROC", 1, "plain"],
  ["shared", 676, 304, 232, 82, "enriched input", "", "PROD", 1, "plain"],
  ["search", 1184, 150, 216, 112, "SearchProcessor", "match · resolve", "PROC", 1, "plain"],
  ["sout", 1184, 304, 224, 92, "search output", "", "PROD", 1, "plain"],
  ["dict", 690, 74, 150, 40, "Dictionary", "«bands»", "PHON", 0, "plain"],
  ["lev", 560, 20, 150, 32, "Levenshtein", "dist 1.0", "PHON", 0, "plain"],
  ["sp", 720, 20, 150, 32, "simplephone", "PARK", "PHON", 0, "plain"],
  ["ipa", 880, 20, 150, 32, "IPA · espeak", "pɜːk", "PHON", 0, "plain"],
  ["mgr", 1184, 20, 150, 32, "CommandsManager", "12", "MATCH", 0, "plain"],
  ["pp", 1184, 60, 150, 32, "PatternParser", "parse", "MATCH", 0, "plain"],
  ["localizer", 1184, 100, 150, 32, "Localizer", "@keys · recognizable", "MATCH", 0, "plain"],
  ["va", 200, 428, 180, 100, "VoiceAssistant", "I/O boundary", "IO", 1, "plain"],
  ["stt", 30, 452, 150, 34, "STT · Vosk", "", "IO", 1, "plain"],
  ["tts", 30, 522, 150, 34, "TTS · Silero", "", "IO", 1, "plain"],
  ["queue", 200, 566, 180, 98, "Response queue", "VA output", "RESP", 1, "queue"],
  ["rp_play", 1656, 300, 204, 44, "“Playing …”", "R0 · +push", "RESP", 0, "plain"],
  ["rp_lights", 1656, 356, 204, 44, "“Lights off.”", "R1 · done", "RESP", 0, "plain"],
];

// cmdCol(1444,150,182,26,5,4): commands column
function cmdCol(): WNode[] {
  const x = 1444, yTop = 150, w = 182, h = 26, gap = 5;
  return CMDL.map(([name, , , ], i) => ({
    id: "cmd_" + name, x, y: yTop + i * (h + gap), w, h, label: name, sub: "", grp: "CMD", main: false, kind: "cmd",
  }));
}

export const NODES: WNode[] = [
  ...BASE.map(([id, x, y, w, h, label, sub, grp, main, kind]) => ({ id, x, y, w, h, label, sub, grp, main: !!main, kind })),
  ...cmdCol(),
];

// group hulls: [name, color, memberIds, pad]  (ENGINE nests PROCESSORS)
export const G12: [string, string, string[], number][] = [
  ["ENGINE", "#c9923f", ["ctx", "ctxtree", "corr", "ner", "shared", "search", "sout"], 28],
  ["PROCESSORS", "#6cc08a", ["corr", "ner", "search"], 13],
  ["VOICE ASSISTANT", "#4a90d9", ["va", "stt", "tts", "queue"], 15],
  ["PHONETIC TOOLS", "#e07aa6", ["dict", "ipa", "sp", "lev"], 13],
  ["MATCHING", "#8f86c9", ["mgr", "pp", "localizer"], 12],
  ["COMMANDS", "#8fd6a8", CMDL.map((c) => "cmd_" + c[0]), 15],
];

// per-node in/out socket rows: [io, key, value]
export type Field = ["in" | "out", string, string];
export const FIELDS: Record<string, Field[]> = {
  ctx: [["in", "heard(txt)", "transcription"], ["out", "process(req)", ""], ["in", "results(m[])", "2"], ["out", "call(cmd,p,ctx)", ""], ["in", "collect(r)", ""], ["out", "reply(r[])", "→ VA"]],
  va: [["in", "audio", "16 kHz mic"], ["out", "stt→text", "“play …”"], ["in", "responses", "×2"], ["out", "tts→speech", ""]],
  corr: [["in", "raw", "“play …”"], ["out", "corrected", "…"], ["out", "uses", "Dictionary"]],
  ner: [["in", "text", "“play …”"], ["out", "entity", "—"]],
  shared: [["out", "enriched", "variants + entities"]],
  search: [["in", "enriched + tree", ""], ["out", "R0", "play_music"], ["out", "R1", "lights_off"]],
  sout: [["out", "R0", "play_music · root"], ["out", "R1", "lights_off · pops"]],
};

// the design's 20 fine transitions (the canonical animation timeline)
export interface HeartAction { e: [string, string][]; t: string; c: string; d: string }
export const HEART_ACTIONS: HeartAction[] = [
  { e: [["va", "ctx"]], t: "VoiceAssistant → CommandsContext", c: "#4a90d9", d: "transcription “…”" },
  { e: [["ctx", "corr"]], t: "CommandsContext → Corrections", c: "#c9923f", d: "raw string + tokens[]" },
  { e: [["corr", "dict"]], t: "Corrections consults its Dictionary", c: "#e07aa6", d: "unknown tokens" },
  { e: [["dict", "lev"], ["dict", "sp"], ["dict", "ipa"]], t: "Dictionary → phonetic tools", c: "#e07aa6", d: "word → ipa → PARK → dist" },
  { e: [["corr", "ner"]], t: "Corrections → SpacyNER", c: "#6cc08a", d: "corrected text" },
  { e: [["corr", "shared"], ["ner", "shared"]], t: "Corrections & NER write enriched input", c: "#7fd3c4", d: "variants + entity spans" },
  // enriched input + context tree feed Search in ONE step
  { e: [["shared", "search"], ["ctxtree", "search"]], t: "enriched input + context tree → SearchProcessor", c: "#6cc08a", d: "TranscriptionString + context vars" },
  { e: [["search", "mgr"], ["search", "pp"], ["localizer", "pp"]], t: "Search uses CommandsManager & PatternParser (+ Localizer)", c: "#8f86c9", d: "compiled patterns · @keys" },
  { e: [["search", "sout"]], t: "SearchProcessor → search output", c: "#8f86c9", d: "SearchResult[2]" },
  { e: [["sout", "ctx"]], t: "results feed back into CommandsContext", c: "#8f86c9", d: "results[] → dispatcher" },
  // CommandsContext dispatches ALL commands in ONE step (parallel)
  { e: [["ctx", "cmd_play_music"], ["ctx", "cmd_lights_off"]], t: "CommandsContext dispatches commands · parallel", c: "#8fd6a8", d: "{band}+ctx · {**}" },
  // commands produce their responses in ONE step (parallel)
  { e: [["cmd_play_music", "rp_play"], ["cmd_lights_off", "rp_lights"]], t: "commands produce responses · parallel", c: "#8fd6a8", d: "R0 · R1" },
  // responses cycle back in ONE step (parallel)
  { e: [["rp_play", "ctx"], ["rp_lights", "ctx"]], t: "responses cycle back to CommandsContext · parallel", c: "#8fd6a8", d: "R0 · R1 → hub" },
  { e: [["ctx", "va"]], t: "CommandsContext forwards responses → VoiceAssistant", c: "#f6b45a", d: "responses[2]" },
  { e: [["va", "queue"]], t: "responses queued for playback", c: "#7fd3c4", d: "queue ← push ×2" },
  { e: [["va", "tts"]], t: "VoiceAssistant → TTS (speak)", c: "#f6b45a", d: "speech audio ×2" },
];

export const coordOf = new Map(NODES.map((n) => [n.id, { x: n.x, y: n.y, w: n.w, h: n.h, cx: n.x + n.w / 2, cy: n.y + n.h / 2 }]));

// the design's edgePath: leaves the box edge, bezier routed horizontally or vertically
export function edgePath(a: string, b: string): string {
  const A = coordOf.get(a)!, B = coordOf.get(b)!;
  const dx = B.cx - A.cx, dy = B.cy - A.cy;
  let sx, sy, ex, ey, c1x, c1y, c2x, c2y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    sx = dx > 0 ? A.x + A.w : A.x; sy = A.cy; ex = dx > 0 ? B.x : B.x + B.w; ey = B.cy;
    const m = Math.abs(ex - sx) * 0.5; c1x = sx + (dx > 0 ? m : -m); c1y = sy; c2x = ex - (dx > 0 ? m : -m); c2y = ey;
  } else {
    sx = A.cx; sy = dy > 0 ? A.y + A.h : A.y; ex = B.cx; ey = dy > 0 ? B.y : B.y + B.h;
    const m = Math.abs(ey - sy) * 0.5; c1x = sx; c1y = sy + (dy > 0 ? m : -m); c2x = ex; c2y = ey - (dy > 0 ? m : -m);
  }
  return `M ${sx} ${sy} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${ex} ${ey}`;
}

// first action index that touches each node -> drives idle/done styling
export const heartReach: Record<string, number> = (() => {
  const r: Record<string, number> = {};
  HEART_ACTIONS.forEach((a, i) => a.e.forEach(([x, y]) => { if (!(x in r)) r[x] = i; if (!(y in r)) r[y] = i; }));
  return r;
})();

// Inject real trace values into a copy of FIELDS + heart data labels.
export function realFields(steps: Step[]): { fields: Record<string, Field[]>; heart: HeartAction[]; matched: string[] } {
  const fields: Record<string, Field[]> = JSON.parse(JSON.stringify(FIELDS));
  const heart = HEART_ACTIONS.map((a) => ({ ...a, e: a.e.map((e) => [...e] as [string, string]) }));

  const get = (label: string) => steps.find((s) => s.label === label);
  const process = get("process");
  const utter = process?.input?.string as string | undefined;
  const corr = get("corrections");
  const results = (get("search")?.output?.results as any[]) || (process?.output?.results as any[]) || [];
  const cmds = results.map((r) => String(r.command).split(".").pop());

  if (utter) {
    fields.ctx[0][2] = `“${utter}”`;
    fields.va[1][2] = `“${utter}”`;
    fields.corr[0][2] = `“${utter}”`;
    fields.ner[0][2] = `“${utter}”`;
    heart[0].d = `transcription “${utter}”`;
  }
  const corrections = (corr?.output?.results as any) || corr?.input || {};
  // corrections come through as MatchResult corrections on results; pull from results
  const corrList: string[] = [];
  results.forEach((r) => (r.match?.corrections || []).forEach((c: string) => corrList.push(c)));
  if (corrList.length) fields.corr[1][2] = corrList.join(" · ");
  if (cmds.length) {
    fields.search[1][2] = cmds[0] || "—";
    fields.search[2][2] = cmds[1] || "—";
    fields.sout[0][2] = `${cmds[0]} · root`;
    fields.sout[1][2] = `${cmds[1] || "—"}`;
  }
  return { fields, heart, matched: cmds.filter(Boolean) as string[] };
}

// ── DYNAMIC wiring: rebuild the command + response nodes and the dispatch/response timeline from
// the ACTUAL request, so the Wiring view reflects each trace instead of the fixed C12R demo. The
// architecture nodes (VA, hub, processors, search, tools) stay put; only the request-specific
// commands/responses and the steps touching them are generated.
export interface WiringModel {
  nodes: WNode[];
  groups: [string, string, string[], number][];
  heart: HeartAction[];
  fields: Record<string, Field[]>;
  matched: string[];
  coordOf: Map<string, { x: number; y: number; w: number; h: number; cx: number; cy: number }>;
  heartReach: Record<string, number>;
}

const _short = (t: string, n = 18) => (t && t.length > n ? t.slice(0, n - 1) + "…" : t || "");

export function buildWiring(steps: Step[]): WiringModel {
  const get = (l: string) => steps.find((s) => s.label === l);
  const process = get("process");
  const utter = process?.input?.string as string | undefined;
  const rawResults = get("search")?.output?.results ?? process?.output?.results;
  const results: any[] = (Array.isArray(rawResults) ? rawResults : []).filter(Boolean);
  const cmds = [...new Set(results.map((r) => String(r.command).split(".").pop() as string))];
  const paramsByCmd: Record<string, string> = {};
  results.forEach((r) => { const c = String(r.command).split(".").pop() as string; const p = r.match?.parameters || {}; paramsByCmd[c] = Object.entries(p).map(([k, v]) => `${k}=${v}`).join(" "); });
  const corrList: string[] = [];
  results.forEach((r) => (r.match?.corrections || []).forEach((c: string) => corrList.push(c)));
  const respByCmd: Record<string, string[]> = {};
  steps.filter((s) => s.label === "respond").forEach((s) => { const c = String(s.input?.command || "").split(".").pop(); const t = (s.input?.response as any)?.text; if (c) (respByCmd[c] ??= []).push(t); });

  const baseNodes: WNode[] = BASE
    .filter((b) => !["rp_play", "rp_lights"].includes(b[0] as string))
    .map(([id, x, y, w, h, label, sub, grp, main, kind]) => ({ id: id as string, x: x as number, y: y as number, w: w as number, h: h as number, label: label as string, sub: sub as string, grp: grp as string, main: !!main, kind: kind as string }));
  const cmdNodes: WNode[] = cmds.map((c, i) => ({ id: "cmd_" + c, x: 1444, y: 150 + i * 31, w: 182, h: 26, label: c, sub: paramsByCmd[c] || "", grp: "CMD", main: false, kind: "cmd" }));
  const respNodes: WNode[] = cmds.map((c, i) => ({ id: "rp_" + c, x: 1656, y: 300 + i * 58, w: 204, h: 44, label: `“${_short(respByCmd[c]?.[0] || "—", 20)}”`, sub: (respByCmd[c]?.length || 0) > 1 ? `R${i} · ×${respByCmd[c].length}` : "R" + i, grp: "RESP", main: false, kind: "plain" }));
  const nodes = [...baseNodes, ...cmdNodes, ...respNodes];
  const coordOf = new Map(nodes.map((n) => [n.id, { x: n.x, y: n.y, w: n.w, h: n.h, cx: n.x + n.w / 2, cy: n.y + n.h / 2 }]));

  const groups = G12.map((g) => (g[0] === "COMMANDS" ? ([g[0], g[1], cmds.map((c) => "cmd_" + c), g[3]] as [string, string, string[], number]) : g));

  const fields: Record<string, Field[]> = JSON.parse(JSON.stringify(FIELDS));
  if (utter) { fields.ctx[0][2] = `“${_short(utter, 26)}”`; fields.va[1][2] = `“${_short(utter, 20)}”`; fields.corr[0][2] = `“${_short(utter, 20)}”`; fields.ner[0][2] = `“${_short(utter, 20)}”`; }
  if (corrList.length) fields.corr[1][2] = corrList.join(" · ");
  const _ents = get("NER")?.output?.entities;
  const ents = (Array.isArray(_ents) ? _ents : []).filter((x) => typeof x === "string");
  if (ents.length) fields.ner[1][2] = ents.join(" · ");
  if (cmds.length) { fields.search[1][2] = cmds[0]; fields.search[2][2] = cmds[1] || "—"; fields.sout[0][2] = `${cmds[0]} · root`; fields.sout[1][2] = cmds[1] || "—"; }

  // fixed architecture prefix (va→ctx … sout→ctx) + dynamic dispatch/produce/cycle + tail (ctx→va, va→queue, va→tts)
  const clone = (a: HeartAction) => ({ ...a, e: a.e.map((e) => [...e] as [string, string]) });
  const prefix = HEART_ACTIONS.slice(0, 10).map(clone);
  if (utter) prefix[0].d = `transcription “${_short(utter, 18)}”`;
  const dispatch: HeartAction = { e: cmds.map((c) => ["ctx", "cmd_" + c] as [string, string]), t: "CommandsContext dispatches commands · parallel", c: "#8fd6a8", d: cmds.join(" · ") || "—" };
  const produce: HeartAction = { e: cmds.map((c) => ["cmd_" + c, "rp_" + c] as [string, string]), t: "commands produce responses · parallel", c: "#8fd6a8", d: cmds.map((c) => _short(respByCmd[c]?.[0] || "", 10)).filter(Boolean).join(" · ") || "responses" };
  const cycle: HeartAction = { e: cmds.map((c) => ["rp_" + c, "ctx"] as [string, string]), t: "responses cycle back to CommandsContext · parallel", c: "#8fd6a8", d: "→ hub" };
  const tail = HEART_ACTIONS.slice(13).map(clone);
  const heart = cmds.length ? [...prefix, dispatch, produce, cycle, ...tail] : [...prefix, ...tail];

  const heartReach: Record<string, number> = {};
  heart.forEach((a, i) => a.e.forEach(([x, y]) => { if (!(x in heartReach)) heartReach[x] = i; if (!(y in heartReach)) heartReach[y] = i; }));

  return { nodes, groups, heart, fields, matched: cmds, coordOf, heartReach };
}

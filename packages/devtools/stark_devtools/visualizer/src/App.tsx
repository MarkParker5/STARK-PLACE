import { useEffect, useMemo, useRef, useState } from "react";
import { BrainGraph } from "./components/BrainGraph";
import { NodeInspector, PayloadInspector } from "./components/Inspector";
import { Transport } from "./components/Transport";
import { Durations } from "./components/Durations";
import { Dashboard } from "./views/Dashboard";
import { Matching } from "./views/Matching";
import { Wiring } from "./views/Wiring";
import { buildWiring } from "./wiringModel";
import { edgeKey } from "./derive";
import { buildRuntime, kindColor } from "./runtime";
import { fetchDemo, fetchSample, submitUtterance, type GraphNode, type Step, type TraceBundle } from "./schema";
import { useReplay } from "./store/useReplay";
import { tokens } from "./tokens";

type View = "dashboard" | "matching" | "wiring" | "brain";
const BASE_MS = 1100;
const SETTINGS_KEY = "stark-edit-settings-v3"; // v3 — new spread/spacing defaults + trace mode

type TraceMode = "keep" | "chain";
interface EditSettings { spread: number; spacing: number; mutedOpacity: number; unmuteAll: boolean; speed: number; traceMode: TraceMode; groupOpacity: number; fadeRate: number; sizeContrast: number }
const DEFAULT_SETTINGS: EditSettings = { spread: 2, spacing: 0.6, mutedOpacity: 0.28, unmuteAll: false, speed: 1, traceMode: "keep", groupOpacity: 1, fadeRate: 0.16, sizeContrast: 50 };
function loadSettings(): EditSettings { try { return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; } catch { return DEFAULT_SETTINGS; } }

const KINDS = [
  ["hub", "CommandsContext"], ["processor", "processors"], ["matching", "manager · parser"],
  ["parse", "parse types"], ["data", "products (corrections · entities · params)"], ["tool", "dictionary · phonetics"],
  ["command", "commands"], ["response", "responses"], ["io", "VA · STT · TTS · queue"],
  ["state", "context tree"], ["infra", "infra (DI · watchdogs)"],
] as const;

export default function App() {
  const [bundle, setBundle] = useState<TraceBundle | null>(null);
  const [cases, setCases] = useState<{ id: string; label: string; text: string }[]>([]);
  const [view, setView] = useState<View>("dashboard");
  const [pickedNode, setPickedNode] = useState<GraphNode | null>(null);
  const [pickedStep, setPickedStep] = useState<Step | null>(null);
  const [pickedInfo, setPickedInfo] = useState<{ title: string; input: any } | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [edit, setEdit] = useState(false);
  const [blacklist, setBlacklist] = useState<Set<string>>(new Set());
  const [showDurations, setShowDurations] = useState(false);
  const [playlistUI, setPlaylistUI] = useState(false);

  const initial = useMemo(loadSettings, []);
  const [spread, setSpread] = useState(initial.spread);
  const [spacing, setSpacing] = useState(initial.spacing);
  const [mutedOpacity, setMutedOpacity] = useState(initial.mutedOpacity);
  const [unmuteAll, setUnmuteAll] = useState(initial.unmuteAll);
  const [traceMode, setTraceMode] = useState<TraceMode>(initial.traceMode);
  const [groupOpacity, setGroupOpacity] = useState(initial.groupOpacity);
  const [fadeRate, setFadeRate] = useState(initial.fadeRate);
  const [sizeContrast, setSizeContrast] = useState(initial.sizeContrast);
  const [layoutReset, setLayoutReset] = useState(0); // bump → BrainGraph clears saved node positions

  useEffect(() => {
    fetchDemo().then(setBundle).catch(() => fetchSample().then(setBundle).catch(() => {}));
    fetch("/api/cases").then((r) => r.json()).then((d) => setCases(d.cases || [])).catch(() => {});
  }, []);

  const steps = bundle?.steps ?? [];
  const runtime = useMemo(() => buildRuntime(steps), [steps]);
  const realFrames = useMemo(() => runtime.frames.filter((f) => !f.nodes.every((n) => blacklist.has(n))), [runtime, blacklist]);
  // brain gets an empty step 0 (the graph sits idle/faded); playback rewinds here when it ends
  const frames = useMemo(() => [{ nodes: [], edges: [], label: "idle", stepIdxs: [] }, ...realFrames], [realFrames]);

  const wiring = useMemo(() => buildWiring(steps), [steps]);
  const timelineLen = view === "wiring" ? wiring.heart.length : view === "brain" ? frames.length : steps.length;
  const playlistOn = useRef(false);
  const playlistIdx = useRef(0);
  const replay = useReplay(Math.max(1, timelineLen), initial.speed, () => advancePlaylist());
  const speedMs = BASE_MS / replay.speed;

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ spread, spacing, mutedOpacity, unmuteAll, speed: replay.speed, traceMode, groupOpacity, fadeRate, sizeContrast }));
  }, [spread, spacing, mutedOpacity, unmuteAll, replay.speed, traceMode, groupOpacity, fadeRate, sizeContrast]);

  const idx = Math.min(replay.index, Math.max(0, timelineLen - 1));

  // brain: runtime impulse frames
  const frame = view === "brain" ? frames[idx] : undefined;
  const brainStep = frame && frame.stepIdxs.length ? steps[frame.stepIdxs[0]] : undefined;

  // ── trace stack: for each edge/node, "how many steps ago it last pulsed" (age) ──
  // Current frame pulses (age 0); ancestor links stay highlighted and fade with age.
  // traceMode: "keep" accumulates the whole replay; "chain" resets when the active
  // set jumps to a disjoint region (a genuinely different chain).
  const trace = useMemo(() => {
    const edgeAge = new Map<string, number>();
    const nodeAge = new Map<string, number>();
    if (view !== "brain" || replay.overview || !frames.length) return { edgeAge, nodeAge };
    let start = 0;
    if (traceMode === "chain") {
      start = idx;
      while (start > 0) {
        const prev = new Set(frames[start - 1].nodes);
        if (frames[start].nodes.some((n) => prev.has(n))) start--; else break;
      }
    }
    for (let s = start; s <= idx; s++) { // increasing s ⇒ shrinking age ⇒ most-recent wins
      const age = idx - s;
      frames[s].nodes.forEach((n) => nodeAge.set(n, age));
      frames[s].edges.forEach(([a, b]) => edgeAge.set(edgeKey(a, b), age));
    }
    return { edgeAge, nodeAge };
  }, [view, replay.overview, frames, idx, traceMode]);

  // the current frame's edges WITH direction — the pulse travels along the hop, not the static edge
  const activeHopEdges = (!replay.overview && view === "brain" && frames[idx] ? frames[idx].edges : []) as [string, string][];

  const dashStep = steps[Math.min(replay.index, Math.max(0, steps.length - 1))];
  const wiringAction = view === "wiring" && !replay.overview ? wiring.heart[Math.min(idx, wiring.heart.length - 1)] : null;

  const pendingPlay = useRef(false);
  const lastRun = useRef<string>("");
  const prefetch = useRef<Map<string, TraceBundle>>(new Map()); // playlist look-ahead cache
  async function loadBundle(q: string): Promise<TraceBundle> {
    const hit = prefetch.current.get(q);
    if (hit) { prefetch.current.delete(q); return hit; }
    return submitUtterance(q);
  }
  async function run(t?: string) {
    const q = (t ?? text).trim(); if (!q) return;
    setBusy(true); lastRun.current = q;
    try { setBundle(await loadBundle(q)); setPickedNode(null); setPickedStep(null); setPickedInfo(null); pendingPlay.current = true; }
    catch { /* offline */ } finally { setBusy(false); }
  }
  // live-animate a freshly-run request: once its trace is in, auto-play from the top
  useEffect(() => {
    if (!pendingPlay.current) return;
    pendingPlay.current = false;
    const id = setTimeout(() => replay.play(), 20); // just enough for the length-reset to settle
    return () => clearTimeout(id);
  }, [bundle]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── demo playlist: auto-run a list of requests, cycling at the list level ──
  const playlistTexts = () => (cases.length ? cases.map((c) => c.text) : [bundle?.utterance || ""].filter(Boolean));
  // fetch the request AFTER the current one in the background so transitions have no network gap
  function prefetchNext() {
    const list = playlistTexts();
    if (list.length < 2) return;
    const next = list[(playlistIdx.current + 1) % list.length];
    if (!prefetch.current.has(next)) submitUtterance(next).then((b) => prefetch.current.set(next, b)).catch(() => {});
  }
  function advancePlaylist(): boolean {
    if (!playlistOn.current) return false;
    const list = playlistTexts();
    if (!list.length) return false;
    playlistIdx.current = (playlistIdx.current + 1) % list.length; // cycle back to the top
    run(list[playlistIdx.current]);
    prefetchNext();
    return true; // tell the replay we're continuing — no fade-to-idle / pause between requests
  }
  function togglePlaylist() {
    if (playlistOn.current) { playlistOn.current = false; setPlaylistUI(false); return; }
    const list = playlistTexts();
    if (!list.length) return;
    playlistOn.current = true; setPlaylistUI(true);
    replay.setLoop(false); // a playlist relies on onEnd firing, so per-request loop must be off
    playlistIdx.current = 0;
    run(list[0]);
    prefetchNext();
  }

  if (!bundle) return <div style={{ padding: 40, color: tokens.text.mid, fontFamily: tokens.font.mono, fontSize: 15 }}>loading…</div>;

  const blacklistNode = (id: string) => { if (!id) return; setBlacklist((s) => new Set(s).add(id)); setPickedNode(null); setPickedInfo(null); };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: tokens.bg.page, color: tokens.text.hi }}>
      {/* header */}
      <header style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 18px", borderBottom: `1px solid ${tokens.border.faint}` }}>
        <span style={{ fontFamily: tokens.font.display, fontWeight: 600, fontSize: 16, letterSpacing: ".12em", color: tokens.impulse }}>◉ S.T.A.R.K</span>
        <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".2em" }}>VISUALIZER</span>
        <div style={{ flex: 1 }} />
        <select onChange={(e) => e.target.value && run(e.target.value)} value="" style={{ ...hbtn(false), fontSize: 14 }} title="preset cases">
          <option value="" style={{ background: tokens.bg.card }}>cases…</option>
          {cases.map((c) => <option key={c.id} value={c.text} style={{ background: tokens.bg.card }}>{c.label}</option>)}
        </select>
        <input value={text} placeholder={bundle.utterance} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()}
          style={{ flex: 1, maxWidth: 420, background: tokens.bg.card, border: `1px solid ${tokens.border.mid}`, borderRadius: 8, padding: "8px 12px", color: tokens.text.hi, fontFamily: tokens.font.mono, fontSize: 15 }} />
        <button onClick={() => run()} disabled={busy} style={hbtn(busy)}>{busy ? "running…" : "▶ run"}</button>
        <button onClick={() => run(lastRun.current || bundle.utterance)} disabled={busy} style={hbtn(false)} title="re-run the same request and replay it">↻</button>
        <button onClick={togglePlaylist} style={hbtn(playlistUI)} title="auto-play through every preset case, cycling (demo mode)">{playlistUI ? "▮▮ playlist" : "▶▶ playlist"}</button>
        <button onClick={() => setEdit((e) => !e)} style={hbtn(edit)}>{edit ? "✓ editing" : "✎ edit"}</button>
      </header>

      {/* tabs */}
      <div style={{ display: "flex", gap: 6, padding: "10px 18px 0" }}>
        {(["dashboard", "matching", "wiring", "brain"] as View[]).map((v) => (
          <button key={v} onClick={() => setView(v)} style={{ ...hbtn(view === v), textTransform: "capitalize" }}>{v === "brain" ? "brain (runtime)" : v}</button>
        ))}
        <div style={{ flex: 1 }} />
        <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, alignSelf: "center" }}>
          {replay.overview ? "◉ overview · all touched" : `▶ playback · step ${idx + 1}`}
        </span>
      </div>

      {/* body */}
      <div style={{ flex: 1, display: "flex", gap: 12, padding: 14, minHeight: 0 }}>
        <main style={{ flex: 1, minWidth: 0, background: tokens.bg.card, border: `1px solid ${tokens.border.faint}`, borderRadius: 12, overflow: view === "dashboard" ? "auto" : "hidden", position: "relative" }}>
          {view === "dashboard" && (
            <div style={{ padding: 14 }}><Dashboard steps={steps} index={Math.min(replay.index, Math.max(0, steps.length - 1))} overview={replay.overview} onPick={setPickedStep} /></div>
          )}
          {view === "matching" && (
            <Matching steps={steps} utterance={bundle.utterance} onPick={(title, input) => { setPickedInfo({ title, input }); setPickedNode(null); setPickedStep(null); }} />
          )}
          {view === "wiring" && (
            <Wiring steps={steps} index={idx} overview={replay.overview} playing={replay.playing} edit={edit} speedMs={speedMs} mutedOpacity={mutedOpacity} unmuteAll={unmuteAll} groupOpacity={groupOpacity} model={wiring}
              onPickNode={(id, label) => { setPickedInfo({ title: label, input: { module: id } }); setPickedNode(null); setPickedStep(null); }} />
          )}
          {view === "brain" && (
            <BrainGraph runtime={runtime} overview={replay.overview} playing={replay.playing}
              edgeAge={trace.edgeAge} nodeAge={trace.nodeAge} activeEdges={activeHopEdges} unmuteAll={unmuteAll} mutedOpacity={mutedOpacity} activeLabel={frame?.label ?? ""} stepKey={idx} resetKey={layoutReset} fadeRate={fadeRate} sizeContrast={sizeContrast}
              spread={spread} spacing={spacing} speedMs={speedMs} blacklist={blacklist} onBlacklist={blacklistNode}
              onPick={(id, label) => { if (id) { setPickedInfo({ title: label, input: { node: id } }); setPickedNode(null); setPickedStep(null); } else { setPickedInfo(null); } }} picked={pickedInfo ? null : null} />
          )}

          {edit && (view === "wiring" || view === "brain") && (
            <div style={{ position: "absolute", top: 12, left: 12, display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap", maxWidth: "74%", background: tokens.bg.card2, border: `1px solid ${tokens.border.strong}`, borderRadius: 10, padding: "8px 12px", boxShadow: "6px 6px 0 rgba(7,10,13,.45)" }}>
              {view === "wiring" ? (
                <>
                  <Ctl label="MUTED" val={unmuteAll ? "off" : mutedOpacity.toFixed(2)}><input type="range" min={0.05} max={0.9} step={0.05} value={mutedOpacity} onChange={(e) => setMutedOpacity(Number(e.target.value))} style={rng} disabled={unmuteAll} /></Ctl>
                  <button onClick={() => setUnmuteAll((u) => !u)} style={hbtn(unmuteAll)}>unmute all</button>
                  <Ctl label="GROUPS" val={groupOpacity.toFixed(2)}><input type="range" min={0} max={1} step={0.05} value={groupOpacity} onChange={(e) => setGroupOpacity(Number(e.target.value))} style={rng} /></Ctl>
                  <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted }}>scroll=zoom · drag=pan · node=move · saved</span>
                </>
              ) : (
                <>
                  <Ctl label="SPREAD" val={`${spread.toFixed(1)}×`}><input type="range" min={0.5} max={3} step={0.1} value={spread} onChange={(e) => setSpread(Number(e.target.value))} style={rng} /></Ctl>
                  <Ctl label="SPACING" val={`${spacing.toFixed(1)}×`}><input type="range" min={0.1} max={3} step={0.1} value={spacing} onChange={(e) => setSpacing(Number(e.target.value))} style={rng} /></Ctl>
                  <Ctl label="FADE" val={fadeRate === 0 ? "off" : fadeRate.toFixed(2)}><input type="range" min={0} max={0.5} step={0.02} value={fadeRate} onChange={(e) => setFadeRate(Number(e.target.value))} style={rng} /></Ctl>
                  <Ctl label="SIZE Δ" val={`${Math.round(sizeContrast)}`}><input type="range" min={0} max={90} step={2} value={sizeContrast} onChange={(e) => setSizeContrast(Number(e.target.value))} style={rng} /></Ctl>
                  <Ctl label="MUTED" val={mutedOpacity.toFixed(2)}><input type="range" min={0.02} max={0.6} step={0.02} value={mutedOpacity} onChange={(e) => setMutedOpacity(Number(e.target.value))} style={rng} /></Ctl>
                  <button onClick={() => setTraceMode((m) => (m === "keep" ? "chain" : "keep"))} style={hbtn(traceMode === "keep")} title="A/B: keep the whole trace faded, or drop it when a new chain starts">
                    trace: {traceMode === "keep" ? "keep all" : "per-chain"}
                  </button>
                  <button onClick={() => setLayoutReset((n) => n + 1)} style={hbtn(false)} title="clear saved dragged positions">reset layout</button>
                  <button onClick={() => setUnmuteAll((u) => !u)} style={hbtn(unmuteAll)} title="show the full architecture (untouched nodes un-muted) in overview">{unmuteAll ? "✓ full arch" : "unmute all"}</button>
                  <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted }}>scroll=zoom · drag=pan · node=drag (position saved) · shift-drag=group · dbl-click=blacklist</span>
                </>
              )}
            </div>
          )}
        </main>

        {/* right rail */}
        <aside style={{ width: 340, display: "flex", flexDirection: "column", gap: 12, minHeight: 0 }}>
          <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
            {pickedNode ? (
              <NodeInspector node={pickedNode} onClose={() => setPickedNode(null)} onBlacklist={blacklistNode} />
            ) : pickedInfo ? (
              <PayloadInspector title={pickedInfo.title} input={pickedInfo.input} output={{}} onClose={() => setPickedInfo(null)} />
            ) : wiringAction ? (
              <PayloadInspector title={`${idx + 1}. ${wiringAction.t}`} input={{ data: wiringAction.d, edges: wiringAction.e.map((e) => e.join("→")).join(", ") }} output={{}} onClose={() => {}} />
            ) : pickedStep ? (
              <PayloadInspector title={pickedStep.label} input={pickedStep.input} output={pickedStep.output} onClose={() => setPickedStep(null)} />
            ) : view === "brain" && brainStep && !replay.overview ? (
              <PayloadInspector title={`${idx + 1}. ${frame?.label}`} input={brainStep.input} output={brainStep.output} onClose={() => {}} />
            ) : dashStep && !replay.overview && view === "dashboard" ? (
              <PayloadInspector title={dashStep.label} input={dashStep.input} output={dashStep.output} onClose={() => {}} />
            ) : (
              <div style={{ background: tokens.bg.card2, border: `1px solid ${tokens.border.soft}`, borderRadius: 12, padding: 14, fontFamily: tokens.font.mono, fontSize: 14, color: tokens.text.mid, lineHeight: 1.7 }}>
                <div style={{ color: tokens.impulse, marginBottom: 6 }}>◉ overview</div>
                {view === "brain" ? "Runtime object graph — actors & impulses of this request." : "Highlighting every module & link this request touched."}<br />
                <span style={{ color: tokens.text.muted }}>{view === "brain" ? `${runtime.nodes.length} objects · ${runtime.edges.length} impulses · ${realFrames.length} steps` : `${steps.length} steps`}</span>
                <br />Press ▶ or step to replay.
              </div>
            )}
          </div>

          {view === "brain" && (
            <div style={{ background: tokens.bg.card2, border: `1px solid ${tokens.border.soft}`, borderRadius: 12, padding: 12 }}>
              {blacklist.size > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: "#e07a7a", letterSpacing: ".1em", marginBottom: 6 }}>BLACKLISTED (dbl-click node)</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {[...blacklist].map((b) => <button key={b} onClick={() => setBlacklist((s) => { const n = new Set(s); n.delete(b); return n; })} style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.mid, border: `1px solid ${tokens.border.mid}`, borderRadius: 999, padding: "2px 9px", background: "none", cursor: "pointer" }}>{b} ✕</button>)}
                  </div>
                </div>
              )}
              <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".12em", marginBottom: 8 }}>RUNTIME KINDS</div>
              {KINDS.map(([k, label]) => (
                <div key={k} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ width: 13, height: 13, borderRadius: 3, background: kindColor(k) }} />
                  <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.mid }}>{label}</span>
                </div>
              ))}
            </div>
          )}
        </aside>
      </div>

      <footer style={{ padding: "12px 18px", borderTop: `1px solid ${tokens.border.faint}` }}>
        <Transport replay={replay} onLabelClick={() => setShowDurations(true)} label={replay.overview ? `overview · ${timelineLen} steps` : `${idx + 1}/${timelineLen}${view === "wiring" && wiringAction ? " · " + wiringAction.t : view === "brain" && frame ? " · " + frame.label : dashStep ? " · " + dashStep.label : ""}`} />
      </footer>

      {showDurations && <Durations events={bundle.events} onClose={() => setShowDurations(false)} />}
    </div>
  );
}

function Ctl({ label, val, children }: { label: string; val: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".08em" }}>{label}</span>
      {children}
      <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.impulse, width: 34 }}>{val}</span>
    </div>
  );
}
const rng: React.CSSProperties = { width: 100, accentColor: tokens.impulse };
function hbtn(active: boolean): React.CSSProperties {
  return { background: active ? "rgba(246,180,90,.12)" : tokens.bg.card2, border: `1px solid ${active ? tokens.impulse : tokens.border.mid}`, color: active ? tokens.impulse : tokens.text.mid, borderRadius: 8, padding: "7px 13px", fontFamily: tokens.font.mono, fontSize: 14, cursor: "pointer" };
}

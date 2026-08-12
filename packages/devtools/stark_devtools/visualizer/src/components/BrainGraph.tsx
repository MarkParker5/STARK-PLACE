import { forceCenter, forceCollide, forceLink, forceManyBody, forceRadial, forceSimulation, type Simulation } from "d3-force";
import { useEffect, useReducer, useRef, useState } from "react";
import { kindColor, type Runtime } from "../runtime";
import { tokens } from "../tokens";

const VB = { w: 1400, h: 760 };

interface SimNode { id: string; label: string; kind: string; weight: number; depth: number; touched: boolean; x: number; y: number; vx?: number; vy?: number; fx?: number | null; fy?: number | null }

interface Props {
  runtime: Runtime;
  overview: boolean;
  playing: boolean;
  edgeAge: Map<string, number>; // undirected edge key -> steps since it last pulsed (0 = current)
  nodeAge: Map<string, number>; // node id -> steps since last touched (0 = current)
  activeEdges: [string, string][]; // current frame's DIRECTED hops — pulse travels from→to
  unmuteAll: boolean; // in overview, show untouched architecture at full instead of muted
  mutedOpacity: number; // how dim untouched / not-yet-reached nodes & links are
  activeLabel: string;
  stepKey: number; // frame index — restarts the one-shot pulse each step
  resetKey: number; // bump → clear saved dragged positions
  fadeRate: number; // how aggressively ancestor links dim per step of age (0 = no fade)
  sizeContrast: number; // radius range max-min; 0 = all nodes ~equal size
  spread: number;
  spacing: number;
  speedMs: number;
  blacklist: Set<string>;
  onBlacklist: (id: string) => void;
  onPick: (id: string, label: string) => void;
  picked: string | null;
}

// undirected key, matches App's edgeKey
const ekey = (a: string, b: string) => (a < b ? `${a}|${b}` : `${b}|${a}`);
// trace fade by age: age 0 = full, older = dimmer but never invisible; rate = aggressiveness
const fade = (age: number, rate: number) => Math.max(0.14, 1 - age * rate);

// manually dragged node positions persist across reloads (pinned where you left them)
const PIN_KEY = "stark-brain-pins-v1";
function loadPins(): Record<string, { x: number; y: number }> { try { return JSON.parse(localStorage.getItem(PIN_KEY) || "{}"); } catch { return {}; } }

export function BrainGraph({ runtime, overview, playing, edgeAge, nodeAge, activeEdges, unmuteAll, mutedOpacity, activeLabel, stepKey, resetKey, fadeRate, sizeContrast, spread, spacing, speedMs, blacklist, onBlacklist, onPick, picked }: Props) {
  const pinsRef = useRef<Record<string, { x: number; y: number }>>(loadPins());
  const nodesRef = useRef<SimNode[]>([]);
  const simRef = useRef<Simulation<SimNode, any> | null>(null);
  const [, tick] = useReducer((x) => x + 1, 0);
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const svgRef = useRef<SVGSVGElement>(null);

  const maxW = Math.max(1, ...runtime.nodes.map((n) => n.weight));
  // steep-ish curve ⇒ big central actors, tiny leaves; sizeContrast = radius spread (0 ⇒ uniform)
  const rMin = 9;
  const radius = (n: { weight: number; kind: string }) => rMin + Math.pow(n.weight / maxW, 0.9) * sizeContrast;

  // (re)build the live simulation when the graph or the spacing/spread change
  useEffect(() => {
    const prev = new Map(nodesRef.current.map((n) => [n.id, n]));
    const nodes: SimNode[] = runtime.nodes
      .filter((n) => !blacklist.has(n.id))
      .map((n) => {
        const pin = pinsRef.current[n.id];
        const p = prev.get(n.id);
        const x = pin?.x ?? p?.x ?? VB.w / 2 + (Math.random() - 0.5) * 200;
        const y = pin?.y ?? p?.y ?? VB.h / 2 + (Math.random() - 0.5) * 200;
        // pinned nodes stay exactly where the user dropped them
        return { ...n, x, y, fx: pin ? pin.x : (p?.fx ?? null), fy: pin ? pin.y : (p?.fy ?? null) };
      });
    const ids = new Set(nodes.map((n) => n.id));
    const links = runtime.edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({ ...e }));
    nodesRef.current = nodes;

    const ring = 118 * spread;
    const sim = forceSimulation<SimNode>(nodes)
      // strong repulsion + big collision radius = "anti-gravity against overlaps"
      .force("charge", forceManyBody<SimNode>().strength((d) => (-260 - radius(d) * 16) * spacing))
      .force("link", forceLink<SimNode, any>(links).id((d: any) => d.id).distance(ring * 0.85).strength(0.2))
      // concentric RINGS by graph depth from the hub -> hierarchical, far fewer edge crossings
      .force("radial", forceRadial<SimNode>((d) => d.depth * ring, VB.w / 2, VB.h / 2).strength(0.7))
      .force("collide", forceCollide<SimNode>((d) => radius(d) + 16 * spacing).strength(0.95))
      .force("center", forceCenter(VB.w / 2, VB.h / 2).strength(0.01));
    sim.on("tick", tick);
    sim.alpha(0.9).restart();
    simRef.current = sim;
    return () => { sim.stop(); };
  }, [runtime, spread, spacing, blacklist]);

  // "reset layout" → drop all saved positions and let the sim re-settle
  const firstReset = useRef(true);
  useEffect(() => {
    if (firstReset.current) { firstReset.current = false; return; }
    pinsRef.current = {};
    localStorage.removeItem(PIN_KEY);
    nodesRef.current.forEach((m) => { m.fx = null; m.fy = null; });
    simRef.current?.alpha(0.9).restart();
  }, [resetKey]);

  const nodes = nodesRef.current;
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const gRef = useRef<SVGGElement>(null);
  // fade transitions track playback speed so dimming reads as motion, not a jump
  const fadeMs = Math.min(650, Math.max(220, Math.round(speedMs * 0.55)));

  // ── coordinate conversion via the real SVG CTM (handles viewBox scale + letterbox) ──
  // viewBox coords (root svg) — for pan/zoom math
  function clientToVB(cx: number, cy: number) {
    const ctm = svgRef.current!.getScreenCTM()!;
    return new DOMPoint(cx, cy).matrixTransform(ctm.inverse());
  }
  // graph coords (inside the translated/scaled <g>) — for dragging nodes
  function clientToGraph(cx: number, cy: number) {
    const ctm = (gRef.current ?? svgRef.current)!.getScreenCTM()!;
    return new DOMPoint(cx, cy).matrixTransform(ctm.inverse());
  }

  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    const v = clientToVB(e.clientX, e.clientY);
    const factor = Math.exp(-Math.max(-40, Math.min(40, e.deltaY)) * 0.0016);
    setView((s) => { const k = Math.max(0.3, Math.min(5, s.k * factor)); const f = k / s.k; return { k, tx: v.x - (v.x - s.tx) * f, ty: v.y - (v.y - s.ty) * f }; });
  }
  function onPanDown(e: React.MouseEvent) {
    if ((e.target as Element).closest?.("[data-node]")) return;
    const start = clientToVB(e.clientX, e.clientY); const orig = { ...view };
    const move = (ev: MouseEvent) => { const v = clientToVB(ev.clientX, ev.clientY); setView({ k: orig.k, tx: orig.tx + (v.x - start.x), ty: orig.ty + (v.y - start.y) }); };
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
  }

  // ── node drag (whole graph responds); shift OR the node's kind-group moves together ──
  function onNodeDown(e: React.MouseEvent, n: SimNode) {
    e.stopPropagation();
    const startG = clientToGraph(e.clientX, e.clientY);
    const group = e.shiftKey ? nodes.filter((m) => m.kind === n.kind) : [n];
    const offsets = group.map((m) => ({ m, dx: m.x - startG.x, dy: m.y - startG.y }));
    let moved = false;
    const sim = simRef.current!;
    const move = (ev: MouseEvent) => {
      moved = true;
      const g = clientToGraph(ev.clientX, ev.clientY);
      offsets.forEach(({ m, dx, dy }) => { m.fx = g.x + dx; m.fy = g.y + dy; });
      sim.alphaTarget(0.3).restart();
    };
    const up = () => {
      if (moved) {
        // keep dragged nodes pinned where dropped, and persist so it survives reload
        offsets.forEach(({ m }) => { m.fx = m.x; m.fy = m.y; pinsRef.current[m.id] = { x: m.x, y: m.y }; });
        localStorage.setItem(PIN_KEY, JSON.stringify(pinsRef.current));
      }
      sim.alphaTarget(0);
      window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up);
      if (!moved) onPick(n.id, n.label);
    };
    window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
  }

  // directed current-frame hops resolved to node objects — drive the moving pulse + chip
  const hops = activeEdges.map(([f, t]) => [byId.get(f), byId.get(t)] as [SimNode | undefined, SimNode | undefined]).filter(([a, b]) => a && b) as [SimNode, SimNode][];
  const chipEdge: [SimNode, SimNode] | null = hops[0] ?? null;

  return (
    <svg ref={svgRef} viewBox={`0 0 ${VB.w} ${VB.h}`} onWheel={onWheel} onMouseDown={onPanDown}
      style={{ width: "100%", height: "100%", display: "block", cursor: "move", userSelect: "none" }} onClick={() => onPick("", "")}>
      <defs>
        <filter id="bglow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="3.5" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
      </defs>
      <rect x={0} y={0} width={VB.w} height={VB.h} fill="transparent" />
      <g ref={gRef} transform={`translate(${view.tx},${view.ty}) scale(${view.k})`}>
        {/* edges — trace stack: current link (age 0) pulses; ancestor links stay lit, fading by age */}
        {runtime.edges.map((e, i) => {
          const a = byId.get(e.source), b = byId.get(e.target);
          if (!a || !b) return null;
          const age = overview ? undefined : edgeAge.get(ekey(e.source, e.target));
          const ovShown = unmuteAll || (a.touched && b.touched); // overview: touched architecture stands out
          const lit = overview ? ovShown : age !== undefined;
          const current = age === 0; // current step's link — brighter, but the pulse is a separate layer
          const op = overview ? (ovShown ? 0.5 : mutedOpacity * 0.45) : age === undefined ? mutedOpacity * 0.6 : fade(age, fadeRate);
          return (
            <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={current ? "#8a611f" : lit ? "#5c4a2a" : "#33414c"} strokeWidth={current ? 3.2 : lit ? 2 : 1} strokeOpacity={op} strokeLinecap="round" style={{ transition: `stroke ${fadeMs}ms, stroke-opacity ${fadeMs}ms, stroke-width ${fadeMs}ms` }} />
          );
        })}

        {/* moving pulse — drawn along the DIRECTED hop (from→to), so bounce-backs visibly reverse */}
        {!overview && hops.map(([a, b], i) => (
          <path key={`imp-${stepKey}-${i}`} d={`M ${a.x} ${a.y} L ${b.x} ${b.y}`} fill="none" stroke={tokens.impulse} strokeWidth={3.4} strokeLinecap="round"
            pathLength={100} strokeDasharray="16 200" style={{ filter: "drop-shadow(0 0 5px rgba(255,200,120,.9))", animation: `t7dash ${playing ? Math.round(speedMs * 0.9) : 2000}ms linear ${playing ? "both" : "infinite"}` }} />
        ))}

        {/* data chip (pause only, no movement) on the current link */}
        {!overview && !playing && activeLabel && chipEdge && (() => {
          const [a, b] = chipEdge!;
          const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2; const t = activeLabel;
          return (
            <g transform={`translate(${mx},${my})`}>
              <rect x={-t.length * 4 - 8} y={-11} width={t.length * 8 + 16} height={22} rx={11} fill="#0a0e12" stroke={tokens.impulse} strokeWidth={1} style={{ filter: "drop-shadow(0 0 12px rgba(246,180,90,.4))" }} />
              <text textAnchor="middle" y={4} fontFamily={tokens.font.mono} fontSize={13} fill={tokens.impulse}>{t}</text>
            </g>
          );
        })()}

        {/* nodes — current step glows; touched-earlier stay lit and fade by age; untouched dim */}
        {nodes.map((n) => {
          const r = radius(n);
          const age = overview ? 0 : nodeAge.get(n.id);
          const ovShown = unmuteAll || n.touched; // overview: untouched architecture is muted
          const lit = overview ? ovShown : age !== undefined;
          const current = !overview && age === 0; // touched this step
          const color = kindColor(n.kind);
          const op = overview ? (ovShown ? 1 : mutedOpacity) : age === undefined ? mutedOpacity : fade(age, fadeRate);
          return (
            <g key={n.id} data-node transform={`translate(${n.x},${n.y})`} opacity={op} style={{ cursor: "pointer", transition: `opacity ${fadeMs}ms` }}
              onMouseDown={(e) => onNodeDown(e, n)} onDoubleClick={(e) => { e.stopPropagation(); onBlacklist(n.id); }}>
              <circle r={r} fill={current ? "#150f07" : tokens.bg.card} stroke={current ? tokens.impulse : lit ? color : "#2a343d"} strokeWidth={current ? 2.5 : picked === n.id ? 2 : 1.4} filter={current || (overview && lit) ? "url(#bglow)" : undefined} style={{ transition: `stroke ${fadeMs}ms, fill ${fadeMs}ms, stroke-width ${fadeMs}ms` }} />
              <circle r={Math.max(3, r * 0.32)} fill={color} opacity={lit ? 0.95 : 0.4} style={{ transition: `opacity ${fadeMs}ms` }} />
              <text y={r + 14} textAnchor="middle" fontFamily={tokens.font.mono} fontSize={13} fill={current ? tokens.impulse : lit ? tokens.text.mid : tokens.text.faint}>{n.label}</text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

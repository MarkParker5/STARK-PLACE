import { useEffect, useMemo, useRef, useState } from "react";
import { edgeKey } from "../derive";
import { layoutGraph, type PositionedNode } from "../layout/forceLayout";
import type { Graph, GraphNode } from "../schema";
import { tokens } from "../tokens";

interface Props {
  graph: Graph;
  width: number;
  height: number;
  spread?: number;
  mode?: "gravity" | "zones";
  minWeight?: number;
  // playback / overview model
  touched: Set<string>;
  touchedEdgeKeys: Set<string>;
  overview: boolean;
  activeId: string | null;
  activeEdge: [string, string] | null;
  activeLabel?: string;
  mutedOpacity?: number;
  unmuteAll?: boolean;
  speedMs?: number;
  playing?: boolean;
  spacing?: number;
  blacklist?: Set<string>;
  centralityOf?: (n: GraphNode) => number;
  // presentation
  colorFor: (n: GraphNode) => string;
  onPick: (n: GraphNode | null) => void;
  picked: string | null;
  edit: boolean;
  storageKey: string;
  hiddenGroups?: Set<string>;
  groupOf?: (n: GraphNode) => string;
  groupOpacity?: Record<string, number>;
}

export function ForceGraph(props: Props) {
  const { graph, width, height, touched, touchedEdgeKeys, overview, activeId, activeEdge, colorFor, onPick, picked, edit, storageKey } = props;
  const groupOf = props.groupOf ?? (() => "all");
  const minWeight = props.minWeight ?? 0;
  const unmuteAll = !!props.unmuteAll;
  const mutedBase = unmuteAll ? 0.75 : props.mutedOpacity ?? 0.16;
  const speedMs = props.speedMs ?? 900;
  const blacklist = props.blacklist ?? new Set<string>();

  const base = useMemo(
    () => layoutGraph(graph, width, height, { spread: props.spread ?? 1.3, spacing: props.spacing ?? 1, mode: props.mode ?? "gravity", groupOf, centralityOf: props.centralityOf }),
    [graph, width, height, props.spread, props.spacing, props.mode, props.centralityOf]
  );
  const basePos = useMemo(() => new Map(base.nodes.map((n) => [n.id, { x: n.x, y: n.y }])), [base]);
  const nodeById = useMemo(() => new Map(base.nodes.map((n) => [n.id, n])), [base]);

  // manual drag overrides (persisted)
  const [overrides, setOverrides] = useState<Record<string, { x: number; y: number }>>({});
  useEffect(() => {
    try {
      setOverrides(JSON.parse(localStorage.getItem(storageKey) || "{}"));
    } catch {
      setOverrides({});
    }
  }, [storageKey]);

  // zoom / pan
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const dragId = useRef<string | null>(null);

  function toViewBox(clientX: number, clientY: number) {
    const rect = svgRef.current!.getBoundingClientRect();
    return { x: ((clientX - rect.left) / rect.width) * width, y: ((clientY - rect.top) / rect.height) * height };
  }
  function toGraph(clientX: number, clientY: number) {
    const v = toViewBox(clientX, clientY);
    return { x: (v.x - view.tx) / view.k, y: (v.y - view.ty) / view.k };
  }

  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    const v = toViewBox(e.clientX, e.clientY);
    // gentle, normalized zoom (smoother + less sensitive)
    const delta = Math.max(-40, Math.min(40, e.deltaY));
    const factor = Math.exp(-delta * 0.0016);
    setView((s) => {
      const k = Math.max(0.3, Math.min(6, s.k * factor));
      const f = k / s.k;
      return { k, tx: v.x - (v.x - s.tx) * f, ty: v.y - (v.y - s.ty) * f };
    });
  }

  function onNodeDown(e: React.MouseEvent, id: string) {
    if (!edit) return;
    e.preventDefault();
    e.stopPropagation();
    dragId.current = id;
    const move = (ev: MouseEvent) => {
      if (!dragId.current) return;
      setOverrides((o) => ({ ...o, [dragId.current!]: toGraph(ev.clientX, ev.clientY) }));
    };
    const up = () => {
      setOverrides((o) => {
        localStorage.setItem(storageKey, JSON.stringify(o));
        return o;
      });
      dragId.current = null;
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function onPanDown(e: React.MouseEvent) {
    if (e.target !== svgRef.current && (e.target as Element).tagName !== "rect") return; // only empty space
    const start = toViewBox(e.clientX, e.clientY);
    const orig = { ...view };
    const move = (ev: MouseEvent) => {
      const v = toViewBox(ev.clientX, ev.clientY);
      setView({ k: orig.k, tx: orig.tx + (v.x - start.x), ty: orig.ty + (v.y - start.y) });
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  const P = (id: string) => overrides[id] ?? basePos.get(id) ?? { x: width / 2, y: height / 2 };
  const hidden = props.hiddenGroups ?? new Set<string>();
  const groupOpacity = props.groupOpacity ?? {};

  function visible(n: PositionedNode): boolean {
    if (blacklist.has(n.id)) return false; // blacklisted -> removed from display AND timeline
    if (!overview && activeId === n.id) return true; // active step always visible
    if (hidden.has(groupOf(n.node))) return false;
    if (n.node.weight < minWeight) return false;
    return true;
  }
  function nodeOpacity(n: PositionedNode): number {
    const go = groupOpacity[groupOf(n.node)] ?? 1;
    if (!touched.has(n.id)) return mutedBase * go; // untouched -> muted (configurable / unmuted)
    return go; // touched -> shown
  }

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "100%", display: "block", cursor: edit ? "grab" : "move", userSelect: "none", WebkitUserSelect: "none" }}
      onWheel={onWheel}
      onMouseDown={onPanDown}
      onClick={() => onPick(null)}
    >
      <defs>
        <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3.5" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <rect x={0} y={0} width={width} height={height} fill="transparent" />

      <g transform={`translate(${view.tx},${view.ty}) scale(${view.k})`} style={{ transition: "transform .1s ease-out" }}>
        {/* edges */}
        <g>
          {base.edges.map((e, i) => {
            const na = nodeById.get(e.source)!;
            const nb = nodeById.get(e.target)!;
            if (!visible(na) || !visible(nb)) return null;
            const a = P(e.source);
            const b = P(e.target);
            const key = edgeKey(e.source, e.target);
            const isTouched = touchedEdgeKeys.has(key);
            const isActive =
              activeEdge &&
              ((activeEdge[0] === e.source && activeEdge[1] === e.target) ||
                (activeEdge[0] === e.target && activeEdge[1] === e.source));
            const op = Math.min(nodeOpacity(na), nodeOpacity(nb));
            const strokeOp = isActive ? 1 : isTouched ? 0.5 : 0.12; // untouched links muted (less than before)
            const dash = e.edge.kind === "inherit" ? "none" : e.edge.kind === "compose" ? "4 4" : "1 3";
            const d = `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
            return (
              <g key={i}>
                <line
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={isActive ? "#7a561f" : isTouched ? "#3a4650" : tokens.border.mid}
                  strokeWidth={isActive ? 3 : isTouched ? 1 : 0.6}
                  strokeDasharray={isActive ? undefined : dash}
                  strokeLinecap="round"
                  strokeOpacity={strokeOp * op}
                />
                {/* design impulse: moving gradient overlaying the gray link (t7dash) */}
                {isActive && (
                  <path
                    d={d}
                    fill="none"
                    stroke={tokens.impulse}
                    strokeWidth={3.2}
                    strokeLinecap="round"
                    pathLength={100}
                    strokeDasharray="16 200"
                    style={{ filter: "drop-shadow(0 0 5px rgba(255,200,120,.9))", animation: `t7dash ${speedMs}ms linear infinite` }}
                  />
                )}
              </g>
            );
          })}
        </g>

        {/* data chip on the active edge — shown only when PAUSED, no movement animation */}
        {!overview &&
          !props.playing &&
          activeEdge &&
          (() => {
            const a = P(activeEdge[0]);
            const b = P(activeEdge[1]);
            if (!nodeById.get(activeEdge[0]) || !nodeById.get(activeEdge[1])) return null;
            const mx = (a.x + b.x) / 2;
            const my = (a.y + b.y) / 2;
            const text = props.activeLabel || "";
            if (!text) return null;
            return (
              <g transform={`translate(${mx},${my})`}>
                <rect x={-text.length * 4 - 8} y={-11} width={text.length * 8 + 16} height={22} rx={11} fill="#0a0e12" stroke={tokens.impulse} strokeWidth={1} style={{ filter: "drop-shadow(0 0 12px rgba(246,180,90,.4))" }} />
                <text textAnchor="middle" y={4} fontFamily={tokens.font.mono} fontSize={13} fill={tokens.impulse}>
                  {text}
                </text>
              </g>
            );
          })()}

        {/* nodes */}
        <g>
          {base.nodes.map((n) => {
            if (!visible(n)) return null;
            const p = P(n.id);
            const op = nodeOpacity(n);
            const isTouched = touched.has(n.id);
            const isActive = !overview && activeId === n.id;
            const highlight = overview ? isTouched : isActive; // overview lights all touched; playback lights current
            const isPicked = picked === n.id;
            const shown = isTouched || unmuteAll; // unmute-all reveals untouched nodes fully
            const color = colorFor(n.node);
            return (
              <g
                key={n.id}
                transform={`translate(${p.x},${p.y})`}
                opacity={op}
                style={{ cursor: edit ? "grab" : "pointer" }}
                onMouseDown={(e) => onNodeDown(e, n.id)}
                onClick={(e) => {
                  e.stopPropagation();
                  onPick(n.node);
                }}
              >
                <circle
                  r={n.r}
                  fill={isActive ? tokens.bg.active : tokens.bg.card}
                  stroke={isActive ? tokens.impulse : isPicked || highlight ? color : shown ? color : "#2a343d"}
                  strokeWidth={isActive ? 2.5 : highlight || isPicked ? 1.6 : shown ? 1.2 : 1}
                  filter={isActive || (overview && isTouched) ? "url(#glow)" : undefined}
                />
                <circle r={Math.max(2, n.r * 0.3)} fill={color} opacity={highlight ? 0.95 : shown ? 0.7 : 0.35} />
                {(n.r > 15 || highlight || isPicked || shown) && (
                  <text
                    y={n.r + 12}
                    textAnchor="middle"
                    fontFamily={tokens.font.mono}
                    fontSize={13}
                    fill={isActive ? tokens.impulse : highlight ? tokens.text.mid : shown ? tokens.text.muted : tokens.text.faint}
                  >
                    {n.node.label}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </g>
    </svg>
  );
}

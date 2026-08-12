import { useEffect, useMemo, useRef, useState } from "react";
import type { Step } from "../schema";
import { ACC, type Field, type WiringModel } from "../wiringModel";
import { tokens } from "../tokens";

const VB = { w: 1900, h: 700 };
const POS_KEY = "stark-wiring-pos";

interface Props {
  steps: Step[];
  index: number;
  overview: boolean;
  playing: boolean;
  edit: boolean;
  speedMs: number;
  mutedOpacity: number;
  unmuteAll: boolean;
  groupOpacity: number;
  model: WiringModel;
  onPickNode: (id: string, label: string) => void;
}

export function Wiring({ steps, index, overview, playing, edit, speedMs, mutedOpacity, unmuteAll, groupOpacity, model, onPickNode }: Props) {
  const { fields, heart, matched, coordOf, heartReach, nodes: NODES, groups: G12 } = model;
  const hstep = Math.min(index, heart.length - 1);
  const curAct = heart[hstep];

  const activeSet = useMemo(() => {
    const s = new Set<string>();
    if (!overview) curAct.e.forEach(([x, y]) => { s.add(x); s.add(y); });
    return s;
  }, [curAct, overview]);

  const touched = useMemo(() => {
    const s = new Set<string>(Object.keys(heartReach));
    matched.forEach((m) => s.add("cmd_" + m));
    return s;
  }, [matched]);

  // position overrides (edit-mode drag, persisted)
  const [overrides, setOverrides] = useState<Record<string, { x: number; y: number }>>({});
  useEffect(() => {
    try { setOverrides(JSON.parse(localStorage.getItem(POS_KEY) || "{}")); } catch { setOverrides({}); }
  }, []);
  const eff = (id: string) => {
    const c = coordOf.get(id)!;
    const o = overrides[id];
    const x = o ? o.x : c.x, y = o ? o.y : c.y;
    return { x, y, w: c.w, h: c.h, cx: x + c.w / 2, cy: y + c.h / 2 };
  };
  function edgePath(a: string, b: string): string {
    const A = eff(a), B = eff(b);
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

  const activeNode = overview ? null : null;
  const activeEdge = null;

  // zoom / pan
  const [view, setView] = useState({ k: 0.62, tx: 40, ty: 20 });
  const wrapRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ id: string } | null>(null);
  function toLocal(cx: number, cy: number) {
    const r = wrapRef.current!.getBoundingClientRect();
    return { x: cx - r.left, y: cy - r.top };
  }
  function toStage(clientX: number, clientY: number) {
    const p = toLocal(clientX, clientY);
    return { x: (p.x - view.tx) / view.k, y: (p.y - view.ty) / view.k };
  }
  function onWheel(e: React.WheelEvent) {
    e.preventDefault();
    const p = toLocal(e.clientX, e.clientY);
    const delta = Math.max(-40, Math.min(40, e.deltaY));
    const factor = Math.exp(-delta * 0.0016);
    setView((s) => {
      const k = Math.max(0.3, Math.min(3, s.k * factor));
      const f = k / s.k;
      return { k, tx: p.x - (p.x - s.tx) * f, ty: p.y - (p.y - s.ty) * f };
    });
  }
  function onPanDown(e: React.MouseEvent) {
    if ((e.target as HTMLElement).closest?.("[data-node]")) return; // node handles its own drag
    const start = { x: e.clientX, y: e.clientY };
    const orig = { ...view };
    const move = (ev: MouseEvent) => setView({ k: orig.k, tx: orig.tx + (ev.clientX - start.x), ty: orig.ty + (ev.clientY - start.y) });
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }
  // drag a whole group by its hull label
  function onGroupDown(e: React.MouseEvent, ids: string[]) {
    if (!edit) return;
    e.preventDefault();
    e.stopPropagation();
    drag.current = { id: ids[0] };
    const start = toStage(e.clientX, e.clientY);
    const origins = ids.map((gid) => { const c = eff(gid); return { gid, dx: c.x - start.x, dy: c.y - start.y }; });
    const move = (ev: MouseEvent) => {
      const g = toStage(ev.clientX, ev.clientY);
      setOverrides((o) => { const n = { ...o }; origins.forEach(({ gid, dx, dy }) => { n[gid] = { x: g.x + dx, y: g.y + dy }; }); return n; });
    };
    const up = () => {
      setOverrides((o) => { localStorage.setItem(POS_KEY, JSON.stringify(o)); return o; });
      drag.current = null;
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function onNodeDown(e: React.MouseEvent, id: string) {
    if (!edit) return;
    e.preventDefault();
    e.stopPropagation(); // don't start a pan
    drag.current = { id };
    // shift-drag moves the whole group (all nodes sharing this node's group)
    const grp = NODES.find((n) => n.id === id)?.grp;
    const groupIds = e.shiftKey && grp ? NODES.filter((n) => n.grp === grp).map((n) => n.id) : [id];
    const start = toStage(e.clientX, e.clientY);
    const origins = groupIds.map((gid) => { const c = eff(gid); return { gid, dx: c.x - start.x, dy: c.y - start.y }; });
    const move = (ev: MouseEvent) => {
      const g = toStage(ev.clientX, ev.clientY);
      setOverrides((o) => { const n = { ...o }; origins.forEach(({ gid, dx, dy }) => { n[gid] = { x: g.x + dx, y: g.y + dy }; }); return n; });
    };
    const up = () => {
      setOverrides((o) => { localStorage.setItem(POS_KEY, JSON.stringify(o)); return o; });
      drag.current = null;
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  const muteOp = unmuteAll ? 0.75 : mutedOpacity;

  function nodeStyle(id: string, grp: string, main: boolean): React.CSSProperties {
    const c = eff(id);
    const accent = ACC[grp] || "#e2e8ee";
    const isStr = grp === "PROD" || grp === "RESP";
    const active = activeSet.has(id);
    const done = id in heartReach && heartReach[id] < hstep;
    const isTouched = touched.has(id);
    const shown = isTouched || unmuteAll;

    let border: string, bg: string, glow = "", op = 1, labelColor: string;
    if (active) {
      border = "#f6b45a"; bg = "#150f07"; glow = "0 0 20px -7px rgba(246,180,90,.55)"; labelColor = "#f6b968";
    } else if (overview && isTouched) {
      border = accent; bg = grp === "PROD" ? "#0b1310" : "#0b0f14"; labelColor = isStr ? "#bfe6de" : accent;
    } else if (!overview && done) {
      border = isStr ? "#2f4a44" : "#33414c"; bg = grp === "PROD" ? "#0b1310" : "#0b0f14"; labelColor = isStr ? "#bfe6de" : accent;
    } else if (shown) {
      border = `${accent}66`; bg = "#0b0f14"; labelColor = "#8b96a1"; op = isTouched ? (main ? 0.7 : 0.5) : muteOp;
    } else {
      border = main ? "#243039" : "#18222b"; bg = "#0b0f14"; labelColor = "#66707a"; op = muteOp;
    }
    return {
      position: "absolute", left: c.x, top: c.y, width: c.w, minHeight: c.h,
      boxSizing: "border-box", padding: main ? "9px 12px" : "6px 10px", border: `1px solid ${border}`, borderRadius: 10, background: bg,
      boxShadow: glow || undefined, opacity: op, color: labelColor,
      transition: drag.current ? "none" : "border-color .4s, box-shadow .45s, opacity .4s, background .4s",
      cursor: edit ? "grab" : "pointer", zIndex: main ? 4 : 3,
    };
  }

  return (
    <div ref={wrapRef} onWheel={onWheel} onMouseDown={onPanDown}
      style={{ width: "100%", height: "100%", overflow: "hidden", position: "relative", cursor: "move", userSelect: "none" }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: VB.w, height: VB.h, transformOrigin: "0 0",
        transform: `translate(${view.tx}px,${view.ty}px) scale(${view.k})`, transition: drag.current ? "none" : "transform .1s ease-out" }}>

        {/* group hulls */}
        {G12.map(([name, col, ids, pad]) => {
          let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
          ids.forEach((id) => { const c = eff(id); if (!c) return; x0 = Math.min(x0, c.x); y0 = Math.min(y0, c.y); x1 = Math.max(x1, c.x + c.w); y1 = Math.max(y1, c.y + c.h); });
          return (
            <div key={name} style={{ position: "absolute", left: x0 - pad, top: y0 - pad, width: x1 - x0 + 2 * pad, height: y1 - y0 + 2 * pad,
              border: `1.4px dashed ${col}55`, borderRadius: 26, zIndex: 1, pointerEvents: "none", opacity: groupOpacity }}>
              <span data-node onMouseDown={(e) => onGroupDown(e, ids)}
                style={{ position: "absolute", left: 12, top: -10, font: "13px 'Space Mono',monospace", letterSpacing: ".18em", color: `${col}dd`, background: tokens.bg.page, padding: "0 6px",
                  pointerEvents: edit ? "auto" : "none", cursor: edit ? "grab" : "default", border: edit ? `1px solid ${col}55` : "none", borderRadius: 6 }}>
                {edit ? "⠿ " : ""}{name}
              </span>
            </div>
          );
        })}

        {/* wires */}
        <svg width={VB.w} height={VB.h} style={{ position: "absolute", left: 0, top: 0, zIndex: 2, overflow: "visible", pointerEvents: "none" }}>
          {heart.map((a, ai) =>
            a.e.map(([x, y]) => {
              const d = edgePath(x, y);
              const on = !overview && ai === hstep;
              const past = !overview && ai < hstep;
              const shown = overview || on || past;
              return (
                <path key={`b${x}${y}${ai}`} d={d} fill="none"
                  stroke={on ? "#7a561f" : past ? "#33414c" : overview ? "#2a3742" : "#182530"}
                  strokeWidth={on ? 3 : past ? 1.6 : 1.1} opacity={shown ? (on ? 0.95 : past ? 0.5 : 0.6) : 0.18} strokeLinecap="round" />
              );
            })
          )}
          {!overview && curAct.e.map(([x, y]) => (
            // one-shot impulse synced to the step interval (no looping / double-repeat)
            <path key={`i${x}${y}${hstep}`} d={edgePath(x, y)} fill="none" stroke={curAct.c || "#ffd28a"} strokeWidth={3.2} strokeLinecap="round"
              pathLength={100} strokeDasharray="16 200" style={{ filter: "drop-shadow(0 0 5px rgba(255,200,120,.9))", animation: `t7dash ${playing ? Math.round(speedMs * 0.9) : 2000}ms linear ${playing ? "both" : "infinite"}` }} />
          ))}
        </svg>

        {/* nodes */}
        {NODES.map((n) => {
          const fld = fields[n.id] as Field[] | undefined;
          return (
            <div key={n.id} data-node style={nodeStyle(n.id, n.grp, n.main)}
              onMouseDown={(e) => onNodeDown(e, n.id)}
              onClick={(e) => { e.stopPropagation(); if (!drag.current) onPickNode(n.id, n.label); }}>
              <div style={{ font: `600 13px ${tokens.font.display}`, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{n.label}</div>
              {n.sub && !fld && <div style={{ font: `11px ${tokens.font.mono}`, letterSpacing: ".05em", textTransform: "uppercase", color: "#4d5761", marginTop: 2 }}>{n.grp === "CMD" ? "" : n.sub}</div>}
              {n.kind === "tree" && <TreeRows />}
              {n.kind === "queue" && <QueueRows steps={steps} />}
              {fld && <FieldBox rows={fld} />}
            </div>
          );
        })}

        {/* data chip — only when PAUSED, no movement animation */}
        {!overview && !playing && (() => {
          const [x, y] = curAct.e[0];
          const A = eff(x), B = eff(y);
          const mx = (A.cx + B.cx) / 2, my = (A.cy + B.cy) / 2;
          return (
            <div style={{ position: "absolute", left: mx, top: my, transform: "translate(-50%,-50%)", zIndex: 9, padding: "3px 11px", borderRadius: 999,
              font: `600 12px ${tokens.font.mono}`, whiteSpace: "nowrap", border: `1px solid ${curAct.c}`, background: "#0a0e12", color: curAct.c, boxShadow: `0 0 15px -3px ${curAct.c}` }}>
              {curAct.d || curAct.t}
            </div>
          );
        })()}
      </div>
    </div>
  );
}

function FieldBox({ rows }: { rows: Field[] }) {
  return (
    <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 3, borderTop: "1px dashed #232d36", paddingTop: 6 }}>
      {rows.map((r, i) => (
        <div key={i} style={{ display: "flex", alignItems: "baseline", gap: 6, font: `600 11px ${tokens.font.mono}`, lineHeight: 1.4 }}>
          <span style={{ color: r[0] === "in" ? "#6cc08a" : "#f6b45a", flex: "none" }}>{r[0] === "in" ? "◂" : "▸"}</span>
          <span style={{ color: "#7f8a94", flex: "none" }}>{r[1]}</span>
          {r[2] && <span style={{ color: "#c2cad2", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r[2]}</span>}
        </div>
      ))}
    </div>
  );
}

function TreeRows() {
  const rows = [{ l: "root", d: 0, c: "12 commands" }, { l: "music ▸ playing", d: 1, c: "stop · next" }];
  return (
    <div style={{ marginTop: 6, borderTop: "1px dashed #232d36", paddingTop: 6 }}>
      {rows.map((r, i) => (
        <div key={i} style={{ font: `11px ${tokens.font.mono}`, color: i === rows.length - 1 ? "#f6b968" : "#7f8a94", paddingLeft: r.d * 12 }}>
          {r.d > 0 ? "▸ " : ""}{r.l} <span style={{ color: "#4d5761" }}>· {r.c}</span>
        </div>
      ))}
    </div>
  );
}

function QueueRows({ steps }: { steps: Step[] }) {
  const resp = steps.filter((s) => s.label === "respond").map((s) => (s.input?.response as any)?.text || "response");
  const items = resp.length ? resp : ["response"];
  return (
    <div style={{ marginTop: 6, borderTop: "1px dashed #232d36", paddingTop: 6, display: "flex", flexDirection: "column", gap: 3 }}>
      {items.slice(0, 3).map((t, i) => (
        <div key={i} style={{ font: `11px ${tokens.font.mono}`, color: "#bfe6de", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{i + 1} ▸ {t}</div>
      ))}
    </div>
  );
}


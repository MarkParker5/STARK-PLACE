import { useMemo, useState } from "react";
import type { ProfileEvent } from "../schema";
import { tokens } from "../tokens";

// Per-function timing rolled up from the raw capture, so "a fast function called many times" is
// visible next to "one slow call". Three sortings answer three different questions.
type Sort = "chrono" | "single" | "combined";
interface Row { symbol: string; count: number; total: number; max: number; firstSeq: number }

const mono = tokens.font.mono;
const ms = (ns: number) => (ns / 1e6 >= 1 ? `${(ns / 1e6).toFixed(2)}ms` : `${(ns / 1e3).toFixed(0)}µs`);

export function Durations({ events, onClose }: { events: ProfileEvent[]; onClose: () => void }) {
  const [sort, setSort] = useState<Sort>("combined");

  const rows = useMemo(() => {
    const by: Record<string, Row> = {};
    for (const e of events) {
      if ((e.phase !== "return" && e.phase !== "error") || e.dur_ns == null) continue;
      const r = (by[e.symbol] ??= { symbol: e.symbol, count: 0, total: 0, max: 0, firstSeq: e.seq });
      r.count++;
      r.total += e.dur_ns;
      r.max = Math.max(r.max, e.dur_ns);
      r.firstSeq = Math.min(r.firstSeq, e.seq);
    }
    const list = Object.values(by);
    const cmp: Record<Sort, (a: Row, b: Row) => number> = {
      chrono: (a, b) => a.firstSeq - b.firstSeq,
      single: (a, b) => b.max - a.max,
      combined: (a, b) => b.total - a.total,
    };
    return list.sort(cmp[sort]);
  }, [events, sort]);

  const maxTotal = Math.max(1, ...rows.map((r) => r.total));
  const grand = rows.reduce((s, r) => s + r.total, 0);

  const tabs: [Sort, string, string][] = [
    ["chrono", "chronological", "call order"],
    ["single", "longest single", "one slow call"],
    ["combined", "longest combined", "many small calls add up"],
  ];

  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(4,7,10,.72)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center", padding: 30 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ width: "min(760px,92vw)", maxHeight: "82vh", display: "flex", flexDirection: "column", background: tokens.bg.card, border: `1px solid ${tokens.border.mid}`, borderRadius: 14, overflow: "hidden", boxShadow: "10px 10px 0 rgba(4,7,10,.5)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "13px 16px", borderBottom: `1px solid ${tokens.border.faint}` }}>
          <span style={{ font: `13px ${mono}`, letterSpacing: ".14em", color: tokens.impulse }}>⏱ STEP DURATIONS</span>
          <span style={{ font: `12px ${mono}`, color: tokens.text.faint }}>{rows.length} functions · Σ {ms(grand)}</span>
          <div style={{ flex: 1 }} />
          <button onClick={onClose} style={{ background: "none", border: "none", color: tokens.text.mid, cursor: "pointer", fontSize: 16 }}>✕</button>
        </div>
        <div style={{ display: "flex", gap: 6, padding: "10px 16px" }}>
          {tabs.map(([k, label, hint]) => (
            <button key={k} onClick={() => setSort(k)} title={hint}
              style={{ background: sort === k ? "rgba(246,180,90,.12)" : tokens.bg.card2, border: `1px solid ${sort === k ? tokens.impulse : tokens.border.mid}`, color: sort === k ? tokens.impulse : tokens.text.mid, borderRadius: 8, padding: "6px 11px", font: `13px ${mono}`, cursor: "pointer" }}>
              {label}
            </button>
          ))}
        </div>
        <div style={{ overflow: "auto", padding: "0 16px 16px" }}>
          {rows.map((r, i) => {
            const [cls, meth] = r.symbol.includes(".") ? [r.symbol.slice(0, r.symbol.lastIndexOf(".")), r.symbol.slice(r.symbol.lastIndexOf(".") + 1)] : ["", r.symbol];
            const pct = (r.total / maxTotal) * 100;
            return (
              <div key={r.symbol} style={{ position: "relative", padding: "7px 10px", borderBottom: `1px solid ${tokens.border.faint}` }}>
                <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${pct}%`, background: "rgba(246,180,90,.09)", borderRadius: 4 }} />
                <div style={{ position: "relative", display: "flex", alignItems: "baseline", gap: 8 }}>
                  <span style={{ font: `12px ${mono}`, color: tokens.text.faint, width: 22 }}>{i + 1}</span>
                  <span style={{ font: `13px ${mono}`, color: tokens.text.hi }}>{meth}</span>
                  {cls && <span style={{ font: `11px ${mono}`, color: tokens.text.faint }}>{cls}</span>}
                  <div style={{ flex: 1 }} />
                  <span style={{ font: `12px ${mono}`, color: tokens.text.muted }}>×{r.count}</span>
                  <span style={{ font: `12px ${mono}`, color: tokens.text.faint, width: 76, textAlign: "right" }} title="slowest single call">max {ms(r.max)}</span>
                  <span style={{ font: `12px ${mono}`, color: tokens.impulse, width: 78, textAlign: "right" }} title="all calls combined">Σ {ms(r.total)}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

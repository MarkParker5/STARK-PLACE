import { useMemo, useState } from "react";
import { reparse, submitUtterance, type ReparseResult, type Step } from "../schema";
import { buildMatching, cmdColor, type Attempt, type CommandMatch } from "../matching";
import { tokens } from "../tokens";

interface Props {
  steps: Step[];
  utterance: string;
  onPick: (title: string, input: any) => void;
}

const mono = tokens.font.mono;

// the raw utterance carved into resolved command spans (overlap resolution, visualised)
function Ribbon({ utterance, results, sel, onSel }: { utterance: string; results: CommandMatch[]; sel: number; onSel: (i: number) => void }) {
  const parts: { text: string; ci: number }[] = [];
  let cur = 0;
  const sorted = results.map((r, i) => ({ r, i })).sort((a, b) => a.r.start - b.r.start);
  sorted.forEach(({ r, i }) => {
    if (r.start > cur) parts.push({ text: utterance.slice(cur, r.start), ci: -1 });
    parts.push({ text: utterance.slice(r.start, r.end), ci: i });
    cur = Math.max(cur, r.end);
  });
  if (cur < utterance.length) parts.push({ text: utterance.slice(cur), ci: -1 });
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 2, alignItems: "center", lineHeight: 2 }}>
      {parts.map((p, k) => {
        if (p.ci < 0) return <span key={k} style={{ font: `15px ${mono}`, color: "#5a6570", padding: "2px 1px" }}>{p.text}</span>;
        const r = results[p.ci];
        const c = cmdColor(r.command);
        const active = p.ci === sel;
        // calm by default: muted unless focused; the selected span alone carries color
        return (
          <span key={k} onClick={() => onSel(p.ci)} title={`${r.command} · index ${r.index}`}
            style={{ font: `600 15px ${mono}`, color: active ? "#0a0e12" : "#aeb7c0", background: active ? c : "transparent", border: `1px solid ${active ? c : "#2a343d"}`, borderRadius: 6, padding: "2px 7px", cursor: "pointer" }}>
            {p.text}
          </span>
        );
      })}
    </div>
  );
}

function Chip({ text, color = "#9aa4ae", bg = "#121820" }: { text: string; color?: string; bg?: string }) {
  return <span style={{ font: `12px ${mono}`, color, background: bg, border: `1px solid #232e38`, borderRadius: 999, padding: "1px 8px", whiteSpace: "nowrap" }}>{text}</span>;
}

// a small labelled container so params and corrections read as separate groups, not one soup
function TagBox({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 7px 3px 8px", border: "1px solid #1c2530", borderRadius: 8, background: "#0a0e12" }}>
      <span style={{ font: `10px ${mono}`, letterSpacing: ".1em", color: "#4d5761", textTransform: "uppercase" }}>{label}</span>
      {children}
    </span>
  );
}

// one node of the recursive parse tree
function TreeNode({ a, depth, accent }: { a: Attempt; depth: number; accent: string }) {
  const paramEntries = Object.entries(a.parameters);
  const inHl = a.matched && a.start != null && a.end != null && a.string.length > (a.end ?? 0);
  return (
    <div style={{ borderLeft: depth > 0 ? `1px dashed ${accent}55` : "none", marginLeft: depth > 0 ? 6 : 0, paddingLeft: depth > 0 ? 14 : 0, marginTop: 8 }}>
      <div style={{ background: tokens.bg.card2, border: `1px solid ${a.matched ? `${accent}66` : "#3a2630"}`, borderRadius: 10, padding: "9px 12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span style={{ font: `600 13px ${mono}`, color: a.matched ? accent : "#c98a8a" }}>{a.pattern}</span>
          <span style={{ flex: 1 }} />
          {a.cached && <Chip text="cached" color="#6f7883" />}
          <Chip text={a.matched ? "✓ match" : "✕ no match"} color={a.matched ? "#7f8a94" : "#c98a8a"} />
        </div>
        {!a.matched && a.why && <div style={{ font: `11px ${mono}`, color: "#6f7883", marginTop: 4 }}>↳ {a.why}</div>}
        {/* IN → OUT */}
        <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ font: `12px ${mono}`, color: "#7f8a94" }}>
            <span style={{ color: "#5a6570" }}>◂ in </span>
            {inHl
              ? <>“{a.string.slice(0, a.start)}<span style={{ color: accent, background: `${accent}22` }}>{a.string.slice(a.start, a.end)}</span>{a.string.slice(a.end)}”</>
              : <>“{a.string.length > 60 ? a.string.slice(0, 60) + "…" : a.string}”</>}
          </div>
          {a.matched && (
            <div style={{ font: `12px ${mono}`, color: "#c2cad2" }}>
              <span style={{ color: accent }}>▸ out </span>“{a.substring}” <span style={{ color: "#4d5761" }}>@ [{a.start},{a.end}]</span>
            </div>
          )}
          {a.corrected && a.corrected !== a.substring && (
            <div style={{ font: `12px ${mono}`, color: "#9aa4ae" }}>▸ corrected → “{a.corrected}”</div>
          )}
        </div>
        {/* corrections applied at this level */}
        {a.corrections.length > 0 && (
          <div style={{ marginTop: 6, display: "flex", gap: 5, flexWrap: "wrap" }}>
            <span style={{ font: `10px ${mono}`, letterSpacing: ".1em", color: "#4d5761", textTransform: "uppercase", alignSelf: "center" }}>fixes</span>
            {a.corrections.map((c, i) => <Chip key={i} text={c} />)}
          </div>
        )}
        {/* parameters — each recurses into the sub-parser that filled it */}
        {paramEntries.length > 0 && (
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 2 }}>
            {paramEntries.map(([k, v]) => {
              const child = a.children.find((c) => c.string === String(v));
              const type = paramType(a.pattern, k); // the Object class that parses this parameter
              return (
                <div key={k}>
                  <div style={{ font: `12px ${mono}`, color: "#8b96a1", marginTop: 4 }}>
                    <span style={{ color: "#5a6570" }}>param</span> ${k}
                    {type && <span style={{ color: "#8b96a1" }}>:{type}</span>}
                    <span style={{ color: "#4d5761" }}> = </span><span style={{ color: "#dfe4ea" }}>{String(v)}</span>
                    {type && <span style={{ color: "#5a6570" }}> · {type} parser</span>}
                    {!child && <span style={{ color: "#5a6570" }}> · literal</span>}
                  </div>
                  {child && <TreeNode a={child} depth={depth + 1} accent={accent} />}
                </div>
              );
            })}
          </div>
        )}
        {/* any nested parses not tied to a named parameter (e.g. Union branches) */}
        {a.children.filter((c) => !paramEntries.some(([, v]) => String(v) === c.string)).map((c, i) => (
          <TreeNode key={i} a={c} depth={depth + 1} accent={accent} />
        ))}
      </div>
    </div>
  );
}

export function Matching({ steps, utterance, onPick }: Props) {
  const m = useMemo(() => buildMatching(steps, utterance), [steps, utterance]);
  const [sel, setSel] = useState(0);
  const [diffText, setDiffText] = useState("");
  const [diff, setDiff] = useState<{ utterance: string; results: CommandMatch[] } | null>(null);
  const [diffBusy, setDiffBusy] = useState(false);
  async function runDiff() {
    const t = diffText.trim(); if (!t) return;
    setDiffBusy(true);
    try { const b = await submitUtterance(t); setDiff(buildMatching(b.steps, b.utterance)); } catch { setDiff(null); } finally { setDiffBusy(false); }
  }

  if (!m.results.length && !m.attempts.length) {
    return <div style={{ padding: 30, font: `15px ${mono}`, color: tokens.text.mid }}>No pattern-matching in this trace (fallback / no command matched).</div>;
  }

  // rejected command patterns (tried, produced nothing) — the other half of "search"
  const rejected = m.attempts.filter((a) => !a.matched);
  const selMatch = m.results[sel];
  const selAttempt = selMatch?.attempt || m.attempts.find((a) => a.matched);

  return (
    <div style={{ height: "100%", overflow: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 14 }}>
      {/* pre-match enrichment */}
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
        <Section title="CORRECTIONS" hint="CorrectionsProcessor">
          {m.corrections.length ? m.corrections.map((c, i) => <Chip key={i} text={c} />) : <Muted t="none" />}
        </Section>
        <Section title="ENTITIES" hint="SpacyNER">
          {m.entities.length ? m.entities.map((e, i) => <Chip key={i} text={e} />) : <Muted t="none" />}
        </Section>
        <div style={{ flex: 1, minWidth: 260 }}>
          <div style={{ font: `12px ${mono}`, letterSpacing: ".12em", color: "#5a6570" }}>DIFF vs another utterance</div>
          <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
            <input value={diffText} onChange={(e) => setDiffText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && runDiff()} placeholder="e.g. play radiohead and set a timer for ten"
              style={{ flex: 1, background: "#0a0e12", border: `1px solid ${tokens.border.mid}`, borderRadius: 7, padding: "6px 9px", color: tokens.text.hi, font: `13px ${mono}` }} />
            <button onClick={runDiff} disabled={diffBusy} style={{ background: tokens.bg.card, border: `1px solid ${tokens.impulse}`, color: tokens.impulse, borderRadius: 7, padding: "6px 11px", font: `13px ${mono}`, cursor: "pointer" }}>{diffBusy ? "…" : "diff"}</button>
          </div>
          {diff && (() => {
            const A = new Map(m.results.map((r) => [r.command, r]));
            const B = new Map(diff.results.map((r) => [r.command, r]));
            const added = diff.results.filter((r) => !A.has(r.command));
            const removed = m.results.filter((r) => !B.has(r.command));
            const changed = m.results.filter((r) => B.has(r.command) && JSON.stringify(B.get(r.command)!.parameters) !== JSON.stringify(r.parameters));
            return (
              <div style={{ marginTop: 7, display: "flex", flexDirection: "column", gap: 3 }}>
                {added.map((r) => <span key={"a" + r.command} style={{ font: `12px ${mono}`, color: "#6cc08a" }}>+ {r.command} {fmtParams(r.parameters)}</span>)}
                {removed.map((r) => <span key={"r" + r.command} style={{ font: `12px ${mono}`, color: "#c98a8a" }}>− {r.command} {fmtParams(r.parameters)}</span>)}
                {changed.map((r) => <span key={"c" + r.command} style={{ font: `12px ${mono}`, color: "#e0a45a" }}>~ {r.command} {fmtParams(r.parameters)} → {fmtParams(B.get(r.command)!.parameters)}</span>)}
                {!added.length && !removed.length && !changed.length && <span style={{ font: `12px ${mono}`, color: "#5a6570" }}>identical command set</span>}
              </div>
            );
          })()}
        </div>
      </div>

      {/* e2e command search — overlap resolution ribbon */}
      <div style={{ background: tokens.bg.card, border: `1px solid ${tokens.border.faint}`, borderRadius: 12, padding: 14 }}>
        <Hdr t="COMMAND SEARCH · overlap resolution" s={`${m.results.length} matched · ${rejected.length} rejected`} />
        <div style={{ marginTop: 10 }}>
          <Ribbon utterance={m.utterance} results={m.results} sel={sel} onSel={setSel} />
        </div>
        <div style={{ marginTop: 6, font: `12px ${mono}`, color: "#5a6570" }}>each span is a resolved command · click to inspect its parse tree · gaps are unmatched filler</div>
        {/* span-conflict inspector */}
        <div style={{ marginTop: 8, paddingTop: 8, borderTop: `1px dashed ${tokens.border.faint}` }}>
          <span style={{ font: `11px ${mono}`, letterSpacing: ".1em", color: "#4d5761" }}>SPAN CONFLICTS</span>{" "}
          {m.conflicts.length === 0
            ? <span style={{ font: `12px ${mono}`, color: "#5a6570" }}>none — every command claimed a disjoint span</span>
            : m.conflicts.map((c, i) => (
              <div key={i} style={{ font: `12px ${mono}`, color: "#c98a8a", marginTop: 3 }}>“{c.substring}” [{c.start},{c.end}] lost to <b style={{ color: "#dfe4ea" }}>{c.lostTo}</b></div>
            ))}
        </div>
      </div>

      {/* two columns: resolved list + recursive parse tree */}
      <div style={{ display: "flex", gap: 14, alignItems: "flex-start", flexWrap: "wrap" }}>
        {/* resolved + rejected */}
        <div style={{ flex: "1 1 340px", minWidth: 300, display: "flex", flexDirection: "column", gap: 8 }}>
          <Hdr t="RESOLVED" s="registry order" />
          {m.results.map((r, i) => {
            const c = cmdColor(r.command);
            const on = i === sel;
            return (
              <div key={i} onClick={() => { setSel(i); onPick(`${r.command}()`, { command: r.full, index: r.index, substring: r.substring, parameters: r.parameters, corrections: r.corrections }); }}
                style={{ background: on ? tokens.bg.active : tokens.bg.card2, border: `1px solid ${on ? c : tokens.border.soft}`, borderRadius: 10, padding: "9px 12px", cursor: "pointer" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 2, background: on ? c : "#3a444d" }} />
                  <span style={{ font: `600 14px ${tokens.font.display}`, color: on ? "#e2e8ee" : tokens.text.mid }}>{r.command}</span>
                  <span style={{ flex: 1 }} />
                  <span style={{ font: `12px ${mono}`, color: "#4d5761" }}>#{r.index} · [{r.start},{r.end}]</span>
                </div>
                <div style={{ font: `12px ${mono}`, color: "#8b96a1", marginTop: 4 }}>“{r.substring}”</div>
                {/* params and corrections in their own labelled boxes */}
                {(Object.keys(r.parameters).length > 0 || r.corrections.length > 0) && (
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 7 }}>
                    {Object.keys(r.parameters).length > 0 && (
                      <TagBox label="params">{Object.entries(r.parameters).map(([k, v]) => <Chip key={k} text={`${k}=${v}`} color="#9aa4ae" />)}</TagBox>
                    )}
                    {r.corrections.length > 0 && (
                      <TagBox label="fixes">{r.corrections.map((x, j) => <Chip key={j} text={x} color="#9aa4ae" />)}</TagBox>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          {rejected.length > 0 && <Hdr t="REJECTED" s="tried · no match" />}
          {rejected.map((a, i) => (
            <div key={i} onClick={() => onPick(a.pattern, { pattern: a.pattern, string: a.string, matched: false, why: a.why })}
              style={{ background: tokens.bg.card2, border: `1px solid #2c2024`, borderRadius: 10, padding: "7px 12px", cursor: "pointer" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ font: `13px ${mono}`, color: "#a88" }}>{a.pattern}</span>
                <span style={{ flex: 1 }} />
                <Chip text="✕ no match" color="#c98a8a" />
              </div>
              {a.why && <div style={{ font: `11px ${mono}`, color: "#6f7883", marginTop: 4 }}>↳ {a.why}</div>}
            </div>
          ))}
        </div>

        {/* recursive parse tree */}
        <div style={{ flex: "2 1 460px", minWidth: 360 }}>
          <Hdr t="PARSE TREE · recursive" s={selMatch ? selMatch.command : selAttempt?.pattern || ""} />
          <div style={{ marginTop: 4 }}>
            {selAttempt
              ? <TreeNode a={selAttempt} depth={0} accent={selMatch ? cmdColor(selMatch.command) : "#8f86c9"} />
              : <Muted t="select a resolved command to trace its parse" />}
          </div>
          {selAttempt && <ReRun pattern={selAttempt.pattern} original={selAttempt.string} />}
        </div>
      </div>
    </div>
  );
}

// edit a parser's input and re-run just that match, in isolation. The result is MODIFIED and
// never touches the stored trace, so the full picture is not distorted by REPL pokes.
function ReRun({ pattern, original }: { pattern: string; original: string }) {
  const [val, setVal] = useState(original);
  const [res, setRes] = useState<ReparseResult | null>(null);
  const [busy, setBusy] = useState(false);
  const dirty = val !== original;
  async function go() {
    setBusy(true);
    try { setRes(await reparse(pattern, val)); } catch { setRes(null); } finally { setBusy(false); }
  }
  const m = res?.matches?.[0];
  return (
    <div style={{ marginTop: 12, padding: 11, border: `1px solid ${tokens.border.soft}`, borderRadius: 10, background: tokens.bg.card2 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ font: `11px ${mono}`, letterSpacing: ".1em", color: "#5a6570" }}>EDIT &amp; RE-RUN</span>
        <span style={{ font: `12px ${mono}`, color: "#8b96a1" }}>{pattern}</span>
        {dirty && <span style={{ font: `10px ${mono}`, color: "#e0a45a", border: "1px solid #5c451f", borderRadius: 999, padding: "0 7px" }}>MODIFIED</span>}
      </div>
      <div style={{ display: "flex", gap: 6, marginTop: 7 }}>
        <input value={val} onChange={(e) => setVal(e.target.value)} onKeyDown={(e) => e.key === "Enter" && go()}
          style={{ flex: 1, background: "#0a0e12", border: `1px solid ${tokens.border.mid}`, borderRadius: 7, padding: "6px 9px", color: tokens.text.hi, font: `13px ${mono}` }} />
        <button onClick={go} disabled={busy} style={{ background: tokens.bg.card, border: `1px solid ${tokens.impulse}`, color: tokens.impulse, borderRadius: 7, padding: "6px 11px", font: `13px ${mono}`, cursor: "pointer" }}>{busy ? "…" : "⟳ run"}</button>
        {dirty && <button onClick={() => { setVal(original); setRes(null); }} style={{ background: tokens.bg.card, border: `1px solid ${tokens.border.mid}`, color: tokens.text.mid, borderRadius: 7, padding: "6px 9px", font: `13px ${mono}`, cursor: "pointer" }}>reset</button>}
      </div>
      {res && (
        <div style={{ marginTop: 8, font: `12px ${mono}` }}>
          {res.error ? <span style={{ color: "#e07a7a" }}>✕ {res.error}</span>
            : m ? <span style={{ color: "#c2cad2" }}><span style={{ color: "#7f8a94" }}>▸ out</span> “{m.substring}” {Object.entries(m.parameters).map(([k, v]) => <span key={k} style={{ color: "#9aa4ae" }}>· {k}={String(v)} </span>)}{m.corrections?.length ? <span style={{ color: "#e0a45a" }}>· fixes {m.corrections.join(", ")}</span> : null}</span>
              : <span style={{ color: "#c98a8a" }}>✕ no match for this input</span>}
        </div>
      )}
    </div>
  );
}

function Section({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ font: `12px ${mono}`, letterSpacing: ".12em", color: "#5a6570" }}>{title} <span style={{ color: "#3a444d" }}>· {hint}</span></div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>{children}</div>
    </div>
  );
}
function Hdr({ t, s }: { t: string; s: string }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
      <span style={{ font: `13px ${mono}`, letterSpacing: ".12em", color: tokens.impulse }}>{t}</span>
      <span style={{ font: `12px ${mono}`, color: "#4d5761" }}>{s}</span>
    </div>
  );
}
function Muted({ t }: { t: string }) { return <span style={{ font: `12px ${mono}`, color: "#4d5761" }}>{t}</span>; }

function fmtParams(p: Record<string, string>): string {
  const e = Object.entries(p);
  return e.length ? "(" + e.map(([k, v]) => `${k}=${v}`).join(", ") + ")" : "";
}

// pull the Object class of a parameter out of a pattern, e.g. "weather in $city:Word" -> "Word"
function paramType(pattern: string, name: string): string | null {
  const m = pattern.match(new RegExp("\\$" + name + ":(\\w+)"));
  return m ? m[1] : null;
}

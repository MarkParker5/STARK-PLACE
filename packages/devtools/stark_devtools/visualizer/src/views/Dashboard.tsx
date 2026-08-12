import { useMemo, useState } from "react";
import type { Step } from "../schema";
import { groupColor, tokens } from "../tokens";

// Design 12a: a fixed CommandsContext "bus" on the left that every stage docks into, and four
// numbered bands on the right (boundary-in → processors → command execution → boundary-out).
// Pattern-match calls collapse inside search; command execution binds dispatch+command+respond
// per command.

function payloadLine(obj: Record<string, any>): string {
  return Object.entries(obj)
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("  ·  ");
}

const BANDS = [
  { n: 1, title: "VoiceAssistant ▸ STT · boundary in", socket: "process_string(txt) ◂" },
  { n: 2, title: "Processors · corrections · NER · search", socket: "process() ▸  results(m[]) ◂" },
  { n: 3, title: "Command execution · each ▸ its own response", socket: "run_command(cmd,p) ▸  respond(r) ◂" },
  { n: 4, title: "VoiceAssistant ▸ TTS · boundary out", socket: "reply(r[]) ▸" },
];

function bandOf(s: Step): number {
  if (s.group === "io_in") return 1;
  if (s.symbol === "CommandsContext.process_string") return 1;
  if (s.group === "execution") return 3;
  if (s.group === "io_out") return 4;
  if (s.label === "handle response" || s.label === "push context") return 4;
  return 2;
}

const SOCKETS = [
  ["◂", "process_string(txt)", "in"],
  ["▸", "process(req)", "out"],
  ["◂", "results(m[])", "in"],
  ["▸", "run_command(cmd,p)", "out"],
  ["◂", "respond(r)", "in"],
  ["▸", "reply(r[])", "out"],
];

interface Props {
  steps: Step[];
  index: number;
  overview: boolean;
  onPick: (s: Step) => void;
}

function cmdName(s: Step): string | null {
  const c = s.input?.command || s.output?.command;
  return typeof c === "string" ? c.split(".").pop()! : null;
}

export function Dashboard({ steps, index, overview, onPick }: Props) {
  const [openMatches, setOpenMatches] = useState(false);
  const [openDict, setOpenDict] = useState(false);

  const bands = useMemo(() => {
    const map = new Map<number, { step: Step; i: number }[]>();
    steps.forEach((step, i) => {
      const b = bandOf(step);
      if (!map.has(b)) map.set(b, []);
      map.get(b)!.push({ step, i });
    });
    return map;
  }, [steps]);

  const registry = useMemo(() => {
    const cmds = new Set<string>();
    steps.forEach((s) => {
      const c = cmdName(s);
      if (c) cmds.add(c);
      (s.output?.results as any[])?.forEach?.((r) => r?.command && cmds.add(String(r.command).split(".").pop()!));
    });
    return [...cmds];
  }, [steps]);

  return (
    <div style={{ display: "flex", gap: 14, padding: 6, minHeight: "100%" }}>
      {/* CommandsContext bus */}
      <aside style={{ width: 220, flexShrink: 0 }}>
        <div style={{ position: "sticky", top: 0, background: tokens.bg.card2, border: `1px solid ${tokens.border.soft}`, borderRadius: 12, padding: 14 }}>
          <div style={{ fontFamily: tokens.font.display, fontWeight: 600, fontSize: 16, color: tokens.text.hi }}>CommandsContext</div>
          <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".1em", marginTop: 3 }}>THE HUB · DISPATCHER</div>
          <div style={{ height: 1, background: tokens.border.faint, margin: "12px 0" }} />
          <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".1em", marginBottom: 8 }}>I/O SOCKETS</div>
          {SOCKETS.map(([arrow, name, dir], i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 7 }}>
              <span style={{ color: tokens.text.faint, fontFamily: tokens.font.mono, fontSize: 14, width: 12 }}>{arrow}</span>
              <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.mid }}>{name}</span>
            </div>
          ))}
          <div style={{ height: 1, background: tokens.border.faint, margin: "12px 0" }} />
          <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".1em", marginBottom: 6 }}>CONTEXT TREE</div>
          <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.mid, lineHeight: 1.7 }}>
            root · {registry.length || "…"} cmds
            <br />
            <span style={{ color: tokens.text.mid }}>▸ music · playing</span>
          </div>
          {registry.length > 0 && (
            <>
              <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".1em", margin: "12px 0 6px" }}>REGISTRY</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                {registry.map((c) => (
                  <span key={c} style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.mid, border: `1px solid ${tokens.border.mid}`, borderRadius: 999, padding: "2px 9px" }}>
                    {c}
                  </span>
                ))}
              </div>
            </>
          )}
        </div>
      </aside>

      {/* bands */}
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
        {BANDS.map((band) => {
          const items = bands.get(band.n) ?? [];
          const bandActive = !overview && items.some((it) => it.i === index);
          // calm: a single muted stripe per band; the only real accent is the active (impulse) state
          const accent = bandActive ? tokens.impulse : "#3a4550";
          return (
            <section
              key={band.n}
              style={{
                border: `1px solid ${bandActive ? tokens.impulse : tokens.border.soft}`,
                borderLeft: `3px solid ${accent}`,
                borderRadius: 12,
                background: bandActive ? tokens.bg.active : tokens.bg.card,
                padding: "12px 16px",
                opacity: items.length ? 1 : 0.4,
                transition: "border-color .3s, background .3s",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, marginBottom: items.length ? 10 : 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: bandActive ? tokens.impulse : tokens.text.muted, border: `1px solid ${bandActive ? tokens.impulse : tokens.border.mid}`, borderRadius: 999, width: 20, height: 20, display: "inline-flex", alignItems: "center", justifyContent: "center" }}>{band.n}</span>
                  <span style={{ fontFamily: tokens.font.display, fontWeight: 600, fontSize: 14, color: tokens.text.hi, textTransform: "uppercase", letterSpacing: ".04em" }}>{band.title}</span>
                </div>
                <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".06em" }}>{band.socket}</span>
              </div>

              {band.n === 2
                ? renderProcessors(items, index, overview, onPick, openMatches, setOpenMatches, openDict, setOpenDict)
                : band.n === 3
                ? renderExecution(items, index, overview, onPick)
                : items.map(({ step, i }) => <StepRow key={step.seq} step={step} active={!overview && i === index} onPick={onPick} />)}
            </section>
          );
        })}
      </div>
    </div>
  );
}

function renderProcessors(
  items: { step: Step; i: number }[],
  index: number,
  overview: boolean,
  onPick: (s: Step) => void,
  openMatches: boolean,
  setOpenMatches: (b: boolean) => void,
  openDict: boolean,
  setOpenDict: (b: boolean) => void
) {
  const matches = items.filter((it) => it.step.label === "pattern match");
  const dicts = items.filter((it) => it.step.label === "dictionary lookup");
  const rest = items.filter((it) => it.step.label !== "pattern match" && it.step.label !== "dictionary lookup");
  const matchActive = !overview && matches.some((m) => m.i === index);
  const dictActive = !overview && dicts.some((m) => m.i === index);
  return (
    <>
      {rest.map(({ step, i }) => (
        <div key={step.seq}>
          <StepRow step={step} active={!overview && i === index} onPick={onPick} />
          {step.label === "corrections" && dicts.length > 0 && (
            <div style={{ marginLeft: 14, marginTop: 4 }}>
              <button onClick={() => setOpenDict(!openDict)} style={{ background: "none", border: "none", cursor: "pointer", fontFamily: tokens.font.mono, fontSize: 13, color: dictActive ? tokens.impulse : tokens.text.muted, padding: "2px 0" }}>
                {openDict ? "▾" : "▸"} {dicts.length} dictionary lookup{dicts.length > 1 ? "s" : ""} {dictActive ? "· active" : ""}
              </button>
              {openDict && dicts.map(({ step: d, i }) => <StepRow key={d.seq} step={d} active={!overview && i === index} onPick={onPick} compact />)}
            </div>
          )}
          {step.label === "search" && matches.length > 0 && (
            <div style={{ marginLeft: 14, marginTop: 4 }}>
              <button onClick={() => setOpenMatches(!openMatches)} style={{ background: "none", border: "none", cursor: "pointer", fontFamily: tokens.font.mono, fontSize: 13, color: matchActive ? tokens.impulse : tokens.text.muted, padding: "2px 0" }}>
                {openMatches ? "▾" : "▸"} {matches.length} pattern match calls {matchActive ? "· active" : ""}
              </button>
              {openMatches && matches.map(({ step: m, i }) => <StepRow key={m.seq} step={m} active={!overview && i === index} onPick={onPick} compact />)}
            </div>
          )}
        </div>
      ))}
    </>
  );
}

function StepRowByIndex({ step, onPick }: { step: Step; i: number; onPick?: (s: Step) => void }) {
  return <StepRow step={step} active={false} onPick={onPick ?? (() => {})} compact />;
}

function renderExecution(items: { step: Step; i: number }[], index: number, overview: boolean, onPick: (s: Step) => void) {
  // bind dispatch + command + respond per command
  const groups = new Map<string, { step: Step; i: number }[]>();
  const order: string[] = [];
  let current = "command";
  for (const it of items) {
    const name = cmdName(it.step);
    if (name) current = name;
    if (!groups.has(current)) {
      groups.set(current, []);
      order.push(current);
    }
    groups.get(current)!.push(it);
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {order.map((name) => {
        const grp = groups.get(name)!;
        const active = !overview && grp.some((g) => g.i === index);
        return (
          <div key={name} style={{ border: `1px solid ${active ? tokens.impulse : tokens.border.faint}`, borderRadius: 10, padding: "8px 10px", background: active ? "rgba(246,180,90,.05)" : "transparent" }}>
            <div style={{ fontFamily: tokens.font.mono, fontSize: 14, color: active ? tokens.impulse : tokens.text.mid, marginBottom: 6 }}>▸ {name}</div>
            {grp.map(({ step, i }) => (
              <StepRow key={step.seq} step={step} active={!overview && i === index} onPick={onPick} compact />
            ))}
          </div>
        );
      })}
    </div>
  );
}

function StepRow({ step, active, onPick, compact }: { step: Step; active: boolean; onPick: (s: Step) => void; compact?: boolean }) {
  const color = groupColor(step.group);
  return (
    <div
      onClick={() => onPick(step)}
      style={{
        display: "flex",
        gap: 10,
        padding: compact ? "5px 8px" : "7px 10px",
        marginTop: 6,
        borderRadius: 8,
        background: active ? "rgba(246,180,90,.08)" : "transparent",
        border: `1px solid ${active ? tokens.impulse : tokens.border.faint}`,
        cursor: "pointer",
      }}
    >
      <span style={{ width: 3, alignSelf: "stretch", background: active ? tokens.impulse : color, borderRadius: 2, opacity: active ? 1 : 0.35 }} />
      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
          <span style={{ fontFamily: tokens.font.display, fontWeight: 600, fontSize: compact ? 13 : 14, color: active ? tokens.impulse : tokens.text.hi }}>{step.label}</span>
          <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, whiteSpace: "nowrap" }}>
            {step.symbol}
            {step.dur_ns != null ? ` · ${(step.dur_ns / 1000).toFixed(0)}µs` : ""}
          </span>
        </div>
        {Object.keys(step.input).length > 0 && <div style={line}><span style={tag}>in</span>{payloadLine(step.input)}</div>}
        {Object.keys(step.output).length > 0 && <div style={line}><span style={tag}>out</span>{payloadLine(step.output)}</div>}
      </div>
    </div>
  );
}

const line: React.CSSProperties = { fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.mid, marginTop: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" };
const tag: React.CSSProperties = { color: tokens.text.faint, marginRight: 6, fontSize: 13, textTransform: "uppercase", letterSpacing: ".08em" };

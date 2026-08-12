import type { Replay } from "../store/useReplay";
import { tokens } from "../tokens";

const SPEEDS = [0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20];

export function Transport({ replay, label, onLabelClick }: { replay: Replay; label?: string; onLabelClick?: () => void }) {
  const btn: React.CSSProperties = {
    background: tokens.bg.card2,
    border: `1px solid ${tokens.border.mid}`,
    color: tokens.text.mid,
    borderRadius: 8,
    padding: "6px 12px",
    fontFamily: tokens.font.mono,
    fontSize: 16,
    cursor: "pointer",
  };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
      <button
        style={{
          ...btn,
          borderColor: replay.overview ? tokens.impulse : tokens.border.mid,
          color: replay.overview ? tokens.impulse : tokens.text.mid,
          background: replay.overview ? "rgba(246,180,90,.12)" : tokens.bg.card2,
        }}
        onClick={replay.goOverview}
        title="overview — highlight everything this request touched"
      >
        ◉ overview
      </button>
      <button style={btn} onClick={replay.prev} title="previous">◀</button>
      <button
        style={{ ...btn, borderColor: replay.playing ? tokens.impulse : tokens.border.mid, color: replay.playing ? tokens.impulse : tokens.text.mid }}
        onClick={replay.toggle}
      >
        {replay.playing ? "❚❚" : "▶"}
      </button>
      <button style={btn} onClick={replay.next} title="next">▶</button>
      <button
        style={{ ...btn, borderColor: replay.loop ? tokens.impulse : tokens.border.mid, color: replay.loop ? tokens.impulse : tokens.text.mid, background: replay.loop ? "rgba(246,180,90,.12)" : tokens.bg.card2 }}
        onClick={() => replay.setLoop((l) => !l)}
        title="auto-restart / loop playback"
      >
        ↻
      </button>

      <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 6 }}>
        <span style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted }}>speed</span>
        <select
          value={replay.speed}
          onChange={(e) => replay.setSpeed(Number(e.target.value))}
          style={{ ...btn, padding: "6px 8px", fontSize: 14, color: tokens.impulse, borderColor: tokens.border.mid }}
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s} style={{ background: tokens.bg.card }}>
              {s}×
            </option>
          ))}
        </select>
      </div>

      <input
        type="range"
        min={0}
        max={Math.max(0, replay.length - 1)}
        value={replay.index}
        onChange={(e) => replay.setIndex(Number(e.target.value))}
        style={{ flex: 1, minWidth: 120, accentColor: tokens.impulse }}
      />
      <span onClick={onLabelClick} title={onLabelClick ? "click for per-function durations" : undefined}
        style={{ fontFamily: tokens.font.mono, fontSize: 14, color: onLabelClick ? tokens.text.mid : tokens.text.muted, whiteSpace: "nowrap", cursor: onLabelClick ? "pointer" : "default", textDecoration: onLabelClick ? "underline dotted" : "none", textUnderlineOffset: 3 }}>
        {label ?? `step ${replay.index + 1} / ${replay.length}`} {onLabelClick ? "⏱" : ""}
      </span>
    </div>
  );
}

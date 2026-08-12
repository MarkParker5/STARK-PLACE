import type { GraphNode } from "../schema";
import { tokens } from "../tokens";

function JsonView({ value }: { value: any }) {
  return (
    <pre
      style={{
        margin: 0,
        fontFamily: tokens.font.mono,
        fontSize: 14,
        lineHeight: 1.5,
        color: tokens.text.hi,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
      }}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function NodeInspector({ node, onClose, onBlacklist }: { node: GraphNode; onClose: () => void; onBlacklist?: (id: string) => void }) {
  const rows: [string, any][] = [
    ["module", node.module],
    ["weight", node.weight],
    ["calls (traced)", node.calls],
    ["public methods", node.public],
    ["private methods", node.private],
    ["OOP relations", node.relations],
    ["active in trace", node.active ? "yes" : "no"],
  ];
  return (
    <div style={panel}>
      <div style={header}>
        <span style={{ fontFamily: tokens.font.display, fontWeight: 600, color: tokens.text.hi }}>{node.label}</span>
        <div style={{ display: "flex", gap: 8 }}>
          {onBlacklist && (
            <button style={{ ...closeBtn, fontSize: 13, color: "#e07a7a" }} onClick={() => onBlacklist(node.id)} title="hide from display and timeline">
              ⊘ blacklist
            </button>
          )}
          <button style={closeBtn} onClick={onClose}>✕</button>
        </div>
      </div>
      <table style={{ borderCollapse: "collapse", width: "100%" }}>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td style={{ ...cell, color: tokens.text.muted, fontFamily: tokens.font.mono, fontSize: 13, textTransform: "uppercase" }}>{k}</td>
              <td style={{ ...cell, color: tokens.text.hi, fontFamily: tokens.font.mono, fontSize: 14 }}>{String(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PayloadInspector({ title, input, output, onClose }: { title: string; input: any; output: any; onClose: () => void }) {
  return (
    <div style={panel}>
      <div style={header}>
        <span style={{ fontFamily: tokens.font.display, fontWeight: 600, color: tokens.text.hi }}>{title}</span>
        <button style={closeBtn} onClick={onClose}>
          ✕
        </button>
      </div>
      <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".1em", marginTop: 6 }}>INPUT</div>
      <JsonView value={input} />
      {output && Object.keys(output).length > 0 && (
        <>
          <div style={{ fontFamily: tokens.font.mono, fontSize: 13, color: tokens.text.muted, letterSpacing: ".1em", marginTop: 10 }}>OUTPUT</div>
          <JsonView value={output} />
        </>
      )}
    </div>
  );
}

const panel: React.CSSProperties = {
  background: tokens.bg.card2,
  border: `1px solid ${tokens.border.strong}`,
  borderRadius: tokens.radius.card,
  padding: 14,
  maxHeight: "100%",
  overflow: "auto",
  boxShadow: "8px 8px 0 rgba(7,10,13,.45)",
};
const header: React.CSSProperties = { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 };
const closeBtn: React.CSSProperties = { background: "none", border: "none", color: tokens.text.muted, cursor: "pointer", fontSize: 17 };
const cell: React.CSSProperties = { padding: "3px 6px", borderBottom: `1px solid ${tokens.border.faint}` };

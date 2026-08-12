// Design tokens from the STARK Visualizer handoff spec.
export const tokens = {
  bg: {
    page: "#070a0d",
    page2: "#090c10",
    card: "#0b0f13",
    card2: "#0c1116",
    active: "#150f07",
  },
  border: {
    faint: "#141b22",
    soft: "#1a232c",
    mid: "#232e38",
    strong: "#33414c",
    active: "#f6b45a",
  },
  // group / accent colors
  group: {
    io_in: "#4a90d9",
    io_out: "#4a90d9",
    engine: "#c9923f",
    processors: "#6cc08a",
    phonetics: "#e07aa6",
    matching: "#8f86c9",
    execution: "#8fd6a8",
    other: "#7fd3c4",
  } as Record<string, string>,
  impulse: "#f6b45a",
  text: {
    hi: "#e2e8ee",
    mid: "#8b96a1",
    muted: "#586472",
    faint: "#4d5761",
  },
  radius: { tile: 10, card: 12, group: 26, pill: 999 },
  font: {
    display: "'Space Grotesk', system-ui, sans-serif",
    mono: "'Space Mono', ui-monospace, monospace",
  },
};

export function groupColor(group: string): string {
  return tokens.group[group] ?? tokens.group.other;
}

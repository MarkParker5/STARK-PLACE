import type { Graph, GraphNode, Step } from "./schema";
import { groupColor } from "./tokens";

export function classOf(symbol: string): string {
  return symbol.split(".", 1)[0];
}

// Top-level stark subpackage -> color (for the auto/brain graph).
const MODULE_COLORS: Record<string, string> = {
  core: "#c9923f",
  "core/processors": "#6cc08a",
  "core/patterns": "#8f86c9",
  "core/types": "#a79ce0",
  voice_assistant: "#4a90d9",
  interfaces: "#7fd3c4",
  tools: "#e07aa6",
  models: "#c9b06a",
  general: "#586472",
};

export function moduleColor(module: string): string {
  const parts = module.split("/");
  const two = parts.slice(0, 2).join("/");
  return MODULE_COLORS[two] ?? MODULE_COLORS[parts[0]] ?? "#7fd3c4";
}

export function moduleGroup(module: string): string {
  const parts = module.split("/");
  const two = parts.slice(0, 2).join("/");
  if (MODULE_COLORS[two]) return two;
  return parts[0] ?? "other";
}

// Build a small SEMANTIC wiring graph purely from the curated steps (no hardcoded coordinates).
// Nodes = the classes that took part; edges = the order the story moved between them.
export function stepsToGraph(steps: Step[]): Graph {
  const nodes = new Map<string, GraphNode>();
  const edges = new Map<string, { source: string; target: string; weight: number }>();

  const add = (cls: string, group: string) => {
    const n = nodes.get(cls);
    if (n) n.calls += 1;
    else
      nodes.set(cls, {
        id: cls,
        label: cls,
        module: group,
        weight: 1,
        calls: 1,
        public: 0,
        private: 0,
        relations: 0,
        active: true,
      });
  };

  let prev: string | null = null;
  for (const s of steps) {
    const cls = classOf(s.symbol);
    add(cls, s.group);
    if (prev && prev !== cls) {
      const key = `${prev}->${cls}`;
      const e = edges.get(key);
      if (e) e.weight += 1;
      else edges.set(key, { source: prev, target: cls, weight: 1 });
    }
    prev = cls;
  }
  for (const n of nodes.values()) n.weight = n.calls;

  return {
    nodes: [...nodes.values()],
    edges: [...edges.values()].map((e) => ({ ...e, kind: "call" as const })),
    meta: { class_count: nodes.size, traced: true, weights: {} },
  };
}

// group -> color for the semantic wiring graph (uses step groups)
export function stepGroupColor(group: string): string {
  return groupColor(group);
}

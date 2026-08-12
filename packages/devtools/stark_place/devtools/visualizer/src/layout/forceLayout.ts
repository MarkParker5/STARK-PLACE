import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceRadial,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import type { Graph, GraphEdge, GraphNode } from "../schema";

export interface PositionedNode extends SimulationNodeDatum {
  id: string;
  node: GraphNode;
  r: number;
  x: number;
  y: number;
}
export interface PositionedEdge {
  source: string;
  target: string;
  edge: GraphEdge;
}

export interface LayoutOpts {
  spread?: number;
  spacing?: number; // extra space between neighbouring nodes (collision radius)
  mode?: "gravity" | "zones";
  groupOf?: (n: GraphNode) => string;
  centralityOf?: (n: GraphNode) => number; // 0..1; 1 = pulled to the centre (main modules), 0 = outer ring (tools)
}

export function radiusFor(weight: number, maxWeight: number): number {
  const t = maxWeight > 0 ? weight / maxWeight : 0;
  return 10 + Math.sqrt(t) * 26;
}

// Two layout rules share the same forces; `mode` swaps how nodes are anchored:
//   * "gravity" — a single radial well: heavier nodes pulled to the centre, everything clusters.
//   * "zones"   — each module GROUP gets its own anchor on a ring; nodes are pulled to their group's
//                 anchor, so the graph separates into readable zones (a multi-tree feel) instead of
//                 one glued blob. Heavier nodes sit nearer their zone's core.
// `spread` scales repulsion + link distance so the whole thing blooms out or packs in.
export function layoutGraph(graph: Graph, width: number, height: number, opts: LayoutOpts = {}): {
  nodes: PositionedNode[];
  edges: PositionedEdge[];
} {
  const spread = opts.spread ?? 1.3;
  const spacing = opts.spacing ?? 1;
  const mode = opts.mode ?? "gravity";
  const groupOf = opts.groupOf ?? (() => "all");
  const centralityOf = opts.centralityOf;

  const maxWeight = Math.max(1, ...graph.nodes.map((n) => n.weight));
  const cx = width / 2;
  const cy = height / 2;
  const maxRadius = Math.min(width, height) / 2 - 30;

  // group anchors on a ring (zones mode)
  const groups = [...new Set(graph.nodes.map((n) => groupOf(n)))].sort();
  const groupAnchor = new Map<string, { x: number; y: number }>();
  const zoneRing = maxRadius * 0.62;
  groups.forEach((g, i) => {
    const a = (i / Math.max(1, groups.length)) * Math.PI * 2 - Math.PI / 2;
    groupAnchor.set(g, { x: cx + Math.cos(a) * zoneRing, y: cy + Math.sin(a) * zoneRing });
  });

  const nodes: PositionedNode[] = graph.nodes.map((n, i) => {
    const anchor = mode === "zones" ? groupAnchor.get(groupOf(n))! : { x: cx, y: cy };
    // golden-angle seeding around the anchor
    const ga = i * 2.399963229728653;
    const rad = Math.sqrt((i + 0.5) / graph.nodes.length) * (mode === "zones" ? maxRadius * 0.28 : maxRadius);
    return {
      id: n.id,
      node: n,
      r: radiusFor(n.weight, maxWeight),
      x: anchor.x + Math.cos(ga) * rad,
      y: anchor.y + Math.sin(ga) * rad,
    };
  });
  const byId = new Map(nodes.map((n) => [n.id, n]));

  const linkMap = new Map<string, { source: string; target: string; edge: GraphEdge; w: number }>();
  for (const e of graph.edges) {
    if (!byId.has(e.source) || !byId.has(e.target)) continue;
    const key = [e.source, e.target].sort().join("|");
    const cur = linkMap.get(key);
    if (cur) cur.w += e.weight;
    else linkMap.set(key, { source: e.source, target: e.target, edge: e, w: e.weight });
  }
  const links = [...linkMap.values()] as any[];

  const sim = forceSimulation(nodes)
    .force("charge", forceManyBody().strength((d: any) => (-110 - (d as PositionedNode).r * 6) * spread))
    .force(
      "link",
      forceLink(links)
        .id((d: any) => (d as PositionedNode).id)
        .distance((l: any) => (150 - Math.min(80, l.w * 6)) * spread)
        .strength((l: any) => Math.min(1, 0.12 + l.w * 0.05))
    )
    .force("collide", forceCollide((d: any) => ((d as PositionedNode).r + 8) * spacing));

  if (mode === "zones") {
    // pull each node to its group's anchor -> zones; heavier nodes nearer the anchor core
    sim
      .force("zoneX", forceX((d: any) => groupAnchor.get(groupOf((d as PositionedNode).node))!.x).strength(0.22))
      .force("zoneY", forceY((d: any) => groupAnchor.get(groupOf((d as PositionedNode).node))!.y).strength(0.22))
      .force(
        "zoneRadial",
        forceRadial(
          (d: any) => (1 - (d as PositionedNode).node.weight / maxWeight) * 60,
          (d: any) => groupAnchor.get(groupOf((d as PositionedNode).node))!.x,
          (d: any) => groupAnchor.get(groupOf((d as PositionedNode).node))!.y
        ).strength(0.25)
      );
  } else {
    // central-abstraction: if centralityOf is given, main modules (high centrality) sit near the
    // centre and their low-level tools push to the outer ring — a clearer lateral structure.
    const targetRadius = (n: PositionedNode) =>
      centralityOf ? maxRadius * (1 - 0.92 * centralityOf(n.node)) : maxRadius * (1 - 0.7 * (n.node.weight / maxWeight));
    sim
      .force("radial", forceRadial((d: any) => targetRadius(d as PositionedNode), cx, cy).strength(0.7))
      .force("center", forceCenter(cx, cy).strength(0.03))
      .force("x", forceX(cx).strength(0.015))
      .force("y", forceY(cy).strength(0.015));
  }

  sim.stop();
  for (let i = 0; i < 420; i++) sim.tick();

  const edges: PositionedEdge[] = links.map((l: any) => ({
    source: (l.source as PositionedNode).id,
    target: (l.target as PositionedNode).id,
    edge: l.edge,
  }));

  return { nodes, edges };
}

// Mirror of profiler/schema.py (ProfileEvent) + the server's trace bundle. The ONLY coupling to STARK.

export interface ProfileEvent {
  trace_id: string;
  seq: number;
  t_ns: number;
  phase: "call" | "return" | "error";
  symbol: string;
  module: string;
  depth: number;
  thread: number;
  dur_ns: number | null;
  data: Record<string, any>;
}

export interface Step {
  seq: number;
  trace_id: string;
  symbol: string;
  label: string;
  group: string;
  depth: number;
  input: Record<string, any>;
  output: Record<string, any>;
  dur_ns: number | null;
  error: string | null;
}

export interface GraphNode {
  id: string;
  label: string;
  module: string;
  weight: number;
  calls: number;
  public: number;
  private: number;
  relations: number;
  active: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  kind: "inherit" | "compose" | "call";
  weight: number;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta: { class_count: number; traced: boolean; weights: Record<string, number> };
}

export interface TraceBundle {
  utterance: string;
  events: ProfileEvent[];
  steps: Step[];
  graph: Graph;
}

// --- API client -------------------------------------------------------------

export async function fetchDemo(): Promise<TraceBundle> {
  const r = await fetch("/api/demo");
  if (!r.ok) throw new Error("demo failed");
  return r.json();
}

export async function submitUtterance(text: string): Promise<TraceBundle> {
  const r = await fetch("/api/utterance", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error("utterance failed");
  return r.json();
}

// Isolated single-parser re-run (Matching page "edit & re-run"). Never persists.
export interface ReparseResult {
  pattern: string;
  string: string;
  matches: { substring: string; start: number; end: number; parameters: Record<string, any>; corrections: string[]; corrected_string: string }[];
  error: string | null;
}
export async function reparse(pattern: string, string: string): Promise<ReparseResult> {
  const r = await fetch("/api/reparse", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pattern, string }) });
  if (!r.ok) throw new Error("reparse failed");
  return r.json();
}

// Fallback: the bundled sample (works with no server running).
export async function fetchSample(): Promise<TraceBundle> {
  const r = await fetch("sample.json");
  return r.json();
}

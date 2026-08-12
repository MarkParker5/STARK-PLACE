// Derive the "what did this request touch" model from a trace, shared by all views.
import type { Step } from "./schema";
import { classOf } from "./graph";

export function edgeKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

export interface Touched {
  nodes: Set<string>; // class names touched during the request
  edgeKeys: Set<string>; // undirected edge keys touched
  // ordered, de-duplicated, self-transitions removed — the playback timeline of real links
  edgeSequence: { from: string; to: string; stepIndex: number }[];
}

// A timeline frame for the graph views: consecutive steps that map to the SAME node collapse into
// one frame (so pattern-match×8 don't waste 8 slots and don't look "invisible", and the commands
// dispatched together read as one parallel step). Blacklisted classes are dropped from both the
// display and the timeline.
export interface Frame {
  primary: string; // the class/node this frame activates
  label: string;
  count: number; // how many raw steps collapsed here
  stepIdxs: number[];
}

export function buildFrames(steps: Step[], blacklist: Set<string>): Frame[] {
  const frames: Frame[] = [];
  steps.forEach((s, i) => {
    const cls = classOf(s.symbol);
    if (blacklist.has(cls)) return;
    const last = frames[frames.length - 1];
    if (last && last.primary === cls) {
      last.count++;
      last.stepIdxs.push(i);
    } else {
      frames.push({ primary: cls, label: s.label, count: 1, stepIdxs: [i] });
    }
  });
  return frames;
}

// Which classes/links the current request actually exercised. Untouched modules stay muted;
// self-transitions (same node twice in a row) are NOT links and are skipped so they don't take a
// place in the playback timeline.
export function deriveTouched(steps: Step[], blacklist?: Set<string>): Touched {
  const nodes = new Set<string>();
  const edgeKeys = new Set<string>();
  const edgeSequence: { from: string; to: string; stepIndex: number }[] = [];

  let prev: string | null = null;
  steps.forEach((s, i) => {
    const cls = classOf(s.symbol);
    if (blacklist?.has(cls)) return;
    nodes.add(cls);
    if (prev && prev !== cls) {
      edgeKeys.add(edgeKey(prev, cls));
      edgeSequence.push({ from: prev, to: cls, stepIndex: i });
    }
    prev = cls;
  });

  return { nodes, edgeKeys, edgeSequence };
}

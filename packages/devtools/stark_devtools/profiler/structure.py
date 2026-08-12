"""Static structure of STARK, for the auto-built ("purely generic") brain graph.

Nothing here hardcodes stark's modules. We parse every `.py` file in the stark package with `ast`
(no imports — so optional deps like vosk/torch/spacy never load) and extract, per class:
  * public vs private method counts (public methods weigh more — they are the real surface);
  * OOP relations — inheritance (bases) and composition (annotations referencing other stark types).

`build_graph(events)` merges this static structure with DYNAMIC call counts from a trace, producing
node weights per the rule: weight rises with call count, method count (public > private) and relation
count. Edges carry inheritance/composition plus call-adjacency from the trace. The visualizer force-
lays it out: heavier nodes pulled to the centre, more-linked nodes pulled together.
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .capture import STARK_DIR
from .schema import CALL, ProfileEvent

# weight contributions (tunable)
W_CALL = 1.0        # per dynamic call observed in the trace
W_PUBLIC = 3.0      # per public method (the real API surface)
W_PRIVATE = 1.0     # per private/dunder method
W_RELATION = 2.0    # per OOP relation (inheritance or composition)


@dataclass
class ClassInfo:
    name: str
    module: str
    public: int = 0
    private: int = 0
    bases: list[str] = field(default_factory=list)         # names, filtered to stark later
    compositions: list[str] = field(default_factory=list)  # referenced stark type names


def _iter_py_files(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _annotation_names(node: ast.AST) -> list[str]:
    """Collect bare identifier names used in a type annotation (Word, list[Command], X | None, ...)."""
    names: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.append(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.append(sub.attr)
    return names


def extract_classes(root: str = STARK_DIR) -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}
    for path in _iter_py_files(root):
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        module = os.path.relpath(path, STARK_DIR)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            info = ClassInfo(name=node.name, module=module)
            info.bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_"):
                        info.private += 1
                    else:
                        info.public += 1
                    # composition via parameter/return annotations
                    for arg in item.args.args + item.args.kwonlyargs:
                        if arg.annotation is not None:
                            info.compositions += _annotation_names(arg.annotation)
                    if item.returns is not None:
                        info.compositions += _annotation_names(item.returns)
                elif isinstance(item, ast.AnnAssign) and item.annotation is not None:
                    info.compositions += _annotation_names(item.annotation)
            # a class may be declared twice across files in theory; last wins, fine for a graph
            classes[node.name] = info
    return classes


def class_call_counts(events: list[ProfileEvent]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        if e.phase != CALL:
            continue
        # symbol is a qualname like "PatternParser.match" or "PatternParser._parse.<locals>.<lambda>"
        head = e.symbol.split(".", 1)[0]
        counts[head] += 1
    return dict(counts)


def call_edges(events: list[ProfileEvent]) -> dict[tuple[str, str], int]:
    """Directed caller→callee edges between classes, inferred from per-thread call nesting."""
    edges: dict[tuple[str, str], int] = defaultdict(int)
    # per-thread stack of class names for currently-open frames
    stacks: dict[int, list[str]] = defaultdict(list)
    for e in events:
        cls = e.symbol.split(".", 1)[0]
        if e.phase == CALL:
            stack = stacks[e.thread]
            if stack and stack[-1] != cls:
                edges[(stack[-1], cls)] += 1
            stack.append(cls)
        elif e.phase in ("return", "error"):
            stack = stacks[e.thread]
            if stack:
                stack.pop()
    return dict(edges)


def build_graph(events: list[ProfileEvent] | None = None, root: str = STARK_DIR) -> dict[str, Any]:
    """Return {nodes, edges, meta} for the auto-graph. Purely derived — no module is hardcoded."""
    classes = extract_classes(root)
    known = set(classes)
    counts = class_call_counts(events or [])
    adjacency = call_edges(events or [])

    # relation edges from static structure (only between known stark classes)
    rel_edges: dict[tuple[str, str, str], int] = defaultdict(int)
    for name, info in classes.items():
        for base in info.bases:
            if base in known and base != name:
                rel_edges[(name, base, "inherit")] += 1
        for comp in set(info.compositions):
            if comp in known and comp != name:
                rel_edges[(name, comp, "compose")] += 1

    # count relations per node (both directions) for weighting
    relation_count: dict[str, int] = defaultdict(int)
    for (a, b, _kind) in rel_edges:
        relation_count[a] += 1
        relation_count[b] += 1

    nodes = []
    for name, info in classes.items():
        calls = counts.get(name, 0)
        rels = relation_count.get(name, 0)
        weight = (
            W_CALL * calls
            + W_PUBLIC * info.public
            + W_PRIVATE * info.private
            + W_RELATION * rels
        )
        nodes.append({
            "id": name,
            "label": name,
            "module": info.module,
            "weight": round(weight, 2),
            "calls": calls,
            "public": info.public,
            "private": info.private,
            "relations": rels,
            "active": calls > 0,  # did this node participate in the traced utterance?
        })

    edges = []
    for (a, b, kind), n in rel_edges.items():
        edges.append({"source": a, "target": b, "kind": kind, "weight": n})
    for (a, b), n in adjacency.items():
        if a in known and b in known:
            edges.append({"source": a, "target": b, "kind": "call", "weight": n})

    nodes.sort(key=lambda n: n["weight"], reverse=True)
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "class_count": len(nodes),
            "traced": bool(events),
            "weights": {"call": W_CALL, "public": W_PUBLIC, "private": W_PRIVATE, "relation": W_RELATION},
        },
    }


def main() -> None:
    import json
    import sys

    events = None
    argv = sys.argv[1:]
    if argv and argv[0].endswith(".jsonl"):
        from .replay import load
        events = load(argv[0])
    graph = build_graph(events)
    if "--json" in argv:
        print(json.dumps(graph, indent=2))
        return
    print(f"{graph['meta']['class_count']} classes, {len(graph['edges'])} edges "
          f"({'traced' if graph['meta']['traced'] else 'static only'})\n")
    print("top nodes by weight:")
    for n in graph["nodes"][:20]:
        flag = "●" if n["active"] else "○"
        print(f"  {flag} {n['label']:34} w={n['weight']:6.1f}  "
              f"calls={n['calls']:3}  pub={n['public']:2} priv={n['private']:2} rel={n['relations']:2}  {n['module']}")


if __name__ == "__main__":
    main()

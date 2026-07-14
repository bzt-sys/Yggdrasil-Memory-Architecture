from __future__ import annotations

"""
Paper-oriented graph visualization helpers for Yggdrasil exports.

Input:
    graph.json exported by :export-evidence

Output:
    filtered_graph.json
    filtered_graph.dot
    filtered_graph_summary.md
    filtered_graph.svg / filtered_graph.png if Graphviz is installed

The goal is not to render the entire substrate. The goal is to create readable
figures that show governance topology: episodes, selected events, meaningful
features, and promoted governance concepts.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


NOISY_FEATURE_VALUES: Set[str] = {
    "a", "an", "and", "are", "as", "ask", "based", "be", "by", "command",
    "for", "from", "here", "how", "if", "imp", "in", "interface", "is", "it",
    "line", "of", "on", "or", "that", "the", "this", "to", "use", "when", "with",
}

ALWAYS_KEEP_ENTITY_VALUES: Set[str] = {
    "python", "linux", "networkx", "mistral-7b-instruct-v0", "mistral-7b-instruct-v0.3",
}

ALWAYS_KEEP_TASK_VALUES: Set[str] = {
    "replay", "context", "decay", "compression", "pruning", "adjudication",
}

EDGE_KEEP_BY_MODE: Dict[str, Set[str]] = {
    "paper": {"contains", "has_feature", "promotes_to"},
    "constraints": {"has_feature", "promotes_to"},
    "concepts": {"contains", "has_feature", "promotes_to"},
    "full-lite": {"contains", "has_feature", "promotes_to", "rel:cooccur"},
}

NODE_ORDER = {
    "episode": 0,
    "event": 1,
    "feature": 2,
    "concept": 3,
    "outcome": 4,
}


def load_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def node_type(node_id: str, node: Optional[Dict[str, Any]] = None) -> str:
    if node and node.get("type"):
        return str(node["type"])
    return str(node_id).split(":", 1)[0]


def raw_key(node_id: str, node: Optional[Dict[str, Any]] = None) -> str:
    if node and node.get("key") is not None:
        return str(node["key"])
    if "::" in node_id:
        return node_id.split("::", 1)[1]
    if ":" in node_id:
        return node_id.split(":", 1)[1]
    return node_id


def feature_kind_and_value(node_id: str) -> Tuple[str, str]:
    # expected: feature:entity::python
    if not node_id.startswith("feature:"):
        return "", ""
    rest = node_id[len("feature:"):]
    if "::" in rest:
        kind, value = rest.split("::", 1)
        return kind.lower(), value.strip().lower()
    return "", rest.strip().lower()


def truncate(text: str, n: int = 46) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= n:
        return text
    return text[: max(0, n - 1)].rstrip() + "…"


def de_lesson(text: str) -> str:
    text = str(text)
    text = re.sub(r"^lesson:\s*", "", text, flags=re.I).strip()
    return text


def is_artifact_like(value: str) -> bool:
    v = value.lower()
    return bool(
        "-" in v
        or "_" in v
        or re.search(r"\d", v)
        or v in ALWAYS_KEEP_ENTITY_VALUES
    )


def is_noisy_feature(node_id: str) -> bool:
    kind, value = feature_kind_and_value(node_id.lower())
    if kind in {"token", "bigram"}:
        return True
    if kind == "entity":
        return value in NOISY_FEATURE_VALUES
    return False


def keep_feature(node_id: str, *, mode: str) -> bool:
    kind, value = feature_kind_and_value(node_id.lower())

    if kind == "constraint":
        return True

    if kind == "task":
        return value in ALWAYS_KEEP_TASK_VALUES

    if kind == "entity":
        if value in NOISY_FEATURE_VALUES:
            return False
        if mode == "full-lite":
            return is_artifact_like(value) or value in ALWAYS_KEEP_ENTITY_VALUES
        return is_artifact_like(value)

    # Paper/default modes suppress lexical implementation details.
    if kind in {"token", "bigram"}:
        return False

    return False


def event_label(node_id: str, node: Dict[str, Any]) -> str:
    data = node.get("data") or {}
    step = data.get("created_step", "?")
    role = data.get("role", "event")
    text = data.get("text") or raw_key(node_id, node)
    text = re.sub(r"\s+", " ", str(text)).strip()
    if text.startswith(":lesson"):
        text = text[len(":lesson"):].strip()
    if text.lower().startswith("lesson:"):
        text = text.split(":", 1)[1].strip()
    return f"{role} step {step}\n{truncate(text, 36)}"


def feature_label(node_id: str) -> str:
    kind, value = feature_kind_and_value(node_id)
    value = de_lesson(value)
    if kind == "entity":
        if is_artifact_like(value):
            return f"Extracted Artifact\n{truncate(value, 34)}"
        return f"Extracted Entity\n{truncate(value, 34)}"
    if kind == "task":
        return f"Task Signal\n{truncate(value, 34)}"
    if kind == "constraint":
        return f"Constraint Feature\n{truncate(value, 42)}"
    if kind == "token":
        return f"Lexical Token\n{truncate(value, 34)}"
    if kind == "bigram":
        return f"Lexical Bigram\n{truncate(value, 34)}"
    return f"Feature\n{truncate(value or node_id, 34)}"


def concept_label(node_id: str, node: Dict[str, Any]) -> str:
    key = raw_key(node_id, node)
    if key.startswith("constraint::"):
        key = key.split("constraint::", 1)[1]
    key = de_lesson(key)
    if key.startswith("entity::"):
        key = key.split("entity::", 1)[1]
        return f"Concept\nentity: {truncate(key, 34)}"
    return f"Governance Concept\n{truncate(key, 42)}"


def display_label(node_id: str, node: Dict[str, Any]) -> str:
    typ = node_type(node_id, node)
    if typ == "episode":
        return "Episode"
    if typ == "event":
        return event_label(node_id, node)
    if typ == "feature":
        return feature_label(node_id)
    if typ == "concept":
        return concept_label(node_id, node)
    if typ == "outcome":
        return "Outcome"
    return truncate(node_id, 42)


def normalize_nodes(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes_raw = graph.get("nodes", [])
    nodes: Dict[str, Dict[str, Any]] = {}
    if isinstance(nodes_raw, dict):
        for k, v in nodes_raw.items():
            row = dict(v or {})
            row.setdefault("id", k)
            nodes[k] = row
    else:
        for n in nodes_raw:
            node_id = str(n.get("id") or n.get("ref") or "")
            if not node_id:
                typ = str(n.get("type", "node"))
                key = str(n.get("key", ""))
                node_id = f"{typ}:{key}"
            nodes[node_id] = dict(n)
            nodes[node_id].setdefault("id", node_id)
    # graph.to_canonical_dict may include nodes only as refs inside edges in some versions.
    for e in graph.get("edges", []) or []:
        for endpoint in ("src", "dst"):
            nid = str(e.get(endpoint, ""))
            if nid and nid not in nodes:
                typ = nid.split(":", 1)[0]
                nodes[nid] = {"id": nid, "type": typ, "key": nid.split(":", 1)[1] if ":" in nid else nid, "data": {}}
    return nodes


def outgoing_edges(edges: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        out[str(e.get("src"))].append(e)
    return out


def incoming_edges(edges: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    inc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edges:
        inc[str(e.get("dst"))].append(e)
    return inc


def select_subgraph(graph: Dict[str, Any], mode: str = "paper", max_events: int = 7) -> Dict[str, Any]:
    nodes = normalize_nodes(graph)
    edges = list(graph.get("edges", []) or [])
    out = outgoing_edges(edges)
    inc = incoming_edges(edges)

    keep_nodes: Set[str] = set()
    keep_edges: List[Dict[str, Any]] = []

    # Always keep episodes.
    for nid, n in nodes.items():
        if node_type(nid, n) == "episode":
            keep_nodes.add(nid)

    # Keep governance concepts.
    concepts = [nid for nid, n in nodes.items() if node_type(nid, n) == "concept"]
    if mode == "constraints":
        concepts = [nid for nid in concepts if "constraint::" in nid.lower()]
    for nid in concepts:
        keep_nodes.add(nid)

    # Keep promoted constraint features and their promote edges.
    for e in edges:
        if e.get("type") != "promotes_to":
            continue
        src = str(e.get("src")); dst = str(e.get("dst"))
        if dst in keep_nodes:
            keep_nodes.add(src)
            keep_edges.append(e)
            # Pull in the originating event for this feature if available.
            for ine in inc.get(src, []):
                if ine.get("type") == "has_feature":
                    ev = str(ine.get("src"))
                    if node_type(ev, nodes.get(ev, {})) == "event":
                        keep_nodes.add(ev)
                        keep_edges.append(ine)

    # Keep a limited number of recent/user events plus assistant events that illustrate behavior.
    events = [nid for nid, n in nodes.items() if node_type(nid, n) == "event"]
    events.sort(key=lambda nid: (nodes[nid].get("data", {}).get("created_step", 0), nid))
    # Prefer events at or after first non-lesson prompt when possible.
    interesting_events = []
    for nid in events:
        data = nodes[nid].get("data") or {}
        text = str(data.get("text", ""))
        step = data.get("created_step", 0)
        if step >= 5 or "graph-fast-compute" in text.lower() or "cluster-sync" in text.lower():
            interesting_events.append(nid)
    for nid in interesting_events[:max_events]:
        keep_nodes.add(nid)

    # Add contains edges for kept events.
    for e in edges:
        if e.get("type") == "contains":
            src = str(e.get("src")); dst = str(e.get("dst"))
            if src in keep_nodes and dst in keep_nodes:
                keep_edges.append(e)

    # Add meaningful features for kept events.
    for ev in list(keep_nodes):
        if node_type(ev, nodes.get(ev, {})) != "event":
            continue
        for e in out.get(ev, []):
            if e.get("type") != "has_feature":
                continue
            feat = str(e.get("dst"))
            if feat not in nodes:
                continue
            if node_type(feat, nodes[feat]) != "feature":
                continue
            if keep_feature(feat, mode=mode):
                keep_nodes.add(feat)
                keep_edges.append(e)

    # For concepts mode, add all concept promotion chains.
    if mode in {"concepts", "full-lite"}:
        for e in edges:
            if e.get("type") == "promotes_to":
                src = str(e.get("src")); dst = str(e.get("dst"))
                keep_nodes.update([src, dst])
                keep_edges.append(e)
                for ine in inc.get(src, []):
                    if ine.get("type") == "has_feature":
                        ev = str(ine.get("src"))
                        keep_nodes.add(ev)
                        keep_edges.append(ine)

    allowed_edges = EDGE_KEEP_BY_MODE.get(mode, EDGE_KEEP_BY_MODE["paper"])
    dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for e in keep_edges:
        typ = str(e.get("type"))
        src = str(e.get("src")); dst = str(e.get("dst"))
        if typ not in allowed_edges:
            continue
        if src not in keep_nodes or dst not in keep_nodes:
            continue
        dedup[(src, dst, typ)] = e

    filtered_nodes = []
    for nid in sorted(keep_nodes, key=lambda x: (NODE_ORDER.get(node_type(x, nodes.get(x, {})), 99), x)):
        n = dict(nodes[nid])
        n["id"] = nid
        n["viz_label"] = display_label(nid, n)
        filtered_nodes.append(n)

    filtered_edges = list(dedup.values())
    filtered_edges.sort(key=lambda e: (str(e.get("type")), str(e.get("src")), str(e.get("dst"))))

    return {"nodes": filtered_nodes, "edges": filtered_edges}


def dot_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def node_style(node_id: str, node: Dict[str, Any]) -> Dict[str, str]:
    typ = node_type(node_id, node)
    if typ == "episode":
        return {"shape": "ellipse", "style": "filled", "fillcolor": "#f6f8fa", "color": "#111111", "penwidth": "1.4"}
    if typ == "event":
        return {"shape": "box", "style": "rounded,filled", "fillcolor": "#e5f3fb", "color": "#111111", "penwidth": "1.2"}
    if typ == "feature":
        if "feature:constraint::" in node_id.lower():
            return {"shape": "note", "style": "filled", "fillcolor": "#fff4ce", "color": "#111111", "penwidth": "1.2"}
        return {"shape": "note", "style": "filled", "fillcolor": "#f4f4f4", "color": "#111111", "penwidth": "1.1"}
    if typ == "concept":
        return {"shape": "box", "style": "rounded,filled", "fillcolor": "#dcfce7", "color": "#111111", "penwidth": "2.2"}
    return {"shape": "box", "style": "filled", "fillcolor": "#ffffff", "color": "#111111"}


def edge_style(edge: Dict[str, Any]) -> Dict[str, str]:
    typ = str(edge.get("type"))
    if typ == "promotes_to":
        return {"color": "#087830", "penwidth": "2.4", "label": "promotes_to"}
    if typ == "contains":
        return {"color": "#64748b", "penwidth": "1.1", "label": "contains"}
    if typ == "has_feature":
        return {"color": "#2563eb", "penwidth": "1.0", "label": "has_feature"}
    return {"color": "#999999", "penwidth": "1.0", "label": typ}


def to_dot(filtered: Dict[str, Any], title: str = "Yggdrasil filtered governance topology") -> str:
    lines = [
        "digraph G {",
        "  graph [rankdir=LR, bgcolor=\"white\", labelloc=\"t\", label=\"%s\", fontsize=22, fontname=\"Arial\", nodesep=0.55, ranksep=0.9];" % dot_escape(title),
        "  node [fontname=\"Arial\", fontsize=10, margin=0.10];",
        "  edge [fontname=\"Arial\", fontsize=8, arrowsize=0.8];",
    ]
    nodes = {str(n.get("id")): n for n in filtered.get("nodes", [])}
    for nid, n in nodes.items():
        attrs = node_style(nid, n)
        attrs["label"] = n.get("viz_label") or display_label(nid, n)
        attr_s = ", ".join(f'{k}=\"{dot_escape(v)}\"' for k, v in attrs.items())
        lines.append(f'  "{dot_escape(nid)}" [{attr_s}];')
    for e in filtered.get("edges", []) or []:
        src = str(e.get("src")); dst = str(e.get("dst"))
        if src not in nodes or dst not in nodes:
            continue
        attrs = edge_style(e)
        attr_s = ", ".join(f'{k}=\"{dot_escape(v)}\"' for k, v in attrs.items())
        lines.append(f'  "{dot_escape(src)}" -> "{dot_escape(dst)}" [{attr_s}];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def summarize(filtered: Dict[str, Any], source: str, mode: str) -> str:
    node_types = Counter(node_type(str(n.get("id")), n) for n in filtered.get("nodes", []))
    edge_types = Counter(str(e.get("type")) for e in filtered.get("edges", []))
    lines = [
        "# Filtered Graph Summary",
        "",
        f"source: `{source}`",
        f"mode: `{mode}`",
        f"nodes: {len(filtered.get('nodes', []))}",
        f"edges: {len(filtered.get('edges', []))}",
        "",
        "## Node types",
    ]
    for k, v in sorted(node_types.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Edge types"]
    for k, v in sorted(edge_types.items()):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Included nodes"]
    for n in filtered.get("nodes", []):
        lines.append(f"- {node_type(str(n.get('id')), n)}: {str(n.get('viz_label','')).replace(chr(10), ' — ')}")
    return "\n".join(lines) + "\n"


def render_with_graphviz(dot_path: str, out_dir: str) -> List[str]:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return []
    outputs = []
    for fmt in ("svg", "png"):
        out_path = os.path.join(out_dir, f"filtered_graph.{fmt}")
        subprocess.run([dot_bin, f"-T{fmt}", dot_path, "-o", out_path], check=True)
        outputs.append(out_path)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Create paper-readable Yggdrasil graph visualizations from graph.json exports.")
    ap.add_argument("--graph", required=True, help="Path to exported graph.json")
    ap.add_argument("--mode", default="paper", choices=["paper", "constraints", "concepts", "full-lite"], help="Filtering mode")
    ap.add_argument("--out-dir", default="paper_figures/graph", help="Output directory")
    ap.add_argument("--max-events", type=int, default=7, help="Maximum recent/interesting events to include")
    ap.add_argument("--title", default="Yggdrasil filtered governance topology", help="Graph title")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    graph = load_graph(args.graph)
    filtered = select_subgraph(graph, mode=args.mode, max_events=args.max_events)

    json_path = os.path.join(args.out_dir, "filtered_graph.json")
    dot_path = os.path.join(args.out_dir, "filtered_graph.dot")
    summary_path = os.path.join(args.out_dir, "filtered_graph_summary.md")

    write_json(json_path, filtered)
    write_text(dot_path, to_dot(filtered, title=args.title))
    write_text(summary_path, summarize(filtered, args.graph, args.mode))

    rendered = []
    try:
        rendered = render_with_graphviz(dot_path, args.out_dir)
    except Exception as exc:
        print(f"[WARN] Graphviz render failed: {exc}")

    print(f"[WRITE] {json_path}")
    print(f"[WRITE] {dot_path}")
    print(f"[WRITE] {summary_path}")
    if rendered:
        for p in rendered:
            print(f"[RENDER] {p}")
    else:
        print("[INFO] Graphviz 'dot' not found or render skipped. DOT/JSON outputs are available.")


if __name__ == "__main__":
    main()

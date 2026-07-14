from __future__ import annotations

"""
Minimal graph substrate primitives for Yggdrasil.

This module provides lightweight in-memory node and edge storage with
secondary indexes for:

- node type
- logical node key
- outbound adjacency
- inbound adjacency

Design goals:
- simple deterministic behavior
- easy inspection and replay support
- stable keyed upserts for logical entities
- minimal implementation surface
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Node:
    """
    Graph node.

    `id` is the runtime identity of the node instance.
    `key` is an optional logical identity used for stable lookup and upsert.
    """

    id: str
    type: str
    key: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: time.time())


@dataclass
class Edge:
    """
    Directed graph edge between two node ids.
    """

    id: str
    src: str
    dst: str
    type: str
    weight: float = 1.0
    confidence: float = 1.0
    ts: float = field(default_factory=lambda: time.time())
    data: Dict[str, Any] = field(default_factory=dict)


class Graph:
    """
    Lightweight in-memory graph with simple secondary indexes.

    This graph favors clarity and deterministic behavior over advanced
    storage features. It is intended for runtime substrate state, audit,
    and replay-friendly inspection.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.by_type: Dict[str, List[str]] = {}
        self.by_key: Dict[Tuple[str, str], str] = {}
        self.out_edges: Dict[str, List[str]] = {}
        self.in_edges: Dict[str, List[str]] = {}

    def _add_node_index(self, node: Node) -> None:
        """Add a node to type and key indexes."""
        self.by_type.setdefault(node.type, []).append(node.id)
        if node.key is not None:
            self.by_key[(node.type, node.key)] = node.id

    def _add_edge_index(self, edge: Edge) -> None:
        """Add an edge to adjacency indexes."""
        self.out_edges.setdefault(edge.src, []).append(edge.id)
        self.in_edges.setdefault(edge.dst, []).append(edge.id)

    def upsert_node(
        self,
        type: str,
        key: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> Node:
        """
        Create or update a node.

        When `key` is provided, `(type, key)` acts as a logical identity and
        repeated calls update the existing node instead of creating a new one.
        """
        if key is not None and (type, key) in self.by_key:
            n = self.nodes[self.by_key[(type, key)]]
            if data:
                n.data.update(data)
            if ts is not None:
                n.ts = ts
            return n

        nid = str(uuid.uuid4())
        n = Node(id=nid, type=type, key=key, data=data or {}, ts=ts or time.time())
        self.nodes[nid] = n
        self._add_node_index(n)
        return n

    def add_edge(
        self,
        src: str,
        dst: str,
        type: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        data: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> Edge:
        """
        Create a directed edge between two existing node ids.
        """
        eid = str(uuid.uuid4())
        e = Edge(
            id=eid,
            src=src,
            dst=dst,
            type=type,
            weight=weight,
            confidence=confidence,
            data=data or {},
            ts=ts or time.time(),
        )
        self.edges[eid] = e
        self._add_edge_index(e)
        return e

    def delete_edge(self, edge_id: str) -> bool:
        """
        Delete an edge and remove it from adjacency indexes.
        """
        e = self.edges.get(edge_id)
        if not e:
            return False

        if e.src in self.out_edges:
            self.out_edges[e.src] = [x for x in self.out_edges[e.src] if x != edge_id]
        if e.dst in self.in_edges:
            self.in_edges[e.dst] = [x for x in self.in_edges[e.dst] if x != edge_id]

        del self.edges[edge_id]
        return True

    def find_nodes(self, type: str) -> List[Node]:
        """
        Return all nodes of a given type.
        """
        return [self.nodes[nid] for nid in self.by_type.get(type, []) if nid in self.nodes]

    def find_node_by_key(self, type: str, key: str) -> Optional[Node]:
        """
        Return a node by logical `(type, key)` identity.
        """
        nid = self.by_key.get((type, key))
        return self.nodes.get(nid) if nid else None

    def neighbors_out(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
    ) -> List[Tuple[Edge, Node]]:
        """
        Return outgoing neighbors as `(edge, destination_node)` pairs.
        """
        res: List[Tuple[Edge, Node]] = []
        for eid in list(self.out_edges.get(node_id, [])):
            e = self.edges.get(eid)
            if not e:
                continue
            if edge_type and e.type != edge_type:
                continue
            res.append((e, self.nodes[e.dst]))
        return res

    def neighbors_in(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
    ) -> List[Tuple[Edge, Node]]:
        """
        Return incoming neighbors as `(edge, source_node)` pairs.
        """
        res: List[Tuple[Edge, Node]] = []
        for eid in list(self.in_edges.get(node_id, [])):
            e = self.edges.get(eid)
            if not e:
                continue
            if edge_type and e.type != edge_type:
                continue
            res.append((e, self.nodes[e.src]))
        return res

    def count_edges(self, edge_type: Optional[str] = None) -> int:
        """
        Count all edges, or only edges of a given type.
        """
        if edge_type is None:
            return len(self.edges)
        return sum(1 for e in self.edges.values() if e.type == edge_type)

# ------------------------------------------------------------------
# Human-readable export / evidence helpers
# ------------------------------------------------------------------

    def node_type_counts(self) -> Dict[str, int]:
        """Return counts grouped by node type."""
        return {typ: len([nid for nid in ids if nid in self.nodes]) for typ, ids in sorted(self.by_type.items())}

    def edge_type_counts(self) -> Dict[str, int]:
        """Return counts grouped by edge type."""
        counts: Dict[str, int] = {}
        for e in self.edges.values():
            counts[e.type] = counts.get(e.type, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> Dict[str, Any]:
        """Return a compact graph summary suitable for reports."""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": self.node_type_counts(),
            "edge_types": self.edge_type_counts(),
        }

    def format_summary(self) -> str:
        """Return a terminal / paper-friendly graph summary."""
        s = self.summary()
        lines = [
            "[GRAPH SUMMARY]",
            f"nodes={s['nodes']} edges={s['edges']}",
            "node_types:",
        ]
        for k, v in s["node_types"].items():
            lines.append(f"  - {k}: {v}")
        lines.append("edge_types:")
        for k, v in s["edge_types"].items():
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    def to_canonical_dict(self, include_runtime_ids: bool = False) -> Dict[str, Any]:
        """
        Export the graph in deterministic order for inspection.

        By default, runtime UUID fields are omitted. The export is intended for
        human-readable evidence and diffable debugging, not as the canonical
        replay source. JSONL events remain the causal source of truth.
        """
        nodes = []
        for n in self.nodes.values():
            row = {
                "type": n.type,
                "key": n.key,
                "data": dict(sorted((n.data or {}).items())),
            }
            if include_runtime_ids:
                row["id"] = n.id
                row["ts"] = n.ts
            nodes.append(row)
        nodes.sort(key=lambda r: (str(r.get("type")), str(r.get("key")), str(r.get("data"))))

        def node_ref(node_id: str) -> str:
            n = self.nodes.get(node_id)
            if not n:
                return "missing"
            if n.key is not None:
                return f"{n.type}:{n.key}"
            return f"{n.type}:{n.id}" if include_runtime_ids else f"{n.type}:<runtime>"

        edges = []
        for e in self.edges.values():
            row = {
                "type": e.type,
                "src": node_ref(e.src),
                "dst": node_ref(e.dst),
                "weight": round(float(e.weight), 8),
                "confidence": round(float(e.confidence), 8),
                "data": dict(sorted((e.data or {}).items())),
            }
            if include_runtime_ids:
                row["id"] = e.id
                row["ts"] = e.ts
            edges.append(row)
        edges.sort(key=lambda r: (r["type"], r["src"], r["dst"], str(r["data"])))
        return {"summary": self.summary(), "nodes": nodes, "edges": edges}

    def format_nodes(self, limit: int = 25) -> str:
        """Return a compact node listing."""
        rows = self.to_canonical_dict().get("nodes", [])[: max(0, int(limit))]
        lines = [f"[NODES] showing={len(rows)} limit={limit}"]
        for r in rows:
            data = r.get("data", {})
            preview = ", ".join(f"{k}={v!r}" for k, v in list(data.items())[:4])
            lines.append(f"- {r.get('type')}::{r.get('key')} {preview}")
        return "\n".join(lines)

    def format_edges(self, limit: int = 25) -> str:
        """Return a compact edge listing."""
        rows = self.to_canonical_dict().get("edges", [])[: max(0, int(limit))]
        lines = [f"[EDGES] showing={len(rows)} limit={limit}"]
        for r in rows:
            lines.append(
                f"- {r.get('src')} -[{r.get('type')} w={r.get('weight')} c={r.get('confidence')}]-> {r.get('dst')}"
            )
        return "\n".join(lines)

    def to_dot(self, max_label: int = 48) -> str:
        """
        Export a Graphviz DOT representation for visualization.

        This is intentionally simple and dependency-free. Render with:
        `dot -Tpng graph.dot -o graph.png`.
        """
        def esc(s: Any) -> str:
            return str(s).replace('\\', '\\\\').replace('"', '\\"')

        def short(s: Any) -> str:
            s = str(s)
            return s if len(s) <= max_label else s[: max_label - 3] + "..."

        lines = ["digraph YggdrasilGraph {", "  rankdir=LR;", "  node [shape=box, fontsize=10];"]
        for n in sorted(self.nodes.values(), key=lambda x: (x.type, str(x.key), x.id)):
            label = f"{n.type}\\n{short(n.key or n.id)}"
            lines.append(f'  "{esc(n.id)}" [label="{esc(label)}"];')
        for e in sorted(self.edges.values(), key=lambda x: (x.type, x.src, x.dst, x.id)):
            label = f"{e.type}\\nw={round(float(e.weight), 3)} c={round(float(e.confidence), 3)}"
            lines.append(f'  "{esc(e.src)}" -> "{esc(e.dst)}" [label="{esc(label)}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

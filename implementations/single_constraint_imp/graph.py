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
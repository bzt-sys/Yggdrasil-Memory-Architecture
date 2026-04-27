from __future__ import annotations

"""
Deterministic graph state hashing for replay and audit checks.

This module computes a stable hash over the logical graph state while
intentionally excluding fields that are expected to vary across runs,
such as runtime UUIDs, raw event text, and timestamps.

The goal is not full graph serialization fidelity. The goal is a stable,
useful replay parity signal for verifying that two executions produced the
same meaningful substrate state.
"""

import hashlib
import json
from typing import Any, Dict, List, Tuple

# Keep node hash inputs stable: include only fields intended to be
# deterministic across equivalent executions.
NODE_DATA_WHITELIST = {
    "event": ["role", "source", "intent"],  # intentionally excludes raw text and timestamps
    "episode": ["status", "opened_step", "closed_step"],
    "feature": ["kind", "value"],
    "concept": ["kind", "support", "count_seen", "promoted_step", "last_seen_step"],
    "outcome": ["score", "label"],
}

# Keep edge hash inputs stable for the same reason.
EDGE_DATA_WHITELIST = {
    "has_feature": ["salience", "novelty", "created_step", "kind"],
    "contains": ["step"],
    "leads_to": ["step"],
    "promotes_to": ["step"],
    "rel:cooccur": ["step"],
    "rel:supports": ["step"],
    "rel:conflicts": ["step"],
    "rel:depends": ["step"],
}


def _pick(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Select a stable subset of fields from a dictionary.

    Float values are rounded so insignificant numeric noise does not affect
    replay parity checks.
    """
    out: Dict[str, Any] = {}
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, float):
                v = round(v, 8)
            out[k] = v
    return out


def compute_state_hash(graph) -> Tuple[str, Dict[str, Any]]:
    """
    Compute a deterministic hash over the logical graph state.

    The hash intentionally ignores runtime-only identifiers and other
    non-deterministic fields. Nodes are keyed by `(type, key)` and edges are
    keyed by logical source / destination identities rather than UUIDs.

    Returns:
    - hex digest of the normalized graph payload
    - compact summary dict with node and edge counts
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Build stable node records keyed by (type, key). Ignore internal UUID ids.
    for n in graph.nodes.values():
        if n.key is None:
            continue
        w = NODE_DATA_WHITELIST.get(n.type, [])
        nodes.append(
            {
                "type": n.type,
                "key": n.key,
                "data": _pick(n.data, w),
            }
        )

    # Convert edge endpoints to logical identities to remove UUID dependence.
    for e in graph.edges.values():
        src = graph.nodes.get(e.src)
        dst = graph.nodes.get(e.dst)
        if not src or not dst or src.key is None or dst.key is None:
            continue

        w = EDGE_DATA_WHITELIST.get(e.type, [])
        edges.append(
            {
                "type": e.type,
                "src": f"{src.type}:{src.key}",
                "dst": f"{dst.type}:{dst.key}",
                "confidence": round(float(e.confidence), 8),
                "weight": round(float(e.weight), 8),
                "data": _pick(e.data, w),
            }
        )

    # Deterministic ordering
    nodes.sort(key=lambda r: (r["type"], r["key"]))
    edges.sort(key=lambda r: (r["type"], r["src"], r["dst"]))

    payload = {"nodes": nodes, "edges": edges}
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()

    summary = {"nodes": len(nodes), "edges": len(edges)}
    return h, summary
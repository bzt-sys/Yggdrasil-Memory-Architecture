from __future__ import annotations

"""
Deterministic graph state hashing for replay and audit checks.

This module computes a stable hash over the logical graph state while
intentionally excluding fields that are expected to vary across runs, such as
runtime UUIDs, raw event text, and timestamps.

The hash is intended as a replay parity signal: two executions that produce the
same meaningful substrate state should produce the same hash even if internal
UUIDs differ.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Tuple

NODE_DATA_WHITELIST = {
    "event": ["role", "source", "intent", "created_step"],
    "episode": ["status", "opened_step", "closed_step"],
    "feature": ["kind", "value"],
    "concept": ["kind", "support", "count_seen", "promoted_step", "last_seen_step"],
    "outcome": ["score", "label", "created_step"],
}

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

# Node types whose keys are usually runtime UUIDs and should not be used as
# logical identity in the canonical hash payload.
VOLATILE_KEY_TYPES = {"event", "episode", "outcome"}
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _pick(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, float):
                v = round(v, 8)
            out[k] = v
    return out


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _uses_volatile_key(node_type: str, key: Any) -> bool:
    return node_type in VOLATILE_KEY_TYPES or (isinstance(key, str) and bool(_UUID_RE.match(key)))


def canonical_state_payload(graph) -> Dict[str, Any]:
    """
    Build the normalized graph payload used for deterministic hashing.

    Runtime UUIDs are mapped to deterministic canonical identities before edges
    are normalized. This allows replay from JSONL causal records to verify state
    parity even when fresh runtime objects receive fresh UUIDs.
    """
    node_rows: List[Tuple[str, str, Dict[str, Any], str]] = []
    id_to_ref: Dict[str, str] = {}

    # First pass: create stable node records and temporary sort keys.
    for n in graph.nodes.values():
        w = NODE_DATA_WHITELIST.get(n.type, [])
        data = _pick(n.data or {}, w)
        if _uses_volatile_key(n.type, n.key):
            # For event/outcome/episode nodes, derive identity from deterministic
            # data and insertion-order fallback. created/opened/closed steps are
            # included where available to avoid collisions between repeated roles.
            sort_key = _stable_json({"type": n.type, "data": data})
            key = ""  # filled with ordinal below
        else:
            key = str(n.key)
            sort_key = _stable_json({"type": n.type, "key": key, "data": data})
        node_rows.append((n.id, n.type, {"type": n.type, "key": key, "data": data}, sort_key))

    # Assign deterministic ordinals for volatile-key nodes after sorting.
    node_rows.sort(key=lambda r: (r[1], r[3], r[0]))
    volatile_counts: Dict[str, int] = {}
    nodes: List[Dict[str, Any]] = []
    for runtime_id, typ, row, sort_key in node_rows:
        if row["key"] == "":
            volatile_counts[typ] = volatile_counts.get(typ, 0) + 1
            row = dict(row)
            row["key"] = f"{typ}#{volatile_counts[typ]:06d}"
        id_to_ref[runtime_id] = f"{row['type']}:{row['key']}"
        nodes.append(row)

    # Second pass: normalize edges through canonical endpoint identities.
    edges: List[Dict[str, Any]] = []
    for e in graph.edges.values():
        if e.src not in id_to_ref or e.dst not in id_to_ref:
            continue
        w = EDGE_DATA_WHITELIST.get(e.type, [])
        edges.append(
            {
                "type": e.type,
                "src": id_to_ref[e.src],
                "dst": id_to_ref[e.dst],
                "confidence": round(float(e.confidence), 8),
                "weight": round(float(e.weight), 8),
                "data": _pick(e.data or {}, w),
            }
        )

    nodes.sort(key=lambda r: (r["type"], r["key"], _stable_json(r["data"])))
    edges.sort(key=lambda r: (r["type"], r["src"], r["dst"], _stable_json(r["data"])))
    return {"nodes": nodes, "edges": edges}


def compute_state_hash(graph) -> Tuple[str, Dict[str, Any]]:
    payload = canonical_state_payload(graph)
    blob = _stable_json(payload).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()
    summary = {"nodes": len(payload["nodes"]), "edges": len(payload["edges"])}
    return h, summary

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
    "event": ["role", "source", "intent", "created_step", "semantic_profile"],
    "episode": ["status", "opened_step", "closed_step"],
    "feature": ["kind", "value"],
    "concept": ["kind", "support", "count_seen", "promoted_step", "last_seen_step", "adaptive_governance", "authority_state", "preservation_state"],
    "outcome": ["score", "label", "created_step", "verified", "source"],
    "candidate": [
        "status", "containment", "verified", "locality", "family",
        "assignment_reason", "locality_status", "locality_depth", "locality_confidence", "locality_path",
        "target_type", "target_key", "proposed_family", "proposed_subfamily", "proposal_reason",
        "salience", "created_step",
        "expires_after_step", "pedigree_status", "integration_action",
        "novelty_score", "similarity_score", "feature_similarity_score", "usefulness_score",
        "recurrence_count", "variant_of", "status_reason", "integration_confidence", "pedigree_evaluated_step",
        "merged_into", "merged_step", "integration_state", "active_for_retrieval",
        "reinforcement_count", "reinforced_by_ids", "variant_count", "variant_ids", "variant_facets",
        "candidate_kind", "anchor_state", "scope_state", "authority_state", "task_id",
        "success_count", "reuse_count", "transfer_success_count", "regression_count", "failed_reuse_count",
        "missed_high_similarity_count", "utility_score", "use_count", "positive_use_count", "negative_use_count",
    ],
    "locality_region": ["family", "target_type", "target_key", "locality_path", "locality_depth", "locality_status", "parent_locality", "ontology_origin", "authority_state", "created_step", "last_seen_step", "established_step", "evidence_seen_count", "artifact_count"],
    "locality_proposal": ["status", "proposal_kind", "family", "proposed_family", "proposed_subfamily", "locality", "locality_path", "created_step", "last_seen_step", "seen_count", "confidence", "reason"],
    "collation_run": ["step", "reason", "max_items", "status", "reviewed_candidates", "protected_candidates", "demoted_candidates", "proposals_created", "promotion_evaluations", "completed_step"],
    "recurrent_episode": ["status", "task_id", "attempt_count", "signatures", "support_candidate_ids", "support_investigation_ids", "interpretation", "created_step", "last_seen_step"],
    "developmental_hypothesis": ["status", "authority_state", "claim", "observation", "lesson", "procedure", "scope", "rejection_condition", "confidence", "validation_score", "validation_state", "governance_ready", "family", "locality", "locality_confidence", "support_candidate_ids", "evidence_count", "recurrence_count", "variant_count", "topology_designation", "topology_designation_source"],
    "investigation": ["status", "task_id", "attempt_number", "error_type", "observed_error", "repeated_failure_signature", "hypotheses", "repair_directions", "information_needs", "confidence", "created_step"],
    "pre_answer_investigation": ["task_id", "attempt_number", "prior_investigation_ids", "unresolved_hypotheses", "repeated_failure_signature", "information_needs", "created_step"],
    "action_rationale": ["rationale_id", "event_id", "task_id", "attempt_number", "summary", "basis", "created_step", "step"],
    "governance_proposal": [
        "status", "activation_state", "durable", "verified", "promotion_status", "source",
        "family", "locality", "locality_status", "locality_depth", "locality_confidence", "locality_path",
        "target_type", "target_key", "proposed_family", "proposed_subfamily",
        "confidence", "evidence_candidate_count", "recurrence_count", "variant_count",
        "created_step", "last_seen_step", "seen_count", "active_for_governance",
        "promotion_evidence_score", "promotion_reasons", "last_promotion_eval_step",
        "promoted_step", "promoted_concept_key", "demoted_step", "authority_state",
    ],
}

EDGE_DATA_WHITELIST = {
    "has_feature": ["salience", "novelty", "created_step", "kind"],
    "contains": ["step"],
    "leads_to": ["step"],
    "promotes_to": ["step", "adaptive"],
    "rel:cooccur": ["step"],
    "rel:supports": ["step"],
    "rel:conflicts": ["step"],
    "rel:depends": ["step"],
    "derived_from": ["step", "containment"],
    "supports": ["step", "provisional"],
    "reinforces": ["step", "similarity", "cache_local"],
    "variant_of": ["step", "similarity", "cache_local", "shared_features", "variant_features", "parent_only_features", "feature_delta_count"],
    "assigned_to": ["step", "artifact_role", "locality_status", "assignment_reason"],
    "reviewed_in": ["step", "preservation_state", "authority_state"],
    "supported_by": ["step", "replay_restore"],
    "contains_locality": ["step", "replay_restore"],
    "proposes_locality": ["step", "replay_restore"],
    "considers_investigation": ["step", "task_id"],
    "continues_investigation": ["step", "task_id"],
    "derived_from_investigation": ["step"],
}

# Node types whose keys are usually runtime UUIDs and should not be used as
# logical identity in the canonical hash payload.
VOLATILE_KEY_TYPES = {"event", "episode", "outcome", "candidate", "governance_proposal", "collation_run", "investigation", "pre_answer_investigation", "action_rationale", "developmental_hypothesis", "recurrent_episode"}
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
        if n.type == "outcome" and "verified" in w and "verified" not in data:
            data["verified"] = False
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


def runtime_state_payload(engine) -> Dict[str, Any]:
    """Canonical operational state affecting continuation and retrieval.

    Runtime UUIDs are represented by semantic order signatures rather than raw
    identifiers so reconstructed engines can be equivalent without sharing
    process-local keys.
    """
    def node_signature(node_type: str, key: Any) -> Dict[str, Any]:
        node = engine.graph.find_node_by_key(node_type, str(key))
        if node is None:
            return {"missing": True}
        data = node.data or {}
        row = {
            "type": node_type,
            "created_step": int(data.get("created_step", data.get("opened_step", data.get("step", 0))) or 0),
            "status": data.get("status"),
            "role": data.get("role"),
            "source": data.get("source"),
        }
        if node_type == "candidate":
            row["family"] = data.get("family")
            row["locality"] = data.get("locality")
        return row
    current = getattr(engine, "_current_episode_id", None)
    return {
        "step": int(getattr(engine, "step", 0)),
        "current_episode": node_signature("episode", current) if current else None,
        "episode_order": [node_signature("episode", x) for x in getattr(engine, "_episode_order", [])],
        "event_order": [node_signature("event", x) for x in getattr(engine, "_event_order", [])],
        "candidate_order": [node_signature("candidate", x) for x in getattr(engine, "_candidate_order", [])],
        "feature_counts": {str(k): int(v) for k, v in sorted(getattr(engine, "_feature_counts", {}).items())},
        "feature_support": {str(k): round(float(v), 8) for k, v in sorted(getattr(engine, "_feature_support", {}).items())},
        "last_collation_step": int(getattr(engine, "_last_collation_step", 0)),
        "ablate_governance": bool(getattr(engine, "ablate_governance", False)),
        "enable_decay": bool(getattr(engine, "enable_decay", True)),
        "candidate_cache_ttl_steps": int(getattr(engine, "candidate_cache_ttl_steps", 0)),
        "max_candidate_cache": int(getattr(engine, "max_candidate_cache", 0)),
        "protected_candidate_ttl_steps": int(getattr(engine, "protected_candidate_ttl_steps", 0)),
        "auto_collation_interval_steps": getattr(engine, "auto_collation_interval_steps", None),
    }


def compute_runtime_state_hash(engine) -> Tuple[str, Dict[str, Any]]:
    payload = runtime_state_payload(engine)
    blob = _stable_json(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), payload

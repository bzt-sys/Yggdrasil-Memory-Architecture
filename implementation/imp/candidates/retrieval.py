from __future__ import annotations

"""Candidate retrieval and ranking helpers.

These functions are deliberately pure with respect to engine state except for
the caller-provided graph node objects. ``YggdrasilEngine`` remains responsible
for expiry, graph mutation, logging, and routing construction; this module owns
candidate projection and ranking policy.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set


SimilarityFn = Callable[[str, str], float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _candidate_kind(data: Dict[str, Any]) -> str:
    """Classify candidate content for context ranking diagnostics."""
    text = str(data.get("text") or "").lower()
    action = str(data.get("integration_action") or "")
    locality = str(data.get("locality") or "").lower()

    if "repair brief" in text or "failure_repair" in text or "objective_failure" in text:
        return "repair_or_failure"
    if "success_reinforcement" in text or "objective_success" in text:
        return "success_pattern"
    if "generic_episode_reflection" in text or "topic_shift" in text:
        return "generic_reflection"
    if action in {"attach_variant", "reinforce_existing"}:
        return "variant_or_reinforcement"
    if "assertionerror" in locality or "nameerror" in locality or "syntaxerror" in locality:
        return "error_locality"
    return "candidate"


def _candidate_priority(data: Dict[str, Any]) -> float:
    """Policy bias used after ordinary salience/routing/similarity scoring.

    The goal is to prevent generic reflection prose from crowding out actionable
    repair evidence during actor context assembly.
    """
    kind = _candidate_kind(data)
    if kind == "repair_or_failure":
        return 0.30
    if kind == "error_locality":
        return 0.18
    if kind == "variant_or_reinforcement":
        return 0.10
    if kind == "success_pattern":
        return 0.03
    if kind == "generic_reflection":
        return -0.18
    return 0.0


def project_candidate_node(
    node: Any,
    *,
    prompt_text: str,
    step: int,
    route: Optional[str],
    similarity_fn: SimilarityFn,
) -> Dict[str, Any]:
    """Project a graph candidate node into the actor-facing context row."""
    data = dict(getattr(node, "data", {}) or {})
    family = str(data.get("family", "general_constraint"))

    score = _safe_float(data.get("salience"), 0.0)
    if data.get("status") == "variant_cached":
        score -= 0.05
    if route and (route == family or route in str(data.get("locality", ""))):
        score += 0.35

    text = str(data.get("text", "") or "")
    prompt_sim = similarity_fn(prompt_text, text) if prompt_text else 0.0
    score += 0.20 * _safe_float(prompt_sim, 0.0)

    policy_bias = _candidate_priority(data)
    score += policy_bias

    # The engine passes live graph nodes; updating last_seen here preserves the
    # previous retrieval side effect while centralizing selection policy.
    data["last_seen_step"] = step
    try:
        node.data["last_seen_step"] = step
    except Exception:
        pass

    candidate_kind = _candidate_kind(data)

    return {
        "candidate_id": getattr(node, "key", None),
        "family": family,
        "locality": data.get("locality"),
        "locality_status": data.get("locality_status"),
        "locality_depth": data.get("locality_depth"),
        "locality_confidence": data.get("locality_confidence"),
        "locality_path": data.get("locality_path", []),
        "target_type": data.get("target_type"),
        "target_key": data.get("target_key"),
        "proposed_family": data.get("proposed_family"),
        "proposed_subfamily": data.get("proposed_subfamily"),
        "proposal_reason": data.get("proposal_reason"),
        "containment": data.get("containment", "candidate_cache"),
        "memory_tier": data.get("memory_tier"),
        "preservation_state": data.get("preservation_state"),
        "organization_state": data.get("organization_state"),
        "authority_state": data.get("authority_state"),
        "rarity_score": data.get("rarity_score"),
        "permanence_score": data.get("permanence_score"),
        "outcome_severity": data.get("outcome_severity"),
        "salience": data.get("salience"),
        "score": round(score, 4),
        "text": data.get("text"),
        "verified": data.get("verified", False),
        "status": data.get("status"),
        "pedigree_status": data.get("pedigree_status", "unchecked"),
        "integration_action": data.get("integration_action", "undecided"),
        "novelty_score": data.get("novelty_score"),
        "similarity_score": data.get("similarity_score"),
        "usefulness_score": data.get("usefulness_score"),
        "recurrence_count": data.get("recurrence_count", 1),
        "variant_of": data.get("variant_of"),
        "integration_confidence": data.get("integration_confidence"),
        "pedigree_reasons": data.get("pedigree_reasons", []),
        "nearest_candidate_id": data.get("nearest_candidate_id"),
        "merged_into": data.get("merged_into"),
        "reinforcement_count": data.get("reinforcement_count", 0),
        "variant_count": data.get("variant_count", 0),
        "variant_ids": data.get("variant_ids", []),
        "variant_facets": data.get("variant_facets", {}),
        "variant_summaries": data.get("variant_summaries", []),
        "candidate_kind": candidate_kind,
        "retrieval_policy_bias": round(policy_bias, 4),
    }


def retrieve_ranked_candidates(
    *,
    graph: Any,
    prompt_text: str,
    k: int,
    step: int,
    routing_state: Dict[str, Any],
    candidate_statuses: Iterable[str],
    similarity_fn: SimilarityFn,
) -> List[Dict[str, Any]]:
    """Return top-k candidate rows using the shared ranking policy."""
    statuses: Set[str] = {str(s) for s in candidate_statuses}
    route = routing_state.get("route") if routing_state else None

    rows: List[Dict[str, Any]] = []
    for node in graph.find_nodes("candidate"):
        data = getattr(node, "data", {}) or {}
        if data.get("status") not in statuses:
            continue
        if data.get("active_for_retrieval") is False:
            continue
        rows.append(
            project_candidate_node(
                node,
                prompt_text=prompt_text,
                step=step,
                route=route,
                similarity_fn=similarity_fn,
            )
        )

    rows.sort(key=lambda r: (-_safe_float(r.get("score"), 0.0), str(r.get("candidate_id"))))
    return rows[: int(k)]

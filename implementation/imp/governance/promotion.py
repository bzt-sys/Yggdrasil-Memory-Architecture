from __future__ import annotations

"""Pure governance proposal / promotion policy helpers.

These helpers intentionally do not mutate the graph. ``memory.py`` remains the
runtime orchestrator and graph owner; this module owns deterministic governance
policy calculations so promotion behavior can be tested independently.
"""

from typing import Any, Callable, Dict, List, Tuple


def governance_proposal_statuses() -> set[str]:
    """Statuses that remain eligible for inactive proposal retrieval/evaluation."""
    return {"proposed", "revised", "held"}


def candidate_evidence_count(candidate_data: Dict[str, Any]) -> int:
    """Return conservative evidence count implied by one candidate artifact."""
    return (
        1
        + int(candidate_data.get("reinforcement_count", 0) or 0)
        + int(candidate_data.get("variant_count", 0) or 0)
    )


def proposal_text_from_candidate_data(candidate_data: Dict[str, Any]) -> str:
    """Create a governance-shaped but inactive proposal statement."""
    family = str(candidate_data.get("family", "general_constraint"))
    locality = str(candidate_data.get("locality", "family::general_constraint"))
    candidate_text = str(candidate_data.get("text", "")).strip()
    return (
        f"For future situations routed to {family} at {locality}, consider this provisional governance: "
        f"{candidate_text}"
    )


def proposal_key_for_candidate_key(candidate_key: str) -> str:
    return f"proposal::{candidate_key}"


def adaptive_constraint_key_from_proposal_data(
    proposal_data: Dict[str, Any],
    *,
    locality_slug: Callable[[str], str],
) -> str:
    """Create a stable active-constraint key from a promoted proposal."""
    family = str(proposal_data.get("family", "general_constraint"))
    target = str(proposal_data.get("target_key") or proposal_data.get("locality") or family)
    text = str(proposal_data.get("proposal_text", ""))
    slug = locality_slug(" ".join([family, target, text])[:180])
    return f"constraint::adaptive::{slug}"


def promotion_evidence_score(proposal_data: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Compute conservative promotion evidence from cached proposal metadata."""
    reasons: List[str] = []
    confidence = float(proposal_data.get("confidence", 0.0) or 0.0)
    evidence_count = int(proposal_data.get("evidence_candidate_count", 1) or 1)
    recurrence = int(proposal_data.get("recurrence_count", 1) or 1)
    variants = int(proposal_data.get("variant_count", 0) or 0)
    locality_conf = float(proposal_data.get("locality_confidence", 0.0) or 0.0)
    seen = int(proposal_data.get("seen_count", 1) or 1)

    if evidence_count >= 3:
        reasons.append("multiple_candidate_evidence")
    if recurrence >= 3:
        reasons.append("recurrence_threshold_met")
    if variants >= 1:
        reasons.append("variant_support_present")
    if locality_conf >= 0.55:
        reasons.append("locality_confidence_sufficient")
    if confidence >= 0.65:
        reasons.append("proposal_confidence_sufficient")
    if seen >= 2:
        reasons.append("proposal_retrieved_or_seen_again")

    evidence_score = (
        0.28 * min(1.0, confidence)
        + 0.22 * min(1.0, locality_conf)
        + 0.20 * min(1.0, evidence_count / 4.0)
        + 0.18 * min(1.0, recurrence / 4.0)
        + 0.12 * min(1.0, variants / 3.0)
    )
    return round(evidence_score, 4), reasons


def promotion_gate_decision(
    proposal_data: Dict[str, Any],
    *,
    step: int,
    candidate_cache_ttl_steps: int,
) -> Dict[str, Any]:
    """Evaluate proposal promotion/demotion gates without mutating graph state."""
    score, reasons = promotion_evidence_score(proposal_data)
    confidence = float(proposal_data.get("confidence", 0.0) or 0.0)
    evidence_count = int(proposal_data.get("evidence_candidate_count", 1) or 1)
    recurrence = int(proposal_data.get("recurrence_count", 1) or 1)
    variants = int(proposal_data.get("variant_count", 0) or 0)
    age = int(step) - int(proposal_data.get("created_step", step) or step)

    promote_gate = (
        score >= 0.68
        and evidence_count >= 3
        and recurrence >= 2
        and confidence >= 0.55
        and proposal_data.get("locality_status") in {"existing", "propose_subfamily"}
    )
    demote_gate = age > (int(candidate_cache_ttl_steps) * 2) and score < 0.35

    if promote_gate:
        decision = "promote"
        promotion_status = "promoted_unverified"
    elif demote_gate:
        decision = "demote"
        promotion_status = "demoted_weak_evidence"
    else:
        decision = "hold"
        promotion_status = "held_for_more_evidence"

    return {
        "decision": decision,
        "promotion_status": promotion_status,
        "evidence_score": score,
        "reasons": reasons,
        "confidence": confidence,
        "evidence_candidate_count": evidence_count,
        "recurrence_count": recurrence,
        "variant_count": variants,
        "age": age,
        "promote_gate": promote_gate,
        "demote_gate": demote_gate,
    }

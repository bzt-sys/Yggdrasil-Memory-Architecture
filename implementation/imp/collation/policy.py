from __future__ import annotations

"""Pure collation review policy.

These helpers intentionally avoid importing graph or engine types. They accept
plain candidate/profile dictionaries and return decisions that ``memory.py`` can
apply to the graph. Keeping this layer pure makes collation easier to test,
replace, and eventually schedule as a slow/offline process.

This file also includes compatibility wrappers for earlier Phase 3 migration
patches that expected ``review_candidate_projection`` and
``summarize_collation_reviews`` exports.
"""

from typing import Any, Dict, Iterable, List, Tuple


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def candidate_sort_key(candidate_data: Dict[str, Any], candidate_key: str = "") -> Tuple[float, int, str]:
    """Stable ordering for bounded collation review.

    Higher permanence/salience is reviewed first. Older candidates win ties so
    repeated runs remain deterministic and long-lived unresolved artifacts are
    not permanently starved by newer cache entries.
    """
    priority = _as_float(candidate_data.get("permanence_score", candidate_data.get("salience", 0.0)))
    created_step = _as_int(candidate_data.get("created_step", 0))
    return (-priority, created_step, str(candidate_key or ""))


def organization_state_for_integration_action(action: str) -> str:
    """Map candidate integration action to slow-path organization state."""
    action = str(action or "")
    if action in {"attach_variant", "reinforce_existing"}:
        return "organized_variant_or_reinforcement"
    if action == "abstract_to_concept":
        return "abstracted_candidate"
    if action == "propose_governance":
        return "proposal_source"
    return "unorganized"


def should_demote_candidate(candidate_data: Dict[str, Any]) -> bool:
    """Return whether collation should demote a weak held cache item."""
    action = str(candidate_data.get("integration_action", ""))
    permanence = _as_float(candidate_data.get("permanence_score", 0.0))
    recurrence = _as_int(candidate_data.get("recurrence_count", 1), default=1)
    return action == "hold_cached" and permanence < 0.28 and recurrence <= 1


def should_propose_governance(candidate_data: Dict[str, Any]) -> bool:
    """Return whether recurrence/usefulness justifies a governance-shaped proposal.

    This does not grant authority. It only says the candidate is strong enough
    to become an inactive proposal reviewed by the promotion gate.
    """
    recurrence = _as_int(candidate_data.get("recurrence_count", 1), default=1)
    usefulness = _as_float(candidate_data.get("usefulness_score", 0.0))
    action = str(candidate_data.get("integration_action", ""))
    return recurrence >= 2 and usefulness >= 0.55 and action not in {"propose_governance", "reinforce_existing"}


def collation_review_reasons(profile: Dict[str, Any], candidate_data: Dict[str, Any]) -> List[str]:
    """Build auditable reason list for one candidate review."""
    reasons = [str(profile.get("protection_reason", "collation_review"))]
    org = organization_state_for_integration_action(str(candidate_data.get("integration_action", "")))
    if org != "unorganized":
        reasons.append(f"organization_state::{org}")
    if should_demote_candidate(candidate_data):
        reasons.append("demote_low_permanence_low_recurrence")
    elif should_propose_governance(candidate_data):
        reasons.append("eligible_for_governance_proposal")
    return reasons


# ---------------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------------

def review_candidate_projection(
    candidate_data: Dict[str, Any],
    *,
    candidate_key: str = "",
    profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compatibility wrapper for earlier Phase 3 collation imports.

    Returns a plain review dictionary assembled from the smaller policy helpers.
    ``memory.py`` remains responsible for applying these decisions to graph
    nodes and logs.
    """
    profile = dict(profile or {})
    action = str(candidate_data.get("integration_action", ""))
    org_state = organization_state_for_integration_action(action)
    demote = should_demote_candidate(candidate_data)
    propose = should_propose_governance(candidate_data)
    reasons = collation_review_reasons(profile, candidate_data)

    preservation_state = str(candidate_data.get("preservation_state", "provisional") or "provisional")
    if demote:
        preservation_state = "demoted"
    elif _as_float(candidate_data.get("permanence_score", 0.0)) >= 0.70:
        preservation_state = "protected"

    return {
        "candidate_key": str(candidate_key or candidate_data.get("candidate_id", "")),
        "organization_state": org_state,
        "preservation_state": preservation_state,
        "should_demote": bool(demote),
        "should_propose_governance": bool(propose),
        "sort_key": candidate_sort_key(candidate_data, candidate_key),
        "reasons": reasons,
    }


def summarize_collation_reviews(reviews: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Compatibility summary for collections of review dictionaries."""
    rows = list(reviews or [])
    return {
        "reviewed_candidates": len(rows),
        "demoted_candidates": sum(1 for r in rows if r.get("should_demote")),
        "proposal_candidates": sum(1 for r in rows if r.get("should_propose_governance")),
        "protected_candidates": sum(1 for r in rows if str(r.get("preservation_state")) == "protected"),
    }

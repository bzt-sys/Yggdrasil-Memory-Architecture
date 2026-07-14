"""Conservative governance promotion policy helpers."""

from __future__ import annotations

from .schemas import GovernanceProposalView


def promotion_readiness_score(proposal: GovernanceProposalView) -> float:
    """Compute a conservative promotion-readiness score.

    This is not a promotion decision. It is a pure helper for diagnostics and
    future extraction from memory.py.
    """
    score = 0.0
    score += min(5, int(proposal.recurrence_count or 1)) * 0.15
    score += min(5, int(proposal.evidence_candidate_count or 1)) * 0.12
    score += min(3, int(proposal.variant_count or 0)) * 0.08
    score += float(proposal.confidence or 0.0) * 0.35
    if proposal.verified:
        score += 0.50
    if proposal.durable:
        score += 0.25
    if proposal.status in {"discarded", "demoted"}:
        score -= 1.0
    return round(score, 6)


def should_hold_proposal(proposal: GovernanceProposalView, *, threshold: float = 1.15) -> bool:
    """Return True when a proposal should remain inactive pending evidence."""
    if proposal.activation_state == "active":
        return False
    if proposal.verified and proposal.evidence_candidate_count >= 3:
        return promotion_readiness_score(proposal) < threshold
    return True

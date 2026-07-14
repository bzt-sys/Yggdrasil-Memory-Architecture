"""Candidate ranking policy helpers.

The policy is intentionally conservative and repair-brief-first. It is meant to
mirror the direction of the Phase 2.23/2.24 work without taking over runtime
behavior yet.
"""

from __future__ import annotations

from typing import Iterable, List

from .schemas import CandidateView


def candidate_priority_score(candidate: CandidateView, *, current_step: int = 0) -> float:
    """Return a retrieval priority score for a candidate projection.

    Ordering preference:
    1. concrete repair evidence,
    2. investigation-linked candidates,
    3. useful/recurrent candidates,
    4. generic reflection text only as a fallback.
    """
    score = 0.0
    score += float(candidate.score or 0.0)
    score += float(candidate.salience or 0.0) * 0.35
    score += float(candidate.confidence or 0.0) * 0.25
    score += min(3, int(candidate.pedigree.recurrence_count or 1)) * 0.10

    if candidate.has_repair_brief:
        score += 1.25
    if candidate.has_investigation_evidence:
        score += 0.65
    if candidate.authority_state in {"active", "governance"}:
        score += 0.50
    if candidate.preservation_state in {"durable", "protected"}:
        score += 0.25
    if candidate.status in {"discarded", "demoted"}:
        score -= 2.0

    if current_step and candidate.last_seen_step:
        age = max(0, int(current_step) - int(candidate.last_seen_step))
        score -= min(0.50, age * 0.01)

    # Generic reflection candidates often pollute prompt execution; keep them as
    # low-priority fallbacks unless supported by investigation evidence.
    text = (candidate.text or "").lower()
    if "generic_episode_reflection" in text and not candidate.has_investigation_evidence:
        score -= 0.75
    if "success_reinforcement" in text and not candidate.has_repair_brief:
        score -= 0.25

    return round(score, 6)


def rank_candidate_views(candidates: Iterable[CandidateView], *, current_step: int = 0) -> List[CandidateView]:
    return sorted(
        list(candidates),
        key=lambda c: (candidate_priority_score(c, current_step=current_step), c.last_seen_step, c.candidate_id),
        reverse=True,
    )

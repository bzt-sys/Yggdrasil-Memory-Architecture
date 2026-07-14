"""Pure evidence policy helpers for future environment adapters."""

from __future__ import annotations

from .schemas import EvidenceObject


def evidence_strength(evidence: EvidenceObject) -> float:
    """Compute a conservative evidence-strength score.

    External/environment evidence is useful, but does not automatically create
    governance. The score is for ranking preservation and investigation effort.
    """
    score = 0.0
    score += max(0.0, min(1.0, evidence.source.reliability)) * 0.35
    score += max(0.0, min(1.0, evidence.verdict.confidence)) * 0.35
    if evidence.error_type:
        score += 0.10
    if evidence.failed_input is not None or evidence.expected is not None or evidence.observed is not None:
        score += 0.15
    if evidence.violated_assumption:
        score += 0.10
    return min(1.0, score)


def should_preserve_as_candidate(evidence: EvidenceObject, threshold: float = 0.45) -> bool:
    return evidence_strength(evidence) >= threshold

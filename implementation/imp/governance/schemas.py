"""Typed governance/locality projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class LocalityView:
    locality: str
    family: str
    locality_status: str = "existing"
    locality_depth: int = 1
    locality_confidence: float = 0.45
    assignment_reason: str = "unknown"
    path: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GovernanceProposalView:
    proposal_id: str
    candidate_id: str
    proposal_text: str
    locality: str
    family: str
    status: str = "proposed"
    activation_state: str = "inactive"
    promotion_status: str = "not_evaluated"
    confidence: float = 0.45
    recurrence_count: int = 1
    evidence_candidate_count: int = 1
    variant_count: int = 0
    verified: bool = False
    durable: bool = False


@dataclass(frozen=True)
class PromotionDecisionView:
    proposal_id: str
    decision: str = "hold"
    confidence: float = 0.0
    evidence_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    promoted_concept_key: Optional[str] = None

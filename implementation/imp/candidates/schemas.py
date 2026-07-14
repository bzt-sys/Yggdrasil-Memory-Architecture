"""Typed candidate-cache views used outside the memory engine.

These are lightweight projections, not authoritative storage records. They let
experiment/reporting/context code reason about candidate state without depending
on graph-node dictionaries directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PedigreeView:
    candidate_id: str
    pedigree_status: str = "unchecked"
    integration_action: str = "undecided"
    novelty_score: float = 0.0
    similarity_score: float = 0.0
    usefulness_score: float = 0.0
    recurrence_count: int = 1
    variant_of: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateView:
    candidate_id: str
    text: str
    locality: str = "family::general_constraint"
    family: str = "general_constraint"
    status: str = "cached"
    salience: float = 0.5
    confidence: float = 0.0
    score: float = 0.0
    created_step: int = 0
    last_seen_step: int = 0
    memory_tier: str = "hot_cache"
    preservation_state: str = "provisional"
    organization_state: str = "unorganized"
    authority_state: str = "none"
    collation_status: str = "pending"
    has_repair_brief: bool = False
    has_investigation_evidence: bool = False
    source: str = "candidate_cache"
    pedigree: PedigreeView = field(default_factory=lambda: PedigreeView(candidate_id=""))
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSelection:
    selected: List[CandidateView] = field(default_factory=list)
    suppressed: List[CandidateView] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    max_items: int = 4

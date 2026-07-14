"""Typed projections for slow-path collation.

These dataclasses intentionally mirror only the stable information needed for
slow/offline review. They should be constructed from graph nodes rather than
becoming a second source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CollationArtifact:
    artifact_id: str
    artifact_type: str = "candidate"
    text: str = ""
    locality: str = ""
    family: str = ""
    status: str = "cached"
    memory_tier: str = "hot_cache"
    authority_state: str = "none"
    preservation_state: str = "provisional"
    organization_state: str = "unorganized"
    recurrence_count: int = 1
    permanence_score: float = 0.0
    rarity_score: float = 0.0
    usefulness_score: float = 0.0
    outcome_severity: float = 0.0
    has_repair_brief: bool = False
    evidence_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollationReview:
    artifact_id: str
    decision: str = "hold"
    preservation_state: str = "provisional"
    organization_state: str = "unorganized"
    authority_state: str = "none"
    memory_tier: str = "hot_cache"
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class CollationSummary:
    reviewed: int = 0
    held: int = 0
    protected: int = 0
    proposed: int = 0
    demoted: int = 0
    promoted_ready: int = 0
    reasons: List[str] = field(default_factory=list)

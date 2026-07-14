"""Structured evidence schemas for internal/external adaptive coupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceSource:
    source_type: str = "environment"  # internal, environment, corpus, tool, human
    source_name: str = "unknown"
    reliability: float = 0.5
    timestamp: float = 0.0


@dataclass
class EvidenceVerdict:
    label: str = "unknown"  # supports, contradicts, ambiguous
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class EvidenceObject:
    evidence_id: str
    episode_id: str = ""
    task_id: str = ""
    attempt_number: int = 0
    evidence_type: str = "outcome"  # traceback, assertion, counterexample, doc, runtime
    raw: str = ""
    error_type: str = ""
    observed: Optional[Any] = None
    expected: Optional[Any] = None
    failed_input: Optional[Any] = None
    violated_assumption: str = ""
    source: EvidenceSource = field(default_factory=EvidenceSource)
    verdict: EvidenceVerdict = field(default_factory=EvidenceVerdict)
    metadata: Dict[str, Any] = field(default_factory=dict)

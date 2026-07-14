"""Typed investigation schemas.

These are deliberately lightweight and serializable. They mirror the current
runtime dictionaries used in memory.py without forcing an immediate migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class InvestigationEvidence:
    """Structured environmental or internal evidence available to investigation."""

    source: str = "unknown"
    evidence_type: str = "unknown"
    error_type: str = ""
    observed_error: str = ""
    failed_input: Optional[str] = None
    expected_output: Optional[str] = None
    observed_output: Optional[str] = None
    confidence: float = 0.45
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepairBrief:
    """Compact actionable repair context for pre-answer or retry use."""

    failure_type: str = "unknown"
    observed_error: str = ""
    contradicted_assumptions: List[str] = field(default_factory=list)
    minimal_experiments: List[str] = field(default_factory=list)
    next_attempt_directives: List[str] = field(default_factory=list)
    information_needs: List[str] = field(default_factory=list)
    confidence: float = 0.45
    status: str = "provisional"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreAnswerInvestigationRecord:
    """Prospective investigation/audit record before answer commitment."""

    record_id: str
    prompt_fingerprint: str = ""
    trigger_reasons: List[str] = field(default_factory=list)
    repair_briefs_considered: List[str] = field(default_factory=list)
    assumptions_to_check: List[str] = field(default_factory=list)
    minimal_mental_experiments: List[str] = field(default_factory=list)
    commitment_checklist: List[str] = field(default_factory=list)
    created_step: int = 0
    confidence: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationDecision:
    """Gate result for whether investigation should run."""

    should_investigate: bool = False
    mode: str = "none"
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

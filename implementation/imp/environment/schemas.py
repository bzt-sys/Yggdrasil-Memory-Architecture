"""Typed environment interface schemas.

These dataclasses define the boundary between internal developmental state and
external validation/interaction. They deliberately avoid depending on the graph
engine so adapters can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentAction:
    """A bounded action requested by an investigation or experiment runner."""

    action_id: str
    action_type: str
    domain: str = "generic"
    intent: str = "observe"
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "investigation"
    max_runtime_s: float = 2.0


@dataclass
class EnvironmentObservation:
    """Raw observation returned from an environment adapter."""

    observation_id: str
    action_id: str
    domain: str
    source: str
    status: str = "unknown"
    raw: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class EvidenceExtraction:
    """Structured evidence distilled from an environment observation."""

    evidence_id: str
    observation_id: str
    evidence_type: str
    source: str
    claim: str
    supports: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    error_type: str = ""
    failed_input: Optional[str] = None
    expected_output: Optional[str] = None
    observed_output: Optional[str] = None
    confidence: float = 0.5
    causal_status: str = "provisional"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentResult:
    """Adapter result containing raw observation plus structured evidence."""

    action: EnvironmentAction
    observation: EnvironmentObservation
    evidence: List[EvidenceExtraction] = field(default_factory=list)
    ok: bool = False

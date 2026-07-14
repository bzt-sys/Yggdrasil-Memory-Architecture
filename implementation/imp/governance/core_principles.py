from __future__ import annotations

"""Core cognitive principles for Yggdrasil governance.

This layer is intentionally different from learned/domain governance.  These
principles are seeded procedural heuristics for uncertainty handling,
assumption revision, hypothesis generation, evidence integration, and problem
 decomposition.  They are low-authority, inspectable, and activated by novelty
or uncertainty rather than only by failure.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


CORE_COGNITIVE_LOCALITY = "core::cognitive_principles"
CORE_COGNITIVE_FAMILY = "core_cognitive_principles"


@dataclass(frozen=True)
class CoreCognitivePrinciple:
    principle_id: str
    title: str
    directive: str
    activation_tags: List[str]
    checklist: List[str]

    def to_context_row(self, *, activation_score: float, reasons: List[str]) -> Dict[str, Any]:
        row = asdict(self)
        row.update({
            "locality": CORE_COGNITIVE_LOCALITY,
            "family": CORE_COGNITIVE_FAMILY,
            "authority_state": "seeded_core_low_authority",
            "activation_score": round(float(activation_score), 4),
            "activation_reasons": list(reasons),
        })
        return row


CORE_COGNITIVE_PRINCIPLES: List[CoreCognitivePrinciple] = [
    CoreCognitivePrinciple(
        principle_id="assumption_identification",
        title="Identify assumptions before commitment",
        directive=(
            "When the situation is novel or uncertain, identify the assumptions "
            "your answer relies on before committing to a solution."
        ),
        activation_tags=["novelty", "uncertainty", "semantic_failure", "hidden_test"],
        checklist=[
            "What am I assuming the task/environment means?",
            "Which assumption would most likely cause failure if wrong?",
            "Can I choose a solution that depends on fewer assumptions?",
        ],
    ),
    CoreCognitivePrinciple(
        principle_id="problem_decomposition",
        title="Decompose the problem into testable parts",
        directive=(
            "Break the problem into contract, inputs, outputs, boundary cases, "
            "and mechanism before generating the final action."
        ),
        activation_tags=["complexity", "novelty", "coding", "planning"],
        checklist=[
            "What is the exact success condition?",
            "What are the smallest and boundary cases?",
            "What subproblem must be solved before the whole task can work?",
        ],
    ),
    CoreCognitivePrinciple(
        principle_id="hypothesis_generation",
        title="Generate alternatives when confidence is weak",
        directive=(
            "When the evidence is incomplete or prior attempts failed, generate at "
            "least one alternative interpretation or strategy instead of repeating "
            "the most familiar pattern."
        ),
        activation_tags=["uncertainty", "repeated_failure", "investigation"],
        checklist=[
            "What is the most likely explanation?",
            "What is a plausible alternative?",
            "Which alternative is easiest to test mentally or environmentally?",
        ],
    ),
    CoreCognitivePrinciple(
        principle_id="evidence_comparison",
        title="Compare action against environment signal",
        directive=(
            "Treat environment feedback as evidence about the prior action. Compare "
            "what was attempted against what the environment signaled before revising."
        ),
        activation_tags=["environment_signal", "failure", "debugging", "feedback"],
        checklist=[
            "What did I do?",
            "What did the environment signal?",
            "Which part of the action is most contradicted by that signal?",
        ],
    ),
    CoreCognitivePrinciple(
        principle_id="uncertainty_calibration",
        title="Prefer calibrated uncertainty over confident guessing",
        directive=(
            "When substrate match, prompt interpretation, or generated strategy is "
            "uncertain, avoid overconfident shortcuts; use verification, direct "
            "derivation, or investigation."
        ),
        activation_tags=["uncertainty", "low_match", "novelty", "overconfidence"],
        checklist=[
            "Do I have enough evidence for this interpretation?",
            "Could a simpler direct derivation reduce uncertainty?",
            "Should I ask, search, test, or mentally simulate before committing?",
        ],
    ),
]


def assess_core_cognitive_activation(
    *,
    prompt_text: str = "",
    routing_state: Dict[str, Any] | None = None,
    prompt_risk: Dict[str, Any] | None = None,
    candidate_count: int = 0,
    investigation_count: int = 0,
    governance_count: int = 0,
) -> Dict[str, Any]:
    """Return a bounded activation decision for seeded cognitive governance.

    Activation is based on uncertainty/novelty rather than failure alone.  This
    is deliberately heuristic and auditable; later executive scheduling can
    replace it with a resource-allocation policy.
    """
    routing_state = routing_state or {}
    prompt_risk = prompt_risk or {}
    reasons: List[str] = []
    score = 0.0

    risk = float(prompt_risk.get("risk", 0.0) or 0.0)
    threshold = float(prompt_risk.get("threshold", 0.35) or 0.35)
    if risk >= max(0.25, threshold):
        score += 0.35
        reasons.append("prompt_risk_exceeds_threshold")

    domain_conf = float(routing_state.get("domain_confidence", 0.0) or 0.0)
    if domain_conf <= 0.45:
        score += 0.18
        reasons.append("low_domain_confidence")

    if routing_state.get("clarification_needed"):
        score += 0.20
        reasons.append("clarification_needed")

    if routing_state.get("operational_intent"):
        score += 0.10
        reasons.append("operational_intent")

    if int(candidate_count) <= 0 and int(governance_count) <= 0:
        score += 0.12
        reasons.append("low_substrate_match")

    if int(investigation_count) > 0:
        score += min(0.20, 0.07 * int(investigation_count))
        reasons.append("similar_investigations_available")

    low = (prompt_text or "").lower()
    novelty_markers = ["new", "unknown", "unfamiliar", "complete", "correctly", "hidden", "debug", "error", "why"]
    marker_hits = [m for m in novelty_markers if m in low]
    if marker_hits:
        score += min(0.15, 0.03 * len(marker_hits))
        reasons.append("novelty_or_uncertainty_markers:" + ",".join(marker_hits[:4]))

    score = max(0.0, min(1.0, score))
    active = score >= 0.35
    return {
        "active": active,
        "activation_score": round(score, 4),
        "reasons": reasons or ["below_activation_threshold"],
        "locality": CORE_COGNITIVE_LOCALITY,
        "family": CORE_COGNITIVE_FAMILY,
    }


def select_core_cognitive_principles(
    *,
    prompt_text: str = "",
    routing_state: Dict[str, Any] | None = None,
    prompt_risk: Dict[str, Any] | None = None,
    candidate_count: int = 0,
    investigation_count: int = 0,
    governance_count: int = 0,
    k: int = 3,
) -> Dict[str, Any]:
    activation = assess_core_cognitive_activation(
        prompt_text=prompt_text,
        routing_state=routing_state,
        prompt_risk=prompt_risk,
        candidate_count=candidate_count,
        investigation_count=investigation_count,
        governance_count=governance_count,
    )
    if not activation.get("active"):
        return {**activation, "principles": []}

    reason_blob = " ".join(activation.get("reasons") or []).lower()
    scored: List[tuple[float, str, CoreCognitivePrinciple]] = []
    for p in CORE_COGNITIVE_PRINCIPLES:
        s = float(activation["activation_score"])
        for tag in p.activation_tags:
            if tag.lower() in reason_blob or tag.lower() in (prompt_text or "").lower():
                s += 0.08
        if int(investigation_count) > 0 and p.principle_id in {"evidence_comparison", "hypothesis_generation"}:
            s += 0.12
        if int(candidate_count) <= 0 and p.principle_id in {"assumption_identification", "uncertainty_calibration"}:
            s += 0.08
        scored.append((min(1.0, s), p.principle_id, p))

    scored.sort(key=lambda x: (-x[0], x[1]))
    rows = [p.to_context_row(activation_score=s, reasons=list(activation.get("reasons") or [])) for s, _, p in scored[: max(1, int(k))]]
    return {**activation, "principles": rows}

from __future__ import annotations

"""Seeded reasoning paradigms for Yggdrasil.

A paradigm is not domain governance and not a benchmark-specific rule.  It is a
low-authority reasoning operator: a compact procedure for shaping the
pre-answer investigation workspace.  Selection is deterministic and auditable;
later developmental policy can learn when each operator helps.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple


REASONING_PARADIGM_LOCALITY = "core::reasoning_paradigms"
REASONING_PARADIGM_FAMILY = "seeded_reasoning_paradigms"


@dataclass(frozen=True)
class ReasoningParadigm:
    paradigm_id: str
    title: str
    intent: str
    activation_tags: List[str]
    procedure: List[str]
    workspace_outputs: List[str]
    failure_modes: List[str]

    def to_context_row(self, *, score: float, reasons: List[str], matched_tags: List[str]) -> Dict[str, Any]:
        row = asdict(self)
        row.update({
            "locality": REASONING_PARADIGM_LOCALITY,
            "family": REASONING_PARADIGM_FAMILY,
            "authority_state": "seeded_operator_low_authority",
            "selection_score": round(float(score), 4),
            "selection_reasons": list(reasons),
            "matched_tags": sorted(set(matched_tags)),
        })
        return row


SEEDED_REASONING_PARADIGMS: List[ReasoningParadigm] = [
    ReasoningParadigm(
        paradigm_id="polya_gap_analysis",
        title="Polya-style gap analysis",
        intent="Use when the objective is clear but the path from knowns to unknowns is not yet secure.",
        activation_tags=["unknown", "uncertainty", "gap", "problem", "solve", "derive", "boundary", "contract"],
        procedure=[
            "Restate the objective and success condition.",
            "Separate what is known, unknown, and constrained.",
            "Look for a simpler related problem, transformation, or invariant.",
            "Construct the solution and then look back for boundary violations.",
        ],
        workspace_outputs=["knowns", "unknowns", "constraints", "candidate_transformation", "look_back_checks"],
        failure_modes=["Can become too local when the environment requires empirical testing or open-ended exploration."],
    ),
    ReasoningParadigm(
        paradigm_id="scientific_inquiry",
        title="Scientific inquiry",
        intent="Use when multiple explanations are plausible and evidence must discriminate between them.",
        activation_tags=["hypothesis", "experiment", "evidence", "observe", "test", "prediction", "uncertainty", "signal"],
        procedure=[
            "List plausible hypotheses that explain the current evidence.",
            "Derive predictions or discriminating observations for each hypothesis.",
            "Choose the smallest informative test or evidence comparison.",
            "Revise the hypothesis set after the signal is interpreted.",
        ],
        workspace_outputs=["hypotheses", "predictions", "minimal_tests", "evidence_comparison", "revision_rule"],
        failure_modes=["Can over-investigate when a direct construction is already available."],
    ),
    ReasoningParadigm(
        paradigm_id="differential_debugging",
        title="Differential debugging",
        intent="Use when there is a committed action, an expected result, and a contradictory signal.",
        activation_tags=["failure", "error", "debug", "signal", "expected", "observed", "contradiction", "repair"],
        procedure=[
            "State the expected behavior and the observed signal without merging them.",
            "Identify the smallest difference between attempted action and required behavior.",
            "Locate the assumption most likely responsible for that difference.",
            "Make the smallest repair and define the next check.",
        ],
        workspace_outputs=["expected_vs_observed", "difference", "violated_assumption", "minimal_repair", "next_check"],
        failure_modes=["Can focus too narrowly on symptoms if the original problem framing was wrong."],
    ),
    ReasoningParadigm(
        paradigm_id="systems_decomposition",
        title="Systems decomposition",
        intent="Use when the situation contains interacting components, feedback loops, or multi-stage effects.",
        activation_tags=["system", "interaction", "loop", "component", "architecture", "pipeline", "feedback", "state"],
        procedure=[
            "Identify the components, boundaries, and information flows.",
            "Separate local effects from cross-component feedback.",
            "Find the state variables that must remain coherent.",
            "Choose an intervention that preserves the larger system objective.",
        ],
        workspace_outputs=["components", "boundaries", "flows", "state_variables", "intervention_points"],
        failure_modes=["Can become too abstract when the task only needs a direct concrete answer."],
    ),
    ReasoningParadigm(
        paradigm_id="decision_analysis",
        title="Decision analysis",
        intent="Use when the main problem is selecting among actions under constraints and tradeoffs.",
        activation_tags=["choose", "tradeoff", "risk", "cost", "benefit", "policy", "priority", "option"],
        procedure=[
            "Enumerate viable options.",
            "State constraints, risks, costs, and likely benefits for each option.",
            "Prefer robust options when uncertainty is high.",
            "Record what evidence would change the decision.",
        ],
        workspace_outputs=["options", "constraints", "tradeoffs", "robust_choice", "change_conditions"],
        failure_modes=["Can delay action if options are over-enumerated without improving the decision."],
    ),
]


def _token_hits(text: str, tags: Iterable[str]) -> List[str]:
    low = (text or "").lower()
    return [str(t) for t in tags if str(t).lower() in low]


def _context_signal_blob(*, routing_state: Dict[str, Any], prompt_risk: Dict[str, Any], investigations: List[Dict[str, Any]], candidates: List[Dict[str, Any]], governance_proposals: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    parts.extend(str(v) for v in (routing_state or {}).values())
    parts.extend(str(v) for v in (prompt_risk or {}).values())
    for row in list(investigations or [])[:4]:
        if isinstance(row, dict):
            parts.append(str(row.get("summary") or row.get("repair_brief") or row.get("text") or row))
        else:
            parts.append(str(row))
    for row in list(candidates or [])[:4]:
        if isinstance(row, dict):
            parts.append(str(row.get("text") or row.get("summary") or row))
        else:
            parts.append(str(row))
    for row in list(governance_proposals or [])[:4]:
        if isinstance(row, dict):
            parts.append(str(row.get("proposal_text") or row.get("summary") or row))
        else:
            parts.append(str(row))
    return "\n".join(parts)


def _phase_bonus(phase: str, paradigm_id: str, *, investigations: List[Dict[str, Any]], outcome_signal: Dict[str, Any], committed_action: str) -> Tuple[float, List[str]]:
    """Return generic phase-specific compatibility evidence."""
    phase = (phase or "pre_action").strip().lower()
    reasons: List[str] = []
    bonus = 0.0
    signal_text = " ".join(str(v) for v in (outcome_signal or {}).values()).lower()
    action_present = bool((committed_action or "").strip())
    outcome_present = bool(outcome_signal)

    if phase in {"post_action", "post_answer", "diagnostic", "reflection"}:
        if outcome_present:
            bonus += 0.10
            reasons.append("post_action_environment_signal_available")
        if action_present:
            bonus += 0.08
            reasons.append("post_action_committed_action_available")
        if investigations:
            bonus += 0.06
            reasons.append("post_action_investigation_chain_available")
        if paradigm_id == "differential_debugging" and (outcome_present or action_present):
            bonus += 0.18
            reasons.append("expected_observed_comparison_needed")
        if paradigm_id == "scientific_inquiry" and ("ambiguous" in signal_text or "unknown" in signal_text or "uncertain" in signal_text):
            bonus += 0.12
            reasons.append("ambiguous_signal_requires_hypothesis_discrimination")
        if paradigm_id == "systems_decomposition" and ("cascade" in signal_text or "interaction" in signal_text or "state" in signal_text):
            bonus += 0.10
            reasons.append("signal_mentions_interacting_state")
    else:
        if paradigm_id in {"polya_gap_analysis", "systems_decomposition", "decision_analysis"}:
            bonus += 0.05
            reasons.append("pre_action_planning_operator")
    return bonus, reasons


def select_reasoning_paradigm(
    *,
    prompt_text: str = "",
    routing_state: Dict[str, Any] | None = None,
    prompt_risk: Dict[str, Any] | None = None,
    candidates: List[Dict[str, Any]] | None = None,
    investigations: List[Dict[str, Any]] | None = None,
    governance_proposals: List[Dict[str, Any]] | None = None,
    core_activation: Dict[str, Any] | None = None,
    phase: str = "pre_action",
    committed_action: str = "",
    outcome_signal: Dict[str, Any] | None = None,
    k: int = 1,
) -> Dict[str, Any]:
    """Deterministically select one seeded reasoning operator.

    The selector uses only generic situation evidence: uncertainty, retrieved
    developmental artifacts, committed action availability, and environment
    signal shape. It deliberately avoids benchmark names, programming-language
    names, and hard-coded exception classes. Older callers may pass ``k``; the
    current default behavior intentionally caps active injection to one paradigm
    so the context space is not over-formatted.
    """
    routing_state = routing_state or {}
    prompt_risk = prompt_risk or {}
    candidates = list(candidates or [])
    investigations = list(investigations or [])
    governance_proposals = list(governance_proposals or [])
    core_activation = core_activation or {}
    outcome_signal = outcome_signal or {}
    phase = (phase or "pre_action").strip().lower()

    global_reasons: List[str] = [f"phase::{phase}"]
    base = 0.16
    risk = float(prompt_risk.get("risk", 0.0) or 0.0)
    if risk >= float(prompt_risk.get("threshold", 0.35) or 0.35):
        base += 0.18
        global_reasons.append("uncertainty_or_risk_above_threshold")
    if investigations:
        base += min(0.18, 0.06 * len(investigations))
        global_reasons.append("prior_investigation_available")
    if candidates:
        base += min(0.12, 0.04 * len(candidates))
        global_reasons.append("candidate_evidence_available")
    if governance_proposals:
        base += 0.08
        global_reasons.append("governance_proposal_available")
    if routing_state.get("operational_intent"):
        base += 0.08
        global_reasons.append("operational_intent")
    if routing_state.get("clarification_needed"):
        base += 0.10
        global_reasons.append("clarification_needed")
    if core_activation.get("active"):
        base += 0.10
        global_reasons.append("core_cognition_active")
    if outcome_signal:
        base += 0.12
        global_reasons.append("environment_or_outcome_signal_available")
    if committed_action:
        base += 0.06
        global_reasons.append("committed_action_available")

    evidence_blob = _context_signal_blob(
        routing_state=routing_state,
        prompt_risk=prompt_risk,
        investigations=investigations,
        candidates=candidates,
        governance_proposals=governance_proposals,
    )
    outcome_blob = "\n".join(f"{k}: {v}" for k, v in sorted((outcome_signal or {}).items()))
    combined = f"{prompt_text}\n{committed_action}\n{outcome_blob}\n{evidence_blob}"

    scored: List[Tuple[float, str, ReasoningParadigm, List[str], List[str]]] = []
    for paradigm in SEEDED_REASONING_PARADIGMS:
        matched = _token_hits(combined, paradigm.activation_tags)
        local_reasons = list(global_reasons)
        if matched:
            local_reasons.append("activation_tags_matched")
        score = base + min(0.35, 0.07 * len(matched))
        pid = paradigm.paradigm_id
        phase_delta, phase_reasons = _phase_bonus(
            phase,
            pid,
            investigations=investigations,
            outcome_signal=outcome_signal,
            committed_action=committed_action,
        )
        score += phase_delta
        local_reasons.extend(phase_reasons)
        # Generic context-shape bonuses.
        if investigations and pid == "differential_debugging":
            score += 0.10
            local_reasons.append("repair_evidence_suits_differential_comparison")
        if risk >= 0.5 and pid in {"polya_gap_analysis", "scientific_inquiry"}:
            score += 0.08
            local_reasons.append("high_uncertainty_suits_gap_or_hypothesis_reasoning")
        if routing_state.get("clarification_needed") and pid == "polya_gap_analysis":
            score += 0.08
            local_reasons.append("clarification_gap_suits_polya")
        if governance_proposals and pid == "decision_analysis":
            score += 0.05
            local_reasons.append("policy_options_suit_decision_analysis")
        scored.append((min(1.0, score), pid, paradigm, matched, local_reasons))

    scored.sort(key=lambda item: (-item[0], item[1]))
    # Context injection is intentionally single-paradigm by default.  The
    # selector still records scores/reasons so future uncertainty-aware expansion
    # can request comparison, but it does not crowd the actor context with an
    # ordered stack during normal operation.
    active = bool(scored and scored[0][0] >= 0.28)
    selected = [
        scored[0][2].to_context_row(score=scored[0][0], reasons=scored[0][4], matched_tags=scored[0][3])
    ] if active else []
    if selected:
        selected[0]["stack_rank"] = 1
        selected[0]["stack_role"] = "primary"
    selected_ids = [row.get("paradigm_id") for row in selected]
    return {
        "active": active,
        "phase": phase,
        "locality": REASONING_PARADIGM_LOCALITY,
        "family": REASONING_PARADIGM_FAMILY,
        "selection_mode": "single_context_aware_paradigm",
        "selection_reasons": global_reasons if global_reasons else ["default_low_intensity_strategy_selection"],
        "selected": selected,
        "selected_reasoning_paradigm_id": selected_ids[0] if selected_ids else None,
        # Kept as a one-item compatibility field for existing audit/trace code.
        "ordered_stack_ids": selected_ids,
    }

from __future__ import annotations

"""Lightweight uncertainty routing for HumanEval-style experiment harnesses.

This module is intentionally small and mostly deterministic. The goal is not to
teach the model a reasoning framework; it is to decide whether the substrate
should stay out of the model's way, inject evidence, or allow a bounded private
candidate/test/revision loop before committing an answer.
"""

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Callable, Optional


@dataclass
class UncertaintyGateDecision:
    confidence: float
    route: str
    context_enabled: bool
    pre_answer_enabled: bool
    deliberation_enabled: bool
    max_deliberation_rounds: int
    reasons: list[str]
    retrieval_budget: dict[str, int]
    suggested_temperature: float = 0.2
    model_probe_used: bool = False
    model_probe_raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    return max(0.0, min(1.0, v))


def parse_gate_json(text: str) -> dict[str, Any]:
    """Parse a permissive JSON object from a short model uncertainty probe."""
    raw = text or ""
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def heuristic_confidence(prompt: str, *, previous_failures: Optional[list[dict[str, Any]]] = None) -> tuple[float, list[str]]:
    """Estimate task risk from observable prompt features and retry state.

    This is deliberately conservative and transparent. It should be replaced or
    calibrated with empirical gate-outcome data, not hidden inside prompt prose.
    """
    text = prompt or ""
    lower = text.lower()
    reasons: list[str] = []
    confidence = 0.72

    if previous_failures:
        confidence -= min(0.35, 0.16 * len(previous_failures))
        reasons.append("previous_failed_attempts")

    branch_words = ["if", "except", "unless", "empty", "none", "negative", "duplicate", "sorted", "order", "substring", "subsequence", "prime", "palindrome"]
    hits = sum(1 for w in branch_words if w in lower)
    if hits >= 4:
        confidence -= 0.10
        reasons.append("many_edge_case_terms")
    elif hits >= 2:
        confidence -= 0.05
        reasons.append("some_edge_case_terms")

    if len(text) > 1800:
        confidence -= 0.08
        reasons.append("long_prompt")
    if len(text) < 350:
        confidence -= 0.04
        reasons.append("sparse_prompt")

    if any(tok in lower for tok in ("matrix", "graph", "tree", "recursive", "dynamic", "combinations", "permutations")):
        confidence -= 0.10
        reasons.append("algorithmic_complexity_signal")

    if any(tok in lower for tok in ("float", "precision", "tolerance")):
        confidence -= 0.08
        reasons.append("numeric_tolerance_signal")

    if not reasons:
        reasons.append("no_obvious_uncertainty_signal")
    return _clip01(confidence), reasons


def decide_uncertainty_gate(
    prompt: str,
    *,
    previous_failures: Optional[list[dict[str, Any]]] = None,
    model_probe: Optional[Callable[[str], str]] = None,
    enable_model_probe: bool = False,
    high_confidence_threshold: float = 0.72,
    low_confidence_threshold: float = 0.45,
    deliberation_threshold: float = 0.62,
    max_deliberation_rounds: int = 2,
) -> UncertaintyGateDecision:
    """Return a routing decision for context and private deliberation.

    Routes:
    - natural_first_pass: very small substrate dosage, no pre-answer investigation.
    - substrate_context: medium substrate dosage, no private deliberation.
    - deliberate_then_commit: larger evidence dosage plus bounded private candidate/test/revision.

    The gate is intentionally graded, not binary. High confidence reduces
    retrieval and lowers suggested temperature; it does not suppress all memory.
    """
    confidence, reasons = heuristic_confidence(prompt, previous_failures=previous_failures)
    raw = ""
    used_probe = False

    if enable_model_probe and model_probe is not None:
        probe_prompt = (
            "Assess your confidence for solving this Python programming task. "
            "Do not solve it. Return JSON only with keys: confidence number 0-1, "
            "needs_memory boolean, needs_experiment boolean, risk_factors array of short strings.\n\n"
            + prompt[:3500]
        )
        try:
            raw = model_probe(probe_prompt)
            parsed = parse_gate_json(raw)
            if parsed:
                model_conf = _clip01(parsed.get("confidence"), confidence)
                # Blend rather than trust self-report absolutely; self-confidence is not calibrated.
                confidence = _clip01((0.55 * confidence) + (0.45 * model_conf), confidence)
                used_probe = True
                if parsed.get("needs_memory"):
                    reasons.append("model_probe_needs_memory")
                if parsed.get("needs_experiment"):
                    reasons.append("model_probe_needs_experiment")
                for rf in parsed.get("risk_factors") or []:
                    if rf:
                        reasons.append("probe:" + str(rf)[:80])
        except Exception as exc:
            reasons.append(f"model_probe_failed:{type(exc).__name__}")

    if previous_failures:
        # After an observed failure, prefer evidence-native substrate repair over natural repeat.
        route = "deliberate_then_commit"
        context_enabled = True
        pre_answer_enabled = True
        deliberation_enabled = True
        retrieval_budget = {"candidates": 4, "investigations": 3, "hypotheses": 3, "governance": 1, "recent": 2, "core_principles": 0}
        suggested_temperature = 0.15
        reasons.append("failure_forces_evidence_route")
    elif confidence >= high_confidence_threshold:
        route = "natural_first_pass"
        context_enabled = True
        pre_answer_enabled = False
        deliberation_enabled = False
        retrieval_budget = {"candidates": 1, "investigations": 0, "hypotheses": 1, "governance": 0, "recent": 1, "core_principles": 0}
        suggested_temperature = 0.10
    elif confidence < low_confidence_threshold or confidence < deliberation_threshold:
        route = "deliberate_then_commit"
        context_enabled = True
        pre_answer_enabled = False
        deliberation_enabled = True
        retrieval_budget = {"candidates": 3, "investigations": 2, "hypotheses": 3, "governance": 1, "recent": 2, "core_principles": 0}
        suggested_temperature = 0.20
    else:
        route = "substrate_context"
        context_enabled = True
        pre_answer_enabled = False
        deliberation_enabled = False
        retrieval_budget = {"candidates": 2, "investigations": 1, "hypotheses": 2, "governance": 0, "recent": 2, "core_principles": 0}
        suggested_temperature = 0.15

    return UncertaintyGateDecision(
        confidence=round(confidence, 3),
        route=route,
        context_enabled=bool(context_enabled),
        pre_answer_enabled=bool(pre_answer_enabled),
        deliberation_enabled=bool(deliberation_enabled),
        max_deliberation_rounds=max(1, int(max_deliberation_rounds)),
        reasons=list(dict.fromkeys(reasons)),
        retrieval_budget=dict(retrieval_budget),
        suggested_temperature=float(suggested_temperature),
        model_probe_used=bool(used_probe),
        model_probe_raw=raw[:1200],
    )

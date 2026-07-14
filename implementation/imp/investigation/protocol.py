"""Reusable investigation protocol helpers.

This module contains pure helpers only. It is safe to import from memory.py or
agent_session.py later because it has no dependency on the graph engine.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .schemas import RepairBrief


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def build_repair_brief(evidence: Dict[str, Any], prior_hypotheses: Iterable[str] | None = None) -> RepairBrief:
    """Create a bounded repair brief from structured evidence.

    This mirrors the current memory.py behavior but keeps the policy isolated so
    later patches can delegate to it without changing external behavior.
    """

    error_type = str(evidence.get("error_type") or evidence.get("failure_type") or "unknown")
    observed_error = str(evidence.get("observed_error") or evidence.get("error") or "")
    et = error_type.lower()
    prior = [str(h) for h in (prior_hypotheses or []) if str(h).strip()]

    contradicted: List[str] = []
    experiments: List[str] = []
    directives: List[str] = []
    needs: List[str] = []

    if "nameerror" in et:
        contradicted.append("Runtime referenced an undefined symbol.")
        experiments.append("Use the missing symbol from the environment error as the repair target.")
        directives.append("Ensure referenced symbols are defined, imported, or removed.")
        needs.append("Identify the exact missing symbol from the environment error.")
    elif "syntaxerror" in et:
        contradicted.append("Evaluator could not parse the extracted Python.")
        experiments.append("Inspect the parser error and extracted code surface.")
        directives.append("Return parseable executable Python.")
        needs.append("Determine whether truncation or extraction caused the syntax failure.")
    elif "assertionerror" in et:
        contradicted.append("Executable code contradicted at least one observed assertion/test.")
        experiments.append("Use the failing assertion or diagnostic case as the repair target.")
        directives.append("Revise behavior to satisfy the observed failing case.")
        needs.append("A concrete failing input/expected/observed triple would improve causal diagnosis.")
    elif "typeerror" in et:
        contradicted.append("Runtime types contradicted an operation or call in the attempted code.")
        experiments.append("Locate the incompatible operation from the traceback text.")
        directives.append("Revise the operation so it matches the observed runtime types.")
        needs.append("Inspect the exact operand or function call named by the traceback.")
    else:
        contradicted.append("The evaluator produced an unclassified failure signal.")
        experiments.append("Use the raw signal as the narrowest available repair target.")
        directives.append("Make one repair grounded in the observed signal.")
        needs.append("Collect error type, traceback, or a counterexample if available.")

    for h in prior[:4]:
        if h not in needs:
            needs.append(f"Prior unresolved hypothesis: {h}")

    return RepairBrief(
        failure_type=error_type,
        observed_error=observed_error[:800],
        contradicted_assumptions=contradicted[:4],
        minimal_experiments=experiments[:4],
        next_attempt_directives=directives[:4],
        information_needs=needs[:5],
        confidence=float(evidence.get("confidence", 0.55) or 0.55),
    )


def build_pre_answer_checklist(repair_briefs: Iterable[Dict[str, Any]] | None = None) -> Dict[str, List[str]]:
    """Create a small pre-answer checklist from retrieved repair briefs."""

    assumptions = [
        "The final answer must satisfy the exact user/task contract, not a nearby generic pattern.",
        "All referenced helpers, imports, and symbols must exist in the execution surface.",
    ]
    experiments = [
        "Mentally execute the candidate solution on the simplest valid input.",
        "Check at least one boundary or adversarial input implied by the prompt.",
    ]
    checklist = [
        "Does the answer directly implement the requested signature/contract?",
        "Does the solution avoid undeclared dependencies or helpers?",
        "Would the proposed answer survive the retrieved repair directives?",
    ]

    for brief in repair_briefs or []:
        assumptions.extend(_as_list(brief.get("contradicted_assumptions"))[:2])
        experiments.extend(_as_list(brief.get("minimal_experiments"))[:2])
        checklist.extend(_as_list(brief.get("next_attempt_directives"))[:2])

    def dedupe(items: List[str], limit: int) -> List[str]:
        out: List[str] = []
        for item in items:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
            if len(out) >= limit:
                break
        return out

    return {
        "assumptions_to_check": dedupe(assumptions, 8),
        "minimal_mental_experiments": dedupe(experiments, 8),
        "commitment_checklist": dedupe(checklist, 10),
    }

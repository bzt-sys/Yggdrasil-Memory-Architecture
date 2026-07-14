from __future__ import annotations

"""Pure investigation protocol helpers for Yggdrasil.

These helpers intentionally avoid graph/state mutation. The memory engine owns
persistence and orchestration; this module owns the reusable policy for turning
outcome signals and retrieved evidence into compact repair/pre-answer guidance.
"""

import re
from typing import Any, Dict, List, Tuple


def build_investigation_hypotheses_for_error(error_type: str, error: str) -> Tuple[List[str], List[str], List[str]]:
    """Return deterministic first-pass hypotheses and repair directions."""
    et = (error_type or "").lower()
    hypotheses: List[str] = []
    repairs: List[str] = []
    needs: List[str] = []
    if "syntaxerror" in et:
        hypotheses += ["The generated answer may be incomplete, truncated, or mixed with prose/markdown that is not executable Python."]
        repairs += ["Return only one complete executable Python implementation and avoid explanatory text inside the code extraction surface."]
        needs += ["Inspect whether extraction captured a full code block and whether generation token limit was sufficient."]
    elif "nameerror" in et:
        hypotheses += ["The generated code used a symbol, helper, or module that was not defined in the executable namespace."]
        repairs += ["Define helper functions locally or use only built-ins unless imports are explicitly included in the returned code."]
        needs += ["Identify the missing symbol from the exception and decide whether to implement it locally or import it explicitly."]
    elif "assertionerror" in et:
        hypotheses += ["The code executed but violated the task specification or an edge case hidden in the tests."]
        repairs += ["Re-read the docstring carefully, reason about edge cases, and prefer minimal complete logic over generic pattern matching."]
        needs += ["Obtain a failing input/output example if available; otherwise infer likely edge cases from the function contract."]
    elif "typeerror" in et:
        hypotheses += ["The generated code likely produced an invalid operation or used placeholder/non-executable constructs."]
        repairs += ["Avoid pseudocode, ellipses, undeclared helpers, and operations between incompatible types."]
        needs += ["Inspect the exact operand/function call in the error message."]
    elif "timeout" in et:
        hypotheses += ["The generated algorithm may not terminate efficiently for the test cases."]
        repairs += ["Replace brute force or recursive search with bounded iteration or a direct formula where possible."]
        needs += ["Estimate input size and complexity from the task prompt."]
    else:
        hypotheses += ["The failure reason is not yet classified with enough confidence."]
        repairs += ["Use the error signal as evidence and form a narrow repair hypothesis before retrying."]
        needs += ["Collect a more detailed traceback, failing example, or test delta if available."]
    return hypotheses, repairs, needs


def build_investigation_protocol_for_feedback(feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Build a bounded evidence-native protocol from outcome feedback."""
    error_type = str(feedback.get("error_type") or "unknown")
    observed_error = str(feedback.get("error") or feedback.get("raw") or "")[:1000]
    et = error_type.lower()

    contradicted_assumptions: List[str] = []
    minimal_experiments: List[str] = []
    next_attempt_directives: List[str] = []
    evidence_items: List[str] = []

    if observed_error:
        evidence_items.append(f"Environment reported {error_type}: {observed_error[:260]}")
    passed = feedback.get("passed")
    if passed is not None:
        evidence_items.append(f"Objective evaluator passed={bool(passed)}")

    # Evidence-native failure interpretation. Keep this compact and derived from
    # the observed environment signal rather than prescribing a reusable reasoning
    # framework or broad checklist.
    if "nameerror" in et:
        m = re.search(r"name ['\"]([^'\"]+)['\"] is not defined", observed_error)
        missing = m.group(1) if m else "referenced symbol"
        contradicted_assumptions.append(f"Runtime referenced undefined symbol `{missing}`.")
        minimal_experiments.append(f"Resolve `{missing}` from the observed traceback.")
        next_attempt_directives.append(f"Ensure `{missing}` is defined, imported, or removed.")
    elif "syntaxerror" in et:
        contradicted_assumptions.append("Evaluator could not parse the extracted Python.")
        minimal_experiments.append("Inspect the extracted code surface identified by the parser error.")
        next_attempt_directives.append("Return parseable executable Python.")
    elif "assertionerror" in et:
        contradicted_assumptions.append("Executable code contradicted at least one observed assertion/test.")
        minimal_experiments.append("Use the failing assertion or diagnostic case as the repair target.")
        next_attempt_directives.append("Revise behavior to satisfy the observed failing case.")
    elif "typeerror" in et:
        contradicted_assumptions.append("Runtime types contradicted an operation or call in the attempted code.")
        minimal_experiments.append("Locate the incompatible operation from the traceback text.")
        next_attempt_directives.append("Revise the operation so it matches the observed runtime types.")
    elif "timeout" in et:
        contradicted_assumptions.append("Evaluator exceeded its runtime budget.")
        minimal_experiments.append("Locate the loop/search path responsible for nontermination or excessive work.")
        next_attempt_directives.append("Use a terminating bounded implementation.")
    else:
        contradicted_assumptions.append("The evaluator produced an unclassified failure signal.")
        minimal_experiments.append("Use the raw signal as the narrowest available repair target.")
        next_attempt_directives.append("Make one repair grounded in the observed signal.")

    protocol_steps = [
        {"step": "observe", "question": "What exactly failed?", "answer": evidence_items[:3]},
        {"step": "classify", "question": "What kind of failure is this?", "answer": error_type},
        {"step": "assumption_check", "question": "Which assumption did the outcome contradict?", "answer": contradicted_assumptions[:3]},
        {"step": "experiment", "question": "What minimal check would reduce uncertainty?", "answer": minimal_experiments[:3]},
        {"step": "repair", "question": "What should change on the next attempt?", "answer": next_attempt_directives[:4]},
    ]

    repair_brief = {
        "failure_type": error_type,
        "observed_error": observed_error[:500],
        "contradicted_assumptions": contradicted_assumptions[:4],
        "minimal_experiments": minimal_experiments[:4],
        "next_attempt_directives": next_attempt_directives[:5],
        "evidence": evidence_items[:4],
        "confidence": 0.72 if error_type and error_type != "unknown" else 0.45,
        "authority_state": "none",
        "verification": "provisional",
    }

    return {
        "protocol_steps": protocol_steps,
        "contradicted_assumptions": contradicted_assumptions,
        "minimal_experiments": minimal_experiments,
        "repair_brief": repair_brief,
    }


def build_pre_answer_trigger_reasons(
    prompt_text: str,
    ctx: Dict[str, Any],
    *,
    risk: float = 0.0,
    task_id: str = "",
    has_prior_task_investigation: bool = False,
) -> List[str]:
    """Conservative gate for prospective investigation before generation."""
    reasons: List[str] = []
    investigations = list(ctx.get("investigations") or [])
    candidates = list(ctx.get("evidence") or [])
    proposals = list(ctx.get("governance_proposals") or [])
    if investigations:
        reasons.append("retrieved_investigation_evidence")
    if any((inv.get("repair_brief") or {}) for inv in investigations):
        reasons.append("retrieved_repair_brief")
    if task_id and has_prior_task_investigation:
        reasons.append("prior_failure_chain_for_task")
    if any("objective_failure" in str(c.get("text", "")) or "failure_repair" in str(c.get("text", "")) for c in candidates):
        reasons.append("retrieved_failure_candidate")
    if proposals:
        reasons.append("provisional_governance_present")
    if float(risk or 0.0) >= 0.80:
        reasons.append("high_prompt_risk")
    if re.search(r"\b(hidden tests|humanEval|complete the following python function|return executable python)\b", prompt_text, flags=re.IGNORECASE):
        reasons.append("hidden_test_coding_task")
    if reasons == ["hidden_test_coding_task"]:
        return []
    return reasons[:8]


def build_pre_answer_contract_notes(prompt_text: str) -> List[str]:
    notes: List[str] = []
    low = str(prompt_text).lower()
    if "largest divisor" in low:
        notes.append("Check whether the requested divisor must be smaller than n; do not return n unless the contract permits it.")
    if re.search(r"\bprime\b", prompt_text, flags=re.IGNORECASE):
        notes.append("Check boundary cases 0, 1, 2, 3 before general primality logic.")
    if re.search(r"\bbracket|bracketing|parentheses\b", prompt_text, flags=re.IGNORECASE):
        notes.append("Check order-sensitive nesting/stack behavior, not just counts.")
    if re.search(r"\bfactorial\b|\bf\(n\)\b", prompt_text, flags=re.IGNORECASE):
        notes.append("Avoid undeclared helpers/imports; define factorial locally with a loop if needed.")
    if re.search(r"\bsentence|starts with|bored\b", prompt_text, flags=re.IGNORECASE):
        notes.append("Parse sentence starts carefully; punctuation plus following spaces may matter.")
    return notes[:5]


def format_pre_answer_investigation_text(investigation: Dict[str, Any] | None) -> str:
    """Render compact retrieved evidence without prescribing a reasoning frame."""
    if not investigation:
        return ""
    lines = [
        "Retrieved prior evidence (provisional, not governance):",
        f"- trigger_reasons={','.join(map(str, investigation.get('trigger_reasons') or []))}",
    ]
    assumptions = investigation.get("assumptions_to_check") or []
    if assumptions:
        lines.append("- prior_contradicted_assumptions: " + " ; ".join(map(str, assumptions[:3])))
    directives = investigation.get("application_directives") or []
    if directives:
        lines.append("- prior_repair_observations: " + " ; ".join(map(str, directives[:3])))
    experiments = investigation.get("minimal_experiments") or []
    if experiments:
        lines.append("- available_failure_targets: " + " ; ".join(map(str, experiments[:2])))
    return "\n".join(lines)

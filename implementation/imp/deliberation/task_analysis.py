from __future__ import annotations

"""Selective task decomposition and information-gap classification.

This module does not prescribe a reasoning framework. It identifies what kind
of support may be useful and, only when warranted, asks the actor for a compact
execution plan grounded in the current task and available evidence.
"""

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Callable, Optional


@dataclass
class TaskAnalysisDecision:
    invoked: bool
    difficulty_type: str
    needs_decomposition: bool
    needs_memory: bool
    needs_external_information: bool
    needs_experiment: bool
    reasons: list[str]
    plan: dict[str, Any]
    model_used: bool = False
    model_raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not m:
        return {}
    try:
        value = json.loads(m.group(0))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def classify_information_gap(prompt: str, *, previous_failures: Optional[list[dict[str, Any]]] = None) -> tuple[str, dict[str, bool], list[str]]:
    lower = (prompt or "").lower()
    reasons: list[str] = []
    needs_memory = bool(previous_failures)
    needs_experiment = bool(previous_failures)
    needs_external = any(x in lower for x in ("api", "library", "version", "documentation", "endpoint", "package", "current", "latest"))
    structural = any(x in lower for x in ("graph", "tree", "matrix", "recursive", "dynamic programming", "combinations", "permutations", "parse", "schedule"))
    edge_density = sum(1 for x in ("empty", "negative", "duplicate", "order", "sorted", "substring", "subsequence", "exactly", "at least", "at most") if x in lower)
    needs_decomposition = structural or edge_density >= 3 or bool(previous_failures)

    if previous_failures:
        reasons.append("observed_failure_requires_revision")
    if structural:
        reasons.append("structural_complexity")
    if edge_density >= 3:
        reasons.append("dense_contract_or_edge_cases")
    if needs_external:
        reasons.append("external_knowledge_signal")

    if needs_external:
        difficulty = "knowledge"
    elif needs_experiment:
        difficulty = "environmental"
    elif needs_memory:
        difficulty = "episodic"
    elif needs_decomposition:
        difficulty = "reasoning"
    else:
        difficulty = "direct"

    return difficulty, {
        "needs_decomposition": needs_decomposition,
        "needs_memory": needs_memory,
        "needs_external_information": needs_external,
        "needs_experiment": needs_experiment,
    }, reasons or ["direct_solution_appears_sufficient"]


def analyze_task(
    prompt: str,
    *,
    previous_failures: Optional[list[dict[str, Any]]] = None,
    gate_route: str = "",
    model_generate: Optional[Callable[[str], str]] = None,
    enable_model_plan: bool = True,
) -> TaskAnalysisDecision:
    difficulty, flags, reasons = classify_information_gap(prompt, previous_failures=previous_failures)
    should_invoke = bool(flags["needs_decomposition"] or gate_route == "deliberate_then_commit")
    plan: dict[str, Any] = {}
    raw = ""
    model_used = False

    if should_invoke and enable_model_plan and model_generate is not None:
        planning_prompt = (
            "Analyze this programming task without solving it. Return JSON only with keys: "
            "goal, subtasks (max 5), checks (max 5), information_gaps, recommended_operation. "
            "Use concise task-specific statements. Do not include implementation code or generic reasoning advice.\n\n"
            + prompt[:4000]
        )
        try:
            raw = model_generate(planning_prompt)
            parsed = _parse_json(raw)
            if parsed:
                plan = {
                    "goal": str(parsed.get("goal") or "")[:300],
                    "subtasks": [str(x)[:240] for x in (parsed.get("subtasks") or [])[:5]],
                    "checks": [str(x)[:240] for x in (parsed.get("checks") or [])[:5]],
                    "information_gaps": [str(x)[:240] for x in (parsed.get("information_gaps") or [])[:4]],
                    "recommended_operation": str(parsed.get("recommended_operation") or difficulty)[:120],
                }
                model_used = True
        except Exception as exc:
            reasons.append(f"planning_model_failed:{type(exc).__name__}")

    if should_invoke and not plan:
        plan = {
            "goal": "Satisfy the function contract and objective tests.",
            "subtasks": ["Extract the exact input/output contract", "Choose a direct algorithm", "Check stated examples and boundary cases"],
            "checks": ["Trace the stated example", "Check one boundary case", "Check traversal/order semantics"],
            "information_gaps": ["External lookup required"] if flags["needs_external_information"] else [],
            "recommended_operation": "experiment" if flags["needs_experiment"] else "decompose_then_execute",
        }

    return TaskAnalysisDecision(
        invoked=should_invoke,
        difficulty_type=difficulty,
        needs_decomposition=bool(flags["needs_decomposition"]),
        needs_memory=bool(flags["needs_memory"]),
        needs_external_information=bool(flags["needs_external_information"]),
        needs_experiment=bool(flags["needs_experiment"]),
        reasons=list(dict.fromkeys(reasons)),
        plan=plan,
        model_used=model_used,
        model_raw=raw[:1200],
    )


def format_task_analysis_context(decision: TaskAnalysisDecision) -> str:
    if not decision.invoked or not decision.plan:
        return ""
    return (
        "TASK ANALYSIS TOOL OUTPUT (advisory; task-specific)\n"
        + json.dumps({
            "difficulty_type": decision.difficulty_type,
            "needs_external_information": decision.needs_external_information,
            "needs_experiment": decision.needs_experiment,
            "plan": decision.plan,
        }, ensure_ascii=False, indent=2)
    )

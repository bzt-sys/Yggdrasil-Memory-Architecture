from __future__ import annotations

"""Replayable cognitive-operation orchestration.

The controller does not solve tasks or prescribe a reasoning framework. It
combines uncertainty, task structure, observed failures, information gaps, and
available tools into an explicit operation plan. The result is intended to be
logged verbatim so route quality can be evaluated independently of task score.
"""

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class OrchestrationDecision:
    primary_route: str
    operations: list[str]
    rationale: list[str]
    context_enabled: bool
    pre_answer_enabled: bool
    deliberation_enabled: bool
    use_task_decomposition: bool
    use_information_retrieval: bool
    use_episodic_retrieval: bool
    use_environment_experiment: bool
    retrieval_budget: dict[str, int]
    suggested_temperature: float
    unresolved_knowledge_gap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _budget(**overrides: int) -> dict[str, int]:
    base = {
        "candidates": 1,
        "investigations": 0,
        "hypotheses": 1,
        "governance": 1,
        "recent": 1,
        "core_principles": 0,
    }
    base.update({k: max(0, int(v)) for k, v in overrides.items()})
    return base


def decide_orchestration(
    *,
    gate: Optional[dict[str, Any]],
    task_analysis: Optional[dict[str, Any]],
    previous_failures: Optional[list[dict[str, Any]]] = None,
    information_retrieval_available: bool = False,
    environment_available: bool = True,
) -> OrchestrationDecision:
    gate = gate or {}
    analysis = task_analysis or {}
    failures = previous_failures or []
    difficulty = str(analysis.get("difficulty_type") or "direct")
    confidence = float(gate.get("confidence") or 0.5)
    reasons: list[str] = []
    operations: list[str] = []

    needs_decomposition = bool(analysis.get("needs_decomposition"))
    needs_memory = bool(analysis.get("needs_memory"))
    needs_external = bool(analysis.get("needs_external_information"))
    needs_experiment = bool(analysis.get("needs_experiment")) or bool(failures)

    # Observed outcomes outrank self-estimated confidence.
    if failures:
        reasons.append("observed_failure_overrides_confidence")
        operations.extend(["episodic_retrieval", "task_decomposition", "environment_experiment"])
        return OrchestrationDecision(
            primary_route="repair_and_experiment",
            operations=operations,
            rationale=reasons,
            context_enabled=True,
            pre_answer_enabled=True,
            deliberation_enabled=bool(environment_available),
            use_task_decomposition=True,
            use_information_retrieval=bool(needs_external and information_retrieval_available),
            use_episodic_retrieval=True,
            use_environment_experiment=bool(environment_available),
            retrieval_budget=_budget(candidates=4, investigations=3, hypotheses=3, governance=2, recent=2),
            suggested_temperature=0.12,
            unresolved_knowledge_gap=bool(needs_external and not information_retrieval_available),
        )

    if needs_external:
        reasons.append("external_information_gap_detected")
        operations.append("information_retrieval")
        if needs_decomposition:
            operations.append("task_decomposition")
        return OrchestrationDecision(
            primary_route="retrieve_then_plan" if information_retrieval_available else "knowledge_gap_unresolved",
            operations=operations,
            rationale=reasons,
            context_enabled=True,
            pre_answer_enabled=False,
            deliberation_enabled=False,
            use_task_decomposition=bool(needs_decomposition),
            use_information_retrieval=bool(information_retrieval_available),
            use_episodic_retrieval=bool(needs_memory),
            use_environment_experiment=False,
            retrieval_budget=_budget(candidates=1 if needs_memory else 0, investigations=0, hypotheses=1, governance=1, recent=1),
            suggested_temperature=0.10,
            unresolved_knowledge_gap=not information_retrieval_available,
        )

    if needs_experiment and environment_available:
        reasons.append("testable_uncertainty_or_environmental_need")
        operations.extend(["task_decomposition", "environment_experiment"])
        if needs_memory:
            operations.insert(0, "episodic_retrieval")
        return OrchestrationDecision(
            primary_route="plan_and_experiment",
            operations=operations,
            rationale=reasons,
            context_enabled=True,
            pre_answer_enabled=bool(needs_memory),
            deliberation_enabled=True,
            use_task_decomposition=True,
            use_information_retrieval=False,
            use_episodic_retrieval=bool(needs_memory),
            use_environment_experiment=True,
            retrieval_budget=_budget(candidates=3 if needs_memory else 1, investigations=2 if needs_memory else 0, hypotheses=2, governance=1, recent=2),
            suggested_temperature=0.16,
        )

    if needs_decomposition or difficulty == "reasoning":
        reasons.append("structural_reasoning_support_needed")
        operations.append("task_decomposition")
        if confidence < 0.62 and environment_available:
            operations.append("environment_experiment")
        return OrchestrationDecision(
            primary_route="plan_then_execute",
            operations=operations,
            rationale=reasons,
            context_enabled=True,
            pre_answer_enabled=False,
            deliberation_enabled=bool(confidence < 0.62 and environment_available),
            use_task_decomposition=True,
            use_information_retrieval=False,
            use_episodic_retrieval=False,
            use_environment_experiment=bool(confidence < 0.62 and environment_available),
            retrieval_budget=_budget(candidates=1, investigations=0, hypotheses=2, governance=1, recent=1),
            suggested_temperature=0.14,
        )

    if needs_memory or difficulty == "episodic":
        reasons.append("relevant_prior_experience_may_help")
        operations.append("episodic_retrieval")
        return OrchestrationDecision(
            primary_route="retrieve_then_execute",
            operations=operations,
            rationale=reasons,
            context_enabled=True,
            pre_answer_enabled=False,
            deliberation_enabled=False,
            use_task_decomposition=False,
            use_information_retrieval=False,
            use_episodic_retrieval=True,
            use_environment_experiment=False,
            retrieval_budget=_budget(candidates=2, investigations=1, hypotheses=2, governance=1, recent=1),
            suggested_temperature=0.12,
        )

    reasons.append("direct_execution_sufficient")
    operations.append("direct_execution")
    return OrchestrationDecision(
        primary_route="direct_execute",
        operations=operations,
        rationale=reasons,
        context_enabled=True,
        pre_answer_enabled=False,
        deliberation_enabled=False,
        use_task_decomposition=False,
        use_information_retrieval=False,
        use_episodic_retrieval=False,
        use_environment_experiment=False,
        retrieval_budget=_budget(candidates=0, investigations=0, hypotheses=0, governance=1, recent=0),
        suggested_temperature=0.08,
    )

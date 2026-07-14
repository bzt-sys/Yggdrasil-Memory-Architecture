from __future__ import annotations

"""Structured environment observation helpers.

This module is intentionally small and dependency-free. It gives experiment
harnesses a stable way to express external feedback before that feedback enters
Yggdrasil's ordinary event stream.

The memory engine still owns graph mutation and episode closure. This boundary
only normalizes external observations into the textual/provenance form currently
consumed by the live runtime, while preserving richer fields for future evidence
objects.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class EnvironmentObservation:
    """Serializable external observation produced by a task environment.

    The fields are general enough for HumanEval today and for later domains
    such as file-system IDE tasks, finance simulators, research workflows, and
    tool-call sandboxes.
    """

    task_id: str = ""
    attempt_number: int = 0
    passed: Optional[bool] = None
    error_type: str = ""
    error: str = ""
    entry_point: str = ""
    mode: str = ""
    source: str = "environment"
    feedback_level: int = 1
    signal_quality: str = "low"
    failed_input: Optional[str] = None
    expected_output: Optional[str] = None
    observed_output: Optional[str] = None
    corpus_snippets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_environment_observation(
    observation: Union[EnvironmentObservation, Dict[str, Any]],
) -> EnvironmentObservation:
    """Coerce a dict or dataclass into an EnvironmentObservation."""
    if isinstance(observation, EnvironmentObservation):
        return observation
    data = dict(observation or {})
    return EnvironmentObservation(
        task_id=str(data.get("task_id") or data.get("task") or ""),
        attempt_number=int(data.get("attempt_number") or data.get("attempt") or 0),
        passed=data.get("passed"),
        error_type=str(data.get("error_type") or ""),
        error=str(data.get("error") or data.get("observed_error") or ""),
        entry_point=str(data.get("entry_point") or ""),
        mode=str(data.get("mode") or ""),
        source=str(data.get("source") or "environment"),
        feedback_level=int(data.get("feedback_level") or 1),
        signal_quality=str(data.get("signal_quality") or "low"),
        failed_input=data.get("failed_input"),
        expected_output=data.get("expected_output"),
        observed_output=data.get("observed_output"),
        corpus_snippets=list(data.get("corpus_snippets") or []),
        metadata=dict(data.get("metadata") or {}),
    )


def _semi_safe(value: Any, *, limit: int = 800) -> str:
    """Render a value into a compact semicolon-safe field."""
    if value is None:
        return ""
    text = str(value).replace("\n", "\\n").replace(";", ",")
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def format_environment_observation(
    observation: Union[EnvironmentObservation, Dict[str, Any]],
    *,
    prefix: str = "Environment objective feedback",
) -> str:
    """Format an external observation for the current event-stream parser.

    The resulting text intentionally retains the existing key=value shape used
    by ``YggdrasilEngine._latest_objective_feedback``. Additional fields are
    appended conservatively so older parsers simply ignore them.
    """
    obs = normalize_environment_observation(observation)
    task_label = obs.task_id or "unknown_task"
    attempt_label = int(obs.attempt_number or 0)
    passed = "" if obs.passed is None else str(bool(obs.passed))
    fields = [
        f"passed={passed}",
        f"error_type={_semi_safe(obs.error_type, limit=160)}",
        f"error={_semi_safe(obs.error, limit=900)}",
        f"entry_point={_semi_safe(obs.entry_point, limit=160)}",
        f"mode={_semi_safe(obs.mode, limit=160)}",
        f"feedback_level={int(obs.feedback_level or 0)}",
        f"signal_quality={_semi_safe(obs.signal_quality, limit=80)}",
    ]
    if obs.failed_input is not None:
        fields.append(f"failed_input={_semi_safe(obs.failed_input, limit=500)}")
    if obs.expected_output is not None:
        fields.append(f"expected_output={_semi_safe(obs.expected_output, limit=500)}")
    if obs.observed_output is not None:
        fields.append(f"observed_output={_semi_safe(obs.observed_output, limit=500)}")
    if obs.corpus_snippets:
        joined = " || ".join(_semi_safe(s, limit=280) for s in obs.corpus_snippets[:3])
        fields.append(f"corpus_snippets={joined}")

    return f"{prefix} for {task_label} attempt {attempt_label}: " + "; ".join(fields) + "."

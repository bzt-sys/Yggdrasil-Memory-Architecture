from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class EnvironmentObservation:
    """Structured external feedback from an environment/tool/evaluator.

    The observation is intentionally conservative. It may contain concrete
    fields such as failing input/output later, but the HumanEval bridge defaults
    to non-answer-leaking pass/fail/error metadata.
    """

    environment: str
    task_id: str = ""
    attempt_number: int = 0
    passed: Optional[bool] = None
    error_type: str = ""
    error: str = ""
    entry_point: str = ""
    mode: str = ""
    feedback_level: int = 1
    signal_quality: str = "low_bandwidth"
    failed_input: Optional[str] = None
    expected_output: Optional[str] = None
    observed_output: Optional[str] = None
    corpus_snippets: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_observation(value: EnvironmentObservation | Mapping[str, Any]) -> EnvironmentObservation:
    """Convert a dataclass or dict payload into an EnvironmentObservation."""
    if isinstance(value, EnvironmentObservation):
        return value
    d = dict(value)
    allowed = set(EnvironmentObservation.__dataclass_fields__.keys())
    kwargs = {k: d.get(k) for k in allowed if k in d}
    if "corpus_snippets" not in kwargs or kwargs.get("corpus_snippets") is None:
        kwargs["corpus_snippets"] = []
    if "metadata" not in kwargs or kwargs.get("metadata") is None:
        kwargs["metadata"] = {}
    return EnvironmentObservation(**kwargs)  # type: ignore[arg-type]


def _short(text: Any, limit: int = 500) -> str:
    s = str(text or "").replace("\n", " ").strip()
    return s[:limit]


def format_environment_observation(obs: EnvironmentObservation | Mapping[str, Any]) -> str:
    """Serialize structured environment evidence into a stable text event.

    Keep the legacy "HumanEval objective feedback ..." prefix for compatibility
    with existing memory.py parsers, then append structured evidence fields in a
    predictable form.
    """
    o = normalize_observation(obs)
    if o.environment.lower() == "humaneval":
        head = (
            f"HumanEval objective feedback for {o.task_id} attempt {int(o.attempt_number or 0)}: "
            f"passed={bool(o.passed)}; "
            f"error_type={o.error_type}; "
            f"error={_short(o.error)}; "
            f"entry_point={o.entry_point}; "
            f"mode={o.mode}."
        )
    else:
        head = (
            f"Environment objective feedback for {o.environment} task={o.task_id} "
            f"attempt={int(o.attempt_number or 0)}: passed={o.passed}; "
            f"error_type={o.error_type}; error={_short(o.error)}."
        )

    extra: List[str] = []
    extra.append(f"feedback_level={int(o.feedback_level or 0)}")
    extra.append(f"signal_quality={o.signal_quality}")
    if o.failed_input is not None:
        extra.append(f"failed_input={_short(o.failed_input)}")
    if o.expected_output is not None:
        extra.append(f"expected_output={_short(o.expected_output)}")
    if o.observed_output is not None:
        extra.append(f"observed_output={_short(o.observed_output)}")
    if o.corpus_snippets:
        titles = [str(s.get("title") or s.get("topic") or "snippet") for s in o.corpus_snippets[:3]]
        extra.append("corpus_snippets=" + " | ".join(titles))
    if extra:
        return head + " Structured environment evidence: " + "; ".join(extra) + "."
    return head

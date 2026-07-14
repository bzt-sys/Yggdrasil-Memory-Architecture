from __future__ import annotations

"""Environment feedback normalization helpers.

These functions convert environment/harness feedback strings into stable
structured dictionaries.  The helpers are intentionally pure so they can be
used by HumanEval today and by richer coding/IDE environments later.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EnvironmentEvidence:
    """Structured evidence extracted from an external environment signal."""

    raw: str = ""
    passed: Optional[bool] = None
    error_type: str = ""
    error: str = ""
    entry_point: str = ""
    mode: str = ""
    task_id: str = ""
    attempt_number: int = 0
    failed_input: str = ""
    expected_output: str = ""
    observed_output: str = ""
    evidence_source: str = "environment"
    feedback_level: int = 1
    signal_quality: str = "low"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_field(text: str, key: str) -> str:
    """Extract a semicolon-delimited key=value field from feedback text."""
    m = re.search(rf"(?:^|;)\s*{re.escape(key)}=([^;]*)(?:;|$)", text)
    if not m:
        return ""
    return m.group(1).strip()


def _feedback_level(evidence: EnvironmentEvidence) -> int:
    """Classify feedback bandwidth without interpreting the task solution."""
    if evidence.failed_input or evidence.expected_output or evidence.observed_output:
        return 2
    if evidence.error_type or evidence.error or evidence.entry_point:
        return 1
    if evidence.passed is not None:
        return 0
    return 0


def _signal_quality(evidence: EnvironmentEvidence) -> str:
    """Conservative quality label for downstream investigation policies."""
    if evidence.passed is True:
        return "high"
    if evidence.failed_input and (evidence.expected_output or evidence.observed_output):
        return "high"
    if evidence.error_type and evidence.error and evidence.error not in {"AssertionError()", "AssertionError"}:
        return "medium"
    if evidence.error_type:
        return "low"
    if evidence.passed is not None:
        return "low"
    return "unknown"


def parse_objective_feedback_text(text: str, *, raw_limit: int = 1200) -> Dict[str, Any]:
    """Parse an objective-feedback event into structured environment evidence.

    Supported current format:
        HumanEval objective feedback for HumanEval/24 attempt 1:
        passed=False; error_type=AssertionError; error=AssertionError();
        entry_point=largest_divisor; mode=response_only.

    Future-compatible optional fields:
        failed_input=...; expected=...; observed=...

    The parser intentionally does not infer the correct answer.  It only
    preserves environmental evidence and feedback bandwidth metadata.
    """
    raw = str(text or "")[: int(raw_limit)]
    evidence = EnvironmentEvidence(raw=raw)

    task_match = re.search(r"objective feedback for ([^:]+) attempt (\d+)", raw, flags=re.IGNORECASE)
    if task_match:
        evidence.task_id = task_match.group(1).strip()
        try:
            evidence.attempt_number = int(task_match.group(2))
        except Exception:
            evidence.attempt_number = 0

    passed_match = re.search(r"passed=(True|False|true|false)", raw)
    if passed_match:
        evidence.passed = passed_match.group(1).lower() == "true"

    evidence.error_type = _extract_field(raw, "error_type")
    evidence.error = _extract_field(raw, "error")
    evidence.entry_point = _extract_field(raw, "entry_point")
    evidence.mode = _extract_field(raw, "mode").rstrip(".")
    evidence.failed_input = _extract_field(raw, "failed_input") or _extract_field(raw, "input")
    evidence.expected_output = _extract_field(raw, "expected") or _extract_field(raw, "expected_output")
    evidence.observed_output = _extract_field(raw, "observed") or _extract_field(raw, "observed_output")

    evidence.feedback_level = _feedback_level(evidence)
    evidence.signal_quality = _signal_quality(evidence)
    return evidence.to_dict()

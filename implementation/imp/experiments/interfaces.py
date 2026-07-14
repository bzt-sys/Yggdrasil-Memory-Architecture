"""Lightweight experiment interface types.

These types define the experiment/runtime boundary without changing current
behavior. They are intentionally small so existing runners can adopt them
incrementally while the substrate remains stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExperimentArtifactPaths:
    """Canonical files produced by an experiment run."""

    run_dir: Path
    records_jsonl: Path
    transcript_md: Path
    manifest_json: Path
    summary_json: Optional[Path] = None
    report_md: Optional[Path] = None


@dataclass
class ExperimentAttemptRecord:
    """Single attempt record emitted by benchmark/task harnesses."""

    run_id: str
    task_id: str
    attempt_number: int
    prompt: str = ""
    response: str = ""
    correctness: Optional[bool] = None
    error_type: str = ""
    error: str = ""
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    context_debug: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRunSummary:
    """Aggregate run summary for reports and comparisons."""

    run_id: str
    task_count: int = 0
    attempt_count: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    error_type_counts: Dict[str, int] = field(default_factory=dict)
    lifecycle_totals: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

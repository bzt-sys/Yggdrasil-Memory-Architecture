"""Controlled Python execution adapter scaffold.

Patch 3.6 provides a minimal, isolated adapter for future HumanEval/coding
curriculum integration. It is not wired into the runtime yet.
"""

from __future__ import annotations

import contextlib
import io
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schemas import EnvironmentAction, EnvironmentObservation, EnvironmentResult, EvidenceExtraction


@dataclass
class PythonExecutionRequest:
    code: str
    entry_point: str = ""
    tests: str = ""
    globals_seed: Dict[str, Any] = field(default_factory=dict)
    mode: str = "exec"
    timeout_s: float = 2.0


class PythonEnvironmentAdapter:
    """Small local Python execution adapter for controlled experiments.

    This intentionally avoids subprocess/sandbox guarantees for now. It is a
    development scaffold for short trusted snippets generated inside the local
    experiment harness. A hardened subprocess sandbox should replace it before
    untrusted deployment.
    """

    domain = "python"

    def build_action(self, request: PythonExecutionRequest, *, source: str = "experiment") -> EnvironmentAction:
        return EnvironmentAction(
            action_id=str(uuid.uuid4()),
            action_type="python_exec",
            domain=self.domain,
            intent="validate_code",
            source=source,
            max_runtime_s=float(request.timeout_s),
            payload={
                "code": request.code,
                "entry_point": request.entry_point,
                "tests": request.tests,
                "mode": request.mode,
            },
        )

    def run(self, action: EnvironmentAction) -> EnvironmentResult:
        code = str(action.payload.get("code") or "")
        tests = str(action.payload.get("tests") or "")
        entry_point = str(action.payload.get("entry_point") or "")
        namespace: Dict[str, Any] = {}
        stdout = io.StringIO()
        status = "pass"
        raw = ""
        data: Dict[str, Any] = {"entry_point": entry_point}
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, namespace, namespace)
                if tests:
                    exec(tests, namespace, namespace)
            raw = stdout.getvalue()
            data.update({"stdout": raw, "passed": True, "error_type": "", "error": ""})
            ok = True
        except Exception as exc:  # pragma: no cover - scaffold path
            status = "fail"
            tb = traceback.format_exc(limit=8)
            raw = tb
            data.update({
                "stdout": stdout.getvalue(),
                "passed": False,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": tb,
            })
            ok = False

        observation = EnvironmentObservation(
            observation_id=str(uuid.uuid4()),
            action_id=action.action_id,
            domain=self.domain,
            source="python_adapter",
            status=status,
            raw=raw[:4000],
            data=data,
            confidence=0.95,
        )
        evidence = self.extract_evidence(observation)
        return EnvironmentResult(action=action, observation=observation, evidence=evidence, ok=ok)

    def extract_evidence(self, observation: EnvironmentObservation) -> List[EvidenceExtraction]:
        data = observation.data or {}
        if data.get("passed") is True:
            return [EvidenceExtraction(
                evidence_id=str(uuid.uuid4()),
                observation_id=observation.observation_id,
                evidence_type="objective_success",
                source="python_adapter",
                claim="Generated code executed and passed the provided tests.",
                supports=["candidate_solution"],
                confidence=0.95,
                causal_status="verified_for_tests",
                metadata={"entry_point": data.get("entry_point", "")},
            )]
        return [EvidenceExtraction(
            evidence_id=str(uuid.uuid4()),
            observation_id=observation.observation_id,
            evidence_type="objective_failure",
            source="python_adapter",
            claim="Generated code failed under Python execution or tests.",
            contradicts=["candidate_solution_correctness"],
            error_type=str(data.get("error_type") or "unknown"),
            observed_output=str(data.get("error") or "")[:500],
            confidence=0.90,
            causal_status="provisional_failure_signal",
            metadata={"entry_point": data.get("entry_point", ""), "traceback": str(data.get("traceback") or "")[:1500]},
        )]

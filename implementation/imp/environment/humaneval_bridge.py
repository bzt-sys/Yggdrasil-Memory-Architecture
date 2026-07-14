from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .observation import EnvironmentObservation, format_environment_observation


_SEED_CORPUS: List[Dict[str, str]] = [
    {
        "topic": "assertion_error",
        "title": "AssertionError means a tested condition was false",
        "text": "In Python tests, AssertionError usually means the program ran but returned a value that contradicted an expected condition. It is often a semantic or edge-case failure rather than a syntax failure.",
    },
    {
        "topic": "name_error",
        "title": "NameError means a name was not defined",
        "text": "A NameError occurs when code references a variable, function, class, or import that is not present in the current namespace. Define helpers or import allowed standard-library functions explicitly.",
    },
    {
        "topic": "imports",
        "title": "Returned benchmark code must include its own imports/helpers",
        "text": "When generated code is executed in isolation, helper functions and imports must be present in the returned code unless the prompt explicitly provides them.",
    },
    {
        "topic": "hidden_tests",
        "title": "Hidden tests check edge cases not visible in the prompt",
        "text": "Passing examples is not enough. Derive exact behavior from the docstring and mentally test empty inputs, boundaries, duplicates, negative values, and special cases.",
    },
    {
        "topic": "divisor",
        "title": "Largest divisor tasks often require proper-divisor semantics",
        "text": "If a task asks for the largest divisor and examples exclude the number itself, verify whether it is asking for the largest proper divisor: the largest divisor strictly smaller than n.",
    },
    {
        "topic": "syntax",
        "title": "SyntaxError means code could not be parsed",
        "text": "SyntaxError indicates the interpreter could not parse the generated code. Check indentation, incomplete fenced extraction, missing parentheses, and unterminated strings.",
    },
]


@dataclass
class HumanEvalEnvironmentBridge:
    """Convert HumanEval correctness results into structured environment evidence."""

    corpus_k: int = 3
    include_corpus: bool = True
    corpus: List[Dict[str, str]] = field(default_factory=lambda: list(_SEED_CORPUS))

    def corpus_lookup(self, correctness: Dict[str, Any], task: Dict[str, Any]) -> List[Dict[str, str]]:
        if not self.include_corpus or self.corpus_k <= 0:
            return []
        error_type = str(correctness.get("error_type") or "").lower()
        prompt = str(task.get("prompt") or "").lower()
        scored: List[tuple[int, Dict[str, str]]] = []
        for row in self.corpus:
            blob = (row.get("topic", "") + " " + row.get("title", "") + " " + row.get("text", "")).lower()
            score = 0
            if error_type and error_type in blob:
                score += 4
            for token in ("divisor", "assertion", "nameerror", "syntax", "hidden", "import", "edge"):
                if token in prompt and token in blob:
                    score += 2
                elif token in blob and token in error_type:
                    score += 1
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(r) for _, r in scored[: self.corpus_k]]

    def from_correctness(
        self,
        *,
        task: Dict[str, Any],
        task_id: str,
        attempt: int,
        correctness: Dict[str, Any],
    ) -> EnvironmentObservation:
        passed = bool(correctness.get("passed"))
        error_type = str(correctness.get("error_type") or "")
        signal_quality = "verified_pass" if passed else ("low_bandwidth_assertion" if error_type == "AssertionError" else "error_class_signal")
        return EnvironmentObservation(
            environment="humaneval",
            task_id=str(task_id),
            attempt_number=int(attempt),
            passed=passed,
            error_type=error_type,
            error=str(correctness.get("error") or ""),
            entry_point=str(correctness.get("entry_point") or task.get("entry_point") or ""),
            mode=str(correctness.get("mode") or ""),
            feedback_level=1,
            signal_quality=signal_quality,
            corpus_snippets=self.corpus_lookup(correctness, task),
            metadata={
                "extracted_chars": correctness.get("extracted_chars"),
                "task_index": task.get("task_index"),
                "estimated_complexity": task.get("estimated_complexity"),
                "complexity_bucket": str(task.get("complexity_bucket")),
                "solution_visible": False,
            },
        )

    def serialize_event_text(self, observation: EnvironmentObservation) -> str:
        return format_environment_observation(observation)

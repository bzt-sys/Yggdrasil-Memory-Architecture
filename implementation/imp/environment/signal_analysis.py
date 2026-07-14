from __future__ import annotations

"""Environment signal comparative analysis.

This module is intentionally domain-general. It compares a committed action
against a later environment signal and returns an auditable analysis object.
For coding benchmarks, the environment signal is usually an exception or test
failure. For future IDE/embodied deployments, the same shape can represent test
logs, linter output, user correction, tool failure, or world-state feedback.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SignalMeaning:
    signal_type: str
    meaning: str
    confidence: float = 0.45
    source: str = "local_environment_signal_corpus"
    uncertainty: str = "medium"


@dataclass
class ActionSignalAnalysis:
    signal_type: str
    signal_text: str
    signal_meaning: str
    prior_action_summary: str
    action_features: List[str] = field(default_factory=list)
    possible_violated_assumptions: List[str] = field(default_factory=list)
    corrective_hypotheses: List[str] = field(default_factory=list)
    information_needs: List[str] = field(default_factory=list)
    corpus_queries: List[str] = field(default_factory=list)
    external_sources_used: List[str] = field(default_factory=list)
    confidence: float = 0.45
    uncertainty: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# This is deliberately a small local corpus, not task-answer data. It teaches the
# system what environment signals mean, not how to solve a benchmark item.
_SIGNAL_CORPUS: Dict[str, SignalMeaning] = {
    "assertionerror": SignalMeaning(
        signal_type="AssertionError",
        meaning=(
            "The generated program executed, but a checked condition failed. "
            "In benchmark tests this usually means the implementation violated "
            "the function contract or missed an edge case rather than failing to parse."
        ),
        confidence=0.86,
        uncertainty="low",
    ),
    "syntaxerror": SignalMeaning(
        signal_type="SyntaxError",
        meaning=(
            "Python could not parse the submitted code. The output may be truncated, "
            "mixed with prose, or contain incomplete blocks/strings/parentheses."
        ),
        confidence=0.9,
        uncertainty="low",
    ),
    "nameerror": SignalMeaning(
        signal_type="NameError",
        meaning=(
            "The program referenced a symbol that was not defined in the executable "
            "namespace. The action likely assumed a helper, variable, class, or import existed."
        ),
        confidence=0.9,
        uncertainty="low",
    ),
    "attributeerror": SignalMeaning(
        signal_type="AttributeError",
        meaning=(
            "The program accessed an attribute or method that the runtime object did not provide. "
            "The action likely assumed an API shape or object capability that was not valid."
        ),
        confidence=0.88,
        uncertainty="low",
    ),
    "importerror": SignalMeaning(
        signal_type="ImportError",
        meaning=(
            "The program attempted to import something that could not be loaded. "
            "The action likely assumed a dependency or import surface was available."
        ),
        confidence=0.86,
        uncertainty="low",
    ),
    "modulenotfounderror": SignalMeaning(
        signal_type="ModuleNotFoundError",
        meaning=(
            "The program attempted to import a module unavailable to the execution environment. "
            "The action likely assumed an external dependency existed."
        ),
        confidence=0.9,
        uncertainty="low",
    ),
    "typeerror": SignalMeaning(
        signal_type="TypeError",
        meaning=(
            "The program used an operation or call incompatible with the runtime values. "
            "The action likely assumed a type, callable signature, or operand behavior."
        ),
        confidence=0.82,
        uncertainty="low",
    ),
    "timeouterror": SignalMeaning(
        signal_type="TimeoutError",
        meaning=(
            "The program did not finish within the environment budget. The action likely used "
            "an algorithmic strategy with excessive or unbounded runtime."
        ),
        confidence=0.8,
        uncertainty="low",
    ),
}


def lookup_signal_meaning(signal_type: str, signal_text: str = "") -> SignalMeaning:
    key = re.sub(r"[^a-zA-Z]", "", str(signal_type or "")).lower()
    if key in _SIGNAL_CORPUS:
        return _SIGNAL_CORPUS[key]
    low = str(signal_text or "").lower()
    for k, meaning in _SIGNAL_CORPUS.items():
        if k in low:
            return meaning
    return SignalMeaning(
        signal_type=str(signal_type or "unknown") or "unknown",
        meaning=(
            "The environment returned a signal whose operational meaning is not yet known with confidence. "
            "The system should query documentation, a local corpus, or a tool manual before forming durable conclusions."
        ),
        confidence=0.35,
        source="unknown_signal_fallback",
        uncertainty="high",
    )


def _compact(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def summarize_prior_action(action_text: str) -> str:
    text = str(action_text or "")
    if not text.strip():
        return "No prior committed action was available for comparison."
    codeish = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("def ", "class ", "import ", "from ")):
            codeish.append(s)
    if codeish:
        return "Prior action included: " + "; ".join(codeish[:5])
    return _compact(text, 420)


def extract_action_features(action_text: str) -> List[str]:
    text = str(action_text or "")
    low = text.lower()
    features: List[str] = []
    if re.search(r"\bimport\b|\bfrom\s+\w+\s+import\b", text):
        features.append("uses_imports")
    if re.search(r"\b(stack|append\(|pop\()", low):
        features.append("uses_stack_or_mutating_list_state")
    if re.search(r"\brecurs|\breturn\s+\w+\(", low):
        features.append("possibly_recursive")
    if re.search(r"\bsort\(|sorted\(", low):
        features.append("uses_sorting")
    if re.search(r"\bre\.|regex|regular expression", low):
        features.append("uses_regex")
    if "lambda" in low:
        features.append("uses_lambda")
    helpers = re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.M)
    if helpers:
        features.append("defines_functions:" + ",".join(helpers[:5]))
    calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    builtin_allow = {"range", "len", "int", "str", "list", "set", "dict", "sum", "min", "max", "abs", "all", "any", "enumerate", "zip", "sorted"}
    unknownish = sorted({c for c in calls if c not in builtin_allow and c not in helpers})
    if unknownish:
        features.append("calls_symbols:" + ",".join(unknownish[:8]))
    return features[:8]


def analyze_action_signal(
    *,
    signal_type: str,
    signal_text: str,
    prior_action_text: str,
    episodic_context: Optional[List[str]] = None,
    governance_context: Optional[List[str]] = None,
    candidate_context: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare a committed action against an environment signal.

    The output is deliberately an auditable report, not hidden reasoning. It is
    suitable for storage in an investigation workspace and later retrieval.
    """
    meaning = lookup_signal_meaning(signal_type, signal_text)
    action_summary = summarize_prior_action(prior_action_text)
    action_features = extract_action_features(prior_action_text)

    assumptions: List[str] = []
    hypotheses: List[str] = []
    needs: List[str] = []

    assumptions.append("The prior committed action was assumed to be compatible with the environment's success criteria.")
    if meaning.uncertainty == "high":
        needs.append(f"Query an external/local source for the meaning of environment signal `{signal_type}` before deriving durable governance.")
    else:
        needs.append(f"Use local signal meaning for `{meaning.signal_type}` as evidence, not as a final answer.")

    low_type = str(meaning.signal_type or signal_type).lower()
    if "assertion" in low_type:
        assumptions.append("The prior action's algorithmic strategy likely matched a familiar pattern but did not satisfy at least one tested contract condition.")
        hypotheses.append("Compare the task contract against the prior action's dominant strategy and change the strategy if it is only pattern-matched.")
        hypotheses.append("Before retrying, identify which part of the specification the prior action may have ignored or overgeneralized.")
        for feat in action_features:
            if feat == "uses_stack_or_mutating_list_state":
                assumptions.append("The prior action used stack/list-state behavior; because the environment signaled semantic failure, treat that strategy as suspect unless the contract explicitly requires it.")
                hypotheses.append("Try a non-stack formulation or justify stack behavior directly from the specification before reusing it.")
    elif any(x in low_type for x in ("name", "attribute", "import", "module")):
        assumptions.append("The prior action may have assumed a symbol, dependency, API, method, or object capability existed in the environment.")
        hypotheses.append("Make the next action self-contained or explicitly define/import every required dependency before use.")
        hypotheses.append("Do not rely on inferred API affordances unless they are provided by the prompt, corpus, docs, or runtime inspection.")
    elif "syntax" in low_type:
        assumptions.append("The prior action assumed the submitted artifact was syntactically complete and executable, but the environment rejected parsing.")
        hypotheses.append("Change the output surface: produce one complete artifact with no prose or truncation in the executable region.")
    elif "type" in low_type:
        assumptions.append("The prior action assumed value types or callable signatures that the environment did not support.")
        hypotheses.append("Audit operations and function calls against the prompt-specified data types before retrying.")
    elif "timeout" in low_type:
        assumptions.append("The prior action assumed its runtime strategy was computationally acceptable.")
        hypotheses.append("Replace unbounded search or heavy recursion with a bounded or direct algorithm.")
    else:
        assumptions.append("The relation between the prior action and environment signal is uncertain.")
        hypotheses.append("Make one narrow change that is causally connected to the signal meaning, then observe whether the environment changes.")

    if episodic_context:
        needs.append("Compare this failure with prior episodic patterns retrieved for the same or similar task.")
    if governance_context:
        needs.append("Check whether any provisional governance already warned against the failed action pattern.")
    if candidate_context:
        needs.append("Use candidate memories as hypotheses to test, not as authoritative rules.")

    queries = [f"What does {signal_type or 'this environment signal'} mean in this execution environment?"]
    if meaning.uncertainty == "high":
        queries.append("Search local tool/runtime documentation for the unknown environment signal.")

    result = ActionSignalAnalysis(
        signal_type=str(signal_type or meaning.signal_type or "unknown"),
        signal_text=_compact(signal_text, 800),
        signal_meaning=meaning.meaning,
        prior_action_summary=action_summary,
        action_features=action_features,
        possible_violated_assumptions=list(dict.fromkeys(assumptions))[:8],
        corrective_hypotheses=list(dict.fromkeys(hypotheses))[:8],
        information_needs=list(dict.fromkeys(needs))[:8],
        corpus_queries=queries[:4],
        external_sources_used=[meaning.source] if meaning.source else [],
        confidence=max(0.35, min(0.9, meaning.confidence)),
        uncertainty=meaning.uncertainty,
    )
    return result.to_dict()


def build_prompt_aware_failure_packet(
    *,
    task_prompt: str = "",
    assistant_response: str = "",
    extracted_code: str = "",
    environment_signal: Dict[str, Any] | None = None,
    retrieved_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    signal = dict(environment_signal or {})
    retrieved = dict(retrieved_context or {})
    error_type = str(signal.get("error_type", "") or signal.get("signal_type", "") or "")
    passed = bool(signal.get("passed", False))

    if passed:
        signal_meaning = "The committed action satisfied the available environment checks."
    elif error_type == "AssertionError":
        signal_meaning = (
            "The committed action executed, but violated an expected condition. "
            "This usually indicates a semantic/specification mismatch rather than syntax failure."
        )
    elif error_type in {"SyntaxError", "IndentationError"}:
        signal_meaning = "The committed action could not be parsed or compiled."
    elif error_type in {"NameError", "AttributeError", "ImportError", "ModuleNotFoundError"}:
        signal_meaning = "The committed action assumed a symbol, attribute, function, or dependency was available."
    elif error_type:
        signal_meaning = "The environment returned a failure signal. Interpret the signal before revising behavior."
    else:
        signal_meaning = "No explicit error type was supplied by the environment."

    return {
        "analysis_type": "prompt_aware_action_signal_comparison",
        "task_prompt_excerpt": (task_prompt or "")[:4000],
        "assistant_response_excerpt": (assistant_response or "")[:4000],
        "extracted_code_excerpt": (extracted_code or "")[:4000],
        "environment_signal": signal,
        "signal_meaning": signal_meaning,
        "retrieved_candidate_ids": retrieved.get("selected_candidate_ids", []),
        "retrieved_investigation_ids": retrieved.get("selected_investigation_ids", []),
        "corrective_questions": [
            "What did the prompt/specification require?",
            "What assumption did the committed action make?",
            "How does the environment signal contradict or fail to support that assumption?",
            "What alternative assumption or implementation strategy should be tested next?",
        ],
        "provisional": True,
    }


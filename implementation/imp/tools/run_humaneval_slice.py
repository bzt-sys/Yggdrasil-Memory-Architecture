from __future__ import annotations

"""
Run a HumanEval mechanism-validation slice through Yggdrasil's structured API.

This runner intentionally does NOT automate the interactive CLI.  It imports
YggdrasilAgentSession directly and receives structured prompt responses,
diagnostics, traces, collation summaries, and HUD snapshots.

This is for mechanism validation, not leaderboard evaluation.
"""

import argparse
import ast
import json
import multiprocessing as mp
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow running as `python imp/tools/run_humaneval_slice.py` from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imp.agent_session import AgentSessionConfig, YggdrasilAgentSession  # noqa: E402
from imp.deliberation.uncertainty_gate import decide_uncertainty_gate  # noqa: E402
from imp.deliberation.task_analysis import analyze_task, format_task_analysis_context  # noqa: E402
from imp.deliberation.information_retrieval import retrieve_information, format_information_retrieval_context  # noqa: E402
from imp.deliberation.orchestration import decide_orchestration  # noqa: E402
from imp.environment.corpus import LocalCorpus  # noqa: E402


DEFAULT_SLICE = Path("data/humaneval/humaneval_mechanism_seed.jsonl")
DEFAULT_RUN_DIR = Path("runs/humaneval")
DEFAULT_MODEL = r"D:\Yggdrasil\imp_split\models\Mistral-7B-Instruct-v0.3"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def infer_resume_start(records: list[dict[str, Any]], default_start: int = 0) -> int:
    """Return the next task index after completed records.

    A task is considered complete if it has a normal completed_utc field and
    no top-level error. This intentionally treats max-attempt failures as
    complete so long developmental runs can resume at the next task instead
    of repeating failed items unless --start is supplied explicitly.
    """
    completed = [
        int(r.get("run_index", -1))
        for r in records
        if r.get("completed_utc") and not r.get("error") and r.get("run_index") is not None
    ]
    if not completed:
        return int(default_start)
    return max(completed) + 1


def append_transcript(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)



def extract_python_code(response: str) -> str:
    """Extract executable Python from a model response.

    Prefer fenced python blocks. If the model starts a fence but is cut off
    before closing it, use the unterminated fenced content rather than the
    surrounding prose. If no fence exists, try to trim to the first `def` or
    `import` line. This intentionally does not use the canonical solution.
    """
    text = response or ""

    # Closed fenced blocks.
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks if block.strip()).strip()

    # Unterminated fenced block: common when local generation maxes out.
    m = re.search(r"```(?:python|py)?\s*(.*)\Z", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # No fence: remove leading prose by starting at likely Python code.
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "import ", "from ", "class ")):
            return "\n".join(lines[idx:]).strip()

    return text.strip()


def normalize_code_only_response(response: str) -> tuple[str, dict[str, Any]]:
    """Return executable-code surface plus a bounded format diagnostic.

    HumanEval prompts ask for code only, but local instruct models often wrap
    code in prose/markdown. The harness should evaluate/log the executable
    candidate while preserving a format violation as developmental evidence.
    """
    raw = response or ""
    extracted = extract_python_code(raw)
    changed = extracted.strip() != raw.strip()
    has_fence = "```" in raw
    leading_prose = bool(raw.strip()) and not raw.lstrip().startswith(("def ", "import ", "from ", "class "))
    diagnostic = {
        "format_violation": bool(changed or has_fence or leading_prose),
        "normalization_applied": bool(changed),
        "had_markdown_fence": bool(has_fence),
        "had_leading_prose": bool(leading_prose),
        "raw_chars": len(raw),
        "normalized_chars": len(extracted),
    }
    return extracted.strip(), diagnostic


class _YggAssertionDiagnostic(AssertionError):
    """Assertion failure carrying bounded environment diagnostic context."""

    def __init__(self, source: str, detail: dict[str, Any]):
        super().__init__(source)
        self.source = source
        self.detail = detail


def _safe_repr(value: Any, limit: int = 500) -> str:
    try:
        text = repr(value)
    except BaseException:
        text = f"<{type(value).__name__}: repr failed>"
    return text[:limit]


def _diagnose_assert_expression(source: str, local_ns: dict[str, Any], global_ns: dict[str, Any]) -> dict[str, Any]:
    """Return bounded diagnostic details for a failed assertion expression.

    This deliberately exposes only the failing expression and evaluated operand
    summaries. It does not reveal the full test suite, canonical solution, or
    future tests. The intent is IDE-style feedback for developmental repair.
    """
    detail: dict[str, Any] = {"assertion": source}
    try:
        expr_ast = ast.parse(source, mode="eval").body
    except BaseException:
        return detail

    if isinstance(expr_ast, ast.Compare):
        try:
            left_value = eval(compile(ast.Expression(expr_ast.left), "<assert-left>", "eval"), global_ns, local_ns)
            detail["left"] = _safe_repr(left_value)
        except BaseException as exc:
            detail["left_eval_error"] = repr(exc)
        comparators: list[str] = []
        for comp in expr_ast.comparators:
            try:
                comparators.append(_safe_repr(eval(compile(ast.Expression(comp), "<assert-right>", "eval"), global_ns, local_ns)))
            except BaseException as exc:
                comparators.append(f"<eval error {type(exc).__name__}: {exc!r}>")
        detail["comparators"] = comparators
        detail["operators"] = [type(op).__name__ for op in expr_ast.ops]
    else:
        try:
            detail["value"] = _safe_repr(eval(compile(ast.Expression(expr_ast), "<assert-expr>", "eval"), global_ns, local_ns))
        except BaseException as exc:
            detail["value_eval_error"] = repr(exc)
    return detail


class _YggAssertInstrumenter(ast.NodeTransformer):
    """Rewrite `assert X` into a diagnostic failure branch for one failing case."""

    def visit_Assert(self, node: ast.Assert) -> Any:  # noqa: N802 - ast API name
        self.generic_visit(node)
        source = ast.unparse(node.test) if hasattr(ast, "unparse") else "<assertion>"
        diagnostic_call = ast.Call(
            func=ast.Name(id="_ygg_diagnose_assert", ctx=ast.Load()),
            args=[ast.Constant(source), ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]), ast.Call(func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[])],
            keywords=[],
        )
        raise_call = ast.Raise(
            exc=ast.Call(func=ast.Name(id="_YggAssertionDiagnostic", ctx=ast.Load()), args=[ast.Constant(source), diagnostic_call], keywords=[]),
            cause=None,
        )
        return ast.copy_location(ast.If(test=ast.UnaryOp(op=ast.Not(), operand=node.test), body=[raise_call], orelse=[]), node)


def _instrument_humaneval_test(test: str) -> str:
    """Instrument asserts so the first failing check yields concrete diagnostics."""
    try:
        tree = ast.parse(test)
        tree = _YggAssertInstrumenter().visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, "<humaneval_instrumented_test>", "exec")  # type: ignore[return-value]
    except BaseException:
        return test


def _humaneval_exec_worker(code: str, test: str, entry_point: str, q: "mp.Queue[Any]", diagnostic_cases: bool = True) -> None:
    """Execute generated HumanEval code in a child process."""
    ns: dict[str, Any] = {
        "_YggAssertionDiagnostic": _YggAssertionDiagnostic,
        "_ygg_diagnose_assert": _diagnose_assert_expression,
    }
    try:
        exec(code, ns, ns)
        test_obj = _instrument_humaneval_test(test) if diagnostic_cases else test
        exec(test_obj, ns, ns)
        if "check" not in ns:
            raise AssertionError("HumanEval test did not define check(candidate).")
        if entry_point not in ns:
            raise AssertionError(f"Generated code did not define entry point {entry_point!r}.")
        ns["check"](ns[entry_point])
        q.put({"passed": True, "error": "", "error_type": ""})
    except _YggAssertionDiagnostic as exc:
        q.put({
            "passed": False,
            "error": repr(exc),
            "error_type": "AssertionError",
            "failure_diagnostic": exc.detail,
            "signal_quality": "concrete_failing_assertion",
        })
    except BaseException as exc:  # benchmark sandbox: capture failures, don't crash runner
        q.put({"passed": False, "error": repr(exc), "error_type": type(exc).__name__})


def evaluate_humaneval_response(task: dict[str, Any], response: str, timeout_s: float = 5.0, diagnostic_cases: bool = True) -> dict[str, Any]:
    """Return binary pass/fail against HumanEval tests without revealing solutions.

    HumanEval does not have one unique textual answer. It has many possible
    correct implementations. The binary signal is whether the generated code
    passes the task tests.
    """
    prompt = str(task.get("prompt", ""))
    test = str(task.get("test", ""))
    entry_point = str(task.get("entry_point", ""))
    extracted, format_diagnostic = normalize_code_only_response(response)

    candidates: list[tuple[str, str]] = []
    if entry_point and f"def {entry_point}" in extracted:
        candidates.append(("response_only", extracted))
    candidates.append(("prompt_plus_response", prompt + "\n" + extracted))

    last: dict[str, Any] = {"passed": False, "error": "no candidate executed", "error_type": "NoCandidate"}
    for mode, code in candidates:
        ctx = mp.get_context("spawn")
        q: mp.Queue[Any] = ctx.Queue()
        proc = ctx.Process(target=_humaneval_exec_worker, args=(code, test, entry_point, q, bool(diagnostic_cases)))
        proc.start()
        proc.join(float(timeout_s))
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            last = {"passed": False, "error": f"timeout after {timeout_s}s", "error_type": "TimeoutError", "mode": mode}
            continue
        if not q.empty():
            last = q.get()
            last["mode"] = mode
        else:
            last = {"passed": False, "error": f"worker exited with code {proc.exitcode}", "error_type": "WorkerExit", "mode": mode}
        if last.get("passed"):
            break

    last["entry_point"] = entry_point
    last["extracted_chars"] = len(extracted)
    last["format_diagnostic"] = format_diagnostic
    if format_diagnostic.get("format_violation"):
        last.setdefault("warnings", []).append("format_violation_normalized_to_code_only")
    return last



def build_environment_feedback(
    *,
    task_id: str,
    attempt: int,
    correctness: dict[str, Any],
    prompt: str,
    response: str,
    use_environment_bridge: bool = False,
    environment_corpus_k: int = 0,
) -> str:
    """Build objective feedback for the substrate.

    The legacy feedback line is preserved for compatibility. When the
    environment bridge flag is enabled, append richer action/signal context so
    failure-action analysis can compare the original task prompt, committed
    response, extracted code, and environment signal.
    """
    extracted, response_format_diagnostic = normalize_code_only_response(response)
    base = (
        f"HumanEval objective feedback for {task_id} attempt {attempt}: "
        f"passed={bool(correctness.get('passed'))}; "
        f"error_type={correctness.get('error_type', '')}; "
        f"error={str(correctness.get('error', ''))[:500]}; "
        f"entry_point={correctness.get('entry_point', '')}; "
        f"mode={correctness.get('mode', '')}; "
        f"format_violation={bool((correctness.get('format_diagnostic') or response_format_diagnostic).get('format_violation'))}."
    )
    if not use_environment_bridge:
        return base

    prompt_excerpt = str(prompt or "")[:3500]
    response_excerpt = str(response or "")[:3500]
    extracted_excerpt = str(extracted or "")[:3500]
    error_type = str(correctness.get("error_type", ""))
    signal_note = ""
    if error_type == "AssertionError":
        signal_note = (
            "Signal interpretation hint: generated code executed, but at least one "
            "asserted behavioral expectation failed. Compare the task specification "
            "against the committed implementation and identify likely violated assumptions."
        )
    elif error_type:
        signal_note = (
            "Signal interpretation hint: compare the environment error against the "
            "symbols, imports, helpers, control flow, and assumptions in the committed response."
        )
    else:
        signal_note = "Signal interpretation hint: no failure signal was emitted."

    return (
        base
        + "\n[environment_bridge]\n"
        + f"task_id={task_id}\n"
        + f"attempt={attempt}\n"
        + f"passed={bool(correctness.get('passed'))}\n"
        + f"error_type={error_type}\n"
        + f"error={str(correctness.get('error', ''))[:1000]}\n"
        + f"entry_point={correctness.get('entry_point', '')}\n"
        + f"mode={correctness.get('mode', '')}\n"
        + f"format_diagnostic={json.dumps(correctness.get('format_diagnostic') or response_format_diagnostic, ensure_ascii=False)}\n"
        + f"corpus_k={int(environment_corpus_k)}\n"
        + ("[failure_diagnostic]\n" + json.dumps(correctness.get("failure_diagnostic"), ensure_ascii=False, indent=2)[:2000] + "\n[/failure_diagnostic]\n" if correctness.get("failure_diagnostic") else "")
        + signal_note
        + "\n\n[task_prompt]\n"
        + prompt_excerpt
        + "\n[/task_prompt]\n\n[committed_response]\n"
        + response_excerpt
        + "\n[/committed_response]\n\n[extracted_code]\n"
        + extracted_excerpt
        + "\n[/extracted_code]\n"
        + "[/environment_bridge]"
    )

def compact_lifecycle_signal(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for row in events:
        typ = str(row.get("type", "unknown"))
        by_type[typ] = by_type.get(typ, 0) + 1
    paradigm_events = _extract_paradigm_selection_events(events)
    paradigm_phase_counts: dict[str, int] = {}
    for ev in paradigm_events:
        phase = str(ev.get("phase") or "unknown")
        paradigm_phase_counts[phase] = paradigm_phase_counts.get(phase, 0) + 1
    return {
        "event_count": len(events),
        "by_type": by_type,
        "episode_closed": by_type.get("episode_closed", 0),
        "investigations_created": by_type.get("investigation_workspace_created", 0),
        "provisional_outcomes": by_type.get("provisional_outcome_written", 0),
        "candidates_cached": by_type.get("candidate_cached", 0),
        "pedigree_evaluations": by_type.get("candidate_pedigree_evaluated", 0),
        "locality_assignments": by_type.get("artifact_assigned_to_locality", 0),
        "locality_proposals": by_type.get("locality_region_proposed", 0),
        "governance_proposals": by_type.get("governance_proposal_created", 0),
        "developmental_hypotheses": by_type.get("developmental_hypothesis_synthesized", 0),
        "developmental_transitions": by_type.get("developmental_transition", 0),
        "artifact_injection_transitions": by_type.get("artifact_injection_transition", 0),
        "recurrent_episode_collations": by_type.get("recurrent_episode_collated", 0),
        "developmental_hypotheses_validation_candidates": sum(1 for e in events if e.get("type") == "developmental_hypothesis_synthesized" and (e.get("data") or {}).get("validation_state") == "validation_candidate"),
        "governance_promotions": sum(1 for e in events if e.get("type") in {"governance_proposal_promotion_evaluated", "governance_promotion_evaluated"} and (e.get("data") or {}).get("decision") == "promote"),
        "governance_holds": sum(1 for e in events if e.get("type") in {"governance_proposal_promotion_evaluated", "governance_promotion_evaluated"} and (e.get("data") or {}).get("decision") == "hold"),
        "promotion_evaluations": by_type.get("governance_proposal_promotion_evaluated", 0) + by_type.get("governance_promotion_evaluated", 0),
        "collation_completed": by_type.get("collation_period_completed", 0) + by_type.get("collation_period_complete", 0),
        "pre_answer_investigations": by_type.get("pre_answer_investigation_created", 0),
        "reasoning_paradigm_selections": len(paradigm_events),
        "reasoning_paradigm_phase_counts": paradigm_phase_counts,
    }

def _extract_selected_paradigm_ids(selection: dict[str, Any]) -> list[str]:
    """Return selected paradigm ids from a selection packet. Compatibility accepts old stack fields."""
    if not isinstance(selection, dict):
        return []
    stack = selection.get("ordered_stack_ids") or selection.get("selected_reasoning_paradigm_ids")
    if stack:
        return [str(x) for x in stack if x]
    selected = selection.get("selected") or []
    ids: list[str] = []
    for row in selected:
        if isinstance(row, dict):
            pid = row.get("paradigm_id") or row.get("id")
            if pid:
                ids.append(str(pid))
    return ids


def _extract_paradigm_selection_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract replay-visible reasoning paradigm selection packets from lifecycle rows."""
    rows: list[dict[str, Any]] = []
    for row in events or []:
        data = row.get("data") if isinstance(row.get("data"), dict) else row
        if not isinstance(data, dict):
            continue
        if data.get("kind") == "reasoning_paradigm_selection":
            payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
            selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
            phase = payload.get("phase") or selection.get("phase") or ("pre_action" if payload.get("mode") == "pre_answer_investigation" else "unknown")
            rows.append({
                "event_id": data.get("event_id"),
                "phase": phase,
                "mode": payload.get("mode"),
                "selected_reasoning_paradigm_ids": payload.get("selected_reasoning_paradigm_ids") or _extract_selected_paradigm_ids(selection),
                "selection": selection,
                "payload_hash": data.get("payload_hash"),
            })
            continue
        # Compatibility: pre-answer investigations may contain the selected
        # stack directly even when no separate developmental_event packet was
        # emitted into the runner lifecycle window. Surface it for records and
        # summaries so paradigm observability does not silently drop to zero.
        if row.get("type") == "pre_answer_investigation_created" and isinstance(data.get("reasoning_paradigm_selection"), dict):
            selection = data.get("reasoning_paradigm_selection") or {}
            rows.append({
                "event_id": data.get("pre_answer_investigation_id"),
                "phase": selection.get("phase") or "pre_action",
                "mode": data.get("mode") or "pre_answer_investigation",
                "selected_reasoning_paradigm_ids": _extract_selected_paradigm_ids(selection),
                "selection": selection,
                "payload_hash": None,
            })
    return rows


def _context_fingerprint(context_text: str, *, limit: int = 2400) -> str:
    import hashlib
    normalized = "\n".join((context_text or "")[:limit].split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def compute_developmental_delta(current: dict[str, Any], previous: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Compare two attempts for the same task and summarize developmental change.

    The metric is intentionally structural and benchmark-agnostic: it asks whether
    retrieved artifacts, reasoning paradigm selections, context, diagnostics, and
    feedback signals changed between attempts.
    """
    if not previous:
        return {"available": False, "reason": "first_attempt_for_task", "score": 0.0, "changed_dimensions": []}

    def ids(row: dict[str, Any], key: str) -> set[str]:
        return {str(x) for x in (row.get(key) or []) if x}

    dimensions: dict[str, bool] = {}
    dimensions["retrieved_candidates_changed"] = ids(current, "selected_candidate_ids") != ids(previous, "selected_candidate_ids")
    dimensions["retrieved_investigations_changed"] = ids(current, "selected_investigation_ids") != ids(previous, "selected_investigation_ids")
    dimensions["retrieved_governance_changed"] = ids(current, "selected_governance_proposal_ids") != ids(previous, "selected_governance_proposal_ids")
    dimensions["pre_action_paradigm_changed"] = list(current.get("pre_action_paradigm_stack") or []) != list(previous.get("pre_action_paradigm_stack") or [])
    dimensions["post_action_paradigm_changed"] = list(current.get("post_action_paradigm_stack") or []) != list(previous.get("post_action_paradigm_stack") or [])
    dimensions["context_fingerprint_changed"] = current.get("context_fingerprint") != previous.get("context_fingerprint")
    dimensions["error_type_changed"] = ((current.get("correctness") or {}).get("error_type")) != ((previous.get("correctness") or {}).get("error_type"))
    dimensions["outcome_changed"] = bool((current.get("correctness") or {}).get("passed")) != bool((previous.get("correctness") or {}).get("passed"))

    cur_life = current.get("lifecycle_signal") or {}
    prev_life = previous.get("lifecycle_signal") or {}
    for key in ("pre_answer_investigations", "investigations_created", "candidates_cached", "locality_assignments", "collation_completed", "governance_proposals", "reasoning_paradigm_selections"):
        dimensions[f"{key}_count_changed"] = int(cur_life.get(key) or 0) != int(prev_life.get(key) or 0)

    changed = [k for k, v in dimensions.items() if v]
    score = round(len(changed) / max(1, len(dimensions)), 3)
    return {
        "available": True,
        "previous_attempt": previous.get("attempt"),
        "current_attempt": current.get("attempt"),
        "score": score,
        "changed_dimension_count": len(changed),
        "total_dimensions": len(dimensions),
        "changed_dimensions": changed,
        "dimensions": dimensions,
        "interpretation": "nonzero_delta_indicates_developmental_state_changed_between_attempts",
    }


def build_prompt(task: dict[str, Any], mode: str = "standard", include_tests: bool = False) -> str:
    task_id = task.get("task_id", "unknown")
    prompt = task.get("prompt", "")

    # Keep the default HumanEval prompt neutral. Earlier versions included
    # generic warnings about imports/libraries; those phrases can accidentally
    # trigger Yggdrasil's library-governance router and obscure whether the
    # benchmark task itself is being injected correctly.
    if mode == "minimal":
        body = f"Complete this Python function. Return executable Python code only.\n\n{prompt}\n"
    elif mode == "uncertainty_pressure":
        body = f"""HumanEval mechanism-validation task: {task_id}

Complete the following Python function correctly from the signature and docstring.
Return one complete executable Python implementation. Do not include prose, markdown, examples, or explanations.
If you use helpers, define them in the returned code.

{prompt}
"""
    else:
        body = f"""HumanEval mechanism-validation task: {task_id}

Complete the following Python function correctly from the signature and docstring.
Return one complete executable Python implementation only. Do not include prose, markdown, examples, or explanations.
If you use helpers, define them in the returned code.

{prompt}
"""

    if include_tests:
        body += "\nVisible tests for debugging this harness, not leaderboard-style evaluation:\n"
        body += str(task.get("test", ""))
        body += "\n"

    return body






def _build_private_repair_prompt(task_prompt: str, candidate_response: str, correctness: dict[str, Any], round_index: int) -> str:
    """Build bounded private repair prompt for deliberation mode.

    This prompt is not stored as a user task prompt and is not a reasoning
    framework. It gives the model objective environment evidence and asks for a
    revised executable implementation.
    """
    failure_diag = correctness.get("failure_diagnostic")
    diag_text = json.dumps(failure_diag, ensure_ascii=False, indent=2)[:1600] if failure_diag else ""
    return (
        "You are revising a candidate Python implementation after objective local execution feedback.\n"
        "Return one complete executable Python implementation only. Do not include prose or markdown.\n\n"
        "[task]\n" + task_prompt[:3500] + "\n[/task]\n\n"
        "[previous_candidate]\n" + str(candidate_response or "")[:3500] + "\n[/previous_candidate]\n\n"
        "[environment_feedback]\n"
        f"round={round_index}\n"
        f"passed={bool(correctness.get('passed'))}\n"
        f"error_type={correctness.get('error_type', '')}\n"
        f"error={str(correctness.get('error', ''))[:1000]}\n"
        + ("failure_diagnostic=" + diag_text + "\n" if diag_text else "")
        + "[/environment_feedback]\n"
    )


def run_private_deliberation(
    *,
    session_api: YggdrasilAgentSession,
    task: dict[str, Any],
    prompt: str,
    gate: Any,
    eval_timeout: float,
    diagnostic_cases: bool,
) -> dict[str, Any]:
    """Generate/test/revise privately, then return the final candidate response.

    The runner commits only the selected final candidate through run_prompt(...,
    forced_response=...). The private trace is saved in records/transcript so it
    remains auditable without flooding the actor-facing task context.
    """
    context_text = ""
    if bool(getattr(gate, "context_enabled", False)):
        try:
            context_text = session_api.api.get_context_text(prompt_text=prompt)
        except Exception:
            context_text = ""

    rounds: list[dict[str, Any]] = []
    candidate_prompt = prompt
    response = session_api.raw_generate(candidate_prompt, context_text=context_text)
    correctness = evaluate_humaneval_response(task, response, timeout_s=eval_timeout, diagnostic_cases=diagnostic_cases)
    rounds.append({
        "round": 1,
        "kind": "initial_candidate",
        "response_preview": response[:1000],
        "response_chars": len(response),
        "correctness": correctness,
    })

    max_rounds = max(1, int(getattr(gate, "max_deliberation_rounds", 1) or 1))
    current_response = response
    current_correctness = correctness
    for round_index in range(2, max_rounds + 1):
        if current_correctness.get("passed"):
            break
        repair_prompt = _build_private_repair_prompt(prompt, current_response, current_correctness, round_index)
        current_response = session_api.raw_generate(repair_prompt, context_text=context_text)
        current_correctness = evaluate_humaneval_response(task, current_response, timeout_s=eval_timeout, diagnostic_cases=diagnostic_cases)
        rounds.append({
            "round": round_index,
            "kind": "repair_candidate",
            "response_preview": current_response[:1000],
            "response_chars": len(current_response),
            "correctness": current_correctness,
        })
        if current_correctness.get("passed"):
            break

    return {
        "used": True,
        "round_count": len(rounds),
        "final_round": rounds[-1] if rounds else None,
        "rounds": rounds,
        "final_response": current_response,
        "final_correctness": current_correctness,
        "private_context_chars": len(context_text),
    }


def _pct(n: int, d: int) -> float:
    return round((100.0 * n / d), 2) if d else 0.0


def summarize_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate HumanEval mechanism-validation records into a run report.

    This is not a leaderboard scorer. It summarizes whether the experimental
    harness exercised the Yggdrasil lifecycle: attempts, pass/fail, retries,
    investigations, candidates, proposals, collation, and retrieved short-term
    evidence. The goal is to make long developmental runs inspectable at a
    glance while preserving per-attempt JSONL records as the source of truth.
    """
    attempts = [r for r in records if r.get("task_id")]
    task_ids = sorted({str(r.get("task_id")) for r in attempts})
    task_groups: dict[str, list[dict[str, Any]]] = {}
    for r in attempts:
        task_groups.setdefault(str(r.get("task_id")), []).append(r)

    passed_attempts = [r for r in attempts if (r.get("correctness") or {}).get("passed")]
    failed_attempts = [r for r in attempts if r.get("correctness") and not (r.get("correctness") or {}).get("passed")]
    errored_attempts = [r for r in attempts if r.get("error")]

    tasks_passed = 0
    attempts_to_pass: list[int] = []
    task_rows: list[dict[str, Any]] = []
    for task_id, rows in task_groups.items():
        ordered = sorted(rows, key=lambda x: int(x.get("attempt") or 0))
        pass_row = next((r for r in ordered if (r.get("correctness") or {}).get("passed")), None)
        if pass_row:
            tasks_passed += 1
            attempts_to_pass.append(int(pass_row.get("attempt") or len(ordered)))
        task_rows.append({
            "task_id": task_id,
            "attempts": len(ordered),
            "passed": bool(pass_row),
            "attempt_to_pass": int(pass_row.get("attempt")) if pass_row else None,
            "final_error_type": (ordered[-1].get("correctness") or {}).get("error_type") if ordered else None,
            "selected_candidate_ids": [cid for r in ordered for cid in (r.get("selected_candidate_ids") or []) if cid],
            "selected_investigation_ids": [iid for r in ordered for iid in (r.get("selected_investigation_ids") or []) if iid],
        })

    lifecycle_totals: dict[str, int] = {}
    for r in attempts:
        signal = r.get("lifecycle_signal") or {}
        for key, value in signal.items():
            if isinstance(value, int):
                lifecycle_totals[key] = lifecycle_totals.get(key, 0) + value

    error_types: dict[str, int] = {}
    for r in failed_attempts:
        err = str((r.get("correctness") or {}).get("error_type") or "unknown")
        error_types[err] = error_types.get(err, 0) + 1

    candidate_retrieval_attempts = sum(1 for r in attempts if r.get("selected_candidate_ids"))
    investigation_retrieval_attempts = sum(1 for r in attempts if r.get("selected_investigation_ids"))
    context_candidate_attempts = sum(1 for r in attempts if r.get("context_has_candidate_cache"))
    context_investigation_attempts = sum(1 for r in attempts if r.get("context_has_investigations"))
    pre_answer_attempts = sum(1 for r in attempts if r.get("pre_answer_investigation"))
    collation_attempts = sum(1 for r in attempts if r.get("collate_ran"))
    pre_paradigm_attempts = sum(1 for r in attempts if r.get("pre_action_paradigm_stack"))
    post_paradigm_attempts = sum(1 for r in attempts if r.get("post_action_paradigm_stack"))
    delta_rows = [r.get("developmental_delta") for r in attempts if (r.get("developmental_delta") or {}).get("available")]
    avg_delta_score = round(sum(float(d.get("score") or 0.0) for d in delta_rows) / len(delta_rows), 3) if delta_rows else None
    changed_delta_rows = sum(1 for d in delta_rows if int(d.get("changed_dimension_count") or 0) > 0)
    gate_rows = [r.get("uncertainty_gate") for r in attempts if (r.get("uncertainty_gate") or {}).get("route")]
    gate_routes: dict[str, int] = {}
    for g in gate_rows:
        route = str(g.get("route") or "unknown")
        gate_routes[route] = gate_routes.get(route, 0) + 1
    deliberation_rows = [r.get("deliberation_trace") for r in attempts if (r.get("deliberation_trace") or {}).get("used")]
    deliberation_passes = sum(1 for r in attempts if (r.get("deliberation_trace") or {}).get("used") and (r.get("correctness") or {}).get("passed"))
    orchestration_rows = [r.get("orchestration") for r in attempts if (r.get("orchestration") or {}).get("primary_route")]
    orchestration_routes: dict[str, int] = {}
    orchestration_operations: dict[str, int] = {}
    for row in orchestration_rows:
        route = str(row.get("primary_route") or "unknown")
        orchestration_routes[route] = orchestration_routes.get(route, 0) + 1
        for op in row.get("operations") or []:
            op = str(op)
            orchestration_operations[op] = orchestration_operations.get(op, 0) + 1

    return {
        "purpose": "HumanEval mechanism-validation run summary; not leaderboard evaluation.",
        "record_count": len(records),
        "attempt_count": len(attempts),
        "task_count": len(task_ids),
        "tasks_passed": tasks_passed,
        "tasks_failed_or_unresolved": max(0, len(task_ids) - tasks_passed),
        "task_pass_rate_percent": _pct(tasks_passed, len(task_ids)),
        "attempt_pass_rate_percent": _pct(len(passed_attempts), len(attempts)),
        "failed_attempt_count": len(failed_attempts),
        "errored_attempt_count": len(errored_attempts),
        "avg_attempts_to_pass": round(sum(attempts_to_pass) / len(attempts_to_pass), 3) if attempts_to_pass else None,
        "max_attempts_to_pass": max(attempts_to_pass) if attempts_to_pass else None,
        "error_types": dict(sorted(error_types.items(), key=lambda kv: (-kv[1], kv[0]))),
        "lifecycle_totals": lifecycle_totals,
        "retrieval": {
            "attempts_with_selected_candidates": candidate_retrieval_attempts,
            "attempts_with_selected_investigations": investigation_retrieval_attempts,
            "attempts_with_candidate_cache_in_context": context_candidate_attempts,
            "attempts_with_investigations_in_context": context_investigation_attempts,
            "candidate_retrieval_rate_percent": _pct(candidate_retrieval_attempts, len(attempts)),
            "investigation_retrieval_rate_percent": _pct(investigation_retrieval_attempts, len(attempts)),
            "attempts_with_pre_answer_investigation": pre_answer_attempts,
            "pre_answer_investigation_rate_percent": _pct(pre_answer_attempts, len(attempts)),
        },
        "collation": {
            "collation_attempts": collation_attempts,
            "collation_rate_percent": _pct(collation_attempts, len(attempts)),
        },
        "reasoning_paradigms": {
            "attempts_with_pre_action_paradigm": pre_paradigm_attempts,
            "attempts_with_post_action_paradigm": post_paradigm_attempts,
            "pre_action_paradigm_rate_percent": _pct(pre_paradigm_attempts, len(attempts)),
            "post_action_paradigm_rate_percent": _pct(post_paradigm_attempts, len(attempts)),
        },
        "developmental_delta": {
            "delta_comparisons": len(delta_rows),
            "comparisons_with_nonzero_delta": changed_delta_rows,
            "nonzero_delta_rate_percent": _pct(changed_delta_rows, len(delta_rows)),
            "avg_delta_score": avg_delta_score,
        },
        "orchestration": {
            "attempts_with_orchestration": len(orchestration_rows),
            "orchestration_rate_percent": _pct(len(orchestration_rows), len(attempts)),
            "routes": dict(sorted(orchestration_routes.items())),
            "operations": dict(sorted(orchestration_operations.items())),
            "unresolved_knowledge_gaps": sum(1 for r in orchestration_rows if r.get("unresolved_knowledge_gap")),
        },
        "uncertainty_gate": {
            "attempts_with_gate": len(gate_rows),
            "gate_rate_percent": _pct(len(gate_rows), len(attempts)),
            "routes": dict(sorted(gate_routes.items())),
            "deliberation_attempts": len(deliberation_rows),
            "deliberation_rate_percent": _pct(len(deliberation_rows), len(attempts)),
            "deliberation_passes": deliberation_passes,
            "deliberation_pass_rate_percent": _pct(deliberation_passes, len(deliberation_rows)),
        },
        "tasks": sorted(task_rows, key=lambda x: x["task_id"]),
    }


def write_run_report(summary: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# HumanEval Mechanism-Validation Run Report\n\n")
    lines.append("This report summarizes Yggdrasil lifecycle behavior. It is not a leaderboard score.\n\n")
    lines.append("## Overview\n\n")
    lines.append(f"- Tasks: {summary.get('task_count')}\n")
    lines.append(f"- Attempts: {summary.get('attempt_count')}\n")
    lines.append(f"- Tasks passed: {summary.get('tasks_passed')} / {summary.get('task_count')} ({summary.get('task_pass_rate_percent')}%)\n")
    lines.append(f"- Attempt pass rate: {summary.get('attempt_pass_rate_percent')}%\n")
    lines.append(f"- Average attempts to pass: {summary.get('avg_attempts_to_pass')}\n")
    lines.append(f"- Errored attempts: {summary.get('errored_attempt_count')}\n\n")

    lines.append("## Lifecycle Totals\n\n")
    lifecycle = summary.get("lifecycle_totals") or {}
    if lifecycle:
        for key in sorted(lifecycle):
            lines.append(f"- {key}: {lifecycle[key]}\n")
    else:
        lines.append("- No lifecycle events recorded.\n")
    lines.append("\n")

    lines.append("## Retrieval\n\n")
    retrieval = summary.get("retrieval") or {}
    for key in sorted(retrieval):
        lines.append(f"- {key}: {retrieval[key]}\n")
    lines.append("\n")

    lines.append("## Collation\n\n")
    collation = summary.get("collation") or {}
    for key in sorted(collation):
        lines.append(f"- {key}: {collation[key]}\n")
    lines.append("\n")

    lines.append("## Reasoning Paradigms\n\n")
    paradigms = summary.get("reasoning_paradigms") or {}
    for key in sorted(paradigms):
        lines.append(f"- {key}: {paradigms[key]}\n")
    lines.append("\n")

    lines.append("## Developmental Delta\n\n")
    deltas = summary.get("developmental_delta") or {}
    for key in sorted(deltas):
        lines.append(f"- {key}: {deltas[key]}\n")
    lines.append("\n")

    orchestration = summary.get("orchestration") or {}
    lines.append("## Orchestration\n\n")
    lines.append(f"- attempts_with_orchestration: {orchestration.get('attempts_with_orchestration', 0)}\n")
    lines.append(f"- orchestration_rate_percent: {orchestration.get('orchestration_rate_percent', 0)}\n")
    lines.append(f"- unresolved_knowledge_gaps: {orchestration.get('unresolved_knowledge_gaps', 0)}\n")
    lines.append(f"- routes: {orchestration.get('routes', {})}\n")
    lines.append(f"- operations: {orchestration.get('operations', {})}\n\n")

    lines.append("## Uncertainty Gate / Deliberation\n\n")
    gate = summary.get("uncertainty_gate") or {}
    for key in sorted(gate):
        lines.append(f"- {key}: {gate[key]}\n")
    lines.append("\n")

    lines.append("## Error Types\n\n")
    errors = summary.get("error_types") or {}
    if errors:
        for key, val in errors.items():
            lines.append(f"- {key}: {val}\n")
    else:
        lines.append("- No failed attempts recorded.\n")
    lines.append("\n")

    lines.append("## Per-Task Summary\n\n")
    lines.append("| Task | Attempts | Passed | Attempt to pass | Final error | Retrieved candidates | Retrieved investigations |\n")
    lines.append("|---|---:|---:|---:|---|---:|---:|\n")
    for row in summary.get("tasks") or []:
        lines.append(
            f"| {row.get('task_id')} | {row.get('attempts')} | {row.get('passed')} | "
            f"{row.get('attempt_to_pass')} | {row.get('final_error_type')} | "
            f"{len(row.get('selected_candidate_ids') or [])} | {len(row.get('selected_investigation_ids') or [])} |\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
def compact_diag_signal(diag: dict[str, Any]) -> dict[str, Any]:
    hud = diag.get("hud") or {}
    counts = hud.get("counts") or {}
    last = hud.get("last_step") or {}
    candidates = diag.get("recent_candidates") or []
    proposals = diag.get("recent_governance_proposals") or []
    investigations = diag.get("recent_investigations") or []
    episodes = diag.get("recent_episodes") or []
    return {
        "step": counts.get("step"),
        "episodes": counts.get("episodes"),
        "episode_open": counts.get("episode_open"),
        "events": counts.get("events"),
        "outcomes": counts.get("outcomes"),
        "candidates": counts.get("candidates"),
        "recent_episode_count": len(episodes),
        "recent_closed_episode_count": sum(1 for e in episodes if e.get("status") == "closed"),
        "recent_investigation_count": len(investigations),
        "recent_investigation_error_types": [inv.get("error_type") for inv in investigations[:5]],
        "recent_candidate_count": len(candidates),
        "recent_candidate_actions": [c.get("integration_action") for c in candidates[:5]],
        "recent_candidate_authority": [c.get("authority_state") for c in candidates[:5]],
        "recent_candidate_localities": [c.get("locality") for c in candidates[:5]],
        "recent_proposal_count": len(proposals),
        "recent_proposal_statuses": [p.get("status") for p in proposals[:5]],
        "recent_collation_count": len(diag.get("recent_collation_runs") or []),
        "candidate_summary": diag.get("candidate_summary"),
        "state_hash": last.get("state_hash"),
    }


def print_task_summary(record: dict[str, Any]) -> None:
    signal = record.get("diagnostic_signal") or {}
    lifecycle = record.get("lifecycle_signal") or {}
    correctness = record.get("correctness") or {}
    correctness_label = "n/a" if not correctness else ("PASS" if correctness.get("passed") else "FAIL")
    print(
        "diag: "
        f"step={signal.get('step')} "
        f"episodes={signal.get('episodes')} "
        f"events={signal.get('events')} "
        f"outcomes={signal.get('outcomes')} "
        f"candidates={signal.get('candidates')} "
        f"proposals_recent={signal.get('recent_proposal_count')} "
        f"correctness={correctness_label}"
    )
    if lifecycle:
        print(
            "lifecycle: "
            f"closed={lifecycle.get('episode_closed')} "
            f"investigations={lifecycle.get('investigations_created')} "
            f"provisional={lifecycle.get('provisional_outcomes')} "
            f"candidates={lifecycle.get('candidates_cached')} "
            f"pedigree={lifecycle.get('pedigree_evaluations')} "
            f"locality={lifecycle.get('locality_assignments')} "
            f"proposals={lifecycle.get('governance_proposals')} "
            f"paradigms={lifecycle.get('reasoning_paradigm_selections')} "
            f"collation={lifecycle.get('collation_completed')}"
        )
    if record.get("pre_action_paradigm_stack") or record.get("post_action_paradigm_stack"):
        print(
            "paradigms: "
            f"pre={record.get('pre_action_paradigm_stack') or []} "
            f"post={record.get('post_action_paradigm_stack') or []}"
        )
    delta = record.get("developmental_delta") or {}
    if delta.get("available"):
        print(f"developmental_delta: score={delta.get('score')} changed={delta.get('changed_dimension_count')}/{delta.get('total_dimensions')}")


def maybe_int_rating(value: str) -> Optional[int]:
    if value == "":
        return None
    rating = int(value)
    if rating < 0 or rating > 9:
        raise ValueError("rating must be 0-9")
    return rating


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HumanEval slice through Yggdrasil structured session API.")
    parser.add_argument(
        "--profile",
        choices=["baseline", "developmental", "replay", "gated"],
        default=None,
        help=(
            "Experiment profile. baseline keeps ordinary benchmark-style output; "
            "developmental enables environment bridge + paradigm/developmental trace observability; "
            "replay enables developmental observability and resume-friendly defaults."
        ),
    )
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-run", default=None, help="Resume an existing HumanEval run id. Implies --resume and reuses the run/session id unless --session is supplied.")
    parser.add_argument("--start", type=int, default=None, help="Starting task index. If omitted with --resume/--resume-run, inferred from records.jsonl.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["standard", "minimal", "uncertainty_pressure"], default="standard")
    parser.add_argument("--session", default=None)
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--resume", action="store_true", help="Resume the Yggdrasil session store instead of starting an empty substrate session.")
    parser.add_argument("--replay-causal-diagnostics", action="store_true", help="Record canonical per-step state and causal-boundary diagnostics for replay mismatch analysis.")
    parser.add_argument("--debug-lexical", action="store_true", default=True)
    parser.add_argument("--decay", action="store_true", help="Enable decay; default keeps --no-decay semantics for controlled evals.")
    parser.add_argument("--actor", choices=["none", "hf"], default="hf")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--diag-n", type=int, default=8)
    parser.add_argument("--trace-n", type=int, default=12)
    parser.add_argument("--collate-n", type=int, default=10)
    parser.add_argument("--no-collate", action="store_true")
    parser.add_argument("--ablate", dest="ablate_governance", action="store_true", help="Disable Yggdrasil governance/context injection for baseline model-performance runs. Alias for --ablate-governance.")
    parser.add_argument("--ablate-governance", dest="ablate_governance", action="store_true", help="Disable governance/context/pre-answer injection while preserving ordinary run logging for ablation comparison.")
    parser.add_argument("--no-ablate-governance", dest="ablate_governance", action="store_false", help="Explicitly keep governance/context injection enabled.")
    parser.set_defaults(ablate_governance=False)
    parser.add_argument("--retry-until-correct", action="store_true", help="Retry each task until local tests pass or --max-attempts is reached.")
    parser.add_argument("--max-attempts", type=int, default=1, help="Maximum attempts per task. Used with --retry-until-correct; also caps ordinary runs.")
    parser.add_argument("--collate-every-candidates", type=int, default=0, help="If >0, run collation after this many new cached candidates since the last collation. In retry mode, 3 is recommended.")
    parser.add_argument("--inject-attempt-feedback", action="store_true", default=True, help="Record objective pass/fail feedback as an outcome-observation event before rating. Default: true; this does not inject feedback into the next user prompt.")
    parser.add_argument("--no-inject-attempt-feedback", dest="inject_attempt_feedback", action="store_false", help="Disable objective feedback observation events. Not recommended for lifecycle validation.")
    parser.add_argument("--default-rating", default="", help="Fallback rating sent after each task. Empty string skips fallback rating.")
    parser.add_argument("--auto-evaluate", action="store_true", default=True, help="Run generated code against HumanEval tests and use binary pass/fail feedback. Default: true.")
    parser.add_argument("--no-auto-evaluate", dest="auto_evaluate", action="store_false", help="Disable local HumanEval correctness execution.")
    parser.add_argument("--eval-timeout", type=float, default=5.0, help="Seconds allowed for each local HumanEval correctness check.")
    parser.add_argument("--correct-rating", type=int, default=8, help="Rating used when generated code passes tests.")
    parser.add_argument("--incorrect-rating", type=int, default=2, help="Rating used when generated code fails tests.")
    parser.add_argument("--interactive-rating", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--show-tests", action="store_true", help="Include tests in prompt for harness debugging only.")
    parser.add_argument("--show-canonical", action="store_true", help="Print canonical solution to transcript only; never sent to model.")
    parser.add_argument("--print-prompt", action="store_true", help="Print each injected prompt to the terminal before generation.")
    parser.add_argument("--print-response", action="store_true", default=True, help="Print assistant responses to the terminal. Default: true.")
    parser.add_argument("--no-print-response", dest="print_response", action="store_false", help="Do not print assistant responses to the terminal.")
    parser.add_argument("--include-attempt-feedback-in-prompt", action="store_true", default=False, help="Debug-only: include previous failed attempt error summaries in retry prompts. Default: false; objective feedback should enter through outcome events/candidate retrieval instead.")
    parser.add_argument("--record-objective-feedback", action="store_true", default=True, help="Record local evaluator pass/fail/error details into the episode before rating/reflection. Default: true.")
    parser.add_argument("--no-record-objective-feedback", dest="record_objective_feedback", action="store_false")
    parser.add_argument("--no-include-attempt-feedback-in-prompt", dest="include_attempt_feedback_in_prompt", action="store_false")
    parser.add_argument("--retry-prefix", action="store_true", default=False, help="Debug-only: prepend a retry notice to repeated attempts. Default: false so retry prompts remain identical and adaptation must come from substrate context.")
    parser.add_argument("--print-candidate-usage", action="store_true", default=True, help="Print candidate/proposal artifacts selected into the pre-response context. Default: true.")
    parser.add_argument("--no-print-candidate-usage", dest="print_candidate_usage", action="store_false")
    parser.add_argument("--print-context", action="store_true", help="Print the actor context block before each attempt for candidate-cache visibility.")
    parser.add_argument("--print-preanswer", action="store_true", default=True, help="Print pre-answer investigations when they are created. Default: true.")
    parser.add_argument("--print-investigations", action="store_true", default=True, help="Print new investigation workspaces after each attempt. Default: true.")
    parser.add_argument("--no-print-investigations", dest="print_investigations", action="store_false")
    parser.add_argument("--print-reflections", action="store_true", default=True, help="Print new provisional reflections after each attempt. Default: true.")
    parser.add_argument("--no-print-reflections", dest="print_reflections", action="store_false")
    parser.add_argument("--use-environment-bridge", action="store_true", help="Append rich prompt/response/code/error context to objective feedback for failure-action analysis.")
    parser.add_argument("--no-use-environment-bridge", dest="use_environment_bridge", action="store_false", help="Disable environment bridge even when a profile would enable it.")
    parser.add_argument("--environment-corpus-k", type=int, default=0, help="Reserved corpus lookup budget for environment-bridge feedback packets.")
    parser.add_argument("--no-diagnostic-failing-case", dest="diagnostic_failing_case", action="store_false", default=True, help="Disable bounded failing-assertion diagnostics in local environment feedback.")
    parser.add_argument("--print-paradigms", action="store_true", help="Print pre/post reasoning paradigm selections for each attempt.")
    parser.add_argument("--print-developmental-trace", action="store_true", help="Print compact developmental trajectory and attempt-to-attempt delta diagnostics.")
    parser.add_argument("--no-run-summary", action="store_true", help="Disable final run_summary.json / RUN_REPORT.md export.")
    parser.add_argument("--uncertainty-gate", action="store_true", help="Route first-pass context/deliberation through a lightweight uncertainty gate.")
    parser.add_argument("--no-uncertainty-gate", dest="uncertainty_gate", action="store_false")
    parser.add_argument("--uncertainty-model-probe", action="store_true", help="Use a short model self-estimate blended with deterministic gate signals.")
    parser.add_argument("--gate-high-confidence", type=float, default=0.72)
    parser.add_argument("--gate-low-confidence", type=float, default=0.45)
    parser.add_argument("--gate-deliberation-threshold", type=float, default=0.62)
    parser.add_argument("--deliberation-rounds", type=int, default=2, help="Maximum private candidate/test/revision rounds inside one attempt when gated deliberation activates.")
    parser.add_argument("--task-analysis-tool", action="store_true", help="Selectively classify information gaps and produce a compact task decomposition when warranted.")
    parser.add_argument("--no-task-analysis-tool", dest="task_analysis_tool", action="store_false")
    parser.add_argument("--task-analysis-model-plan", action="store_true", default=True, help="Allow a short actor-generated JSON decomposition when the task-analysis tool activates.")
    parser.add_argument("--no-task-analysis-model-plan", dest="task_analysis_model_plan", action="store_false")
    parser.add_argument("--information-retrieval-tool", action="store_true", help="Retrieve grounded reference material when task analysis identifies an external knowledge gap.")
    parser.add_argument("--no-information-retrieval-tool", dest="information_retrieval_tool", action="store_false")
    parser.add_argument("--knowledge-corpus-dir", type=Path, default=None, help="Directory of .md/.txt reference documents available to the local retrieval provider.")
    parser.add_argument("--information-retrieval-k", type=int, default=3, help="Maximum local reference hits injected for a knowledge-limited task.")
    parser.add_argument("--orchestration-layer", action="store_true", help="Select cognitive operations from uncertainty, task structure, evidence, and available tools.")
    parser.add_argument("--no-orchestration-layer", dest="orchestration_layer", action="store_false")
    args = parser.parse_args()

    if args.replay_causal_diagnostics:
        os.environ["YGG_REPLAY_CAUSAL_DIAGNOSTICS"] = "1"

    # Experiment profiles are intentionally applied after argparse so explicit
    # flags still work. These profiles stabilize the runner as the architecture
    # evolves: developmental mechanisms should not disappear behind flag drift.
    if args.profile == "baseline":
        args.ablate_governance = True
        args.use_environment_bridge = False
        args.include_attempt_feedback_in_prompt = False
        args.no_collate = True
        args.print_context = False
        args.print_preanswer = False
        args.print_investigations = False
        args.print_reflections = False
        args.print_candidate_usage = False
        args.print_paradigms = False
        args.print_developmental_trace = False
    elif args.profile == "developmental":
        args.use_environment_bridge = True
        args.print_context = True
        args.print_preanswer = True
        args.print_investigations = True
        args.print_reflections = True
        args.print_candidate_usage = True
        args.print_paradigms = True
        args.print_developmental_trace = True
    elif args.profile == "replay":
        args.resume = True
        args.use_environment_bridge = True
        args.print_context = True
        args.print_preanswer = True
        args.print_investigations = True
        args.print_candidate_usage = True
        args.print_paradigms = True
        args.print_developmental_trace = True
    elif args.profile == "gated":
        args.use_environment_bridge = True
        args.uncertainty_gate = True
        args.task_analysis_tool = True
        args.information_retrieval_tool = True
        args.orchestration_layer = True
        args.print_context = True
        args.print_preanswer = True
        args.print_investigations = True
        args.print_reflections = True
        args.print_candidate_usage = True
        args.print_paradigms = True
        args.print_developmental_trace = True

    tasks = load_jsonl(args.slice)
    knowledge_corpus = LocalCorpus.from_directory(args.knowledge_corpus_dir) if args.knowledge_corpus_dir else LocalCorpus([])

    if args.resume_run and args.run_id and args.resume_run != args.run_id:
        raise ValueError("Use either --resume-run or --run-id for a resumed run, not conflicting values.")

    resume_requested = bool(args.resume or args.resume_run)
    if resume_requested and not args.resume_run and not args.run_id:
        existing_run_dirs = [p for p in args.run_dir.iterdir() if p.is_dir() and (p / "records.jsonl").exists()] if args.run_dir.exists() else []
        if not existing_run_dirs:
            raise ValueError("--resume was supplied but no prior run was found. Use --resume-run <run_id> or run a persisted session first.")
        existing_run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        args.resume_run = existing_run_dirs[0].name

    run_id = args.resume_run or args.run_id or f"humaneval_{utc_stamp()}"
    session = args.session or run_id
    run_dir = args.run_dir / run_id
    records_path = run_dir / "records.jsonl"
    transcript_path = run_dir / "run_transcript.md"
    manifest_path = run_dir / "run_manifest.json"
    diagnostics_dir = run_dir / "diagnostics"
    traces_dir = run_dir / "traces"
    lifecycle_dir = run_dir / "lifecycle"
    correctness_dir = run_dir / "correctness"
    run_summary_path = run_dir / "run_summary.json"
    run_report_path = run_dir / "RUN_REPORT.md"

    run_dir.mkdir(parents=True, exist_ok=True)

    existing_records = load_existing_records(records_path) if resume_requested else []
    effective_start = int(args.start) if args.start is not None else (infer_resume_start(existing_records, 0) if resume_requested else 0)
    selected = tasks[effective_start : effective_start + args.limit if args.limit else None]

    cfg = AgentSessionConfig(
        session=session,
        session_root=args.session_root,
        use_session_store=True,
        resume=resume_requested,
        debug_lexical=bool(args.debug_lexical),
        no_decay=not bool(args.decay),
        actor=args.actor,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )

    manifest = {
        "run_id": run_id,
        "profile": args.profile,
        "created_utc": utc_stamp(),
        "slice": str(args.slice),
        "start": effective_start,
        "limit": args.limit,
        "selected_count": len(selected),
        "session": session,
        "resume_requested": resume_requested,
        "resume_run": args.resume_run,
        "existing_record_count": len(existing_records),
        "config": cfg.__dict__,
        "diag_n": args.diag_n,
        "trace_n": args.trace_n,
        "collate_n": None if args.no_collate else args.collate_n,
        "ablate_governance": bool(args.ablate_governance),
        "prompt_mode": args.prompt_mode,
        "default_rating": args.default_rating,
        "retry_until_correct": bool(args.retry_until_correct),
        "max_attempts": int(args.max_attempts),
        "include_attempt_feedback_in_prompt": bool(args.include_attempt_feedback_in_prompt),
        "print_context": bool(args.print_context),
        "collate_every_candidates": int(args.collate_every_candidates),
        "inject_attempt_feedback": bool(args.inject_attempt_feedback),
        "auto_evaluate": bool(args.auto_evaluate),
        "eval_timeout": args.eval_timeout,
        "correct_rating": args.correct_rating,
        "incorrect_rating": args.incorrect_rating,
        "use_environment_bridge": bool(args.use_environment_bridge),
        "environment_corpus_k": int(args.environment_corpus_k),
        "diagnostic_failing_case": bool(args.diagnostic_failing_case),
        "print_paradigms": bool(args.print_paradigms),
        "print_developmental_trace": bool(args.print_developmental_trace),
        "uncertainty_gate": bool(args.uncertainty_gate),
        "uncertainty_model_probe": bool(args.uncertainty_model_probe),
        "gate_high_confidence": float(args.gate_high_confidence),
        "gate_low_confidence": float(args.gate_low_confidence),
        "gate_deliberation_threshold": float(args.gate_deliberation_threshold),
        "deliberation_rounds": int(args.deliberation_rounds),
        "purpose": "Yggdrasil HumanEval mechanism validation through structured API, not CLI automation.",
        "replay_causal_diagnostics": bool(args.replay_causal_diagnostics),
    }
    if resume_requested and manifest_path.exists():
        resume_manifest_path = run_dir / f"run_manifest_resume_{utc_stamp()}.json"
        write_json(resume_manifest_path, manifest)
    else:
        write_json(manifest_path, manifest)

    print("=" * 88)
    print("Yggdrasil HumanEval Slice Runner — Structured API")
    print("=" * 88)
    print(f"Run id:      {run_id}")
    print(f"Resume:      {resume_requested}")
    print(f"Start:       {effective_start}")
    print(f"Tasks:       {len(selected)}")
    print(f"Records:     {records_path}")
    print(f"Transcript:  {transcript_path}")
    print(f"Manifest:    {manifest_path}")
    print(f"Actor:       {cfg.actor}")
    print(f"Model:       {cfg.model if cfg.actor == 'hf' else '[none]'}")
    print(f"Ablation:    {'ON' if args.ablate_governance else 'OFF'}")
    print("\nLoading Yggdrasil session / model...")

    if resume_requested:
        append_transcript(transcript_path, f"\n\n# HumanEval Run {run_id} — resumed {utc_stamp()}\n\n")
    else:
        append_transcript(transcript_path, f"# HumanEval Run {run_id}\n\n")
    append_transcript(transcript_path, "## Manifest / Resume Manifest\n\n```json\n" + json.dumps(manifest, indent=2, ensure_ascii=False) + "\n```\n\n")

    session_api = YggdrasilAgentSession(cfg)
    if args.ablate_governance:
        ablation_report = session_api.set_ablation(True)
        append_transcript(
            transcript_path,
            "## Ablation\n\n"
            "Governance/context/pre-answer injection disabled for this run.\n\n```json\n"
            + json.dumps(ablation_report, indent=2, ensure_ascii=False)
            + "\n```\n\n",
        )
    print("Session ready.")
    if resume_requested:
        resume_report = getattr(session_api, "resume_report", None)
        print(f"Resume report: {resume_report if resume_report is not None else '[none]'}")
        append_transcript(transcript_path, "## Resume Report\n\n```json\n" + json.dumps(resume_report, indent=2, ensure_ascii=False, default=str) + "\n```\n\n")

    try:
        candidates_since_collation = 0
        for offset, task in enumerate(selected):
            i = effective_start + offset
            task_id = task.get("task_id", f"task_{i}")
            max_attempts = max(1, int(args.max_attempts if args.retry_until_correct else min(args.max_attempts, 1)))

            if args.pause:
                input(f"\nPress Enter to run task {i} ({task_id})...")

            print("\n" + "=" * 88)
            print(f"TASK {i}: {task_id}")
            print("=" * 88)
            append_transcript(transcript_path, f"\n---\n\n## Task {i}: {task_id}\n\n")

            task_passed = False
            task_attempt_records: list[dict[str, Any]] = []
            previous_failures: list[dict[str, Any]] = []

            for attempt in range(1, max_attempts + 1):
                prompt = build_prompt(task, mode=args.prompt_mode, include_tests=bool(args.show_tests))
                if args.include_attempt_feedback_in_prompt and previous_failures:
                    # Debug-only escape hatch. Default runs keep the task prompt clean;
                    # objective feedback is recorded as an outcome/event and should
                    # reappear only through candidate/context retrieval.
                    prompt += "\nPrevious failed attempts for this same task (debug prompt feedback; hidden tests remain hidden):\n"
                    for failure in previous_failures[-3:]:
                        prompt += (
                            f"- Attempt {failure.get('attempt')}: "
                            f"error_type={failure.get('error_type')}; "
                            f"error={str(failure.get('error', ''))[:500]}\n"
                        )
                    prompt += "Use this feedback to change the implementation, not to repeat the same pattern. Return executable code only.\n"
                if args.retry_prefix and args.retry_until_correct and attempt > 1:
                    # Debug-only. By default, retry attempts use the same task prompt so
                    # behavioral change must be mediated by Yggdrasil context.
                    prompt = f"Attempt {attempt} for {task_id}.\n\n" + prompt

                print(f"\n--- Attempt {attempt}/{max_attempts} ---")

                record: dict[str, Any] = {
                    "run_id": run_id,
                    "run_index": i,
                    "task_id": task_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "retry_until_correct": bool(args.retry_until_correct),
                    "complexity_bucket": str(task.get("complexity_bucket")),
                    "estimated_complexity": task.get("estimated_complexity"),
                    "prompt_mode": args.prompt_mode,
                    "started_utc": utc_stamp(),
                }

                append_transcript(transcript_path, f"\n### Attempt {attempt}/{max_attempts}\n\n")
                append_transcript(transcript_path, "#### Prompt\n\n```text\n" + prompt + "\n```\n\n")
                if args.print_prompt:
                    print("\n[INJECTED PROMPT]\n" + prompt + "\n[/INJECTED PROMPT]")
                if args.show_canonical:
                    append_transcript(transcript_path, "#### Canonical solution [not sent to model]\n\n```python\n" + str(task.get("canonical_solution", "")) + "\n```\n\n")

                try:
                    lifecycle_cursor = session_api.log_cursor()
                    gate_decision = None
                    deliberation_trace = {"used": False}
                    forced_response = None
                    forced_correctness = None
                    context_enabled_for_attempt = True
                    pre_answer_enabled_for_attempt = True
                    context_mode_for_attempt = "full"
                    task_analysis_context = ""
                    task_analysis_decision = None
                    information_retrieval_context = ""
                    information_retrieval_result = None
                    orchestration_decision = None
                    if args.uncertainty_gate and not bool(args.ablate_governance):
                        gate_decision = decide_uncertainty_gate(
                            prompt,
                            previous_failures=previous_failures,
                            model_probe=(lambda probe_prompt: session_api.raw_generate(probe_prompt, context_text="")),
                            enable_model_probe=bool(args.uncertainty_model_probe),
                            high_confidence_threshold=float(args.gate_high_confidence),
                            low_confidence_threshold=float(args.gate_low_confidence),
                            deliberation_threshold=float(args.gate_deliberation_threshold),
                            max_deliberation_rounds=int(args.deliberation_rounds),
                        )
                        record["uncertainty_gate"] = gate_decision.to_dict()
                        record["retrieval_budget"] = dict(getattr(gate_decision, "retrieval_budget", {}) or {})
                        context_enabled_for_attempt = bool(gate_decision.context_enabled)
                        pre_answer_enabled_for_attempt = bool(gate_decision.pre_answer_enabled)
                        context_mode_for_attempt = {"route": str(gate_decision.route), "retrieval_budget": dict(getattr(gate_decision, "retrieval_budget", {}) or {})}
                        append_transcript(transcript_path, "#### Uncertainty gate\n\n```json\n" + json.dumps(gate_decision.to_dict(), indent=2, ensure_ascii=False) + "\n```\n\n")
                        if bool(args.task_analysis_tool):
                            task_analysis_decision = analyze_task(
                                prompt,
                                previous_failures=previous_failures,
                                gate_route=str(gate_decision.route),
                                model_generate=(lambda planning_prompt: session_api.raw_generate(planning_prompt, context_text="")),
                                enable_model_plan=bool(args.task_analysis_model_plan),
                            )
                            record["task_analysis"] = task_analysis_decision.to_dict()
                            if bool(args.orchestration_layer):
                                orchestration_decision = decide_orchestration(
                                    gate=record["uncertainty_gate"],
                                    task_analysis=record["task_analysis"],
                                    previous_failures=previous_failures,
                                    information_retrieval_available=bool(args.information_retrieval_tool and knowledge_corpus.docs),
                                    environment_available=bool(args.auto_evaluate),
                                )
                                record["orchestration"] = orchestration_decision.to_dict()
                                gate_decision.context_enabled = orchestration_decision.context_enabled
                                gate_decision.pre_answer_enabled = orchestration_decision.pre_answer_enabled
                                gate_decision.deliberation_enabled = orchestration_decision.deliberation_enabled
                                gate_decision.retrieval_budget = dict(orchestration_decision.retrieval_budget)
                                gate_decision.suggested_temperature = orchestration_decision.suggested_temperature
                                record["retrieval_budget"] = dict(orchestration_decision.retrieval_budget)
                                context_enabled_for_attempt = orchestration_decision.context_enabled
                                pre_answer_enabled_for_attempt = orchestration_decision.pre_answer_enabled
                                context_mode_for_attempt = {"route": orchestration_decision.primary_route, "retrieval_budget": dict(orchestration_decision.retrieval_budget)}
                                append_transcript(transcript_path, "#### Orchestration decision\n\n```json\n" + json.dumps(record["orchestration"], indent=2, ensure_ascii=False) + "\n```\n\n")
                                print("[ORCHESTRATION] " + f"route={orchestration_decision.primary_route} | confidence={float(record['uncertainty_gate'].get('confidence', 0.0)):.2f} | operations={','.join(orchestration_decision.operations)} | budget=" + "/".join(f"{k[:1]}{v}" for k,v in orchestration_decision.retrieval_budget.items() if int(v or 0)>0) + f" | temperature={float(orchestration_decision.suggested_temperature):.2f}")

                            use_plan = bool(task_analysis_decision.invoked)
                            if orchestration_decision is not None:
                                use_plan = bool(orchestration_decision.use_task_decomposition)
                            task_analysis_context = format_task_analysis_context(task_analysis_decision) if use_plan else ""
                            append_transcript(transcript_path, "#### Task analysis tool\n\n```json\n" + json.dumps(record["task_analysis"], indent=2, ensure_ascii=False) + "\n```\n\n")
                            ta = record["task_analysis"]
                            plan = ta.get("plan") or {}
                            print("[TASK ANALYSIS] " + f"difficulty={ta.get('difficulty_type')} | goal={str(plan.get('goal') or '')[:180]} | subtasks={len(plan.get('subtasks') or [])} | checks={len(plan.get('checks') or [])} | next={str(plan.get('recommended_operation') or '')[:180]}")
                            use_ir = bool(args.information_retrieval_tool)
                            if orchestration_decision is not None:
                                use_ir = bool(orchestration_decision.use_information_retrieval)
                            if use_ir:
                                information_retrieval_result = retrieve_information(
                                    prompt,
                                    task_analysis=record["task_analysis"],
                                    corpus=knowledge_corpus,
                                    max_hits=int(args.information_retrieval_k),
                                )
                                record["information_retrieval"] = information_retrieval_result.to_dict()
                                information_retrieval_context = format_information_retrieval_context(information_retrieval_result)
                                append_transcript(transcript_path, "#### Information retrieval tool\n\n```json\n" + json.dumps(record["information_retrieval"], indent=2, ensure_ascii=False) + "\n```\n\n")
                                if information_retrieval_result.invoked:
                                    print("[INFORMATION RETRIEVAL] " + json.dumps(record["information_retrieval"], ensure_ascii=False))

                        if bool(gate_decision.deliberation_enabled) and int(args.deliberation_rounds) > 0 and args.auto_evaluate:
                            deliberation_trace = run_private_deliberation(
                                session_api=session_api,
                                task=task,
                                prompt=prompt,
                                gate=gate_decision,
                                eval_timeout=float(args.eval_timeout),
                                diagnostic_cases=bool(args.diagnostic_failing_case),
                            )
                            forced_response = str(deliberation_trace.get("final_response") or "")
                            forced_correctness = deliberation_trace.get("final_correctness")
                            record["deliberation_trace"] = {k: v for k, v in deliberation_trace.items() if k != "final_response"}
                            append_transcript(transcript_path, "#### Private deliberation trace\n\n```json\n" + json.dumps(record["deliberation_trace"], indent=2, ensure_ascii=False) + "\n```\n\n")
                    else:
                        record["uncertainty_gate"] = {"active": False, "reason": "disabled_or_ablation"}

                    result = session_api.run_prompt(
                        prompt,
                        context_enabled=context_enabled_for_attempt,
                        pre_answer_enabled=pre_answer_enabled_for_attempt,
                        forced_response=forced_response,
                        context_mode=context_mode_for_attempt,
                        extra_context_text="\n\n".join(x for x in (task_analysis_context, information_retrieval_context) if x),
                    )
                    response = result.get("response", "")
                    normalized_response, response_format_diagnostic = normalize_code_only_response(response)
                    if response_format_diagnostic.get("format_violation"):
                        record["response_format_diagnostic"] = response_format_diagnostic
                        response_for_evaluation = normalized_response
                    else:
                        record["response_format_diagnostic"] = response_format_diagnostic
                        response_for_evaluation = response
                    context_payload = result.get("context") or {}
                    selected_candidates = list(context_payload.get("selected_candidates") or [])
                    selected_proposals = list(context_payload.get("selected_governance_proposals") or [])
                    selected_hypotheses = list(context_payload.get("selected_developmental_hypotheses") or [])
                    selected_investigations = list(context_payload.get("selected_investigations") or [])
                    pre_answer_investigation = context_payload.get("pre_answer_investigation")
                    pre_action_paradigm_selection = context_payload.get("reasoning_paradigm_selection") or {}
                    pre_action_paradigm_stack = _extract_selected_paradigm_ids(pre_action_paradigm_selection)
                    record["pre_answer_investigation"] = pre_answer_investigation
                    record["pre_action_paradigm_selection"] = pre_action_paradigm_selection
                    record["pre_action_paradigm_stack"] = pre_action_paradigm_stack
                    if args.print_paradigms and pre_action_paradigm_stack:
                        print("\n[PRE-ACTION PARADIGM STACK]")
                        print(json.dumps({"selected": pre_action_paradigm_stack[:1]}, ensure_ascii=False, indent=2))
                        print("[/PRE-ACTION PARADIGM STACK]")
                    if pre_action_paradigm_selection:
                        append_transcript(transcript_path, "#### Pre-action reasoning paradigm selection\n\n```json\n" + json.dumps(pre_action_paradigm_selection, indent=2, ensure_ascii=False) + "\n```\n\n")
                    if args.print_preanswer and pre_answer_investigation:
                        print("\n[PRE-ANSWER INVESTIGATION]")
                        print(json.dumps({
                            "task_id": pre_answer_investigation.get("task_id"),
                            "trigger_reasons": pre_answer_investigation.get("trigger_reasons", [])[:4],
                            "contract_notes": pre_answer_investigation.get("contract_notes", [])[:3],
                            "next_operations": pre_answer_investigation.get("application_directives", [])[:3],
                            "confidence": pre_answer_investigation.get("confidence"),
                        }, ensure_ascii=False, indent=2))
                        print("[/PRE-ANSWER INVESTIGATION]")
                    if pre_answer_investigation:
                        append_transcript(transcript_path, "#### Pre-answer investigation\n\n```json\n" + json.dumps(pre_answer_investigation, indent=2, ensure_ascii=False) + "\n```\n\n")
                    record["selected_candidate_ids"] = [c.get("candidate_id") for c in selected_candidates]
                    record["selected_investigation_ids"] = [inv.get("investigation_id") for inv in selected_investigations]
                    record["selected_candidates"] = selected_candidates
                    record["selected_investigations"] = selected_investigations
                    record["selected_governance_proposals"] = selected_proposals
                    record["selected_developmental_hypotheses"] = selected_hypotheses
                    record["selected_developmental_hypothesis_ids"] = [h.get("hypothesis_id") for h in selected_hypotheses]
                    record["selected_governance_proposal_ids"] = [p.get("proposal_id") for p in selected_proposals]
                    if args.print_candidate_usage:
                        print("\n[CANDIDATE CONTEXT USED]")
                        if selected_candidates:
                            for c in selected_candidates:
                                print(
                                    f"- {c.get('candidate_id')} | score={float(c.get('score') or 0.0):.2f} "
                                    f"| locality={c.get('locality')} | action={c.get('integration_action')} "
                                    f"| status={c.get('status')} | text={str(c.get('text', ''))[:220]}"
                                )
                        else:
                            print("- none")
                        if selected_investigations:
                            print("[INVESTIGATION CONTEXT USED]")
                            for inv in selected_investigations:
                                print(
                                    f"- {inv.get('investigation_id')} | task={inv.get('task_id', '')} "
                                    f"| attempt={inv.get('attempt_number', 0)} | chain={inv.get('investigation_chain_id', '')} "
                                    f"| error={inv.get('error_type')} | score={float(inv.get('score') or 0.0):.2f} "
                                    f"| repair={'; '.join(map(str, inv.get('repair_directions') or []))[:220]}"
                                )
                        if selected_hypotheses:
                            print("[DEVELOPMENTAL HYPOTHESES IN CONTEXT]")
                            for h in selected_hypotheses:
                                print(
                                    f"- {h.get('hypothesis_id')} | validation={h.get('validation_state')} "
                                    f"| signature={h.get('signature')} | lesson={str(h.get('lesson', ''))[:220]}"
                                )
                        if selected_proposals:
                            print("[GOVERNANCE PROPOSALS IN CONTEXT]")
                            for p in selected_proposals:
                                print(
                                    f"- {p.get('proposal_id')} | status={p.get('status')} "
                                    f"| promotion={p.get('promotion_status')} | text={str(p.get('proposal_text', ''))[:220]}"
                                )
                        print("[/CANDIDATE CONTEXT USED]")
                    append_transcript(transcript_path, "#### Retrieved Context Artifacts Used\n\n```json\n" + json.dumps({"candidates": selected_candidates, "investigations": selected_investigations, "developmental_hypotheses": selected_hypotheses, "governance_proposals": selected_proposals}, indent=2, ensure_ascii=False) + "\n```\n\n")
                    record["assistant_event_id"] = result.get("assistant_event_id")
                    record["user_event_id"] = result.get("user_event_id")
                    record["prompt_chars"] = len(prompt)
                    record["response_chars"] = len(response)
                    record["response_preview"] = response[:1000]
                    record["context_preview"] = str(context_payload.get("context_text", ""))[:1500]
                    record["hud_after_response"] = result.get("hud")
                    append_transcript(transcript_path, "#### Response\n\n```text\n" + response + "\n```\n\n")
                    context_text_for_record = str(context_payload.get("context_text", ""))
                    record["context_fingerprint"] = _context_fingerprint(context_text_for_record)
                    record["context_has_investigations"] = bool(selected_investigations) or ("investigation" in context_text_for_record.lower())
                    record["context_has_candidate_cache"] = "Candidate cache memories" in context_text_for_record
                    record["context_has_governance_proposals"] = bool(selected_proposals) or ("governance" in context_text_for_record.lower())
                    record["context_has_developmental_hypotheses"] = bool(selected_hypotheses) or ("developmental hypoth" in context_text_for_record.lower())
                    record["selected_investigations"] = selected_investigations
                    append_transcript(transcript_path, "#### Context Preview\n\n```text\n" + context_text_for_record[:4000] + "\n```\n\n")
                    if args.print_context:
                        print("\n[CONTEXT PREVIEW]")
                        print(context_text_for_record[:3000] if context_text_for_record else "[empty context]")
                        print("[/CONTEXT PREVIEW]")
                    if args.print_response:
                        print("\n[ASSISTANT RESPONSE]")
                        print(response if response else "[empty response]")
                        print("[/ASSISTANT RESPONSE]")

                    correctness: Optional[dict[str, Any]] = None
                    if args.auto_evaluate:
                        if forced_correctness is not None:
                            correctness = dict(forced_correctness)
                            correctness["from_private_deliberation"] = True
                        else:
                            correctness = evaluate_humaneval_response(task, response_for_evaluation, timeout_s=args.eval_timeout, diagnostic_cases=bool(args.diagnostic_failing_case))
                        record["correctness"] = correctness
                        write_json(correctness_dir / f"task_{i:04d}_{task_id.replace('/', '_')}_attempt_{attempt:02d}.json", correctness)
                        append_transcript(transcript_path, "#### Correctness [local tests, solution hidden]\n\n```json\n" + json.dumps(correctness, indent=2, ensure_ascii=False) + "\n```\n\n")
                        print("[CORRECTNESS] " + ("PASS" if correctness.get("passed") else "FAIL") + (f" | {correctness.get('error_type')}: {correctness.get('error')}" if not correctness.get("passed") else ""))
                        if not correctness.get("passed"):
                            previous_failures.append({
                                "attempt": attempt,
                                "error_type": correctness.get("error_type", ""),
                                "error": correctness.get("error", ""),
                            })

                    if correctness is not None and (bool(args.record_objective_feedback) or bool(args.inject_attempt_feedback)):
                        feedback = build_environment_feedback(
                            task_id=str(task_id),
                            attempt=int(attempt),
                            correctness=correctness,
                            prompt=prompt,
                            response=response,
                            use_environment_bridge=bool(args.use_environment_bridge) or (not bool(correctness.get("passed"))),
                            environment_corpus_k=int(args.environment_corpus_k),
                        )
                        obs = session_api.record_observation(feedback, role="system", end_step=False)
                        record["feedback_event_id"] = obs.get("event_id")
                        record["environment_bridge_used"] = bool(args.use_environment_bridge) or (not bool(correctness.get("passed")))
                        append_transcript(transcript_path, "#### Objective Outcome Observation Event\n\n```text\n" + feedback + "\n```\n\n")

                    if correctness is not None:
                        utility_result = session_api.engine.assess_attempt_artifact_utility(
                            task_id=str(task_id), prompt_text=prompt, passed=bool(correctness.get("passed")),
                            selected_candidate_ids=record.get("selected_candidate_ids") or [],
                            selected_investigation_ids=[x.get("investigation_id") for x in (record.get("selected_investigations") or []) if x.get("investigation_id")],
                        )
                        record["regression_aware_utility"] = utility_result
                        if correctness.get("passed"):
                            anchor_result = session_api.engine.record_successful_action_anchor(
                                task_id=str(task_id), prompt_text=prompt, action_text=response_for_evaluation,
                                selected_candidate_ids=record.get("selected_candidate_ids") or [],
                                selected_investigation_ids=[x.get("investigation_id") for x in (record.get("selected_investigations") or []) if x.get("investigation_id")],
                            )
                            record["successful_action_anchor"] = anchor_result
                            print("[SUCCESS ANCHOR] " + json.dumps(anchor_result, ensure_ascii=False))
                        if utility_result.get("regression"):
                            print("[REGRESSION AWARE UTILITY] " + json.dumps(utility_result.get("regression"), ensure_ascii=False))
                        append_transcript(transcript_path, "#### Success anchor / regression-aware utility\n\n```json\n" + json.dumps({"utility": utility_result, "anchor": record.get("successful_action_anchor")}, indent=2, ensure_ascii=False) + "\n```\n\n")

                    rating: Optional[int]
                    if args.interactive_rating:
                        raw = input("Rating to send for this attempt 0-9, blank to skip :rate: ").strip()
                        rating = maybe_int_rating(raw)
                    elif correctness is not None:
                        rating = int(args.correct_rating) if correctness.get("passed") else int(args.incorrect_rating)
                    else:
                        rating = maybe_int_rating(args.default_rating)

                    record["rating"] = rating
                    record["rating_source"] = "interactive" if args.interactive_rating else ("auto_correctness" if correctness is not None else "default")
                    if rating is not None:
                        tags = ["humaneval", "mechanism_validation", f"task_{i}", f"attempt_{attempt}"]
                        if correctness is not None:
                            tags.append("correct" if correctness.get("passed") else "incorrect")
                            if correctness.get("error_type"):
                                tags.append("eval_error_" + str(correctness.get("error_type")).lower())
                        rate_result = session_api.rate(rating, tags=tags)
                        record["rate_result"] = rate_result

                    diag = session_api.diagnostics(recent=args.diag_n)
                    trace = session_api.trace(args.trace_n)
                    record["diagnostic_signal"] = compact_diag_signal(diag)

                    lifecycle_events = session_api.lifecycle_events_since(lifecycle_cursor, max_events=300)
                    lifecycle_signal = compact_lifecycle_signal(lifecycle_events)
                    record["lifecycle_signal"] = lifecycle_signal
                    paradigm_selection_events = _extract_paradigm_selection_events(lifecycle_events)
                    record["reasoning_paradigm_selection_events"] = paradigm_selection_events
                    post_action_events = [e for e in paradigm_selection_events if e.get("phase") == "post_action"]
                    record["post_action_paradigm_selection"] = (post_action_events[-1].get("selection") if post_action_events else {})
                    record["post_action_paradigm_stack"] = (post_action_events[-1].get("selected_reasoning_paradigm_ids") if post_action_events else [])
                    if args.print_paradigms and any(e.get("selected_reasoning_paradigm_ids") for e in paradigm_selection_events):
                        print("\n[REASONING PARADIGM SELECTION EVENTS]")
                        print(json.dumps(paradigm_selection_events, ensure_ascii=False, indent=2))
                        print("[/REASONING PARADIGM SELECTION EVENTS]")
                    new_investigations = [e for e in lifecycle_events if e.get("type") == "investigation_workspace_created" and any(e.get(k) is not None for k in ("investigation_id","error_type","observed_error","task_id"))]
                    new_pre_answer_investigations = [e for e in lifecycle_events if e.get("type") == "pre_answer_investigation_created"]
                    new_reflections = [e for e in lifecycle_events if e.get("type") == "provisional_outcome_written" and any(e.get(k) is not None for k in ("outcome_id","reflection_mode","investigation_id","summary"))]
                    record["new_pre_answer_investigations"] = new_pre_answer_investigations
                    record["new_investigations"] = new_investigations
                    record["new_reflections"] = new_reflections
                    if args.print_investigations and new_investigations:
                        print("\n[NEW INVESTIGATIONS]")
                        for inv in new_investigations:
                            print(json.dumps({
                                "investigation_id": inv.get("investigation_id"),
                                "error_type": inv.get("error_type"),
                                "observed_error": inv.get("observed_error"),
                                "task_id": inv.get("task_id"),
                                "attempt_number": inv.get("attempt_number"),
                                "investigation_chain_id": inv.get("investigation_chain_id"),
                                "prior_investigation_ids": inv.get("prior_investigation_ids"),
                                "unresolved_hypotheses": inv.get("unresolved_hypotheses"),
                                "repeated_failure_signature": inv.get("repeated_failure_signature"),
                                "hypotheses": inv.get("hypotheses"),
                                "repair_directions": inv.get("repair_directions"),
                                "information_needs": inv.get("information_needs"),
                            }, ensure_ascii=False, indent=2))
                        print("[/NEW INVESTIGATIONS]")
                    if args.print_reflections and new_reflections:
                        print("\n[NEW REFLECTIONS]")
                        for ref in new_reflections:
                            print(json.dumps({
                                "outcome_id": ref.get("outcome_id"),
                                "reflection_mode": ref.get("reflection_mode"),
                                "investigation_id": ref.get("investigation_id"),
                                "summary": ref.get("summary"),
                            }, ensure_ascii=False, indent=2))
                        print("[/NEW REFLECTIONS]")
                    write_json(lifecycle_dir / f"task_{i:04d}_{task_id.replace('/', '_')}_attempt_{attempt:02d}.json", lifecycle_events)

                    new_candidates = int(lifecycle_signal.get("candidates_cached", 0) or 0)
                    candidates_since_collation += new_candidates

                    write_json(diagnostics_dir / f"task_{i:04d}_{task_id.replace('/', '_')}_attempt_{attempt:02d}.json", diag)
                    write_json(traces_dir / f"task_{i:04d}_{task_id.replace('/', '_')}_attempt_{attempt:02d}.json", trace)
                    append_transcript(transcript_path, "#### Lifecycle signal\n\n```json\n" + json.dumps(lifecycle_signal, indent=2, ensure_ascii=False) + "\n```\n\n")
                    append_transcript(transcript_path, "#### Diagnostic signal\n\n```json\n" + json.dumps(record["diagnostic_signal"], indent=2, ensure_ascii=False) + "\n```\n\n")

                    recurrent_episode = None
                    if correctness is not None and not correctness.get("passed") and attempt >= 2:
                        failure_sequence = list(previous_failures[-min(4, len(previous_failures)):])
                        if len(failure_sequence) >= 2:
                            candidate_ids = []
                            investigation_ids = []
                            for prior in task_attempt_records[-2:]:
                                candidate_ids.extend(prior.get("selected_candidate_ids") or [])
                                investigation_ids.extend(prior.get("selected_investigation_ids") or [])
                            candidate_ids.extend(record.get("selected_candidate_ids") or [])
                            investigation_ids.extend(record.get("selected_investigation_ids") or [])
                            recurrent_episode = session_api.engine.record_recurrent_failure_episode(
                                task_id=str(task_id), attempts=failure_sequence, prompt_text=prompt,
                                selected_candidate_ids=candidate_ids, selected_investigation_ids=investigation_ids,
                            )
                            record["recurrent_episode_collation"] = recurrent_episode
                            append_transcript(transcript_path, "#### Recurrent failure collation\n\n```json\n" + json.dumps(recurrent_episode, indent=2, ensure_ascii=False) + "\n```\n\n")
                            print("[RECURRENT FAILURE COLLATION] " + json.dumps(recurrent_episode, ensure_ascii=False))

                    should_collate = False
                    collation_reason = None
                    if not args.no_collate:
                        if recurrent_episode and recurrent_episode.get("created"):
                            should_collate = True
                            collation_reason = "recurrent_failure_meso_collation"
                        elif int(args.collate_every_candidates) > 0:
                            should_collate = candidates_since_collation >= int(args.collate_every_candidates)
                            collation_reason = f"candidate_threshold_{args.collate_every_candidates}"
                        else:
                            should_collate = True
                            collation_reason = "humaneval_runner"

                    if should_collate:
                        collation = session_api.collate(max_items=args.collate_n, reason=collation_reason or "humaneval_runner")
                        candidates_since_collation = 0
                        # Recapture lifecycle after collation so completed collation, synthesized
                        # provisional governance, and promotion-evaluation events are reflected
                        # in this attempt's record/report.
                        lifecycle_events = session_api.lifecycle_events_since(lifecycle_cursor, max_events=500)
                        lifecycle_signal = compact_lifecycle_signal(lifecycle_events)
                        record["lifecycle_signal"] = lifecycle_signal
                        record["lifecycle_events_after_collation"] = lifecycle_events
                        paradigm_selection_events = _extract_paradigm_selection_events(lifecycle_events)
                        record["reasoning_paradigm_selection_events"] = paradigm_selection_events
                        post_action_events = [e for e in paradigm_selection_events if e.get("phase") == "post_action"]
                        record["post_action_paradigm_selection"] = (post_action_events[-1].get("selection") if post_action_events else record.get("post_action_paradigm_selection", {}))
                        record["post_action_paradigm_stack"] = (post_action_events[-1].get("selected_reasoning_paradigm_ids") if post_action_events else record.get("post_action_paradigm_stack", []))
                        write_json(lifecycle_dir / f"task_{i:04d}_{task_id.replace('/', '_')}_attempt_{attempt:02d}.json", lifecycle_events)
                        diag_after = session_api.diagnostics(recent=args.diag_n)
                        record["collate_ran"] = True
                        record["collation"] = collation
                        record["diagnostic_signal_after_collation"] = compact_diag_signal(diag_after)
                        write_json(diagnostics_dir / f"task_{i:04d}_{task_id.replace('/', '_')}_attempt_{attempt:02d}_after_collation.json", diag_after)
                        append_transcript(transcript_path, "#### Collation\n\n```json\n" + json.dumps(collation, indent=2, ensure_ascii=False) + "\n```\n\n")
                        append_transcript(transcript_path, "#### Lifecycle signal after collation\n\n```json\n" + json.dumps(lifecycle_signal, indent=2, ensure_ascii=False) + "\n```\n\n")
                    else:
                        record["collate_ran"] = False
                        record["candidates_since_collation"] = candidates_since_collation

                    if args.pause:
                        record["notes"] = input("Notes / observed lifecycle behavior, blank if none: ").strip()
                    else:
                        record["notes"] = ""

                    previous_attempt_record = task_attempt_records[-1] if task_attempt_records else None
                    developmental_delta = compute_developmental_delta(record, previous_attempt_record)
                    record["developmental_delta"] = developmental_delta
                    if args.print_developmental_trace:
                        print("\n[DEVELOPMENTAL TRACE]")
                        print(json.dumps({
                            "task_id": task_id,
                            "attempt": attempt,
                            "pre_action_paradigm_stack": record.get("pre_action_paradigm_stack"),
                            "post_action_paradigm_stack": record.get("post_action_paradigm_stack"),
                            "selected_candidate_ids": record.get("selected_candidate_ids"),
                            "selected_investigation_ids": record.get("selected_investigation_ids"),
                            "selected_governance_proposal_ids": record.get("selected_governance_proposal_ids"),
                            "lifecycle_signal": record.get("lifecycle_signal"),
                            "developmental_delta": developmental_delta,
                        }, ensure_ascii=False, indent=2))
                        print("[/DEVELOPMENTAL TRACE]")
                        append_transcript(transcript_path, "#### Developmental trace / delta\n\n```json\n" + json.dumps(record.get("developmental_delta"), indent=2, ensure_ascii=False) + "\n```\n\n")

                    record["completed_utc"] = utc_stamp()
                    append_jsonl(records_path, record)
                    task_attempt_records.append(record)
                    print_task_summary(record)
                    print(f"Logged task {task_id} attempt {attempt}")

                    if correctness is not None and correctness.get("passed"):
                        task_passed = True
                        break
                    if not args.retry_until_correct:
                        break

                except Exception as exc:
                    record["error"] = repr(exc)
                    record["traceback"] = traceback.format_exc()
                    record["completed_utc"] = utc_stamp()
                    append_jsonl(records_path, record)
                    task_attempt_records.append(record)
                    append_transcript(transcript_path, "#### ERROR\n\n```text\n" + record["traceback"] + "\n```\n\n")
                    print(f"ERROR on {task_id} attempt {attempt}: {exc!r}")
                    if args.pause:
                        cont = input("Continue after error? [y/N] ").strip().lower()
                        if cont != "y":
                            raise
                    else:
                        raise

            task_summary = {
                "run_id": run_id,
                "task_id": task_id,
                "run_index": i,
                "attempts": len(task_attempt_records),
                "passed": bool(task_passed),
                "stopped_reason": "passed" if task_passed else ("max_attempts" if args.retry_until_correct else "single_pass"),
                "attempt_record_indices": [r.get("attempt") for r in task_attempt_records],
            }
            write_json(run_dir / "task_summaries" / f"task_{i:04d}_{task_id.replace('/', '_')}.json", task_summary)
            append_transcript(transcript_path, "### Task Summary\n\n```json\n" + json.dumps(task_summary, indent=2, ensure_ascii=False) + "\n```\n\n")

    finally:
        session_api.close()

    if not args.no_run_summary:
        all_records = load_existing_records(records_path)
        summary = summarize_run(all_records)
        summary["run_id"] = run_id
        summary["session"] = session
        summary["profile"] = args.profile
        summary["resume_requested"] = bool(resume_requested)
        summary["resume_run"] = args.resume_run
        summary["existing_record_count"] = len(existing_records)
        summary["effective_start"] = effective_start
        summary["session_root"] = str(args.session_root)
        summary["run_dir"] = str(run_dir)
        summary["replay_verification"] = getattr(session_api, "resume_report", None)
        write_json(run_summary_path, summary)
        write_run_report(summary, run_report_path)
        print("\n[RUN SUMMARY]")
        print(
            f"tasks={summary.get('task_count')} "
            f"attempts={summary.get('attempt_count')} "
            f"tasks_passed={summary.get('tasks_passed')} "
            f"task_pass_rate={summary.get('task_pass_rate_percent')}% "
            f"candidate_retrieval_rate={summary.get('retrieval', {}).get('candidate_retrieval_rate_percent')}% "
            f"investigation_retrieval_rate={summary.get('retrieval', {}).get('investigation_retrieval_rate_percent')}%"
        )
        print(f"Run summary: {run_summary_path}")
        print(f"Run report:  {run_report_path}")

    print("\nDone.")
    print(f"Records:    {records_path}")
    print(f"Transcript: {transcript_path}")


if __name__ == "__main__":
    main()

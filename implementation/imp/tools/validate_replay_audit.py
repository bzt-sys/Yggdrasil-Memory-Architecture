from __future__ import annotations

"""Replay fidelity audit for Yggdrasil's generic developmental event log.

The audit is intentionally benchmark-agnostic. It creates a tiny deterministic
experience/action/environment/outcome sequence, verifies that richer
``developmental_event_written`` packets are emitted, rebuilds from the JSONL
causal history, and compares structural counts, state hashes, and retrieval
context summaries.
"""

import argparse
import sys
import json
import shutil
import tempfile
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any, Dict, List, Tuple

from imp.agent_session import YggdrasilAgentSession, AgentSessionConfig
from imp.session_state import SessionStore
from imp.storage import read_jsonl
from imp.substrate_api import SubstrateAPI
from imp.state_hash import compute_state_hash, compute_runtime_state_hash


def _count_rows(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r.get("type", "")] = out.get(r.get("type", ""), 0) + 1
    return out


def _count_nodes(engine: Any) -> Dict[str, int]:
    return {k: len(v) for k, v in sorted(engine.graph.by_type.items())}


def _developmental_kinds(rows: List[Dict[str, Any]]) -> List[str]:
    kinds = []
    for r in rows:
        if r.get("type") == "developmental_event_written":
            k = (r.get("data") or {}).get("kind")
            if k:
                kinds.append(str(k))
    return kinds


def _paradigm_phase_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        if r.get("type") != "developmental_event_written":
            continue
        data = r.get("data") or {}
        if data.get("kind") != "reasoning_paradigm_selection":
            continue
        payload = data.get("payload") or {}
        phase = str(payload.get("phase") or (payload.get("selection") or {}).get("phase") or payload.get("mode") or "unknown")
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def _max_paradigm_stack_size(rows: List[Dict[str, Any]]) -> int:
    max_size = 0
    for r in rows:
        if r.get("type") != "developmental_event_written":
            continue
        data = r.get("data") or {}
        if data.get("kind") != "reasoning_paradigm_selection":
            continue
        payload = data.get("payload") or {}
        selection = payload.get("selection") or {}
        stack = selection.get("ordered_stack_ids") or payload.get("selected_reasoning_paradigm_ids") or []
        max_size = max(max_size, len(stack))
    return max_size


def _context_fingerprint(api: SubstrateAPI, prompt: str) -> Dict[str, Any]:
    ctx = api.get_context_bundle(prompt_text=prompt, max_items=6)
    structured = ctx.get("context_json") or ctx
    return {
        "candidate_ids": [c.get("candidate_id") for c in structured.get("selected_candidates", []) if isinstance(c, dict)],
        "investigation_ids": [i.get("investigation_id") for i in structured.get("selected_investigations", []) if isinstance(i, dict)],
        "proposal_ids": [g.get("proposal_id") for g in structured.get("selected_governance_proposals", []) if isinstance(g, dict)],
        "text_len": len(ctx.get("context_text") or ""),
    }


def _result(name: str, ok: bool, detail: str = "") -> Tuple[str, bool, str]:
    return (name, bool(ok), detail)


def run_audit(root: Path, session_id: str, keep: bool = False) -> int:
    if root.exists() and not keep:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    cfg = AgentSessionConfig(session=session_id, session_root=str(root), actor="none", resume=False)
    session = YggdrasilAgentSession(cfg)

    prompt = "Given a novel task, identify assumptions before committing to an action."
    first = session.run_prompt(prompt, end_step=True)
    assistant_id = first.get("assistant_event_id")

    observation_text = (
        "Environment objective feedback for generic_task attempt 1: "
        "passed=False; signal_type=constraint_violation; error=The committed action skipped explicit assumption checking; "
        "signal_quality=medium; feedback_level=2."
    )
    session.record_observation(
        observation_text,
        role="system",
        end_step=False,
    )
    session.rate(2, tags=["audit", "objective_feedback", "generic_failure"])
    # Add one post-feedback prompt so retrieval/context has something to select.
    session.run_prompt("Repeat the task, but use the prior signal as evidence.", end_step=True)

    store = SessionStore(root=str(root))
    store.write_snapshot(session_id, session.engine)
    report = store.verify_session(session_id, enable_decay=session.engine.enable_decay)
    rebuilt = store.rebuild_engine(session_id, enable_decay=session.engine.enable_decay)

    rows = read_jsonl(str(store.paths(session_id).events_jsonl))
    row_counts = _count_rows(rows)
    kinds = _developmental_kinds(rows)
    phase_counts = _paradigm_phase_counts(rows)
    max_stack_size = _max_paradigm_stack_size(rows)
    original_hash, original_summary = compute_state_hash(session.engine.graph)
    rebuilt_hash, rebuilt_summary = compute_state_hash(rebuilt.graph)
    original_counts = _count_nodes(session.engine)
    rebuilt_counts = _count_nodes(rebuilt)
    original_ctx = _context_fingerprint(session.api, prompt)
    rebuilt_api = SubstrateAPI(rebuilt)
    rebuilt_ctx = _context_fingerprint(rebuilt_api, prompt)
    original_runtime_hash, original_runtime = compute_runtime_state_hash(session.engine)
    rebuilt_runtime_hash, rebuilt_runtime = compute_runtime_state_hash(rebuilt)

    # Continuation equivalence: apply the same next causal event to continuous
    # and reconstructed engines, then require equal state and retrieval.
    continuation_text = "A later related experience asks the agent to make its assumptions explicit."
    session.api.record_event(role="user", text=continuation_text, source="replay_audit_continuation", provenance={"audit": True})
    session.api.end_step()
    rebuilt_api.record_event(role="user", text=continuation_text, source="replay_audit_continuation", provenance={"audit": True})
    rebuilt_api.end_step()
    continuous_hash, _ = compute_state_hash(session.engine.graph)
    continued_replay_hash, _ = compute_state_hash(rebuilt.graph)
    continuous_ctx = _context_fingerprint(session.api, continuation_text)
    continued_replay_ctx = _context_fingerprint(rebuilt_api, continuation_text)

    checks = [
        _result("event rows written", row_counts.get("event_written", 0) >= 4, str(row_counts.get("event_written", 0))),
        _result("developmental packets written", row_counts.get("developmental_event_written", 0) >= 4, str(row_counts.get("developmental_event_written", 0))),
        _result("prompt packet present", "experience_prompt" in kinds, ",".join(kinds)),
        _result("action packet present", "committed_action" in kinds, ",".join(kinds)),
        _result("environment packet present", "environment_observation" in kinds, ",".join(kinds)),
        _result("outcome packet present", "outcome_signal" in kinds, ",".join(kinds)),
        _result("pre-action paradigm injection disabled", phase_counts.get("pre_action", 0) == 0, json.dumps(phase_counts, sort_keys=True)),
        _result("post-action paradigm injection disabled", phase_counts.get("post_action", 0) == 0, json.dumps(phase_counts, sort_keys=True)),
        _result("no diagnostic paradigm stack injected", max_stack_size == 0, str(max_stack_size)),
        _result("action rationale packet restored", original_counts.get("action_rationale", 0) == rebuilt_counts.get("action_rationale", 0) and original_counts.get("action_rationale", 0) >= 1, f"original={original_counts.get('action_rationale', 0)} rebuilt={rebuilt_counts.get('action_rationale', 0)}"),
        _result("state hash replay verifier ok", bool(report.get("ok")), json.dumps(report.get("first_mismatch"), ensure_ascii=False)),
        _result("final state hash restored", original_hash == rebuilt_hash, f"original={original_hash} rebuilt={rebuilt_hash}"),
        _result("node counts restored", original_counts == rebuilt_counts, f"original={original_counts} rebuilt={rebuilt_counts}"),
        _result("retrieval fingerprint stable", original_ctx == rebuilt_ctx, f"original={original_ctx} rebuilt={rebuilt_ctx}"),
        _result("runtime indexes restored", original_runtime_hash == rebuilt_runtime_hash, f"original={original_runtime} rebuilt={rebuilt_runtime}"),
        _result("continuation state equivalent", continuous_hash == continued_replay_hash, f"continuous={continuous_hash} replayed={continued_replay_hash}"),
        _result("continuation retrieval equivalent", continuous_ctx == continued_replay_ctx, f"continuous={continuous_ctx} replayed={continued_replay_ctx}"),
    ]

    print("Replay audit session:", session_id)
    print("Session root:", root)
    print("Row counts:", json.dumps(row_counts, sort_keys=True))
    print("Developmental kinds:", ", ".join(kinds))
    print("Paradigm phase counts:", json.dumps(phase_counts, sort_keys=True))
    print("Max injected paradigm count:", max_stack_size)
    print("Original summary:", json.dumps(original_summary, sort_keys=True))
    print("Rebuilt summary:", json.dumps(rebuilt_summary, sort_keys=True))
    print("")

    failed = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" :: {detail}" if detail else ""))
        if not ok:
            failed += 1

    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Yggdrasil generic developmental replay fidelity.")
    parser.add_argument("--session", default="replay_audit_dev_events", help="Session id to create/use.")
    parser.add_argument("--root", default="", help="Session root. Defaults to a temporary directory.")
    parser.add_argument("--keep", action="store_true", help="Keep an existing root/session instead of deleting it first.")
    args = parser.parse_args()

    if args.root:
        return run_audit(Path(args.root), args.session, keep=args.keep)
    with tempfile.TemporaryDirectory(prefix="ygg_replay_audit_") as tmp:
        return run_audit(Path(tmp), args.session, keep=True)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

"""
Deterministic replay verifier for Yggdrasil JSONL logs.

This tool rebuilds engine state from the recorded causal inputs
(events and outcomes) and compares the resulting state-hash sequence
against the hashes captured during the original run.

The verifier intentionally ignores derivative log rows such as
extraction summaries or episode bookkeeping, since those should emerge
from replay rather than be injected directly.
"""

import argparse
import json
from typing import Any, Dict, List

from .memory import HybridAdjudicator, YggdrasilEngine
from .storage import JsonlLogger
from .substrate_api import SubstrateAPI


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL log file into a list of row dictionaries.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    """
    Replay a recorded run and compare state hashes step-by-step.
    """
    ap = argparse.ArgumentParser(
        description="Replay a Yggdrasil JSONL log and verify deterministic state-hash parity."
    )
    ap.add_argument(
        "--log",
        required=True,
        help="Path to the source JSONL log (for example: logs/demo.jsonl).",
    )
    ap.add_argument(
        "--session",
        default="replay-session",
        help="Session id to use for the replay run.",
    )
    args = ap.parse_args()

    rows = _read_jsonl(args.log)

    # Fresh engine with fresh graph; drive only via the stable API surface
    # so replay behavior depends on the same causal operations as the original run.
    logger = JsonlLogger("logs/replay.jsonl")
    adjudicator = HybridAdjudicator()
    eng = YggdrasilEngine(
        session_id=args.session,
        logger=logger,
        adjudicator=adjudicator,
    )
    api = SubstrateAPI(eng)

    expected_hashes: Dict[int, str] = {}
    for r in rows:
        t = r.get("type")
        d = r.get("data", {})
        if t == "state_hash":
            expected_hashes[int(d["step"])] = str(d["hash"])

    replay_hashes: Dict[int, str] = {}

    for r in rows:
        t = r.get("type")
        d = r.get("data", {})

        if t == "event_written":
            # Re-apply the recorded causal input. UUIDs will differ, but the
            # replay state hash is intentionally independent of runtime UUIDs.
            eng.start_step()
            api.record_event(
                role=d.get("role", "user"),
                text=d.get("text", ""),
                source=d.get("source", "replay"),
            )
            api.end_step()

        elif t == "outcome_written":
            eng.start_step()
            api.log_outcome(
                score=float(d.get("score", 0.0)),
                label=str(d.get("label", "") or ""),
                tags=d.get("tags", []),
            )
            api.end_step()

        # Ignore derivative rows such as extraction summaries, episode open/close
        # events, and other secondary bookkeeping. These should emerge from replay.

        sh = eng._last_step_metrics.get("state_hash")
        if sh:
            replay_hashes[int(eng.step)] = sh

    steps = sorted(set(expected_hashes.keys()) | set(replay_hashes.keys()))
    ok = True

    for s in steps:
        expected = expected_hashes.get(s)
        replayed = replay_hashes.get(s)
        if expected != replayed:
            ok = False
            print(f"[MISMATCH] step={s} expected={expected} replay={replayed}")
            break

    if ok:
        print(f"[PASS] replay hash sequence identical for {len(expected_hashes)} steps.")
    else:
        print("[FAIL] replay hash mismatch. See first mismatch above.")


if __name__ == "__main__":
    main()
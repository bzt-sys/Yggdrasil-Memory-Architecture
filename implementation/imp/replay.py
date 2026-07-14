from __future__ import annotations

"""Deterministic replay verifier for Yggdrasil JSONL logs."""

import argparse
import os

from .memory import HybridAdjudicator
from .session_state import SessionStore, format_replay_report
from .evidence import export_evidence_bundle


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay a Yggdrasil JSONL log and verify deterministic state-hash parity."
    )
    ap.add_argument("--log", required=True, help="Path to source JSONL log.")
    ap.add_argument("--session", default="replay-session", help="Session id used for report paths.")
    ap.add_argument("--session-root", default="sessions", help="Directory for session artifacts.")
    ap.add_argument("--report-json", default="", help="Optional explicit JSON report path.")
    ap.add_argument("--report-md", default="", help="Optional explicit Markdown report path.")
    ap.add_argument("--export-evidence", default="", help="Optional directory for graph/trace evidence exports.")
    ap.add_argument("--debug-lexical", action="store_true", help="Enable lexical extractor during replay.")
    ap.add_argument("--no-decay", action="store_true", help="Disable decay/compression/failsafe maintenance during replay verification.")
    args = ap.parse_args()

    store = SessionStore(args.session_root)
    adjudicator = HybridAdjudicator(enable_lexical_default=bool(args.debug_lexical))
    report = store.verify_session(
        args.session,
        adjudicator=adjudicator,
        source_log=args.log,
        write_report=False,
        enable_decay=not bool(args.no_decay),
    )

    print(format_replay_report(report).strip())
    if report.get("ok"):
        print(f"[PASS] replay hash sequence identical for {report.get('expected_steps')} steps.")
    else:
        print("[FAIL] replay hash mismatch. See report above.")

    if args.report_json:
        import json
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    if args.report_md:
        os.makedirs(os.path.dirname(args.report_md) or ".", exist_ok=True)
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write(format_replay_report(report))

    if args.export_evidence:
        eng = store.rebuild_engine(args.session, adjudicator=adjudicator, source_log=args.log, enable_decay=not bool(args.no_decay))
        export_evidence_bundle(eng, args.export_evidence)
        print(f"[EXPORT] evidence written to {args.export_evidence}")


if __name__ == "__main__":
    main()

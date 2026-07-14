from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imp.session_state import SessionStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run detailed causal-boundary replay diagnostics for an existing Yggdrasil session.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--decay", action="store_true", help="Construct replay with decay enabled. Default is disabled for controlled HumanEval sessions.")
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    store = SessionStore(root=args.session_root)
    report = store.verify_session(
        args.session,
        enable_decay=bool(args.decay),
        write_report=not args.no_write_report,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

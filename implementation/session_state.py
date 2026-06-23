from __future__ import annotations

"""
Session persistence, resume, and replay utilities for Yggdrasil.

Canonical source of truth:
    sessions/<session_id>/events.jsonl

Fast/human-readable artifacts:
    sessions/<session_id>/artifacts/latest_snapshot.json
    sessions/<session_id>/artifacts/replay_report.json
    sessions/<session_id>/artifacts/replay_report.md

The JSONL log remains the causal history. Snapshots are convenience artifacts,
not the authoritative replay source.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .memory import HybridAdjudicator, YggdrasilEngine
from .storage import JsonlLogger, NullLogger, read_jsonl, write_json
from .substrate_api import SubstrateAPI
from .state_hash import compute_state_hash, canonical_state_payload


CAUSAL_ROW_TYPES = {"event_written", "outcome_written"}


@dataclass
class SessionPaths:
    root: str
    session_id: str

    @property
    def session_dir(self) -> str:
        return os.path.join(self.root, self.session_id)

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.session_dir, "artifacts")

    @property
    def events_jsonl(self) -> str:
        return os.path.join(self.session_dir, "events.jsonl")

    @property
    def snapshot_json(self) -> str:
        return os.path.join(self.artifacts_dir, "latest_snapshot.json")

    @property
    def replay_report_json(self) -> str:
        return os.path.join(self.artifacts_dir, "replay_report.json")

    @property
    def replay_report_md(self) -> str:
        return os.path.join(self.artifacts_dir, "replay_report.md")

    @property
    def graph_json(self) -> str:
        return os.path.join(self.artifacts_dir, "graph.json")

    @property
    def graph_dot(self) -> str:
        return os.path.join(self.artifacts_dir, "graph.dot")

    def ensure(self) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)


class SessionStore:
    """Manage canonical session paths and reconstruction from event logs."""

    def __init__(self, root: str = "sessions") -> None:
        self.root = root

    def paths(self, session_id: str) -> SessionPaths:
        p = SessionPaths(root=self.root, session_id=session_id)
        p.ensure()
        return p

    def logger(self, session_id: str) -> JsonlLogger:
        return JsonlLogger(self.paths(session_id).events_jsonl)

    def rebuild_engine(
        self,
        session_id: str,
        *,
        adjudicator: Optional[HybridAdjudicator] = None,
        source_log: Optional[str] = None,
        logger: Optional[Any] = None,
        enable_decay: bool = True,
    ) -> YggdrasilEngine:
        """
        Rebuild an engine from a JSONL causal history.

        Reconstruction is grouped by original state_hash rows: causal rows are
        applied until a state_hash row is encountered, then end_step() is called.
        This preserves the original step grouping instead of incrementing once
        per event row.
        """
        p = self.paths(session_id)
        rows = read_jsonl(source_log or p.events_jsonl)
        eng = YggdrasilEngine(
            session_id=session_id,
            logger=logger or NullLogger(),
            adjudicator=adjudicator or HybridAdjudicator(),
            enable_decay=bool(enable_decay),
        )
        api = SubstrateAPI(eng)
        eng.start_step()
        pending_causal = False

        for r in rows:
            typ = r.get("type")
            data = r.get("data", {}) or {}
            if typ == "event_written":
                api.record_event(
                    role=data.get("role", "user"),
                    text=data.get("text", ""),
                    source=data.get("source", "replay"),
                    event_id=data.get("event_id"),
                )
                pending_causal = True
            elif typ == "outcome_written":
                api.log_outcome(
                    score=float(data.get("score", 0.0)),
                    label=str(data.get("label", "") or ""),
                    tags=data.get("tags", []),
                    outcome_id=data.get("outcome_id"),
                )
                pending_causal = True
            elif typ == "state_hash" and pending_causal:
                api.end_step()
                eng.start_step()
                pending_causal = False

        if pending_causal:
            api.end_step()
        return eng

    def resume_engine(
        self,
        session_id: str,
        *,
        adjudicator: Optional[HybridAdjudicator] = None,
        append_log_path: Optional[str] = None,
        verify: bool = True,
        enable_decay: bool = True,
    ) -> YggdrasilEngine:
        """
        Rebuild prior state and attach an append logger so experimentation can continue.
        """
        p = self.paths(session_id)
        report = self.verify_session(session_id, adjudicator=adjudicator, enable_decay=enable_decay) if verify else None
        eng = self.rebuild_engine(session_id, adjudicator=adjudicator, logger=NullLogger(), enable_decay=enable_decay)
        eng.log = JsonlLogger(append_log_path or p.events_jsonl)
        if report:
            eng._last_resume_report = report  # type: ignore[attr-defined]
        return eng

    def verify_session(
        self,
        session_id: str,
        *,
        adjudicator: Optional[HybridAdjudicator] = None,
        source_log: Optional[str] = None,
        write_report: bool = True,
        enable_decay: bool = True,
    ) -> Dict[str, Any]:
        """Replay a session and compare replayed state hashes to recorded hashes."""
        p = self.paths(session_id)
        rows = read_jsonl(source_log or p.events_jsonl)
        expected: Dict[int, str] = {}
        for r in rows:
            if r.get("type") == "state_hash":
                d = r.get("data", {}) or {}
                if "step" in d and "hash" in d:
                    expected[int(d["step"])] = str(d["hash"])

        replay_hashes: Dict[int, str] = {}
        eng = YggdrasilEngine(
            session_id=f"{session_id}-replay",
            logger=NullLogger(),
            adjudicator=adjudicator or HybridAdjudicator(),
            enable_decay=bool(enable_decay),
        )
        api = SubstrateAPI(eng)
        eng.start_step()
        pending_causal = False
        first_mismatch: Optional[Dict[str, Any]] = None

        for r in rows:
            typ = r.get("type")
            d = r.get("data", {}) or {}
            if typ == "event_written":
                api.record_event(
                    role=d.get("role", "user"),
                    text=d.get("text", ""),
                    source=d.get("source", "replay"),
                    event_id=d.get("event_id"),
                )
                pending_causal = True
            elif typ == "outcome_written":
                api.log_outcome(
                    score=float(d.get("score", 0.0)),
                    label=str(d.get("label", "") or ""),
                    tags=d.get("tags", []),
                    outcome_id=d.get("outcome_id"),
                )
                pending_causal = True
            elif typ == "state_hash" and pending_causal:
                api.end_step()
                step = int(eng.step)
                h = eng._last_step_metrics.get("state_hash")
                if h:
                    replay_hashes[step] = str(h)
                exp = expected.get(step)
                if exp != h and first_mismatch is None:
                    first_mismatch = {"step": step, "expected": exp, "replayed": h}
                eng.start_step()
                pending_causal = False

        if pending_causal:
            api.end_step()
            h = eng._last_step_metrics.get("state_hash")
            if h:
                replay_hashes[int(eng.step)] = str(h)

        ok = first_mismatch is None and expected == replay_hashes
        final_hash, final_summary = compute_state_hash(eng.graph)
        report = {
            "ok": bool(ok),
            "session_id": session_id,
            "source_log": source_log or p.events_jsonl,
            "expected_steps": len(expected),
            "replayed_steps": len(replay_hashes),
            "first_mismatch": first_mismatch,
            "final_hash": final_hash,
            "final_summary": final_summary,
        }
        if write_report:
            write_json(p.replay_report_json, report)
            with open(p.replay_report_md, "w", encoding="utf-8") as f:
                f.write(format_replay_report(report))
        return report

    def write_snapshot(self, session_id: str, eng: YggdrasilEngine) -> Dict[str, Any]:
        """Write a convenience snapshot and graph export for human inspection."""
        p = self.paths(session_id)
        h, summary = compute_state_hash(eng.graph)
        payload = {
            "session_id": session_id,
            "step": eng.step,
            "state_hash": h,
            "summary": summary,
            "hud": eng.hud(),
            "canonical_state": canonical_state_payload(eng.graph),
        }
        write_json(p.snapshot_json, payload)
        write_json(p.graph_json, eng.graph.to_canonical_dict())
        with open(p.graph_dot, "w", encoding="utf-8") as f:
            f.write(eng.graph.to_dot())
        return payload


def format_replay_report(report: Dict[str, Any]) -> str:
    status = "PASS" if report.get("ok") else "FAIL"
    lines = [
        f"# Replay Report — {status}",
        "",
        f"session_id: `{report.get('session_id')}`",
        f"source_log: `{report.get('source_log')}`",
        f"expected_steps: {report.get('expected_steps')}",
        f"replayed_steps: {report.get('replayed_steps')}",
        f"final_hash: `{report.get('final_hash')}`",
        f"final_summary: `{report.get('final_summary')}`",
    ]
    if report.get("first_mismatch"):
        lines += ["", "## First mismatch", "", "```json", json.dumps(report["first_mismatch"], indent=2, sort_keys=True), "```"]
    else:
        lines += ["", "Replay hash sequence matched recorded state hashes."]
    return "\n".join(lines) + "\n"

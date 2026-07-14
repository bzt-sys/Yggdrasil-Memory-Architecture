from __future__ import annotations

"""Human-readable evidence export helpers for Yggdrasil."""

import json
import os
from typing import Any, Dict, Optional

from .state_hash import compute_state_hash, canonical_state_payload
from .storage import write_json


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def format_graph_summary(graph) -> str:
    return graph.format_summary()


def format_hud(eng) -> str:
    hud = eng.hud()
    c = hud.get("counts", {})
    last = hud.get("last_step", {})
    lines = [
        "[HUD]",
        f"step={c.get('step')} episodes={c.get('episodes')} open={c.get('episode_open')} events={c.get('events')}",
        f"features={c.get('features')} concepts={c.get('concepts')} outcomes={c.get('outcomes')} detail_edges={c.get('detail_edges')}",
        f"state_hash={last.get('state_hash', '')}",
        "top_concepts:",
    ]
    for k, v in hud.get("top_concepts", []) or []:
        lines.append(f"  - {k} ({float(v):.4f})")
    return "\n".join(lines) + "\n"


def format_trace(eng, n: int = 10) -> str:
    rows = eng.trace_last(n)
    lines = [f"[TRACE] last={len(rows)}"]
    for r in rows:
        text = (r.get("text") or "").replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        lines.append(f"- {str(r.get('event_id'))[:8]} role={r.get('role')} intent={r.get('intent')} :: {text}")
        feats = r.get("features", []) or []
        for f in feats[:6]:
            lines.append(f"    * {f.get('kind')} {f.get('key')} conf={float(f.get('conf', 0)):.2f} sal={float(f.get('sal', 0)):.2f}")
    return "\n".join(lines) + "\n"


def export_evidence_bundle(eng, out_dir: str, trace_n: int = 12) -> Dict[str, str]:
    """
    Export paper-friendly evidence artifacts from a live or reconstructed engine.
    """
    ensure_dir(out_dir)
    state_hash, summary = compute_state_hash(eng.graph)
    manifest = {
        "state_hash": state_hash,
        "summary": summary,
        "files": {},
    }

    paths = {
        "hud": os.path.join(out_dir, "hud.txt"),
        "trace": os.path.join(out_dir, "trace.md"),
        "graph_summary": os.path.join(out_dir, "graph_summary.txt"),
        "graph_json": os.path.join(out_dir, "graph.json"),
        "graph_dot": os.path.join(out_dir, "graph.dot"),
        "canonical_state": os.path.join(out_dir, "canonical_state.json"),
        "manifest": os.path.join(out_dir, "manifest.json"),
    }

    write_text(paths["hud"], format_hud(eng))
    write_text(paths["trace"], format_trace(eng, trace_n))
    write_text(paths["graph_summary"], eng.graph.format_summary() + "\n\n" + eng.graph.format_nodes(20) + "\n\n" + eng.graph.format_edges(20) + "\n")
    write_json(paths["graph_json"], eng.graph.to_canonical_dict())
    write_text(paths["graph_dot"], eng.graph.to_dot())
    write_json(paths["canonical_state"], canonical_state_payload(eng.graph))

    manifest["files"] = {k: v for k, v in paths.items() if k != "manifest"}
    write_json(paths["manifest"], manifest)
    return paths

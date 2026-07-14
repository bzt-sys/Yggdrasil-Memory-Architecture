from __future__ import annotations

"""Generic developmental event helpers for event-sourced replay.

This module deliberately avoids benchmark- or coding-specific schemas.  It
records the richer causal unit that newer Yggdrasil mechanisms operate on:
experience -> committed action -> external/reflective signal -> analysis ->
developmental artifact.  Graph features remain useful derived indexes, but the
causal log should preserve these packets directly.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "developmental_event_v1"


def stable_payload_hash(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


@dataclass
class DevelopmentalEvent:
    """Canonical, domain-general causal packet."""

    event_id: str
    session_id: str
    step: int
    kind: str
    parent_ids: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_log_data(self) -> Dict[str, Any]:
        data = asdict(self)
        data["payload_hash"] = stable_payload_hash(self.payload)
        return data


def emit_developmental_event(
    logger: Any,
    *,
    event_id: str,
    session_id: str,
    step: int,
    kind: str,
    payload: Dict[str, Any],
    parent_ids: Optional[List[str]] = None,
) -> None:
    """Append a developmental event when the active logger supports JSONL logging."""
    if logger is None or not hasattr(logger, "log"):
        return
    packet = DevelopmentalEvent(
        event_id=str(event_id),
        session_id=str(session_id),
        step=int(step),
        kind=str(kind),
        parent_ids=[str(p) for p in (parent_ids or []) if p],
        payload=dict(payload or {}),
    )
    logger.log("developmental_event_written", packet.to_log_data())


def classify_experience_kind(*, role: str, source: str, provenance: Optional[Dict[str, Any]] = None) -> str:
    """Map runtime events into generic causal roles without benchmark coupling."""
    prov = provenance or {}
    explicit = prov.get("developmental_kind") or prov.get("kind")
    if explicit:
        return str(explicit)
    src = (source or "").lower()
    r = (role or "").lower()
    if "environment" in src or "observation" in src:
        return "environment_observation"
    if r == "assistant":
        return "committed_action"
    if r == "system":
        return "system_signal"
    return "experience_prompt"


_SIGNAL_KEYS = (
    "passed", "score", "label", "error_type", "error", "signal_type",
    "signal_quality", "feedback_level", "observed_output", "expected_output",
)


def semantic_profile_from_text(text: str, *, role: str = "", source: str = "", provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compact semantic profile used as index material, not causal ground truth."""
    prov = dict(provenance or {})
    low = (text or "").lower()
    tags: List[str] = []
    if any(cue in low for cue in ("do not", "don't", "avoid", "must not", "never", "always", "prefer")):
        tags.append("constraint_like")
    if "?" in text:
        tags.append("question_like")
    if "environment objective feedback" in low or "objective feedback" in low:
        tags.append("objective_feedback")
    if "uncertain" in low or "unknown" in low or "cannot verify" in low:
        tags.append("uncertainty_explicit")

    extracted: Dict[str, Any] = {}
    for key in _SIGNAL_KEYS:
        m = re.search(rf"\b{re.escape(key)}\s*=\s*([^;\.]+)", text or "", flags=re.IGNORECASE)
        if m:
            extracted[key] = m.group(1).strip()[:240]

    profile = {
        "surface_kind": classify_experience_kind(role=role, source=source, provenance=prov),
        "role": role,
        "source": source,
        "tags": sorted(set(tags + list(prov.get("tags", []) or [])))[:12],
        "signal": extracted,
        "text_length": len(text or ""),
    }
    if prov.get("semantic_profile") and isinstance(prov.get("semantic_profile"), dict):
        merged = dict(profile)
        merged.update(prov["semantic_profile"])
        profile = merged
    return profile

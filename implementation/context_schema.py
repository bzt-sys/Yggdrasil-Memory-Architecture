from __future__ import annotations

"""
Canonical context bundle schema for Yggdrasil.

This module defines the minimal structured context format exchanged between
the substrate and downstream actor surfaces. The schema is intentionally
small, deterministic, and easy to serialize for logging, replay, and audit.
"""

import json
from typing import Any, Dict, List, Optional


def _stable_dumps(obj: Any) -> str:
    """
    Serialize an object to deterministic JSON.

    Output is normalized with sorted keys and compact separators so repeated
    serializations of equivalent objects produce identical text.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_context_bundle(
    *,
    step: int,
    constraints: List[Dict[str, Any]],
    goals: List[Dict[str, Any]],
    concepts: List[Dict[str, Any]],
    recent_episodes: List[Dict[str, Any]],
    evidence: Optional[List[Dict[str, Any]]] = None,
    routing_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the canonical structured context bundle.

    The schema is intentionally minimal:
    - stable enough for deterministic serialization
    - expressive enough for actor-facing context assembly
    - easy to inspect in logs and replay workflows
    """
    bundle = {
        "schema_version": 1,
        "step": int(step),
        "constraints": constraints,
        "goals": goals,
        "concepts": concepts,
        "recent_episodes": recent_episodes,
        "evidence": evidence or [],
        "routing_state": routing_state or {},
    }
    return bundle


def serialize_context(bundle: Dict[str, Any]) -> str:
    """
    Serialize a context bundle to canonical JSON text.
    """
    return _stable_dumps(bundle)
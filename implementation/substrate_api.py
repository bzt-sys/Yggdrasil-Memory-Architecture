from __future__ import annotations

"""
Stable public façade for the Yggdrasil engine.

This module provides a minimal wrapper around ``YggdrasilEngine`` so that
callers do not depend directly on internal engine method names. The goal is
to preserve a small, readable, and relatively stable integration surface for:

- interactive entrypoints
- test / replay utilities
- future adapters or service layers

The API intentionally exposes only the operations most external callers need:
event recording, outcome logging, deterministic context retrieval, and basic
runtime inspection.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .memory import YggdrasilEngine


@dataclass
class SubstrateAPI:
    """
    Thin stable wrapper around ``YggdrasilEngine``.

    This façade exists to reduce coupling between the engine internals and the
    rest of the repository. Public callers should prefer this surface when they
    do not need direct access to lower-level engine behavior.
    """

    engine: YggdrasilEngine

    # ------------------------------------------------------------------
    # Event and outcome recording
    # ------------------------------------------------------------------

    def record_event(
        self,
        *,
        role: str,
        text: str,
        source: str = "cli",
        provenance: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """
        Record an event and immediately ingest it into the substrate.

        Parameters:
        - role: event role such as ``user`` or ``assistant``
        - text: raw event text
        - source: source label for provenance tracking
        - provenance: optional structured provenance metadata

        Returns:
        - the created event id
        """
        event_id = self.engine.write_event(
            role=role,
            text=text,
            source=source,
            provenance=provenance,
            event_id=event_id,
        )
        self.engine.adjudicate_and_ingest(event_id)
        return event_id

    def log_outcome(
        self,
        *,
        score: float,
        label: str = "",
        tags: Optional[List[str]] = None,
        outcome_id: Optional[str] = None,
    ) -> str:
        """
        Record an outcome node for the current episode or step context.

        Parameters:
        - score: normalized scalar outcome score
        - label: optional human-readable label such as ``good`` or ``bad``
        - tags: optional free-form tags for later inspection

        Returns:
        - the created outcome id
        """
        return self.engine.write_outcome(score=score, label=label, tags=tags, outcome_id=outcome_id)

    def end_step(self) -> None:
        """
        Finalize the current engine step.

        This allows decay, promotion, compression, hashing, and other
        end-of-step maintenance logic to run.
        """
        self.engine.end_step()

    # ------------------------------------------------------------------
    # Deterministic context surface
    # ------------------------------------------------------------------

    def get_context_bundle(self) -> Dict[str, Any]:
        """
        Return the structured deterministic context bundle.

        This is the canonical machine-readable context representation exposed
        by the engine.
        """
        return self.engine.get_context()

    def get_context_json(self) -> str:
        """
        Return the structured context bundle as canonical JSON text.
        """
        return self.engine.get_context_json()

    def get_context_text(self) -> str:
        """
        Return the actor-facing rendered context text.
        """
        return self.engine.get_context_text()

    # ------------------------------------------------------------------
    # Diagnostics and inspection
    # ------------------------------------------------------------------

    def trace(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return the most recent ``n`` traced events.
        """
        return self.engine.trace_last(n=n)

    def hud(self) -> Dict[str, Any]:
        """
        Return a compact runtime summary of engine state.
        """
        return self.engine.hud()
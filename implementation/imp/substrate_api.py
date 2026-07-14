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
runtime inspection / maintenance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .evidence import export_evidence_bundle
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
        """Record an event and immediately ingest it into the substrate."""
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
        """Record an outcome node for the current episode or step context."""
        return self.engine.write_outcome(
            score=score,
            label=label,
            tags=tags,
            outcome_id=outcome_id,
        )

    def record_action_rationale(
        self,
        *,
        prompt_text: str,
        action_text: str,
        action_event_id: str = "",
        context_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a compact replayable action-basis packet for post-action diagnosis."""
        return self.engine.record_action_rationale(
            prompt_text=prompt_text,
            action_text=action_text,
            action_event_id=action_event_id,
            context_summary=context_summary or {},
        )

    def end_step(self) -> None:
        """Finalize the current engine step."""
        self.engine.end_step()

    # ------------------------------------------------------------------
    # Deterministic context surface
    # ------------------------------------------------------------------

    def get_context_bundle(self, *, prompt_text: str = "", max_items: int = 8) -> Dict[str, Any]:
        """
        Return the structured deterministic context bundle.

        Optional prompt text keeps context retrieval/ranking prompt-aware
        without callers reaching into engine internals. The nested fallbacks
        keep compatibility with older engines whose context signatures were
        less expressive.
        """
        try:
            return self.engine.get_context(prompt_text=prompt_text, max_items=max_items)
        except TypeError:
            try:
                return self.engine.get_context(prompt_text=prompt_text)
            except TypeError:
                return self.engine.get_context()

    def get_context_json(self, *, prompt_text: str = "", max_items: int = 8) -> str:
        """Return the structured context bundle as canonical JSON text."""
        try:
            return self.engine.get_context_json(prompt_text=prompt_text, max_items=max_items)
        except TypeError:
            try:
                return self.engine.get_context_json(prompt_text=prompt_text)
            except TypeError:
                return self.engine.get_context_json()

    def get_context_text(self, *, prompt_text: str = "", max_items: int = 8) -> str:
        """Return the actor-facing rendered context text."""
        try:
            return self.engine.get_context_text(prompt_text=prompt_text, max_items=max_items)
        except TypeError:
            try:
                return self.engine.get_context_text(prompt_text=prompt_text)
            except TypeError:
                return self.engine.get_context_text()

    # ------------------------------------------------------------------
    # Diagnostics and inspection
    # ------------------------------------------------------------------

    def diagnostics_summary(self, recent: int = 5) -> Dict[str, Any]:
        """Return a compact diagnostics summary without exposing engine internals."""
        return self.engine.diagnostics_summary(recent=int(recent))

    def diagnostics(self, recent: int = 5) -> Dict[str, Any]:
        """Alias for diagnostics_summary used by some callers."""
        return self.diagnostics_summary(recent=recent)

    def trace(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent ``n`` traced events."""
        return self.engine.trace_last(n=int(n))

    def hud(self) -> Dict[str, Any]:
        """Return a compact runtime summary of engine state."""
        return self.engine.hud()

    def graph_summary(self) -> str:
        """Return the graph's human-readable summary."""
        return self.engine.graph.format_summary()

    def lifecycle_events_since(self, cursor: int, max_events: int = 300) -> List[Dict[str, Any]]:
        """Return logger events after ``cursor`` when the engine logger supports it.

        Most session code implements this directly by reading the JSONL log.
        This method is intentionally best-effort and safe for callers that want
        to avoid direct engine/logger coupling later.
        """
        try:
            path = getattr(self.engine.log, "path", None)
            if not path:
                return []
            from .storage import read_jsonl

            rows = read_jsonl(path)
            return rows[int(cursor): int(cursor) + int(max_events)]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Maintenance / adaptive lifecycle controls
    # ------------------------------------------------------------------

    def run_collation_period(self, max_items: int = 64, reason: str = "api") -> Dict[str, Any]:
        """Run the engine collation period through the public API surface."""
        return self.engine.run_collation_period(max_items=int(max_items), reason=reason)

    def collate(self, max_items: int = 64, reason: str = "api") -> Dict[str, Any]:
        """Alias for run_collation_period."""
        return self.run_collation_period(max_items=max_items, reason=reason)

    def set_ablation(self, enabled: bool) -> None:
        """Enable or disable governance ablation."""
        self.engine.set_ablation(bool(enabled))

    def export_evidence(self, out_dir: Union[str, Path]) -> Dict[str, Any]:
        """Export evidence artifacts for the current engine state."""
        return export_evidence_bundle(self.engine, str(out_dir))

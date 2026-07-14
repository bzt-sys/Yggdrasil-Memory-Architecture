from __future__ import annotations

"""Structured evidence helpers for Yggdrasil.

The evidence layer translates raw environment feedback into bounded, typed
signals before investigation/reflection consume it.  It intentionally avoids
graph mutation; the memory engine remains the runtime owner of storage.
"""

from .environment_feedback import EnvironmentEvidence, parse_objective_feedback_text

__all__ = ["EnvironmentEvidence", "parse_objective_feedback_text"]

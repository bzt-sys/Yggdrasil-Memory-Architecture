"""Candidate cache helpers for Yggdrasil.

This package owns pure candidate projection/ranking policy. The live engine
still owns graph mutation, expiry, and lifecycle orchestration.
"""

from .retrieval import retrieve_ranked_candidates

__all__ = ["retrieve_ranked_candidates"]

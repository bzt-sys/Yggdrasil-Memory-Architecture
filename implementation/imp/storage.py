from __future__ import annotations

"""
Append-only JSONL logging utilities for Yggdrasil.

This module provides a minimal logger that writes structured records to a
JSONL file. The format is intentionally simple and append-only so logs are
easy to inspect, diff, replay, and audit.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class JsonlLogger:
    """
    Minimal append-only JSONL logger.
    """

    path: str

    def __post_init__(self) -> None:
        """
        Create the parent directory for the log path when needed.
        """
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def log(
        self,
        rec_type: str,
        data: Dict[str, Any],
        ts: Optional[float] = None,
    ) -> None:
        """
        Append a structured log row to the JSONL file.

        Each row contains:
        - ts: event timestamp
        - type: record type name
        - data: structured record payload
        """
        row = {
            "ts": float(ts if ts is not None else time.time()),
            "type": rec_type,
            "data": data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

class NullLogger:
    """
    Logger with the same interface as JsonlLogger that discards records.

    Used for resume/replay reconstruction when we need to rebuild state without
    appending duplicate causal rows to the source session log.
    """

    def log(self, rec_type: str, data: Dict[str, Any], ts: Optional[float] = None) -> None:
        return None


def read_jsonl(path: str) -> list[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    rows: list[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a deterministic JSON artifact."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def append_text(path: str, text: str) -> None:
    """Append text to a file, creating parent directories as needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)

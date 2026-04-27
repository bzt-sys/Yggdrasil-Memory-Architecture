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
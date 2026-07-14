"""Post-patch validation checks for Yggdrasil development builds.

This script is deliberately lightweight. It does not load the local LLM or run a
benchmark. It verifies that the current code exposes the structured interfaces
needed by the experiment harness and catches common refactor mistakes such as
methods being nested under the wrong indentation level.

Usage:
    python imp/tools/validate_yggdrasil_install.py
"""

from __future__ import annotations

import importlib
import sys
from typing import Iterable, List, Tuple


REQUIRED_MODULES: Tuple[str, ...] = (
    "imp.memory",
    "imp.agent_session",
    "imp.context_schema",
    "imp.graph",
    "imp.state_hash",
    "imp.storage",
)

REQUIRED_ENGINE_METHODS: Tuple[str, ...] = (
    "write_event",
    "adjudicate_and_ingest",
    "close_episode_with_reflection",
    "get_context",
    "get_context_json",
    "get_context_text",
    "hud",
)

OPTIONAL_ENGINE_METHODS: Tuple[str, ...] = (
    "retrieve_candidate_context",
    "retrieve_investigations_for_prompt",
    "retrieve_governance_proposals_for_prompt",
    "open_pre_answer_investigation",
)

REQUIRED_SESSION_METHODS: Tuple[str, ...] = (
    "run_prompt",
    "diagnostics",
)


def _check_imports(module_names: Iterable[str]) -> List[str]:
    errors: List[str] = []
    for name in module_names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            errors.append(f"import failed: {name}: {exc!r}")
    return errors


def _check_attrs(obj: object, names: Iterable[str], label: str, *, required: bool = True) -> List[str]:
    messages: List[str] = []
    for name in names:
        ok = hasattr(obj, name)
        prefix = "missing" if required else "optional missing"
        if not ok:
            messages.append(f"{prefix}: {label}.{name}")
    return messages


def main() -> int:
    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(_check_imports(REQUIRED_MODULES))

    try:
        memory = importlib.import_module("imp.memory")
        engine_cls = getattr(memory, "YggdrasilEngine")
        errors.extend(_check_attrs(engine_cls, REQUIRED_ENGINE_METHODS, "YggdrasilEngine"))
        warnings.extend(_check_attrs(engine_cls, OPTIONAL_ENGINE_METHODS, "YggdrasilEngine", required=False))
    except Exception as exc:
        errors.append(f"engine inspection failed: {exc!r}")

    try:
        session_mod = importlib.import_module("imp.agent_session")
        session_cls = getattr(session_mod, "YggdrasilAgentSession")
        errors.extend(_check_attrs(session_cls, REQUIRED_SESSION_METHODS, "YggdrasilAgentSession"))
    except Exception as exc:
        errors.append(f"session inspection failed: {exc!r}")

    print("Yggdrasil install validation")
    print("=" * 32)

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
        print()

    if errors:
        print("FAILED")
        for e in errors:
            print(f"- {e}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

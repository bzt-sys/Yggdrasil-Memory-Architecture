from __future__ import annotations

"""
Interactive CLI entrypoint for Yggdrasil.

This module exposes the primary user-facing loop for the runtime
constraint substrate. It supports:

- normal user / assistant interaction
- explicit outcome scoring via :rate
- policy / lesson ingestion via :lesson
- governance ablation toggling via :ablate
- state inspection via :stats and :trace

The intended workflow is:

1. user submits a prompt
2. the engine ingests and adjudicates the event
3. substrate context is assembled
4. the actor generates a response
5. the assistant response is ingested back into the substrate
6. the user may optionally score the result with :rate
7. the user may optionally convert that evaluation into a reusable
   behavioral update with :lesson

This CLI is the canonical interactive demo surface for the project.
"""

import argparse
from typing import List, Optional

from .actor_hf import ActorConfig, HFActor
from .memory import HybridAdjudicator, YggdrasilEngine
from .storage import JsonlLogger
from .session_state import SessionStore
from .evidence import export_evidence_bundle


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    ap = argparse.ArgumentParser(
        description="Interactive CLI for the Yggdrasil runtime constraint substrate."
    )

    ap.add_argument("--session", default="demo-session", help="Session identifier.")
    ap.add_argument("--log", default="./logs/demo.jsonl", help="Path to the JSONL log file.")
    ap.add_argument("--session-root", default="sessions", help="Root directory for standardized session artifacts.")
    ap.add_argument("--use-session-store", action="store_true", help="Write canonical session log to sessions/<session>/events.jsonl.")
    ap.add_argument("--resume", action="store_true", help="Rebuild previous session state from sessions/<session>/events.jsonl before accepting prompts.")

    ap.add_argument(
        "--debug-lexical",
        action="store_true",
        help=(
            "Enable token / bigram extraction by default for USER events. "
            "Assistant events remain structured-only."
        ),
    )
    ap.add_argument(
        "--trace-default",
        type=int,
        default=0,
        help="Automatically print :trace N after each completed step if N > 0.",
    )

    ap.add_argument(
        "--no-decay",
        action="store_true",
        help=(
            "Disable decay/compression/failsafe maintenance for controlled replay "
            "experiments. Governance topology is still formed; pruning dynamics are skipped."
        ),
    )

    ap.add_argument(
        "--export-evidence",
        action="store_true",
        help="Export graph/state/evidence artifacts on startup/resume before entering the interactive loop.",
    )

    ap.add_argument(
        "--export-dir",
        default=None,
        help="Optional directory for exported evidence artifacts.",
    )

    ap.add_argument(
        "--actor",
        choices=["none", "hf"],
        default="none",
        help="Actor backend to use. 'none' returns a placeholder response.",
    )
    ap.add_argument("--model", default="", help="HF model id or local model path.")
    ap.add_argument("--max-new-tokens", type=int, default=192, help="Maximum response length.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    ap.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for generation.")
    ap.add_argument(
        "--device",
        default="auto",
        help="Transformers device_map setting (for example: auto, cuda, cpu).",
    )
    ap.add_argument("--dtype", default="auto", help="Model dtype selection (for example: auto, fp16).")

    return ap


def make_actor(args: argparse.Namespace) -> Optional[HFActor]:
    """
    Construct the optional actor backend.

    Deterministic defaults are preferred so that behavior differences are
    more attributable to substrate context than to sampling noise.
    """
    if args.actor == "none":
        return None

    if args.actor == "hf":
        if not args.model:
            raise SystemExit("--actor hf requires --model (HF repo id or local path).")

        cfg = ActorConfig(
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
        )
        return HFActor(
            model_id_or_path=args.model,
            cfg=cfg,
            device=args.device,
            dtype=args.dtype,
        )

    raise SystemExit(f"Unsupported actor backend: {args.actor}")


def print_banner() -> None:
    """Print the interactive command banner."""
    print(
        "Yggdrasil v0.28\n"
        "Commands: :rate 0-9 [tags...], :lesson <text>, :ablate [on|off], "
        ":trace N, :stats, :graph, :export-evidence [dir], :quit"
    )

def default_evidence_dir(args: argparse.Namespace, store: SessionStore) -> str:
    """Choose the default evidence export directory."""
    if args.export_dir:
        return args.export_dir

    if args.resume or args.use_session_store:
        return store.paths(args.session).artifacts_dir

    return f"evidence/{args.session}"

def handle_rate(
    user_in: str,
    eng: YggdrasilEngine,
    last_assistant_id: Optional[str],
) -> Optional[dict]:
    """
    Handle a :rate command.

    A rating closes the current episode outcome and stores lightweight
    provenance so a subsequent :lesson can be attached to the scored result.
    """
    parts = user_in.strip().split()
    if len(parts) < 2:
        print("Usage: :rate <0-9> [tags...]")
        return None

    try:
        rating = int(parts[1])
    except Exception:
        print("Usage: :rate <0-9> [tags...]")
        return None

    if rating < 0 or rating > 9:
        print("Rating must be an integer from 0 to 9.")
        return None

    tags = parts[2:]

    # Normalize the rating to [-1.0, +1.0].
    score = (float(rating) - 4.5) / 4.5

    if rating >= 6:
        label = "good"
    elif rating <= 3:
        label = "bad"
    else:
        label = "neutral"

    eng.write_outcome(score=score, label=label, tags=tags)

    pending_rating = {
        "rating": rating,
        "score": score,
        "label": label,
        "tags": tags,
        "target": (last_assistant_id[:8] if last_assistant_id else "none"),
    }

    # Retained only for debugging / inspection; not part of memory formation.
    eng._pending_rating = pending_rating  # type: ignore[attr-defined]

    eng.end_step()
    _print_hud(eng.hud())
    print("Recorded. Optionally follow with :lesson <text> to create a policy update.")
    return pending_rating


def handle_lesson(
    user_in: str,
    eng: YggdrasilEngine,
    pending_rating: Optional[dict],
) -> Optional[dict]:
    """
    Handle a :lesson command.

    Lessons are written as user events with an explicit 'lesson:' prefix so
    downstream extraction remains deterministic and easy to interpret.
    """
    lesson_text = user_in[len(":lesson") :].strip()
    if not lesson_text:
        print("Usage: :lesson <text>")
        return pending_rating

    payload = "lesson: " + lesson_text

    if pending_rating:
        payload += (
            f"\n[rating_ref] score={pending_rating['score']:.3f} "
            f"label={pending_rating['label']} tags={','.join(pending_rating['tags'])} "
            f"target={pending_rating['target']}"
        )
        pending_rating = None
        try:
            delattr(eng, "_pending_rating")
        except Exception:
            pass

    ev_id = eng.write_event(role="user", text=payload)
    eng.adjudicate_and_ingest(ev_id)

    eng.end_step()
    _print_hud(eng.hud())
    return pending_rating


def handle_ablation(user_in: str, eng: YggdrasilEngine) -> None:
    """Handle governance ablation toggling."""
    parts = user_in.split()

    if len(parts) == 1:
        eng.set_ablation(not eng.ablate_governance)
    else:
        arg = parts[1].strip().lower()
        if arg in ("on", "true", "1", "enable", "enabled"):
            eng.set_ablation(True)
        elif arg in ("off", "false", "0", "disable", "disabled"):
            eng.set_ablation(False)
        else:
            print("Usage: :ablate [on|off]")
            return

    state = "ON" if eng.ablate_governance else "OFF"
    print(f"Governance ablation: {state}")
    _print_hud(eng.hud())


def run_interaction(
    user_in: str,
    eng: YggdrasilEngine,
    actor: Optional[HFActor],
    logger: JsonlLogger,
) -> str:
    """
    Execute one normal user → assistant interaction step.

    Returns the assistant event id so that future :rate commands can point
    back to the most recently scored assistant response.
    """
    ev_id = eng.write_event(role="user", text=user_in)
    eng.adjudicate_and_ingest(ev_id)

    context_text = eng.get_context_text(prompt_text=user_in)
    if context_text:
        if getattr(eng, "_last_context_json", ""):
            logger.log(
                "context_selected",
                {"step": eng.step, "context_json": eng._last_context_json},
            )
        else:
            logger.log(
                "context_selected",
                {"step": eng.step, "context_text": context_text},
            )

    if actor is None:
        actor_text = "[placeholder actor response]"
    else:
        actor_text = actor.generate(context_text=context_text, user_text=user_in)

    print(f"\nAssistant: {actor_text}\n")

    assistant_id = eng.write_event(role="assistant", text=actor_text)
    eng.adjudicate_and_ingest(assistant_id)
    return assistant_id


def main() -> None:
    """Run the interactive Yggdrasil CLI."""
    args = build_parser().parse_args()

    adjudicator = HybridAdjudicator(enable_lexical_default=bool(args.debug_lexical))
    store = SessionStore(args.session_root)

    if args.resume or args.use_session_store:
        session_paths = store.paths(args.session)
        logger = JsonlLogger(session_paths.events_jsonl)
        if args.resume:
            eng = store.resume_engine(
                args.session,
                adjudicator=adjudicator,
                append_log_path=session_paths.events_jsonl,
                verify=True,
                enable_decay=not bool(args.no_decay),
            )
            report = getattr(eng, "_last_resume_report", None)
            if report:
                status = "PASS" if report.get("ok") else "FAIL"
                print(f"[RESUME] {status} step={eng.step} hash={report.get('final_hash', '')[:12]}...")
        else:
            eng = YggdrasilEngine(
                session_id=args.session,
                logger=logger,
                adjudicator=adjudicator,
                enable_decay=not bool(args.no_decay),
            )
    else:
        logger = JsonlLogger(args.log)
        eng = YggdrasilEngine(
            session_id=args.session,
            logger=logger,
            adjudicator=adjudicator,
            enable_decay=not bool(args.no_decay),
        )
    
    actor = make_actor(args)

    if args.export_evidence:
        out_dir = default_evidence_dir(args, store)
        export_evidence_bundle(eng, out_dir)
        print(f"[EXPORT] evidence written to {out_dir}")

    last_assistant_id: Optional[str] = None
    pending_rating: Optional[dict] = None

    print_banner()

    while True:
        eng.start_step()

        try:
            user_in = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_in:
            continue

        if user_in.startswith(":quit"):
            break

        if user_in.startswith(":rate"):
            pending_rating = handle_rate(user_in, eng, last_assistant_id)
            continue

        if user_in.startswith(":lesson"):
            pending_rating = handle_lesson(user_in, eng, pending_rating)
            continue

        if user_in.startswith(":ablate") or user_in.startswith(":ablation"):
            handle_ablation(user_in, eng)
            continue

        if user_in.startswith(":graph"):
            print(eng.graph.format_summary())
            continue

        if user_in.startswith(":export-evidence"):
            parts = user_in.split(maxsplit=1)
            out_dir = parts[1].strip() if len(parts) > 1 else default_evidence_dir(args, store)
            export_evidence_bundle(eng, out_dir)
            print(f"[EXPORT] evidence written to {out_dir}")
            continue

        if user_in.startswith(":stats"):
            _print_hud(eng.hud())
            continue

        if user_in.startswith(":trace"):
            try:
                n = int(user_in.split()[1])
            except Exception:
                n = 5
            _print_trace(eng.trace_last(n))
            continue

        last_assistant_id = run_interaction(user_in, eng, actor, logger)

        eng.end_step()
        _print_hud(eng.hud())

        if args.trace_default and args.trace_default > 0:
            _print_trace(eng.trace_last(args.trace_default))


def _print_hud(hud: dict) -> None:
    """Print a compact runtime summary for the current engine state."""
    c = hud["counts"]
    last = hud.get("last_step", {})
    ablated = bool((hud.get("ablation") or {}).get("governance_disabled"))
    ablation_txt = " | ablation=ON" if ablated else " | ablation=OFF"

    print(
        f"[HUD] step={c['step']} | episodes={c['episodes']} "
        f"(open={c['episode_open']}) | events={c['events']} "
        f"| vocab_features={c['features']} | concepts={c['concepts']} "
        f"| detail_edges={c['detail_edges']} | outcomes={c['outcomes']}"
        f"{ablation_txt}"
    )
    print(
        f"      decay: decayed={last.get('decayed', 0)} "
        f"deleted={last.get('deleted_edges', 0)} "
        f"compressed={last.get('compressed_edges', 0)} "
        f"promoted={last.get('promoted', 0)}"
    )

    if last.get("state_hash"):
        print(f"      state_hash: {last.get('state_hash')[:12]}...")

    top = hud.get("top_concepts") or []
    if top:
        print("Top concepts:", ", ".join([f"{k}({v:.2f})" for k, v in top]))


def _print_trace(rows: List[dict]) -> None:
    """Print a compact trace of recent events and extracted features."""
    print("---- TRACE ----")
    for r in rows:
        txt = (r.get("text") or "").replace("\n", " ")
        if len(txt) > 90:
            txt = txt[:87] + "..."

        meta = r.get("meta") or {}
        print(
            f"- {r['event_id'][:8]} role={r['role']} ep={str(r['episode_id'])[:8]} "
            f"intent={r.get('intent')} structured={meta.get('structured_count')} "
            f"lexical={meta.get('lexical_enabled')} :: {txt}"
        )

        feats = r.get("features", [])
        if feats:
            show = []
            for f in feats[:10]:
                show.append(
                    f"{f.get('kind')}:{f.get('key')} "
                    f"sal={float(f.get('sal', 0.0)):.3f} "
                    f"conf={float(f.get('conf', 0.0)):.2f}"
                )
            print("  feats:", " | ".join(show))
    print("--------------")


if __name__ == "__main__":
    main()
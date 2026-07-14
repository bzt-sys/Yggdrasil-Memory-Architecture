from __future__ import annotations

"""
Structured execution API for Yggdrasil agent experiments.

This module moves benchmark / experiment execution away from the interactive
CLI.  The CLI remains useful for human demos, but experiment harnesses should
instantiate YggdrasilAgentSession directly so they can receive structured
responses, diagnostics, traces, and collation summaries without brittle terminal
I/O synchronization.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .actor_hf import ActorConfig, HFActor
from .memory import HybridAdjudicator, YggdrasilEngine
from .session_state import SessionStore
from .storage import JsonlLogger, read_jsonl
from .substrate_api import SubstrateAPI


@dataclass
class AgentSessionConfig:
    session: str = "demo-session"
    log: str = "./logs/demo.jsonl"
    session_root: str = "sessions"
    use_session_store: bool = True
    resume: bool = False
    debug_lexical: bool = False
    no_decay: bool = False
    actor: str = "none"
    model: str = "D:\Yggdrasil\imp_split\models\Mistral-7B-Instruct-v0.3"
    max_new_tokens: int = 192
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"


def make_actor_from_config(cfg: AgentSessionConfig) -> Optional[HFActor]:
    """Construct the optional actor backend from a structured config."""
    if cfg.actor == "none":
        return None

    if cfg.actor == "hf":
        if not cfg.model:
            raise ValueError("actor='hf' requires a model path or model id.")

        actor_cfg = ActorConfig(
            max_new_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            seed=int(cfg.seed),
        )
        return HFActor(
            model_id_or_path=cfg.model,
            cfg=actor_cfg,
            device=cfg.device,
            dtype=cfg.dtype,
        )

    raise ValueError(f"Unsupported actor backend: {cfg.actor}")


class YggdrasilAgentSession:
    """
    Callable Yggdrasil runtime session.

    This object owns a YggdrasilEngine, optional actor backend, and event logger.
    Experiment harnesses should call `run_prompt`, `rate`, `diagnostics`,
    `trace`, and `collate` directly instead of automating the CLI.
    """

    def __init__(self, cfg: AgentSessionConfig):
        self.cfg = cfg
        self.adjudicator = HybridAdjudicator(enable_lexical_default=bool(cfg.debug_lexical))
        self.store = SessionStore(cfg.session_root)

        if cfg.resume or cfg.use_session_store:
            session_paths = self.store.paths(cfg.session)
            self.logger = JsonlLogger(session_paths.events_jsonl)
            if cfg.resume:
                self.engine = self.store.resume_engine(
                    cfg.session,
                    adjudicator=self.adjudicator,
                    append_log_path=session_paths.events_jsonl,
                    verify=True,
                    enable_decay=not bool(cfg.no_decay),
                )
                self.resume_report = getattr(self.engine, "_last_resume_report", None)
            else:
                self.engine = YggdrasilEngine(
                    session_id=cfg.session,
                    logger=self.logger,
                    adjudicator=self.adjudicator,
                    enable_decay=not bool(cfg.no_decay),
                )
                self.resume_report = None
        else:
            self.logger = JsonlLogger(cfg.log)
            self.engine = YggdrasilEngine(
                session_id=cfg.session,
                logger=self.logger,
                adjudicator=self.adjudicator,
                enable_decay=not bool(cfg.no_decay),
            )
            self.resume_report = None

        self.api = SubstrateAPI(self.engine)
        self.actor = make_actor_from_config(cfg)
        if self.actor is not None:
            self.engine.governance_synthesis_fn = self._synthesize_governance_with_actor
            self.engine.locality_designation_fn = self._designate_locality_with_actor
            self.engine.recurrent_episode_interpretation_fn = self._interpret_recurrent_episode_with_actor
        self.last_assistant_id: Optional[str] = None
        self.pending_rating: Optional[Dict[str, Any]] = None

    def _synthesize_governance_with_actor(self, packet: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use the actor as the collation synthesis kernel.

        The actor is asked to derive provisional governance from experience and
        return strict JSON. The engine still parses, bounds, stores, and later
        promotes/demotes the result; the model does not directly mutate state.
        """
        if self.actor is None:
            return []
        system_text = (
            "You are a governance collation synthesizer for an adaptive agent. "
            "Infer provisional behavioral guidance from recurrent experience. "
            "Do not produce benchmark-specific hacks or final answers. Return JSON only."
        )
        user_text = (
            "Given this collation packet, derive up to three provisional governance proposals. "
            "Each proposal should be low-authority guidance to inject in similar future contexts and pressure-test. "
            "Return exactly this JSON shape: {\"proposals\":[{\"proposal_text\":str,\"family\":str,\"locality\":str,"
            "\"rationale\":str,\"confidence\":float,\"source_candidate_ids\":[str],\"evidence_candidate_count\":int,"
            "\"pressure_test_plan\":str}]}\n\n"
            + json.dumps(packet, ensure_ascii=False, indent=2)[:12000]
        )
        text = self.actor.generate(context_text="", user_text=user_text, system_text=system_text)
        return self.engine._parse_synthesis_output(text)


    def _interpret_recurrent_episode_with_actor(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        if self.actor is None:
            return {}
        system_text = (
            "You are interpreting a sequence of consecutive failed repair attempts as one recurrent developmental episode. "
            "Do not discard repeated evidence merely because it was already retrieved. Identify what distinction, evidence, "
            "test, or assumption is still missing. Return JSON only."
        )
        user_text = (
            "Return exactly: {\"summary\":str,\"recurrent_pattern\":str,\"missing_evidence_questions\":[str],"
            "\"contradicted_assumptions\":[str],\"next_operations\":[str],\"candidate_abstraction\":str,"
            "\"confidence\":float}. Ground every field in the supplied failed-attempt sequence.\n\n"
            + json.dumps(packet, ensure_ascii=False, indent=2)[:12000]
        )
        text = self.actor.generate(context_text="", user_text=user_text, system_text=system_text)
        try:
            obj = json.loads(text.strip())
            return obj if isinstance(obj, dict) else {}
        except Exception:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(text[start:end+1])
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
            return {}

    def _designate_locality_with_actor(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the actor to propose topology placement; engine adjudicates it."""
        if self.actor is None:
            return {}
        system_text = (
            "You are proposing the minimum useful conceptual placement for a developmental hypothesis. "
            "Begin from uncertainty, reuse the deepest adequate existing node, and create at most one new level. "
            "Do not mutate state and do not invent a deep taxonomy. Return JSON only."
        )
        user_text = (
            "Choose one action: attach_existing, propose_family, propose_subfamily, or hold_unclassified. "
            "Return exactly: {\"action\":str,\"family\":str,\"locality\":str,\"parent_locality\":str|null,"
            "\"proposed_name\":str,\"scope\":str,\"rationale\":str,\"confidence\":float}. "
            "A family must be directly beneath root::unknown. A subfamily must be one level beneath an existing family. "
            "Use hold_unclassified when evidence is insufficient.\n\n" + json.dumps(packet, ensure_ascii=False, indent=2)[:12000]
        )
        text = self.actor.generate(context_text="", user_text=user_text, system_text=system_text)
        try:
            obj = json.loads(text.strip())
            return obj if isinstance(obj, dict) else {}
        except Exception:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(text[start:end+1])
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    pass
            return {}

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)

    def raw_generate(self, prompt: str, *, context_text: str = "", system_text: Optional[str] = None) -> str:
        """Generate text without mutating Yggdrasil state.

        Experiment harnesses use this for uncertainty probes and private
        deliberation candidates. Because this method does not record events, any
        caller that relies on it should explicitly log the resulting gate or
        deliberation trace through the runner record/session observation.
        """
        if self.actor is None:
            return "[placeholder actor response]"
        return self.actor.generate(context_text=context_text, user_text=prompt, system_text=system_text)

    def run_prompt(
        self,
        prompt: str,
        *,
        end_step: bool = True,
        context_enabled: bool = True,
        pre_answer_enabled: bool = True,
        forced_response: Optional[str] = None,
        context_mode: str = "full",
        extra_context_text: str = "",
    ) -> Dict[str, Any]:
        """Run one user prompt through Yggdrasil and return structured output.

        context_enabled=False preserves normal event/action logging while
        suppressing substrate context injection for uncertainty-gated natural
        first-pass answers. forced_response lets an experiment commit a privately
        tested deliberation candidate into the session log without regenerating.
        """
        self.engine.start_step()

        user_event_id = self.api.record_event(role="user", text=prompt, source="structured_session")

        context_text = ""
        context_json = ""
        selected_candidates: List[Dict[str, Any]] = []
        selected_governance_proposals: List[Dict[str, Any]] = []
        selected_developmental_hypotheses: List[Dict[str, Any]] = []
        selected_investigations: List[Dict[str, Any]] = []
        parsed_context: Dict[str, Any] = {}
        retrieval_budget: Dict[str, int] = {}
        if isinstance(context_mode, dict):
            retrieval_budget = {str(k): int(v) for k, v in (context_mode.get("retrieval_budget") or {}).items()}
            context_mode_label = str(context_mode.get("route") or "graded")
        else:
            context_mode_label = str(context_mode)
        if context_enabled:
            if retrieval_budget:
                setattr(self.engine, "_context_retrieval_budget", retrieval_budget)
            try:
                context_text = self.api.get_context_text(prompt_text=prompt)
            finally:
                if retrieval_budget:
                    try:
                        delattr(self.engine, "_context_retrieval_budget")
                    except Exception:
                        pass
            context_json = getattr(self.engine, "_last_context_json", "")
            if context_json:
                try:
                    parsed_context = json.loads(context_json)
                    selected_candidates = list(parsed_context.get("evidence") or [])
                    selected_governance_proposals = list(parsed_context.get("governance_proposals") or [])
                    selected_developmental_hypotheses = list(parsed_context.get("developmental_hypotheses") or [])
                    selected_investigations = list(parsed_context.get("investigations") or [])
                except Exception:
                    selected_candidates = []
                    selected_governance_proposals = []
                    selected_developmental_hypotheses = []
                    selected_investigations = []
                    parsed_context = {}
        else:
            try:
                empty_ctx = {
                    "step": self.engine.step,
                    "constraints": [],
                    "goals": [],
                    "concepts": [],
                    "recent_episodes": [],
                    "evidence": [],
                    "governance_proposals": [],
                    "developmental_hypotheses": [],
                    "investigations": [],
                    "core_cognitive_principles": [],
                    "core_cognitive_activation": {"active": False, "reasons": ["uncertainty_gate_context_suppressed"]},
                    "reasoning_paradigm_selection": {"active": False, "selected": [], "selection_reasons": ["uncertainty_gate_context_suppressed"]},
                    "meta": {"context_mode": "none", "reason": "uncertainty_gate"},
                }
                from .context_schema import serialize_context
                context_json = serialize_context(empty_ctx)
                self.engine._last_context_json = context_json
            except Exception:
                context_json = ""

        if extra_context_text:
            context_text = (context_text + "\n\n" + str(extra_context_text).strip()).strip() if context_text else str(extra_context_text).strip()

        pre_answer_investigation = None
        if context_enabled and pre_answer_enabled:
            pre_answer_investigation = self.engine.open_pre_answer_investigation(prompt_text=prompt, context=parsed_context or {})
            pre_answer_text = self.engine.format_pre_answer_investigation(pre_answer_investigation)
            if pre_answer_text:
                context_text = (context_text + "\n\n" + pre_answer_text).strip() if context_text else pre_answer_text

        context_payload: Dict[str, Any] = {
            "context_text": context_text,
            "context_json": context_json,
            "selected_candidates": selected_candidates,
            "selected_governance_proposals": selected_governance_proposals,
            "selected_developmental_hypotheses": selected_developmental_hypotheses,
            "selected_investigations": selected_investigations,
            "pre_answer_investigation": pre_answer_investigation,
            "context_enabled": bool(context_enabled),
            "pre_answer_enabled": bool(pre_answer_enabled),
            "context_mode": context_mode_label,
            "retrieval_budget": retrieval_budget,
            "extra_context_text": str(extra_context_text or ""),
        }

        # Canonical developmental injection ledger: retrieval and rendering are
        # separate state transitions so later outcome attribution can identify
        # where an artifact entered the actor-facing process.
        for artifact_type, rows, id_key in [
            ("candidate", selected_candidates, "candidate_id"),
            ("investigation", selected_investigations, "investigation_id"),
            ("developmental_hypothesis", selected_developmental_hypotheses, "hypothesis_id"),
            ("governance_proposal", selected_governance_proposals, "proposal_id"),
        ]:
            for rank, row in enumerate(rows, 1):
                aid = row.get(id_key)
                self.logger.log("artifact_injection_transition", {
                    "step": self.engine.step, "artifact_type": artifact_type, "artifact_id": aid,
                    "transition": "selected_and_rendered", "rank": rank,
                    "score": row.get("score"), "authority_role": row.get("authority_role"),
                    "context_mode": context_mode_label, "retrieval_budget": retrieval_budget,
                })

        if context_text:
            if context_payload["context_json"]:
                self.logger.log(
                    "context_selected",
                    {"step": self.engine.step, "context_json": context_payload["context_json"], "context_mode": context_mode_label, "retrieval_budget": retrieval_budget},
                )
            else:
                self.logger.log(
                    "context_selected",
                    {"step": self.engine.step, "context_text": context_text, "context_mode": context_mode_label, "retrieval_budget": retrieval_budget},
                )

        if forced_response is not None:
            response_text = str(forced_response)
        elif self.actor is None:
            response_text = "[placeholder actor response]"
        else:
            response_text = self.actor.generate(context_text=context_text, user_text=prompt)

        assistant_event_id = self.api.record_event(role="assistant", text=response_text, source="structured_session")
        self.last_assistant_id = assistant_event_id
        action_rationale = self.api.record_action_rationale(
            prompt_text=prompt,
            action_text=response_text,
            action_event_id=assistant_event_id,
            context_summary={
                "selected_candidate_ids": [c.get("candidate_id") for c in selected_candidates],
                "selected_investigation_ids": [i.get("investigation_id") for i in selected_investigations],
                "selected_governance_proposal_ids": [g.get("proposal_id") for g in selected_governance_proposals],
                "selected_developmental_hypothesis_ids": [h.get("hypothesis_id") for h in selected_developmental_hypotheses],
                "pre_answer_investigation_id": (pre_answer_investigation or {}).get("pre_answer_investigation_id") if isinstance(pre_answer_investigation, dict) else None,
            },
        )
        context_payload["action_rationale"] = action_rationale

        if end_step:
            self.api.end_step()

        return {
            "session": self.cfg.session,
            "step": self.engine.step,
            "user_event_id": user_event_id,
            "assistant_event_id": assistant_event_id,
            "prompt": prompt,
            "response": response_text,
            "context": context_payload,
            "hud": self.engine.hud(),
        }


    def record_observation(self, text: str, *, role: str = "system", end_step: bool = False) -> Dict[str, Any]:
        """Record non-generative experiment/environment feedback into the current episode.

        HumanEval and future environment harnesses use this to place objective
        feedback such as pass/fail, traceback class, or tool outcome into the
        episode before a rating closes it. The actor does not see this event
        until future prompts retrieve it through ordinary candidate/context paths.
        """
        if not text.strip():
            return {"event_id": None, "hud": self.engine.hud()}
        self.engine.start_step()
        event_id = self.api.record_event(role=role, text=text.strip(), source="environment_observation")
        if end_step:
            self.api.end_step()
        return {"event_id": event_id, "hud": self.engine.hud()}

    def rate(self, rating: int, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Record an outcome rating and close/update the current episode path."""
        tags = list(tags or [])
        rating = int(rating)
        if rating < 0 or rating > 9:
            raise ValueError("rating must be an integer from 0 to 9")

        self.engine.start_step()

        score = (float(rating) - 4.5) / 4.5
        if rating >= 6:
            label = "good"
        elif rating <= 3:
            label = "bad"
        else:
            label = "neutral"

        self.api.log_outcome(score=score, label=label, tags=tags)
        self.pending_rating = {
            "rating": rating,
            "score": score,
            "label": label,
            "tags": tags,
            "target": (self.last_assistant_id[:8] if self.last_assistant_id else "none"),
        }
        self.engine._pending_rating = self.pending_rating  # type: ignore[attr-defined]
        self.api.end_step()

        return {"pending_rating": self.pending_rating, "hud": self.engine.hud()}

    def lesson(self, text: str) -> Dict[str, Any]:
        """Create a manual lesson/policy update, mirroring CLI :lesson behavior."""
        if not text.strip():
            raise ValueError("lesson text may not be empty")

        self.engine.start_step()
        payload = "lesson: " + text.strip()
        if self.pending_rating:
            payload += (
                f"\n[rating_ref] score={self.pending_rating['score']:.3f} "
                f"label={self.pending_rating['label']} tags={','.join(self.pending_rating['tags'])} "
                f"target={self.pending_rating['target']}"
            )
            self.pending_rating = None
            try:
                delattr(self.engine, "_pending_rating")
            except Exception:
                pass

        event_id = self.api.record_event(role="user", text=payload, source="lesson")
        self.api.end_step()
        return {"event_id": event_id, "hud": self.engine.hud()}


    def log_cursor(self) -> int:
        """Return the current session-log row count for per-task lifecycle slicing."""
        try:
            return len(read_jsonl(self.logger.path))
        except Exception:
            return 0

    def lifecycle_events_since(self, cursor: int, max_events: int = 200) -> List[Dict[str, Any]]:
        """Return lifecycle-relevant log rows appended after `cursor`.

        This is experiment instrumentation. It lets benchmark harnesses capture
        the exact causal trail for one task without scraping terminal output.
        """
        rows = read_jsonl(self.logger.path)
        new_rows = rows[max(0, int(cursor)):]
        lifecycle_types = {
            "event_written",
            "episode_opened",
            "episode_closed",
            "outcome_written",
            "provisional_outcome_written",
            "candidate_cached",
            "candidate_pedigree_evaluated",
            "candidate_reinforcement_integrated",
            "candidate_variant_integrated",
            "artifact_assigned_to_locality",
            "locality_region_proposed",
            "governance_proposal_created",
            "governance_proposal_promotion_evaluated",
            "governance_promotion_evaluated",  # legacy alias
            "governance_proposal_promoted",
            "governance_proposal_demoted",
            "collation_period_started",
            "collation_period_completed",
            "collation_period_complete",  # legacy alias
            "collation_candidate_reviewed",
            "developmental_hypothesis_synthesized",
            "developmental_transition",
            "recurrent_episode_collated",
            "artifact_injection_transition",
            "investigation_workspace_created",
            "pre_answer_investigation_created",
            "action_rationale_recorded",
            "developmental_event_written",
            "state_hash",
        }
        filtered = [r for r in new_rows if str(r.get("type")) in lifecycle_types]
        return filtered[-int(max_events):]

    def diagnostics(self, recent: int = 5) -> Dict[str, Any]:
        return self.api.diagnostics_summary(recent=int(recent))

    def trace(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.api.trace(int(n))

    def collate(self, max_items: int = 64, reason: str = "experiment") -> Dict[str, Any]:
        return self.api.run_collation_period(max_items=int(max_items), reason=reason)

    def set_ablation(self, enabled: bool) -> Dict[str, Any]:
        self.api.set_ablation(bool(enabled))
        return {"ablation_enabled": bool(enabled), "hud": self.api.hud()}

    def export_evidence(self, out_dir: str | Path) -> Dict[str, Any]:
        self.api.export_evidence(out_dir)
        return {"export_dir": str(out_dir)}

    def close(self) -> None:
        """Placeholder for symmetry with subprocess-based harnesses."""
        return None

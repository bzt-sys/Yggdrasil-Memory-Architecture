from __future__ import annotations

"""
Session persistence, resume, and replay utilities for Yggdrasil.

Canonical source of truth:
    sessions/<session_id>/events.jsonl

Fast/human-readable artifacts:
    sessions/<session_id>/artifacts/latest_snapshot.json
    sessions/<session_id>/artifacts/replay_report.json
    sessions/<session_id>/artifacts/replay_report.md

The JSONL log remains the causal history. Snapshots are convenience artifacts,
not the authoritative replay source.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .memory import HybridAdjudicator, YggdrasilEngine
from .storage import JsonlLogger, NullLogger, read_jsonl, write_json
from .substrate_api import SubstrateAPI
from .state_hash import compute_state_hash, canonical_state_payload
from .replay_diagnostics import boundary_audit, canonical_payload_diff, replay_precondition_snapshot


CAUSAL_ROW_TYPES = {"event_written", "outcome_written", "developmental_event_written"}


@dataclass
class SessionPaths:
    root: str
    session_id: str

    @property
    def session_dir(self) -> str:
        return os.path.join(self.root, self.session_id)

    @property
    def artifacts_dir(self) -> str:
        return os.path.join(self.session_dir, "artifacts")

    @property
    def events_jsonl(self) -> str:
        return os.path.join(self.session_dir, "events.jsonl")

    @property
    def snapshot_json(self) -> str:
        return os.path.join(self.artifacts_dir, "latest_snapshot.json")

    @property
    def replay_report_json(self) -> str:
        return os.path.join(self.artifacts_dir, "replay_report.json")

    @property
    def replay_report_md(self) -> str:
        return os.path.join(self.artifacts_dir, "replay_report.md")

    @property
    def graph_json(self) -> str:
        return os.path.join(self.artifacts_dir, "graph.json")

    @property
    def graph_dot(self) -> str:
        return os.path.join(self.artifacts_dir, "graph.dot")

    def ensure(self) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)


def _restore_pre_answer_investigation(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    """Restore an explicitly logged pre-answer investigation artifact.

    Pre-answer investigations are prospective developmental artifacts. They are
    created during context assembly, so legacy replay from event/outcome rows can
    otherwise miss them even though they participate in the graph. Restoring the
    logged artifact keeps replay fidelity without making the schema benchmark-
    or task-specific.
    """
    inv_id = str(data.get("pre_answer_investigation_id") or data.get("id") or "")
    if not inv_id or eng.graph.find_node_by_key("pre_answer_investigation", inv_id):
        return
    payload = dict(data or {})
    payload.pop("pre_answer_investigation_id", None)
    node = eng.graph.upsert_node("pre_answer_investigation", key=inv_id, data=payload)
    investigation_nodes = list(eng.graph.find_nodes("investigation"))
    for prior_id in payload.get("prior_investigation_ids", []) or []:
        prior = eng.graph.find_node_by_key("investigation", str(prior_id))
        # Some investigation IDs are generated inside reflection during replay.
        # When an exact logged ID is unavailable, connect to the most recent
        # reconstructed investigation so the prospective artifact keeps its
        # causal dependency without requiring benchmark-specific ID plumbing.
        if prior is None and investigation_nodes:
            prior = investigation_nodes[-1]
        if prior:
            eng.graph.add_edge(node.id, prior.id, "considers_investigation", confidence=0.70, data={"step": int(data.get("step", eng.step)), "task_id": payload.get("task_id", "")})

def _restore_action_rationale(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    """Restore a replay-visible action-rationale artifact.

    Action rationales are deliberately compact action-basis packets, not hidden
    chain-of-thought.  They are logged separately from the assistant event so
    post-action investigation can compare the committed action, its visible
    basis, and the environment signal during resume/replay.
    """
    rid = str(data.get("rationale_id") or data.get("event_id") or data.get("id") or "")
    if not rid or eng.graph.find_node_by_key("action_rationale", rid):
        return
    payload = dict(data or {})
    payload.setdefault("rationale_id", rid)
    eng.graph.upsert_node("action_rationale", key=rid, data=payload)


def _restore_node_from_log(
    eng: YggdrasilEngine,
    *,
    node_type: str,
    key_fields: Iterable[str],
    data: Dict[str, Any],
) -> Optional[Any]:
    """Merge a lifecycle packet into the causally reconstructed artifact.

    Replay commonly regenerates an equivalent node with a fresh runtime UUID.
    Prefer a same-step, unclaimed node over creating a duplicate, and retain an
    alias from the logged identity for later packet references.
    """
    key = next((str(data.get(f)) for f in key_fields if data.get(f)), "")
    if not key:
        return None
    aliases = getattr(eng, "_replay_id_alias", {})
    claimed = getattr(eng, "_replay_claimed_nodes", set())
    existing = eng.graph.find_node_by_key(node_type, key)
    if existing is None and key in aliases:
        existing = eng.graph.find_node_by_key(node_type, aliases[key])
    if existing is None:
        target_step = int(data.get("step", getattr(eng, "step", 0)) or 0)
        candidates = [n for n in eng.graph.find_nodes(node_type) if n.id not in claimed]
        def match_score(node: Any) -> tuple[int, int, str]:
            score = 0
            node_step = int(node.data.get("created_step", node.data.get("step", target_step)) or target_step)
            if node_step == target_step:
                score += 4
            for field in ("family", "locality", "label", "status", "source", "task_id", "episode_id", "provisional_outcome_id"):
                expected = data.get(field)
                actual = node.data.get(field)
                if expected is not None and actual is not None and str(expected) == str(actual):
                    score += 3
            if data.get("score") is not None and node.data.get("score") is not None:
                if abs(float(data.get("score")) - float(node.data.get("score"))) < 1e-8:
                    score += 3
            if node_type == "outcome" and data.get("provisional_outcome_id"):
                if node.data.get("label") == "provisional_reflection" or node.data.get("source") == "reflection":
                    score += 8
                else:
                    score -= 8
            return (score, -node_step, str(node.key))
        if candidates:
            existing = max(candidates, key=match_score)
            aliases[key] = str(existing.key)
    payload = dict(data or {})
    payload.setdefault("replay_restored_from", "lifecycle_packet")
    payload.setdefault("step", int(data.get("step", getattr(eng, "step", 0)) or 0))
    node = existing or eng.graph.upsert_node(node_type, key=key, data={})
    node.data.update(payload)
    claimed.add(node.id)
    eng._replay_id_alias = aliases  # type: ignore[attr-defined]
    eng._replay_claimed_nodes = claimed  # type: ignore[attr-defined]
    return node


def _restore_investigation_workspace(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    node = _restore_node_from_log(eng, node_type="investigation", key_fields=("investigation_id", "id"), data=data)
    if not node:
        return
    node.data.setdefault("status", data.get("status", "open"))
    ep_id = data.get("episode_id")
    if ep_id:
        ep = eng.graph.find_node_by_key("episode", str(ep_id))
        if ep:
            eng.graph.add_edge(node.id, ep.id, "investigates", confidence=float(data.get("confidence", 0.6) or 0.6), data={"step": int(data.get("step", eng.step) or 0), "reason": data.get("reason")})
    for prior_id in data.get("prior_investigation_ids", []) or []:
        prior = eng.graph.find_node_by_key("investigation", str(prior_id))
        if prior:
            eng.graph.add_edge(node.id, prior.id, "continues_investigation", confidence=float(data.get("confidence", 0.6) or 0.6), data={"step": int(data.get("step", eng.step) or 0), "task_id": data.get("task_id", "")})


def _restore_provisional_outcome(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    node = _restore_node_from_log(eng, node_type="outcome", key_fields=("outcome_id", "provisional_outcome_id", "id"), data=data)
    if not node:
        return
    node.data.setdefault("status", data.get("status", "provisional"))
    ep_id = data.get("episode_id")
    if ep_id:
        ep = eng.graph.find_node_by_key("episode", str(ep_id))
        if ep:
            eng.graph.add_edge(node.id, ep.id, "summarizes", confidence=0.65, data={"step": int(data.get("step", eng.step) or 0), "reason": data.get("reason", "replay_restore")})
    inv_id = data.get("investigation_id")
    if inv_id:
        inv = eng.graph.find_node_by_key("investigation", str(inv_id))
        if inv:
            eng.graph.add_edge(node.id, inv.id, "derived_from_investigation", confidence=0.65, data={"step": int(data.get("step", eng.step) or 0)})


def _restore_candidate(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    node = _restore_node_from_log(eng, node_type="candidate", key_fields=("candidate_id", "id"), data=data)
    if not node:
        return
    node.data.setdefault("status", "cached")
    node.data.setdefault("pedigree_status", data.get("pedigree_status", "unchecked"))
    node.data.setdefault("integration_action", data.get("integration_action", "undecided"))
    node.data.setdefault("recurrence_count", int(data.get("recurrence_count", 1) or 1))
    if str(node.data.get("candidate_kind") or data.get("candidate_kind") or "") == "successful_action_anchor":
        locality = str(node.data.get("locality") or data.get("locality") or "")
        if locality:
            parent = str(data.get("parent_locality") or ((data.get("locality_path") or [None, "root::unknown"])[-2] if len(data.get("locality_path") or []) > 1 else "root::unknown"))
            region = eng.graph.upsert_node("locality_region", key=locality, data={
                "locality": locality, "family": data.get("family"), "target_type": data.get("target_type"),
                "target_key": data.get("target_key"), "locality_path": list(data.get("locality_path", []) or []),
                "locality_depth": data.get("locality_depth"), "locality_status": data.get("locality_status"),
                "parent_locality": parent, "ontology_origin": "experience_proposed",
                "created_step": int(data.get("created_step", data.get("step", eng.step)) or 0),
                "last_seen_step": int(data.get("step", eng.step) or 0), "artifact_count": 1,
            })
            eng.graph.add_edge(node.id, region.id, "assigned_to", confidence=float(data.get("locality_confidence", 0.5) or 0.5), data={
                "step": int(data.get("step", eng.step) or 0), "artifact_role": "successful_action_anchor",
                "locality_status": data.get("locality_status"), "assignment_reason": data.get("assignment_reason"),
            })
            parent_node = eng.graph.find_node_by_key("locality_region", parent)
            if parent_node:
                eng.graph.add_edge(parent_node.id, region.id, "contains_locality", confidence=float(data.get("locality_confidence", 0.5) or 0.5), data={"step": int(data.get("step", eng.step) or 0)})
    ep_id = data.get("episode_id")
    if ep_id:
        ep = eng.graph.find_node_by_key("episode", str(ep_id))
        if ep:
            eng.graph.add_edge(node.id, ep.id, "derived_from", confidence=0.65, data={"step": int(data.get("step", eng.step) or 0), "replay_restore": True})
    out_id = data.get("provisional_outcome_id") or data.get("outcome_id")
    if out_id:
        out = eng.graph.find_node_by_key("outcome", str(out_id))
        if out:
            eng.graph.add_edge(node.id, out.id, "supports", confidence=0.65, data={"step": int(data.get("step", eng.step) or 0), "provisional": True, "replay_restore": True})


def _restore_candidate_pedigree(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    cid = str(data.get("candidate_id") or data.get("id") or "")
    if not cid:
        return
    aliases = getattr(eng, "_replay_id_alias", {})
    node = eng.graph.find_node_by_key("candidate", cid)
    if node is None and cid in aliases:
        node = eng.graph.find_node_by_key("candidate", aliases[cid])
    if node is None:
        node = _restore_node_from_log(eng, node_type="candidate", key_fields=("candidate_id", "id"), data=data)
    if node is None:
        return
    node.data.update({
        "pedigree_status": "evaluated",
        "integration_action": data.get("action") or data.get("integration_action"),
        "integration_confidence": data.get("integration_confidence"),
        "novelty_score": data.get("novelty_score", data.get("novelty")),
        "similarity_score": data.get("similarity_score", data.get("similarity")),
        "feature_similarity_score": data.get("feature_similarity_score", data.get("feature_similarity")),
        "usefulness_score": data.get("usefulness_score", data.get("usefulness")),
        "nearest_candidate_id": data.get("nearest_candidate_id"),
        "variant_of": data.get("variant_of"),
        "recurrence_count": data.get("recurrence_count", node.data.get("recurrence_count", 1)),
        "pedigree_reasons": data.get("reasons", []),
        "pedigree_replay_packet": dict(data or {}),
    })
    nearest = data.get("nearest_candidate_id") or data.get("variant_of")
    if nearest:
        prior = eng.graph.find_node_by_key("candidate", str(nearest))
        if prior:
            eng.graph.add_edge(node.id, prior.id, "pedigree_related_to", confidence=float(data.get("integration_confidence", 0.5) or 0.5), data={"step": int(data.get("step", eng.step) or 0), "action": data.get("action")})


def _restore_locality_assignment(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    locality = str(data.get("locality") or "")
    if locality:
        loc = eng.graph.upsert_node("locality_region", key=locality, data={
            "locality": locality,
            "family": data.get("family"),
            "locality_status": data.get("locality_status"),
            "assignment_reason": data.get("assignment_reason"),
            "locality_path": list(data.get("locality_path", []) or []),
            "locality_depth": data.get("locality_depth"),
            "parent_locality": data.get("parent_locality"),
            "replay_restored_from": "artifact_assigned_to_locality",
            "last_seen_step": int(data.get("step", eng.step) or 0),
        })
    else:
        loc = None
    artifact_type = str(data.get("artifact_type") or "")
    artifact_key = data.get("artifact_key")
    if artifact_type and artifact_key:
        aliases = getattr(eng, "_replay_id_alias", {})
        artifact = eng.graph.find_node_by_key(artifact_type, str(artifact_key))
        if artifact is None and str(artifact_key) in aliases:
            artifact = eng.graph.find_node_by_key(artifact_type, aliases[str(artifact_key)])
        if artifact is None:
            artifact = _restore_node_from_log(
                eng, node_type=artifact_type, key_fields=("artifact_key",), data=data
            )
        if artifact is None:
            return
        artifact.data.update({
            "locality": locality or artifact.data.get("locality"),
            "family": data.get("family", artifact.data.get("family")),
            "locality_status": data.get("locality_status", artifact.data.get("locality_status")),
            "locality_assignment_reason": data.get("assignment_reason"),
        })
        # Causal reconstruction already materializes assignment edges for
        # endogenous artifacts. The packet supplies exact metadata only; adding
        # another edge here would duplicate or cross-wire assignments.



def _restore_locality_proposal(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    key = str(data.get("proposal_key") or data.get("locality") or "")
    if not key:
        return
    locality = str(data.get("locality") or key)
    parent = str(data.get("parent_locality") or "root::unknown")
    proposal = eng.graph.upsert_node("locality_proposal", key=key, data={
        **dict(data or {}),
        "status": data.get("status", "proposed"),
        "ontology_origin": "experience_proposed",
        "replay_restored_from": "locality_region_proposed",
    })
    region = eng.graph.upsert_node("locality_region", key=locality, data={
        "locality": locality,
        "family": data.get("family"),
        "locality_status": data.get("proposal_kind", "proposed_family"),
        "parent_locality": parent,
        "ontology_origin": "experience_proposed",
        "last_seen_step": int(data.get("step", eng.step) or 0),
    })
    parent_node = eng.graph.find_node_by_key("locality_region", parent)
    if parent_node:
        eng.graph.add_edge(parent_node.id, region.id, "contains_locality", confidence=float(data.get("confidence", 0.5) or 0.5), data={"step": int(data.get("step", eng.step) or 0), "replay_restore": True})
    # The original runtime records the proposal and region but does not always
    # materialize a proposal edge. Preserve graph parity; the packet itself
    # remains the canonical provenance record.


def _restore_locality_state_change(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    locality = str(data.get("locality") or "")
    if not locality:
        return
    region = eng.graph.find_node_by_key("locality_region", locality) or eng.graph.upsert_node("locality_region", key=locality, data={})
    region.data.update({
        "family": data.get("family", region.data.get("family")),
        "locality_status": data.get("new_status", "existing_learned"),
        "established_step": data.get("step"),
        "evidence_seen_count": data.get("seen_count"),
        "parent_locality": data.get("parent_locality", region.data.get("parent_locality")),
        "ontology_origin": "experience_validated",
        "replay_restored_from": "locality_region_state_changed",
    })



def _restore_hypothesis_locality_designation(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    hid = str(data.get("hypothesis_id") or "")
    if not hid:
        return
    node = eng.graph.find_node_by_key("developmental_hypothesis", hid) or eng.graph.upsert_node("developmental_hypothesis", key=hid, data={})
    designation = dict(data.get("designation", {}) or {})
    node.data.update({
        "topology_designation": designation,
        "model_topology_designation_raw": dict(data.get("raw_designation", {}) or {}),
        "topology_designation_source": data.get("source", "model_proposed_substrate_adjudicated"),
        "family": designation.get("family", node.data.get("family")),
        "locality": designation.get("locality", node.data.get("locality")),
        "locality_confidence": designation.get("confidence", node.data.get("locality_confidence", 0.45)),
        "replay_restored_from": "hypothesis_locality_designated",
    })

def _restore_governance_proposal(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    node = _restore_node_from_log(eng, node_type="governance_proposal", key_fields=("proposal_id", "id"), data=data)
    if not node:
        return
    node.data.setdefault("status", data.get("status", "proposed"))
    node.data.setdefault("activation_state", data.get("activation_state", "inactive"))
    node.data.setdefault("durable", bool(data.get("durable", False)))
    node.data.setdefault("verified", bool(data.get("verified", False)))
    cid = data.get("candidate_id") or data.get("source_candidate_id")
    if cid:
        cand = eng.graph.find_node_by_key("candidate", str(cid))
        if cand:
            eng.graph.add_edge(node.id, cand.id, "proposed_from", confidence=float(data.get("confidence", 0.6) or 0.6), data={"step": int(data.get("step", eng.step) or 0), "replay_restore": True})


def _restore_governance_promotion_eval(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    pid = str(data.get("proposal_id") or data.get("id") or "")
    if not pid:
        return
    aliases = getattr(eng, "_replay_id_alias", {})
    node = eng.graph.find_node_by_key("governance_proposal", pid)
    if node is None and pid in aliases:
        node = eng.graph.find_node_by_key("governance_proposal", aliases[pid])
    if node is None:
        node = _restore_node_from_log(eng, node_type="governance_proposal", key_fields=("proposal_id", "id"), data=data)
    if node is None:
        return
    node.data.update({
        "status": data.get("decision", node.data.get("status")),
        "promotion_status": data.get("promotion_status", node.data.get("promotion_status")),
        "promotion_evidence_score": data.get("evidence_score"),
        "last_promotion_eval_step": data.get("step", eng.step),
        "promotion_reasons": data.get("reasons", []),
        "promotion_replay_packet": dict(data or {}),
    })


def _restore_collation_run(eng: YggdrasilEngine, data: Dict[str, Any], *, completed: bool) -> None:
    run_id = str(data.get("run_id") or data.get("id") or "")
    if not run_id:
        return
    payload = dict(data or {})
    payload.setdefault("status", "complete" if completed else "running")
    payload.setdefault("replay_restored_from", "collation_period_completed" if completed else "collation_period_started")
    eng.graph.upsert_node("collation_run", key=run_id, data=payload)


def _restore_developmental_hypothesis(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    hid = str(data.get("hypothesis_id") or data.get("id") or "")
    if not hid:
        return
    payload = dict(data or {})
    payload.setdefault("artifact_type", "developmental_hypothesis")
    payload.setdefault("status", "hypothesis")
    payload.setdefault("authority_state", "advisory_hypothesis_only")
    node = eng.graph.upsert_node("developmental_hypothesis", key=hid, data=payload)
    for cid in payload.get("support_candidate_ids", []) or []:
        cand = eng.graph.find_node_by_key("candidate", str(cid))
        if cand:
            eng.graph.add_edge(node.id, cand.id, "supported_by", confidence=float(payload.get("confidence", 0.5) or 0.5), data={"step": int(payload.get("step", eng.step) or 0), "replay_restore": True})



def _restore_recurrent_episode(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    rid = str(data.get("recurrent_episode_id") or data.get("id") or "")
    if not rid:
        return
    payload = dict(data or {})
    payload.setdefault("status", "provisional_meso_episode")
    payload.setdefault("created_step", int(payload.get("step", eng.step) or 0))
    node = eng.graph.upsert_node("recurrent_episode", key=rid, data=payload)
    for cid in payload.get("selected_candidate_ids", []) or payload.get("support_candidate_ids", []) or []:
        cand = eng.graph.find_node_by_key("candidate", str(cid))
        if cand:
            eng.graph.add_edge(node.id, cand.id, "supported_by", confidence=0.7, data={"step": int(payload.get("step", eng.step) or 0), "replay_restore": True})
    for iid in payload.get("selected_investigation_ids", []) or payload.get("support_investigation_ids", []) or []:
        inv = eng.graph.find_node_by_key("investigation", str(iid))
        if inv:
            eng.graph.add_edge(node.id, inv.id, "supported_by", confidence=0.7, data={"step": int(payload.get("step", eng.step) or 0), "replay_restore": True})

def _restore_candidate_utility(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    cid = str(data.get("candidate_id") or "")
    if not cid:
        return
    node = eng.graph.find_node_by_key("candidate", cid)
    if node is None:
        aliases = getattr(eng, "_replay_id_alias", {})
        if cid in aliases:
            node = eng.graph.find_node_by_key("candidate", aliases[cid])
    if node is None:
        return
    for key in ("utility_score", "use_count", "positive_use_count", "negative_use_count", "transfer_success_count", "failed_reuse_count", "success_count", "salience"):
        if key in data:
            node.data[key] = data.get(key)
    node.data["last_utility_step"] = int(data.get("step", eng.step) or 0)


def _restore_success_anchor_regression(eng: YggdrasilEngine, data: Dict[str, Any]) -> None:
    cid = str(data.get("anchor_id") or "")
    node = eng.graph.find_node_by_key("candidate", cid) if cid else None
    if node is None:
        return
    if "regression_count" in data:
        node.data["regression_count"] = data.get("regression_count")
    if "missed_high_similarity_count" in data:
        node.data["missed_high_similarity_count"] = data.get("missed_high_similarity_count")


def _restore_lifecycle_packet(eng: YggdrasilEngine, typ: str, data: Dict[str, Any]) -> bool:
    """Restore non-event developmental artifacts that materially affect resume context."""
    if typ == "investigation_workspace_created":
        _restore_investigation_workspace(eng, data); return True
    if typ == "provisional_outcome_written":
        _restore_provisional_outcome(eng, data); return True
    if typ == "candidate_cached":
        _restore_candidate(eng, data); return True
    if typ == "candidate_pedigree_evaluated":
        _restore_candidate_pedigree(eng, data); return True
    if typ in {"candidate_utility_updated", "successful_action_anchor_reinforced"}:
        _restore_candidate_utility(eng, data); return True
    if typ == "successful_anchor_regression_observed":
        _restore_success_anchor_regression(eng, data); return True
    if typ == "artifact_assigned_to_locality":
        _restore_locality_assignment(eng, data); return True
    if typ == "locality_region_proposed":
        _restore_locality_proposal(eng, data); return True
    if typ == "locality_region_state_changed":
        _restore_locality_state_change(eng, data); return True
    if typ == "developmental_hypothesis_synthesized":
        _restore_developmental_hypothesis(eng, data); return True
    if typ == "recurrent_episode_collated":
        _restore_recurrent_episode(eng, data); return True
    if typ == "hypothesis_locality_designated":
        _restore_hypothesis_locality_designation(eng, data); return True
    if typ == "governance_proposal_created":
        _restore_governance_proposal(eng, data); return True
    if typ in {"governance_proposal_promotion_evaluated", "governance_promotion_evaluated"}:
        _restore_governance_promotion_eval(eng, data); return True
    if typ == "collation_period_started":
        _restore_collation_run(eng, data, completed=False); return True
    if typ in {"collation_period_completed", "collation_period_complete"}:
        _restore_collation_run(eng, data, completed=True); return True
    return False


class ReplayReducer:
    """Canonical reducer shared by verification and resume."""

    def __init__(self, engine: YggdrasilEngine) -> None:
        self.engine = engine
        self.api = SubstrateAPI(engine)
        self.pending = False
        self.developmental_packets: List[Dict[str, Any]] = []
        self.lifecycle_queue: List[tuple[str, Dict[str, Any]]] = []
        self.applied_counts: Dict[str, int] = {}
        self.last_runtime_state: Dict[str, Any] = {}
        self.finish_diagnostics: List[Dict[str, Any]] = []
        self.start_step_count = 0
        self.engine.start_step()
        self.start_step_count += 1

    def _count(self, typ: str) -> None:
        self.applied_counts[typ] = self.applied_counts.get(typ, 0) + 1

    def _flush_lifecycle(self) -> None:
        # Causal APIs first reconstruct endogenous effects. Logged packets then
        # merge exact identities/content without influencing causal generation.
        for typ, data in self.lifecycle_queue:
            if typ == "pre_answer_investigation_created":
                _restore_pre_answer_investigation(self.engine, data)
            elif typ == "action_rationale_recorded":
                _restore_action_rationale(self.engine, data)
            else:
                _restore_lifecycle_packet(self.engine, typ, data)
        self.lifecycle_queue.clear()

    def apply(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        typ = str(row.get("type") or "")
        data = dict(row.get("data", {}) or {})
        self._count(typ)
        if typ == "event_written":
            self.api.record_event(role=data.get("role", "user"), text=data.get("text", ""),
                                  source=data.get("source", "replay"), provenance=data.get("provenance", {}) or {},
                                  event_id=data.get("event_id"))
            self.pending = True
        elif typ == "outcome_written":
            self.api.log_outcome(score=float(data.get("score", 0.0)), label=str(data.get("label", "") or ""),
                                 tags=data.get("tags", []), outcome_id=data.get("outcome_id"))
            self.pending = True
        elif typ == "developmental_event_written":
            self.developmental_packets.append(data)
        elif typ in {"pre_answer_investigation_created", "action_rationale_recorded"} or typ in {
            "investigation_workspace_created", "provisional_outcome_written", "candidate_cached",
            "candidate_pedigree_evaluated", "artifact_assigned_to_locality", "locality_region_proposed",
            "locality_region_state_changed", "developmental_hypothesis_synthesized", "recurrent_episode_collated",
            "hypothesis_locality_designated", "governance_proposal_created",
            "governance_proposal_promotion_evaluated", "governance_promotion_evaluated",
            "collation_period_started", "collation_period_completed", "collation_period_complete",
            "candidate_utility_updated", "successful_action_anchor_reinforced",
            "successful_anchor_regression_observed",
        }:
            self.lifecycle_queue.append((typ, data)); self.pending = True
        elif typ == "state_hash":
            return self.finish_step(recorded=data)
        return None

    def finish_step(self, recorded: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.pending:
            return None
        # Reconstruct endogenous causal effects first. Lifecycle packets are
        # exact audit overlays and must not influence episode/candidate creation.
        preconditions = replay_precondition_snapshot(self.engine)
        recorded_runtime = dict((recorded or {}).get("runtime_state", {}) or {})
        self.api.end_step()
        self._flush_lifecycle()
        removed_duplicate_edges = deduplicate_logical_edges(self.engine)
        runtime = recorded_runtime
        if runtime:
            self.last_runtime_state = dict(runtime)
            self.engine._feature_counts = {str(k): int(v) for k, v in (runtime.get("feature_counts", {}) or {}).items()}
            self.engine._feature_support = {str(k): float(v) for k, v in (runtime.get("feature_support", {}) or {}).items()}
            self.engine._last_collation_step = int(runtime.get("last_collation_step", self.engine._last_collation_step) or 0)
            self.engine.ablate_governance = bool(runtime.get("ablate_governance", self.engine.ablate_governance))
            self.engine.candidate_cache_ttl_steps = int(runtime.get("candidate_cache_ttl_steps", self.engine.candidate_cache_ttl_steps) or self.engine.candidate_cache_ttl_steps)
            self.engine.max_candidate_cache = int(runtime.get("max_candidate_cache", self.engine.max_candidate_cache) or self.engine.max_candidate_cache)
            self.engine.protected_candidate_ttl_steps = int(runtime.get("protected_candidate_ttl_steps", self.engine.protected_candidate_ttl_steps) or self.engine.protected_candidate_ttl_steps)
            self.engine.auto_collation_interval_steps = runtime.get("auto_collation_interval_steps", self.engine.auto_collation_interval_steps)
        merged_hash, summary = compute_state_hash(self.engine.graph)
        actual_payload = canonical_state_payload(self.engine.graph)
        diagnostic = {
            "step": int(self.engine.step),
            "preconditions": preconditions,
            "recorded_runtime_state": recorded_runtime,
            "recorded_diagnostic_config": dict((recorded or {}).get("diagnostic_config", {}) or {}),
            "removed_duplicate_edges": int(removed_duplicate_edges),
            "replay_start_step_count": int(self.start_step_count),
            "actual_summary": summary,
        }
        self.finish_diagnostics.append(diagnostic)
        result = {"step": int(self.engine.step), "hash": merged_hash,
                  "recorded": dict(recorded or {}),
                  "canonical_state": actual_payload,
                  "diagnostic": diagnostic}
        self.engine.start_step(); self.start_step_count += 1; self.pending = False
        self.engine._replay_claimed_nodes = set()  # type: ignore[attr-defined]
        return result

    def finalize(self) -> Optional[Dict[str, Any]]:
        result = self.finish_step()
        self.engine._replay_developmental_packets = list(self.developmental_packets)  # type: ignore[attr-defined]
        self.engine._replay_applied_counts = dict(self.applied_counts)  # type: ignore[attr-defined]
        rebuild_runtime_indexes(self.engine)
        if self.last_runtime_state:
            runtime = self.last_runtime_state
            self.engine._feature_counts = {str(k): int(v) for k, v in (runtime.get("feature_counts", {}) or {}).items()}
            self.engine._feature_support = {str(k): float(v) for k, v in (runtime.get("feature_support", {}) or {}).items()}
            self.engine._last_collation_step = int(runtime.get("last_collation_step", self.engine._last_collation_step) or 0)
        return result


def deduplicate_logical_edges(eng: YggdrasilEngine) -> int:
    """Remove replay-overlay duplicates while preserving the causal edge."""
    seen = set()
    removed = 0
    for edge_id, edge in list(eng.graph.edges.items()):
        key = (edge.type, edge.src, edge.dst)
        if key in seen:
            if eng.graph.delete_edge(edge_id):
                removed += 1
        else:
            seen.add(key)
    return removed


def rebuild_runtime_indexes(eng: YggdrasilEngine) -> None:
    """Rebuild operational indexes that affect retrieval and continuation."""
    def step_of(node: Any, *keys: str) -> int:
        for key in keys:
            try:
                if node.data.get(key) is not None:
                    return int(node.data.get(key) or 0)
            except Exception:
                pass
        return 0

    episodes = list(eng.graph.find_nodes("episode"))
    events = list(eng.graph.find_nodes("event"))
    candidates = list(eng.graph.find_nodes("candidate"))
    if not getattr(eng, "_episode_order", None):
        eng._episode_order = [str(n.key) for n in sorted(episodes, key=lambda n: (step_of(n, "opened_step", "created_step"), str(n.key)))]
    if not getattr(eng, "_event_order", None):
        eng._event_order = [str(n.key) for n in sorted(events, key=lambda n: (step_of(n, "created_step"), str(n.key)))]
    if not getattr(eng, "_candidate_order", None):
        eng._candidate_order = [str(n.key) for n in sorted(candidates, key=lambda n: (step_of(n, "created_step", "step"), str(n.key)))]
    open_eps = [n for n in episodes if str(n.data.get("status", "")) == "open"]
    eng._current_episode_id = str(sorted(open_eps, key=lambda n: (step_of(n, "opened_step"), str(n.key)))[-1].key) if open_eps else None
    eng._feature_counts = {}
    eng._feature_support = {}
    for node in eng.graph.find_nodes("feature"):
        key = str(node.key)
        eng._feature_counts[key] = int(node.data.get("count_seen", node.data.get("count", 1)) or 1)
        eng._feature_support[key] = float(node.data.get("support", 0.0) or 0.0)
    runs = list(eng.graph.find_nodes("collation_run"))
    eng._last_collation_step = max([step_of(n, "completed_step", "step") for n in runs] or [0])


class SessionStore:
    """Manage canonical session paths and reconstruction from event logs."""

    def __init__(self, root: str = "sessions") -> None:
        self.root = root

    def paths(self, session_id: str) -> SessionPaths:
        p = SessionPaths(root=self.root, session_id=session_id)
        p.ensure()
        return p

    def logger(self, session_id: str) -> JsonlLogger:
        return JsonlLogger(self.paths(session_id).events_jsonl)

    def rebuild_engine(
        self,
        session_id: str,
        *,
        adjudicator: Optional[HybridAdjudicator] = None,
        source_log: Optional[str] = None,
        logger: Optional[Any] = None,
        enable_decay: bool = True,
        restore_lifecycle_packets: bool = True,
    ) -> YggdrasilEngine:
        """Rebuild complete developmental state through the canonical reducer."""
        p = self.paths(session_id)
        rows = read_jsonl(source_log or p.events_jsonl)
        eng = YggdrasilEngine(
            session_id=session_id, logger=logger or NullLogger(),
            adjudicator=adjudicator or HybridAdjudicator(), enable_decay=bool(enable_decay),
        )
        reducer = ReplayReducer(eng)
        for row in rows:
            # The argument remains for legacy callers; full lifecycle restoration
            # is now the canonical contract and cannot diverge from verification.
            reducer.apply(row)
        reducer.finalize()
        return eng

    def resume_engine(
        self,
        session_id: str,
        *,
        adjudicator: Optional[HybridAdjudicator] = None,
        append_log_path: Optional[str] = None,
        verify: bool = True,
        enable_decay: bool = True,
    ) -> YggdrasilEngine:
        """
        Rebuild prior state and attach an append logger so experimentation can continue.
        """
        p = self.paths(session_id)
        report = self.verify_session(session_id, adjudicator=adjudicator, enable_decay=enable_decay) if verify else None
        eng = self.rebuild_engine(
            session_id,
            adjudicator=adjudicator,
            logger=NullLogger(),
            enable_decay=enable_decay,
            restore_lifecycle_packets=True,
        )
        eng.log = JsonlLogger(append_log_path or p.events_jsonl)
        if report:
            eng._last_resume_report = report  # type: ignore[attr-defined]
        return eng

    def verify_session(
        self,
        session_id: str,
        *,
        adjudicator: Optional[HybridAdjudicator] = None,
        source_log: Optional[str] = None,
        write_report: bool = True,
        enable_decay: bool = True,
    ) -> Dict[str, Any]:
        """Verify the same full reconstruction contract used by resume."""
        p = self.paths(session_id)
        rows = read_jsonl(source_log or p.events_jsonl)
        boundary_diagnostics = boundary_audit(rows)
        expected: Dict[int, str] = {}
        expected_payloads: Dict[int, Dict[str, Any]] = {}
        expected_rows: List[Dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            if row.get("type") == "state_hash":
                d = row.get("data", {}) or {}
                if "step" in d and "hash" in d:
                    step = int(d["step"])
                    expected_rows.append({"row_index": row_index, "step": step, "hash": str(d["hash"])})
                    expected[step] = str(d["hash"])
                    if isinstance(d.get("diagnostic_canonical_state"), dict):
                        expected_payloads[step] = d.get("diagnostic_canonical_state") or {}

        eng = YggdrasilEngine(
            session_id=f"{session_id}-replay", logger=NullLogger(),
            adjudicator=adjudicator or HybridAdjudicator(), enable_decay=bool(enable_decay),
        )
        reducer = ReplayReducer(eng)
        replay_hashes: Dict[int, str] = {}
        first_mismatch: Optional[Dict[str, Any]] = None
        for row in rows:
            result = reducer.apply(row)
            if result:
                step = int(result["step"])
                h = str(result.get("hash") or "")
                replay_hashes[step] = h
                exp = expected.get(step)
                if exp != h and first_mismatch is None:
                    first_mismatch = {
                        "step": step, "expected": exp, "replayed": h,
                        "source_state_hash_row": next((x for x in expected_rows if x.get("step") == step), None),
                        "replay_diagnostic": result.get("diagnostic") or {},
                        "canonical_diff": canonical_payload_diff(expected_payloads.get(step), result.get("canonical_state") or {}),
                        "boundary": (boundary_diagnostics.get("boundaries") or {}).get(str(step), {}),
                    }
        tail = reducer.finalize()
        if tail and tail.get("hash"):
            replay_hashes[int(tail["step"])] = str(tail["hash"])

        ok = first_mismatch is None and expected == replay_hashes
        final_hash, final_summary = compute_state_hash(eng.graph)
        from .state_hash import compute_runtime_state_hash
        runtime_hash, runtime_payload = compute_runtime_state_hash(eng)
        report = {
            "ok": bool(ok), "session_id": session_id,
            "source_log": source_log or p.events_jsonl,
            "expected_steps": len(expected), "replayed_steps": len(replay_hashes),
            "first_mismatch": first_mismatch, "final_hash": final_hash,
            "final_summary": final_summary, "runtime_state_hash": runtime_hash,
            "runtime_state": runtime_payload,
            "developmental_packets_restored": len(reducer.developmental_packets),
            "reducer_applied_counts": reducer.applied_counts,
            "reducer_finish_diagnostics": reducer.finish_diagnostics,
            "causal_boundary_audit": boundary_diagnostics,
            "state_hash_rows": expected_rows,
            "duplicate_state_hash_steps": boundary_diagnostics.get("duplicate_step_numbers", {}),
            "reconstruction_contract": "full_lifecycle_v2+causal_diagnostics_v1",
        }
        if write_report:
            write_json(p.replay_report_json, report)
            with open(p.replay_report_md, "w", encoding="utf-8") as f:
                f.write(format_replay_report(report))
        return report

    def write_snapshot(self, session_id: str, eng: YggdrasilEngine) -> Dict[str, Any]:
        """Write a convenience snapshot and graph export for human inspection."""
        p = self.paths(session_id)
        h, summary = compute_state_hash(eng.graph)
        payload = {
            "session_id": session_id,
            "step": eng.step,
            "state_hash": h,
            "summary": summary,
            "hud": eng.hud(),
            "canonical_state": canonical_state_payload(eng.graph),
        }
        write_json(p.snapshot_json, payload)
        write_json(p.graph_json, eng.graph.to_canonical_dict())
        with open(p.graph_dot, "w", encoding="utf-8") as f:
            f.write(eng.graph.to_dot())
        return payload


def format_replay_report(report: Dict[str, Any]) -> str:
    status = "PASS" if report.get("ok") else "FAIL"
    lines = [
        f"# Replay Report — {status}",
        "",
        f"session_id: `{report.get('session_id')}`",
        f"source_log: `{report.get('source_log')}`",
        f"expected_steps: {report.get('expected_steps')}",
        f"replayed_steps: {report.get('replayed_steps')}",
        f"final_hash: `{report.get('final_hash')}`",
        f"final_summary: `{report.get('final_summary')}`",
    ]
    if report.get("first_mismatch"):
        lines += ["", "## First mismatch", "", "```json", json.dumps(report["first_mismatch"], indent=2, sort_keys=True), "```"]
    else:
        lines += ["", "Replay hash sequence matched recorded state hashes."]
    return "\n".join(lines) + "\n"

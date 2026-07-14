from __future__ import annotations

"""Evidence-validated developmental hypothesis synthesis for collation.

This module is intentionally deterministic. It provides the substrate with an
intermediate layer between raw episodic candidates and active governance:

candidate memories -> evidence clusters -> developmental hypotheses -> future validation -> governance

A hypothesis is not a policy. It is a falsifiable developmental claim with scope,
expected benefit, rejection conditions, and explicit evidence provenance.
"""

import hashlib
import re
from typing import Any, Dict, Iterable, List, Tuple


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _slug(text: str, max_len: int = 64) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).lower()).strip("_")
    if not s:
        s = "general"
    return s[:max_len].strip("_") or "general"


def _stable_short_hash(parts: Iterable[str], n: int = 10) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:n]


def infer_evidence_polarity(data: Dict[str, Any]) -> str:
    text = str(data.get("text", "")).lower()
    if "objective_success" in text or "success_reinforcement" in text or "passed" in text and "failed" not in text:
        return "success"
    if "objective_failure" in text or "failure_repair" in text or "assertionerror" in text or "syntaxerror" in text or "nameerror" in text or "typeerror" in text:
        return "failure"
    label = str(data.get("label", data.get("outcome_label", ""))).lower()
    if "pass" in label or "success" in label:
        return "success"
    if "fail" in label or "error" in label:
        return "failure"
    return "mixed"


def infer_failure_signature(data: Dict[str, Any]) -> str:
    text = str(data.get("text", "")).lower()
    features = " ".join(map(str, data.get("feature_keys", []) or [])).lower()
    blob = f"{text} {features}"
    if "syntaxerror" in blob:
        return "syntax_error"
    if "nameerror" in blob or "not defined" in blob:
        return "undefined_name"
    if "typeerror" in blob:
        return "type_mismatch"
    if "assertionerror" in blob:
        if any(x in blob for x in ["empty", "[]", "boundary", "edge case", "zero", "one"]):
            return "assertion_boundary_case"
        if any(x in blob for x in ["sort", "order", "ordered"]):
            return "assertion_ordering_contract"
        if any(x in blob for x in ["string", "substring", "subsequence", "character"]):
            return "assertion_string_contract"
        return "assertion_contract_mismatch"
    if infer_evidence_polarity(data) == "success":
        return "success_contract_match"
    return "general_behavioral_delta"


def cluster_key_for_candidate(candidate_id: str, data: Dict[str, Any]) -> Tuple[str, str, str, str]:
    family = str(data.get("family") or "general_constraint")
    locality = str(data.get("locality") or f"family::{family}")
    polarity = infer_evidence_polarity(data)
    signature = infer_failure_signature(data)
    return (family, locality, polarity, signature)


def build_evidence_clusters(
    candidates: Iterable[Tuple[str, Dict[str, Any]]],
    *,
    min_evidence: int = 2,
    max_clusters: int = 12,
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str, str], List[Tuple[str, Dict[str, Any]]]] = {}
    for cid, data in candidates:
        if str(data.get("status", "")) not in {"cached", "staged", "variant_cached"} and str(data.get("pedigree_status", "")) != "evaluated":
            continue
        key = cluster_key_for_candidate(cid, data)
        buckets.setdefault(key, []).append((str(cid), dict(data)))

    rows: List[Dict[str, Any]] = []
    for (family, locality, polarity, signature), items in buckets.items():
        if len(items) < int(min_evidence):
            continue
        support_ids = [cid for cid, _ in items]
        recurrence = sum(max(1, _as_int(d.get("recurrence_count", 1), 1)) for _, d in items)
        variants = sum(max(0, _as_int(d.get("variant_count", 0), 0)) for _, d in items)
        usefulness = max((_as_float(d.get("usefulness_score", 0.0)) for _, d in items), default=0.0)
        locality_conf = max((_as_float(d.get("locality_confidence", 0.0)) for _, d in items), default=0.0)
        locality_statuses = [str(d.get("locality_status", "")) for _, d in items]
        locality_paths = [list(d.get("locality_path", []) or []) for _, d in items if d.get("locality_path")]
        evidence_count = len(items) + variants + max(0, recurrence - len(items))
        score = round(min(1.0, 0.22 * len(items) + 0.10 * variants + 0.08 * recurrence + 0.25 * usefulness + 0.15 * locality_conf), 4)
        key_slug = _slug("__".join([family, locality, polarity, signature]))
        hid = f"hypothesis::{key_slug}::{_stable_short_hash([family, locality, polarity, signature])}"
        rows.append({
            "hypothesis_id": hid,
            "family": family,
            "locality": locality,
            "polarity": polarity,
            "signature": signature,
            "support_candidate_ids": support_ids[:24],
            "evidence_count": int(evidence_count),
            "candidate_count": len(items),
            "variant_count": int(variants),
            "recurrence_count": int(recurrence),
            "usefulness_score": round(usefulness, 4),
            "locality_confidence": round(locality_conf, 4),
            "locality_status": next((x for x in locality_statuses if x), "held_unclassified"),
            "locality_path": max(locality_paths, key=len) if locality_paths else ["root::unknown"],
            "topology_designation": {
                "action": "attach_existing" if any(x.startswith("existing") for x in locality_statuses) else ("propose_subfamily" if "proposed_subfamily" in locality_statuses else ("propose_family" if "proposed_family" in locality_statuses else "hold_unclassified")),
                "family": family,
                "locality": locality,
                "highest_matching_level": locality,
            },
            "confidence": score,
        })
    rows.sort(key=lambda r: (-float(r.get("confidence", 0.0)), -int(r.get("evidence_count", 0)), str(r.get("hypothesis_id"))))
    return rows[: int(max_clusters)]


def synthesize_hypothesis_payload(cluster: Dict[str, Any]) -> Dict[str, Any]:
    family = str(cluster.get("family", "general_constraint"))
    locality = str(cluster.get("locality", f"family::{family}"))
    polarity = str(cluster.get("polarity", "mixed"))
    signature = str(cluster.get("signature", "general_behavioral_delta"))
    evidence_count = int(cluster.get("evidence_count", 0) or 0)
    confidence = float(cluster.get("confidence", 0.0) or 0.0)

    if family in {"python_function_correctness", "coding_contract_repair", "coding"}:
        if polarity == "failure":
            if signature == "syntax_error":
                observation = "Repeated coding failures include non-executable or syntactically invalid outputs."
                lesson = "Executable validity must be checked before a solution is treated as a final answer."
                procedure = "Before committing, prefer a minimal complete function body and avoid explanatory text inside code output."
                reject_if = "Future executable-code episodes in this locality pass despite skipping syntax validation."
            elif signature == "undefined_name":
                observation = "Repeated coding failures involve names or helpers that are not defined in the submitted solution."
                lesson = "All helper functions, imports, and variables required by the contract must be locally defined."
                procedure = "Scan the candidate for undeclared helpers/imports before final commitment."
                reject_if = "Future failures stop involving undefined names after this pattern is no longer retrieved."
            elif signature == "type_mismatch":
                observation = "Repeated coding failures involve mismatched input/output types or unsupported operations."
                lesson = "The implementation must preserve the exact type contract implied by the signature and examples."
                procedure = "Check expected input and return types against examples and hidden-boundary possibilities before final commitment."
                reject_if = "Future type-sensitive tasks pass without type-contract checks."
            elif signature == "assertion_boundary_case":
                observation = "Repeated assertion failures appear tied to boundary or edge-case behavior."
                lesson = "Boundary cases are a recurring source of incorrect Python function behavior."
                procedure = "Test or mentally simulate empty, singleton, zero, duplicate, and extremal cases before committing."
                reject_if = "Future boundary-heavy tasks pass reliably without explicit boundary review."
            else:
                observation = "Repeated assertion failures indicate mismatch between plausible implementation and exact task contract."
                lesson = "A plausible solution is not enough; the candidate must be checked against the precise behavioral contract."
                procedure = "Compare the candidate against all examples, infer hidden contract constraints, and revise before final commitment."
                reject_if = "Future assertion failures are not reduced by contract-focused revision."
        elif polarity == "success":
            observation = "Successful coding episodes share contract-faithful executable implementations."
            lesson = "Simple implementations that directly match the signature, examples, and boundary implications should be preserved."
            procedure = "Favor minimal executable code with locally defined helpers and behavior traceable to the prompt contract."
            reject_if = "Future tasks in this locality fail because this simplicity bias ignores required complexity."
        else:
            observation = "Mixed coding evidence suggests an unresolved behavioral pattern requiring future validation."
            lesson = "Do not promote this pattern to governance until future episodes clarify whether it predicts success or failure."
            procedure = "Track future similar episodes and compare retrieved guidance against objective pass/fail outcomes."
            reject_if = "Future evidence splits evenly across success and failure with no identifiable differentiator."
    else:
        observation = f"Repeated episodes in {family} show a recurring {polarity} pattern: {signature}."
        lesson = "Treat recurring evidence as a provisional developmental hypothesis, not a durable rule."
        procedure = "Use the hypothesis as advisory context only when future situations match its locality and evidence signature."
        reject_if = "Future matched episodes contradict the predicted improvement or show the scope was too broad."

    scope = f"Apply only to {locality} when evidence signature resembles {signature}."
    claim = f"In {locality}, {signature} is a recurring {polarity} developmental signal supported by {evidence_count} evidence units."
    return {
        "claim": claim,
        "observation": observation,
        "lesson": lesson,
        "procedure": procedure,
        "scope": scope,
        "reject_if": reject_if,
        "expected_next_change": "Future attempts should show fewer repeated failures or cleaner first-pass behavior when this hypothesis is relevant.",
        "artifact_type": "developmental_hypothesis",
        "status": "hypothesis",
        "validation_state": "needs_future_validation" if confidence < 0.78 or evidence_count < 4 else "validation_candidate",
        "governance_ready": bool(confidence >= 0.78 and evidence_count >= 4),
        "authority_state": "advisory_hypothesis_only",
        "confidence": round(confidence, 4),
        "topology_designation": dict(cluster.get("topology_designation", {}) or {}),
        "locality_status": cluster.get("locality_status", "held_unclassified"),
        "locality_path": list(cluster.get("locality_path", []) or []),
    }


def governance_text_from_hypothesis(data: Dict[str, Any]) -> str:
    family = str(data.get("family", "general_constraint"))
    locality = str(data.get("locality", f"family::{family}"))
    procedure = str(data.get("procedure") or data.get("lesson") or data.get("claim") or "Use validated evidence cautiously.")
    scope = str(data.get("scope") or locality)
    reject_if = str(data.get("reject_if") or "future matched evidence contradicts this directive")
    return (
        f"For future situations routed to {family} at {locality}, apply this validated developmental procedure: "
        f"{procedure} Scope: {scope} Reconsider if: {reject_if}"
    )[:900]

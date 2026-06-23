"""
Core substrate memory engine for Yggdrasil.

This module contains two primary layers:

1. HybridAdjudicator
   Deterministic-first extraction of structured signals from raw text,
   with optional bounded LLM-assisted fill when structured extraction is sparse.

2. YggdrasilEngine
   Runtime memory substrate that stores events, features, concepts, episodes,
   outcomes, and traceable support signals in a graph. The engine is responsible
   for:
   - event and outcome recording
   - extraction ingestion
   - concept promotion
   - detail decay and compression
   - deterministic context assembly
   - replay / audit compatibility

Design goals:
- deterministic and inspectable behavior
- bounded memory growth
- immediate usability for interactive agent loops
- traceable runtime adaptation without model retraining
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .context_schema import build_context_bundle, serialize_context
from .graph import Graph
from .state_hash import compute_state_hash
from .storage import JsonlLogger

# ------------------------------------------------------------------
# Packet / extraction data structures
# ------------------------------------------------------------------

@dataclass
class EventPacket:
    """Serializable event payload recorded into the substrate."""
    event_id: str
    session_id: str
    ts: float
    role: str
    text: str
    source: str = "cli"
    provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutcomePacket:
    """Serializable scalar outcome attached to the current episode context."""
    outcome_id: str
    session_id: str
    ts: float
    score: float
    label: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = "user"

@dataclass
class FeatureItem:
    """Structured feature extracted from an event."""
    key: str
    kind: str
    confidence: float = 0.6
    value: Optional[str] = None
    evidence_span: Optional[Tuple[int,int]] = None

@dataclass
class RelationItem:
    """Typed relation between two extracted features."""
    src: str
    rel_type: str
    dst: str
    confidence: float = 0.5

@dataclass
class EpisodeSignal:
    """Heuristic signal describing whether an event starts, continues, or ends an episode."""
    boundary: str
    confidence: float = 0.5

@dataclass
class ExtractionRecord:
    """Complete adjudication result for a single event."""
    event_id: str
    intent: str
    features: List[FeatureItem] = field(default_factory=list)
    relations: List[RelationItem] = field(default_factory=list)
    episode_signal: EpisodeSignal = field(default_factory=lambda: EpisodeSignal("unknown", 0.5))
    meta: Dict[str, Any] = field(default_factory=dict)

# ------------------------------------------------------------------
# Adjudication
# ------------------------------------------------------------------

class HybridAdjudicator:
    """
    Deterministic-first text adjudicator.

    This component extracts bounded structured signals from raw text using
    rule-based heuristics, then optionally augments sparse results with a
    validated LLM fill step.

    The adjudicator is intentionally conservative:
    - deterministic extraction is preferred
    - lexical fallback is bounded
    - LLM output is schema-validated before use
    """
    def __init__(
        self,
        llm_fill: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        enable_lexical_default: bool = False,
        lexical_fallback_min_structured: int = 2,
        max_features: int = 32,
        max_relations: int = 48,
    ) -> None:
        self.llm_fill = llm_fill
        self.enable_lexical_default = enable_lexical_default
        self.lexical_fallback_min_structured = lexical_fallback_min_structured
        self.max_features = max_features
        self.max_relations = max_relations
        self._entity_stop = set([
            "constraint","constraints","goal","goals","task","tasks","plan","plans",
            "episode","episodes","concept","concepts","memory","yggdrasil"
        ])

    def adjudicate(self, text: str, event_id: str, role: str="user", episode_open: bool=True) -> ExtractionRecord:
        intent = self._infer_intent(text)
        feats: List[FeatureItem] = []
        rels: List[RelationItem] = []

        feats += self._extract_goal(text)
        feats += self._extract_constraint(text)
        feats += self._extract_entities(text)
        feats += self._extract_taskish(text)

        structured = [f for f in feats if f.kind in ("goal","constraint","entity","task","preference")]

        allow_lexical = self.enable_lexical_default and role != "assistant"
        fallback = (len(structured) < self.lexical_fallback_min_structured) and role != "assistant"
        enable_lexical = allow_lexical or fallback
        if enable_lexical:
            feats += self._extract_lexical(text)

        keys = [f.key for f in structured][:8]
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                rels.append(RelationItem(src=keys[i], rel_type="rel:cooccur", dst=keys[j], confidence=0.5))

        if self.llm_fill and len(structured) < 3:
            llm_rec = self._validate_llm_fill(event_id, self.llm_fill({"text": text, "intent": intent, "role": role}))
            existing = {f.key for f in feats}
            for f in llm_rec.features:
                if f.key not in existing:
                    feats.append(f)
            rels.extend(llm_rec.relations)

        feats = feats[: self.max_features]
        rels = rels[: self.max_relations]

        boundary = "continue" if episode_open else "start"
        if re.search(r"\b(done|finished|that works|thanks)\b", text.lower()):
            boundary = "end"

        return ExtractionRecord(
            event_id=event_id,
            intent=intent,
            features=feats,
            relations=rels,
            episode_signal=EpisodeSignal(boundary, 0.55),
            meta={"structured_count": len(structured), "lexical_enabled": enable_lexical, "role": role},
        )

    def _infer_intent(self, t: str) -> str:
        s = t.strip().lower()
        if s.endswith("?"):
            return "question"
        if s.startswith("goal:") or s.startswith("constraint:"):
            return "statement"
        if "propose" in s or ("tasks" in s and "next" in s):
            return "plan"
        return "request"

    def _canon(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def _extract_goal(self, text: str) -> List[FeatureItem]:
        m = re.search(r"(?im)^\s*goal\s*(?::|::)\s*(.+)\s*$", text)
        if not m:
            return []
        v = m.group(1).strip()
        return [FeatureItem(key=f"goal::{self._canon(v)}", kind="goal", confidence=0.85, value=v)]

    def _extract_constraint(self, text: str) -> List[FeatureItem]:
        """
        Extract constraint-like statements from:
        1) explicit tags:  'constraint: ...'
        2) explicit variants: 'rule:', 'policy:', 'lesson:', 'guardrail:'
        3) natural language constraint sentences (don't / do not / avoid / must not / should not / never / always / prefer ...)
        Keeps output deterministic and bounded.
        """
        out: List[FeatureItem] = []
        seen_keys = set()

        def add(v: str, conf: float) -> None:
            v = (v or "").strip()
            if not v:
                return
            # Keep extracted constraints bounded for readability and stable context rendering.
            if len(v) < 6:
                return
            if len(v) > 240:
                v = v[:240].rstrip() + "…"
            key = f"constraint::{self._canon(v)}"
            if key in seen_keys:
                return
            seen_keys.add(key)
            out.append(FeatureItem(key=key, kind="constraint", confidence=conf, value=v))

        # 1) explicit "constraint:" lines (existing behavior)
        for m in re.finditer(r"(?im)^\s*constraint\s*:\s*(.+?)\s*$", text):
            add(m.group(1), 0.90)

        # 2) common explicit variants
        for m in re.finditer(r"(?im)^\s*(rule|policy|lesson|guardrail)\s*:\s*(.+?)\s*$", text):
            add(m.group(2), 0.86)

        # 3) bullet-list forms like:
        #    - do not ...
        #    - avoid ...
        #    * never ...
        for m in re.finditer(r"(?im)^\s*[-*•]\s*(.+?)\s*$", text):
            line = m.group(1).strip()
            if re.search(r"\b(don't|do not|must not|should not|never|avoid|always|prefer)\b", line.lower()):
                add(line, 0.78)

        # 4) sentence-level heuristic extraction (natural language)
        # Split into rough sentences/lines while preserving determinism.
        chunks = re.split(r"(?<=[\.\!\?])\s+|\n+", text.strip())
        cue_re = re.compile(
            r"\b("
            r"don't|do not|must not|should not|never|avoid|"
            r"always|prefer|only if|unless|"
            r"do(?:\s+this)?\s+instead|"
            r"stop\s+\w+|"
            r"rule\s+of\s+thumb"
            r")\b",
            re.IGNORECASE,
        )

        for c in chunks:
            s = c.strip()
            if not s:
                continue
            low = s.lower()

            # Skip if it's obviously a question (often speculative)
            if low.endswith("?"):
                continue

            # Only consider sentences that look constraint-like
            if not cue_re.search(low):
                continue

            # Avoid extracting meta-talk like "constraint: ..." again
            if low.startswith(("constraint:", "goal:", "task:")):
                continue

            # Slightly reduce false positives: require at least one verb-ish cue
            # (good enough for demo; can be refined later)
            add(s, 0.72)

            # keep bounded
            if len(out) >= 6:
                break

        return out


    def _extract_entities(self, text: str) -> List[FeatureItem]:
        ents = set()
        for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", text):
            token = m.group(0).lower()
            if token in self._entity_stop:
                continue
            ents.add(token)
        out = []
        for e in sorted(ents)[:6]:
            out.append(FeatureItem(key=f"entity::{e}", kind="entity", confidence=0.65, value=e))
        return out

    def _extract_taskish(self, text: str) -> List[FeatureItem]:
        t = text.lower()
        out = []
        for phrase in ("episode segmentation","concept promotion","adjudication","replay","pruning","context","anchors","boundedness","decay","compression"):
            if phrase in t:
                out.append(FeatureItem(key=f"task::{phrase.replace(' ','_')}", kind="task", confidence=0.7, value=phrase))
        return out

    def _extract_lexical(self, text: str) -> List[FeatureItem]:
        toks = re.findall(r"[a-zA-Z0-9_']+", text.lower())
        toks = [t for t in toks if len(t) >= 3][:10]
        out = [FeatureItem(key=f"token::{t}", kind="token", confidence=0.25, value=t) for t in toks]
        out += [FeatureItem(key=f"bigram::{toks[i]}_{toks[i+1]}", kind="bigram", confidence=0.2) for i in range(min(len(toks)-1, 6))]
        return out

    def _validate_llm_fill(self, event_id: str, llm_out: Dict[str, Any]) -> ExtractionRecord:
        try:
            intent = str(llm_out.get("intent","request"))
            feats: List[FeatureItem] = []
            for f in llm_out.get("features", []):
                key = str(f["key"]); kind = str(f["kind"])
                conf = float(f.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
                if kind not in ("goal","constraint","entity","task","preference","token","bigram"):
                    continue
                feats.append(FeatureItem(key=key, kind=kind, confidence=conf, value=f.get("value")))
            rels: List[RelationItem] = []
            for r in llm_out.get("relations", []):
                rel_type = str(r.get("rel_type","rel:cooccur"))
                if rel_type not in ("rel:cooccur","rel:supports","rel:conflicts","rel:depends"):
                    continue
                rels.append(RelationItem(src=str(r["src"]), rel_type=rel_type, dst=str(r["dst"]), confidence=float(r.get("confidence",0.5))))
            return ExtractionRecord(event_id=event_id, intent=intent, features=feats, relations=rels)
        except Exception:
            return ExtractionRecord(event_id=event_id, intent="request")

# ------------------------------------------------------------------
# Engine lifecycle
# ------------------------------------------------------------------

class YggdrasilEngine:
    """
    Graph-backed runtime memory engine for Yggdrasil.

    The engine records events and outcomes, ingests extracted features into a
    graph substrate, promotes recurring features into concepts, applies bounded
    maintenance over time, and assembles deterministic context for downstream
    actor use.
    """
    def __init__(
        self,
        session_id: str,
        logger: JsonlLogger,
        adjudicator: HybridAdjudicator,
        promote_threshold: int = 3,
        base_decay: float = 0.06,
        min_detail_salience: float = 0.05,
        keep_recent_details_per_feature: int = 2,
        hard_failsafe_detail_edges: int = 2000,
        hard_failsafe_target: int = 1500,
        enable_decay: bool = True,
    ) -> None:
        self.session_id = session_id
        self.log = logger
        self.graph = Graph()
        self.adjudicator = adjudicator
        self.promote_threshold = int(promote_threshold)
        self.base_decay = float(base_decay)
        self.min_detail_salience = float(min_detail_salience)
        self.keep_recent_details_per_feature = int(keep_recent_details_per_feature)
        self.hard_failsafe_detail_edges = int(hard_failsafe_detail_edges)
        self.hard_failsafe_target = int(hard_failsafe_target)
        self.enable_decay = bool(enable_decay)
        self._last_context_json = ""
        self.ablate_governance: bool = False
        self._last_prompt_risk: Dict[str, Any] = {}

        self.step: int = 0
        self._current_episode_id: Optional[str] = None
        self._episode_order: List[str] = []
        self._event_order: List[str] = []
        self._feature_counts: Dict[str, int] = {}
        self._feature_support: Dict[str, float] = {}
        self._last_step_metrics: Dict[str, Any] = {"decayed": 0, "deleted_edges": 0, "compressed_edges": 0, "promoted": 0}

    def start_step(self) -> None:
        self._last_step_metrics = {"decayed": 0, "deleted_edges": 0, "compressed_edges": 0, "promoted": 0}

    def write_event(
        self,
        role: str,
        text: str,
        source: str="cli",
        provenance: Optional[Dict[str, Any]]=None,
        event_id: Optional[str]=None,
    ) -> str:
        ev_id = event_id or str(uuid.uuid4())
        ev_node = self.graph.upsert_node("event", key=ev_id, data={
            "session_id": self.session_id,
            "role": role,
            "source": source,
            "text": text,
            "provenance": provenance or {},
            "initial_confidence": float((provenance or {}).get("confidence", 1.0)),
            "created_step": self.step,
        }, ts=time.time())
        self._event_order.append(ev_id)
        self.log.log("event_written", {"event_id": ev_id, "session_id": self.session_id, "role": role, "source": source, "text": text})

        if self._current_episode_id is None:
            self._current_episode_id = str(uuid.uuid4())
            self.graph.upsert_node("episode", key=self._current_episode_id, data={"session_id": self.session_id, "status": "open", "opened_step": self.step})
            self._episode_order.append(self._current_episode_id)
            self.log.log("episode_opened", {"session_id": self.session_id, "episode_id": self._current_episode_id, "step": self.step})

        ep_node = self.graph.find_node_by_key("episode", self._current_episode_id)
        if ep_node:
            self.graph.add_edge(ep_node.id, ev_node.id, "contains", confidence=1.0, data={"step": self.step})
        return ev_id

# ------------------------------------------------------------------
# Ingestion and promotion
# ------------------------------------------------------------------

    def adjudicate_and_ingest(self, event_id: str) -> None:
        ev_node = self.graph.find_node_by_key("event", event_id)
        if not ev_node or not self._current_episode_id:
            self.log.log("event_rejected", {"event_id": event_id, "reason": "missing_event_or_episode"})
            return
        role = str(ev_node.data.get("role","user"))
        rec = self.adjudicator.adjudicate(ev_node.data["text"], event_id=event_id, role=role, episode_open=True)
        self.ingest_extraction(self._current_episode_id, rec)

    def ingest_extraction(self, episode_id: str, rec: ExtractionRecord) -> None:
        ep_node = self.graph.find_node_by_key("episode", episode_id)
        ev_node = self.graph.find_node_by_key("event", rec.event_id)
        if not ep_node or not ev_node:
            self.log.log("extraction_rejected", {"event_id": rec.event_id, "reason": "missing_nodes"})
            return

        ev_node.data["intent"] = rec.intent
        ev_node.data["extraction_meta"] = rec.meta

        created_rel = 0
        promoted_now = 0
        present_keys = set()

        for f in rec.features:
            present_keys.add(f.key)
            feat = self.graph.upsert_node("feature", key=f.key, data={"kind": f.kind, "value": f.value})

            prev = self._feature_counts.get(f.key, 0)
            novelty = 1.0 / (1.0 + float(prev))
            self._feature_counts[f.key] = prev + 1

            # Explicit human-provided constraints are promoted immediately so they can
            # influence the next turn without requiring repeated exposure.
            # If a constraint was provided explicitly (e.g., 'lesson:'/'policy:' lines),
            # promote it immediately so it is available on the very next turn.
            promoted = False
            if f.kind == "constraint" and float(f.confidence) >= 0.86:
                promoted = self._promote_now(f.key, f.kind, concept_kind="manual")
            else:
                promoted = self._maybe_promote(f.key, f.kind)

            if promoted:
                promoted_now += 1

            concept = self.graph.find_node_by_key("concept", f.key)
            is_structured = f.kind in ("goal","constraint","entity","task","preference")

            if concept and is_structured:
                concept.data["count_seen"] = int(concept.data.get("count_seen", 0)) + 1
                concept.data["last_seen_step"] = self.step
                continue

            init_sal = self._initial_salience(f.kind, float(f.confidence), novelty)
            self.graph.add_edge(ev_node.id, feat.id, "has_feature", confidence=float(f.confidence), data={
                "salience": init_sal,
                "novelty": novelty,
                "created_step": self.step,
                "kind": f.kind,
            })

        feat_nodes = {k: self.graph.find_node_by_key("feature", k) for k in present_keys}
        for r in rec.relations:
            if r.src not in present_keys or r.dst not in present_keys:
                continue
            src = feat_nodes.get(r.src); dst = feat_nodes.get(r.dst)
            if not src or not dst:
                continue
            self.graph.add_edge(src.id, dst.id, r.rel_type, confidence=float(r.confidence), data={"step": self.step})
            created_rel += 1

        self._last_step_metrics["promoted"] += promoted_now
        self.log.log("extraction_ingested", {
            "event_id": rec.event_id,
            "episode_id": episode_id,
            "intent": rec.intent,
            "features": len(rec.features),
            "relations": len(rec.relations),
            "created_rel": created_rel,
            "lexical": rec.meta.get("lexical_enabled"),
            "structured_count": rec.meta.get("structured_count"),
            "promoted_now": promoted_now,
            "role": rec.meta.get("role"),
        })

    def _initial_salience(self, kind: str, confidence: float, novelty: float) -> float:
        kind_weight = {
            "constraint": 1.2,
            "goal": 1.1,
            "task": 1.0,
            "entity": 0.9,
            "preference": 1.0,
            "token": 0.2,
            "bigram": 0.15,
        }.get(kind, 0.5)
        return float(max(0.0, min(2.0, confidence * kind_weight * (0.7 + 0.6 * novelty))))

    def _promote_now(self, feature_key: str, kind: str, concept_kind: str="manual") -> bool:
        """
        Immediately promote a feature into a concept node (used for explicit, human-provided constraints).
        Returns True if promotion occurred now, False if the concept already existed or kind is not promotable.
        """
        if kind not in ("goal","constraint","entity","task","preference"):
            return False
        if self.graph.find_node_by_key("concept", feature_key):
            return False
        count = int(self._feature_counts.get(feature_key, 0))
        concept = self.graph.upsert_node("concept", key=feature_key, data={
            "kind": concept_kind,
            "support": float(self._feature_support.get(feature_key, 0.0)),
            "count_seen": int(max(1, count)),
            "promoted_step": self.step,
            "last_seen_step": self.step,
        })
        feat = self.graph.find_node_by_key("feature", feature_key)
        if feat:
            self.graph.add_edge(feat.id, concept.id, "promotes_to", confidence=1.0, data={"step": self.step, "forced": True})
        self.log.log("concept_promoted", {"key": feature_key, "concept_id": concept.id, "count": count, "step": self.step, "forced": True, "concept_kind": concept_kind})
        return True

    def _maybe_promote(self, feature_key: str, kind: str) -> bool:
        if kind not in ("goal","constraint","entity","task","preference"):
            return False
        count = self._feature_counts.get(feature_key, 0)
        if count != self.promote_threshold:
            return False
        concept = self.graph.upsert_node("concept", key=feature_key, data={
            "kind": "emergent",
            "support": float(self._feature_support.get(feature_key, 0.0)),
            "count_seen": int(count),
            "promoted_step": self.step,
            "last_seen_step": self.step,
        })
        feat = self.graph.find_node_by_key("feature", feature_key)
        if feat:
            self.graph.add_edge(feat.id, concept.id, "promotes_to", confidence=1.0, data={"step": self.step})
        self.log.log("concept_promoted", {"key": feature_key, "concept_id": concept.id, "count": count, "step": self.step})
        return True

    def write_outcome(
        self,
        score: float,
        label: str="",
        tags: Optional[List[str]]=None,
        outcome_id: Optional[str]=None,
    ) -> str:
        out_id = outcome_id or str(uuid.uuid4())
        out_node = self.graph.upsert_node("outcome", key=out_id, data={
            "session_id": self.session_id,
            "score": float(score),
            "label": label,
            "tags": tags or [],
            "created_step": self.step,
        }, ts=time.time())

        targets: List[str] = []
        if self._current_episode_id:
            targets.append(self._current_episode_id)

        for ep_key in targets:
            ep = self.graph.find_node_by_key("episode", ep_key)
            if ep:
                self.graph.add_edge(out_node.id, ep.id, "leads_to", weight=float(score), confidence=1.0, data={"step": self.step})
        self.log.log("outcome_written", {"outcome_id": out_id, "session_id": self.session_id, "score": float(score), "targets": targets, "label": label, "tags": tags or []})

        if self._current_episode_id:
            ep = self.graph.find_node_by_key("episode", self._current_episode_id)
            if ep:
                ep.data["status"] = "closed"
                ep.data["closed_step"] = self.step
            self.log.log("episode_closed", {"session_id": self.session_id, "episode_id": self._current_episode_id, "step": self.step})
            self._current_episode_id = None

        self._apply_outcome_support(float(score))
        return out_id
    
# ------------------------------------------------------------------
# Maintenance: decay, compression, failsafes
# ------------------------------------------------------------------

    def end_step(self) -> None:
        self.step += 1
        if self.enable_decay:
            self._decay_detail_edges()
            self._compress_promoted_features()
            self._failsafe_if_needed()
        else:
            self.log.log("decay_skipped", {"step": self.step, "reason": "enable_decay_false"})
        h, summary = compute_state_hash(self.graph)
        self.log.log("state_hash", {"step": self.step, "hash": h, **summary})
        self.log.log("step_end", {"step": self.step, **self._last_step_metrics})
        # also store for HUD/debug
        self._last_step_metrics["state_hash"] = h


    def _decay_detail_edges(self) -> None:
        decayed = 0
        deleted = 0
        for eid, e in list(self.graph.edges.items()):
            if e.type != "has_feature":
                continue
            sal = float(e.data.get("salience", 0.0))
            novelty = float(e.data.get("novelty", 0.5))
            repeat_penalty = (1.0 - novelty)
            rate = self.base_decay * (1.0 + repeat_penalty)
            new_sal = sal * (1.0 - rate)
            e.data["salience_prev"] = sal
            e.data["salience"] = new_sal
            decayed += 1
            if new_sal < self.min_detail_salience:
                if self.graph.delete_edge(eid):
                    deleted += 1
        self._last_step_metrics["decayed"] += decayed
        self._last_step_metrics["deleted_edges"] += deleted
        self.log.log("decay_step", {"step": self.step, "decayed_edges": decayed, "deleted_edges": deleted, "min_salience": self.min_detail_salience})

    def _compress_promoted_features(self) -> None:
        compressed = 0
        by_feature: Dict[str, List[tuple]] = {}
        for eid, e in self.graph.edges.items():
            if e.type != "has_feature":
                continue
            feat = self.graph.nodes.get(e.dst)
            if not feat or feat.type != "feature" or not feat.key:
                continue
            if not self.graph.find_node_by_key("concept", feat.key):
                continue
            created_step = float(e.data.get("created_step", 0))
            by_feature.setdefault(feat.key, []).append((eid, created_step))
        for fkey, lst in by_feature.items():
            # Stable ordering for deterministic replay. Prefer newer details,
            # then use the edge id only as a final within-run tie breaker. In
            # controlled replay tests compression can be disabled with --no-decay.
            lst.sort(key=lambda x: (x[1], x[0]), reverse=True)
            for eid, _ in lst[self.keep_recent_details_per_feature:]:
                if self.graph.delete_edge(eid):
                    compressed += 1
        if compressed:
            self._last_step_metrics["compressed_edges"] += compressed
            self.log.log("compressed", {"step": self.step, "compressed_edges": compressed, "keep_recent_per_feature": self.keep_recent_details_per_feature})

    def _failsafe_if_needed(self) -> None:
        detail_count = self.graph.count_edges("has_feature")
        if detail_count <= self.hard_failsafe_detail_edges:
            return
        edges = []
        for eid, e in self.graph.edges.items():
            if e.type != "has_feature":
                continue
            edges.append((float(e.data.get("salience", 0.0)), float(e.data.get("created_step", 0)), eid))
        edges.sort(key=lambda x: (x[0], x[1]))
        deleted = 0
        while self.graph.count_edges("has_feature") > self.hard_failsafe_target and edges:
            _, _, eid = edges.pop(0)
            if self.graph.delete_edge(eid):
                deleted += 1
        self._last_step_metrics["deleted_edges"] += deleted
        self.log.log("failsafe_prune", {"step": self.step, "detail_edges_before": detail_count, "deleted": deleted, "target": self.hard_failsafe_target})

    def set_ablation(self, enabled: bool) -> None:
        self.ablate_governance = bool(enabled)
        self.log.log("ablation_toggled", {"step": self.step, "enabled": self.ablate_governance})

    def _apply_outcome_support(self, score: float) -> None:
        """
        Apply scalar outcome feedback to recently-seen features.

        For current runtime behavior:
        - Constraints get reinforced by abs(score) so a *bad* outcome makes the corrective
          constraint MORE salient (rather than burying it with negative support).
        - Other kinds keep signed reinforcement.
        """
        recent_event_keys = self._event_order[-50:]
        for ev_key in recent_event_keys:
            ev = self.graph.find_node_by_key("event", ev_key)
            if not ev:
                continue
            for e, feat in self.graph.neighbors_out(ev.id, edge_type="has_feature"):
                if not feat.key:
                    continue

                kind = str(feat.data.get("kind") or "")
                bump = float(abs(score) if kind == "constraint" else score)

                self._feature_support[feat.key] = self._feature_support.get(feat.key, 0.0) + bump * float(e.confidence)

                sal = float(e.data.get("salience", 0.0))
                new_sal = min(2.0, sal + 0.15 * bump * float(e.confidence))
                e.data["salience_prev_outcome"] = sal
                e.data["salience"] = new_sal

                c = self.graph.find_node_by_key("concept", feat.key)
                if c:
                    c.data["support"] = float(self._feature_support[feat.key])

        self.log.log("reinforced", {"step": self.step, "score": score, "window_events": len(recent_event_keys)})

# ------------------------------------------------------------------
# Context and diagnostics
# ------------------------------------------------------------------

    def hud(self) -> Dict[str, Any]:
        detail_edges = self.graph.count_edges("has_feature")
        counts = {
            "step": self.step,
            "episodes": len(self.graph.by_type.get("episode", [])),
            "events": len(self.graph.by_type.get("event", [])),
            "features": len(self.graph.by_type.get("feature", [])),
            "concepts": len(self.graph.by_type.get("concept", [])),
            "outcomes": len(self.graph.by_type.get("outcome", [])),
            "detail_edges": detail_edges,
            "episode_open": 1 if self._current_episode_id else 0,
        }
        return {"counts": counts, "top_concepts": self.top_concepts(5), "last_step": dict(self._last_step_metrics), "ablation": {"governance_disabled": bool(self.ablate_governance)}}

    def top_concepts(self, k: int=5):
        rows = []
        for n in self.graph.find_nodes("concept"):
            rows.append((n.key or n.id, float(n.data.get("support", 0.0))))
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows[:k]
    
    def _norm_tokens(self, text: str) -> set[str]:
        toks = re.findall(r"[a-zA-Z0-9_+-]+", (text or "").lower())
        stop = {
            "the", "and", "or", "for", "with", "using", "use", "write",
            "create", "design", "a", "an", "to", "of", "in", "it", "is",
            "as", "this", "that", "please"
        }
        return {t for t in toks if len(t) >= 3 and t not in stop}

    def _prompt_signals(self, prompt_text: str) -> Dict[str, bool]:
        s = (prompt_text or "").lower()

        fictional = bool(re.search(
            r"\b(fictional|not a real package|made-up|invented for this example)\b",
            s,
        ))

        design_framing = bool(re.search(
            r"\b(mock|toy|pseudocode|interface design|demonstration only|hypothetical|api shape|architecture sketch)\b",
            s,
        ))
        production = bool(re.search(
            r"\b(production|real installed|installed|executable|final code|import|imports|pip|package|library|dependency)\b",
            s,
        ))
        asks_code = bool(re.search(
            r"\b(write|implement|script|code|import|install|execute|compute)\b",
            s,
        ))

        assumption_present = bool(re.search(r"\b(assume|pretend|suppose|treat it as|imagine it is)\b", s))

        usage_request = bool(re.search(
            r"\b(show how to use|usage|use it|might be used|used in a system|example code|implement|script|code)\b",
            s,
        ))

        install_request = bool(re.search(
            r"\b(pip install|npm install|install|installation|requirements\.txt|dependency)\b",
            s,
        ))

        explicit_fictional = bool(re.search(
            r"\b(fictional|not a real package|made-up|invented for this example)\b",
            s,
        ))

        design_only = bool(re.search(
            r"\b(api shape|interface design|architecture sketch|toy architecture sketch)\b", s,
        ))

        conceptual_code = bool(re.search(
            r"\b(conceptual version|example code|pseudocode|mock implementation)\b", s,
        ))

        usage_sketch = bool(re.search(
            r"\b(might be used|used in a system|show how it would be used|how it works in a system)\b", s,
        ))

        mentions_cli = bool(re.search(r"\b(cli|command-line|terminal|shell|bash|zsh|command|flags?|subcommands?)\b", s))

        cli_usage_request = bool(re.search(
            r"\b(show basic usage|run|execute|use .* flags|flags? for|subcommands?|shell script|deployment script|command)\b",
            s,
        ))

        deploy_request = bool(re.search(
            r"\b(deploy|deployment|pipeline|syncing nodes|across regions|cluster|orchestrate)\b",
            s,
        ))

        package_manager_request = bool(re.search(
            r"\b(brew install|apt install|yum install|cargo install|pip install|npm install|curl .* install)\b",
            s,
        ))

        mentions_database = bool(re.search(
            r"\b(database|sql|postgres|postgresql|mysql|sqlite|extension|plugin|index type|storage engine|query planner)\b",
            s,
        ))

        database_usage_request = bool(re.search(
            r"\b(create extension|sql|query|schema|index|migration|database config|storage engine|plugin)\b",
            s,
        ))

        mentions_cloud = bool(re.search(
            r"\b(cloud|api|sdk|endpoint|iam|aws|gcp|azure|lambda|service account|deployment target|hosted service)\b",
            s,
        ))

        cloud_usage_request = bool(re.search(
            r"\b(endpoint|sdk call|iam policy|deploy|deployment|credentials|service account|cloud function|hosted api)\b",
            s,
        ))

        mentions_ml = bool(re.search(
            r"\b(ml|machine learning|model|checkpoint|tokenizer|inference|training|fine-tune|framework|transformers|pytorch|tensorflow)\b",
            s,
        ))

        ml_usage_request = bool(re.search(
            r"\b(load model|run inference|tokenizer|checkpoint|train|fine-tune|model config|pipeline)\b",
            s,
        ))
        return {
            "fictional_design": fictional,
            "design_framing": design_framing,
            "explicit_fictional": explicit_fictional,
            "design_only": design_only,
            "conceptual_code": conceptual_code,
            "usage_sketch": usage_sketch,
            "real_dependency_request": production and not explicit_fictional,
            "asks_for_code": asks_code,
            "mentions_library": bool(re.search(r"\b(library|package|dependency|api|import|pip)\b", s)),
            "mentions_cli": mentions_cli,
            "assumption_present": assumption_present,
            "usage_request": usage_request,
            "install_request": install_request,
            "cli_usage_request": cli_usage_request,
            "deploy_request": deploy_request,
            "package_manager_request": package_manager_request,
            "mentions_database": mentions_database,
            "database_usage_request": database_usage_request,
            "mentions_cloud": mentions_cloud,
            "cloud_usage_request": cloud_usage_request,
            "mentions_ml": mentions_ml,
            "ml_usage_request": ml_usage_request,
                    }

    def _build_routing_state(self, prompt_text: str) -> Dict[str, Any]:
        """
        Build a prompt-level routing frame before constraint-family selection.

        This does not select constraints yet. It only determines:
        - whether the prompt contains an unfamiliar artifact-like token
        - whether the user is asking for operational action
        - whether the artifact domain is clear enough to route directly
        - whether clarification should be preferred before domain-specific action
        """
        signals = self._prompt_signals(prompt_text)
        s = (prompt_text or "").lower()

        artifact_like = bool(re.search(
            r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+){2,}\b",
            prompt_text or "",
        ))

        operational_intent = any([
            signals.get("asks_for_code", False),
            signals.get("usage_request", False),
            signals.get("install_request", False),
            signals.get("package_manager_request", False),
            signals.get("cli_usage_request", False),
            signals.get("deploy_request", False),
            signals.get("database_usage_request", False),
            signals.get("cloud_usage_request", False),
            signals.get("ml_usage_request", False),
        ])

        explicit_library = bool(re.search(
            r"\b(python library|library|package|dependency|pip|import|imports)\b",
            s,
        ))

        explicit_cli = bool(re.search(
            r"\b(cli|command-line|terminal|shell|bash|zsh|flags?|subcommands?|brew)\b",
            s,
        ))

        domain_scores = {
            "library": 0.0,
            "cli": 0.0,
            "database": 0.0,
            "cloud": 0.0,
            "ml": 0.0,
        }

        if explicit_library:
            domain_scores["library"] += 0.75
        if "pip install" in s:
            domain_scores["library"] += 0.35
        if "python" in s and signals.get("asks_for_code", False):
            domain_scores["library"] += 0.35

        if explicit_cli:
            domain_scores["cli"] += 0.75
        if signals.get("cli_usage_request", False):
            domain_scores["cli"] += 0.35
        if signals.get("deploy_request", False):
            domain_scores["cli"] += 0.20
        if "brew install" in s:
            domain_scores["cli"] += 0.35

        # Database / extension route
        if signals.get("mentions_database", False):
            domain_scores["database"] += 0.75
        if signals.get("database_usage_request", False):
            domain_scores["database"] += 0.35
        if re.search(r"\b(create extension|sql|postgres|postgresql|mysql|sqlite|index|schema|migration)\b", s):
            domain_scores["database"] += 0.25

        # Cloud / hosted API route
        if signals.get("mentions_cloud", False):
            domain_scores["cloud"] += 0.75
        if signals.get("cloud_usage_request", False):
            domain_scores["cloud"] += 0.35
        if re.search(r"\b(aws|gcp|azure|lambda|iam|sdk|endpoint|service account|cloud function)\b", s):
            domain_scores["cloud"] += 0.25

        # ML model / framework route
        if signals.get("mentions_ml", False):
            domain_scores["ml"] += 0.75
        if signals.get("ml_usage_request", False):
            domain_scores["ml"] += 0.35
        if re.search(r"\b(model|checkpoint|tokenizer|inference|training|fine-tune|pytorch|tensorflow|transformers)\b", s):
            domain_scores["ml"] += 0.25

        best_domain = max(domain_scores, key=domain_scores.get)
        best_score = float(domain_scores[best_domain])

        domain_clear = best_score >= 0.70

        unknown_artifact_uncertainty = (
            artifact_like
            and operational_intent
            and not domain_clear
            and not signals.get("explicit_fictional", False)
        )

        if signals.get("explicit_fictional", False):
            route = "fictional_design"
        elif unknown_artifact_uncertainty:
            route = "unknown_artifact_uncertainty"
        elif domain_clear:
            route = best_domain
        else:
            route = "general"

        clarification_needed = (
            route == "unknown_artifact_uncertainty"
            and operational_intent
            and not signals.get("explicit_fictional", False)
        )

        return {
            "signals": signals,
            "artifact_like": artifact_like,
            "operational_intent": operational_intent,
            "domain_scores": domain_scores,
            "domain": best_domain if domain_clear else "unknown",
            "domain_confidence": best_score,
            "domain_clear": domain_clear,
            "route": route,
            "clarification_needed": clarification_needed,
        }

    def _family_route_allowed(self, family: str, routing_state: Dict[str, Any]) -> bool:
        """
        Decide whether a constraint family is allowed under the current route.

        This prevents domain-specific families from activating before the prompt
        has enough evidence to route into that domain.
        """
        route = routing_state.get("route")

        if route == "fictional_design":
            return False

        if family == "unknown_artifact_uncertainty":
            return route == "unknown_artifact_uncertainty"

        if route == "library":
            return family in {
                "unverified_dependency_existence",
                "placeholder_api_bypass",
                "install_command_risk",
                "alternative_suggestion_policy",
                "general_constraint",
            }

        if route == "cli":
            return family in {
                "unverified_cli_existence",
                "cli_command_bypass",
                "install_command_risk",
                "alternative_suggestion_policy",
                "general_constraint",
            }

        if route == "database":
            return family in {
                "unverified_database_extension",
                "database_syntax_bypass",
                "install_command_risk",
                "alternative_suggestion_policy",
                "general_constraint",
            }

        if route == "cloud":
            return family in {
                "unverified_cloud_service",
                "cloud_api_bypass",
                "alternative_suggestion_policy",
                "general_constraint",
            }

        if route == "ml":
            return family in {
                "unverified_ml_artifact",
                "ml_api_bypass",
                "install_command_risk",
                "alternative_suggestion_policy",
                "general_constraint",
            }

        if route == "unknown_artifact_uncertainty":
            return family in {
                "unknown_artifact_uncertainty",
                "general_constraint",
            }

        return family in {
            "unknown_artifact_uncertainty",
            "general_constraint",
        }

    def _constraint_family(self, constraint_key: str) -> str:
        k = (constraint_key or "").lower()

        # Broad uncertainty must be checked first.
        if any(x in k for x in (
            "unfamiliar artifact",
            "do not assume what type of artifact",
            "proper-noun system",
            "before providing domain-specific",
            "commands, imports, apis, flags",
        )):
            return "unknown_artifact_uncertainty"

        # Domain-specific bypass families should be checked before generic
        # "cannot be verified" existence language.
        if any(x in k for x in (
            "sql syntax",
            "database config",
            "database configuration",
            "create extension",
            "extension apis",
            "storage engine",
            "query planner",
        )):
            return "database_syntax_bypass"

        if any(x in k for x in (
            "database extension",
            "sql module",
            "database plugin",
            "index type",
            "storage engine",
            "unverified database",
        )):
            return "unverified_database_extension"

        if any(x in k for x in (
            "endpoints",
            "sdk calls",
            "iam policies",
            "credentials",
            "deployment steps",
            "provider-specific",
        )):
            return "cloud_api_bypass"

        if any(x in k for x in (
            "cloud service",
            "hosted api",
            "sdk",
            "iam role",
            "deployment target",
            "provider-specific feature",
        )):
            return "unverified_cloud_service"

        if any(x in k for x in (
            "model calls",
            "configs",
            "checkpoints",
            "training code",
            "tokenizer",
            "inference api",
        )):
            return "ml_api_bypass"

        if any(x in k for x in (
            "ml framework",
            "model name",
            "checkpoint",
            "training package",
            "machine learning framework",
        )):
            return "unverified_ml_artifact"

        if any(x in k for x in (
            "command-line flags",
            "subcommands",
            "config schemas",
            "shell pipelines",
            "deployment scripts",
        )):
            return "cli_command_bypass"

        if any(x in k for x in (
            "cli tool",
            "command-line",
            "fabricating commands",
            "flags",
            "subcommands",
            "shell scripts",
        )):
            return "unverified_cli_existence"

        if any(x in k for x in (
            "pip install",
            "installation commands",
            "install commands",
            "npm install",
            "brew install",
            "curl install",
            "package-manager commands",
        )):
            return "install_command_risk"

        if any(x in k for x in (
            "hypothetical",
            "placeholder",
            "mock interfaces",
            "example usage",
            "inventing",
            "fabricating imports",
        )):
            return "placeholder_api_bypass"

        if any(x in k for x in (
            "suggest well-known alternatives",
            "alternatives only if",
            "well-known alternatives",
        )):
            return "alternative_suggestion_policy"

        # Generic unverifiable dependency existence should come late,
        # otherwise it absorbs CLI/database/cloud/ML constraints.
        if any(x in k for x in (
            "software library",
            "package name",
            "unfamiliar",
            "cannot be verified",
            "do not assume it exists",
            "verified as real",
        )):
            return "unverified_dependency_existence"

        return "general_constraint"
        
    def _family_threshold(self, family: str, risk: float, signals: Dict[str, bool]) -> float:
        """
        Family-specific applicability threshold.
        Lower = easier to inject.

        Iteration 4.4:
        - distinguishes explicit fiction from design-only prompts
        - prevents toy/interface design from triggering placeholder constraints
        - keeps conceptual code and usage sketches risk-governed
        """

        if signals.get("explicit_fictional"):
            if family in (
                "unverified_dependency_existence",
                "placeholder_api_bypass",
                "install_command_risk",
                "alternative_suggestion_policy",
            ):
                return 0.85

        if family == "install_command_risk":
            return 0.30 if signals.get("install_request") else 0.85

        if family == "placeholder_api_bypass":
            if signals.get("design_only") and not signals.get("conceptual_code") and not signals.get("usage_sketch"):
                return 0.85

            if (
                signals.get("conceptual_code")
                or signals.get("usage_sketch")
                or signals.get("usage_request")
                or signals.get("assumption_present")
            ):
                return 0.35 if risk >= 0.50 else 0.45

            if signals.get("asks_for_code"):
                return 0.40 if risk >= 0.50 else 0.50

            return 0.70

        if family == "unverified_dependency_existence":
            if signals.get("design_only") and not signals.get("usage_sketch"):
                return 0.75

            if (
                signals.get("mentions_library")
                or signals.get("usage_sketch")
                or signals.get("usage_request")
                or signals.get("assumption_present")
                or signals.get("real_dependency_request")
            ):
                return 0.35 if risk >= 0.50 else 0.45

            return 0.65

        if family == "alternative_suggestion_policy":
            if signals.get("design_only"):
                return 0.80

            if signals.get("real_dependency_request") or signals.get("usage_request"):
                return 0.45

            return 0.70
        
        if family == "unverified_cli_existence":
            if signals.get("explicit_fictional"):
                return 0.85
            if signals.get("design_only"):
                return 0.75
            if signals.get("mentions_cli") or signals.get("cli_usage_request") or signals.get("deploy_request"):
                return 0.35 if risk >= 0.50 else 0.45
            return 0.65

        if family == "cli_command_bypass":
            if signals.get("explicit_fictional"):
                return 0.85
            if signals.get("design_only") and not signals.get("cli_usage_request"):
                return 0.80
            if signals.get("cli_usage_request") or signals.get("deploy_request") or signals.get("assumption_present"):
                return 0.35 if risk >= 0.50 else 0.45
            return 0.70
        
        if family == "unverified_database_extension":
            if signals.get("explicit_fictional"):
                return 0.85
            if signals.get("design_only"):
                return 0.75
            if signals.get("mentions_database") or signals.get("database_usage_request"):
                return 0.35 if risk >= 0.50 else 0.45
            return 0.65

        if family == "database_syntax_bypass":
            if signals.get("explicit_fictional"):
                return 0.85
            if signals.get("design_only") and not signals.get("database_usage_request"):
                return 0.80
            if signals.get("database_usage_request") or signals.get("assumption_present"):
                return 0.35 if risk >= 0.50 else 0.45
            return 0.70

        if family == "unverified_cloud_service":
            if signals.get("explicit_fictional"):
                return 0.85
            if signals.get("design_only"):
                return 0.75
            if signals.get("mentions_cloud") or signals.get("cloud_usage_request") or signals.get("deploy_request"):
                return 0.35 if risk >= 0.50 else 0.45
            return 0.65

        if family == "cloud_api_bypass":
            if signals.get("explicit_fictional"):
                return 0.85
            if signals.get("design_only") and not signals.get("cloud_usage_request"):
                return 0.80
            if signals.get("cloud_usage_request") or signals.get("deploy_request") or signals.get("assumption_present"):
                return 0.35 if risk >= 0.50 else 0.45
            return 0.70

        if family == "unverified_ml_artifact":
            if signals.get("explicit_fictional"):
                return 0.85
            if signals.get("design_only"):
                return 0.75
            if signals.get("mentions_ml") or signals.get("ml_usage_request"):
                return 0.35 if risk >= 0.50 else 0.45
            return 0.65

        if family == "ml_api_bypass":
            if signals.get("explicit_fictional"):
                return 0.85
            if signals.get("design_only") and not signals.get("ml_usage_request"):
                return 0.80
            if signals.get("ml_usage_request") or signals.get("assumption_present"):
                return 0.35 if risk >= 0.50 else 0.45
            return 0.70

        return self._dynamic_threshold(risk)
    
    def _compute_prompt_risk(self, prompt_text: str) -> float:
        """
        Heuristic risk scoring for a prompt.
        Returns a float in [0.0, 1.0].
        """

        signals = self._prompt_signals(prompt_text)
        s = prompt_text.lower()

        risk = 0.0

        # 1. Unknown entity / fabricated library risk
        if re.search(r"[a-zA-Z0-9_-]+-[a-zA-Z0-9_-]+", s):
            risk += 0.35

        # 2. Code generation / execution intent
        if signals["asks_for_code"]:
            risk += 0.25

        # 3. Dependency / library context
        if signals["mentions_library"] or signals["real_dependency_request"]:
            risk += 0.25

        # 4. Assumption injection (strong signal)
        if "assume" in s:
            risk += 0.35

        # 5. Conceptual / fictional reduces risk (but not to zero)
        if signals["explicit_fictional"]:
            risk -= 0.35
        elif signals.get("design_framing"):
            risk -= 0.15

        # Clamp
        return max(0.0, min(1.0, risk))

    def _dynamic_threshold(self, risk: float) -> float:
        """
        Convert risk score into applicability threshold.
        Lower threshold = easier to inject constraints.
        """

        base = 0.45

        if risk >= 0.75:
            return 0.25
        elif risk >= 0.50:
            return 0.35
        elif risk >= 0.25:
            return 0.45
        else:
            return 0.60
        
    def _score_constraint_for_prompt(
        self,
        constraint_key: str,
        prompt_text: str,
        support: float,
        last_seen: int,
    ) -> Dict[str, Any]:
        prompt_tokens = self._norm_tokens(prompt_text)
        constraint_tokens = self._norm_tokens(constraint_key.replace("constraint::", ""))
        overlap = prompt_tokens.intersection(constraint_tokens)

        signals = self._prompt_signals(prompt_text)
        family = self._constraint_family(constraint_key)

        score = 0.0
        reasons: List[str] = []

        if family == "unverified_dependency_existence":
            if (
                signals["real_dependency_request"]
                or signals["mentions_library"]
                or signals["usage_request"]
                or signals["assumption_present"]
                or signals["database_usage_request"]
                or signals["cloud_usage_request"]
                or signals["ml_usage_request"]
            ):
                score += 0.55
                reasons.append("dependency_existence_context")
            if signals["explicit_fictional"]:
                score -= 0.45
                reasons.append("explicit_fictional_suppression")
            if re.search(r"[a-zA-Z0-9_-]+-[a-zA-Z0-9_-]+", prompt_text.lower()):
                score += 0.20
                reasons.append("hyphenated_unknown_entity")

        elif family == "unknown_artifact_uncertainty":
            if re.search(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+){2,}", prompt_text):
                score += 0.45
                reasons.append("artifact_like_uncertainty")
            if signals["usage_request"] or signals["cli_usage_request"] or signals["deploy_request"] or signals["install_request"]:
                score += 0.35
                reasons.append("operational_uncertainty")

        elif family == "unverified_cli_existence":
            if signals.get("mentions_cli") or signals["cli_usage_request"] or signals["deploy_request"]:
                score += 0.55
                reasons.append("cli_existence_context")
            if re.search(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+){2,}", prompt_text):
                score += 0.20
                reasons.append("hyphenated_unknown_tool")

        elif family == "cli_command_bypass":
            if signals["cli_usage_request"] or signals["deploy_request"] or signals["assumption_present"]:
                score += 0.65
                reasons.append("cli_command_or_flag_bypass_risk")

        elif family == "placeholder_api_bypass":
            if signals["asks_for_code"] or signals["usage_request"] or signals["assumption_present"]:
                score += 0.65
                reasons.append("placeholder_or_usage_bypass_risk")
            if signals["explicit_fictional"]:
                score -= 0.85
                reasons.append("explicit_fictional_suppression")
        
        elif family == "unverified_database_extension":
            if signals.get("mentions_database") or signals.get("database_usage_request"):
                score += 0.55
                reasons.append("database_extension_context")
            if re.search(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+){2,}", prompt_text):
                score += 0.20
                reasons.append("hyphenated_unknown_database_artifact")
            if signals.get("explicit_fictional"):
                score -= 0.45
                reasons.append("explicit_fictional_suppression")

        elif family == "database_syntax_bypass":
            if signals.get("database_usage_request") or signals.get("assumption_present"):
                score += 0.65
                reasons.append("database_syntax_or_config_bypass_risk")
            if signals.get("explicit_fictional"):
                score -= 0.85
                reasons.append("explicit_fictional_suppression")

        elif family == "unverified_cloud_service":
            if signals.get("mentions_cloud") or signals.get("cloud_usage_request") or signals.get("deploy_request"):
                score += 0.55
                reasons.append("cloud_service_context")
            if re.search(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+){2,}", prompt_text):
                score += 0.20
                reasons.append("hyphenated_unknown_cloud_artifact")
            if signals.get("explicit_fictional"):
                score -= 0.45
                reasons.append("explicit_fictional_suppression")

        elif family == "cloud_api_bypass":
            if signals.get("cloud_usage_request") or signals.get("deploy_request") or signals.get("assumption_present"):
                score += 0.65
                reasons.append("cloud_api_or_deployment_bypass_risk")
            if signals.get("explicit_fictional"):
                score -= 0.85
                reasons.append("explicit_fictional_suppression")

        elif family == "unverified_ml_artifact":
            if signals.get("mentions_ml") or signals.get("ml_usage_request"):
                score += 0.55
                reasons.append("ml_artifact_context")
            if re.search(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+){2,}", prompt_text):
                score += 0.20
                reasons.append("hyphenated_unknown_ml_artifact")
            if signals.get("explicit_fictional"):
                score -= 0.45
                reasons.append("explicit_fictional_suppression")

        elif family == "ml_api_bypass":
            if signals.get("ml_usage_request") or signals.get("assumption_present"):
                score += 0.65
                reasons.append("ml_api_or_model_usage_bypass_risk")
            if signals.get("explicit_fictional"):
                score -= 0.85
                reasons.append("explicit_fictional_suppression")

        elif family == "install_command_risk":
            if signals["install_request"]:
                score += 0.75
                reasons.append("install_request")
            else:
                score -= 0.20
                reasons.append("no_install_intent")

        elif family == "alternative_suggestion_policy":
            if signals["real_dependency_request"] or signals["usage_request"] or signals["mentions_library"]:
                score += 0.45
                reasons.append("real_usage_or_alternative_context")
            if signals["explicit_fictional"]:
                score -= 0.50
                reasons.append("explicit_fictional_suppression")

        else:
            if overlap:
                score += 0.15
                reasons.append("general_overlap")
            else:
                score -= 0.25
                reasons.append("no_relevance_signal")

        return {
            "key": constraint_key,
            "support": support,
            "last_seen_step": last_seen,
            "family": family,
            "applicability": score,
            "selected": score >= 0.45,
            "reasons": reasons,
        }

    def select_constraints_for_prompt(
        self,
        prompt_text: str,
        k_constraints: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []

        for n in self.graph.find_nodes("concept"):
            if not n.key or not n.key.startswith("constraint::"):
                continue

            support = float(n.data.get("support", 0.0))
            last_seen = int(n.data.get("last_seen_step", -1))

            rows.append(
                self._score_constraint_for_prompt(
                    constraint_key=n.key,
                    prompt_text=prompt_text,
                    support=support,
                    last_seen=last_seen,
                )
            )

        rows.sort(key=lambda r: (-float(r["applicability"]), -float(r["support"]), -int(r["last_seen_step"]), r["key"]))
        risk = self._compute_prompt_risk(prompt_text)
        routing_state = self._build_routing_state(prompt_text)
        signals = self._prompt_signals(prompt_text)
        

        global_threshold = self._dynamic_threshold(risk)

        for r in rows:
            allowed = self._family_route_allowed(
                family=str(r.get("family", "general_constraint")),
                routing_state=routing_state,
            )

            r["route"] = routing_state.get("route")
            r["route_allowed"] = allowed

            if not allowed:
                r["risk"] = risk
                r["global_threshold"] = global_threshold
                r["threshold"] = 1.0
                r["signals"] = signals
                r["selected"] = False
                r.setdefault("reasons", []).append("route_suppressed")
                continue

            family_threshold = self._family_threshold(
                family=str(r.get("family", "general_constraint")),
                risk=risk,
                signals=signals,
            )
            r["risk"] = risk
            r["global_threshold"] = global_threshold
            r["threshold"] = family_threshold
            r["signals"] = signals
            r["selected"] = r["applicability"] >= family_threshold
        
        self._last_prompt_risk = {
            "risk": risk,
            "threshold": global_threshold,
            "prompt": prompt_text,
            "signals": self._prompt_signals(prompt_text),
            "routing_state": routing_state,
        }

        selected = [r for r in rows if r["selected"]][:k_constraints]
        rejected = [r for r in rows if not r["selected"]]

        self.log.log(
            "constraint_selection",
            {
                "step": self.step,
                "prompt": prompt_text,
                "risk": risk,
                "global_threshold": global_threshold,
                "signals": signals,
                "routing_state": routing_state,
                "selected": selected,
                "rejected": rejected,
            },
        )

        return selected, rejected
    
    def get_context(self, prompt_text: str = "", k_constraints: int = 3, k_goals: int = 2, k_recent: int = 3) -> Dict[str, Any]:
        """
        Deterministic context selection (Phase 3):
        - top constraints (by concept support, tie-break by last_seen_step)
        - top goals
        - recent episode summaries (last k episodes, minimal)
        Everything is traceable.
        """
        concepts = self.graph.find_nodes("concept")

        def pick(kind: str, k: int):
            rows = []
            for n in concepts:
                if not n.key:
                    continue
                # concept kind is "emergent"; infer from feature prefix for now
                # keys look like: constraint::..., goal::..., task::..., entity::...
                if not n.key.startswith(f"{kind}::"):
                    continue
                support = float(n.data.get("support", 0.0))
                last_seen = int(n.data.get("last_seen_step", -1))
                promoted = int(n.data.get("promoted_step", -1))
                rows.append((n.key, support, last_seen, promoted))
            # deterministic ordering: support desc, last_seen desc, promoted desc, key asc
            rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
            return rows[:k]

        routing_state = self._build_routing_state(prompt_text)
        selected_constraints, rejected_constraints = self.select_constraints_for_prompt(
            prompt_text=prompt_text,
            k_constraints=k_constraints,
        )
        goals = pick("goal", k_goals)

        # recent episodes: use episode node metadata only (no embeddings, no magic)
        eps = []
        for ep_key in self._episode_order[-k_recent:]:
            ep = self.graph.find_node_by_key("episode", ep_key)
            if not ep:
                continue
            eps.append({
                "episode_id": ep_key,
                "status": ep.data.get("status"),
                "opened_step": ep.data.get("opened_step"),
                "closed_step": ep.data.get("closed_step"),
            })

        bundle = build_context_bundle(
            step=self.step,
            constraints=[
                {
                "key": c["key"],
                "support": c["support"],
                "last_seen_step": c["last_seen_step"],
                "family": c["family"],
                "applicability": c["applicability"],
                "risk": c.get("risk", 0.0),
                "threshold": c.get("threshold", 0.45),
                "global_threshold": c.get("global_threshold", 0.45),
                "selected": c.get("selected", False),
                "signals": c.get("signals", {}),
                "route": c.get("route"),
                "route_allowed": c.get("route_allowed", True),
                "reasons": c["reasons"],
            }
                for c in selected_constraints
            ],
            goals=[{"key": k, "support": s, "last_seen_step": ls} for (k, s, ls, _) in goals],
            concepts=[],  # keep empty for v1 until top-k non-constraint/goals later
            recent_episodes=eps,
            evidence=[],
            routing_state=routing_state,
        )
        return bundle


    def get_context_json(self, prompt_text: str = ""):
        """
        Deterministic JSON context bundle (Phase 1.2).
        Stored on self._last_context_json for logging.
        When governance ablation is enabled, the returned bundle is empty so the
        actor receives no policy-memory guidance.
        """
        if self.ablate_governance:
            ctx = {
                "step": self.step,
                "constraints": [],
                "goals": [],
                "concepts": [],
                "recent_episodes": [],
                "evidence": [],
                "meta": {"ablation": "governance_disabled"},
            }
        else:
            ctx = self.get_context(prompt_text=prompt_text)
        self._last_context_json = serialize_context(ctx)
        return self._last_context_json


    def get_context_text(self, prompt_text: str = "") -> str:
        """
        Human-readable context block injected into the actor.
        Keep it small, deterministic, and citeable.
        When governance ablation is enabled, no context is injected.
        """
        if self.ablate_governance:
            self.get_context_json()
            return ""

        ctx = self.get_context(prompt_text=prompt_text)
        # ensure JSON is always available for logs/audits
        self._last_context_json = serialize_context(ctx)
        risk_info = getattr(self, "_last_prompt_risk", {}) or {}
        print("\n[PROMPT RISK]")
        print(
            f"risk={float(risk_info.get('risk', 0.0)):.2f} | "
            f"threshold={float(risk_info.get('threshold', 0.0)):.2f} | "
            f"signals={risk_info.get('signals', {})}"
        )
        print("[/PROMPT RISK]\n")

        routing_state = ctx.get("routing_state", {})

        print("\n[ROUTING]")
        print(
            f"route={routing_state.get('route')} | "
            f"domain={routing_state.get('domain')} | "
            f"domain_confidence={float(routing_state.get('domain_confidence', 0.0)):.2f} | "
            f"artifact_like={routing_state.get('artifact_like')} | "
            f"operational_intent={routing_state.get('operational_intent')} | "
            f"clarification_needed={routing_state.get('clarification_needed')}"
        )
        print("[/ROUTING]\n")
        # Debug visibility for Iteration 3:
        # Print only the constraints that are actually being injected.
        if ctx["constraints"]:
            print("\n[INJECTED CONSTRAINTS]")
            for c in ctx["constraints"]:
                print(
            f"- {c['key']} | "
            f"family={c.get('family')} | "
            f"applicability={c.get('applicability', 0.0):.2f} | "
            f"support={c['support']:.2f} | "
            f"risk={c.get('risk', 0.0):.2f} | "
            f"threshold={c.get('threshold', 0.0):.2f} | "
            f"route={c.get('route')} | "
            f"route_allowed={c.get('route_allowed')} | "
            f"reasons={','.join(c.get('reasons', []))}"
        )
            print("[/INJECTED CONSTRAINTS]\n")
        else:
            print("\n[INJECTED CONSTRAINTS]\n- none\n[/INJECTED CONSTRAINTS]\n")

        lines = []

        routing_state = ctx.get("routing_state", {})

        if routing_state.get("clarification_needed"):
            lines.append("Routing guidance:")
            lines.append(
                "- The user referenced an unfamiliar artifact and the artifact type is not clear enough for safe domain-specific action. "
                "Ask a clarifying question or state uncertainty before providing commands, imports, APIs, flags, configuration, deployment steps, or usage."
            )

        if ctx["constraints"]:
            lines.append("Active constraints (highest support):")
            for c in ctx["constraints"]:
                lines.append(
                f"- {c['key']} | support={c['support']:.2f} "
                f"| applicability={c.get('applicability', 0.0):.2f} "
                f"| family={c.get('family', 'unknown')}"
            )
        if ctx["goals"]:
            lines.append("Active goals (highest support):")
            for g in ctx["goals"]:
                lines.append(f"- {g['key']} | support={g['support']:.2f} | last_seen_step={g['last_seen_step']}")
        return "\n".join(lines).strip()


    def trace_last(self, n: int=5):
        out = []
        for ev_key in self._event_order[-n:]:
            ev = self.graph.find_node_by_key("event", ev_key)
            if not ev:
                continue
            ep_key = None
            for e, src in self.graph.neighbors_in(ev.id, edge_type="contains"):
                if src.type == "episode":
                    ep_key = src.key
                    break
            feats = []
            for e, feat in self.graph.neighbors_out(ev.id, edge_type="has_feature"):
                feats.append({"key": feat.key, "kind": feat.data.get("kind"), "conf": e.confidence, "sal": float(e.data.get("salience", 0.0))})
            feats.sort(key=lambda x: x["sal"], reverse=True)
            out.append({"event_id": ev_key, "role": ev.data.get("role"), "episode_id": ep_key, "text": ev.data.get("text"), "features": feats[:12], "intent": ev.data.get("intent"), "meta": ev.data.get("extraction_meta")})
        return out
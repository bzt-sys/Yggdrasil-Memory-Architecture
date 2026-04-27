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
        for e in list(ents)[:6]:
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
        self._last_context_json = ""
        self.ablate_governance: bool = False

        self.step: int = 0
        self._current_episode_id: Optional[str] = None
        self._episode_order: List[str] = []
        self._event_order: List[str] = []
        self._feature_counts: Dict[str, int] = {}
        self._feature_support: Dict[str, float] = {}
        self._last_step_metrics: Dict[str, Any] = {"decayed": 0, "deleted_edges": 0, "compressed_edges": 0, "promoted": 0}

    def start_step(self) -> None:
        self._last_step_metrics = {"decayed": 0, "deleted_edges": 0, "compressed_edges": 0, "promoted": 0}

    def write_event(self, role: str, text: str, source: str="cli", provenance: Optional[Dict[str, Any]]=None) -> str:
        ev_id = str(uuid.uuid4())
        ev_node = self.graph.upsert_node("event", key=ev_id, data={
            "session_id": self.session_id,
            "role": role,
            "source": source,
            "text": text,
            "provenance": provenance or {},
            "initial_confidence": float((provenance or {}).get("confidence", 1.0)),
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

    def write_outcome(self, score: float, label: str="", tags: Optional[List[str]]=None) -> str:
        out_id = str(uuid.uuid4())
        out_node = self.graph.upsert_node("outcome", key=out_id, data={"session_id": self.session_id, "score": float(score), "label": label, "tags": tags or []}, ts=time.time())

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
        self._decay_detail_edges()
        self._compress_promoted_features()
        self._failsafe_if_needed()
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
            lst.sort(key=lambda x: x[1], reverse=True)
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
    
    def get_context(self, k_constraints: int = 3, k_goals: int = 2, k_recent: int = 3) -> Dict[str, Any]:
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

        constraints = pick("constraint", k_constraints)
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
            constraints=[{"key": k, "support": s, "last_seen_step": ls} for (k, s, ls, _) in constraints],
            goals=[{"key": k, "support": s, "last_seen_step": ls} for (k, s, ls, _) in goals],
            concepts=[],  # keep empty for v1 until top-k non-constraint/goals later
            recent_episodes=eps,
            evidence=[],
        )
        return bundle


    def get_context_json(self) -> str:
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
            ctx = self.get_context()
        self._last_context_json = serialize_context(ctx)
        return self._last_context_json


    def get_context_text(self) -> str:
        """
        Human-readable context block injected into the actor.
        Keep it small, deterministic, and citeable.
        When governance ablation is enabled, no context is injected.
        """
        if self.ablate_governance:
            self.get_context_json()
            return ""

        ctx = self.get_context()
        # ensure JSON is always available for logs/audits
        self._last_context_json = serialize_context(ctx)

        lines = []
        if ctx["constraints"]:
            lines.append("Active constraints (highest support):")
            for c in ctx["constraints"]:
                lines.append(f"- {c['key']} | support={c['support']:.2f} | last_seen_step={c['last_seen_step']}")
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
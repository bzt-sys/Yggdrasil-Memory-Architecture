from __future__ import annotations

"""Selective information retrieval for knowledge-limited tasks.

The interface is intentionally provider-agnostic. The first implementation is
an auditable deterministic local corpus. When no provider or no relevant hit is
available, the tool returns an explicit unresolved information gap rather than
encouraging the actor to invent missing facts.
"""

from dataclasses import asdict, dataclass
from typing import Any, Optional

from imp.environment.corpus import LocalCorpus


@dataclass
class InformationRetrievalResult:
    invoked: bool
    provider: str
    query: str
    requested_source_types: list[str]
    hits: list[dict[str, Any]]
    resolved: bool
    unresolved_gaps: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_retrieval_query(prompt: str, task_analysis: Optional[dict[str, Any]] = None) -> str:
    analysis = task_analysis or {}
    plan = analysis.get("plan") or {}
    gaps = [str(x).strip() for x in (plan.get("information_gaps") or []) if str(x).strip()]
    goal = str(plan.get("goal") or "").strip()
    parts = gaps[:4] + ([goal] if goal else []) + [str(prompt or "")[:1800]]
    return "\n".join(p for p in parts if p).strip()


def retrieve_information(
    prompt: str,
    *,
    task_analysis: Optional[dict[str, Any]] = None,
    corpus: Optional[LocalCorpus] = None,
    max_hits: int = 3,
) -> InformationRetrievalResult:
    analysis = task_analysis or {}
    needs_external = bool(analysis.get("needs_external_information"))
    plan = analysis.get("plan") or {}
    gaps = [str(x)[:240] for x in (plan.get("information_gaps") or []) if str(x).strip()]
    query = build_retrieval_query(prompt, analysis)

    if not needs_external:
        return InformationRetrievalResult(
            invoked=False,
            provider="none",
            query=query,
            requested_source_types=[],
            hits=[],
            resolved=False,
            unresolved_gaps=[],
            reason="task_analysis_did_not_identify_external_knowledge_need",
        )

    requested = ["local_documentation", "reference_material"]
    if corpus is None or not corpus.docs:
        return InformationRetrievalResult(
            invoked=True,
            provider="local_corpus",
            query=query,
            requested_source_types=requested,
            hits=[],
            resolved=False,
            unresolved_gaps=gaps or ["External information was requested, but no retrieval source is configured."],
            reason="no_retrieval_source_available",
        )

    raw_hits = corpus.search(query, k=max(1, int(max_hits)))
    hits = [
        {
            "hit_id": h.hit_id,
            "source": h.source,
            "title": h.title,
            "text": h.text[:1800],
            "score": h.score,
            "tags": list(h.tags),
        }
        for h in raw_hits
    ]
    return InformationRetrievalResult(
        invoked=True,
        provider="local_corpus",
        query=query,
        requested_source_types=requested,
        hits=hits,
        resolved=bool(hits),
        unresolved_gaps=[] if hits else (gaps or ["No relevant source was found in the configured corpus."]),
        reason="relevant_sources_found" if hits else "no_relevant_source_found",
    )


def format_information_retrieval_context(result: InformationRetrievalResult) -> str:
    if not result.invoked:
        return ""
    if not result.hits:
        return (
            "INFORMATION RETRIEVAL TOOL OUTPUT (advisory)\n"
            "No grounded source was available for the identified knowledge gap. "
            "Do not invent missing facts; solve only from the task contract or explicitly preserve uncertainty."
        )
    lines = ["INFORMATION RETRIEVAL TOOL OUTPUT (grounded reference; advisory)"]
    for idx, hit in enumerate(result.hits, start=1):
        lines.append(f"SOURCE {idx}: {hit.get('title')} | {hit.get('source')} | score={hit.get('score')}")
        lines.append(str(hit.get("text") or "")[:1800])
    return "\n\n".join(lines)

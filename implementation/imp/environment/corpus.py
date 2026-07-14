"""Local corpus scaffold for controlled external knowledge lookup."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class CorpusHit:
    hit_id: str
    source: str
    title: str
    text: str
    score: float = 0.0
    tags: List[str] = field(default_factory=list)


class LocalCorpus:
    """Tiny deterministic local text corpus.

    Intended first use: non-answer Python reference snippets such as exception
    definitions, import rules, assert semantics, and debugging principles.
    """

    def __init__(self, docs: Optional[Iterable[CorpusHit]] = None) -> None:
        self.docs: List[CorpusHit] = list(docs or [])

    @classmethod
    def from_directory(cls, path: str | Path) -> "LocalCorpus":
        root = Path(path)
        docs: List[CorpusHit] = []
        if not root.exists():
            return cls([])
        for p in sorted(root.rglob("*.md")) + sorted(root.rglob("*.txt")):
            text = p.read_text(encoding="utf-8", errors="replace")
            title = p.stem.replace("_", " ")
            docs.append(CorpusHit(hit_id=str(uuid.uuid4()), source=str(p), title=title, text=text[:6000], score=0.0))
        return cls(docs)

    def search(self, query: str, *, k: int = 4) -> List[CorpusHit]:
        terms = [t for t in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(t) >= 3]
        rows: List[CorpusHit] = []
        for d in self.docs:
            blob = f"{d.title}\n{d.text}".lower()
            score = 0.0
            for t in terms:
                score += blob.count(t)
            if score <= 0:
                continue
            rows.append(CorpusHit(hit_id=d.hit_id, source=d.source, title=d.title, text=d.text, score=float(score), tags=list(d.tags)))
        rows.sort(key=lambda h: h.score, reverse=True)
        return rows[: int(k)]

"""Groundedness signal computation for retrieval probes.

All features are cheap (string/regex over already-retrieved chunks). No LLM
calls. The combined scalar g(probe) drives topology selection deterministically.
"""
from __future__ import annotations
import re
from .types import RetrievedChunk

_CAPITALISED = re.compile(r'\b[A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+)*')
_DATE_LIKE = re.compile(r'\b(?:1[6-9]\d{2}|20\d{2})\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b', re.IGNORECASE)
_NUMBER_LIKE = re.compile(r'\b\d+(?:\.\d+)?\b')


def _named_entities(text: str) -> set[str]:
    return {m.group(0).lower() for m in _CAPITALISED.finditer(text)}


def ne_coverage(question: str, top1_text: str) -> float:
    qe = _named_entities(question)
    te = _named_entities(top1_text)
    if not qe:
        return 0.0
    overlap = sum(1 for e in qe if any(e == t or e in t or t in e for t in te))
    return overlap / len(qe)


def wh_target_extractable(question: str, top1_text: str) -> bool:
    """Cheap heuristic: does top-1 contain a span of the expected answer type?"""
    q = question.lower()
    if any(w in q for w in ('when', 'what year', 'what date', 'what month')):
        return bool(_DATE_LIKE.search(top1_text))
    if any(w in q for w in ('how many', 'how much', 'number of')):
        return bool(_NUMBER_LIKE.search(top1_text))
    return bool(_CAPITALISED.search(top1_text))


def compute_groundedness(question: str, chunks: list[RetrievedChunk]) -> tuple[float, dict[str, float]]:
    """Returns (g, components) where g in [0,1] is a combined groundedness score."""
    if not chunks:
        return 0.0, {'top1_score': 0.0, 'score_gap_1to5': 0.0, 'ne_coverage': 0.0, 'wh_target_extractable': 0.0}
    top1 = chunks[0]
    top1_score = float(top1.score)
    if len(chunks) >= 2:
        score_gap = max(0.0, top1.score - chunks[-1].score)
    else:
        score_gap = 0.0
    ne = ne_coverage(question, top1.text)
    wh = 1.0 if wh_target_extractable(question, top1.text) else 0.0
    norm_score = max(0.0, min(1.0, (top1_score - 0.5) / 0.5))
    g = 0.4 * norm_score + 0.2 * (score_gap * 5) + 0.2 * ne + 0.2 * wh
    g = max(0.0, min(1.0, g))
    components = {
        'top1_score': top1_score,
        'score_gap_1to5': score_gap,
        'ne_coverage': ne,
        'wh_target_extractable': wh,
    }
    return g, components

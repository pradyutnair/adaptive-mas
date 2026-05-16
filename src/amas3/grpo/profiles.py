"""Canonical 5-class reasoning-type profiling for cross-dataset transfer.

The profile string is used as the indexing key for the experience library and
must be dataset-agnostic so a ``bridge`` insight learned on HotpotQA can fire
on 2WikiQA bridge questions too.

Priority:
  1. GPT-4o annotation cache loaded from ``data/annotations/annot_*.jsonl``
     (produced by ``scripts/annotate_profiles.py``).
  2. Keyword heuristic on the question text.

Vocabulary: ``bridge`` / ``intersection`` / ``temporal`` / ``causal`` / ``any``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROFILE_CLASSES: tuple[str, ...] = ("bridge", "intersection", "temporal", "causal", "any")

_ANNOT_CACHE: dict[str, str] | None = None


def _annotation_search_paths() -> list[Path]:
    """Directories searched for GPT-4o reasoning-type annotations."""
    candidates: list[Path] = []
    env_dir = os.environ.get("AMAS_ANNOTATIONS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "data" / "annotations")
    candidates.append(Path("data/annotations"))
    return candidates


def _load_annot_cache() -> dict[str, str]:
    """Load qid -> canonical reasoning-type once per process."""
    global _ANNOT_CACHE
    if _ANNOT_CACHE is not None:
        return _ANNOT_CACHE
    cache: dict[str, str] = {}
    for cand in _annotation_search_paths():
        if not cand.is_dir():
            continue
        loaded = False
        for jsonl in cand.glob("annot_*.jsonl"):
            with jsonl.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = str(d.get("id", "")).strip()
                    rt = str(d.get("reasoning_type", "")).strip().lower()
                    if qid and rt in PROFILE_CLASSES:
                        cache[qid] = rt
                        loaded = True
        if loaded:
            break
    _ANNOT_CACHE = cache
    return cache


def characterize_query_profile(
    question: str,
    dataset: str = "default",
    qid: str | None = None,
) -> str:
    """Return the canonical 5-class label for ``question``.

    ``dataset`` is accepted only so reward shaping can still read the
    per-dataset baseline; it is intentionally NOT encoded into the label so
    insights transfer across datasets sharing the same reasoning type.
    """
    if qid:
        cache = _load_annot_cache()
        annotated = cache.get(str(qid).strip())
        if annotated:
            return annotated
    q = (question or "").lower()
    if any(x in q for x in (
        "why ", "cause", " led to", " result in", " due to", " because ",
        " effect of ", " consequence ",
    )):
        return "causal"
    # Intersection / comparison is checked before temporal because comparison
    # questions ("who was born earlier, X or Y?") often contain temporal
    # vocabulary; the comparative form is more specific so it wins.
    if any(x in q for x in (
        "compare", "which of", "which one", "larger", "smaller", "older",
        "younger", "earlier", "later", " or ", "both ", "each ",
        " same ", " different",
    )):
        return "intersection"
    if any(x in q for x in (
        "when ", "what year", "what date", "what month", " before ", " after ",
        " during ", "first ", "last ",
    )):
        return "temporal"
    if len(q.split()) >= 12 and any(x in q for x in (
        " of ", " by ", "who wrote", "directed by", "starred", "featured in",
        "founder of", "author of", "creator of", "produced by",
    )):
        return "bridge"
    return "any"

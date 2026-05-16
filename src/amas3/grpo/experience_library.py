"""Structured experience library for HERA-style experience-guided orchestration.

Stores actionable insights learned from HERA-style training. Supports semantic
retrieval with utility/diversity scoring, JSON persistence, and prompt formatting.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[a-z0-9]+")
MAX_EXPERIENCE_WORDS = 32
STOP_WORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "what",
    "which", "who", "when", "where", "is", "are", "was", "were", "did",
    "does", "with", "by", "from", "that", "this", "how", "why",
})


@dataclass
class ExperienceEntry:
    id: str
    profile: str
    insight: str
    utility: float = 0.5
    target_roles: tuple[str, ...] = ("planner", "solver", "synthesizer")
    applies_when: str = ""
    avoid_when: str = ""
    usage_count: int = 0
    success_count: int = 0
    embedding: list[float] = field(default_factory=list)


_EMBEDDER: Any | None = None
_EMBEDDER_FAILED = False


def words(text: str) -> set[str]:
    """Extract content words from text, filtering stopwords and short tokens."""
    return {w for w in WORD_RE.findall((text or "").lower())
            if len(w) > 2 and w not in STOP_WORDS}


def cap_experience_text(text: str, max_words: int = MAX_EXPERIENCE_WORDS) -> str:
    """Keep learned experiences compact, as TF-GRPO uses short reusable lessons."""
    pieces = (text or "").split()
    if len(pieces) <= max_words:
        return " ".join(pieces)
    return " ".join(pieces[:max_words]).rstrip(" ,;:.") + "."


def _entry_text(entry: ExperienceEntry) -> str:
    return " ".join([entry.profile, entry.insight, entry.applies_when, entry.avoid_when]).strip()


def _embedder():
    global _EMBEDDER, _EMBEDDER_FAILED
    if _EMBEDDER_FAILED:
        return None
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        cache_folder = os.environ.get("AMAS_EXPERIENCE_CACHE", "/local/yzheng/pnair/.cache/huggingface")
        Path(cache_folder).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_folder)
        os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_folder) / "hub"))
        os.environ.setdefault("HF_XET_CACHE", str(Path(cache_folder) / "xet"))
        os.environ.setdefault("XDG_CACHE_HOME", "/local/yzheng/pnair/.cache")
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("AMAS_EXPERIENCE_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
        _EMBEDDER = SentenceTransformer(model_name, cache_folder=cache_folder)
        return _EMBEDDER
    except Exception:
        _EMBEDDER_FAILED = True
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _lexical_similarity(question: str, entry: ExperienceEntry) -> float:
    qw = words(question)
    ew = words(_entry_text(entry))
    return len(qw & ew) / max(1, len(qw | ew))


class ExperienceLibrary:
    """Structured collection of experience entries with retrieval and management."""

    def __init__(self, entries: list[ExperienceEntry] | None = None) -> None:
        self.entries: dict[str, ExperienceEntry] = {}
        if entries:
            for e in entries:
                self.entries[e.id] = e
        self.next_id_counter = len(self.entries) + 1

    def __len__(self) -> int:
        return self.size()

    def generate_id(self) -> str:
        eid = f"E{self.next_id_counter:03d}"
        self.next_id_counter += 1
        while eid in self.entries:
            eid = f"E{self.next_id_counter:03d}"
            self.next_id_counter += 1
        return eid

    def add(self, entry: ExperienceEntry) -> str:
        """Add a new entry. Assigns an ID if empty. Returns the entry ID."""
        if not entry.id:
            entry.id = self.generate_id()
        capped = cap_experience_text(entry.insight)
        if capped != entry.insight:
            entry.insight = capped
            entry.embedding = []
        self._ensure_embedding(entry)
        self.entries[entry.id] = entry
        return entry.id

    def modify(self, target_id: str, revised: ExperienceEntry) -> None:
        """Refine an existing entry in place, matching TF-GRPO modify operations."""
        if target_id not in self.entries:
            self.add(revised)
            return
        target = self.entries[target_id]
        if revised.profile:
            target.profile = revised.profile
        if revised.insight:
            target.insight = cap_experience_text(revised.insight)
        if revised.applies_when:
            target.applies_when = revised.applies_when
        if revised.avoid_when:
            target.avoid_when = revised.avoid_when
        if revised.target_roles:
            target.target_roles = tuple(revised.target_roles)
        target.utility = max(target.utility, revised.utility)
        target.embedding = []
        self._ensure_embedding(target)

    def merge(self, target_id: str, source: ExperienceEntry) -> None:
        """Merge source insight into existing entry, compacting duplicate guidance."""
        if target_id not in self.entries:
            self.add(source)
            return
        target = self.entries[target_id]
        target_usage = target.usage_count
        source_usage = source.usage_count
        total = target_usage + source_usage
        target.insight = cap_experience_text(self._compact_merge_text(target.insight, source.insight))
        target.applies_when = self._compact_merge_text(target.applies_when, source.applies_when, max_chars=140)
        target.avoid_when = self._compact_merge_text(target.avoid_when, source.avoid_when, max_chars=120)
        if total > 0:
            target.utility = (target.utility * target_usage + source.utility * source_usage) / total
        target.usage_count = total
        target.success_count += source.success_count
        combined_roles = set(target.target_roles) | set(source.target_roles)
        target.target_roles = tuple(sorted(combined_roles))
        target.embedding = []
        self._ensure_embedding(target)

    def prune(self, entry_id: str) -> None:
        """Remove an entry by ID."""
        self.entries.pop(entry_id, None)

    def keep(self, entry_id: str) -> None:
        """Explicitly keep an entry (no-op, used for logging clarity in update loops)."""
        pass

    def retrieve(
        self, question: str, role: str = "", limit: int = 5
    ) -> list[ExperienceEntry]:
        """Retrieve top entries by semantic similarity + utility + diversity."""
        if not self.entries:
            return []
        role = (role or "").lower()
        embedder = _embedder()
        q_embedding: list[float] = []
        if embedder is not None:
            try:
                q_embedding = embedder.encode([question], normalize_embeddings=True)[0].tolist()
            except Exception:
                q_embedding = []

        candidates: list[tuple[float, ExperienceEntry]] = []

        for entry in self.entries.values():
            sem = 0.0
            if q_embedding:
                self._ensure_embedding(entry)
                sem = _cosine(q_embedding, entry.embedding)
            if sem <= 0.0:
                sem = _lexical_similarity(question, entry)
            role_bonus = 0.08 if (not role or not entry.target_roles or role in entry.target_roles or "all" in entry.target_roles) else -0.04
            utility_weight = 0.25 * entry.utility
            usage_penalty = 0.04 * math.log1p(max(0, entry.usage_count))
            score = 0.65 * sem + utility_weight + role_bonus - usage_penalty
            candidates.append((score, entry))

        selected: list[ExperienceEntry] = []
        remaining = candidates[:]
        while remaining and len(selected) < max(0, limit):
            best_idx = 0
            best_score = -1e9
            for idx, (base_score, entry) in enumerate(remaining):
                diversity_penalty = 0.0
                if selected:
                    self._ensure_embedding(entry)
                    diversity_penalty = max(_cosine(entry.embedding, s.embedding) for s in selected)
                mmr_score = base_score - 0.20 * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            _, chosen = remaining.pop(best_idx)
            self._ensure_embedding(chosen)
            selected.append(chosen)
        return selected

    def retrieve_for_orchestrator(
        self,
        question: str,
        limit: int = 5,
        similarity_floor: float = 0.05,
        diversity_threshold: float = 0.85,
    ) -> list[ExperienceEntry]:
        """HERA Algorithm 4: experience-guided topology sampling retrieval.

        For each (c', z, u) in E sorted by u descending:
          - keep if SIMILAR(c', c) and NOT REDUNDANT(z, E_rel)
        balancing utility with diversity. No flat thresholds, no per-type
        tables: just utility-prioritized semantic retrieval over E.
        """
        if not self.entries:
            return []
        embedder = _embedder()
        q_embedding: list[float] = []
        if embedder is not None:
            try:
                q_embedding = embedder.encode([question], normalize_embeddings=True)[0].tolist()
            except Exception:
                q_embedding = []

        scored: list[tuple[ExperienceEntry, float]] = []
        for entry in self.entries.values():
            self._ensure_embedding(entry)
            sim = _cosine(q_embedding, entry.embedding) if q_embedding else _lexical_similarity(question, entry)
            if sim < similarity_floor:
                continue
            scored.append((entry, sim))

        scored.sort(key=lambda x: (-x[0].utility, -x[1]))

        selected: list[ExperienceEntry] = []
        for entry, _sim in scored:
            redundant = False
            for already in selected:
                self._ensure_embedding(already)
                if _cosine(entry.embedding, already.embedding) >= diversity_threshold:
                    redundant = True
                    break
            if redundant:
                continue
            selected.append(entry)
            if len(selected) >= limit:
                break
        return selected

    def _ensure_embedding(self, entry: ExperienceEntry) -> None:
        if entry.embedding:
            return
        embedder = _embedder()
        if embedder is None:
            return
        try:
            entry.embedding = embedder.encode([_entry_text(entry)], normalize_embeddings=True)[0].tolist()
        except Exception:
            entry.embedding = []

    @staticmethod
    def _compact_merge_text(a: str, b: str, max_chars: int = 240) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for raw in (a, b):
            for piece in re.split(r"\s+(?:Additionally:|Also:)\s+|[.;]\s+", raw or ""):
                text = " ".join(piece.split()).strip(" .;")
                if not text:
                    continue
                key = " ".join(sorted(words(text)))
                if key in seen:
                    continue
                seen.add(key)
                parts.append(text)
        merged = "; ".join(parts)
        return merged[:max_chars].rstrip(" ;")

    def update_utility(self, entry_id: str, success: bool) -> None:
        """Update utility score based on whether the entry led to a successful outcome."""
        if entry_id not in self.entries:
            return
        entry = self.entries[entry_id]
        entry.usage_count += 1
        if success:
            entry.success_count += 1
        if entry.usage_count > 0:
            entry.utility = entry.success_count / entry.usage_count

    def save(self, path: str | Path) -> None:
        """Serialize library to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        data = self.to_serializable()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ExperienceLibrary:
        """Deserialize library from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        entries = []
        if isinstance(data, dict):
            data = data.get("entries", [])
        for raw in data:
            d = dict(raw)
            if "text" in d and "insight" not in d:
                d["insight"] = d.pop("text")
            if "eid" in d and "id" not in d:
                d["id"] = d.pop("eid")
            d.setdefault("id", "")
            d.setdefault("profile", "general")
            d.setdefault("insight", "")
            d["target_roles"] = tuple(d.get("target_roles", d.get("roles", ())))
            allowed = {
                "id", "profile", "insight", "utility", "target_roles",
                "applies_when", "avoid_when", "usage_count", "success_count",
                "embedding",
            }
            entries.append(ExperienceEntry(**{k: v for k, v in d.items() if k in allowed}))
        lib = cls(entries)
        return lib

    def size(self) -> int:
        return len(self.entries)

    def to_text(self) -> str:
        """Serialize all entries as newline-delimited JSON (for config injection)."""
        lines = []
        for entry in self.entries.values():
            d = {
                "profile": entry.profile,
                "insight": entry.insight,
                "utility": round(entry.utility, 2),
                "roles": list(entry.target_roles),
                "applies_when": entry.applies_when,
                "avoid_when": entry.avoid_when,
            }
            lines.append(json.dumps(d))
        return "\n".join(lines)

    def to_serializable(self) -> list[dict]:
        """Return JSON-serializable entries."""
        data = []
        for entry in self.entries.values():
            d = asdict(entry)
            d["target_roles"] = list(entry.target_roles)
            data.append(d)
        return data

    def format_for_prompt(
        self, question: str = "", role: str = "", limit: int = 5
    ) -> str:
        """Retrieve and format entries for a prompt."""
        entries = self.retrieve(question, role=role, limit=limit) if question else list(self.entries.values())[:limit]
        return format_for_prompt(entries, max_entries=limit)


def format_for_prompt(
    entries: list[ExperienceEntry],
    max_entries: int = 5,
    max_insight_chars: int = 300,
) -> str:
    """Format selected experience entries for injection into LLM prompts.

    Caps each insight to max_insight_chars to bound total token overhead.
    """
    if not entries:
        return ""
    lines = []
    for entry in entries[:max_entries]:
        insight = entry.insight[:max_insight_chars]
        lines.append(insight)
    return "Rules: " + " | ".join(lines)


def format_for_orchestrator(
    entries: list[ExperienceEntry],
    max_entries: int = 5,
    max_insight_chars: int = 200,
) -> str:
    """Format retrieved experiences for pi_O exactly as HERA Appendix B states.

    Each entry block has the canonical fields:
      - Query Type: {profile}
        Insight: {insight}
        Utility score: {utility}

    plus optional applies_when / avoid_when conditions when present. This
    makes pi_O reason over both the rule and its empirical confidence.
    """
    if not entries:
        return "(no prior experiences)"
    blocks = []
    for entry in entries[:max_entries]:
        insight = entry.insight[:max_insight_chars]
        parts = [
            f"- Query Type: {entry.profile}",
            f"  Insight: {insight}",
            f"  Utility score: {entry.utility:.2f}",
        ]
        if entry.applies_when:
            parts.append(f"  Applies when: {entry.applies_when}")
        if entry.avoid_when:
            parts.append(f"  Avoid when: {entry.avoid_when}")
        blocks.append("\n".join(parts))
    return "\n".join(blocks)

"""HERA experience library: Profile-Insight-Utility entries with ADD / MERGE / PRUNE / KEEP."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Paper §D.1: 6 question types (bridge / intersection / comparison / temporal / causal / ambiguous).
# `any` is internal default for unannotated/cross-type insights.
PROFILES = (
    "any",
    "bridge",
    "intersection",
    "comparison",
    "temporal",
    "causal",
    "ambiguous",
)

# Loaded lazily from data/annotations/*.jsonl when available.
_ANNOTATION_CACHE: dict[str, str] | None = None


def _load_annotations(annot_dir: str | None = None) -> dict[str, str]:
    import os
    global _ANNOTATION_CACHE
    if _ANNOTATION_CACHE is not None:
        return _ANNOTATION_CACHE
    cache: dict[str, str] = {}
    if annot_dir is None:
        env = os.getenv("AMAS_ANNOTATIONS_DIR")
        if env:
            annot_dir = env
        else:
            for cand in [
                "/projects/prjs1800/hera/data/annotations",
                "/local/yzheng/pnair/workspace/hera/data/annotations",
            ]:
                if os.path.isdir(cand):
                    annot_dir = cand
                    break
    if not annot_dir:
        _ANNOTATION_CACHE = cache
        return cache
    p = __import__("pathlib").Path(annot_dir)
    if p.exists():
        for f in p.glob("annot_*.jsonl"):
            try:
                with open(f) as fh:
                    for line in fh:
                        d = json.loads(line)
                        qid = str(d.get("id", ""))
                        rt = str(d.get("reasoning_type", ""))
                        if qid and rt in PROFILES:
                            cache[qid] = rt
            except Exception:
                continue
    _ANNOTATION_CACHE = cache
    return cache


@dataclass
class ExpEntry:
    id: str
    profile: str  # query type / characteristic
    insight: str
    utility: int = 0
    uses: int = 0
    rationale: str = ""

    def utility_rate(self) -> float:
        return self.utility / max(1, self.uses or 1)


@dataclass
class ExperienceLibrary:
    entries: list[ExpEntry] = field(default_factory=list)
    next_num: int = 1
    max_entries: int = 30

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceLibrary":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
        except Exception:
            return cls()
        entries = [
            ExpEntry(
                id=str(x.get("id", "")),
                profile=str(x.get("profile", "any")),
                insight=str(x.get("insight", "")),
                utility=int(x.get("utility", 0)),
                uses=int(x.get("uses", 0)),
                rationale=str(x.get("rationale", "")),
            )
            for x in data.get("entries", [])
        ]
        return cls(
            entries=entries,
            next_num=int(data.get("next_num", len(entries) + 1)),
            max_entries=int(data.get("max_entries", 30)),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_num": self.next_num,
            "max_entries": self.max_entries,
            "entries": [asdict(e) for e in self.entries],
        }

    def add(self, profile: str, insight: str, rationale: str = "") -> str:
        ins = insight.strip()
        if not ins:
            return ""
        prof = profile if profile in PROFILES else "any"
        # Skip near-duplicate.
        for e in self.entries:
            if e.profile == prof and _token_overlap(ins, e.insight) > 0.85:
                return e.id
        eid = f"E-{self.next_num:03d}"
        self.next_num += 1
        self.entries.append(ExpEntry(id=eid, profile=prof, insight=ins, rationale=rationale.strip()))
        self.prune()
        return eid

    def delete(self, eid: str) -> None:
        self.entries = [e for e in self.entries if e.id != eid]

    def merge(self, ids: list[str], merged_insight: str, profile: str = "any", rationale: str = "") -> str:
        ids_set = set(ids)
        kept = [e for e in self.entries if e.id not in ids_set]
        merged_id = f"E-{self.next_num:03d}"
        self.next_num += 1
        kept.append(ExpEntry(id=merged_id, profile=profile if profile in PROFILES else "any",
                             insight=merged_insight.strip(), rationale=rationale.strip()))
        self.entries = kept
        self.prune()
        return merged_id

    def reward(self, eid: str, hit: bool) -> None:
        for e in self.entries:
            if e.id == eid:
                e.uses += 1
                if hit:
                    e.utility += 1
                return

    def retrieve(self, profile: str, top_k: int = 5) -> list[ExpEntry]:
        if not self.entries:
            return []

        def score(e: ExpEntry) -> float:
            match = 2.0 if e.profile == profile else (1.0 if e.profile == "any" else 0.0)
            return match + e.utility_rate()

        rows = sorted(self.entries, key=score, reverse=True)
        # Diversity filter (>0.7 token overlap suppressed).
        out: list[ExpEntry] = []
        seen_tokens: list[set[str]] = []
        for e in rows:
            toks = set(re.findall(r"[a-z]+", e.insight.lower()))
            if any(len(toks & s) / max(1, len(toks)) > 0.7 for s in seen_tokens):
                continue
            out.append(e)
            seen_tokens.append(toks)
            if len(out) >= top_k:
                break
        return out

    def to_text(self, profile: str | None = None, top_k: int | None = None) -> str:
        rows = self.retrieve(profile, top_k=top_k or len(self.entries)) if profile else list(self.entries)
        return "\n".join(f"{e.id} [{e.profile}|u={e.utility}/{e.uses}]: {e.insight}" for e in rows)

    def to_paper_format(self, profile: str | None = None, top_k: int | None = None) -> str:
        """Bullet format per HERA paper Appendix B (Query Type / Insight / Utility score)."""
        rows = self.retrieve(profile, top_k=top_k or len(self.entries)) if profile else list(self.entries)
        if not rows:
            return ""
        blocks = []
        for e in rows:
            blocks.append(
                f"- id: {e.id}\n"
                f"  Query Type: {e.profile}\n"
                f"  Insight: {e.insight}\n"
                f"  Utility score: {e.utility}/{max(1, e.uses)}"
            )
        return "\n".join(blocks)

    def apply_ops(self, ops: list[dict[str, Any]]) -> None:
        for op in ops or []:
            kind = str(op.get("operation", op.get("op", "KEEP"))).upper()
            if kind == "ADD":
                self.add(
                    profile=str(op.get("profile", op.get("query_type", "any"))),
                    insight=str(op.get("new_insight", op.get("insight", op.get("text", "")))),
                    rationale=str(op.get("rationale", "")),
                )
            elif kind == "MERGE":
                ids = list(op.get("target_entry_ids", op.get("ids", [])))
                merged = str(op.get("merged_insight", op.get("text", "")))
                profile = str(op.get("profile", "any"))
                if ids and merged:
                    self.merge(ids, merged, profile=profile, rationale=str(op.get("rationale", "")))
            elif kind == "PRUNE":
                ids = list(op.get("target_entry_ids", op.get("ids", [])))
                for eid in ids:
                    self.delete(eid)
                if not ids:
                    self.prune()
            elif kind == "KEEP":
                continue
        self.prune()

    # ---------- Paper Algorithm 3: ExperienceLibraryUpdate (deterministic dispatch) ----------

    def algorithm3_update(self, new_insights: list[dict[str, Any]],
                            similarity_threshold: float = 0.55,
                            conflict_threshold: float = 0.45) -> None:
        """Deterministic ADD/MERGE/PRUNE/KEEP per paper Algorithm 3.

        For each new insight z with profile c:
          matches ← Retrieve(E, c)
          if matches = ∅ → ADD((c, z, u=0))
          elif COMPLEMENTARY(z, matches) → MERGE(z, matches)
          elif CONFLICTS(z, matches) → low_u = FilterLowUtility(matches); PRUNE(low_u)
          else → KEEP

        Used as a non-LLM fallback / verification path to match the paper's deterministic alg.
        """
        for ins in new_insights or []:
            profile = str(ins.get("query_type", ins.get("profile", "any")))
            text = str(ins.get("insight", ins.get("text", "")))
            if not text:
                continue
            matches = [e for e in self.entries
                       if e.profile in (profile, "any") and _token_overlap(e.insight, text) >= conflict_threshold]
            if not matches:
                self.add(profile=profile, insight=text, rationale=str(ins.get("rationale", "")))
                continue
            # COMPLEMENTARY: same profile, moderate overlap (similar topic, compatible direction).
            complementary = [e for e in matches
                             if e.profile == profile and similarity_threshold >= _token_overlap(e.insight, text) > conflict_threshold]
            high_overlap = [e for e in matches if _token_overlap(e.insight, text) > similarity_threshold]
            if complementary:
                ids = [e.id for e in complementary]
                merged = self._naive_merge(text, complementary)
                self.merge(ids, merged, profile=profile, rationale="Algorithm 3 COMPLEMENTARY")
                continue
            if high_overlap:
                # Treat as conflict only if utility differs substantially.
                low_u = [e for e in high_overlap if e.utility_rate() < 0.5]
                if low_u and any(e.utility_rate() >= 0.5 for e in high_overlap):
                    for e in low_u:
                        self.delete(e.id)
                    continue
            # Otherwise KEEP.
        self.prune()

    @staticmethod
    def _naive_merge(new_text: str, matches: list[ExpEntry]) -> str:
        """Concatenate new + best existing insight, dedupe sentences."""
        parts = [new_text] + [e.insight for e in matches]
        seen: set[str] = set()
        out: list[str] = []
        for p in parts:
            key = re.sub(r"\s+", " ", p.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                out.append(p.strip())
        return " ".join(out)[:500]

    def prune(self) -> None:
        if self.max_entries <= 0 or len(self.entries) <= self.max_entries:
            return
        self.entries.sort(key=lambda e: (e.utility_rate(), e.utility, -e.uses), reverse=True)
        self.entries = self.entries[: self.max_entries]


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z]+", a.lower()))
    tb = set(re.findall(r"[a-z]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# ---------------- Question profiling ----------------

def profile_question(q: str, hint: str | None = None, qid: str | None = None) -> str:
    """Profile lookup. Priority: hint > GPT-annotation cache (paper §4) > rule heuristic.

    GPT-4o annotations are loaded from data/annotations/*.jsonl when produced by
    scripts/annotate_profiles.py. Falls back to keyword heuristic when missing.
    """
    if hint and hint in PROFILES:
        return hint
    if qid:
        annots = _load_annotations()
        if qid in annots:
            return annots[qid]
    s = q.lower()
    if any(w in s for w in ["why ", "cause of", "reason ", "lead to", "result in"]):
        return "causal"
    if any(w in s for w in [" or ", "which film was released first", "which one ", "smaller", "larger",
                              "older", "younger", "earlier", "later"]):
        return "comparison"
    if any(w in s for w in ["both ", "each ", "all of "]):
        return "intersection"
    if any(w in s for w in ["when ", "what year", "what date", "what month", "before ", "after "]):
        return "temporal"
    if any(w in s for w in ["plausible", "ambiguous", "interpret"]):
        return "ambiguous"
    return "bridge"

"""Dataset loaders + stratified subsampling for HERA training/evaluation."""
from __future__ import annotations

import glob
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QAExample:
    id: str
    question: str
    answer: str | list[str]
    source: str
    question_type: str = ""
    extra: dict[str, Any] | None = None


# ---------- Train loaders ----------

def load_musique_train(path: str = "/local/yzheng/pnair/baseline/ircot_repo/raw_data/musique/musique_ans_v1.0_train.jsonl",
                       limit: int | None = None) -> list[QAExample]:
    out: list[QAExample] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                ex = json.loads(line)
            except Exception:
                continue
            qid = ex.get("id", "")
            qtype = "comparison" if "compositional" in qid else (
                "bridge" if "hop" in qid else "")
            out.append(QAExample(
                id=str(qid),
                question=ex.get("question", ""),
                answer=ex.get("answer", ""),
                source="musique",
                question_type=qtype,
            ))
    return out


def load_2wiki_train(path: str = "/local/yzheng/pnair/baseline/ircot_repo/processed_data/2wikimultihopqa/train.jsonl",
                     limit: int | None = None) -> list[QAExample]:
    out: list[QAExample] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                ex = json.loads(line)
            except Exception:
                continue
            ans_obj = ex.get("answers_objects", [{}])[0] if ex.get("answers_objects") else {}
            spans = ans_obj.get("spans", []) if isinstance(ans_obj, dict) else []
            answer = spans[0] if spans else ""
            out.append(QAExample(
                id=str(ex.get("question_id", "")),
                question=ex.get("question_text", ""),
                answer=spans if len(spans) > 1 else answer,
                source="2wikimultihop",
                question_type=str(ex.get("type", "")),
            ))
    return out


def load_hotpotqa_train(path: str = "/local/yzheng/pnair/baseline/ircot_repo/processed_data/hotpotqa/train.jsonl",
                        limit: int | None = None) -> list[QAExample]:
    out: list[QAExample] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                ex = json.loads(line)
            except Exception:
                continue
            ans_obj = ex.get("answers_objects", [{}])[0] if ex.get("answers_objects") else {}
            spans = ans_obj.get("spans", []) if isinstance(ans_obj, dict) else []
            answer = spans[0] if spans else ""
            out.append(QAExample(
                id=str(ex.get("question_id", "")),
                question=ex.get("question_text", ""),
                answer=spans if len(spans) > 1 else answer,
                source="hotpotqa",
                question_type=str(ex.get("type", "")),
            ))
    return out


# ---------- Test loaders ----------

def load_test_json(path: str, source: str) -> list[QAExample]:
    raw = json.loads(Path(path).read_text())
    out: list[QAExample] = []
    if isinstance(raw, list):
        for ex in raw:
            ans = ex.get("answer", "")
            qtype = ex.get("question_type", "")
            out.append(QAExample(
                id=str(ex.get("id", "")),
                question=ex.get("question", ""),
                answer=ans,
                source=source,
                question_type=qtype,
            ))
    return out


def load_bamboogle_125(shard_dir: str | None = None) -> list[QAExample]:
    import os
    if shard_dir is None:
        env = os.getenv("AMAS_BAMBOOGLE_DIR")
        if env and os.path.isdir(env):
            shard_dir = env
        else:
            for cand in [
                "/projects/prjs1800/hera/data/bamboogle",
                "/local/yzheng/pnair/workspace/tmp/04-sage-autonomous/data/node408_shards/bamboogle",
            ]:
                if os.path.isdir(cand):
                    shard_dir = cand
                    break
    if not shard_dir:
        return []
    out: list[QAExample] = []
    for f in sorted(glob.glob(f"{shard_dir}/*.json")):
        data = json.loads(Path(f).read_text())
        for ex in data:
            out.append(QAExample(
                id=str(ex.get("id", "")),
                question=ex.get("question", ""),
                answer=ex.get("answer", ""),
                source="bamboogle",
                question_type="bridge",
            ))
    return out


# ---------- Stratified train sampling ----------

def stratified_train(per_dataset: int = 80, seed: int = 42,
                      datasets: tuple[str, ...] = ("musique", "2wikimultihop", "hotpotqa")
                      ) -> list[QAExample]:
    """Balanced across datasets, with diversity over question_type within each."""
    rng = random.Random(seed)
    pool: dict[str, list[QAExample]] = {}
    if "musique" in datasets:
        pool["musique"] = load_musique_train(limit=10000)
    if "2wikimultihop" in datasets:
        pool["2wikimultihop"] = load_2wiki_train(limit=10000)
    if "hotpotqa" in datasets:
        pool["hotpotqa"] = load_hotpotqa_train(limit=10000)

    selected: list[QAExample] = []
    for src, exs in pool.items():
        # Bucket by question_type
        buckets: dict[str, list[QAExample]] = defaultdict(list)
        for e in exs:
            buckets[e.question_type or "other"].append(e)
        types = list(buckets.keys())
        rng.shuffle(types)
        per_type = max(1, per_dataset // max(1, len(types)))
        chosen: list[QAExample] = []
        for t in types:
            rng.shuffle(buckets[t])
            chosen.extend(buckets[t][:per_type])
        rng.shuffle(chosen)
        selected.extend(chosen[:per_dataset])
    rng.shuffle(selected)
    return selected


def save_qa_jsonl(examples: list[QAExample], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for ex in examples:
            f.write(json.dumps({
                "id": ex.id, "question": ex.question, "answer": ex.answer,
                "source": ex.source, "question_type": ex.question_type
            }, ensure_ascii=False) + "\n")


def load_qa_jsonl(path: str | Path) -> list[QAExample]:
    out: list[QAExample] = []
    with open(path) as f:
        for line in f:
            try:
                ex = json.loads(line)
            except Exception:
                continue
            out.append(QAExample(
                id=str(ex.get("id", "")),
                question=ex.get("question", ""),
                answer=ex.get("answer", ""),
                source=ex.get("source", ""),
                question_type=ex.get("question_type", ""),
            ))
    return out

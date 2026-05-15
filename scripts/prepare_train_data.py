"""Sample ~150 training questions per dataset with ground truth, excluding test IDs.

Sources:
  - HotpotQA: data/multidataset/fresh_pool/hotpotqa.json (500, has question_type)
  - 2WikiMultiHop: data/multidataset/fresh_pool/2wikimultihop.json (500, has question_type)
  - MuSiQue: data/multidataset/fresh_pool/musique.json (500) +
             data/musique/musique_train_500.json (500, no question_type)

Test exclusion sets:
  - data/hotpotqa/questions_1000_seed42.json
  - data/2wikimultihop/questions_1000_seed42.json
  - data/musique/questions_1000_seedfull_combined.json

Output: data/{dataset}/train_150_seed42.json
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
N_PER_DATASET = 150
ROOT = Path(__file__).resolve().parent.parent

DATASET_CONFIGS = {
    "hotpotqa": {
        "pools": [
            ROOT / "data" / "multidataset" / "fresh_pool" / "hotpotqa.json",
        ],
        "test": ROOT / "data" / "hotpotqa" / "questions_1000_seed42.json",
    },
    "2wikimultihop": {
        "pools": [
            ROOT / "data" / "multidataset" / "fresh_pool" / "2wikimultihop.json",
        ],
        "test": ROOT / "data" / "2wikimultihop" / "questions_1000_seed42.json",
    },
    "musique": {
        "pools": [
            ROOT / "data" / "multidataset" / "fresh_pool" / "musique.json",
            ROOT / "data" / "musique" / "musique_train_500.json",
        ],
        "test": ROOT / "data" / "musique" / "questions_1000_seedfull_combined.json",
    },
}


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_test_ids(path: Path) -> set[str]:
    data = load_json(path)
    return {str(item["id"]) for item in data}


def infer_type(item: dict) -> str:
    """Extract question_type from item, falling back to hop-count heuristic for MuSiQue."""
    qt = item.get("question_type") or item.get("type") or ""
    if qt:
        return qt
    qid = str(item.get("id", ""))
    if "3hop" in qid:
        return "3hop"
    if "4hop" in qid:
        return "4hop"
    if "2hop" in qid:
        return "2hop"
    return "unknown"


def stratified_sample(
    pool: list[dict], n: int, rng: random.Random
) -> list[dict]:
    """Sample n items, stratified by type if type annotations exist."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for item in pool:
        by_type[infer_type(item)].append(item)

    # If only one type bucket or all "unknown", just random sample
    if len(by_type) <= 1:
        return rng.sample(pool, min(n, len(pool)))

    # Proportional stratified sampling
    sampled: list[dict] = []
    type_counts = {t: len(items) for t, items in by_type.items()}
    total = sum(type_counts.values())

    for qtype, items in sorted(by_type.items()):
        quota = max(1, round(n * len(items) / total))
        chosen = rng.sample(items, min(quota, len(items)))
        sampled.extend(chosen)

    # Adjust to exactly n
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)
    elif len(sampled) < n:
        remaining_ids = {id(s) for s in sampled}
        leftovers = [x for x in pool if id(x) not in remaining_ids]
        extra = rng.sample(leftovers, min(n - len(sampled), len(leftovers)))
        sampled.extend(extra)

    return sampled


def prepare_dataset(name: str, config: dict, rng: random.Random) -> None:
    log.info("Processing %s...", name)

    # Load test IDs to exclude
    test_ids = load_test_ids(config["test"])
    log.info("  Test exclusion set: %d IDs", len(test_ids))

    # Load and merge all pool files, dedup by ID
    seen_ids: set[str] = set()
    pool: list[dict] = []
    for pool_path in config["pools"]:
        if not pool_path.exists():
            log.warning("  Pool file not found: %s", pool_path)
            continue
        items = load_json(pool_path)
        for item in items:
            qid = str(item["id"])
            if qid in test_ids or qid in seen_ids:
                continue
            seen_ids.add(qid)
            pool.append(item)
    log.info("  Pool after exclusions: %d items", len(pool))

    if len(pool) < N_PER_DATASET:
        log.warning("  Pool smaller than target (%d < %d)", len(pool), N_PER_DATASET)

    sampled = stratified_sample(pool, N_PER_DATASET, rng)

    # Format output
    output = []
    for item in sampled:
        output.append({
            "id": str(item["id"]),
            "question": item["question"],
            "answer": item["answer"],
            "type": infer_type(item),
        })

    # Log type distribution
    from collections import Counter
    dist = Counter(o["type"] for o in output)
    log.info("  Sampled %d questions. Type distribution: %s", len(output), dict(dist))

    out_dir = ROOT / "data" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_150_seed42.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved to %s", out_path)


def main() -> None:
    rng = random.Random(SEED)
    for name, config in DATASET_CONFIGS.items():
        prepare_dataset(name, config, rng)
    log.info("Done.")


if __name__ == "__main__":
    main()

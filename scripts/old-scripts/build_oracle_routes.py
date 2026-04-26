"""Build a per-question oracle routing table from an S0 predictions file.

For each question, ``easy = (contain(S0_answer, gold) == 1.0)``. The output
JSONL is consumed by the sufficiency-controller oracle-route ablation
(`ablation.sufficiency_oracle_path`).

Usage:

    python3 scripts/build_oracle_routes.py \\
        --s0-predictions results/s0_matched/<dataset>/predictions.jsonl \\
        --questions data/<dataset>/questions_1000_seed42.json \\
        --output results/oracle/oracle_routes_<dataset>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_offline import contain  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-predictions", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as handle:
        gold = {str(q.get("id", "")).strip(): str(q.get("answer", "")) for q in json.load(handle)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    easy = 0
    with open(args.s0_predictions, "r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("id", "")).strip()
            if not qid or qid not in gold:
                continue
            is_easy = bool(contain(str(row.get("answer", "")), gold[qid]))
            dst.write(json.dumps({"id": qid, "easy": is_easy}, ensure_ascii=False) + "\n")
            n += 1
            easy += int(is_easy)

    print(json.dumps({"n": n, "easy": easy, "easy_rate": easy / n if n else 0.0}, indent=2))


if __name__ == "__main__":
    main()

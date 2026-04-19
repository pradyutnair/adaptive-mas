"""Setup script for musique-200q ablations.

- Slice first 200 questions from data/musique/questions_1000_seedfull_combined.json.
- Build oracle routing table from paper_results/latest/musique/s0_matched/predictions.jsonl
  for those 200 IDs.
- Materialise a copy of configs/m1_2.abl_oracle_route.yaml that points to the
  freshly-written oracle path.

Outputs:
    data/musique/questions_200_seedfull_first.json
    results/oracle/oracle_routes_musique_200.jsonl
    configs/_runtime/m1_2.abl_oracle_route_musique200.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_offline import contain  # noqa: E402

CANONICAL_QFILE = ROOT / "data/musique/questions_1000_seedfull_combined.json"
SUBSET_QFILE = ROOT / "data/musique/questions_200_seedfull_first.json"
S0_PRED = ROOT / "paper_results/latest/musique/s0_matched/predictions.jsonl"
ORACLE_PATH = ROOT / "results/oracle/oracle_routes_musique_200.jsonl"
ABL_ORACLE_SRC = ROOT / "configs/m1_2.abl_oracle_route.yaml"
ABL_ORACLE_DST = ROOT / "configs/_runtime/m1_2.abl_oracle_route_musique200.yaml"


def main() -> None:
    questions = json.loads(CANONICAL_QFILE.read_text())
    subset = questions[:200]
    SUBSET_QFILE.write_text(json.dumps(subset, ensure_ascii=False))
    print(f"wrote {SUBSET_QFILE} ({len(subset)} questions)")

    subset_ids = {str(q["id"]) for q in subset}
    gold_by_id = {str(q["id"]): str(q.get("answer", "")) for q in subset}

    ORACLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_easy = 0
    n_total = 0
    with open(S0_PRED, "r", encoding="utf-8") as src, open(
        ORACLE_PATH, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("id", ""))
            if qid not in subset_ids:
                continue
            gold = gold_by_id[qid]
            ans = str(row.get("answer", ""))
            is_easy = bool(contain(ans, gold))
            dst.write(json.dumps({"id": qid, "easy": is_easy}, ensure_ascii=False) + "\n")
            n_total += 1
            n_easy += int(is_easy)
    print(
        f"wrote {ORACLE_PATH} (n={n_total}, easy={n_easy}, easy_rate="
        f"{n_easy / n_total if n_total else 0:.3f})"
    )

    ABL_ORACLE_DST.parent.mkdir(parents=True, exist_ok=True)
    src_text = ABL_ORACLE_SRC.read_text()
    dst_text = src_text.replace(
        "results/oracle/oracle_routes_TBD.jsonl",
        str(ORACLE_PATH.relative_to(ROOT)),
    )
    ABL_ORACLE_DST.write_text(dst_text)
    print(f"wrote {ABL_ORACLE_DST}")


if __name__ == "__main__":
    main()

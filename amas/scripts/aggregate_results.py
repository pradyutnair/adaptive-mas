"""Aggregate per-dataset summary.json files into a unified Pareto table.

Usage:
  python scripts/aggregate_results.py --root results/run01 --out results/run01/aggregate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            sp = run_dir / "summary.json"
            if not sp.exists():
                continue
            d = json.loads(sp.read_text())
            d["dataset"] = ds_dir.name
            d["run"] = run_dir.name
            rows.append(d)
    rows.sort(key=lambda r: (r["dataset"], r["run"]))

    out = args.out or (root / "aggregate.json")
    Path(out).write_text(json.dumps(rows, indent=2))
    # Markdown view
    md = ["| dataset | run | n | em | f1 | acc | avg_tokens | avg_turns | sas_rate |",
          "|---------|-----|---|----|----|-----|-----------|----------|----------|"]
    for r in rows:
        md.append(
            f"| {r['dataset']} | {r['run']} | {r['n']} | {r['em']:.3f} | {r['f1']:.3f} | "
            f"{r['acc']:.3f} | {r['avg_tokens']:.0f} | {r['avg_turns']:.2f} | "
            f"{r.get('sas_rate', 0):.2f} |"
        )
    md_path = Path(out).with_suffix(".md")
    md_path.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

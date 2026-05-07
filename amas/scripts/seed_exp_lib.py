"""Stage 5: seed an empty AMAS-only experience library + base prompts.

Writes `exp_lib/<name>/library.json` (empty entries, max_entries=40) and
`exp_lib/<name>/prompts.json` (base SEED_PROMPTS — no warm-start operational
rules / behavioural principles from HERA).

Usage:
    python scripts/seed_exp_lib.py --name amas_v1
    python scripts/seed_exp_lib.py --name amas_v2  # for the GRPO output target
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from amas.agents import load_seed_prompts, save_prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="amas_v1",
                    help="Library directory under exp_lib/")
    ap.add_argument("--max-entries", type=int, default=40)
    ap.add_argument("--root", default=str(ROOT),
                    help="Repo root (defaults to the script's parent)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing exp_lib/<name>/")
    args = ap.parse_args()

    target = Path(args.root) / "exp_lib" / args.name
    if target.exists() and any(target.iterdir()) and not args.force:
        print(f"refuse to overwrite non-empty {target} (pass --force)", file=sys.stderr)
        sys.exit(2)
    target.mkdir(parents=True, exist_ok=True)

    lib_path = target / "library.json"
    lib_path.write_text(json.dumps(
        {"next_num": 1, "max_entries": int(args.max_entries), "entries": []},
        indent=2,
    ))

    prompts_path = target / "prompts.json"
    save_prompts(load_seed_prompts(), prompts_path)

    print(f"wrote {lib_path} ({lib_path.stat().st_size} B)")
    print(f"wrote {prompts_path} ({prompts_path.stat().st_size} B)")


if __name__ == "__main__":
    main()

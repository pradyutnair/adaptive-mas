#!/usr/bin/env python3
"""Validate that a completed stage has full coverage and no placeholder error rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_ids(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [str(row.get("id", "")) for row in json.load(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check stage health.")
    parser.add_argument("--questions", required=True, help="Combined questions JSON")
    parser.add_argument("--inputs", nargs="+", required=True, help="predictions.jsonl files")
    args = parser.parse_args()

    expected_ids = set(_load_ids(Path(args.questions)))
    seen_ids: set[str] = set()
    error_rows = 0
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            print(f"missing_file: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = str(row.get("id", ""))
                if qid:
                    seen_ids.add(qid)
                if (row.get("metadata", {}) or {}).get("error"):
                    error_rows += 1

    missing_ids = expected_ids - seen_ids
    if missing_ids or error_rows:
        print(f"missing_ids={len(missing_ids)} error_rows={error_rows}")
        sys.exit(1)

    print(f"ok count={len(seen_ids)} error_rows=0")


if __name__ == "__main__":
    main()

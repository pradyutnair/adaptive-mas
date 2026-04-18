#!/usr/bin/env python3
"""Convert 05-mas question JSON into Adaptive-RAG reader format."""

from __future__ import annotations

import argparse
import json


def answer_object(answer: str) -> dict:
    return {
        "number": "",
        "spans": [answer],
        "date": {"day": "", "month": "", "year": ""},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open(args.output, "w", encoding="utf-8") as out:
        for item in questions:
            row = {
                "question_id": item["id"],
                "question_text": item["question"],
                "answers_objects": [answer_object(item.get("answer", ""))],
                "contexts": [],
                "pinned_contexts": [],
            }
            out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

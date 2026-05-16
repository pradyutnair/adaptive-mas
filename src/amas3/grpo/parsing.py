"""Tolerant JSON object / array extraction for LM outputs."""
from __future__ import annotations

import json


def parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start:end + 1])
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return []


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}

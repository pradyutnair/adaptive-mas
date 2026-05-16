"""Answer-side scoring: EM / F1 / Contain and the composite task reward.

Kept independent of every other GRPO module so the reward and reflection
layers can share the same primitives without circular imports.
"""
from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = _ARTICLES.sub("", s)
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    s = " ".join(s.split())
    return s.strip()


def compute_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum(1 for t in pred_tokens if t in gold_tokens)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_contain(pred: str, gold: str) -> float:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return 0.0
    return 1.0 if gold_norm in pred_norm else 0.0


def compute_task_reward(em: float, f1: float, contain: float) -> float:
    """Task reward aligned to eval: contain is primary, F1/EM preserved."""
    return 0.45 * contain + 0.35 * f1 + 0.20 * em

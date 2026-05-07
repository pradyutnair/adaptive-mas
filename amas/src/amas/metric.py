"""EM / F1 / contain metrics. Standard SQuAD-style normalization."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str | list[str]) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def f1_score(pred: str, gold: str | list[str]) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    best = 0.0
    p_toks = normalize_answer(pred).split()
    if not p_toks:
        # If both empty: 1.0; else 0.0.
        for g in golds:
            if not normalize_answer(g).split():
                return 1.0
        return 0.0
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not g_toks:
            continue
        common = Counter(p_toks) & Counter(g_toks)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        prec = n_common / len(p_toks)
        rec = n_common / len(g_toks)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best:
            best = f1
    return best


def contain(pred: str, gold: str | list[str]) -> float:
    golds = gold if isinstance(gold, list) else [gold]
    p = normalize_answer(pred)
    return float(any(normalize_answer(g) and normalize_answer(g) in p for g in golds))


def accuracy(pred: str, gold: str | list[str]) -> float:
    """Acc = max(EM, contain) per HERA paper convention for QA."""
    return max(exact_match(pred, gold), contain(pred, gold))

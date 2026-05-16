"""Orchestrator and topology optimization via training-free GRPO analysis.

Learns query-type-specific routing, retrieval budget, and topology selection
rules from training results. Encodes them as ExperienceEntry objects.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import dspy

from .experience_library import ExperienceEntry, ExperienceLibrary
from .gepa import normalize_answer, compute_f1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_query_type(question: str) -> str:
    """Simple heuristic query type classification."""
    q = question.lower().strip()
    if q.startswith("how many") or q.startswith("how much"):
        return "count"
    if q.startswith("when") or "what year" in q or "what date" in q:
        return "temporal"
    if q.startswith("where") or "which city" in q or "which country" in q:
        return "location"
    if q.startswith("who"):
        return "person"
    if " or " in q and ("which" in q or "what" in q):
        return "comparison"
    if "both" in q or "and" in q and q.startswith("what"):
        return "intersection"
    return "bridge"


def contain(prediction: str, gold: str) -> float:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return 0.0
    return 1.0 if gold_norm in pred_norm else 0.0


def task_score(metrics: dict) -> float:
    return 0.45 * float(metrics.get("contain", 0.0)) + 0.35 * float(metrics.get("f1", 0.0)) + 0.20 * float(metrics.get("em", 0.0))


def semantic_topology_label(row: dict) -> str:
    """Compact label for the sampled orchestrator topology, not internal pipeline shape."""
    topo = row.get("sampled_topology") or {}
    if isinstance(topo, dict) and topo:
        sig = topo.get("_topology_signature")
        if sig:
            return "|".join(str(x) for x in sig)
        return (
            f"{topo.get('routing_strategy', 'strategy_unknown')}|"
            f"b{topo.get('retrieval_budget', '?')}|"
            f"repair{int(bool(topo.get('repair', False)))}"
        )
    return str(row.get("topology", "linear"))


def result_metrics(result_dict: dict, gold_answer: str) -> dict:
    """Compute eval-aligned EM/F1/Contain/tokens from a result dict."""
    pred = result_dict.get("answer", "")
    em = int(normalize_answer(pred) == normalize_answer(gold_answer))
    f1 = compute_f1(pred, gold_answer)
    cont = contain(pred, gold_answer)
    tokens = result_dict.get("total_tokens", 0)
    return {"em": em, "f1": f1, "contain": cont, "tokens": tokens}


# ---------------------------------------------------------------------------
# Routing analysis
# ---------------------------------------------------------------------------

def analyze_routing_decisions(results: list[dict]) -> dict:
    """From training results, compute per-query-type SAS vs MAS effectiveness.

    Each result dict should have: question, gold_answer, answer, total_tokens,
    sas_collapse (bool), topology, sas_attempt_confidence, sas_escalated.
    """
    by_type: dict[str, dict] = defaultdict(lambda: {
        "sas_correct": 0, "sas_total": 0, "sas_tokens": [],
        "mas_correct": 0, "mas_total": 0, "mas_tokens": [],
        "sas_confidences": [],
    })

    for r in results:
        question = r.get("question", "")
        gold = r.get("gold_answer", "")
        qtype = classify_query_type(question)
        metrics = result_metrics(r, gold)
        bucket = by_type[qtype]

        if r.get("sas_collapse", False):
            bucket["sas_total"] += 1
            bucket["sas_correct"] += 1 if (metrics["em"] > 0 or metrics.get("contain", 0.0) > 0) else 0
            bucket["sas_tokens"].append(metrics["tokens"])
            bucket["sas_confidences"].append(r.get("sas_attempt_confidence", 0.0))
        else:
            bucket["mas_total"] += 1
            bucket["mas_correct"] += 1 if (metrics["em"] > 0 or metrics.get("contain", 0.0) > 0) else 0
            bucket["mas_tokens"].append(metrics["tokens"])

    analysis = {}
    for qtype, bucket in by_type.items():
        sas_acc = bucket["sas_correct"] / max(bucket["sas_total"], 1)
        mas_acc = bucket["mas_correct"] / max(bucket["mas_total"], 1)
        sas_avg_tok = sum(bucket["sas_tokens"]) / max(len(bucket["sas_tokens"]), 1)
        mas_avg_tok = sum(bucket["mas_tokens"]) / max(len(bucket["mas_tokens"]), 1)
        sas_avg_conf = sum(bucket["sas_confidences"]) / max(len(bucket["sas_confidences"]), 1)
        analysis[qtype] = {
            "sas_accuracy": round(sas_acc, 4),
            "sas_count": bucket["sas_total"],
            "sas_avg_tokens": round(sas_avg_tok, 1),
            "sas_avg_confidence": round(sas_avg_conf, 3),
            "mas_accuracy": round(mas_acc, 4),
            "mas_count": bucket["mas_total"],
            "mas_avg_tokens": round(mas_avg_tok, 1),
            "prefer_sas": sas_acc >= mas_acc and sas_avg_tok < mas_avg_tok * 0.6,
        }

    return analysis


def derive_routing_rules(analysis: dict) -> list[dict]:
    """Generate experience entries encoding optimal routing per query type."""
    rules = []
    for qtype, stats in analysis.items():
        if stats["sas_count"] < 3 and stats["mas_count"] < 3:
            continue

        if stats["prefer_sas"]:
            rules.append({
                "profile": f"routing_{qtype}",
                "insight": (
                    f"For {qtype} queries, prefer sas_first. "
                    f"SAS-first accuracy: {stats['sas_accuracy']:.1%}, direct-MAS accuracy: {stats['mas_accuracy']:.1%}, "
                    f"SAS saves {stats['mas_avg_tokens'] - stats['sas_avg_tokens']:.0f} tokens on average."
                ),
                "target_roles": ("orchestrator",),
                "applies_when": f"Query type is {qtype} and probe groundedness is high",
                "avoid_when": f"Query type is {qtype} but evidence is sparse",
            })
        elif stats["mas_accuracy"] > stats["sas_accuracy"] + 0.1:
            rules.append({
                "profile": f"routing_{qtype}",
                "insight": (
                    f"For {qtype} queries, prefer direct_mas decomposition. "
                    f"Direct-MAS accuracy: {stats['mas_accuracy']:.1%} vs SAS-first: {stats['sas_accuracy']:.1%}."
                ),
                "target_roles": ("orchestrator",),
                "applies_when": f"Query type is {qtype}",
                "avoid_when": "",
            })

    return rules


# ---------------------------------------------------------------------------
# Budget analysis
# ---------------------------------------------------------------------------

def analyze_retrieval_budgets(results: list[dict]) -> dict:
    """Per query-type, compute which retrieval budget yields best accuracy/token tradeoff.

    Each result dict should have: question, gold_answer, answer, total_tokens,
    n_retrieval_calls, plan_subgoals.
    """
    by_type: dict[str, list] = defaultdict(list)

    for r in results:
        question = r.get("question", "")
        gold = r.get("gold_answer", "")
        qtype = classify_query_type(question)
        metrics = result_metrics(r, gold)
        retrievals = r.get("n_retrieval_calls", 0)
        by_type[qtype].append({
            "em": metrics["em"],
            "f1": metrics["f1"],
            "contain": metrics.get("contain", 0.0),
            "tokens": metrics["tokens"],
            "retrievals": retrievals,
        })

    analysis = {}
    for qtype, entries in by_type.items():
        # Bucket by retrieval count: 1, 2, 3+
        buckets: dict[int, list] = {1: [], 2: [], 3: []}
        for e in entries:
            r = min(e["retrievals"], 3)
            r = max(r, 1)
            buckets[r].append(e)

        budget_stats = {}
        for budget, items in buckets.items():
            if not items:
                continue
            avg_em = sum(x["em"] for x in items) / len(items)
            avg_f1 = sum(x["f1"] for x in items) / len(items)
            avg_contain = sum(x.get("contain", 0.0) for x in items) / len(items)
            avg_tok = sum(x["tokens"] for x in items) / len(items)
            budget_stats[budget] = {
                "accuracy": round(avg_em, 4),
                "f1": round(avg_f1, 4),
                "contain": round(avg_contain, 4),
                "avg_tokens": round(avg_tok, 1),
                "count": len(items),
            }

        # Find optimal budget: best accuracy, then fewest tokens on tie
        best_budget = 2  # default
        best_score = -1.0
        for budget, stats in budget_stats.items():
            score = stats.get("contain", stats["accuracy"]) + stats["f1"] * 0.35 + stats["accuracy"] * 0.2 - (stats["avg_tokens"] / 20000)
            if score > best_score:
                best_score = score
                best_budget = budget

        analysis[qtype] = {
            "budget_stats": budget_stats,
            "optimal_budget": best_budget,
        }

    return analysis


def derive_budget_rules(analysis: dict) -> list[dict]:
    """Generate experience entries for adaptive budget allocation."""
    rules = []
    for qtype, stats in analysis.items():
        budget = stats["optimal_budget"]
        budget_info = stats["budget_stats"].get(budget, {})
        if not budget_info or budget_info.get("count", 0) < 3:
            continue

        rules.append({
            "profile": f"budget_{qtype}",
            "insight": (
                f"For {qtype} queries, use retrieval budget={budget}. "
                f"Contain: {budget_info.get('contain', 0):.1%}, "
                f"Accuracy: {budget_info.get('accuracy', 0):.1%}, "
                f"avg tokens: {budget_info.get('avg_tokens', 0):.0f}."
            ),
            "target_roles": ("planner", "solver"),
            "applies_when": f"Query type is {qtype}",
            "avoid_when": "Evidence is very sparse (groundedness < 0.3)",
        })

    return rules


# ---------------------------------------------------------------------------
# Topology analysis
# ---------------------------------------------------------------------------

def analyze_topology_effectiveness(results: list[dict]) -> dict:
    """Which topologies work best for which query types.

    Each result dict should have: question, gold_answer, answer, total_tokens, topology.
    """
    by_type_topo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        question = r.get("question", "")
        gold = r.get("gold_answer", "")
        qtype = classify_query_type(question)
        topo = semantic_topology_label(r)
        metrics = result_metrics(r, gold)
        by_type_topo[qtype][topo].append(metrics)

    analysis = {}
    for qtype, topo_entries in by_type_topo.items():
        topo_stats = {}
        for topo, entries in topo_entries.items():
            if not entries:
                continue
            avg_em = sum(e["em"] for e in entries) / len(entries)
            avg_f1 = sum(e["f1"] for e in entries) / len(entries)
            avg_contain = sum(e.get("contain", 0.0) for e in entries) / len(entries)
            avg_tok = sum(e["tokens"] for e in entries) / len(entries)
            topo_stats[topo] = {
                "accuracy": round(avg_em, 4),
                "f1": round(avg_f1, 4),
                "contain": round(avg_contain, 4),
                "avg_tokens": round(avg_tok, 1),
                "count": len(entries),
            }

        # Find best topology
        best_topo = "linear"
        best_score = -1.0
        for topo, stats in topo_stats.items():
            score = stats.get("contain", stats["accuracy"]) + stats["f1"] * 0.35 + stats["accuracy"] * 0.2 - (stats["avg_tokens"] / 20000)
            if score > best_score:
                best_score = score
                best_topo = topo

        analysis[qtype] = {
            "topology_stats": topo_stats,
            "best_topology": best_topo,
        }

    return analysis


def derive_topology_rules(analysis: dict) -> list[dict]:
    """Experience entries for topology selection."""
    rules = []
    for qtype, stats in analysis.items():
        best = stats["best_topology"]
        topo_info = stats["topology_stats"].get(best, {})
        if not topo_info or topo_info.get("count", 0) < 3:
            continue

        rules.append({
            "profile": f"topology_{qtype}",
            "insight": (
                f"For {qtype} queries, prefer topology='{best}'. "
                f"Accuracy: {topo_info.get('accuracy', 0):.1%}, "
                f"Contain: {topo_info.get('contain', 0):.1%}, "
                f"F1: {topo_info.get('f1', 0):.1%}, "
                f"avg tokens: {topo_info.get('avg_tokens', 0):.0f}."
            ),
            "target_roles": ("planner", "orchestrator"),
            "applies_when": f"Query type is {qtype}",
            "avoid_when": "",
        })

    return rules


# ---------------------------------------------------------------------------
# Integration: orchestrator analysis -> experience-library insights
#
# HERA's orchestrator policy pi_O is updated ONLY via the experience library E.
# This module derives natural-language insights from training trajectories and
# adds them as ExperienceEntry rows. It does NOT emit flat threshold tables or
# per-query-type config_overrides: those would turn eval into a grid search
# instead of adaptive topology sampling pi_O(Gamma | q, E, N).
# ---------------------------------------------------------------------------


def optimize_orchestration(
    reflection_lm: dspy.LM,
    training_results: list[dict],
    experience_library: ExperienceLibrary,
) -> dict:
    """Distill training trajectories into experience-library insights.

    Pipeline (HERA Algorithm 3 spirit, applied at the orchestrator level):
      1. Analyze routing / retrieval-budget / topology effectiveness.
      2. Convert the analyses into ExperienceEntry insights.
      3. Add them to the library so the orchestrator LM can condition on them.

    Returns diagnostics only (no config_overrides). `reflection_lm` is kept in
    the signature for backwards compatibility but is no longer used to emit a
    flat config dict.
    """
    del reflection_lm  # No longer used; analyses become library entries.

    routing = analyze_routing_decisions(training_results)
    budgets = analyze_retrieval_budgets(training_results)
    topology = analyze_topology_effectiveness(training_results)

    all_rules = (
        derive_routing_rules(routing)
        + derive_budget_rules(budgets)
        + derive_topology_rules(topology)
    )

    new_entries: list[ExperienceEntry] = []
    for rule in all_rules:
        entry = ExperienceEntry(
            id=f"orch_{rule['profile']}",
            profile=rule["profile"],
            insight=rule["insight"],
            utility=0.6,
            target_roles=tuple(rule.get("target_roles", ())),
            applies_when=rule.get("applies_when", ""),
            avoid_when=rule.get("avoid_when", ""),
        )
        experience_library.add(entry)
        new_entries.append(entry)

    logger.info(
        "Orchestrator analysis -> library: %d routing, %d budget, %d topology insights",
        len(derive_routing_rules(routing)),
        len(derive_budget_rules(budgets)),
        len(derive_topology_rules(topology)),
    )

    return {
        "new_entries": new_entries,
        "routing_analysis": routing,
        "budget_analysis": budgets,
        "topology_analysis": topology,
    }

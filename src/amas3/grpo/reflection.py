"""Semantic-advantage extraction (TF-GRPO Algorithm 2 reflection step).

The reflection step turns a mixed-outcome (or efficiency-contrasting) group
into <=3 short natural-language insights that the experience-library update
step can consume.
"""
from __future__ import annotations

import logging

import dspy

from .experience_library import ExperienceLibrary
from .metrics import compute_task_reward
from .parsing import parse_json_array, parse_json_object
from .profiles import characterize_query_profile
from .prompts import SEMANTIC_ADVANTAGE_PROMPT, TRAJECTORY_SUMMARY_PROMPT
from .rewards import TOKEN_BUDGET_BASELINES
from .rollout import GroupResult, Rollout
from .topology import format_rollout_for_reflection

logger = logging.getLogger(__name__)


def summarize_rollout(r: Rollout, reflection_lm: dspy.LM) -> str:
    """TF-GRPO per-rollout summarization before group advantage extraction."""
    prompt = TRAJECTORY_SUMMARY_PROMPT.format(
        question=r.question,
        gold_answer=r.gold_answer,
        em=r.em,
        f1=r.f1,
        contain=r.contain,
        tokens=r.total_tokens,
        trajectory=format_rollout_for_reflection(r),
    )
    with dspy.context(lm=reflection_lm):
        response = reflection_lm(prompt)
    text = response[0] if isinstance(response, list) else str(response)
    return text.strip()


def format_library_for_reflection(
    library: ExperienceLibrary | None, question: str = "", limit: int = 12,
) -> str:
    if library is None or not library.entries:
        return "(empty)"
    entries = library.retrieve(question, limit=limit) if question else list(library.entries.values())[:limit]
    return "\n".join(
        f"[{entry.id}] profile={entry.profile}, utility={entry.utility:.2f}, "
        f"roles={list(entry.target_roles)}: {entry.insight}"
        for entry in entries
    ) or "(empty)"


def _has_efficiency_contrast(group: GroupResult) -> bool:
    """Admit reflection on tied-quality groups with meaningful token spread.

    HERA reflects on ranked group trajectories. In practice the local Qwen
    backbone often gives identical correctness across samples; dropping those
    groups starves the experience library. This relaxation keeps the
    group-relative update intact but admits:
      - tied-quality groups with material token spread,
      - high-cost plateaus (all correct but wasteful),
      - all-failure groups (need negative signal to break topology collapse).
    """
    if len(group.rollouts) < 2:
        return False
    scores = [compute_task_reward(float(r.em), float(r.f1), float(r.contain)) for r in group.rollouts]
    tokens = [max(1, int(r.total_tokens)) for r in group.rollouts]
    baseline = TOKEN_BUDGET_BASELINES.get(group.rollouts[0].dataset, TOKEN_BUDGET_BASELINES["default"])
    avg_tokens = sum(tokens) / len(tokens)
    if max(scores) < 0.05:
        return True
    tied_quality = max(scores) - min(scores) <= 0.12
    token_spread = max(tokens) >= 1.10 * min(tokens)
    high_cost_plateau = avg_tokens >= 0.90 * baseline
    all_correct = min(scores) >= 0.40
    return tied_quality and (token_spread or high_cost_plateau or all_correct)


def extract_semantic_advantages(
    group: GroupResult,
    reflection_lm: dspy.LM,
    query_type: str | None = None,
    library: ExperienceLibrary | None = None,
) -> list[dict]:
    """Extract <=3 insights from a mixed-outcome or efficiency-contrasting group."""
    if not group.has_mixed_outcomes and not _has_efficiency_contrast(group):
        return []

    ranked = sorted(
        group.rollouts,
        key=lambda r: (-compute_task_reward(float(r.em), float(r.f1), float(r.contain)), int(r.total_tokens)),
    )
    summary_blocks = []
    for idx, rollout in enumerate(ranked, start=1):
        summary = summarize_rollout(rollout, reflection_lm)
        summary_blocks.append(
            f"Rollout {idx}: EM={rollout.em:.1f}, F1={rollout.f1:.2f}, "
            f"Contain={rollout.contain:.1f}, tokens={rollout.total_tokens}\n{summary}"
        )
    trajectory_summaries = "\n---\n".join(summary_blocks)

    prompt = SEMANTIC_ADVANTAGE_PROMPT.format(
        question=group.question,
        query_type=query_type or characterize_query_profile(
            group.question,
            group.rollouts[0].dataset if group.rollouts else "default",
        ),
        library_text=format_library_for_reflection(library, group.question),
        trajectory_summaries=trajectory_summaries,
    )

    with dspy.context(lm=reflection_lm):
        response = reflection_lm(prompt)

    raw_text = response[0] if isinstance(response, list) else str(response)
    obj = parse_json_object(raw_text)
    if obj and isinstance(obj.get("insights"), list):
        insights = obj.get("insights", [])
        for ins in insights:
            if isinstance(ins, dict) and "insight" in ins:
                ins["insight"] = " ".join(str(ins["insight"]).split()[:32])
        return insights
    return parse_json_array(raw_text)

"""Experience-library credit assignment and consolidation.

Two operations:
  - ``update_experience_credit_from_group``: rewards entries that actually
    conditioned pi_O (HERA utility update), with a cold-start fallback for
    early epochs when no entry was injected.
  - ``apply_experience_updates``: runs the LM-driven 5-operation menu
    (ADD / MERGE / DELETE / MODIFY / KEEP) plus pre/post pruning to keep the
    library bounded.
"""
from __future__ import annotations

import json
import logging

import dspy

from .experience_library import ExperienceEntry, ExperienceLibrary
from .metrics import compute_task_reward
from .parsing import parse_json_array
from .profiles import characterize_query_profile
from .prompts import EXPERIENCE_UPDATE_PROMPT
from .rollout import GroupResult

logger = logging.getLogger(__name__)


def update_experience_credit_from_group(
    library: ExperienceLibrary,
    group: GroupResult,
) -> None:
    """Credit only experience entries that were actually injected into pi_O."""
    if not library.entries:
        return
    best_task = max(
        (compute_task_reward(float(r.em), float(r.f1), float(r.contain)) for r in group.rollouts),
        default=0.0,
    )
    touched: set[str] = set()
    for rollout in group.rollouts:
        ids = (rollout.sampled_topology or {}).get("_experience_entry_ids") or []
        if not ids:
            continue
        score = compute_task_reward(float(rollout.em), float(rollout.f1), float(rollout.contain))
        success = (
            score >= max(0.45, best_task - 0.05)
            and bool((rollout.predicted_answer or "").strip())
            and rollout.token_efficiency >= 0.25
        )
        for eid in ids:
            if eid in library.entries:
                library.update_utility(eid, success=success)
                touched.add(eid)

    if touched:
        return

    # Cold-start fallback: nothing was injected, credit only the most
    # semantically similar entries, not the whole retrieved set.
    dataset = group.rollouts[0].dataset if group.rollouts else "default"
    entries = library.retrieve(
        characterize_query_profile(group.question, dataset) + " " + group.question,
        limit=3,
    )
    any_success = any(
        compute_task_reward(float(r.em), float(r.f1), float(r.contain)) >= 0.45
        and bool((r.predicted_answer or "").strip())
        for r in group.rollouts
    )
    for entry in entries:
        library.update_utility(entry.id, success=any_success)


def apply_experience_updates(
    library: ExperienceLibrary,
    new_insights: list[dict],
    reflection_lm: dspy.LM,
    max_library_size: int = 40,
) -> ExperienceLibrary:
    """Run the TF-GRPO 5-operation menu over the current batch of insights."""
    if not new_insights:
        return library

    # Pre-prune: remove obviously stale entries before the LM sees the library.
    stale_ids = [
        eid for eid, entry in library.entries.items()
        if entry.utility < 0.2 and entry.usage_count > 3
    ]
    for eid in stale_ids:
        library.prune(eid)
        logger.info(
            "Pre-pruned stale entry %s (utility=%.2f)", eid,
            library.entries.get(eid, ExperienceEntry(id="", profile="", insight="")).utility,
        )

    lib_lines = []
    for entry in library.entries.values():
        lib_lines.append(
            f"  [{entry.id}] profile={entry.profile}, utility={entry.utility:.2f}, "
            f"usage={entry.usage_count}, roles={list(entry.target_roles)}: {entry.insight[:100]}"
        )
    library_text = "\n".join(lib_lines) if lib_lines else "(empty library)"

    truncated_insights = []
    for ins in new_insights:
        if isinstance(ins, dict):
            ins_copy = dict(ins)
            if "insight" in ins_copy:
                ins_copy["insight"] = " ".join(str(ins_copy["insight"]).split()[:32])
            truncated_insights.append(ins_copy)
    insights_text = json.dumps(truncated_insights, indent=2)

    prompt = EXPERIENCE_UPDATE_PROMPT.format(
        max_entries=max_library_size,
        n_entries=library.size(),
        library_text=library_text,
        new_insights_text=insights_text,
    )

    with dspy.context(lm=reflection_lm):
        response = reflection_lm(prompt)

    raw_text = response[0] if isinstance(response, list) else str(response)
    updates = parse_json_array(raw_text)
    if not updates:
        # Heuristic fallback: ADD remaining insights if the LM produced no
        # parseable plan. Bounded by the remaining capacity.
        updates = []
        for ins in truncated_insights[: max(0, max_library_size - library.size())]:
            if not isinstance(ins, dict) or not ins.get("insight"):
                continue
            updates.append({
                "action": "ADD",
                "target_id": "",
                "insight": {
                    "profile": ins.get("query_type") or ins.get("profile") or "train_group",
                    "insight": " ".join(str(ins.get("insight", "")).split()[:32]),
                    "target_roles": ins.get("target_roles", ["orchestrator", "planner", "solver"]),
                    "applies_when": ins.get("applies_when", "similar query profile"),
                    "avoid_when": ins.get("avoid_when", ""),
                },
            })

    for update in updates:
        action = str(update.get("operation", update.get("action", update.get("option", "KEEP")))).upper()
        target_ids = update.get("target_entry_ids") or []
        if isinstance(target_ids, str):
            target_ids = [target_ids]
        target_id = str(
            update.get("target_id", "")
            or update.get("modified_from", "")
            or (target_ids[0] if target_ids else "")
        )
        insight_data = update.get("insight", {}) or {}
        if not insight_data and (update.get("merged_insight") or update.get("new_insight")):
            insight_data = {
                "profile": "train_group",
                "insight": update.get("merged_insight") or update.get("new_insight"),
                "target_roles": ["orchestrator", "planner", "solver"],
                "applies_when": "similar query profile",
                "avoid_when": "",
            }

        # TF-GRPO five-operation menu: ADD / MERGE / DELETE / MODIFY / KEEP.
        # PRUNE accepted as a legacy alias for DELETE.
        if action == "KEEP":
            continue
        if action in ("DELETE", "PRUNE") and target_id:
            library.prune(target_id)
            continue

        if not insight_data or not insight_data.get("insight"):
            continue

        insight_text = " ".join(str(insight_data["insight"]).split()[:32])
        roles = insight_data.get("target_roles", ["planner", "solver", "synthesizer"])
        if isinstance(roles, str):
            roles = [roles]

        new_entry = ExperienceEntry(
            id="",
            profile=insight_data.get("profile", "general"),
            insight=insight_text,
            utility=0.5,
            target_roles=tuple(roles),
            applies_when=insight_data.get("applies_when", ""),
            avoid_when=insight_data.get("avoid_when", ""),
        )

        if action == "ADD" and library.size() < max_library_size:
            library.add(new_entry)
        elif action == "MODIFY" and target_id:
            library.modify(target_id, new_entry)
        elif action == "MERGE" and target_id:
            library.merge(target_id, new_entry)

    # Post-prune if the library is still over the cap.
    while library.size() > max_library_size:
        lowest = min(library.entries.values(), key=lambda e: (e.utility, -e.usage_count))
        library.prune(lowest.id)
        logger.info("Post-pruned entry %s to stay under %d", lowest.id, max_library_size)

    return library

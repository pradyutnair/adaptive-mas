"""TF-GRPO learning subsystem for the AMAS multi-agent RAG pipeline.

This subpackage groups every module that participates in the training-free
group-relative policy optimization loop. Public API is re-exported below so
callers can ``from amas3.grpo import X`` without caring which file owns ``X``.

Module map:

  experience_library  Library + entry dataclasses and retrieval primitives.
  gepa                Role-aware Prompt Evolution (RoPE) for agent prompts.
  orch_grpo           Training-trajectory analyses producing orchestrator
                      insights (no flat threshold tables emitted).
  metrics             EM / F1 / Contain and the composite task reward.
  rewards             Token-efficiency reward, over-budget penalty,
                      correctness-gated dual reward.
  profiles            Canonical 5-class query profiler with GPT-4o cache.
  prompts             Every prompt template used by GRPO.
  parsing             Tolerant JSON object / array extraction.
  rollout             Rollout / GroupResult dataclasses.
  topology            Topology sampling (pi_O), routing normalization, mutations,
                      config translation.
  rollouts            Single + group rollout drivers.
  reflection          Semantic advantage extraction (Algorithm 2 reflection).
  library_update      Utility credit + ADD/MERGE/DELETE/MODIFY/KEEP application.
"""
from .experience_library import (
    ExperienceEntry,
    ExperienceLibrary,
    format_for_orchestrator,
    format_for_prompt,
)
from .gepa import (
    FailureBuffer,
    MAX_PROMPT_CHARS,
    run_gepa_epoch,
    save_evolved_prompts,
)
from .library_update import (
    apply_experience_updates,
    update_experience_credit_from_group,
)
from .metrics import (
    compute_contain,
    compute_em,
    compute_f1,
    compute_task_reward,
    normalize_answer,
)
from .orch_grpo import optimize_orchestration
from .parsing import parse_json_array, parse_json_object
from .profiles import PROFILE_CLASSES, characterize_query_profile
from .reflection import extract_semantic_advantages
from .rewards import (
    TOKEN_BUDGET_BASELINES,
    compute_dual_reward,
    compute_over_budget_penalty,
    compute_token_efficiency_reward,
)
from .rollout import GroupResult, Rollout
from .rollouts import run_group_rollouts, run_single_rollout
from .topology import (
    config_from_topology,
    normalize_strategy,
    sample_topology,
    topology_mutations,
    topology_signature,
)

__all__ = [
    "ExperienceEntry",
    "ExperienceLibrary",
    "FailureBuffer",
    "GroupResult",
    "MAX_PROMPT_CHARS",
    "PROFILE_CLASSES",
    "Rollout",
    "TOKEN_BUDGET_BASELINES",
    "apply_experience_updates",
    "characterize_query_profile",
    "compute_contain",
    "compute_dual_reward",
    "compute_em",
    "compute_f1",
    "compute_over_budget_penalty",
    "compute_task_reward",
    "compute_token_efficiency_reward",
    "config_from_topology",
    "extract_semantic_advantages",
    "format_for_orchestrator",
    "format_for_prompt",
    "normalize_answer",
    "normalize_strategy",
    "optimize_orchestration",
    "parse_json_array",
    "parse_json_object",
    "run_gepa_epoch",
    "run_group_rollouts",
    "run_single_rollout",
    "sample_topology",
    "save_evolved_prompts",
    "topology_mutations",
    "topology_signature",
    "update_experience_credit_from_group",
]

#!/usr/bin/env bash

# Draft only. Do not run without explicit approval.
#
# Purpose:
# - budget-capped A1 near the S0 token band
# - MuSiQue Pareto sweeps for S0 / frozen M1.1 / A1
#
# Assumptions:
# - MuSiQue 1000q split files already exist or will be created as:
#   data/musique/questions_1000_shard0.json
#   data/musique/questions_1000_shard1.json
#   data/musique/questions_1000_shard2.json
# - Ports map one-to-one to GPUs:
#   8001 -> shard0
#   8002 -> shard1
#   8003 -> shard2
#
# Frozen reference config:
# - frozen/iter16_best/config/m1_1.iter16.yaml


# ---------------------------------------------------------------------------
# 1) Budget-capped A1 draft
# Target: approximate S0 token band by shrinking A1 depth + per-step evidence.
# This is a draft only; expected budget must be measured after a pilot.
# ---------------------------------------------------------------------------

# cp configs/a1.yaml configs/a1_budget_s0draft.yaml
# python3 - <<'PY'
# import yaml
# path = "configs/a1_budget_s0draft.yaml"
# with open(path) as f:
#     cfg = yaml.safe_load(f)
# cfg["variant"] = "a1_budget_s0draft"
# cfg["orchestrator"]["max_steps"] = 1
# cfg.setdefault("investigator", {})
# cfg["investigator"]["search_top_k"] = 2
# cfg["investigator"]["evidence_capsule_limit"] = 1
# cfg["runner"]["checkpoint"] = True
# with open(path, "w") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False)
# PY

# python3 scripts/runner.py \
#   --config configs/a1_budget_s0draft.yaml \
#   --questions data/musique/questions_1000_shard0.json \
#   --output-dir results/A1_budget_s0draft_shard0 \
#   --server-url http://localhost:8001/v1 \
#   --concurrency 24

# python3 scripts/runner.py \
#   --config configs/a1_budget_s0draft.yaml \
#   --questions data/musique/questions_1000_shard1.json \
#   --output-dir results/A1_budget_s0draft_shard1 \
#   --server-url http://localhost:8002/v1 \
#   --concurrency 24

# python3 scripts/runner.py \
#   --config configs/a1_budget_s0draft.yaml \
#   --questions data/musique/questions_1000_shard2.json \
#   --output-dir results/A1_budget_s0draft_shard2 \
#   --server-url http://localhost:8003/v1 \
#   --concurrency 24


# ---------------------------------------------------------------------------
# 2) Pareto sweep draft
# Budget axes:
# - S0: retrieval width / capsule width
# - M1.1: recursive depth + retrieval width / capsule width
# - A1: recursive depth + retrieval width / capsule width
# ---------------------------------------------------------------------------

# S0 low budget
# cp configs/s0.yaml configs/s0_pareto_lowdraft.yaml
# python3 - <<'PY'
# import yaml
# path = "configs/s0_pareto_lowdraft.yaml"
# with open(path) as f:
#     cfg = yaml.safe_load(f)
# cfg["variant"] = "s0_pareto_lowdraft"
# cfg.setdefault("investigator", {})
# cfg["investigator"]["search_top_k"] = 2
# cfg["investigator"]["evidence_capsule_limit"] = 1
# with open(path, "w") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False)
# PY

# S0 high budget
# cp configs/s0.yaml configs/s0_pareto_highdraft.yaml
# python3 - <<'PY'
# import yaml
# path = "configs/s0_pareto_highdraft.yaml"
# with open(path) as f:
#     cfg = yaml.safe_load(f)
# cfg["variant"] = "s0_pareto_highdraft"
# cfg.setdefault("investigator", {})
# cfg["investigator"]["search_top_k"] = 8
# cfg["investigator"]["evidence_capsule_limit"] = 4
# with open(path, "w") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False)
# PY

# M1.1 low budget
# cp frozen/iter16_best/config/m1_1.iter16.yaml configs/m1_1_pareto_lowdraft.yaml
# python3 - <<'PY'
# import yaml
# path = "configs/m1_1_pareto_lowdraft.yaml"
# with open(path) as f:
#     cfg = yaml.safe_load(f)
# cfg["variant"] = "m1_1_pareto_lowdraft"
# cfg["orchestrator"]["max_steps"] = 2
# cfg.setdefault("investigator", {})
# cfg["investigator"]["search_top_k"] = 3
# cfg["investigator"]["evidence_capsule_limit"] = 2
# with open(path, "w") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False)
# PY

# M1.1 high budget
# cp frozen/iter16_best/config/m1_1.iter16.yaml configs/m1_1_pareto_highdraft.yaml
# python3 - <<'PY'
# import yaml
# path = "configs/m1_1_pareto_highdraft.yaml"
# with open(path) as f:
#     cfg = yaml.safe_load(f)
# cfg["variant"] = "m1_1_pareto_highdraft"
# cfg["orchestrator"]["max_steps"] = 5
# cfg.setdefault("investigator", {})
# cfg["investigator"]["search_top_k"] = 8
# cfg["investigator"]["evidence_capsule_limit"] = 4
# with open(path, "w") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False)
# PY

# A1 medium budget
# cp configs/a1.yaml configs/a1_pareto_middraft.yaml
# python3 - <<'PY'
# import yaml
# path = "configs/a1_pareto_middraft.yaml"
# with open(path) as f:
#     cfg = yaml.safe_load(f)
# cfg["variant"] = "a1_pareto_middraft"
# cfg["orchestrator"]["max_steps"] = 2
# cfg.setdefault("investigator", {})
# cfg["investigator"]["search_top_k"] = 3
# cfg["investigator"]["evidence_capsule_limit"] = 2
# with open(path, "w") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False)
# PY

# A1 low budget
# cp configs/a1.yaml configs/a1_pareto_lowdraft.yaml
# python3 - <<'PY'
# import yaml
# path = "configs/a1_pareto_lowdraft.yaml"
# with open(path) as f:
#     cfg = yaml.safe_load(f)
# cfg["variant"] = "a1_pareto_lowdraft"
# cfg["orchestrator"]["max_steps"] = 1
# cfg.setdefault("investigator", {})
# cfg["investigator"]["search_top_k"] = 2
# cfg["investigator"]["evidence_capsule_limit"] = 1
# with open(path, "w") as f:
#     yaml.safe_dump(cfg, f, sort_keys=False)
# PY

# Example shard command template for any draft config:
# python3 scripts/runner.py \
#   --config <CONFIG>.yaml \
#   --questions data/musique/questions_1000_shard0.json \
#   --output-dir results/<RUN_NAME>_shard0 \
#   --server-url http://localhost:8001/v1 \
#   --concurrency 24
#
# python3 scripts/runner.py \
#   --config <CONFIG>.yaml \
#   --questions data/musique/questions_1000_shard1.json \
#   --output-dir results/<RUN_NAME>_shard1 \
#   --server-url http://localhost:8002/v1 \
#   --concurrency 24
#
# python3 scripts/runner.py \
#   --config <CONFIG>.yaml \
#   --questions data/musique/questions_1000_shard2.json \
#   --output-dir results/<RUN_NAME>_shard2 \
#   --server-url http://localhost:8003/v1 \
#   --concurrency 24

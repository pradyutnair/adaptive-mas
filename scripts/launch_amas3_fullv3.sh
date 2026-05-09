#!/usr/bin/env bash
set -euo pipefail

cd /local/yzheng/pnair/workspace/adaptive-mas

stamp="${1:-$(date +%Y%m%d_%H%M%S)}"
root="results/amas3_joint_tfgrpo_fullv3_${stamp}"
mkdir -p "$root"

BASE_ARGS=(
  --worker qwen14b_nothink
  --synth-mode qwen14b_nothink
  --planner-model qwen3-14b
  --planner-mode nothink
  --planner-budget 768
  --solver-budget 768
  --synth-budget 768
  --use-sas-collapse
  --tau-sas-g 0.65
  --tau-sas-conf 0.75
  --adaptive-solver-budget
  --min-retrievals-per-solver 1
  --medium-retrievals-per-solver 2
  --no-repair
  --synth-no-cot
  --concurrency 8
)

summarize_tokens() {
  local predictions="$1"
  local output="$2"
  .venv/bin/python - "$predictions" > "$output" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
def meta(row, key):
    return row.get("metadata", {}).get(key, 0)
toks = [row.get("metadata", {}).get("total_tokens", row.get("total_tokens", 0)) for row in rows]
summary = {
    "rows": len(rows),
    "mean_total_tokens_actual": sum(toks) / len(toks) if toks else 0,
    "mean_planner_tokens": sum(meta(row, "planner_tokens") for row in rows) / len(rows) if rows else 0,
    "mean_solver_tokens": sum(meta(row, "solver_tokens") for row in rows) / len(rows) if rows else 0,
    "mean_synth_tokens": sum(meta(row, "synth_tokens") for row in rows) / len(rows) if rows else 0,
    "mean_rewrite_tokens": sum(meta(row, "rewrite_tokens") for row in rows) / len(rows) if rows else 0,
    "mean_sas_tokens": sum(meta(row, "sas_tokens") for row in rows) / len(rows) if rows else 0,
    "sas_exits": sum(1 for row in rows if row.get("metadata", {}).get("sas_collapse")),
}
print(json.dumps(summary, indent=2))
PY
}

run_eval() {
  local name="$1"
  local qfile="$2"
  local out="$3"
  shift 3
  mkdir -p "$out"
  {
    echo "START $name $(date -Is)"
    .venv/bin/python scripts/run_amas.py \
      --questions "$qfile" \
      --output-dir "$out" \
      --retriever-url http://node408:8003 \
      "${BASE_ARGS[@]}" \
      "$@"
    .venv/bin/python scripts/eval_offline.py \
      --predictions "$out/predictions.jsonl" \
      --questions "$qfile" \
      --output "$out/eval.json"
    summarize_tokens "$out/predictions.jsonl" "$out/token_summary.json"
    echo "DONE $name $(date -Is)"
  } 2>&1 | tee -a "$out/run.log"
}

{
  echo "root=$root"
  echo "policy=source-aware v3: musique plan6_retrieval3_compact8; hotpot compact5; 2wiki compact8; bamboogle compact5"
  run_eval musique data/musique/questions_1000_seedfull_combined.json "$root/musique1000" \
    --max-retrievals 3 --max-plan-subgoals 6 --synth-max-chunks 8 --synth-excerpt-chars 420
  run_eval hotpotqa data/hotpotqa/questions_1000_seed42.json "$root/hotpot1000" \
    --max-retrievals 2 --max-plan-subgoals 3 --synth-max-chunks 5 --synth-excerpt-chars 300
  run_eval 2wikimultihop data/2wikimultihop/questions_1000_seed42.json "$root/2wiki1000" \
    --max-retrievals 2 --max-plan-subgoals 3 --synth-max-chunks 8 --synth-excerpt-chars 420
  run_eval bamboogle data/bamboogle/questions_125.json "$root/bamboogle125" \
    --max-retrievals 2 --max-plan-subgoals 3 --synth-max-chunks 5 --synth-excerpt-chars 300
  echo "FULL_V3_DONE $(date -Is)"
} 2>&1 | tee -a "$root/fullv3.log"

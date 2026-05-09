#!/usr/bin/env bash
# Base method pilot runs: ISA + full MAS with generous defaults.
# Usage: bash scripts/launch_base_pilots.sh [25|100|full]
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT="${1:-25}"
TIMESTAMP=$(date +%Y%m%d_%H%M)
TAG="base_isa"

BASE_ARGS=(
  --worker qwen14b_nothink
  --synth-mode qwen14b_nothink
  --planner-model qwen3-14b
  --planner-mode think
  --solver-budget 1024
  --synth-budget 2048
  --use-isa
  --isa-max-rounds 3
  --isa-accept-threshold 0.7
  --use-bridge-resolver
  --bridge-g-threshold 0.45
  --repair
  --max-retrievals 3
  --max-plan-subgoals 6
  --max-repairs 2
  --synth-max-chunks 15
  --synth-excerpt-chars 700
  --concurrency 24
)

run_eval() {
  local name="$1" qfile="$2" out="$3"
  shift 3
  echo "=== $name ==="
  echo "  questions: $qfile"
  echo "  output:    $out"

  if [ "$LIMIT" != "full" ]; then
    .venv/bin/python scripts/run_amas.py \
      --questions "$qfile" \
      --output-dir "$out" \
      --retriever-url http://node408:8003 \
      --limit "$LIMIT" \
      "${BASE_ARGS[@]}" "$@"
  else
    .venv/bin/python scripts/run_amas.py \
      --questions "$qfile" \
      --output-dir "$out" \
      --retriever-url http://node408:8003 \
      "${BASE_ARGS[@]}" "$@"
  fi

  .venv/bin/python scripts/eval_offline.py \
    --predictions "$out/predictions.jsonl" \
    --questions "$qfile" \
    --output "$out/eval.json"

  echo "  eval: $(cat "$out/eval.json")"
  echo ""
}

OUTBASE="results/${TAG}_${LIMIT}q_${TIMESTAMP}"

run_eval "MuSiQue" \
  "data/pilot/musique_100_seed409.json" \
  "${OUTBASE}/musique"

run_eval "HotpotQA" \
  "data/pilot/hotpotqa_100_seed409.json" \
  "${OUTBASE}/hotpotqa"

run_eval "2Wiki" \
  "data/pilot/2wikimultihop_100_seed409.json" \
  "${OUTBASE}/2wiki"

run_eval "Bamboogle" \
  "data/bamboogle/questions_125.json" \
  "${OUTBASE}/bamboogle"

echo "=== All pilots complete ==="
echo "Results in: ${OUTBASE}/"

#!/usr/bin/env bash
# AMAS-Het full 1000q eval across {gate=off,bayesian,conformal} × {musique,hotpotqa,2wikimultihop,bamboogle}.
# Sequentially within each gate (vLLM throughput cap); parallel across CPUs is bound by
# --concurrency below. Each run logs to wandb project amas-eval.
#
# Args:
#   $1  out_root        e.g. results/run01_eval
#   $2  gates           e.g. "off bayesian conformal"  (space-separated)
#   $3  n_per_dataset   e.g. 1000

set -e
OUT_ROOT=${1:-results/run01_eval}
GATES=${2:-"off bayesian conformal"}
N=${3:-1000}
CONCURRENCY=${4:-16}
CFG=${5:-configs/base.yaml}

cd "$(dirname "$0")/.."
mkdir -p "$OUT_ROOT" logs

declare -A QF
QF[musique]=/local/yzheng/pnair/data/musique/questions_1000_seedfull_combined.json
QF[hotpotqa]=/local/yzheng/pnair/data/hotpotqa/questions_1000_seed42.json
QF[2wikimultihop]=/local/yzheng/pnair/data/2wikimultihop/questions_1000_seed42.json
QF[bamboogle]=/local/yzheng/pnair/workspace/reproduction/sparc-rag/data/bamboogle_125.json

PY=/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python
export PYTHONPATH=src

for gate in $GATES; do
  for ds in musique hotpotqa 2wikimultihop bamboogle; do
    out="$OUT_ROOT/$ds/$gate"
    mkdir -p "$out"
    log="logs/eval_${ds}_${gate}.log"
    echo "[$(date)] $ds $gate -> $out (log=$log)"
    n_arg=$N
    if [ "$ds" = "bamboogle" ]; then n_arg=125; fi
    $PY scripts/run_amas.py \
      --config "$CFG" \
      --questions "${QF[$ds]}" \
      --out-dir "$out" \
      --gate "$gate" \
      --n "$n_arg" \
      --concurrency "$CONCURRENCY" \
      --run-name "${ds}_${gate}_$(basename $OUT_ROOT)" \
      > "$log" 2>&1 || echo "  FAILED $ds $gate"
  done
done

echo "[$(date)] aggregating"
$PY scripts/aggregate_results.py --root "$OUT_ROOT" --out "$OUT_ROOT/aggregate.json"

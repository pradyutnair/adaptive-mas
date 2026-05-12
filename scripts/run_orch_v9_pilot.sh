#!/usr/bin/env bash
# v9 pilot: probe gate REMOVED, verifier ON, followups=2
set -uo pipefail
ROOT=/local/yzheng/pnair/workspace/adaptive-mas
PY="$ROOT/.venv/bin/python3"
RUN="$PY $ROOT/scripts/run_amas.py"
EVAL="$PY $ROOT/scripts/eval_offline.py"
STAMP="$(date +%Y%m%d_%H%M)"
RES="$ROOT/results/amas_pro/orch_v9_pilot100_${STAMP}"
mkdir -p "$RES"
BASE_PLANNER="--planner-model qwen3-14b --planner-mode nothink --planner-budget 768"
BASE_LM="--worker qwen14b_nothink --synth-mode qwen14b_nothink --solver-budget 768 --synth-budget 768"
BASE_PIPE="--max-retrievals 3 --concurrency 24"
WINNER="--no-repair --use-orchestrator --orch-probe-min-g 0.0 --orch-max-followups 2 --orch-min-confidence 0.65 --orch-budget 384 --orch-excerpt-chars 180 --orch-max-chunks 4 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim --synth-excerpt-chars 200 --synth-max-excerpts 5"
declare -A QFILES=( ["2wiki"]="$ROOT/data/2wikimultihop/questions_1000_seed42.json" ["hotpot"]="$ROOT/data/hotpotqa/questions_1000_seed42.json" ["musique"]="$ROOT/data/musique/questions_1000_seedfull_combined.json" ["bamboogle"]="$ROOT/data/bamboogle/questions_125.json" )
declare -A LIMIT=(["2wiki"]="100" ["hotpot"]="100" ["musique"]="100" ["bamboogle"]="125")
for DS in 2wiki hotpot musique bamboogle; do
  OUTDIR="$RES/$DS"
  mkdir -p "$OUTDIR"
  QFILE="${QFILES[$DS]}"; LIM="${LIMIT[$DS]}"
  echo "[run] $DS at $(date +%H:%M:%S) limit=$LIM"
  $RUN --questions "$QFILE" --output-dir "$OUTDIR" $BASE_PLANNER $BASE_LM $BASE_PIPE $WINNER --limit "$LIM" > "$OUTDIR/run.log" 2>&1
  $EVAL --predictions "$OUTDIR/predictions.jsonl" --questions "$QFILE" --output "$OUTDIR/eval.json" > "$OUTDIR/eval.log" 2>&1 || echo "[warn] eval failed for $DS"
done
echo
echo "===== V9 PILOT 100Q ====="
$PY - <<PYEOF
import json, os, statistics
res="$RES"
print(f"{'dataset':10s} {'EM':>6s} {'F1':>6s} {'CT':>6s} {'tok':>7s} {'n':>4s}")
for ds in ['2wiki','hotpot','musique','bamboogle']:
    d=os.path.join(res,ds); ev=os.path.join(d,'eval.json'); pr=os.path.join(d,'predictions.jsonl')
    if not os.path.exists(ev): print(f'{ds:10s} MISSING'); continue
    e=json.load(open(ev))
    toks=[json.loads(l).get('metadata',{}).get('total_tokens',0) for l in open(pr)]
    mt=int(statistics.mean(toks)) if toks else 0
    print(f"{ds:10s} {e['norm_em']:6.3f} {e['token_f1']:6.3f} {e['contain']:6.3f} {mt:7d} {e['total']:4d}")
PYEOF

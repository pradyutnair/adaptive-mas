#!/usr/bin/env bash
# v10 pilot: probe gate removed, tighter prompt, followups=0 (single LLM call), verifier ON
set -uo pipefail
ROOT=/local/yzheng/pnair/workspace/adaptive-mas
PY="$ROOT/.venv/bin/python3"
RUN="$PY $ROOT/scripts/run_amas.py"
EVAL="$PY $ROOT/scripts/eval_offline.py"
STAMP="$(date +%Y%m%d_%H%M)"
RES="$ROOT/results/amas_pro/orch_v10_pilot200_${STAMP}"
mkdir -p "$RES"
BASE_PLANNER="--planner-model qwen3-14b --planner-mode nothink --planner-budget 768"
BASE_LM="--worker qwen14b_nothink --synth-mode qwen14b_nothink --solver-budget 768 --synth-budget 768"
BASE_PIPE="--max-retrievals 3 --concurrency 24"
WINNER="--no-repair --use-orchestrator --orch-probe-min-g 0.0 --orch-max-followups 0 --orch-min-confidence 0.7 --orch-budget 384 --orch-excerpt-chars 180 --orch-max-chunks 4 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim --synth-excerpt-chars 200 --synth-max-excerpts 5"
declare -A QFILES=( ["2wiki"]="$ROOT/data/2wikimultihop/questions_1000_seed42.json" ["hotpot"]="$ROOT/data/hotpotqa/questions_1000_seed42.json" ["musique"]="$ROOT/data/musique/questions_1000_seedfull_combined.json" ["bamboogle"]="$ROOT/data/bamboogle/questions_125.json" )
declare -A LIMIT=(["2wiki"]="200" ["hotpot"]="200" ["musique"]="200" ["bamboogle"]="125")
for DS in 2wiki hotpot musique bamboogle; do
  OUTDIR="$RES/$DS"
  mkdir -p "$OUTDIR"
  QFILE="${QFILES[$DS]}"; LIM="${LIMIT[$DS]}"
  echo "[run] $DS at $(date +%H:%M:%S) limit=$LIM"
  $RUN --questions "$QFILE" --output-dir "$OUTDIR" $BASE_PLANNER $BASE_LM $BASE_PIPE $WINNER --limit "$LIM" > "$OUTDIR/run.log" 2>&1
  $EVAL --predictions "$OUTDIR/predictions.jsonl" --questions "$QFILE" --output "$OUTDIR/eval.json" > "$OUTDIR/eval.log" 2>&1 || echo "[warn] eval failed for $DS"
done
echo
echo "===== V10 PILOT 200Q vs clean base ====="
$PY - <<PYEOF
import json, os, statistics
res="$RES"
target = { '2wiki': (0.418, 0.480, 0.463, 10269), 'hotpot': (0.420, 0.529, 0.455, 8775), 'musique': (0.205, 0.295, 0.231, 8550), 'bamboogle': (0.456, 0.561, 0.472, 6504) }
print(f"{'dataset':10s} {'EM':>6s} {'F1':>6s} {'CT':>6s} {'tok':>7s}  ||  base  ΔEM   Δtok%")
for ds in ['2wiki','hotpot','musique','bamboogle']:
    d=os.path.join(res,ds); ev=os.path.join(d,'eval.json'); pr=os.path.join(d,'predictions.jsonl')
    if not os.path.exists(ev): print(f'{ds:10s} MISSING'); continue
    e=json.load(open(ev))
    toks=[json.loads(l).get('metadata',{}).get('total_tokens',0) for l in open(pr)]
    mt=int(statistics.mean(toks)) if toks else 0
    t=target[ds]
    dEM=e['norm_em']-t[0]; dTOK=100*(mt-t[3])/t[3]
    print(f"{ds:10s} {e['norm_em']:6.3f} {e['token_f1']:6.3f} {e['contain']:6.3f} {mt:7d}  ||  {t[3]:5d}  {dEM:+.3f}  {dTOK:+.1f}%")
PYEOF

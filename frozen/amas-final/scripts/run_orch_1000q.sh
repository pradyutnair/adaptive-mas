#!/usr/bin/env bash
# Reproduce v6/v7 results. Runs 1000q on {2wiki, hotpot, musique} + 125q bamboogle.
# Usage:
#   bash scripts/run_orch_1000q.sh "<extra flags>"
# Example (v7 — best quality):
#   AMAS_MAX_SUBGOALS=4 bash scripts/run_orch_1000q.sh \
#     "--no-repair --use-orchestrator --orch-max-followups 1 --orch-min-confidence 0.65 \
#      --orch-budget 384 --orch-probe-min-g 0.65 --orch-excerpt-chars 180 --orch-max-chunks 4 \
#      --synth-slim --synth-excerpt-chars 200 --synth-max-excerpts 5 \
#      --planner-budget 640 --solver-budget 640"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

# Python: prefer ROOT/.venv, else system python3
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
  PY="$ROOT/.venv/bin/python3"
else
  PY="$(command -v python3)"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RUN="$PY $ROOT/scripts/run_amas.py"
EVAL="$PY $ROOT/scripts/eval_offline.py"
STAMP="$(date +%Y%m%d_%H%M)"
RES="$ROOT/results/run_${STAMP}"
mkdir -p "$RES"

WINNER="${1:-}"
if [[ -z "$WINNER" ]]; then echo "usage: $0 \"<extra flags>\""; exit 1; fi

BASE_PLANNER="--planner-model qwen3-14b --planner-mode nothink --planner-budget 768"
BASE_LM="--worker qwen14b_nothink --synth-mode qwen14b_nothink --solver-budget 768 --synth-budget 768"
BASE_PIPE="--max-retrievals 3 --concurrency 24"

declare -A QFILES=(
  ["2wiki"]="$ROOT/data/2wikimultihop/questions_1000_seed42.json"
  ["hotpot"]="$ROOT/data/hotpotqa/questions_1000_seed42.json"
  ["musique"]="$ROOT/data/musique/questions_1000_seedfull_combined.json"
  ["bamboogle"]="$ROOT/data/bamboogle/questions_125.json"
)
declare -A LIMIT=(["2wiki"]="1000" ["hotpot"]="1000" ["musique"]="1000" ["bamboogle"]="125")

for DS in 2wiki hotpot musique bamboogle; do
  OUTDIR="$RES/$DS"
  if [[ -f "$OUTDIR/eval.json" ]]; then echo "[skip] $DS"; continue; fi
  mkdir -p "$OUTDIR"
  QFILE="${QFILES[$DS]}"; LIM="${LIMIT[$DS]}"
  echo "[run] $DS at $(date +%H:%M:%S) limit=$LIM"
  $RUN --questions "$QFILE" --output-dir "$OUTDIR" \
    $BASE_PLANNER $BASE_LM $BASE_PIPE \
    $WINNER --limit "$LIM" \
    > "$OUTDIR/run.log" 2>&1
  $EVAL --predictions "$OUTDIR/predictions.jsonl" --questions "$QFILE" --output "$OUTDIR/eval.json" \
    > "$OUTDIR/eval.log" 2>&1 || echo "[warn] eval failed for $DS"
done

echo
echo "===== FINAL 1000Q SUMMARY ====="
$PY - <<PYEOF
import json, glob, os, statistics
res = "$RES"
target = {
  "2wiki": (0.418, 0.480, 0.463, 10269),
  "hotpot": (0.420, 0.529, 0.455, 8775),
  "musique": (0.205, 0.295, 0.231, 8550),
  "bamboogle": (0.456, 0.561, 0.472, 6504),
}
print(f"{'dataset':10s} {'EM':>6s} {'F1':>6s} {'CT':>6s} {'tok':>7s}  ||  {'tgt_EM':>6s} {'tgt_F1':>6s} {'tgt_CT':>6s} {'tgt_tok':>7s}")
for ds in ["2wiki","hotpot","musique","bamboogle"]:
    d = os.path.join(res, ds)
    ev = os.path.join(d, "eval.json"); pr = os.path.join(d, "predictions.jsonl")
    if not os.path.exists(ev): print(f"{ds:10s}  MISSING"); continue
    e = json.load(open(ev))
    toks = [json.loads(l).get("metadata",{}).get("total_tokens",0) for l in open(pr)]
    mt = int(statistics.mean(toks)) if toks else 0
    t = target[ds]
    print(f"{ds:10s} {e['norm_em']:6.3f} {e['token_f1']:6.3f} {e['contain']:6.3f} {mt:7d}  ||  {t[0]:6.3f} {t[1]:6.3f} {t[2]:6.3f} {t[3]:7d}")
PYEOF

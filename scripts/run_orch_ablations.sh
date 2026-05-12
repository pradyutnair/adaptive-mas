#!/usr/bin/env bash
# Ablation grid for amas3 orchestrator + slim-synth + amas-pro components.
# Runs each cell on 200q × {musique, hotpot}, scores via eval_offline.py.

set -uo pipefail

ROOT=/local/yzheng/pnair/workspace/adaptive-mas
PY="$ROOT/.venv/bin/python3"
RUN="$PY $ROOT/scripts/run_amas.py"
EVAL="$PY $ROOT/scripts/eval_offline.py"
RES="$ROOT/results/amas_pro/ablations_$(date +%Y%m%d)"
mkdir -p "$RES"

# Shared base flags
BASE_PLANNER="--planner-model qwen3-14b --planner-mode nothink --planner-budget 768"
BASE_LM="--worker qwen14b_nothink --synth-mode qwen14b_nothink --solver-budget 768 --synth-budget 768"
BASE_PIPE="--max-retrievals 3 --concurrency 24"

# Cell flags map: cell_name -> "extra args"
declare -a CELLS=(
  "A0_baseline:--no-repair"
  "A1_repair_on:"
  "A2_slim_synth:--no-repair --synth-slim"
  "A3_orch_noverif:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.65"
  "A4_orch_verif:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.65 --orch-use-verifier --orch-verifier-min-confidence 0.6"
  "A5_orch_verif_slim:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.65 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim"
  "A6_orch_verif_slim_conf055:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.55 --orch-use-verifier --orch-verifier-min-confidence 0.55 --synth-slim"
  "A7_orch_verif_slim_conf075:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.75 --orch-use-verifier --orch-verifier-min-confidence 0.65 --synth-slim"
  "A8_orch_verif_slim_fu3:--no-repair --use-orchestrator --orch-max-followups 3 --orch-min-confidence 0.65 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim"
  "A9_orch_verif_slim_multiplan:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.65 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim --use-multi-plan --K-plans 3"
  "A10_orch_verif_slim_bridge:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.65 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim --use-bridge-resolver --bridge-g-threshold 0.45"
  "A11_orch_verif_slim_repair:--use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.65 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim"
  "A12_orch_verif_tighter_synth:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.65 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim --synth-max-excerpts 4 --synth-excerpt-chars 180"
  "A13_orch_verif_sas_collapse:--no-repair --use-orchestrator --orch-max-followups 2 --orch-min-confidence 0.65 --orch-use-verifier --orch-verifier-min-confidence 0.6 --synth-slim --use-sas-collapse"
)

# Datasets
declare -A QFILES=(
  ["hotpot"]="$ROOT/data/hotpotqa/questions_1000_seed42.json"
  ["musique"]="$ROOT/data/musique/questions_pilot200_seed42.json"
)
declare -A LIMIT=(["hotpot"]="200" ["musique"]="200")

for cellspec in "${CELLS[@]}"; do
  CELL="${cellspec%%:*}"
  EXTRA="${cellspec#*:}"
  for DS in hotpot musique; do
    OUTDIR="$RES/${CELL}__${DS}"
    if [[ -f "$OUTDIR/eval.json" ]]; then
      echo "[skip] $CELL $DS (eval exists)"; continue
    fi
    mkdir -p "$OUTDIR"
    QFILE="${QFILES[$DS]}"
    LIM="${LIMIT[$DS]}"
    echo "[run] $CELL $DS at $(date +%H:%M:%S)"
    $RUN --questions "$QFILE" --output-dir "$OUTDIR" \
      $BASE_PLANNER $BASE_LM $BASE_PIPE \
      $EXTRA --limit "$LIM" \
      > "$OUTDIR/run.log" 2>&1
    if [[ -f "$OUTDIR/predictions.jsonl" ]]; then
      $EVAL --predictions "$OUTDIR/predictions.jsonl" --questions "$QFILE" --output "$OUTDIR/eval.json" \
        > "$OUTDIR/eval.log" 2>&1 || echo "[warn] eval failed for $CELL $DS"
    else
      echo "[err] no predictions for $CELL $DS"
    fi
  done
done

echo
echo "===== ABLATION SUMMARY ====="
$PY - <<PYEOF
import json, glob, os, statistics
res = "$RES"
rows = []
for d in sorted(glob.glob(os.path.join(res, "*__*"))):
    cell = os.path.basename(d).split("__")[0]
    ds = os.path.basename(d).split("__")[1]
    ev = os.path.join(d, "eval.json")
    pr = os.path.join(d, "predictions.jsonl")
    if not os.path.exists(ev) or not os.path.exists(pr):
        continue
    e = json.load(open(ev))
    toks = []
    routes = {"orchestrator_answer":0, "other":0}
    for ln in open(pr):
        r = json.loads(ln)
        m = r.get("metadata", {})
        toks.append(m.get("total_tokens", 0))
        if m.get("topology") == "orchestrator_answer":
            routes["orchestrator_answer"] += 1
        else:
            routes["other"] += 1
    rows.append((cell, ds, e.get("norm_em",0), e.get("token_f1",0), e.get("contain",0), int(statistics.mean(toks)) if toks else 0, routes["orchestrator_answer"], len(toks)))
print(f"{'cell':35s} {'ds':8s} {'EM':>6s} {'F1':>6s} {'CT':>6s} {'tok':>7s} {'orch%':>7s}")
for c,ds,em,f1,ct,tk,oa,n in rows:
    print(f"{c:35s} {ds:8s} {em:6.3f} {f1:6.3f} {ct:6.3f} {tk:7d} {100*oa/n:6.1f}%")
PYEOF

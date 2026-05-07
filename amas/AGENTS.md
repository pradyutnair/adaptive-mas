# AMAS — Adaptive Multi-Agent Search

Active build at `/local/yzheng/pnair/workspace/amas/`. Plan at `/local/yzheng/pnair/.cursor/plans/amas_dual_gate_thesis.plan.md`.

## Hard rules

1. Do NOT modify anything in `/local/yzheng/pnair/workspace/reproduction/`. Read-only.
2. Always log GRPO + RoPE + evals to wandb (`amas-grpo`, `amas-rope`, `amas-eval`).
3. **Het regime only** for now (Qwen3-14B orchestrator + GPT-4o-mini agents). Skip Hom.

## Layout

```
src/amas/
  lm.py retriever.py library.py rope.py grpo.py orchestrator.py agents.py metric.py data.py config.py  (ported verbatim from hera/)
  ledger.py    # Evidence Ledger + Belief State
  probe.py     # turn-0 probe (G=3 self-consistency)
  pipeline.py  # multi-turn loop: probe → gate → MAS turns → gate
  gates/
    base.py conformal.py bayesian.py misc.py __init__.py
scripts/
  run_amas.py      # main runner (wandb-logged)
  smoke.py         # 3-question end-to-end smoke test
  calibrate_routeA.py  # split-conformal calibration (planned)
configs/base.yaml
data/      → reproduction/hera/data/   (symlink, read-only)
prompts/   → reproduction/hera/prompts/ (symlink, read-only)
```

## Infra

- vLLM Qwen3-14B × 3 replicas at `localhost:{8001,8002,8003}/v1`
- Retriever (E5 + wiki18) at `http://node408:8003/retrieve`, top-k=5
- OpenAI `gpt-4o-mini` for 8 HERA agents + Route A verifier
- Test sets: `/local/yzheng/pnair/data/{musique,hotpotqa,2wikimultihop}/questions_1000_*.json`, `reproduction/sparc-rag/data/bamboogle_125.json`

## Run examples

```bash
cd /local/yzheng/pnair/workspace/amas
. /local/yzheng/pnair/.env
export PYTHONPATH=src

# Smoke (3 questions × 3 gates)
python scripts/smoke.py

# 10q pilot, gate=off, no wandb
python scripts/run_amas.py \
  --questions /local/yzheng/pnair/data/musique/questions_1000_seedfull_combined.json \
  --out-dir results/p0/musique_off --n 10 --gate off --concurrency 4 --no-wandb

# Full 1000q × 4 gates (each writes own wandb run)
for gate in off bayesian conformal; do
  for ds in musique hotpotqa 2wikimultihop; do
    python scripts/run_amas.py \
      --questions /local/yzheng/pnair/data/$ds/questions_1000_*.json \
      --out-dir results/run01/$ds/$gate --gate $gate --concurrency 16
  done
done
```

## Phasing checkpoints

- P0 scaffold: smoke passes (3q × 3 gates, no exceptions, EM>0 on smoke-1).
- P1 ledger: 30q val MuSiQue with `--gate off` matches HERA-repro Acc ±1pt.
- P2 gates: 200q calibration of Route A; both gates Pareto-improve over off.
- P3 TF-GRPO: convergence on 200q val (EM ≥ HERA-repro+3pt OR <70% tokens).
- P4 RoPE: per-agent prompt evolution.
- P5 full eval: 1000q × 4 datasets × all gates.

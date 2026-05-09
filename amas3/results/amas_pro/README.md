# AMAS-PRO results

Probe-driven adaptive multi-agent collaborative search for multi-hop QA on MuSiQue / wiki18-corpus (top-K=5 fixed retrieval).

Two backbones, both training-free:

```
amas_pro/
├── homogeneous_qwen3_14b/         # all agents = Qwen3-14B (no closed-source)
│   ├── README.md
│   ├── 1000q_2026-05-01/          # FINAL homogeneous 1000q result (norm_em 0.226)
│   ├── strat100_baseline/         # stratified-100 validation pre-1000q (norm_em 0.230)
│   └── strat100_sas_ablation/     # ablation showing SAS-collapse FP rate
└── heterogeneous_qwen3_14b_4omini/   # planner+synth=Qwen3-14B, solver=GPT-4o-mini
    ├── 1000q_final/                # prior best 1000q (norm_em 0.211, synth-union)
    ├── 1000q_remainder596/         # mid-run resume after vLLM crash
    ├── 1000q_empties_rerun/        # rerun for empty answers
    ├── 1000q_initial/              # pre-synth-union 1000q
    ├── strat100/                   # stratified-100 with synth-union
    ├── strat100_pre_synthunion/    # earlier stratified-100
    └── strat100_qwen3_8b/          # Qwen3-8B planner variant
```

## Headline numbers (1000q MuSiQue, wiki18 top-K=5)

| Run | norm_em | F1 | tokens/q | Notes |
|---|---|---|---|---|
| `homogeneous_qwen3_14b/1000q_2026-05-01/` | **0.226** | 0.307 | 21,704 | All Qwen3-14B, training-free, RecursiveMAS-spirit recursion |
| `heterogeneous_qwen3_14b_4omini/1000q_final/` | 0.211 | 0.306 | ~8,000 | Qwen3-14B planner+synth + GPT-4o-mini solver, prior best |

**Delta:** homogeneous variant gains +1.5 EM by upgrading solver from GPT-4o-mini to Qwen3-14B-think + adding solver-recursion + final-answer-contract synth, but at 3× token cost (Qwen3-14B-think emits long `<think>` blocks).

## Architecture (training-free)

- **Planner** — Plan*RAG-style atomic decomposition emitting `<A.I>` parent-tag DAG
- **Probe layer** — N+1 parallel retrievals; per-probe groundedness `g ∈ [0,1]` via top-1 score, score gap, NE coverage, wh-target-extractable
- **Topology selector** — deterministic over probe signals: SAS / Linear / Fan-DAG / Bridge-first
- **Bridge resolver** — pre-step when `g(original) < 0.45` (low groundedness on original Q)
- **DAG executor** — same-depth nodes via `asyncio.gather` (parallel), different-depth sequential, parent answers interpolated via `<A.I>` tags
- **Solver** — Qwen3-14B-think extraction with entity-grounding defense + same-agent refinement (`RefineAnswerSpan`) when conf < 0.7. RecursiveMAS spirit, training-free.
- **Synth** — wh-target-aligned with explicit final-answer contract: identify wh-target type → list forbidden bridge entities (intermediate findings) → identify required relation → search final-evidence chunks for span satisfying all three. Two recursion rounds.
- **Findings Bus** — append-only NL collaboration substrate with parent-tag interpolation.

## Reproduce homogeneous 1000q

```bash
.venv/bin/python scripts/run_amas.py \
  --questions data/musique/questions.json \
  --output-dir results/amas_pro/homogeneous_qwen3_14b/1000q_2026-05-01 \
  --planner-model qwen3-14b \
  --worker qwen14b_think_small --synth-mode qwen14b_nothink \
  --solver-budget 1024 --synth-budget 1024 \
  --max-retrievals 3 --concurrency 10 \
  --use-bridge-resolver --bridge-g-threshold 0.45 \
  --synth-recursion-rounds 2

.venv/bin/python scripts/eval_offline.py \
  --predictions results/amas_pro/homogeneous_qwen3_14b/1000q_2026-05-01/predictions.jsonl \
  --questions data/musique/questions.json \
  --output results/amas_pro/homogeneous_qwen3_14b/1000q_2026-05-01/eval_final.json

.venv/bin/python scripts/scaling_law.py \
  --predictions results/amas_pro/homogeneous_qwen3_14b/1000q_2026-05-01/predictions.jsonl \
  --output results/amas_pro/homogeneous_qwen3_14b/1000q_2026-05-01/scaling_final.json
```

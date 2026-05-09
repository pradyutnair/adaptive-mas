# AMAS-PRO Homogeneous Qwen3-14B — 2026-05-01 overnight runs

## Headline result

**`amas_pro_homog14b_1000q_v4/` — MuSiQue 1000q, homogeneous Qwen3-14B, no GPT-4o-mini, training-free**

| Metric | Value |
|---|---|
| norm_em | **0.226 (22.6%)** |
| token_f1 | 0.307 |
| contain | 0.247 |
| answered | 953/1000 |
| mean tokens/q | 21,704 |
| mean wallclock/q | 291 s |

Compared to prior heterogeneous baseline (`amas_pro_synthunion_qwen3_14b_4omini_musique_1000q/` at **21.1 EM**): +1.5 EM with no API dependency.
Below user target band (24–27 EM) by 1.4 EM.

## Recursion-depth scaling-law (RQ3)

`scaling_final.json`:

| depth | share | mean_tok | mean_sec | norm_em | F1 | contain |
|---|---|---|---|---|---|---|
| 1 (errored) | 4.5% | 0 | 0 | 0.000 | 0.000 | 0.000 |
| 3 (synth-recursion) | 87.6% | 22 475 | 296 | **0.250** | 0.342 | 0.301 |
| 4 (+ bridge resolver) | 7.9% | 25 517 | 398 | 0.127 | 0.179 | 0.190 |

Bridge resolver actively hurts EM at depth=4. Synth-recursion at depth=3 carries the result.

## Topology distribution

linear (Plan*RAG DAG, parallel/sequential per hop): 91.4%
bridge_first: 2.5% • sas: 1.2% • fan_dag: 0.4% • errored: 4.5%

## Architecture (all training-free)

- Probe-grounded SAS-collapse (disabled in this run — see ablation `amas_pro_homog14b_strat100_v3/`)
- Plan*RAG-style atomic decomposition with `<A.I>` parent-tag interpolation
- Parallel/sequential DAG executor (same-depth nodes via `asyncio.gather`)
- **Solver-recursion** (RecursiveMAS spirit): same-agent re-extracts when conf < 0.7
- **Synth-recursion** (RecursiveMAS spirit): synth re-examines own answer with forbidden-bridge wh-target contract
- Bridge-resolver pre-step for low-groundedness questions
- Entity-grounding defense: solver downgrades unsupported answers to LOW_CONFIDENCE
- Final-answer contract in synth: explicit Step 1–5 wh-target / forbidden-bridge / required-relation reasoning

## Reference baselines

- **`amas_pro_homog14b_strat100_v4/`** — homogeneous Qwen3-14B stratified-100 (52 d2 + 32 d3 + 16 d4). norm_em 0.230, F1 0.313. Used to validate config before launching 1000q.
- **`amas_pro_homog14b_strat100_v3/`** — ablation: same stack with SAS-collapse enabled. SAS fired 11/24 questions with 91% false-positive rate (e.g. accepts "Calvert" for "Charles County"). EM 0.154 vs 0.230 without SAS. Evidence that probe-grounded entity-type SAS-collapse is precision-toxic and should be restricted to date/number/yes_no.

## Tokens-per-EM honesty

22.6 EM at 21.7k tokens/q is 3× the heterogeneous baseline's token cost. Sources:
1. Qwen3-14B-think emits long `<think>` blocks (500–1500 tokens per call) — vs GPT-4o-mini which doesn't.
2. Solver-recursion + Synth-recursion = ~doubles per-question LM-call count.
3. Concurrency was 10 with vLLM headroom for 20+ — runtime could have been ~4 h instead of 8.5 h.

## Reproduce

```bash
cd /local/yzheng/pnair/workspace/adaptive-mas
.venv/bin/python scripts/run_amas.py \
  --questions data/musique/questions.json \
  --output-dir results/amas_pro_homog14b_1000q_v4 \
  --planner-model qwen3-14b \
  --worker qwen14b_think_small --synth-mode qwen14b_nothink \
  --solver-budget 1024 --synth-budget 1024 \
  --max-retrievals 3 --concurrency 10 \
  --use-bridge-resolver --bridge-g-threshold 0.45 \
  --synth-recursion-rounds 2
```

Eval + scaling-law analysis:
```bash
.venv/bin/python scripts/eval_offline.py --predictions results/amas_pro_homog14b_1000q_v4/predictions.jsonl \
  --questions data/musique/questions.json --output results/amas_pro_homog14b_1000q_v4/eval_final.json
.venv/bin/python scripts/scaling_law.py --predictions results/amas_pro_homog14b_1000q_v4/predictions.jsonl \
  --output results/amas_pro_homog14b_1000q_v4/scaling_final.json
```

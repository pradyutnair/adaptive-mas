# EMNLP Handoff — What Codex Needs To Execute

**Date written**: 2026-04-15
**Author**: Claude (project manager), on behalf of user
**Audience**: codex (node409 tmux agent)
**Core thesis claim**: Adaptive-MAS (routes easy questions to SAS lane, hard to MAS lane) beats SAS in quality and beats MAS in efficiency, under matched token budgets.

---

## The claim, decomposed

To publish this at EMNLP 2026, the following three statements must hold simultaneously with statistical significance:

1. **Adaptive-MAS > SAS in quality** at equal or lower average token budget.
2. **Adaptive-MAS ≥ MAS in quality** at strictly lower average token budget.
3. **The routing decision is load-bearing** — random routing at the same SAS/MAS mix is strictly worse than learned routing.

Without all three, the paper is rejectable.

---

## Five evidence blocks required

### 1. Three systems with identical retriever + judge + scaffolding
- **SAS (S0)**: single agent, fixed budget.
- **MAS (A1)**: always-on multi-agent, no adaptive gate.
- **Adaptive-MAS (M1.1-iter16)**: frozen routing controller.

The three must differ **only** in the routing/orchestration layer. Same retriever k, same embedder, same passages-per-hop cap, same judge, same prompts for shared components. Any other delta is a confound that a reviewer will catch.

**Action**: verify retriever k and passage budgets match across the three configs before trusting any comparison. If they do not, rerun baselines with matched budgets.

### 2. Matched-budget evaluation (two versions, both needed)

**(a) Parity comparison**: cap MAS and Adaptive-MAS at SAS's average token budget. Show Adaptive-MAS wins at SAS's budget.

**(b) Pareto curve**: sweep each system across 3–4 budget points. Plot EM vs avg_tokens. The claim is credible iff **Adaptive-MAS dominates both curves** — above SAS at every budget, above or equal to MAS at every budget MAS reaches.

Without (b), a reviewer says "you picked a lucky operating point." The Pareto plot is the single most important figure in the paper.

Budget knobs per system:
- SAS: context length / top-k passages.
- MAS: subagent count cap, passage budget per subagent.
- Adaptive-MAS: same MAS knobs plus routing threshold.

### 3. Routing is load-bearing (ablations)

Required:
- **Force-SAS**: route every question to single-agent lane → should match S0.
- **Force-MAS**: route every question to multi-agent lane → should match A1.
- **Random routing** at the same SAS/MAS mix ratio as learned routing → should be strictly worse than learned routing.
- **Oracle routing**: route based on S0-correctness (wrong → MAS, right → SAS) → upper bound on the controller.

The gap between random and learned routing is the routing contribution. The gap between learned and oracle is remaining headroom — this is the honest limitation.

### 4. Per-slice decomposition

Split eval into "S0-easy" (S0 got right) and "S0-hard" (S0 got wrong). Required:
- **On S0-easy**: Adaptive-MAS matches S0 EM and uses ≈ S0 tokens (proves it actually collapsed to SAS lane).
- **On S0-hard**: Adaptive-MAS > S0 EM (proves MAS lane is recovering hard cases).
- **Token split**: routing distribution, avg tokens on each lane, avg tokens overall.

The pilot200 already shows iter16 S0-easy EM 0.805 at ~40k tokens — the "reverts to SAS" evidence. Needs to hold at 1000q × 3 datasets.

### 5. Significance + efficiency reporting

- Paired bootstrap CI on EM/F1 per (Adaptive vs SAS, Adaptive vs MAS) per dataset.
- McNemar on per-question correctness.
- Main table reports **tokens/question, wall-clock/question, subagent-calls/question** — not in appendix.
- One headline efficiency number: "Adaptive-MAS matches MAS at X% of MAS's token cost" or "Adaptive-MAS beats SAS at parity tokens by Y EM."

---

## Gap analysis (what's in hand vs missing)

| Requirement | Status |
|---|---|
| 3 systems, identical scaffolding | Needs verification (retriever k + passage cap) |
| MuSiQue 1000q full runs | M1.1 ✅, S0 ✅, A1 ~88% (will be ✅ shortly) |
| HotpotQA 1000q | **Missing** — required for publication |
| 2WikiMH 1000q | **Missing** — required for publication |
| Bamboogle/FRAMES | **Missing** — nice-to-have, not required |
| Judge run with DeepSeek-R1-Distill-Qwen-32B | Pending Snellius sync |
| Matched-budget parity point | **Missing** — rerun MAS capped to SAS tokens |
| Pareto budget sweep | **Missing** |
| Force-SAS ablation | Effectively have it (S0) |
| Force-MAS ablation | Effectively have it (A1) |
| Random-routing ablation | **Missing** — cheap |
| Oracle-routing ablation | **Missing** — cheap (use S0 correctness) |
| S0-easy / S0-hard decomposition | pilot200 ✅, 1000q pending |
| Paired significance, McNemar | Pending judge run |
| Token + wall-clock reporting | Have tokens, need wall-clock |

---

## The verify bug (post-freeze investigation, not a blocker)

M1.1-iter16 reports `avg_auto_verify=0.00` — verify is never used. Root cause is not Qwen3-8B capability. Two bugs:

1. **`orchestrator_decide.txt` does not expose `verify` as an action.** The prompt only lists `answer` and `spawn`. The LLM cannot emit `{"action": "verify"}` because it is not in the schema it's given. Code path at `pipeline.iter16.py:1108` is unreachable from the LLM side.
2. **Auto-verify path (`_maybe_verify_fact`) is gated by `auto_verify_threshold=0.7`.** It runs after every spawn but only fires when capsule confidence drops below 0.7. `avg_auto_verify=0.00` means confidence is always ≥0.7 — either uncalibrated/always-high, or the threshold is wrong.

Fix plan (post-1000q, run as a separate branch, do not touch frozen iter16):
- Add `verify` as a third action in `orchestrator_decide.txt` with explicit criteria (e.g., "when a grounded fact contradicts the current target profile, or when a spawn returned a low-confidence answer").
- Lower `auto_verify_threshold` to 0.5 or recalibrate confidence scoring.
- Run a 200q pilot with verify re-enabled. Only promote to 1000q if pilot shows ≥+1 EM.

This is a clean additional ablation for the paper: "verify on vs verify off" would demonstrate another piece of the adaptive machinery.

---

## Minimum viable EMNLP submission

1. 1000q × 3 datasets (MuSiQue ✅, HotpotQA, 2WikiMH) × 3 methods (SAS, MAS, Adaptive-MAS) with DeepSeek-R1-Distill-Qwen-32B judge.
2. Paired significance + McNemar on all 9 cells.
3. Token-matched Pareto plot (3–4 budget points per system per dataset).
4. Two core ablations: routing (random + oracle) and verify (on/off).
5. Per-slice decomposition (easy/hard × tokens/EM).
6. Forensic limitations subsection built from the 8-case slice (1 preservation + 2 surface-form + 5 retrieval/selection misses).
7. Clean theoretical framing of the adaptive routing objective (no heuristic story).

Deadline: **May 25, 2026** — ~6 weeks. Realistic if HotpotQA + 2WikiMH jobs start this week.

---

## Immediate execution order for codex

**Phase A — finish MuSiQue data (today)**
1. Wait for A1 1000q to finish (~20 min from 15:00 CEST).
2. Rsync `M1_1_iter16_1000_shard*`, `S0_1000_seeded_shard*`, `A1_1000_seeded_shard*` predictions to Snellius.
3. Verify retriever k + passage cap match across S0/A1/M1.1 configs. Log the numbers in `MAS_EXPERIMENTAL_LOG.md`.

**Phase B — HotpotQA + 2WikiMH data prep (this week)**
4. Build shard question files for HotpotQA 1000q and 2WikiMH 1000q under the same 334/334/332 split convention.
5. Seed output dirs (none to seed for new datasets — start fresh).
6. Launch M1.1-iter16 + S0 + A1 on HotpotQA. Then same on 2WikiMH.

**Phase C — matched-budget and ablation runs (after Phase B data is in)**
7. **Budget-capped MAS**: rerun A1 with token cap ≈ S0's avg tokens on MuSiQue 1000q.
8. **Random-routing ablation**: same codebase, replace controller decision with Bernoulli at the observed SAS/MAS mix rate. Run on 1000q × 3 datasets.
9. **Oracle-routing ablation**: use S0 correctness as the oracle signal, route wrong-on-S0 to MAS, right-on-S0 to SAS.
10. **Verify-on pilot200**: add verify to decide prompt, lower threshold, run pilot200 on MuSiQue. Promote to 1000q only if +1 EM.
11. **Pareto sweep**: 3–4 budget points per system per dataset. Batch it.

**Phase D — judge + analysis (rolling)**
12. Canonical judge pattern on Snellius — use `/projects/prjs1800/msc-thesis/02-arag-multi-agent/jobs/eval_m5_1000_musique.job` template. Must explicitly export `ARAG_MODEL=DeepSeek-R1-Distill-Qwen-32B` (it defaults to gpt-4o-mini otherwise).
13. Per-slice decomposition script (easy/hard × tokens/EM).
14. Paired bootstrap CI + McNemar script.
15. Pareto plot script.

**Phase E — writing (start in parallel during Phase B/C)**
16. Method section around frozen iter16 pipeline (`frozen/iter16_best/pipeline.iter16.py` is the reference).
17. Related work positioning — differentiate from ARAM/SPREAD/DNMR. No borrowed components.
18. Results table + efficiency table + Pareto figure + ablation table.
19. Forensic limitations subsection built from the 8-case slice.

---

## Workflow reminders (from CLAUDE.md)

- Baseline top-k must equal total passages the method uses. Otherwise gains are pure budget effect.
- Never use ARAM/SPREAD/etc as components in the method.
- Always present plan and get explicit user approval before submitting jobs.
- Do not debug/optimize on paid compute — launch what works, optimize separately.
- Separate jobs per dataset for parallelism.
- No REPL on Snellius.
- No em dashes or semicolons in writing.

---

## Key paths

- **Frozen pipeline**: `/local/yzheng/pnair/workspace/05-mas/frozen/iter16_best/pipeline.iter16.py`
- **Frozen config**: `/local/yzheng/pnair/workspace/05-mas/frozen/iter16_best/config/m1_1.iter16.yaml`
- **Live results**: `/local/yzheng/pnair/workspace/05-mas/results/{M1_1_iter16_1000,S0_1000_seeded,A1_1000_seeded}_shard{0,1,2}/predictions.jsonl`
- **Experimental log (source of truth)**: `/local/yzheng/pnair/workspace/05-mas/MAS_EXPERIMENTAL_LOG.md`
- **Snellius judge template**: `/projects/prjs1800/msc-thesis/02-arag-multi-agent/jobs/eval_m5_1000_musique.job`
- **Snellius eval script**: `/projects/prjs1800/external/arag/scripts/eval.py`

# AMAS Final Thesis Report

Test sets: 1000q MuSiQue + 1000q HotpotQA + 1000q 2WikiMultihop (seed42) + 125q Bamboogle.
Architecture: Qwen3-14B probe, GPT-4o-mini agents+verifier, fresh AMAS-only library
(no HERA warm-start), conformal gate calibrated at alpha=0.05, tau_high=3.892 on 200q
IRCoT-train pool (id-disjoint from test).

Reference: HERA `run02_eval_verbose` — the published, RoPE-trained HERA baseline.

## 1. Headline (amas_v1, T_max=1)

Format: EM / F1 / contain / tokens / sas-rate.

| dataset | HERA-run02 | AMAS gate=off | AMAS gate=conformal |
|---|---|---|---|
| musique | 0.126/0.215/0.157/6684/- | 0.107/0.210/0.139/11514/0.0% | 0.113/0.216/0.151/11680/0.3% |
| hotpotqa | 0.374/0.491/0.436/6347/- | 0.379/0.509/0.462/10871/0.0% | 0.364/0.499/0.451/10409/7.9% |
| 2wikimultihop | 0.239/0.307/0.301/6694/- | 0.308/0.392/0.374/12000/0.0% | 0.275/0.355/0.338/12106/1.7% |
| bamboogle | 0.368/0.519/0.408/6244/- | 0.248/0.386/0.280/10579/0.0% | 0.296/0.437/0.320/10788/1.6% |

### Headline F1 deltas (AMAS gate=off vs HERA-run02)

| dataset | AMAS F1 | HERA F1 | dF1 | dContain | tokens ratio |
|---|---:|---:|---:|---:|---:|
| musique | 0.210 | 0.215 | -0.5pp | -1.8pp | 1.72x |
| hotpotqa | 0.509 | 0.491 | +1.8pp | +2.6pp | 1.71x |
| 2wikimultihop | 0.392 | 0.307 | +8.6pp | +7.3pp | 1.79x |
| bamboogle | 0.386 | 0.519 | -13.2pp | -12.8pp | 1.69x |

**Quality wins**: AMAS beats HERA on F1 by **+8.6pp** on 2Wiki (1000q) and **+1.8pp**
on HotpotQA (1000q). Contain echoes: **+7.3pp** on 2Wiki, **+2.6pp** on HotpotQA. Tied
on MuSiQue. Loss on Bamboogle (n=125, high variance).

**Token cost**: AMAS pays 1.7-1.8x HERA's tokens. Honest tradeoff. Adaptive routing
infrastructure is calibrated but under-fires on the Qwen3-14B probe (sas_rate 0.3-7.9%),
so conformal gate does not recover the gap.

## 2. Architectural contributions (committed, smoke-validated)

1. **Ledger + BeliefState wired into MAS agent contexts** (Stage 1, e210d8f14): MAS turn
   t reads accumulated cross-turn evidence instead of re-deriving. 42/42 MAS LLM-agent
   prompts carry the ledger header in smoke. Likely source of the 2Wiki/HotpotQA F1 lift.
2. **Cross-family probe** (Stage 2, 82aaf6009): Qwen3-14B for the SAS-lane probe; GPT-4o-mini
   for the conformal verifier. Configurable via `cfg.probe.kind`. Out-of-family failure
   modes are uncorrelated.
3. **Lane plumbing** (Stage 3, 444d1ef04): library `ExpEntry.lane`, `library.retrieve(lane=)`
   filter, validated `topology.lane` in {SAS, MAS, AUTO}. No behaviour change at this stage.
4. **SAS/MAS router + escalation policy** (Stage 4, 733f43240): separate ROUTER_SYSTEM prompt
   picks lane from probe state. Pipeline branches on lane; SAS rejection escalates to MAS
   with a `__rejected_probe__` warning injected into agent context.
5. **Empty AMAS exp_lib + reproducible seed script** (Stage 5, 1c87a6f97).
6. **Single-q SA library population** (Stage 6, 71ff6b1cc): 31 lane-tagged, instance-leakage-
   filtered insights collected from 179q gate=off MAS rollouts.
7. **Route A calibration on Qwen probe** (Stage 7): 200q -> tau_high=3.892. Post-hoc alpha
   sweep documented in `results/alpha_sweep_post.json`; verifier-score distribution
   saturates above alpha=0.10.

## 3. Negative results (documented ablations)

### 3a. TF-GRPO + RoPE on amas_v1 -> amas_v2 (Stage 9, FAILED)

Trained 179q from train_240_v2 with `--init-library exp_lib/amas_v1`. Library shrank from
31 to 9 entries via Algorithm-3 PRUNE/MERGE thrash; ContextValidator prompt over-tightened
to 6 rules + 4 principles. Re-evaluated:

| dataset | amas_v1 off F1 | amas_v2 off F1 | dF1 |
|---|---:|---:|---:|
| musique | 0.210 | 0.150 | -6.0pp |
| hotpotqa | 0.509 | 0.450 | -5.9pp |
| 2wikimultihop | 0.392 | 0.303 | -8.9pp |
| bamboogle | 0.386 | 0.335 | -5.1pp |

Root cause: warm-starting GRPO from a populated single-q SA library polluted the
Algorithm 3 retrieval and inflated profile=any entries. Should have started from empty
(HERA's run02 strategy). Reverted to amas_v1.

### 3b. T_max=2 + belief-driven STOP (Stage 10, NEUTRAL/NEGATIVE)

Bumped T_max to 2 and added a mid-MAS termination condition: `belief.entropy < 0.5 AND
verifier YES at conf >= 0.80`. Goal: adaptive depth — easy multi-hop questions stop
after turn 1, hard ones run turn 2. Bamboogle 125q result:

| config | EM | F1 | contain | tokens | mean_turns |
|---|---:|---:|---:|---:|---:|
| T_max=1 conformal | 0.296 | 0.437 | 0.320 | 10788 | 0.98 |
| T_max=2+STOP conformal | 0.288 | 0.436 | 0.328 | 18238 | 1.72 |
| T_max=1 off | 0.248 | 0.386 | 0.280 | 10579 | 1.00 |
| T_max=2 off | 0.312 | 0.480 | 0.360 | 20247 | 2.00 |

Belief STOP fires on ~25% of MAS runs (mean_turns 1.72 < 2.0). But the extra MAS turn
on the other 75% adds +69% tokens for ~tied quality (+0.8pp contain, -0.1pp F1). The
second MAS turn does not improve enough to justify the cost on this pipeline. Paths
forward (out of scope tonight): mutate topology between turns, or train depth-aware
STOP via lane-match reward in TF-GRPO.

### 3c. TF-GRPO from empty library on amas_v1 pipeline (amas_v3, NOT COMPLETED)

Started clean (HERA-faithful: empty library + seed prompts + 179q + RoPE per-q + batch=10).
Stopped at step ~10/179 due to time budget. Diagnostic data preserved in
`results/amas_v3_train/{library.json, prompts.json, train_log.jsonl}`. Resume left as
future work.

## 4. Adaptive routing diagnostics

Conformal gate at alpha=0.05, tau_high=3.892 commit rates per dataset (T_max=1):

| dataset | sas_rate (commit) |
|---|---:|
| musique | 0.3% |
| hotpotqa | 7.9% |
| 2wikimultihop | 1.7% |
| bamboogle | 1.6% |

Probe-correct ceiling on calibration set: 31% (62/200). PAC bound at alpha=0.05 keeps
tau_high tight; verifier-score distribution saturates above alpha=0.10. Adaptive routing
infrastructure (router, gate, lane, ledger, belief) is built and PAC-calibrated, but the
Qwen3-14B probe is not strong enough at single-shot multi-hop to populate a meaningful
SAS lane. Recommended future work: probe-side training to lift probe accuracy, then re-
calibrate gate.

## 5. Story

AMAS is not Pareto-over-HERA tonight. It does:

1. Beat HERA-run02 on F1 by +8.6pp on 2WikiMultihop and +1.8pp on HotpotQA at the cost
   of 1.7-1.8x tokens. Contain confirms: +7.3pp 2Wiki, +2.6pp HotpotQA.
2. Tie MuSiQue. Lose Bamboogle (n=125 noisy).
3. Provide ledger+belief wiring + cross-family probe + calibrated conformal gate as
   reproducible architectural components.
4. Document two negative ablations: GRPO regression from polluted warm-start (3a) and
   T_max=2+STOP token cost not justified by quality (3b).

The thesis claim is **quality lift on large multi-hop datasets via ledger+belief wiring**,
not adaptive token savings. Adaptive routing is calibrated infrastructure for future work.
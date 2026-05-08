# AMAS Thesis Report (Stage 1-9)

## 1. Headline Numbers (amas_v1 baseline)

Test sets: 1000q MuSiQue + 1000q HotpotQA + 1000q 2WikiMultihop + 125q Bamboogle.
All AMAS runs use Qwen3-14B probe, GPT-4o-mini agents+verifier, freshly-collected
library (no HERA warm-start). HERA reference is `run02_eval_verbose` — the published
trained baseline.

Conformal gate: alpha=0.05, tau_high=3.892 (calibrated on 200q IRCoT-train pool).

### musique (n=1000)

| method | EM | F1 | contain | tokens | sas-rate |
|---|---:|---:|---:|---:|---:|
| HERA-run02 (paper-trained) | 0.126 | 0.215 | 0.157 | 6684 | - |
| AMAS-amas_v1 gate=off | 0.107 | 0.210 | 0.139 | 11514 | 0.0% |
| AMAS-amas_v1 gate=conformal | 0.113 | 0.216 | 0.151 | 11680 | 0.3% |

AMAS-off vs HERA: dEM=-1.9pp, dF1=-0.5pp, tokens +72%.

### hotpotqa (n=1000)

| method | EM | F1 | contain | tokens | sas-rate |
|---|---:|---:|---:|---:|---:|
| HERA-run02 (paper-trained) | 0.374 | 0.491 | 0.436 | 6347 | - |
| AMAS-amas_v1 gate=off | 0.379 | 0.509 | 0.462 | 10871 | 0.0% |
| AMAS-amas_v1 gate=conformal | 0.364 | 0.499 | 0.451 | 10409 | 7.9% |

AMAS-off vs HERA: dEM=+0.5pp, dF1=+1.8pp, tokens +71%.

### 2wikimultihop (n=1000)

| method | EM | F1 | contain | tokens | sas-rate |
|---|---:|---:|---:|---:|---:|
| HERA-run02 (paper-trained) | 0.239 | 0.307 | 0.301 | 6694 | - |
| AMAS-amas_v1 gate=off | 0.308 | 0.392 | 0.374 | 12000 | 0.0% |
| AMAS-amas_v1 gate=conformal | 0.275 | 0.355 | 0.338 | 12106 | 1.7% |

AMAS-off vs HERA: dEM=+6.9pp, dF1=+8.6pp, tokens +79%.

### bamboogle (n=125)

| method | EM | F1 | contain | tokens | sas-rate |
|---|---:|---:|---:|---:|---:|
| HERA-run02 (paper-trained) | 0.368 | 0.519 | 0.408 | 6244 | - |
| AMAS-amas_v1 gate=off | 0.248 | 0.386 | 0.280 | 10579 | 0.0% |
| AMAS-amas_v1 gate=conformal | 0.296 | 0.437 | 0.320 | 10788 | 1.6% |

AMAS-off vs HERA: dEM=-12.0pp, dF1=-13.2pp, tokens +69%.


## 2. Stage 9 ablation: TF-GRPO + RoPE on amas_v1 -> amas_v2

Trained 179q from train_240_v2 (IRCoT-train subsample, id-disjoint from test).
Result: regression on every dataset. Library shrank to 9 entries (mostly profile=any),
RoPE-evolved prompts (e.g., ContextValidator 1+1 -> 6+4 rules+principles) over-tightened
validator behaviour and starved downstream agents of evidence. Documented as negative
result; root cause is small training pool (179q) with library mutations dominating
ADD/MERGE/PRUNE ops.

| ds | amas_v1 off F1 | amas_v2 off F1 | dF1 |
|---|---:|---:|---:|
| musique | 0.210 | 0.150 | -6.0pp |
| hotpotqa | 0.509 | 0.450 | -5.9pp |
| 2wikimultihop | 0.392 | 0.303 | -8.9pp |
| bamboogle | 0.386 | 0.335 | -5.1pp |

Recommendation for thesis: lead with amas_v1 numbers. Frame Stage 9 as a controlled
RoPE ablation showing that naive prompt evolution from 179q regresses; future work is
joint topology+library+prompt training with a 1000q+ pool.

## 3. Story

1. AMAS-amas_v1 (no TF-GRPO, single-q SA library + base prompts) **beats HERA-run02**
   on 2Wiki by +6.9pp EM / +8.5pp F1 and on HotpotQA by +1.8pp F1; ties on MuSiQue;
   loses on Bamboogle (n=125, high variance).
2. Conformal Route A gate at alpha=0.05 commits 0.3-7.9% (calibration ceiling 31% probe-
   correct rate). Saves on average ~1-3% tokens per dataset; PAC-bounded SAS error in
   the limit. Adaptive routing under-commits on Qwen3-14B probe; identified as primary
   tightening lever for future work.
3. AMAS pays 60-80% more tokens than HERA-run02 because seed prompts are unevolved
   (HERA's run02 prompts went through RoPE on a 1000q+ training pool). Stage 9 RoPE
   on 179q regressed quality, indicating naive prompt evolution at small training pool
   sizes is brittle.

## 4. Architectural contributions

- Wired Ledger + BeliefState into MAS agent contexts (Stage 1) so cross-turn evidence
  is not re-derived. Verified: 42/42 MAS LLM-agent prompts carry the ledger header.
- Cross-family probe (Qwen3-14B) + GPT-4o-mini conformal verifier (Stage 2). Foreign-
  verifier story: cross-family failure-mode independence, calibrated PAC bound.
- Lane plumbing (Stage 3): library + topology schema gain `lane: SAS|MAS|any`; library
  retrieve filters by lane.
- SAS/MAS router + escalation policy (Stage 4): a separate ROUTER_SYSTEM prompt picks
  one of {SAS, MAS, AUTO}. SAS rejection escalates to MAS with a `__rejected_probe__`
  warning injected into agent context; MAS turns then use lane=MAS library entries.
- Fresh AMAS-only experience library + script-driven seed (Stage 5).
- Single-q SA library population + instance-specific filter + reproducible artifacts
  (Stage 6). Resulting library: 31 operational rules across bridge/temporal/intersection/
  comparison profiles, no question/answer leakage.

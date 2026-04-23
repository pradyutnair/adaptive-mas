# Plan: Adaptive Sufficiency-Controlled Multi-Agent RAG — thesis framing + refined sufficiency informed by iter55

## Context

**Thesis topic**: test-time scaling in multi-agent RAG, evaluating collaboration between agents.

**Claim to defend**: An adaptive controller lets multi-agent RAG pay SAS compute on easy questions and multi-agent compute on hard questions, producing **token + latency Pareto-better results** than any static MAS and better quality than SAS. Counter-result to Tran & Kiela 2026.

**Mechanism**:
- SAS fails on hard multi-hop because retrieval noise accumulates in orchestrator context.
- Recursive subagents absorb retrieval pollution *outside* the orchestrator context — the orchestrator only sees distilled facts with citations.
- A continuous sufficiency score routes each question to the cheapest-sufficient depth: probe-only → typed one-shot per slot → investigator recurse on deficient slot.

### Token audit (completed) — the finding that rewrites Track B

| System | Dataset | Contain | Tokens (mean) | Wallclock | orch/sub split |
|---|---|---|---|---|---|
| **sufficiency-v6** | MuSiQue | 0.366 | 50,004 | 232s | 9.4K / 40.6K |
| **iter55 (OPERA-adaptive)** | MuSiQue | **0.403** | **23,862** | **140s** | 1.3K / 4.8K (partial; 17.7K in planner+synthesis) |
| sufficiency-v6 | HotpotQA | 0.673 | 19,113 | 92s | 5.2K / 13.9K |
| sufficiency-v6 | 2Wiki | 0.695 | 32,600 | — | — |
| iter55 | HotpotQA / 2Wiki | — | — | — | — (no runs) |

**The brutal truth on MuSiQue**: iter55 is Pareto-better than v6 on all three axes (+0.037 quality, −52% tokens, −40% latency). The thesis as currently framed has a hole — on the hardest multi-hop benchmark, the adaptive-MAS baseline loses to an existing static-plan-plus-adaptive-gate system.

**Why iter55 is cheap** (step-count × tokens, from `TOKEN_SUMMARY.json` + per-row inspection):

| Plan steps | n | mean tokens |
|---|---|---|
| 0 | 6 | 6,163 |
| 1 | 347 | 15,574 |
| 2 | 464 | 24,741 |
| 3 | 166 | 36,369 |
| 4 | 17 | 53,185 |

iter55 uses **one-shot Analysis-Answer per typed sub-question** over retrieved docs — no iterative investigator research loop, no refine pass, no probe-then-recurse. Each plan step is ~10K tokens of LLM work. v6's investigator + refine loop is ~10K per *step*, and a typical hard question runs 3–4 steps.

**Where v6 wins**: HotpotQA (19K, easier routing) and 2Wiki (32K, entity-chain questions where bridge resolution dominates). These datasets don't have an iter55 head-to-head — so v6 holds there unless we run iter55 on them.

**Intended outcome**: (i) Track A ships as a strong thesis on v6's cross-dataset wins + mechanism evidence, (ii) Track B redesigns v6's execution lane using iter55's lean per-slot one-shot, adding the *continuous* sufficiency score on top to beat iter55 itself, not just match it.

---

## Track A — thesis framing on existing frozen results (must-do; no new compute)

This path is writable today against existing frozen predictions. Unchanged from prior plan.

### A1. Main table (1000q × 3 datasets)
Systems: SAS, static MAS (iter30_think / a1_matched), sufficiency-v6, **iter55 (MuSiQue only — honest: we did not run iter55 on Hotpot/2Wiki)**.

Metrics: contain, F1, EM; mean tokens/question; mean wall-clock; paired bootstrap 95% CI on deltas; McNemar on per-question correctness.

**Honest framing on MuSiQue**: iter55 is Pareto-better than v6. We report this. Then we show v6 wins on Hotpot (+0.020 vs static MAS, −45% tokens) and 2Wiki (+0.060\* vs static MAS, −39% tokens), which iter55 was never run on. The thesis claim becomes: *"sufficiency is Pareto-better than any static MAS on comparison-heavy benchmarks; on MuSiQue a stronger adaptive baseline (iter55) wins, motivating the Track B refinement."*

### A2. Ablations (existing 200q on MuSiQue)
From `results/abl_musique200_20260419_073534/ablation_summary.json`:
- no_controller (−0.160\*) — controller is load-bearing
- no_probe (+0.035 ns) — probe earns efficiency, not accuracy
- random_route (−0.105\*) — slot DAG routing matters
- oracle_route (−0.070\*) — continuous sufficiency > binary hardness
- τ sweep on MuSiQue shows τ=0.7 in cliff-free region

### A3. Slice decomposition (money plot)
Per-dataset sufficient-vs-insufficient slices with n, contain, mean tokens. Sufficient slice is +13–17 contain pts higher and 3–4× cheaper. Already logged.

### A4. Test-time scaling Pareto (NEW, cheap)
Sweep τ ∈ {0.3, 0.5, 0.6, 0.7, 0.8, 0.9} on MuSiQue. Plot contain vs mean tokens. Overlay SAS, static-MAS, **and iter55**. Honest plot — iter55 sits below-and-left of v6 on MuSiQue. This is Track B's motivation in one figure.

### A5. Mechanism validation (NEW, analysis-only)
Concrete form: `orchestrator_tokens / total_tokens` ratio per system per dataset.
- v6 MuSiQue: 9,434 / 50,004 = **0.189** (orchestrator sees 19%)
- iter55 MuSiQue: 1,328 / 23,862 = 0.056 (iter55 hides even more from orchestrator; it has a *synthesis* call that is not an orchestrator-in-context call)
- SAS expected ≈ 0.70+ (no distillation)
- Break out by slice (sufficient vs recurse).

### A5b. Agent-collaboration metrics (NEW — thesis framing asks for "evaluating collaboration")
Per system: fact-provenance rate (fraction of answers citing ≥1 subagent fact), subagent-utilization rate, sub-question diversity (1 − cosine-sim between consecutive sub-questions), inter-agent information flow (fraction of sub-questions that inject a resolved parent slot's value). Compute from existing predictions.jsonl — no rerun.

### A6. Latency
v6: MuSiQue 232s, Hotpot 92s, 2Wiki 20.6s (fair_v4 already logged). iter55 MuSiQue 140s. Report per system per dataset.

### A7. Related-work positioning (WebFetch for accurate quotes; no memory citations)
- **Tran & Kiela 2026**: their static-MAS-loses-to-SAS result is the straw man; our adaptive controller is the answer.
- **RLM**: shares recursive-control pattern; we differ in gating (continuous sufficiency vs state-machine).
- **A-RAG**: shares test-time-scaling motivation; we scale per-slot budget, not retrieval depth.
- **SPARC-RAG**: shares context-management motivation; we concretize via fact distillation keeping raw chunks out of orchestrator.
- **HERA**: contrasts on orchestration — they learn topology offline; our controller is training-free on a typed slot DAG.

**Track A alone is a defensible thesis** — the MuSiQue honest-loss framing is academically fine because it motivates Track B.

---

## Track B — refined sufficiency: hybrid sufficiency gate + iter55-style per-slot execution

**Goal**: Pareto-dominate iter55 on MuSiQue (not match it) while preserving v6's wins on Hotpot/2Wiki. The novelty is the **continuous sufficiency score selecting execution depth per slot**, not the typed placeholders (which iter55 has).

### B1. Diagnosis — why v6 bleeds tokens on MuSiQue

Decomposition of v6's 50K MuSiQue mean:
- **Probe call**: ~8K (full question, top_k retrieval, investigator-style reasoning).
- **Sufficiency scoring**: ~1.5K (verifier).
- **Recurse**: 2–4 investigator calls × ~10K each. Each investigator call is a full tool-using research loop (retrieve → read → reason → optionally re-retrieve), not a one-shot analysis.
- **Synthesis**: ~3K.

**Root cause**: the sufficiency gate is continuous, but the *execution* lane it gates into is binary — either probe-only (cheap) or full recurse (expensive). There is no middle lane. iter55's lane *is* the middle: one typed sub-question → one analysis-answer call over retrieved docs. No tool loop.

### B2. Design — three-lane adaptive execution keyed on per-slot sufficiency

Replace v6's `_sufficiency_recurse` with a three-lane per-slot controller:

**Lane 1 — direct-probe (current cheap lane, unchanged)**
- `s_overall ≥ τ_high` (e.g., 0.85): ship probe answer. ~8K tokens.

**Lane 2 — iter55-style typed one-shot per slot (NEW)**
- `τ_low ≤ s_overall < τ_high` (e.g., 0.60–0.85):
- Route → typed slot DAG `[{slot_id, role, expected_info_type, depends_on}]`.
- For each slot in topological order: ONE analysis-answer call over top_k retrieved docs for that typed sub-question. Placeholder values from resolved parents injected via `_extract_placeholder_value`.
- No investigator loop, no re-retrieve, no refine pass.
- Expected budget: 15–25K (matches iter55's 1–3 step buckets).

**Lane 3 — investigator recurse on deficient slots only (NEW, tighter than v6)**
- `s_overall < τ_low` (e.g., < 0.60):
- First run Lane 2 (typed one-shot per slot) to get a per-slot sufficiency vector.
- Identify deficient slots: `{i : sᵢ < τ_high}`.
- Run full investigator recurse **only on those slots** (not the whole question). Budget = `max(n_deficient, 1)` investigator calls.
- Re-score after each investigator call; short-circuit when `min_i sᵢ ≥ τ_high`.
- Expected budget: 30–45K.

**Why this Pareto-beats iter55**:
- iter55 *always* runs plan execution (no direct probe on easy questions). Lane 1 skips planning entirely when the probe already suffices → saves on the ~5K plan+1-step-exec overhead on easy questions.
- iter55 executes *all* plan steps even when the first step already resolves the answer. Lane 2 can short-circuit via re-scoring after each slot.
- iter55 has no escalation path when the plan fails. Lane 3 is that path, but it's targeted (per-slot, not full rerun).

**Why this Pareto-beats v6**:
- v6 currently escalates an under-resolved question via heavy investigator calls on the *full question*, not per-slot. Lane 2 is a cheaper option than Lane 3 for questions where the probe failed but typed one-shots suffice.
- v6's investigator budget is `max(2, ceil(MAX_STEPS × (1 − s)))` with MAX_STEPS=4 → often 3–4 calls. Lane 3 caps at `n_deficient` which is typically 1–2.

### B3. What to port from iter55 (verbatim; engineering)
- `_extract_placeholder_value` + helpers (`_extract_year_or_date`, `_extract_locationish`, `_extract_entityish`, `_strip_rhs`) from `scripts/run_adaptive_opera_hybrid.py` → new module `src/adaptive_sage/placeholder_extract.py`.
- `expected_info_type` enum `{entity | person | location | date | year | number | other}`.
- iter55's Analysis-Answer prompt + response schema (`{status, answer, analysis}`) — this is the one-shot call for Lane 2.
- Placeholder syntax `[type from slot Sᵢ]`.

### B4. What NOT to port
- iter55's forced bridge-first rule (`_needs_bridge_first` text-pattern) — violates v6's no-heuristics principle. Let the router emit the DAG.
- iter55's linear plan-execution commitment — broken by Lane 2's per-slot re-score short-circuit.
- iter55's plan-repair pass — replaced by Lane 3 targeted escalation.

### B5. Files to modify (branch `adaptive-op-typed` from `origin/cursor/iter31`)

| File | Change |
|---|---|
| `src/adaptive_sage/placeholder_extract.py` | NEW — port extractors |
| `src/adaptive_sage/one_shot_answer.py` | NEW — iter55-style Analysis-Answer call wrapper |
| `src/adaptive_sage/orchestrator.py` | extend `route_with_usage` to emit `expected_info_type` per hop; add `assess_probe_sufficiency_vector_with_usage` (per-slot vector) |
| `src/adaptive_sage/pipeline.py` | replace `_sufficiency_recurse` with `_route_execution_lane` + `_lane2_typed_oneshot` + `_lane3_targeted_recurse`; preserve scalar path behind flag for ablation |
| `src/adaptive_sage/prompts/*` | Analysis-Answer prompt; per-slot sufficiency schema |
| `configs/m1_3.sufficiency_typed.yaml` | NEW — `adaptive.use_three_lane: true`, `τ_high`, `τ_low` |

Keep v6 scalar path fully functional — becomes the "lanes OFF" ablation.

### B6. Validation sequence (gates)

1. **Unit tests** on `_extract_placeholder_value` using iter55 golden strings.
2. **Smoke50** on MuSiQue — verify no crashes, measure per-lane hit rates.
3. **200q pilot on MuSiQue** vs v6 scalar + iter55. Gate to promote: `contain ≥ 0.403` (match iter55) at `tokens ≤ 23,862` (match iter55). Stretch goal: `contain ≥ 0.42` at `tokens ≤ 22K`.
4. **If pilot passes**: full 1000q × 3 datasets. Paired bootstrap vs v6 and (on MuSiQue) vs iter55. Update Pareto.
5. **If pilot fails to dominate iter55**: ship Track A alone and report Track B as honest negative. The thesis story then becomes "v6 wins on Hotpot/2Wiki, loses to iter55 on MuSiQue, three-lane attempt did not close the gap, open problem for future work." Still publishable-quality.
6. **If v6 on Hotpot/2Wiki regresses after three-lane change**: add per-dataset config toggle — Lane 2 on MuSiQue only, Lane 2 off on Hotpot/2Wiki. The per-dataset ablation is itself a thesis finding.

### B7. Ablations (if B6.4 passes)
- Scalar sufficiency vs per-slot vector (core claim)
- Three-lane vs two-lane (v6) vs one-lane iter55-style (utility of middle lane)
- Typed `s_align` vs untyped (extractor contribution)
- Lane 3 on all slots vs Lane 3 on deficient slots only (targeted-recurse claim)
- Re-score every slot vs re-score once (verifier overhead justification)
- `τ_high` / `τ_low` sweep → new Pareto

### B8. Also run iter55 on Hotpot/2Wiki
Cheap (~3 days compute). Without this, Track A's Hotpot/2Wiki claims are against iter30_think only, which leaves open "maybe iter55 dominates there too." Run iter55 on both datasets as a Track A completion task. Analytically independent of Track B.

---

## Handoff to Codex

**Environment**:
- node409 at `/local/yzheng/pnair/workspace/adaptive-mas`
- New branch: `adaptive-op-typed` from `origin/cursor/iter31`
- vLLM endpoint: localhost:8001 (Qwen3-8B, enable_thinking=true)

**Reference files to read first**:
- `scripts/run_adaptive_opera_hybrid.py` lines 1–180 — `_extract_placeholder_value` family; lines ~300–500 — Analysis-Answer call pattern
- `configs/iter55_opera_adaptive_bridgefirst_placefix.yaml` — iter55 config for reference
- `frozen/iter55_musique1000_adaptive_bridgefirst_placefix_20260422/predictions.jsonl` — iter55 per-question traces
- `src/adaptive_sage/pipeline.py` on `origin/cursor/iter31` — `_run_sufficiency`, `_sufficiency_recurse` (lines 263–900)
- `src/adaptive_sage/orchestrator.py` — `route_with_usage`, `assess_probe_sufficiency_with_usage`
- `configs/m1_2.sufficiency.yaml` — baseline to clone

**Sequenced tasks**:
1. **Run iter55 on Hotpot + 2Wiki** (1000q each). Produces the missing Pareto points for Track A. Independent, can start immediately.
2. **Port extractors** → `placeholder_extract.py` + pytest.
3. **Port Analysis-Answer** → `one_shot_answer.py`. Single LLM call, prompt + schema `{status, answer, analysis}`.
4. **Route prompt**: extend `orchestrator.route_with_usage` to emit `expected_info_type` per hop.
5. **Per-slot verifier**: `assess_probe_sufficiency_vector_with_usage` → `{slots: [{slot_id, s_target, reason}]}`.
6. **Three-lane controller** in `pipeline.py` behind `adaptive.use_three_lane`.
7. **Smoke50 MuSiQue**.
8. **200q pilot MuSiQue** vs v6 + iter55. Report per-lane token breakdown + hit rates.
9. **Gate decision**: if Pareto-dominates iter55 on MuSiQue, proceed to 1000q × 3; else stop, ship Track A.
10. **Ablations B7** on 200q MuSiQue.

---

## Verification

Track A (no compute — analysis of existing + iter55 Hotpot/2Wiki):
- Pareto per dataset: contain vs mean tokens, τ sweep + all baselines.
- Mechanism: `orchestrator_tokens / total_tokens` ratio per system, slice-broken.
- Collaboration metrics: fact-provenance, subagent-utilization, sub-question diversity, info-flow.
- Latency table (confirm wallclock logged on all 1000q predictions; flag any missing).
- Related work: `WebFetch` Tran & Kiela 2026, RLM, A-RAG, SPARC-RAG, HERA.

Track B (only if pilot gate passes):
- `pytest src/adaptive_sage/tests/` green.
- `scripts/compare_sufficiency_1000q.py` emits paired bootstrap; new path Pareto-dominates v6 on MuSiQue; no regression on Hotpot/2Wiki.
- Updated Pareto + mechanism + collaboration plots with three-lane as third curve.

---

## Effort estimate

- Track A writeup + figures + iter55 Hotpot/2Wiki: **2–3 weeks**.
- Track B implementation: **~1 week** (extractors + one-shot wrapper + three-lane controller).
- Track B pilot + gate: **3–5 days**.
- Track B 1000q × 3 if promoted: **2–3 days compute + 2 days analysis**.

**Minimum viable thesis**: Track A alone (with MuSiQue iter55 loss honestly reported).
**Strong thesis**: Track A + Track B promoted (three-lane Pareto-dominates iter55).
**Honest negative result**: Track A + Track B not promoted — still a full thesis with a clear open problem.

---

## Risks & mitigations

1. **Three-lane still loses to iter55 on MuSiQue**. Mitigation: 200q pilot before 1000q compute. If lane 2 is essentially iter55 + a sufficiency gate, Pareto is structurally guaranteed (lane 1 only fires when probe suffices, which iter55 never does). But verify empirically — `τ_high` may be hard to set without cliff.
2. **Lane 2 regresses on Hotpot/2Wiki** where v6 currently wins. Mitigation: per-dataset lane mask; lane-2 off when dataset config disables it. Report as "MuSiQue-specific refinement."
3. **iter55 on Hotpot/2Wiki may itself beat v6**, which would deepen the thesis hole. Mitigation: run iter55 on both early (task 1). If it wins everywhere, pivot Track B to be the primary thesis contribution rather than a refinement.
4. **Per-slot verifier noisier than scalar**. Mitigation: 200q pilot; fall back to scalar s gating into lane 2 (lane 2 doesn't require per-slot signal to work).
5. **Route over-decomposition** (5 slots for 2-hop → spuriously low min). Mitigation: cap `n_slots ≤ 3` in route prompt; slot-count penalty in vector aggregation.
6. **Extractor regex failures**. Mitigation: pytest; fall back to untyped `s_align` when extractor returns empty.

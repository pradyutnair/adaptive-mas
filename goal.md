 Plan: Verification Depth Scaling Study for Multi-Agent Multi-Hop QA                                                     
                                     
 Context

 MSc thesis "Retrieval-Augmented Generation with Multi-Agent Collaborative Search" (deadline June 19, 2026). AMAS-PRO V3
 is the strongest base system (0.193/0.414/0.429/0.400 EM on MuSiQue/2Wiki/HotpotQA/Bamboogle, Qwen3-14B no-think,
 wiki18). We've added: LLM verifier, LLM router, context-aware child retrieval, structured working memory, enhanced
 traces. These are implemented and smoke-tested (5q passed).

 Target headline finding: "Per-hop LLM verification is a more cost-effective scaling axis than additional retrieval
 attempts for multi-hop QA." Demonstrated via Pareto frontier: verification configs dominate retrieval-only configs in
 EM-vs-tokens space.

 Why max_retrievals is not a good scaling axis: With confidence-based early stopping, the solver exits early when
 confidence is high — increasing the ceiling from 3→5 doesn't change behavior on most questions. Best-of-N enforcement
 (always exhaust budget) fixes this but costs 3x LLM calls (~100s/q vs target 15-20s/q). The real axis is what triggers
 retries and how: heuristic confidence vs LLM verification.

 Positioning vs Related Work

 ReAgent (Zhao et al., EMNLP 2025): Multi-agent + LLM verification + backtracking for multi-hop QA.
 - Uses GPT-4o only ($645-546/dataset). We use Qwen3-14B (free, local GPUs).
 - Their novelty is state rollback (revert knowledge sets). Ours is verify-then-retry (rewrite query + re-retrieve within
 same hop).
 - No cost-efficiency analysis, no adaptive routing, no retrieval budget scaling.
 - Same 3 datasets (HotpotQA, 2Wiki, MuSiQue, 1000q each).

 SPARC-RAG (Yang et al., 2026): W x D scaling framework for flat RAG.
 - W = parallel rewritten queries per round, D = sequential rounds. Grid search on 100q subset, scale best to full eval.
 - Does NOT decompose questions — rewrites the SAME question W different ways. Our approach decomposes into a DAG of
 sub-questions.
 - Verification at answer level (whole answer, end of round). Ours at hop level (per sub-question).
 - Uses Qwen2.5-7B/Qwen3-32B. Our Qwen3-14B already competitive because decomposition helps on multi-hop.

 Our framing: "SPARC-RAG scales flat retrieve-and-generate along W x D. ReAgent adds backtracking to GPT-4o. We scale
 hierarchical decompose-and-solve along verification depth with open-source models, showing per-hop verification prevents
 error cascading across hops more cost-effectively than additional retrieval attempts or parallel speculation."

 ---
 System Architecture (already implemented)

 Location: /local/yzheng/pnair/workspace/adaptive-mas/src/amas3/

 Pipeline flow:
 Question -> Probe -> Route (SAS or DAG) -> [DAG: Plan -> Execute DAG -> Repair] -> Synthesize -> Answer

 Key components (all implemented, smoke-tested):

 - LLM Router (router.py): RouteQuestion dspy.Signature. direct -> SAS lane, decompose -> DAG lane. Toggle:
 use_llm_router. Fallback: heuristic g >= tau_sas_g.
 - Per-Hop Verifier (verifier.py): VerifyExtraction dspy.Signature. Returns accept/reject + reason. ~100-150 tokens/call.
 - Working Memory (working_memory.py): Renamed FindingsBus -> WorkingMemory. EvidenceCapsule with evidence_excerpts,
 query_rewrites, verification, parent_ids.
 - Context-Aware Retrieval (pipeline.py): Parent Q/A appended to child retrieval query.
 - Enhanced Traces (pipeline.py, run_amas.py): route, router_reason, verifier_calls/accepts/rejects,
 working_memory_capsules, per_node_retrievals/latency, config_snapshot.

 What STILL needs changing:

 1. Revert solver.py to retry-on-trigger (currently has best-of-N enforcement, causes ~100s/q)

 Current: always perform max_retrievals attempts, no early stopping.
 New: extract -> check trigger -> retry only if triggered, up to max_retrievals ceiling.

 Trigger depends on config (one rule per config, no overlap):
 - Verifier OFF (V0, V1, R1): retry if confidence < 0.5. Stop when confidence >= 0.5 OR attempts == max_retrievals. This
 is original AMAS-PRO V3 behavior.
 - Verifier ON (V2, V3): retry if NOT verifier.accept. Stop when verifier accepts OR attempts == max_retrievals. Verifier
 replaces confidence as the gate — not a union. This ensures V1 vs V2 isolates the trigger mechanism cleanly.

 Per config:
 - V0 (max_r=1): always 1 attempt, no retries possible
 - V1 (max_r=3, no verifier): retry on conf < 0.5, stop on conf >= 0.5
 - V2 (max_r=3, verifier): retry on verifier reject, stop on verifier accept
 - V3 (max_r=3, verifier, repair): same trigger as V2, plus system-level repair
 - R1 (max_r=3, no verifier, top_k=10): same trigger as V1 but more chunks per retrieval

 File: src/amas3/solver.py — revert from v4 (best-of-N) back to retry-on-trigger with verifier integration.

 2. Make top_k configurable in retriever.py

 Currently: _FIXED_TOPK hardcoded. Add top_k parameter to retrieve() with default=5. Pass through from AmasPipelineConfig.

 File: src/amas3/retriever.py — add top_k param to retrieve().
 File: src/amas3/pipeline.py — pass config top_k to retriever calls.
 File: scripts/run_amas.py — add --top-k CLI flag.

 ---
 Scaling Study Design

 Methodology (following SPARC-RAG)

 1. Grid search on 200q subset x 4 datasets (MuSiQue, 2Wiki, HotpotQA, Bamboogle)
 2. Select best config by mean norm_em across {MuSiQue, 2Wiki, HotpotQA}; Bamboogle is OOD
 3. Scale best config to 1000q x 3 + 125q Bamboogle for final numbers
 4. Report all 5 configs in the grid search table (like SPARC-RAG Table 7)

 Verification Depth Grid (5 configs, all use heuristic SAS routing)

 Critical design choice: All grid search configs use heuristic routing (g >= tau_sas_g, 0.65). LLM router is tested
 separately as an ablation. This isolates the verification depth axis cleanly — no confound between router and verifier
 improvements.

 ┌───────────────────┬──────────┬──────────────┬───────┬───────┬────────┬─────────────────────────────────────────────┐
 │      Config       │ Verifier │    Retry     │ max_r │ top_k │ Repair │                What it tests                │
 │                   │          │   trigger    │       │       │        │                                             │
 ├───────────────────┼──────────┼──────────────┼───────┼───────┼────────┼─────────────────────────────────────────────┤
 │ V0: One-shot      │ off      │ none         │ 1     │ 5     │ off    │ Floor: single extract per hop, no retries   │
 ├───────────────────┼──────────┼──────────────┼───────┼───────┼────────┼─────────────────────────────────────────────┤
 │ V1: Confidence    │ off      │ conf < 0.5   │ 3     │ 5     │ off    │ Original AMAS-PRO V3 behavior               │
 │ retry             │          │              │       │       │        │                                             │
 ├───────────────────┼──────────┼──────────────┼───────┼───────┼────────┼─────────────────────────────────────────────┤
 │ V2: Verified      │ on       │ verifier     │ 3     │ 5     │ off    │ Per-hop LLM verification as retry trigger   │
 │                   │          │ reject       │       │       │        │                                             │
 ├───────────────────┼──────────┼──────────────┼───────┼───────┼────────┼─────────────────────────────────────────────┤
 │ V3: Verified +    │ on       │ verifier     │ 3     │ 5     │ on     │ Full system: verification + system-level    │
 │ Repair            │          │ reject       │       │       │        │ repair                                      │
 ├───────────────────┼──────────┼──────────────┼───────┼───────┼────────┼─────────────────────────────────────────────┤
 │ R1: More          │ off      │ conf < 0.5   │ 3     │ 10    │ off    │ Alternative axis: more chunks instead of    │
 │ retrieval         │          │              │       │       │        │ smarter verification                        │
 └───────────────────┴──────────┴──────────────┴───────┴───────┴────────┴─────────────────────────────────────────────┘

 Hypotheses (stated upfront, falsifiable)

 - H1: V0 ~ V1 (confidence retries are rarely triggered, adding max_r from 1->3 with heuristic trigger doesn't help much)
 — validates that retrieval depth alone is a weak axis
 - H2: V2 >> V1 (per-hop LLM verification catches errors that confidence misses) — the headline finding
 - H3: V2 dominates R1 on Pareto frontier (verification is more token-efficient than more retrieval) — the
 alternative-axis comparison
 - H4: V3 >= V2 (repair adds value when verification catches real failures) — system-level repair matters

 If H1 fails (V1 >> V0): confidence retries ARE useful, and the scaling story becomes "verification + retries" rather than
  "verification instead of retries." Still publishable, different framing.

 If H2 fails (V2 ~ V1): verification doesn't help — negative result. Report as "per-hop verification is not cost-effective
  for Qwen3-14B" and fall back to V1 as best config. Still useful finding.

 Pairwise comparisons and what they isolate

 - V0 vs V1: Does retry-on-low-confidence help? (retrieval depth, same verification: none)
 - V1 vs V2: Does LLM verification help over heuristic confidence? (same retrieval budget, different trigger)
 - V1 vs R1: Does more retrieval (top_k 5->10) help without verification? (same trigger, more chunks)
 - V2 vs R1: Verification vs more retrieval — which is more token-efficient? (Pareto comparison)
 - V2 vs V3: Does system-level repair add value on top of per-hop verification?

 Multi-hop error cascading argument

 Per-hop verification prevents errors from propagating across hops. In a 3-hop chain A->B->C, an error at A corrupts B and
  C. Answer-level verification (SPARC-RAG) catches this only at the end. Per-hop verification catches it at A, triggers a
 retry, and prevents cascading. This is our structural advantage — quantified by comparing per-node accuracy across V1 and
  V2.

 Time estimate

 - 200q grid search: 5 configs x (200q x 3 + 125q Bamboogle) = 3,625 questions
 - At ~15-20s/q with retry-on-trigger: ~15-20h sequential
 - With 3 GPUs, 3 shards per dataset: ~5-7h total
 - 1000q final eval: 1 config x (1000q x 3 + 125q) = 3,125 questions, ~3-4h with 3 GPUs
 - Ablations: ~6 configs x 200q x 4 datasets on best config, ~4-5h with 3 GPUs

 ---
 Ablations (200q, best config from grid)

 Component Ablations

 ┌─────────────────┬──────────────────────────────────────────┬───────────────────────────────────────────┐
 │    Ablation     │               What changes               │               What it tests               │
 ├─────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────┤
 │ No verifier     │ Verifier OFF (V1 config)                 │ Already in grid — reuse V1 result         │
 ├─────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────┤
 │ No repair       │ max_repairs=0                            │ Do system-level second chances matter?    │
 ├─────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────┤
 │ No synthesizer  │ Final node answer directly               │ Is LLM aggregation + bridge guard needed? │
 ├─────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────┤
 │ No bridge guard │ Synthesizer without bridge entity filter │ Bridge guard contribution alone           │
 └─────────────────┴──────────────────────────────────────────┴───────────────────────────────────────────┘

 Collaboration Ablations

 ┌───────────────────────────┬──────────────────────────────────────────────┬─────────────────────────────────────────┐
 │         Ablation          │                 What changes                 │              What it tests              │
 ├───────────────────────────┼──────────────────────────────────────────────┼─────────────────────────────────────────┤
 │ Answer-only working       │ Synthesizer sees answers only, no            │ Does richer WM help beyond answer       │
 │ memory                    │ evidence/confidence                          │ chains?                                 │
 ├───────────────────────────┼──────────────────────────────────────────────┼─────────────────────────────────────────┤
 │ Single-agent at matched   │ No decomposition, same mean token budget     │ Is multi-agent decomposition worth the  │
 │ budget                    │                                              │ overhead?                               │
 └───────────────────────────┴──────────────────────────────────────────────┴─────────────────────────────────────────┘

 Adaptivity Ablations

 ┌────────────────────────────┬──────────────────────────────────────────┬────────────────────────────────────┐
 │          Ablation          │               What changes               │           What it tests            │
 ├────────────────────────────┼──────────────────────────────────────────┼────────────────────────────────────┤
 │ LLM router vs heuristic    │ Best config + LLM router ON vs heuristic │ Is LLM router worth ~100 tokens/q? │
 ├────────────────────────────┼──────────────────────────────────────────┼────────────────────────────────────┤
 │ No context-aware retrieval │ Disable parent Q/A in child retrieval    │ Does context-aware retrieval help? │
 └────────────────────────────┴──────────────────────────────────────────┴────────────────────────────────────┘

 ---
 Code Changes Required

 Modify (before grid search)

 ┌────────────────────────┬───────────────────────────────────────────────────────────────────────────────────┬────────┐
 │          File          │                                      Change                                       │ ~Lines │
 ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼────────┤
 │ src/amas3/solver.py    │ Revert from best-of-N to retry-on-trigger. Verifier as alternative trigger when   │ ~80    │
 │                        │ enabled. Early stop on accept + high confidence.                                  │        │
 ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼────────┤
 │ src/amas3/retriever.py │ Make top_k a parameter to retrieve() with default=5                               │ ~10    │
 ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼────────┤
 │ src/amas3/pipeline.py  │ Pass top_k from config to retriever                                               │ ~5     │
 ├────────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼────────┤
 │ scripts/run_amas.py    │ Add --top-k CLI flag                                                              │ ~5     │
 └────────────────────────┴───────────────────────────────────────────────────────────────────────────────────┴────────┘

 Already implemented (verified via smoke test)

 ┌─────────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
 │            File             │                                   Status                                    │
 ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ src/amas3/verifier.py       │ Working — VerifyExtraction signature, 82% accept rate on smoke test         │
 ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ src/amas3/router.py         │ Working — RouteQuestion signature, correctly routes easy->SAS               │
 ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ src/amas3/working_memory.py │ Working — EvidenceCapsule, capsules populated in traces                     │
 ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ src/amas3/types.py          │ Working — EvidenceCapsule dataclass                                         │
 ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ src/amas3/pipeline.py       │ Working — LLM router integration, context-aware retrieval, new trace fields │
 ├─────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
 │ scripts/run_amas.py         │ Working — trace fields saved to predictions.jsonl                           │
 └─────────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘

 New (for sweep execution)

 ┌────────────────────────────┬───────────────────────────────────────────────────────────────┬────────┐
 │            File            │                            Purpose                            │ ~Lines │
 ├────────────────────────────┼───────────────────────────────────────────────────────────────┼────────┤
 │ scripts/run_grid_search.sh │ Shell script: runs 5 configs x 4 datasets with 3-GPU sharding │ ~80    │
 ├────────────────────────────┼───────────────────────────────────────────────────────────────┼────────┤
 │ scripts/analyze_grid.py    │ Pareto frontier, config selection, tables                     │ ~100   │
 └────────────────────────────┴───────────────────────────────────────────────────────────────┴────────┘

 ---
 Execution Plan

 ┌─────────────────────┬────────┬─────────────────────────────────────────────────────────────────────────────────────┐
 │        Phase        │  Time  │                                        What                                         │
 ├─────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ 0: Fix solver +     │ ~1h    │ Revert solver to retry-on-trigger, make top_k configurable                          │
 │ top_k               │        │                                                                                     │
 ├─────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ 1: Re-validate      │ ~5 min │ 5q smoke test with corrected solver — verify ~15-20s/q, traces complete             │
 ├─────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ 1.5: R1 ctx check   │ ~1 min │ Run 1 three-hop question with top_k=10 — verify it fits in Qwen3-14B's 8192 ctx     │
 │                     │        │ without truncation                                                                  │
 ├─────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ 2: Kill old job     │ ~1 min │ Kill PID 2085869 (101q validation using wrong solver)                               │
 ├─────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ 3: 200q grid search │ ~5-7h  │ 5 configs x (200q x 3 + 125q) on 3 GPUs                                             │
 ├─────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ 4: Select best      │ ~30    │ Analyze grid, plot Pareto, select winner                                            │
 │ config              │ min    │                                                                                     │
 ├─────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ 5: 1000q final eval │ ~3-4h  │ Best config on full datasets                                                        │
 ├─────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────────────────┤
 │ 6: Ablations        │ ~4-5h  │ Component + collaboration + adaptivity ablations on 200q                            │
 └─────────────────────┴────────┴─────────────────────────────────────────────────────────────────────────────────────┘

 Total wall-clock: ~14-17h across 2 days.

 ---
 Datasets

 - MuSiQue 1000q: data/musique/questions_1000_seedfull_combined.json (200q subset: first 200 or stratified)
 - 2WikiMultiHopQA 1000q: data/2wikimultihop/questions_1000_seed42.json
 - HotpotQA 1000q: data/hotpotqa/questions_1000_seed42.json
 - Bamboogle 125q: data/bamboogle/questions_125.json

 Best config criterion: highest mean norm_em across {MuSiQue, 2Wiki, HotpotQA}. If within 0.005 EM, prefer lower mean
 tokens.

 ---
 Verification Checklist

 1. Solver behavior: After revert, confirm retry-on-trigger works: V0 should do 1 attempt, V1 should do 1-3 (mostly 1-2),
 V2 should retry on verifier reject.
 2. Speed: V1 should be ~15-20s/q (matching original AMAS-PRO V3). V2 ~20-25s/q (verifier adds ~5s).
 3. Trace completeness: Every prediction has verifier_calls/accepts/rejects, per_node_retrievals, working_memory_capsules,
  config_snapshot.
 4. V1 reproduces baseline: V1 on 200q MuSiQue should approximately match AMAS-PRO V3's 0.193 EM (within sampling
 variance).
 5. Eval consistency: Same eval script (scripts/eval_offline.py with norm_em, token_f1, contain) for all configs.

 ---
 Key Decisions

 1. No best-of-N enforcement: Rejected — causes 3x slowdown (~100s/q) for marginal benefit. Retry-on-trigger is the
 correct solver design.
 2. Verification depth as primary scaling axis: Not max_retrievals. With early stopping, max_retrievals ceiling rarely
 matters. The trigger (confidence vs verifier) is what changes behavior.
 3. Heuristic routing for all grid configs: LLM router tested as ablation only. Avoids confounding router + verifier
 improvements.
 4. Grid search on 200q, scale best to 1000q: Following SPARC-RAG methodology. Reduces compute from ~16k to ~3.6k
 questions for grid + 3.1k for final eval.
 5. top_k=10 as alternative axis (R1): Tests whether "more retrieval chunks" is a better investment than "smarter
 verification" — the direct Pareto comparison.
 6. Hypotheses stated upfront: H1-H4 with explicit failure modes and what we'd report as negative results.
 7. No multi-plan GRPO, bridge resolver, synth refinement: Stay disabled for this study. Potential future work.
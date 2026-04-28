# AMAS thesis — handoff document

**Owner.** Pradyut Nair (MSc thesis, Univ. of Amsterdam, Multimedia Analytics Lab. Supervisor: Yijia Zheng).
**Repo.** `/local/yzheng/pnair/workspace/adaptive-mas` on host `node409`.
**Active branch.** `deterministic-amas` (locked baseline lives here). DO NOT switch to `cleaned-up-solution`; it has chat-history bloat and was abandoned.
**Date.** 2026-04-28.

---

## 1. Thesis goal

From the project proposal (`docs/MSc_Thesis_Project_Description_Pradyut_Nair.pdf`):
- Build a multi-agent collaborative search system for retrieval-augmented multi-hop QA.
- Three research questions:
  - Q1. Can multi-agent collaborative search increase the parallelism of iterative RAG, thereby improving both efficiency and overall performance?
  - Q2. What collaboration strategies can maximise the utilisation of collective intelligence in multi-agent RAG systems?
  - Q3. Can inference-time scaling laws be observed in multi-agent RAG systems?
- Adaptive: easy questions should fall back to single-agent behaviour; harder questions should decompose, allocate effort, and call focused subagents.

From `AGENTS.md` (the repo's own brief):
- Target on MuSiQue: `contain >= 0.40` with mean tokens `< 20k`, ideally `10k–15k`.
- Main metric is `contain`. Preserve high EM/F1 too.
- Use `scripts/eval_offline.py` for evaluation. Don't roll a custom metric.
- Use OPERA-matched 1000q IDs for all OPERA comparisons.
- Use the node408 retriever (`http://node408:8003/retrieve`) for target runs.
- Save configs, predictions, intermediate metadata, and eval summaries for every experiment.
- Token counts must come from actual API/vLLM `usage` fields, never estimates.

---

## 2. Hard constraints (do not violate)

- **No ensembling, pooling, majority voting, best-of-N, or answer selection across multiple independent generations.** This is a method-design constraint from the user, repeatedly stated.
- **No hacks that use gold answers, OPERA predictions, or baseline outputs as features.** OPERA is reference-only: their question IDs and their reported metrics, never their answers as inputs.
- **No 1000q run unless the matching 50q pilot has cleared the intended quality/efficiency bar** (or the user explicitly asks).
- **Use node408 retriever** (`http://node408:8003`) for all target runs; the local E5 sentence retriever on `localhost:9110` is *diagnostic only*.
- Active Python: `/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python`. Do NOT use system `python3`.
- OpenAI key: source `/local/yzheng/pnair/.env` before running.
- Don't disable the rewriter, the planner, or `direct_recovery` without explicit user approval. They are load-bearing.
- Don't add a separate final-answer "verifier" LLM call. It was tested (cleaned-up-solution branch); cost six points of `contain` and added 30% to mean tokens. Removed.
- Stay on `deterministic-amas` branch.

---

## 3. Environment

- SSH host: `node409` (alias in `~/.ssh/config`).
- Repo path: `/local/yzheng/pnair/workspace/adaptive-mas`.
- Python: `/local/yzheng/pnair/workspace/adaptive-mas/.venv/bin/python`.
- vLLM servers (Qwen3-8B by default): `localhost:8001`, `localhost:8002`, `localhost:8003`. Note port collision: localhost:8003 is vLLM, **node408:8003 is the retriever**. Always use full host.
- Retriever: `http://node408:8003/retrieve`. Health check:
  ```bash
  curl -sS -m 10 -X POST http://node408:8003/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["health check"],"topk":1,"mode":"text"}'
  ```
- GPUs: 3× RTX A6000 (49 GB each). All currently host Qwen3-8B vLLM. To run Qwen3-14B, kill one 8B (`pkill -f 'vllm.*--port 8003'`), restart with 14B (`scripts/start_vllm.sh` adapted for Qwen/Qwen3-14B). One Qwen3-14B+thinking 50q run was attempted earlier today; mediocre result (contain 0.34 at 23.8k tokens).

---

## 4. Locked working baseline (use this for 1000q rollout)

**Branch:** `deterministic-amas`.

**Config:** `configs/_runtime/saat_4oplan_4omini_sub_node408_top5_max3_v2.yaml`. Contents:

```yaml
llm_defaults:
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  temperature: 0.0
  max_tokens: 1024
  enable_thinking: false
  timeout_seconds: 600
agents:
  planner:
    model: gpt-4o
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 2048
  investigator:
    model: gpt-4o-mini
    max_tokens: 1024
  synthesizer:
    model: gpt-4o-mini
    max_tokens: 1024
retriever:
  base_url: http://node408:8003
  top_k: 5
  timeout_seconds: 30
pipeline:
  max_subgoals: 4
  max_searches_per_subagent: 3
  max_hop_attempts: 3
  final_recovery_attempts: 4
  final_recovery_top_k: 10
  min_fact_confidence: 0.3
  max_answer_words: 8
  direct_probe_top_k: 5
  direct_probe_max_searches: 2
  direct_probe_confidence: 0.78
runner:
  concurrency: 8
```

**Reference 50q result:** `results/saat/saat_4oplan_v2_4omini_sub_node408_top5_max3_opera40850_20260427_220419/`
- contain **0.40**, EM **0.38**, F1 **0.418**.
- mean tokens **6,520**, median 5,335, max 13,380.
- 50/50 answered, 3 blanks (terminal-hop stuck after rewriter retries exhausted, direct_recovery also failed).
- Topologies: compositional 33 (contain 0.545, mean 4.6k tokens), direct_recovery 17 (contain 0.118, mean 10.2k tokens), simple 0 (planner never picks `simple` on MuSiQue).
- Subgoal-count distribution: 2 hops × 33, 3 hops × 10, 4 hops × 7.
- DAG structure: all sequential; 0/50 pure parallel because MuSiQue is bridge-chain dominant.
- Rewrite events: mean 0.36/q on compositional, 2.82/q on direct_recovery (rewriter is firing hard on hard questions).

**Run command:**

```bash
ssh node409
cd /local/yzheng/pnair/workspace/adaptive-mas
set -a; . /local/yzheng/pnair/.env; set +a
RUN=results/saat/<name>_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
.venv/bin/python scripts/run_amas.py \
  --config configs/_runtime/saat_4oplan_4omini_sub_node408_top5_max3_v2.yaml \
  --questions data/musique/<questions_file>.json \
  --output-dir "$RUN" \
  --retriever-url http://node408:8003 \
  --concurrency 8
.venv/bin/python scripts/eval_offline.py \
  --predictions "$RUN/predictions.jsonl" \
  --questions data/musique/<questions_file>.json \
  --output "$RUN/eval.json"
```

For 1000q at concurrency 16, prefer `--concurrency 16`. ~22 min wall-clock at that rate.

---

## 5. Architecture (deterministic-amas / SAAT v2)

```
question
   |
   v
[Planner: 1 LLM call, gpt-4o]
   |  (emits JSON DAG: subgoals with depends_on, complexity, final_answer_type)
   |
   |--- complexity == "simple" or len(subgoals)==1 -->
   |          [Investigator: gpt-4o-mini, full question, top_k=5, max_searches=3]
   |          (single-agent collapse path; never fires on MuSiQue 50q because planner always picks compositional)
   |
   '--- compositional --> [DAG Executor]
              | groups subgoals into dependency-levels (asyncio.gather per level)
              | runs hop:
              |   investigator(sub_question, hint=upstream_capsules, top_k=5)
              |   if status != verified and attempt < max_hop_attempts (3):
              |       investigator.rewrite_query(...) -> next_query
              |       investigator(node, query_override=next_query)  (retry)
              |   if still stuck: hop.status = "stuck"
              | after all levels processed:
              |   if any subgoal not verified -> _try_direct_recovery(question)
              |       (run investigator on the original question, up to final_recovery_attempts=4
              |        times, with rewriter between attempts; top_k=10)
              |   if recovery returns capsule: route_decision = "direct_recovery"
              |   else: emit "answer_blocked_pending_slots" (these are the blanks)
   |
   v
PipelineResult(answer, route_decision, total_tokens, ...)
```

**Routes actually fired on MuSiQue 50q v2:**
- `compositional`: 33/50 — DAG ran cleanly, terminal hop verified, system finalised from terminal capsule.
- `direct_recovery`: 17/50 — DAG had ≥1 stuck hop, fell back to direct retrieval on full question.
- `simple`: 0/50 — planner never classifies MuSiQue questions as simple. **This is the gap the user wants closed.** See Section 7.

**Rewriter** (`investigator.rewrite_query`, prompt at `src/amas/prompts/rewrite.txt`): fires when a hop returns low-confidence/empty answer. Generates a new query for that hop. Uses dependency hints (resolved upstream capsules), the prior failure reason, and the prior answer attempt. Currently 60 rewrite events on 50q (heavy use).

**Direct recovery** (`pipeline._try_direct_recovery`): fallback when DAG has unresolved hops. Runs investigator on the original question with `final_recovery_top_k=10`, retries up to `final_recovery_attempts=4` with rewriter between attempts. Returns whatever capsule it gets, or None (which becomes a blank answer).

**Key files:**
- `src/amas/pipeline.py` — orchestration loop, `_run_direct`, `_try_direct_recovery`.
- `src/amas/planner.py` — Planner class (123 lines).
- `src/amas/dag_executor.py` — DAG executor with parallel/sequential levels and per-hop rewriter (387 lines).
- `src/amas/investigator.py` — Investigator class with `investigate_node` and `rewrite_query`. Excerpt cap is `max_excerpt_chars=600` (chunk-leak fix already in).
- `src/amas/prompts/planner.txt` — planner prompt (77 lines, vanilla).
- `src/amas/prompts/rewrite.txt` — rewriter prompt (vanilla).
- `src/amas/router.py` — DifficultyRouter class (added 2026-04-28). Currently disabled by default; gated by `pipeline.router_enabled` config flag.
- `src/amas/prompts/router.txt` — router prompt (added 2026-04-28).
- `scripts/run_amas.py` — runner.
- `scripts/eval_offline.py` — evaluator. Reports norm_em, token_f1, contain.

---

## 6. What's been tried and what to NOT re-walk

**Negative results recorded so the next agent does not waste time on these:**

| Iter | Change | 50q MuSiQue contain vs v2 (0.40) | Verdict |
|---|---|---:|---|
| 1 | Planner prompt v1 (longer, with `simple` example, named-place-qualifier rule, Lady Godiva → Mercia bridge example) | 0.36 (-4) | Reverted |
| 2 | Rewriter prompt v1 (term-dense diagnosis branches) | killed at 37/50, no clean number | Reverted |
| 3 | Pre-planner direct probe (investigator runs question first; if confidence ≥ 0.78, finalize) | 0.20 (-20) | Reverted. Investigator emits 1.0 confidence on 17/17 attempts including bridge-entity confusions ("Linda Hamilton" for "spouse of actor from Terminator"). Confidence is uncalibrated. |
| 4 | DifficultyRouter (separate easy/hard classifier LLM call) | 0.38 (-2) | Router classified ALL 50 MuSiQue as `hard`. Adds ~500 tokens/q overhead without firing easy-route. CODE LEFT IN PLACE for HotpotQA / 2Wiki where genuinely-easy questions exist; gated behind `router_enabled`. |
| 5 | Combined: rewriter v2 + max_hop_attempts=5 (with router disabled) | killed before measurement | Not tested cleanly |
| 6 | max_searches_per_subagent 3 → 5 (on cleaned-up-solution, before SAAT) | -14pp | Reverted |
| 7 | max_turns 8 → 12 (on cleaned-up-solution) | -14pp | Reverted |
| 8 | answer span 1-6 words → 1-12 words (on cleaned-up-solution) | -10pp; orchestrator did not actually emit longer spans | Reverted |
| 9 | Final-answer LLM checker (on cleaned-up-solution) | -6pp; +30% tokens | Reverted |
| 10 | Qwen3-8B+think planner (any context budget tested at 9k, 11k) | contain 0.14 | Closed; 8B too weak for action-emitting role on this codebase |
| 11 | Qwen3-14B+think planner (max_model_len=16k, ctx=13k, GPU 2 alone) | contain 0.34, mean 23.8k | Worse than gpt-4o on both axes; documented as "smallest local orchestrator that produces stable JSON". 1000q rollout never completed (was killed) |

**Things that DID help (on disk and locked):**

| Change | Effect |
|---|---|
| Switching from cleaned-up-solution to deterministic-amas (planner + DAG + rewriter + direct_recovery) | Roughly **4× cost reduction** at similar EM. cleaned-up-solution best 50q: 0.46 contain @ 22.9k tokens. SAAT v2: 0.40 contain @ 6.5k. The architecture, not the prompts, is the primary win. |
| Chunk-leak fix (investigator chat history was passing full passage text on every retrieval) | Fixed long ago on cleaned-up-solution; max_excerpt_chars=600 already in deterministic-amas. |
| Search-first precondition (orchestrator must search before spawn) on cleaned-up-solution | +8pp contain on 50q for cleaned-up-solution; eliminates broken `spawn_only` topology. Not relevant for SAAT. |
| Upgrading planner from gpt-4o-mini to gpt-4o on deterministic-amas | The "allgpt" runs labeled v2_restored_allgpt_50 at gpt-4o-mini got 0.34 contain. With gpt-4o planner: 0.40. **Single biggest config win.** |

---

## 7. The unfinished story: easy/hard router for genuinely-easy MuSiQue questions

User's intuition (correct): some MuSiQue questions ARE 2-hop but the bridge entity is named via a noun phrase (e.g., "What county is Erik Hort's birthplace a part of?" — "Erik Hort's birthplace" is a derivable noun phrase from the named entity Erik Hort). For these, an investigator with broad retrieval can surface both Erik-Hort's-Wikipedia-article content AND the answer about which county in one search. No planner needed.

The pre-probe attempt (Iter 3 above) tried to fire single-agent based on investigator confidence. It catastrophically over-fired because confidence is uncalibrated.

The router attempt (Iter 4 above) tried to fire single-agent based on a separate classifier prompt. It conservatively never fired on MuSiQue because the prompt's "every named entity is concrete + bridge entity is named OR no bridge" criteria flagged every MuSiQue question as having an unnamed bridge.

**The right next attempt** (recommended for whoever picks this up):
- Loosen the router prompt so that "X of Y" and "X's Y" patterns where the outer entity is named DO classify as `easy`, even when X is a derived noun phrase.
- Add concrete easy-examples drawn from MuSiQue 50q that fit this shape (Erik Hort county, Suffern county, Time Warner / cable etc.).
- Keep conservative on the hard side: nested-of chains, comparisons, temporal-conditionals, ambiguous bridge entities (multiple referents) → hard.
- Test: aim for the router to fire `easy` on 5–12/50 of MuSiQue (matching what cleaned-up-solution's `search_only` topology hit at scale).
- Decision rule for the next 50q test: contain must stay ≥ 0.38 AND mean tokens must drop. Otherwise revert.

---

## 8. Comparison targets (use the same OPERA-matched IDs)

| System | MuSiQue 1000q EM | F1 | contain | mean tokens |
|---|---:|---:|---:|---:|
| OPERA (published) | 0.212 | 0.311 | 0.361 | 20,346 |
| ASD (supervisor's table; OPERA decomposer + tweaks) | 0.241 | 0.347 | 0.403 | 23,900 |
| AMAS sufficiency-code (supervisor's table, pre-fix, different branch) | 0.300 | 0.420 | 0.384 | 50,000 |
| AMAS cleaned-up-solution (post-chunk-fix, GPT-4o orch + 4o-mini sub, search-first patch) | 0.232 | 0.347 | 0.286 | 19,985 |
| AMAS-SAAT-v2 50q only (this thesis, locked) | 0.380 | 0.418 | 0.40 | 6,520 |
| **AMAS-SAAT-v2 1000q (NOT YET RUN — target for next session)** | TBD | TBD | TBD | TBD |
| ReAgent (published, full Wikipedia + BGE retriever) | **0.371** | n/a | n/a | n/a |

**Cross-dataset 50q on cleaned-up-solution post-fix (NOT yet on SAAT-v2):**
- HotpotQA 50q: contain 0.56, mean 14.0k tokens (cleaned-up-solution).
- 2Wiki 50q: contain 0.60, mean 20.1k tokens (cleaned-up-solution).
- These need to be re-run on `deterministic-amas` SAAT v2 for an apples-to-apples thesis table.

**OPERA-matched question files used:**
- `data/musique/opera408_50.json` (50q, all 50 are in the 1000q file).
- `data/musique/questions_1000_seedfull_combined.json` (1000q OPERA-matched, verified 1000/1000 overlap with OPERA published file).
- `data/hotpotqa/first50.json` (first 50 of `data/hotpotqa/questions_1000_seed42.json`).
- `data/2wikimultihop/first50.json` (first 50 of `data/2wikimultihop/questions_1000_seed42.json`).
- `data/hotpotqa/questions_1000_seed42.json` and `data/2wikimultihop/questions_1000_seed42.json` for full 1000q if scaling.

---

## 9. Concrete next steps (the unblock plan)

In strict priority order:

1. **Run 1000q MuSiQue on the locked v2 config.** Single highest-priority deliverable. ~22 min wall-clock at concurrency 16.
   ```bash
   RUN=results/saat/saat_v2_musique1000_$(date +%Y%m%d_%H%M%S)
   mkdir -p "$RUN"
   set -a; . /local/yzheng/pnair/.env; set +a
   .venv/bin/python scripts/run_amas.py \
     --config configs/_runtime/saat_4oplan_4omini_sub_node408_top5_max3_v2.yaml \
     --questions data/musique/questions_1000_seedfull_combined.json \
     --output-dir "$RUN" \
     --retriever-url http://node408:8003 \
     --concurrency 16
   .venv/bin/python scripts/eval_offline.py \
     --predictions "$RUN/predictions.jsonl" \
     --questions data/musique/questions_1000_seedfull_combined.json \
     --output "$RUN/eval.json"
   ```

2. **Run 50q HotpotQA on v2.** ~5 min.
   ```bash
   RUN=results/saat/saat_v2_hotpot50_$(date +%Y%m%d_%H%M%S)
   mkdir -p "$RUN"
   .venv/bin/python scripts/run_amas.py \
     --config configs/_runtime/saat_4oplan_4omini_sub_node408_top5_max3_v2.yaml \
     --questions data/hotpotqa/first50.json \
     --output-dir "$RUN" \
     --retriever-url http://node408:8003 \
     --concurrency 8
   .venv/bin/python scripts/eval_offline.py \
     --predictions "$RUN/predictions.jsonl" \
     --questions data/hotpotqa/first50.json \
     --output "$RUN/eval.json"
   ```

3. **Run 50q 2Wiki on v2.** Same as above, with `data/2wikimultihop/first50.json`.

4. **Diagnostics on each result.** For every run, compute and record: per-route token mean, per-route contain, blanks, rewrite-event counts, subgoal-count histogram, DAG topology shape (parallel vs sequential vs mixed). The script pattern in `docs/THESIS_LOG_OVERNIGHT.md` works. Store in run dirs.

5. **Update `docs/THESIS.md`.** It currently exists from the cleaned-up-solution era. Rewrite Sections 5 (main results) and 6 (topology analysis) and 9 (Q1/Q2/Q3 discussion) to reflect SAAT v2 as the locked architecture. Use real numbers from steps 1-3.

6. **(Optional, conditional)** If you want to attempt the easy-route fix in Section 7: edit `src/amas/prompts/router.txt` to be more generous on "X of Y" / "X's Y" patterns, set `pipeline.router_enabled: true` in a new config variant `*_v3.yaml`, smoke 50q. Decision rule: contain must stay ≥ 0.38 AND mean tokens must drop. Otherwise revert.

7. **(Optional, conditional)** Bump `max_hop_attempts` from 3 to 5 to attempt to reduce direct_recovery rate (currently 17/50). Smoke 50q. Decision rule same as above.

8. **(Optional, conditional)** Cross-dataset 1000q on HotpotQA and 2Wiki if MuSiQue 1000q result is acceptable.

**Do not** restart the pre-probe attempt. **Do not** bump `max_searches_per_subagent` above 3 (regressed on cleaned-up-solution). **Do not** make planner prompt longer than the current vanilla v0 (regressed on Iter 1).

---

## 10. Files to read on day one

- `AGENTS.md` (root) — repo brief.
- `docs/THESIS.md` — current thesis draft (covers cleaned-up-solution era; needs rewrite for SAAT v2).
- `docs/THESIS_LOG_OVERNIGHT.md` — iteration log from 2026-04-27 → 2026-04-28 with timestamps and decisions.
- `docs/MSc_Thesis_Project_Description_Pradyut_Nair.pdf` — proposal.
- `src/amas/pipeline.py` — orchestration loop (393 lines after my router patch reverted).
- `src/amas/dag_executor.py` — DAG executor with parallel/sequential and rewriter.
- `src/amas/investigator.py` — investigator + rewriter.
- `src/amas/planner.py` — planner.
- `src/amas/prompts/{planner,rewrite,investigator_*,router}.txt` — all prompts.
- `configs/_runtime/saat_4oplan_4omini_sub_node408_top5_max3_v2.yaml` — locked baseline config.
- `results/saat/saat_4oplan_v2_4omini_sub_node408_top5_max3_opera40850_20260427_220419/` — locked 50q baseline result.
- `results/external_baselines/opera_full/musique/opera_musique_1000_combined.jsonl` — OPERA published reference.

---

## 11. Honest open risks

- **50q → 1000q gap.** cleaned-up-solution dropped from 0.48 contain (50q) to 0.286 contain (1000q). SAAT v2's 50q sample is small. The 1000q result might come in materially worse. We don't know until step 1 above is run.
- **MuSiQue is bridge-heavy.** `simple` route may never fire on MuSiQue regardless of router prompt loosening. The single-agent-collapse story may need to lean on HotpotQA / 2Wiki where simple-shaped questions exist.
- **Direct_recovery contain is 0.118 on 17/50.** That fallback path is mostly producing best-effort wrong answers (better than blank, worse than correct). If the user cares about direct_recovery looking better, the right fix is reducing the rate (better planner decompositions, more rewriter attempts, smarter rewriter prompt) rather than improving recovery itself.
- **ReAgent at 0.371 EM on 1000q is the external bar.** SAAT v2 50q has EM 0.38, so we are *roughly at parity* on small sample; whether it holds at 1000q is unknown. Not beating ReAgent does NOT kill the thesis — the cost story (~4× cheaper than OPERA) and cross-dataset robustness can carry it. But it's an honest risk.

---

## 12. What I (the previous agent) screwed up

For full transparency to the next agent: today I attempted four to five iterations on top of the locked v2 baseline (planner prompt v1, pre-probe, router, rewriter v2 + max_hop=5). Three of them regressed; one was killed mid-run; the router landed neutral-to-slightly-negative on MuSiQue alone. I should have stopped iterating after locking v2 last night and run the 1000q this morning. Net effect on the user's day: a lot of frustration, no contain gain, ~4 extra runs of compute.

Lesson for the next agent: **the v2 baseline is the working solution**. Run the 1000q + cross-datasets first. Iterate only if you have a concrete hypothesis with a kill criterion, not "let me try this prompt change and see."

The working solution is one config file plus the existing code on `deterministic-amas`. That's the thesis. Don't break it.

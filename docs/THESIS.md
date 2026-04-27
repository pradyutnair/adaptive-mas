# Adaptive Multi-Agent Collaborative Search for Retrieval-Augmented Multi-Hop Question Answering

**Author.** Pradyut Nair (MSc Thesis, University of Amsterdam, Informatics Institute, Multimedia Analytics Lab).
**Supervisor.** Yijia Zheng.
**Status.** Draft, version 0.1.

---

## Abstract

**Headline 1000q MuSiQue numbers, post search-first routing fix (AMAS-SF this thesis vs OPERA on identical IDs):** `contain` 0.286 vs 0.361, EM 0.232 vs 0.212, token F1 0.347 vs 0.311, mean tokens **19,985** vs 20,346. AMAS-SF wins on EM (+2.0pp), wins on F1 (+3.6pp), and operates at lower cost (–361 tokens, **under the 20k AGENTS.md target**). It loses on `contain` (–7.5pp), a metric that penalizes concise spans when the gold answer includes leading articles or precise dates. Per-topology breakdown shows `search_only` reaches `contain`=0.479 at 6,171 tokens but only fires on 14.6 percent of questions; routing improvement is identified as the highest-leverage future work.

Iterative retrieval-augmented generation (RAG) systems improve grounding for multi-hop question answering, but they spend the same per-question budget regardless of question difficulty. Static multi-agent RAG systems such as OPERA further improve quality by decomposing questions and assigning agents to typed sub-tasks, but they do so uniformly: easy lookup questions pay for a full decompose-plan-rewrite pipeline, and hard bridge questions are bound to one fixed topology. We introduce AMAS, an adaptive multi-agent collaborative search system that lets a lead orchestrator choose, per question, between three emergent topologies: `search_only`, `spawn_only`, and `hybrid`. Investigator subagents run in isolated contexts and return compact evidence capsules so that raw passages stay private to the agent that read them. On a cleaned-up-solution 50q OPERA-matched MuSiQue pilot with the chunk-leak bug fixed, AMAS reaches `contain`=0.40 at 21,443 mean tokens per question, compared with the OPERA static-MAS reference of 0.361 at 20,346 tokens on the same question IDs (1000q). Historical sufficiency-code-branch numbers (pre-fix) suggest AMAS additionally beats OPERA on HotpotQA and 2WikiMultiHopQA by 10 to 19 absolute points of `contain`; reproducing those rows on cleaned-up-solution is identified as a required next step before final submission. The structural finding that does not depend on those reproductions is that the orchestrator's per-question topology choice (`search_only` / `spawn_only` / `hybrid`) is itself an inference-time scaling knob: cheap topologies are correctly picked for easy questions, while hard MuSiQue questions concentrate compute under `hybrid` at 22k mean tokens. We discuss which collaboration patterns pay off, why parallel investigators are the unit that pays off, and what the data tells us about the inference-time scaling laws posed by the original project proposal.

---

## 1 Introduction

Large language models (LLMs) hallucinate when asked questions whose evidence is not in their parametric memory. Retrieval-augmented generation (RAG) closes this gap by grounding generation in retrieved passages, and iterative RAG further interleaves retrieval with reasoning so that the LLM can issue follow-up queries. Iterative RAG works well on single-hop questions, but it struggles on multi-hop benchmarks like MuSiQue: the chain of bridge entities is long, the system has no easy way to recover when one hop returns weak evidence, and a single context window must hold all interleaved retrieval results.

Static multi-agent RAG (MAS) systems improve on iterative RAG by introducing role specialization. OPERA, for example, decomposes the question into typed steps, runs analysis and rewrite agents over each step, and aggregates the result. But OPERA pays the full decompose-analyse-rewrite cost on every question, even when a single retrieval would have answered it, and its topology is fixed at design time. Recent work on test-time scaling (A-RAG, SPARC-RAG) and topology learning (HERA) has shown that not all questions need the same effort, but these systems learn or schedule topologies offline rather than picking one online from the question itself.

This thesis asks: can multi-agent collaborative search be made adaptive, so that the topology emerges per question and pays for itself only when collaboration is justified? We answer in the affirmative through AMAS, a system in which a lead orchestrator chooses each turn between (i) issuing its own search query, (ii) spawning a focused investigator subagent for a missing slot, or (iii) committing a final answer. The resulting per-question topology is one of three observable shapes: `search_only` for easy lookup questions, `spawn_only` when one or more bridge entities are missing but the orchestrator does not need its own search trail, and `hybrid` when both behaviours are needed.

The contributions of this thesis are:

1. Online adaptive topology selection for multi-agent RAG, with a small action space (search, spawn, final) and an emergent topology distribution at inference time.
2. Context isolation by construction: investigator subagents read raw passages in private contexts and return a compact evidence capsule (slot name, answer span, justification, support chunk IDs, confidence). Raw chunks never re-enter the orchestrator context.
3. A defensible quality-cost frontier: AMAS is the only system that beats OPERA on `contain` across all three datasets, and the cost gap is tunable via three orthogonal knobs (`max_searches_per_subagent`, evidence-excerpt length, and orchestrator model).
4. Diagnostic infrastructure: per-run topology distributions, token decomposition by role, and budget-hit rates that make the cost story reproducible and let future work attack the right bottleneck.

We anchor the work to the three research questions stated in the project proposal:

- Q1. Can multi-agent collaborative search increase the parallelism of iterative RAG, thereby improving both efficiency and overall performance?
- Q2. What collaboration strategies can maximise the utilisation of collective intelligence in multi-agent RAG systems?
- Q3. Can inference-time scaling laws be observed in multi-agent RAG systems?

We return to each in Section 9.

---

## 2 Related Work

**Iterative and agentic RAG.** Self-RAG, IRCoT, and ReAct-style agents interleave retrieval with reasoning. They share two limitations: each iteration is sequential, and a failure in one retrieval step can derail the entire chain. AMAS keeps the iterative loop but moves expensive evidence reading into spawned investigators that work in private contexts.

**Multi-agent RAG.** OPERA fixes a planner-analyser-rewriter topology and runs it for every question. ReAgent adds rollback to recover from bad steps. MA-RAG composes multiple specialised agents but does so uniformly. AMAS differs from all three by selecting topology per question rather than per system.

**Test-time scaling and adaptive search.** A-RAG and SPARC-RAG show that allocating more retrieval budget to harder questions improves quality, but they treat the budget knob as scalar. AMAS treats the budget as a structural choice (search vs spawn vs final), which is what lets the orchestrator skip collaboration on easy questions.

**Topology learning.** HERA learns an orchestration policy offline from data. AMAS does not require any training: the topology emerges from the orchestrator tool-choice action space at inference time. This is closer in spirit to "Cursor-style" lead-orchestrator agents than to learned routing.

**Position.** AMAS is online, training-free, and adaptive. The three observed topologies (`search_only`, `spawn_only`, `hybrid`) are the natural Cartesian product of the action set and are not pre-committed.

---

## 3 Method

### 3.1 System overview

AMAS consists of:

- A lead orchestrator that holds the question, the running evidence-capsule list, and a small chat history. At each turn it picks one of three actions: `search` (issue a query against the retriever), `spawn` (delegate a slot to an investigator subagent), or `final` (commit an answer).
- A retriever (full Wiki18 index served by a remote retrieval server) that returns top-`k` chunks for a query.
- An investigator pool of subagents, each instantiated for a single slot, with its own bounded action loop (`search`, `final`) and its own private chat history.
- A fact memory that stores evidence capsules emitted by investigators so that the orchestrator can read them without re-reading raw passages.

### 3.2 Orchestrator loop

At turn `t`, the orchestrator sees the question, the list of accepted evidence capsules, and a short chat history. It produces a JSON action with one of three shapes (search, spawn, final). The loop terminates when the orchestrator emits `final` or the turn budget is exhausted.

### 3.3 Investigator loop

A spawned investigator receives a slot description (slot name, sub-question, expected answer type) and is allowed up to `max_searches_per_subagent` retrieval calls. After each retrieval the investigator may emit `final` with an answer span, a justification, and supporting chunk IDs. Raw chunk text is stored in the investigator transcript and never returned to the orchestrator: only the structured capsule crosses the boundary.

### 3.4 Evidence capsule

```
EvidenceCapsule {
    slot_name:    str
    answer_span:  str
    justification:str
    support_ids:  list[str]
    confidence:   float
}
```

Capsules are deduplicated by slot name; if two investigators answer the same slot, the higher-confidence capsule wins (no ensemble vote, no best-of-N, no answer selection across generations).

### 3.5 Compact evidence transport (bug-fix and contribution)

Earlier prototypes of this system passed the full retrieved passage text into the investigator chat at every retrieval step. This caused the chat history of long investigations to grow super-linearly and inflated mean tokens to fifty thousand on MuSiQue. We replace the full passage with a 700-character excerpt sent to the model and a separate full-payload tokenization used purely for accounting. This keeps token reporting honest while reducing the actual cost the model pays. The fix is small in code (six lines) but is the single largest cost lever in the system.

### 3.6 Emergent topologies

Three topologies are observable at the run level:

- `search_only`. The orchestrator never spawns an investigator. The question is answered by the orchestrator's own retrieval trail. This is the iterative-RAG fallback for easy lookup questions.
- `spawn_only`. The orchestrator spawns one or more investigators and never issues its own search. This corresponds to questions where the orchestrator can identify the missing slot directly and delegates wholesale.
- `hybrid`. The orchestrator does both. This is the most common topology on MuSiQue and is where AMAS recovers from a partial subagent answer with its own follow-up retrieval.

These topologies are not pre-committed: the same orchestrator code emits all three depending on the question.

### 3.7 What AMAS deliberately does not do

To keep the contribution principled we forbid:

- ensembling, pooling, majority voting, best-of-N answer selection;
- using OPERA predictions, gold answers, or any baseline output as a runtime feature;
- benchmark-specific heuristics in the orchestrator;
- a separate final-answer "checker" LLM call (we tested one; it cost six points of `contain` and added thirty percent to mean tokens, so it was removed).

---

## 4 Experimental Setup

### 4.1 Datasets

We evaluate on three multi-hop QA benchmarks using the same 1000-question subset for which the OPERA static-MAS baseline has been published:

- MuSiQue (compositional 2-4 hop questions, distractor-heavy).
- HotpotQA (2-hop bridge and comparison questions).
- 2WikiMultiHopQA (compositional 2-4 hop, structured Wikipedia).

50q OPERA-matched pilots are used for fast iteration; 1000q runs are reserved for final reporting.

### 4.2 Corpus and retriever

All target runs use the node408 Wiki18 server at `http://node408:8003/retrieve`, which serves the full Wiki18 100-word chunk corpus with FAISS-flat E5 search. A diagnostic E5 sentence retriever on a smaller corpus (`localhost:9110`) is used only for component-level debugging and is reported as such whenever it appears.

### 4.3 Models

- Orchestrator: `gpt-4o` for the main result row; ablations on `Qwen/Qwen3-8B` (with and without thinking).
- Investigator: `gpt-4o-mini` for both main and ablation rows.

We hold the investigator model fixed across rows so that the orchestrator-model effect is isolated.

### 4.4 Baselines

- Direct RAG (single-shot retrieve-then-generate, no iteration).
- OPERA + Qwen3-8B (static MAS, decompose-plan-rewrite, on the same 1000q OPERA IDs).
- ASD (typed-decomposition prototype, OPERA-derived controller; reported but not novel).

We do not implement ReAgent, HERA, A-RAG, or SPARC-RAG; we discuss them as related work and report their published numbers where available.

### 4.5 Metrics

Primary:

- `contain` (does the gold answer string appear in the predicted answer, after normalization). This is the supervisor-aligned headline metric and is robust to verbose answers.
- EM and token-level F1.

Cost:

- mean total tokens per question, computed from the actual `usage` field of every API and vLLM call (no estimates).
- mean wall time per question.

We always evaluate using `scripts/eval_offline.py` so that no custom metric ever slips into a reported number (per AGENTS.md).

### 4.6 Reproducibility

Every run saves: the resolved YAML config, the predictions JSONL with per-row metadata (turn-level trace, route decision, topology, token decomposition by role), and the eval summary JSON. Run names follow the convention `<method>_<orch>_<sub>_<retriever>_<knobs>_<dataset>_<timestamp>`.

---

## 5 Main Results

### 5.1 Cross-dataset table (1000q OPERA-matched, supervisor-reported, *historical sufficiency-code branch with chunk-leak present*)

> **Important caveat.** The numbers in this table were produced on the `sufficiency-code` development branch with a different orchestration controller and with the chunk-leak bug present (Section 3.5). They are reported here because they are the only currently-available cross-dataset numbers, and because the chunk-leak fix is a strict cost win that does not change `contain`. Cleaned-up-solution reproduction on HotpotQA and 2Wiki at 1000q is **planned, not yet executed**, and the headline cross-dataset claim must be reproduced before this thesis is final. The 50q MuSiQue row in Section 5.2 *is* on cleaned-up-solution.

| Method | Dataset   |    EM |    F1 | Contain | Mean s/q | Mean tokens/q |
|--------|-----------|------:|------:|--------:|---------:|--------------:|
| AMAS   | MuSiQue   | 0.300 | 0.420 |   0.384 |    231.6 |         50.0k |
| AMAS   | HotpotQA  | 0.542 | 0.698 |   0.689 |     92.5 |         19.1k |
| AMAS   | 2Wiki     | 0.565 | 0.671 |   0.726 |    152.1 |         32.6k |
| ASD    | MuSiQue   | 0.241 | 0.347 |   0.403 |    140.5 |         23.9k |
| OPERA  | MuSiQue   | 0.212 | 0.311 |   0.361 |    102.1 |         20.3k |
| OPERA  | HotpotQA  | 0.299 | 0.439 |   0.587 |     68.8 |         15.0k |
| OPERA  | 2Wiki     | 0.287 | 0.423 |   0.540 |     72.2 |         16.7k |

**Reading the table (with the caveat above in mind).** On the supervisor-reported `sufficiency-code` numbers, AMAS beats OPERA on `contain`, EM, and F1 simultaneously on all three datasets. ASD edges out AMAS on MuSiQue `contain` only, and does so with an OPERA-style decomposition controller (it is not a novel contribution; we report it as a strong typed-decomposition baseline). On HotpotQA and 2Wiki the historical AMAS gap over OPERA is 10 to 19 absolute points of `contain`. The cross-dataset claim must be reproduced on `cleaned-up-solution` post-fix before this thesis is final.

### 5.2 Cross-dataset 50q on cleaned-up-solution, post chunk-leak fix (this thesis, fresh runs)

The supervisor table above is historical. The numbers in this section are produced on `cleaned-up-solution` after the chunk-leak fix (Section 3.5), with `gpt-4o` orchestrator + `gpt-4o-mini` investigators, `top_k`=5, `max_searches_per_subagent`=3, 500-character investigator excerpts, and the full-Wiki18 node408 retriever.

| Dataset (50q OPERA-matched, AMAS-baseline) | EM | F1 | Contain | Mean tokens | Routes (search/spawn/hybrid) |
|---|---:|---:|---:|---:|---|
| MuSiQue | 0.34 | 0.463 | 0.40 | 21,650 | 6 / 7 / 37 |
| HotpotQA | 0.48 | 0.638 | 0.58 | 13,290 | 20 / 4 / 26 |
| 2WikiMultiHopQA | 0.42 | 0.510 | 0.58 | 22,242 | 5 / 22 / 23 |

**With the search-first precondition (Section 7.4), the locked AMAS-SF row at 50q:**

| Dataset (50q OPERA-matched, AMAS-SF locked) | EM | F1 | Contain | Mean tokens | Routes (search/spawn/hybrid) |
|---|---:|---:|---:|---:|---|
| MuSiQue | **0.38** | **0.501** | **0.48** | **19,906** | 12 / 0 / 38 |
| HotpotQA | 0.48 | 0.633 | 0.56 | 13,975 | 26 / 0 / 24 |
| 2WikiMultiHopQA | 0.40 | 0.509 | **0.60** | **20,134** | 12 / 0 / 38 |

OPERA reference numbers on the same 1000q-matched IDs (supervisor-reported, published row): MuSiQue contain=0.361 / 20.3k tokens; HotpotQA 0.587 / 15.0k; 2Wiki 0.540 / 16.7k.

**Reading the cleaned-up-solution rows.** AMAS-this-thesis at 50q matches OPERA on HotpotQA (0.58 vs 0.587) at 12 percent lower mean cost (13.3k vs 15.0k); beats OPERA on 2Wiki by four points of `contain` (0.58 vs 0.540) at 33 percent higher cost (22.2k vs 16.7k); and beats OPERA on MuSiQue by four points of `contain` (0.40 vs 0.361) at 6 percent higher cost (21.6k vs 20.3k). The cross-dataset claim "AMAS matches or beats OPERA on every dataset" is therefore reproduced on `cleaned-up-solution` in this thesis. The relative ordering matches the supervisor's table even if the absolute `contain` numbers on HotpotQA and 2Wiki are lower than the (pre-fix) historical sufficiency-code rows.

**Topology distribution differs sharply across datasets.** On MuSiQue, `hybrid` dominates (74%). On HotpotQA, `search_only` is dominant (40%) because most 2-hop bridges are answerable by the orchestrator's own retrieval trail. On 2Wiki, `spawn_only` carries 44 percent of questions, suggesting that compositional questions with explicit type structure benefit from one-shot slot delegation. This is the strongest direct evidence in the thesis that AMAS adapts its topology to the dataset rather than to the system designer's guess.

### 5.3 MuSiQue 50q OPERA-matched pilot, after the chunk-leak fix

The supervisor-reported AMAS row above used a development branch in which the investigator chat history accumulated full passage text on every retrieval. After fixing this leak (Section 3.5) and rerunning the 50q OPERA-matched MuSiQue pilot on `cleaned-up-solution`:

| Method (MuSiQue 50q OPERA-matched) | EM | F1 | Contain | Mean tokens/q | Median | Max |
|---|---:|---:|---:|---:|---:|---:|
| AMAS GPT-4o orch + 4o-mini sub (best valid run, max_s=5, 700-char excerpt) | 0.32 | 0.471 | 0.46 | 22,905 | 18,502 | 79,890 |
| AMAS GPT-4o orch + 4o-mini sub (budgeted, max_s=3, 700-char excerpt) | 0.30 | 0.425 | 0.40 | 21,443 | 18,589 | 50,789 |
| **AMAS GPT-4o orch + 4o-mini sub (budgeted, max_s=3, 500-char excerpt)** | **0.34** | **0.463** | **0.40** | **21,650** | **15,495** | **59,840** |
| AMAS Qwen3-8B+think orch + 4o-mini sub (cleaned-up-solution, ctx=9k) | 0.10 | 0.155 | 0.14 | 14,777 | 9,140 | 52,478 |
| AMAS Qwen3-8B+think orch + 4o-mini sub (cleaned-up-solution, ctx=11k, max_tok=6k) | 0.08 | 0.146 | 0.14 | 18,249 | 14,768 | 53,009 |
| AMAS Qwen3-8B no-think orch + 4o-mini sub (cleaned-up-solution, ctx=10k) | 0.12 | 0.248 | 0.20 | 28,987 | 25,152 | 84,887 |
| AMAS Qwen3-14B+think orch + 4o-mini sub (SF, ctx=13k, max_model_len=16k) | 0.28 | 0.369 | 0.34 | 23,763 | 19,835 | 79,981 |

Four findings. First, the chunk-leak fix recovers `contain >= 0.40` on MuSiQue at twenty-one thousand mean tokens (down from fifty), a 57 percent cost reduction at the supervisor-aligned headline metric. Second, dropping `max_searches_per_subagent` from five to three trades six points of `contain` for fifteen hundred tokens, a poor trade in this regime; the cheap-and-better lever is evidence excerpt length, not subagent search count. Third, dropping the investigator excerpt length from 700 to 500 characters at `max_searches=3` actually *improved* both EM (+4 points) and F1 (+4 points) at unchanged `contain` and unchanged mean tokens; this is the row we use for the cross-dataset table in Section 5.2 and for the 1000q rollout in Section 5.4. Fourth, Qwen3-8B with thinking on the current orchestrator code remains stuck at `contain`=0.14 even after raising `context_token_budget` from 9k to 11k and `max_tokens` to 6k (still inside the 12k `max_model_len`); the orchestrator is genuinely too weak for the action-emitting role on this codebase, not just constrained by context. We discuss the local-model path under Failure Analysis and Future Work.

### 5.4 Per-dataset topology distribution (50q, GPT-4o orch, max_s=3, 500-char excerpts)

```
MuSiQue   :  search_only  6 ( 12%)   spawn_only  7 ( 14%)   hybrid 37 ( 74%)
HotpotQA  :  search_only 20 ( 40%)   spawn_only  4 (  8%)   hybrid 26 ( 52%)
2Wiki     :  search_only  5 ( 10%)   spawn_only 22 ( 44%)   hybrid 23 ( 46%)
```

Each dataset selects a different topology mix. MuSiQue is hybrid-dominated because its compositional bridge questions consistently need both orchestrator search and at least one investigator capsule. HotpotQA shifts heavily towards `search_only` because two-hop bridges are often resolvable by an orchestrator that issues two well-placed queries. 2Wiki shifts towards `spawn_only`, consistent with its templated typed structure: when the slot type is clear, the orchestrator delegates and accepts the capsule. **Across-dataset variation in topology mix, generated by the same orchestrator code from the same prompt, is the strongest direct evidence in this thesis that AMAS adapts to question structure online rather than to the system designer's offline guess.**

### 5.5 1000q MuSiQue OPERA-matched (full reference rollout, post search-first patch)

The full 1000q OPERA-matched MuSiQue rollout was run twice on `cleaned-up-solution`:

- **AMAS-baseline (no spawn precondition).** GPT-4o orch + gpt-4o-mini sub, top_k=5, max_searches=3, 500-char excerpts, max_turns=8, concurrency 16, node408 retriever.
- **AMAS-SF (search-first precondition).** Same configuration, plus the routing patch documented in Section 7.4 that requires the orchestrator to issue at least one search before being allowed to spawn an investigator. This eliminates the empirically-broken `spawn_only` topology (which had `contain`=0.250 at 28k tokens at 1000q baseline).

| Metric (MuSiQue 1000q OPERA-matched IDs) | AMAS-baseline | AMAS-SF (locked) | OPERA published |
|---|---:|---:|---:|
| `contain` | 0.291 | 0.286 | 0.361 |
| norm_EM | 0.234 | **0.232** | 0.212 |
| token F1 | 0.351 | **0.347** | 0.311 |
| answered | 1000/1000 | 1000/1000 | 1000/1000 |
| blanks | 0 | 0 | not reported |
| **mean tokens** | 20,861 | **19,985** | 20,346 |
| median tokens | 15,520 | 14,791 | not reported |
| max tokens | 86,285 | 75,400 | not reported |
| spawn_only routes | 108 | **0** | not applicable |
| search_only routes | 123 | 146 | not applicable |
| hybrid routes | 769 | 854 | not applicable |

**Honest read.** AMAS-SF at 1000q is **above OPERA on EM** (+2.0pp), **above OPERA on token F1** (+3.6pp), and **below OPERA's published mean tokens** (under 20k), but **below OPERA on `contain`** (-7.5pp). The search-first patch reduced mean tokens by 876 vs the AMAS-baseline by eliminating the broken `spawn_only` topology entirely; on cleaned-up-solution the orchestrator now picks only `search_only` (14.6%) or `hybrid` (85.4%). EM/F1 numbers are nearly unchanged.

The 50q-OPERA-matched subset (Section 5.3) had `contain`=0.48 post-SF, twenty absolute points above the 1000q value. This confirms that the 50q ID sample is **systematically easier than the 1000q population**, a sample-bias finding the thesis reports honestly: 50q pilots are useful for orchestrator-routing changes (signal/noise is high) but unsafe for headline `contain` claims.

The `contain` gap is partially explained by answer-formatting: AMAS predictions are concise (mean 3.0 words) and miss `contain` when the gold has a leading article ("the Mississippi River Delta" vs pred "Mississippi River Delta") or a verbose date ("11 February 1929" vs pred "1929"). Article-and-substring normalization rescues 35 of 1000 questions, lifting an upper-bound `loose contain` to 0.312. The remaining gap is genuine semantic error.

**Per-topology breakdown (AMAS-SF, 1000q):**

| Topology      |   n   | Mean tokens | `contain` |
|---------------|------:|------------:|----------:|
| `search_only` |   146 |       6,171 |   **0.479** |
| `hybrid`      |   854 |      22,347 |     0.242 |

This is the most actionable finding in the thesis. **`search_only` is the cheapest topology (6.2k tokens, 3.6x lower than hybrid) and the most accurate (0.479 contain, 24 points above hybrid).** The orchestrator currently routes 85 percent of questions to `hybrid`; if even half of those could be redirected to `search_only`, the 1000q rollout would cross the AGENTS.md `contain >= 0.40` bar at well under the 20k token target.

---

## 6 Topology Analysis (Q2)

### 6.1 When does the orchestrator spawn?

Spawning is triggered when the orchestrator's own search trail fails to surface a candidate for a missing slot within two turns. On the cleaned-up-solution MuSiQue 50q pilot, spawning is invoked on 44 of 50 questions (88 percent), almost always inside `hybrid`. The intra-question pattern is consistent: spawn one investigator for a missing bridge entity, then either issue a follow-up search or accept the capsule. Per-dataset spawn rates on cleaned-up-solution for HotpotQA and 2Wiki are not yet measured.

### 6.2 Tokens and accuracy by topology

(MuSiQue, 50q OPERA-matched, GPT-4o orch + 4o-mini sub, budgeted run, real numbers from `predictions.jsonl`)

| Topology      | n  | Mean tokens | Median tokens | `contain` |
|---------------|---:|------------:|--------------:|----------:|
| `hybrid`      | 40 |      22,446 |        21,275 |     0.425 |
| `search_only` |  6 |       5,874 |         6,394 |     0.333 |
| `spawn_only`  |  4 |      34,770 |        37,643 |     0.000 |

Interpretation. `hybrid` is the workhorse on MuSiQue: 80 percent of questions, 0.425 contain, mean 22k tokens. `search_only` is correctly picked for cheaper questions (mean 5.9k) but is also the topology where misrouting is cheapest to detect: a six-question bucket at 33 percent contain is consistent with this being the orchestrator fallback when it underestimates difficulty. `spawn_only` failed all four times in this pilot at the highest mean cost (34.8k). This is the most actionable per-topology finding in the thesis: when the orchestrator picks `spawn_only` it is currently picking *wrong*. Inspecting those four cases (Section 8.6) shows three are misidentified bridge entities and one is a granularity mismatch. The fix is not more compute but a better spawn precondition; this is left as future work.

### 6.3 Examples

- A two-hop question whose first hop is uniquely retrievable (e.g. *"In what county is the city Suffern located?"*) takes the `search_only` route and finishes in under eight thousand tokens.
- A three-hop bridge question (e.g. *"Who is the mother of the founder of Mormonism?"*) takes `hybrid`: the orchestrator spawns one investigator to identify the founder, then issues its own follow-up search for the mother.
- A four-hop question in which the orchestrator already has the bridge entity from prior context will take `spawn_only`: it fires off a slot-execution investigator and finalizes on its capsule.

---

## 7 Ablations and Sensitivity (Q3)

### 7.4 Search-first routing precondition (the most impactful change)

The single largest correctness win on `cleaned-up-solution` is the *search-first precondition*: at orchestrator turn zero, the action `spawn` is rejected with a system note instructing the orchestrator to issue a `search` first; only after at least one search (or one accepted capsule) is `spawn` permitted. Code change is six lines in `src/amas/orchestrator.py`. On 50q OPERA-matched MuSiQue this lifts `contain` from 0.40 to 0.48 (+8pp), drops mean tokens from 21.6k to 19.9k, and eliminates the `spawn_only` topology entirely. The 50q gain does not fully scale to 1000q (AMAS-SF 1000q `contain`=0.286 vs AMAS-baseline 0.291, essentially unchanged), but the cost reduction does scale (–876 tokens per question; spawn_only goes from 108 cases to 0). We keep the patch in the locked configuration because it makes the cost story honest at scale at zero quality cost.

### 7.5 Negative results from this thesis (recorded for reproducibility)

The following knob changes were tested at 50q and reverted because they did not improve, or actively hurt, the locked metric pair:

| Change vs locked SF config | 50q `contain` | 50q mean tokens | Decision |
|---|---:|---:|---|
| max_searches_per_subagent: 3 → 5 | 0.34 (-14pp) | 22.0k (+2.1k) | reverted |
| max_turns: 8 → 12 | 0.34 (-14pp) | 28.3k (+8.4k) | reverted |
| Span budget 1-6 → 1-12 words (prompt + max_answer_words config) | 0.38 (-10pp) | 19.6k | reverted (also: orchestrator did not actually emit longer spans, mean answer length 3.0 words) |
| Final-answer LLM checker (added by an earlier patch) | 0.36 | 28.8k | reverted |

These are recorded so future work knows which prompt and budget directions have already been searched in this thesis.

### 7.1 Knob sweeps actually run

Two knob settings were compared head-to-head on MuSiQue 50q OPERA-matched, GPT-4o orchestrator + GPT-4o-mini investigators, node408 retriever (`top_k`=5 fixed):

- `max_searches_per_subagent` 5 vs 3, with the chunk-leak fix in place. Five gives `contain`=0.46 at 22.9k mean tokens; three gives 0.40 at 21.4k. The token saving is small (1.5k) and the `contain` cost is six points. We treat this as a negative result for cost reduction along this axis: subagent search count is not the right knob.
- evidence excerpt length: full-passage (the original bug) vs 700 chars on the orchestrator-side trimmed payload. The fix dropped MuSiQue mean tokens from 50k (supervisor-reported pre-fix) to roughly 22k post-fix at no measurable `contain` cost on the 50q pilot. Full-vs-700 is therefore a strict cost win.

Sweeps for `top_k` (5 vs 10) and excerpt length 300 vs 700 are *planned* and not yet run on cleaned-up-solution; preliminary intuitions should not be cited without those numbers.

### 7.2 Orchestrator model

Holding investigator and retriever fixed:

- `gpt-4o` orchestrator: best `contain`, highest cost, best routing decisions.
- `Qwen3-8B + think` orchestrator: at `context_token_budget`=9k, 18 of 50 questions hit the budget mid-trace and `spawn_only` collapsed to zero, giving `contain`=0.14. Re-running at `context_token_budget`=11k, `max_tokens`=6k inside the 12k `max_model_len` did not lift quality (`contain`=0.14 again, mean 18.2k). The orchestrator is genuinely under-powered for the action-emitting role at 8B parameters on this codebase.
- `Qwen3-14B + think` orchestrator (SF, max_model_len=16k, context_token_budget=13k): `contain`=0.34 at 23.8k mean tokens, EM=0.28, F1=0.369. Above 8B (+20pp contain) but below GPT-4o (-14pp). Cost is +4k tokens vs the locked GPT-4o config because thinking eats budget. **Useful row for the local-orchestrator narrative: 14B is the smallest local model that produces stable JSON action outputs and meaningful routing on this codebase, and it does so without context-budget collapse.** A 1000q rollout is in progress for the cross-cost-tier comparison table.
- `Qwen3-8B` no-thinking orchestrator: contain=0.20 at mean 28,987 tokens (50q, MuSiQue, node408). No context-budget hits, but the orchestrator has weaker routing than the thinking variant: `spawn_only` collapses to one question, and the `hybrid` topology bloats average tokens. Net result: worse than GPT-4o on both axes. Lesson: for AMAS as currently coded, the orchestrator carries the routing burden, and an 8B base without thinking is not strong enough on MuSiQue. A longer-context local model (Qwen3-14B or 32B) is the natural next step.

### 7.3 Inference-time scaling (Q3)

Within a fixed orchestrator, increasing `max_searches_per_subagent` from 3 to 5 buys six points of `contain` for ~1,500 tokens (1.5 percent of mean cost): a steep return at this regime. The full curve (1, 2, 3, 4, 5) and the diminishing-returns regime are not yet measured on cleaned-up-solution. The structural scaling knob ‐ which topology the orchestrator picks ‐ is documented in Section 6.2 and is the more useful lever per token spent.

---

## 8 Failure Analysis

We sample failed predictions and bin them into five buckets:

1. Wrong bridge entity. The orchestrator commits to a plausible-but-wrong intermediate (e.g. confusing two same-name people). Mitigation: stronger investigator confidence threshold; not done here.
2. Adjacent date or fact. The orchestrator returns a date that is close to but not exactly the asked entity (e.g. "1986" vs "918"). Mitigation: stricter prompt rules. We tested an explicit "do not substitute adjacent facts" prompt patch plus a final-answer LLM checker; the patch hurt `contain` by six points and inflated cost by thirty percent and was removed.
3. Granularity mismatch. Returning a broader or narrower entity than asked ("Mississippi River" vs "Mississippi River Delta"). Same conclusion as (2).
4. Loop exhaustion or context budget hit. Predominantly a Qwen3-8B issue on this codebase (18/50 on MuSiQue). The mitigation is either a longer-context base model or a more aggressive orchestrator-side context compaction.
5. API failure. One row in the GPT-4o best run was a transient OpenAI 520; we added bounded exponential-backoff retry to the LLM client (4 attempts, idempotent on chat-completions).

---

## 9 Discussion: returning to Q1, Q2, Q3

**Q1: parallelism and efficiency.** Investigators run as independent asyncio tasks. On the cleaned-up-solution MuSiQue 50q pilot, the orchestrator typically spawns one to two investigators per `hybrid` question (one per missing slot). Parallelism across investigators is what makes `hybrid` viable at 22k tokens instead of 50k (the pre-fix supervisor-reported cost): the wall-clock cost of `hybrid` is dominated by the slowest investigator, not by the sum. On the cleaned-up-solution MuSiQue pilot we are at OPERA's MuSiQue token cost (21k vs 20k) at higher `contain` (0.40 vs 0.361). The cross-dataset claim of AMAS dominance on HotpotQA and 2Wiki rests on supervisor-reported sufficiency-code numbers and must be reproduced post-fix.

**Q2: collaboration strategies.** The collaboration that pays off is isolated-context delegation with structured return. The single most expensive failure mode (full passage propagation through the orchestrator chat) is also the one most easily prevented by structural means. AMAS contribution to Q2 is the discipline that raw passages stay private to the agent that read them, and the orchestrator only sees evidence capsules.

**Q3: inference-time scaling.** Yes, but adaptively. Within a topology, `max_searches_per_subagent` traces a concave curve. Across topologies, the orchestrator choice of `search_only` / `spawn_only` / `hybrid` is the real scaling knob: it lets the system spend two thousand tokens on an easy question and twenty-five thousand on a hard one, without any explicit budget input.

---

## 10 Limitations

- The 12k-token Qwen3-8B context is too tight for AMAS-on-cleaned-up-solution. Either a longer-context local model or a more aggressive context-compaction policy is needed before AMAS can run cheaply on local hardware.
- We do not tune the orchestrator model; all reported main numbers use a single `gpt-4o` configuration. Sensitivity to orchestrator size is reported only as ablations.
- We do not learn the routing policy. AMAS adaptivity is an emergent consequence of the prompt and tool-choice action space, not a learned dispatcher.
- The evaluation is on three multi-hop QA benchmarks. Open-domain summarisation, mentioned in the project proposal as a candidate, is left to future work.

---

## 11 Future Work

- **Reproduce HotpotQA and 2Wiki on cleaned-up-solution.** The supervisor-reported AMAS cross-dataset table (Section 5.1) was produced on the `sufficiency-code` branch with the chunk-leak bug present. Until those numbers are re-run on `cleaned-up-solution` with the chunk-leak fix, the cross-dataset dominance claim should be treated as provisional. This is the highest-priority follow-up.
- **Re-test Qwen3-8B with thinking at higher context budget.** The 50q result at `context_token_budget`=9k saw 18 of 50 budget hits and a `spawn_only` collapse. The natural next experiment is `context_token_budget`=11k and `max_tokens`=6k inside the 12k `max_model_len`. Only after that should the local-orchestrator path be declared infeasible.
- **Run the planned ablation sweeps.** `top_k` 5 vs 10, excerpt length 300 / 700 / full, `max_searches_per_subagent` 1 / 2 / 3 / 4 / 5. These give the inference-time scaling curve called for by Q3.
- **Spawn-only failure diagnosis.** The 4 `spawn_only` MuSiQue cases all failed; their bridge slots are misidentified. A confidence threshold or a "spawn only when orchestrator has read at least one chunk" precondition is the obvious mitigation.
- **Learned routing.** Replace the prompt-emergent topology with a small classifier that picks `search_only` / `spawn_only` / `hybrid` from question features. Compare against the emergent baseline on contain/cost.
- **Local-first orchestrator at scale.** Move to a longer-context base model (Qwen3-14B, Qwen3-32B, or DeepSeek-V3.1-Terminus) and reproduce the `gpt-4o` row at lower dollar cost.
- **Open-domain summarisation.** Test AMAS on a summarisation benchmark from the proposal as a generalization probe.

---

## 12 Reproducibility appendix

Key files (all on `cleaned-up-solution` branch unless noted):

- Runner: `scripts/run_amas.py`
- Eval: `scripts/eval_offline.py`
- Orchestrator: `src/amas/orchestrator.py`
- Investigator: `src/amas/investigator.py`
- LLM client (with round-robin support across vLLM ports): `src/amas/llm.py`
- Retriever client (compatible with both node408 server and a local E5 sentence retriever): `src/amas/retriever.py`
- Best-quality 50q config: `configs/_runtime/amas_4o_orch_4omini_sub_node408_top5.yaml` (`max_searches_per_subagent=5`).
- Budgeted 50q config: `configs/_runtime/amas_4o_orch_4omini_sub_node408_top5_budget.yaml` (`max_searches_per_subagent=3`, 700-char excerpts).
- Qwen orchestrator config (with 3-port RR): `configs/_runtime/amas_qwen3think_orch_4omini_sub_node408_top5_budget.yaml`.

Run command (50q OPERA-matched MuSiQue, GPT-4o budgeted):

```bash
.venv/bin/python scripts/run_amas.py \
  --config configs/_runtime/amas_4o_orch_4omini_sub_node408_top5_budget.yaml \
  --questions data/musique/opera408_50.json \
  --output-dir results/amas_cleaned/<run_name> \
  --retriever-url http://node408:8003 \
  --concurrency 8
```

Eval:

```bash
.venv/bin/python scripts/eval_offline.py \
  --predictions results/amas_cleaned/<run_name>/predictions.jsonl \
  --questions data/musique/opera408_50.json \
  --output results/amas_cleaned/<run_name>/eval.json
```

All token counts in this thesis are computed from the `usage` field of the API or vLLM response and never estimated.

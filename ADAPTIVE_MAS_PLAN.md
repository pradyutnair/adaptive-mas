# Adaptive Recursive SAGE Plan

**Status**: EMNLP-ready research plan  
**Phase**: MuSiQue-first  
**Codebase**: `/projects/prjs1800/tmp/04-sage-autonomous/`  
**Working copy inspected**: `/local/yzheng/pnair/workspace/04-sage-autonomous` on `node409`

## 1. Core Claim

The contribution is **not** “static MAS with a better gate.” The contribution is an **adaptive recursive SAGE** controller that keeps a clean fact memory and decides one step at a time:

- `answer`: answer now from the clean fact state
- `spawn`: delegate one missing-fact retrieval problem to a fresh subagent
- `verify`: check one brittle or conflicting claim before continuing

The mechanism claim is strict:

- Single-agent / static-MAS systems fail on hard multi-hop because raw retrieval noise accumulates and corrupts later reasoning.
- Recursive subagents help by **absorbing retrieval pollution outside the orchestrator context**.
- The orchestrator only sees short distilled facts with citations.
- Adaptivity is the control law: easy questions should stay near SAS compute, hard questions should spend extra only when the orchestrator cannot justify the next step.

This is the intended response to the equal-budget SAS `>` MAS result in [Tran & Kiela 2026](https://arxiv.org/abs/2604.02460): static coordination loses, but **adaptive recursive subagents** can improve hard cases without paying universal MAS overhead.

Positioning:

- [RLM](https://arxiv.org/abs/2512.24601): recursive control pattern
- [A-RAG](https://arxiv.org/abs/2602.03442): hierarchical retrieval tools and test-time scaling
- [SPARC-RAG](https://arxiv.org/abs/2602.00083): context management and adaptive search motivation
- [HERA](https://arxiv.org/abs/2604.00901): contrast with offline / topology-learning orchestration

## 2. What Current SAGE-Auto Is Actually Failing On

Observed on `node409` from `results/auto_1k_strict/musique/predictions.jsonl`:

- Strict MuSiQue: **30.2 EM / 40.4 F1 / 40.7 contain**
- Strict no-thinking MuSiQue: **17.8 EM / 28.4 F1 / 25.6 contain**
- Therefore, reasoning capacity matters; the issue is not solved by simply removing thought tokens.

Failure severity grows sharply with search chain length:

| Slice | N | EM |
|---|---:|---:|
| All questions | 1000 | 30.2 |
| `num_waves <= 2` | 276 | 46.0 |
| `num_waves >= 3` | 724 | 24.2 |
| `num_sub_questions <= 2` | 658 | 34.5 |
| `num_sub_questions >= 3` | 342 | 21.9 |
| `question_type = bridge` | 856 | 32.0 |
| `question_type = single_hop` | 125 | 21.6 |

Dominant error modes among the 698 strict errors:

| Failure mode | Count | Interpretation |
|---|---:|---|
| `no_final_answer` | 255 | the chain never produced a usable final answer |
| `multi_step_wrong_chain` | 236 | multiple intermediate answers existed, but the chain drifted into the wrong reasoning path |
| `single_step_semantic_mismatch` | 129 | the system answered a nearby concept/type, not the asked abstraction |
| `fully_unresolved` | 61 | retrieval / decomposition never produced a viable step |
| `aggregator_lost_correct_intermediate` | 17 | an intermediate agent had the right answer, but synthesis still failed |

What this means for adaptive recursive subagents:

- **Best-fit target regime**:
  - late-wave cases
  - `3+` sub-question chains
  - empty-final failures caused by unresolved missing facts
  - wrong-chain failures where the system kept chasing a bad intermediate result
- **Not the main target regime**:
  - ontology / abstraction mismatches like “sports league” vs “NFL”
  - answer normalization issues like “Fort Lee” vs “Fort Lee, New Jersey”
  - final formatting issues when an agent already had the gold answer

So the experiments must test whether recursive subagents help the **late-wave wrong-chain / missing-fact** regime, not whether they magically solve every SAS failure.

## 3. Updated Architecture

### 3.1 Main method: Adaptive Recursive SAGE

State:

- `question`
- `distilled_facts[]`
- `step_trace[]`
- `remaining_steps`
- `verify_budget`

Orchestrator API:

```python
decide(question, facts, trace) -> {
    "action": "answer" | "spawn" | "verify",
    "sub_question": str | None,
    "goal": str | None,
    "prior_fact_ids": list[int] | None,
    "claim": str | None,
}
```

Subagent API:

```python
run_subagent(sub_question, goal, prior_facts) -> {
    "answer": str,
    "fact": {
        "text": str,
        "confidence": float,
        "support_ids": list[str],
        "source_step": int,
    },
    "retrieved_doc_ids": list[str],
}
```

Constraints:

- the orchestrator receives **bounded evidence capsules** from subagents: answer, distilled fact, confidence, support snippets, and support IDs
- the orchestrator does **not** receive full subagent retrieval dumps or accumulate full cross-hop passage contexts in the default method
- fact text stays short and citation-bearing
- subagents receive prior distilled facts and a required goal
- the orchestrator is also the implicit verifier

### 3.2 Ablation-only baselines inside this codebase

- `SAS-static-hop`: the current planner / hop-chain / wave pipeline
- `recursive-upfront-plan`: same recursive subagent return format, but with the full hop structure planned upfront

The old gated-DAG adaptive story is demoted to ablation-only material. It is not the main proposed system.

## 4. Failure-Driven Hypotheses

### H1. Context-isolation hypothesis

Recursive subagents should help primarily on:

- `num_waves >= 3`
- `num_sub_questions >= 3`
- current `no_final_answer`
- current `multi_step_wrong_chain`

because these are exactly the cases where the present system keeps extending polluted reasoning chains.

### H2. Efficiency-preservation hypothesis

Adaptive recursion should **not** fire much on:

- easy / short-chain questions
- current SAS-correct subset
- abstraction / type-mismatch questions where more retrieval is not the bottleneck

This is how we stay aligned with Tran & Kiela rather than reintroducing MAS waste everywhere.

### H3. Verification hypothesis

Verification should matter mostly when:

- a high-confidence intermediate fact would otherwise induce a wrong next search
- multiple plausible entity/value candidates exist
- the current chain has evidence but the next missing fact is still ambiguous

### H4. Bounded-evidence hypothesis

The orchestrator should work best when it sees:

- short distilled facts
- bounded support snippets
- provenance IDs

and should degrade if it sees either:

- too little evidence to verify, or
- too much raw evidence that recreates context rot.

## 5. Pre-Mortem: Where Adaptive Recursive SAGE Can Fail

This method is not guaranteed to win. The plan must explicitly test the likely failure modes.

### F1. Premature fallback to SAS mode

The orchestrator may answer too early when it should have spawned a subagent.

Expected symptom:

- low subagent-call rate on `SAS-wrong`, `3-hop`, `4-hop`, or `waves>=3`
- no improvement over `S0`

How to test:

- compare `M1` vs `S0`
- inspect spawn rates on hard subsets

### F2. Over-delegation and efficiency collapse

The orchestrator may spawn too often, effectively recreating static MAS.

Expected symptom:

- high subagent-call rate on `SAS-correct` or easy subsets
- realized compute drifts far above SAS without sufficient gain

How to test:

- compare `M1` vs `A1`
- report realized compute on `SAS-correct`, `waves<=2`, and `subq<=2`

### F3. Lossy fact compression

The subagent may compress evidence into a wrong or underspecified fact.

Expected symptom:

- wrong chains despite apparently good retrieval
- verification fails because qualifiers or disambiguators were dropped

How to test:

- evidence-capsule-size ablation
- bounded-evidence vs pollution tradeoff

### F4. Verifier blindness

The orchestrator may receive too little support to actually verify claims.

Expected symptom:

- verify calls occur but do not improve wrong-chain errors
- `A5` and `M1` behave too similarly

How to test:

- `A5` no-verification
- `A6` always-verify
- support-packet-size ablation

### F5. Poisoned high-confidence fact

A wrong but confident subagent return can steer all later steps.

Expected symptom:

- wrong intermediate entity induces a plausible but wrong downstream chain
- error rate increases after the first confident wrong fact

How to test:

- verify-yield analysis
- hard-subset breakdown on `multi_step_wrong_chain`

### F6. Myopic next-missing-fact selection

The orchestrator may ask the wrong next question even if the current facts are correct.

Expected symptom:

- recursive system underperforms the upfront-plan ablation on some deep chains
- many respawns without useful progress

How to test:

- `M1` vs `A2`
- step-trace audit on pilot errors

### F7. Looping / non-termination behavior

The system may repeatedly recall or verify the same unresolved fact.

Expected symptom:

- many repeated sub-questions
- low marginal gain after additional steps

How to test:

- per-step marginal utility in `S0-S4`
- duplicate-subquestion rate in traces

### F8. Non-target failure regime

Some SAS failures are not caused by context rot and should not be oversold.

Expected symptom:

- little or no gain on `single_step_semantic_mismatch`
- continued errors on abstraction, ontology, and answer normalization

How to test:

- failure-category breakdown before and after recursion

The paper should state this explicitly rather than hiding it.

## 6. Main Experiment Package

All new runs are **MuSiQue only** for now. Existing MuSiQue `SAS` and `SAS-IRCoT` runs are fixed baselines and are reused as-is.

### 6.1 Sanity and pilot

| ID | Size | Purpose |
|---|---:|---|
| `P0` | 50 | validate JSON actions, trace logging, fact serialization, support-ID capture |
| `P1` | 200 | run every contribution variant once before full-scale launch |

### 6.2 Main method

| ID | Size | Variant |
|---|---:|---|
| `M1` | 1000 | adaptive recursive orchestrator, `max_steps=4`, `max_verify_calls=1`, clean fact memory cap, informed subagents |

### 6.3 Test-time scaling

Same method, same prompts, same logging, only vary recursion budget:

| ID | Size | `max_steps` |
|---|---:|---:|
| `S0` | 1000 | 0 |
| `S1` | 1000 | 1 |
| `S2` | 1000 | 2 |
| `S3` | 1000 | 3 |
| `S4` | 1000 | 4 |

These are the core scaling experiments. Use **realized** tokens and **realized** subagent calls as the x-axis, not configured step caps alone.

### 6.4 Contribution ablations

| ID | Size | Variant | Why |
|---|---:|---|---|
| `A1` | 1000 | forced recursion, always spawn until step cap | tests whether adaptivity itself matters |
| `A2` | 1000 | upfront decomposition before recursive execution | tests incremental next-missing-fact vs preplanned chain |
| `A3` | 1000 | pollution ablation: pass raw snippet summaries back to orchestrator | tests the context-hygiene mechanism directly |
| `A4` | 1000 | blind subagent: no prior facts, no explicit goal | tests whether informed delegation matters |
| `A5` | 1000 | no verification: first returned fact is always accepted | tests orchestrator-as-verifier |
| `A6` | 1000 | always verify after every spawn | tests whether verification must be adaptive rather than universal |
| `A7` | 1000 | memory-cap ablation: small vs large / unbounded orchestrator fact memory | tests whether bounded working memory is actually important |
| `A8` | 1000 | evidence-capsule-size ablation: 1 vs 2 vs 4 support snippets | tests verifier blindness vs context pollution |
| `A9` | optional | direct-answer disabled at step 0 | only if pilot leaves the adaptive story ambiguous |

### 6.5 Extra EMNLP-grade diagnostics

These are not optional if the goal is a strong paper rather than a promising prototype.

| ID | Size | Variant | Why |
|---|---:|---|---|
| `D1` | 200 | oracle / hindsight routing upper bound | estimate headroom if the controller knew exactly when recursion was needed |
| `D2` | 200 | step-trace audit on `SAS-wrong` pilot subset | manually validate whether the orchestrator chooses the right next missing fact |
| `D3` | 1000 | failure-category before/after comparison | show which failure modes recursion actually fixes |

`D1` can be implemented as a lightweight offline analysis if a full rerun is too expensive: route only the questions in the pilot where `M1` materially helps and report the oracle frontier as an upper bound, not as a deployable system.

### 6.6 Launch waves on `3xA6000`

- Wave 1: `P0`, `P1`
- Wave 2: `M1`, `S0-S4`
- Wave 3: `A1-A8`
- `A9` only if Wave 1 or Wave 2 results are hard to interpret
- `D1-D3` can run partly offline and should not block the main launch sequence

## 7. Failure-Conditioned Evaluation

The paper cannot rely only on aggregate MuSiQue EM. The analysis must be keyed to the actual SAS failure profile.

### 7.1 Required top-level table

For `M1` vs SAS vs SAS-IRCoT:

- EM
- F1
- contain
- LLM-Acc
- answer rate
- total tokens
- wall-clock
- mean subagent calls
- mean verify calls

### 7.2 Mandatory subset breakdowns

Report all of these:

- MuSiQue `2-hop`, `3-hop`, `4-hop`
- current `SAS-correct` vs `SAS-wrong`
- `num_waves <= 2` vs `num_waves >= 3`
- `num_sub_questions <= 2` vs `num_sub_questions >= 3`
- current `no_final_answer`
- current `multi_step_wrong_chain`
- current `single_step_semantic_mismatch`
- current `aggregator_lost_correct_intermediate`

This is essential because recursive subagents are expected to help only certain error regimes.

### 7.3 Required figures

- EM/F1 vs mean realized tokens for `S0-S4`, with SAS and SAS-IRCoT overlays
- distribution of `0/1/2/3/4` subagent calls
- mean subagent calls by MuSiQue hop count
- orchestrator context size vs performance for `M1` vs `A3`
- verify-call rate and correction yield
- performance on `SAS-wrong` subset vs overall compute
- evidence-capsule size vs hard-subset performance
- memory-cap size vs hard-subset performance

## 8. Acceptance Criteria

The adaptive-recursive story is supported only if all of the following are true.

### 8.1 Hard-case gain

`M1` must beat both `S0` and `A1` on at least one of:

- `3-hop`
- `4-hop`
- `num_waves >= 3`
- `SAS-wrong`

If not, recursive subagents are not helping the regime they were introduced for.

### 8.2 Easy-case efficiency protection

On easy subsets:

- `num_waves <= 2`
- `num_sub_questions <= 2`
- `SAS-correct`

`M1` should keep subagent usage low and remain close to SAS in realized compute. If it pays large overhead here, it violates the intended Tran & Kiela alignment.

### 8.3 Mechanism validation

- `A3` must hurt on hard subsets. If pollution does not hurt, the context-isolation claim is weak.
- `A2` must be weaker than `M1` on deep / hard subsets. If not, incremental decomposition is not buying anything.
- At least one of `A4` or `A5` must hurt on hard subsets. Otherwise “prior evidence” or “implicit verification” is not doing useful work.
- `A6` should not dominate `M1` on the efficiency frontier. If always-verify wins cleanly, the adaptive verifier policy is not justified.
- At least one setting in `A7` and `A8` must reveal a real tradeoff. If memory cap and evidence-packet size do not matter, the core context-hygiene story is incomplete.

### 8.4 Negative-result discipline

If recursive subagents do **not** help `single_step_semantic_mismatch`, say so explicitly. Those cases are likely type / ontology / normalization errors, not context-rot errors.

### 8.5 EMNLP sufficiency check

Before claiming the plan is paper-ready, verify that the final package contains:

- one main result table
- one scaling figure
- one hard-subset table
- one mechanism-validation table
- one failure-category before/after table
- one efficiency-fairness argument against static MAS

If any of these are missing, the paper is not yet ready.

## 9. Recommended Interpretation of Each Ablation

| Variant | Expected result | Interpretation if it fails |
|---|---|---|
| `M1` | best hard-case Pareto point | the whole adaptive-recursive story is weak |
| `A1` | more expensive, not better overall | adaptivity matters |
| `A2` | worse on deep chains | incremental “next missing fact” matters |
| `A3` | worse on deep chains | clean-context mechanism is real |
| `A4` | worse on empty-final / late-wave cases | subagents need prior evidence and goals |
| `A5` | more wrong-chain errors | verification matters |
| `A6` | more expensive, not better overall | universal verification is unnecessary |
| `A7` | one bounded-memory setting should dominate unbounded | memory hygiene matters |
| `A8` | medium evidence packet should dominate too-small and too-large | bounded evidence matters |
| `A9` | mostly diagnostic | tells us whether direct answering is masking recursion effects |

## 10. Implementation Implications For The Upcoming Code Work

The eventual code implementation should support this plan, but this document is the source of truth for what to build:

- main pipeline state is recursive, not DAG-first
- log every step action and every returned fact
- preserve per-step support IDs and retrieved document IDs
- record `num_subagent_calls`, `num_verify_calls`, `orchestrator_tokens`, `subagent_tokens`, `facts_used`, `retrieved_docs_total`
- record evidence-capsule size and memory-cap settings in every run config
- log duplicate / repeated sub-question rates for loop analysis
- keep the old hop-chain SAGE path available as an ablation path

## 11. Explicit Deferrals

Not in this phase:

- HotpotQA
- 2Wiki
- Bamboogle
- FRAMES
- any ensembling / pooling / majority voting
- any claim that recursive subagents solve ontology / normalization failures by default

Those only come after MuSiQue shows a strong signal.

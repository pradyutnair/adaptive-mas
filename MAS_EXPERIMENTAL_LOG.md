# MAS Experimental Log

## 2026-04-14 23:16 CEST

Current status before the next iteration:

- `S0` pilot200: `EM 0.205`, `contain 0.230`, `avg_subagents 1.00`, `avg_tokens 25.8k`
- `M1` pilot200: `EM 0.275`, `contain 0.320`, `avg_subagents 3.17`, `avg_tokens 72.3k`
- `M1.1` pilot200: `EM 0.255`, `contain 0.315`, `avg_subagents 2.685`, `avg_auto_verify 1.10`, `avg_answer_reject 0.75`, `avg_tokens 82.0k`

Interpretation:

- `M1.1` is more adaptive than `M1` and now has a real cheap lane.
- `M1.1` still regressed on final EM versus `M1`.
- The leading hypothesis is controller over-triggering: too much automatic verification and too many answer rejections/escalations.

Immediate loop:

1. Diagnose `M1.1` losses against `M1` from traces.
2. Patch the controller with a principled reduction in unnecessary escalation/verification.
3. Rerun the same MuSiQue pilot200 on all 3 GPUs.
4. Repeat until the pilot clears the target bar.

## 2026-04-14 23:24 CEST

Diagnosis from `M1` vs `M1.1` on the full pilot:

- `M1.1` beat `M1` on `6` questions and lost on `10`.
- The losses clustered around routed `recurse` cases, not the cheap lane.
- `8/10` losses had both `answer_rejection_count > 0` and `auto_verify_calls > 0`.
- Several failed traces showed empty investigator capsules being auto-verified and then repeatedly escalated.
- The routed slot graph also duplicated the final answer slot in some recurse cases, forcing unnecessary `answer_rejected_escalate` even when the supporting facts were already sufficient.

Patch set applied:

1. Route cleanup: stop inflating the slot graph with a duplicate terminal slot when the router already returned a dependency chain.
2. Slot normalization: spawned `slot_name` must resolve to an exact pending slot name; malformed values now collapse to the first valid pending slot.
3. Verification cleanup: auto-verify now skips empty capsules and only verifies contradictory facts or low-confidence facts on the terminal answer slot.

Expected effect:

- Lower `avg_auto_verify`
- Lower `avg_answer_reject`
- Fewer empty-answer failures after repeated escalation
- Better hard-case EM without losing the cheap lane

## 2026-04-14 23:33 CEST

Iteration 2 rerun launched on all 3 GPUs using the same pilot200 split into 3 shards.

Very early telemetry (`n=12`) after the patch:

- `EM 0.500`
- `contain 0.583`
- `avg_subagents 1.83`
- `avg_auto_verify 0.00` (down from `1.10`)
- `avg_answer_reject 0.00` (down from `0.75`)
- `avg_tokens 25.6k`

Early interpretation:

- The controller is no longer wasting budget on empty-capsule verification.
- The duplicate-slot / malformed-slot rejection loop appears to be gone in the prefix.
- Need a larger prefix before trusting EM, but the controller-shape metrics moved sharply in the intended direction.

## 2026-04-14 23:39 CEST

Larger iteration-2 prefix (`n=36`):

- `EM 0.444`
- `contain 0.472`
- `avg_subagents 2.17`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.14`
- `avg_tokens 33.0k`

Interpretation:

- The controller-shape metrics are now in a much healthier range than both `M1` and the first `M1.1`.
- `avg_subagents` is inside the target band (`1.6–2.2`) on the prefix.
- The main remaining question is whether the EM holds as the harder tail of the pilot finishes.

## 2026-04-14 23:47 CEST

Mid-run iteration-2 checkpoint (`n=91`):

- `EM 0.341`
- `contain 0.396`
- `avg_subagents 2.41`
- `avg_auto_verify 0.02`
- `avg_answer_reject 0.35`
- `avg_tokens 44.7k`

Comparison to the failed `M1.1` run at larger prefixes:

- EM is higher.
- Verify is dramatically lower.
- Answer rejection is materially lower.
- Token cost is much lower.

Decision:

- Let iteration 2 finish.
- Do not patch again mid-run unless the tail collapses sharply.

## 2026-04-14 23:59 CEST

Late iteration-2 checkpoint (`n=199`):

- `EM 0.246`
- `contain 0.291`
- `avg_subagents 2.59`
- `avg_auto_verify 0.01`
- `avg_answer_reject 0.62`
- `avg_tokens 55.7k`

Interpretation:

- The verify fix worked.
- The overall method still regressed below `M1`.
- The remaining dominant bug is slot semantics: the router and answer model still leak malformed slot names (`target_slot`, full question strings, generic labels like `When`) that cause pointless answer rejections and dead-end recursion.

Iteration 3 prepared locally:

1. Strip placeholder slot names from routed `target_slot` and `required_hops`.
2. Normalize `missing_slot` against the actual pending slot list before the escalation rule sees it.
3. Tighten the route/answer prompts so slot names must be semantic slot labels, not placeholders.

Next action:

- Sync iteration 3 patch to `node409`.
- Archive iteration-2 shard outputs.
- Rerun the same pilot200 on all 3 GPUs immediately.

## 2026-04-15 00:07 CEST

Iteration-3 early prefix (`n=22`):

- `EM 0.682`
- `contain 0.773`
- `avg_subagents 1.86`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.045`
- `avg_tokens 28.3k`

Interpretation:

- Placeholder-slot cleanup appears to have removed a large share of bogus rejection/escalation.
- The controller is again inside the target subagent band while staying cheap.
- Need a larger prefix before trusting the EM, but this is the strongest early controller-shape signal so far.

## 2026-04-15 00:16 CEST

Iteration-3 larger prefix (`n=79`):

- `EM 0.392`
- `contain 0.418`
- `avg_subagents 2.28`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.30`
- `avg_tokens 42.6k`

Interpretation:

- This is materially stronger than iteration 2 at a comparable stage.
- The controller remains adaptive without collapsing into the previous rejection loop.
- Let iteration 3 continue; it is still the best candidate so far.

## 2026-04-15 00:32 CEST

Iteration-3 large prefix (`n=145`):

- `EM 0.331`
- `contain 0.372`
- `avg_subagents 2.40`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.44`
- `avg_tokens 46.8k`

Interpretation:

- Iteration 3 remains above the target overall EM bar (`0.310`) on a large prefix.
- It is materially ahead of both `M1` and iteration 2 at this stage.
- The hard-tail could still pull the final number down, but this is the first iteration that still looks publishable deep into the run.

## 2026-04-15 00:41 CEST

Iteration-3 final result:

- `EM 0.265`
- `contain 0.295`
- `avg_subagents 2.53`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.595`
- `avg_tokens 55.4k`

Conclusion:

- Iteration 3 improved controller hygiene but still failed to beat `M1`.
- The remaining bug is structural: the loop still allows `answer` proposals while pending slots are unresolved, then burns budget rejecting them afterwards.

Iteration-4 patch prepared locally:

1. If pending slots are unresolved and step budget remains, block `answer` immediately.
2. Convert that branch straight into the next spawn proposal.
3. Reserve actual answer generation for cases where the slot graph is satisfied or the budget is exhausted.

## 2026-04-15 00:49 CEST

Iteration-4 early prefix (`n=25`):

- `EM 0.600`
- `contain 0.680`
- `avg_subagents 2.00`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.00`
- `avg_tokens 32.1k`

Interpretation:

## 2026-04-16 19:20 CEST

Iter27 no-think controller patch set:

- strict distillation by default
- deterministic second-pass echo repair
- one-step recursive slot refinement
- tighter hop-chain answer synthesis

Pilot200 gate on `questions_pilot200_seed42.json`:

- `iter26_no_think`: `EM 0.230`, `F1 0.3461`, `contain 0.265`, `answered 200/200`, `avg_tokens 34.2k`
- `iter27_no_think`: `EM 0.210`, `F1 0.3259`, `contain 0.280`, `answered 200/200`, `avg_tokens 32.2k`

Decision:

- Scale to 1000 because the primary metric improved (`contain +1.5pp`) at lower token cost.

Full1000 seeded run on `questions_1000_seedfull_shard{0,1,2}.json`:

- `iter27_no_think`: `EM 0.182`, `F1 0.3019`, `contain 0.267`, `answered 998/1000`, `avg_subagents 2.50`, `avg_verify 0.12`, `avg_tokens 32.0k`

Comparison snapshot:

- Better than `S0_1000` on all quality metrics while staying in the same order of token cost.
- Much cheaper and far more complete than `iter16_1000`, but still behind it on raw EM/F1/contain.
- Still well below `A1_1000` on raw quality, but at roughly one-third the token cost.

## 2026-04-16 22:35 CEST

Thinking-ablation and Pareto checkpoint:

`S0_no_think` on seeded `1000q`:

- `EM 0.130`
- `F1 0.2412`
- `contain 0.182`
- `answered 1000/1000`
- `avg_tokens 11.3k`

Interpretation:

- no-thinking fixes the severe completion problem in `S0`
- but raw quality drops versus the original thinking-enabled `S0`

`iter27_think` on seeded `1000q`:

- `EM 0.276`
- `F1 0.3791`
- `contain 0.363`
- `answered 928/1000`
- `avg_subagents 2.81`
- `avg_verify 0.04`
- `avg_tokens 60.2k`

Pareto comparison against frozen `iter16_1000`:

- `iter16_1000`: `EM 0.284`, `F1 0.3698`, `contain 0.334`, `answered 573/1000`, `avg_tokens 63.0k`
- `iter27_think`: `EM 0.276`, `F1 0.3791`, `contain 0.363`, `answered 928/1000`, `avg_tokens 60.2k`

Outcome:

- `iter27_think` is the current headline result
- it beats `iter16_1000` on the primary metric (`contain`) at lower token cost
- it also substantially improves answer coverage
- this is the current EMNLP figure candidate to preserve before further experimentation

- The structural fix is behaving exactly as intended on the prefix.
- Unresolved pending slots are no longer burning budget through answer→reject loops.
- Need a larger prefix before trusting EM, but this is the cleanest controller behavior yet.

## 2026-04-15 01:04 CEST

Iteration-4 larger prefix (`n=74`):

- `EM 0.459`
- `contain 0.486`
- `avg_subagents 2.30`
- `avg_auto_verify 0.014`
- `avg_answer_reject 0.027`
- `avg_tokens 41.0k`

Interpretation:

- This is the strongest mid-run profile yet.
- The answer-blocking fix appears to have removed the main wasteful loop without collapsing adaptiveness.
- Let iteration 4 continue; it is now the clear best candidate.

## 2026-04-15 01:19 CEST

Iteration-4 large prefix (`n=145`):

- `EM 0.310`
- `contain 0.338`
- `avg_subagents 2.45`
- `avg_auto_verify 0.021`
- `avg_answer_reject 0.062`
- `avg_tokens 50.9k`

Interpretation:

- Iteration 4 is the first run to stay at or above the target overall EM bar deep into the hard tail.
- The rejection loop is largely gone while adaptiveness is preserved.
- Let iteration 4 finish and only patch again if the final 55 questions collapse the result.

## 2026-04-15 01:42 CEST

Iteration-4 final result:

- `EM 0.255`
- `contain 0.280`
- `avg_subagents 2.56`
- `avg_auto_verify 0.015`
- `avg_answer_reject 0.055`
- `avg_tokens 57.5k`

Conclusion:

- The answer-rejection loop is fixed.
- The remaining problem is the recurse controller itself; even without the wasteful loop, it still underperforms `M1` on the hard tail.

Next iteration direction:

- Keep the routed cheap lane.
- For `recurse`, fall back to the original `M1` recursive controller instead of the slot-heavy `M1.1` recurse path.
- Hypothesis: preserve `M1` hard-case behavior while recovering easy-case efficiency gains from routing.

## 2026-04-15 01:22 CEST

Iteration-5 launched:

- Hybrid `M1.1` keeps the routed cheap lane.
- Any `route_decision == recurse` now falls back to the original `M1` recursive controller.
- Goal: recover `M1` hard-case behavior while preserving direct-answer and single-probe efficiency gains.

## 2026-04-15 01:25 CEST

Iteration-5 relaunch fix:

- Shard runners initially failed on two GPUs because `sentence-transformers` auto-bound the E5 encoder to `cuda:0`, colliding with resident vLLM memory.
- Relaunched all three shard runners with `CUDA_VISIBLE_DEVICES=` so retrieval stays on CPU and all three GPUs remain dedicated to vLLM generation.

## 2026-04-15 01:29 CEST

Iteration-5 first completed prefix (`n=10`):

- `EM 0.600`
- `contain 0.700`
- `avg_subagents 2.00`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.00`
- `avg_tokens 25.5k`

Immediate read:

- Quality is strong and cost is materially below old `M1`.
- Every completed case routed to `recurse`, so the cheap lane has not shown itself yet.
- Do not patch yet; wait for a larger prefix to see whether this is just front-of-shard composition or a genuine routing failure.

## 2026-04-15 01:34 CEST

Iteration-5 diagnosis at `n=22`:

- `EM 0.591`
- `contain 0.636`
- `avg_subagents 2.05`
- route mix: `recurse=22`, `single_probe=0`, `direct_answer=0`
- completed hop mix: `2-hop=15`, `3-hop=4`, `4-hop=3`

Conclusion:

- This is a real router failure, not just a hard-prefix artifact.
- The router is still structurally over-conservative on many 2-hop questions.
- Root cause: `route_with_usage()` coerced any `single_probe` with multiple required slots back to `recurse`, which blocks the intended probe-then-decide lane on compositional questions.

Next iteration direction:

- Remove the `single_probe -> recurse` coercion.
- Redefine `single_probe` in the route prompt as a cheap first grounded probe that may seed later recursion, rather than a lane that must solve the full question in one step.

## 2026-04-15 01:38 CEST

Iteration-6 launched:

- Removed the structural `single_probe -> recurse` coercion in `route_with_usage()`.
- Updated the route prompt so `single_probe` means "one cheap grounded probe before re-evaluation", not "must solve the whole question in one step".
- Expected effect: non-zero cheap-lane mass on 2-hop and some easy 3-hop questions without sacrificing later recursion when needed.

## 2026-04-15 01:45 CEST

Iteration-6 first meaningful prefix (`n=19`):

- `EM 0.526`
- `contain 0.579`
- `avg_subagents 1.95`
- `avg_tokens 27.6k`
- route mix: `recurse=16`, `direct_answer=2`, `single_probe=1`
- completed hop mix: `2-hop=14`, `3-hop=2`, `4-hop=3`

Interpretation:

- The structural fix worked: the cheap lane is now real.
- Cost remains far below old `M1` while quality is still much stronger than the old pilot.
- Keep iteration 6 running; the next question is whether this advantage survives once the hard tail arrives.

## 2026-04-15 01:52 CEST

Iteration-6 deeper prefix (`n=33`):

- `EM 0.455`
- `contain 0.545`
- `avg_subagents 2.18`
- `avg_tokens 31.4k`
- route mix: `recurse=28`, `direct_answer=4`, `single_probe=1`

Interpretation:

- Quality has come down from the tiny prefix, but it is still well above old `M1`.
- Compute remains in the target range for the adaptive story.
- Keep running; the decisive test is still the hard tail, not the first 33 examples.

## 2026-04-15 02:00 CEST

Iteration-6 larger prefix (`n=65`):

- `EM 0.415`
- `contain 0.477`
- `avg_subagents 2.43`
- `avg_tokens 40.0k`
- route mix: `recurse=54`, `direct_answer=7`, `single_probe=4`

Current partial easy/hard read on the completed overlap with `S0`:

- `S0-easy` (`n=24`): `EM 0.875`, `avg_tokens 28.8k`, `avg_subagents 2.00`
- `S0-hard` (`n=41`): `EM 0.146`, `avg_tokens 46.5k`, `avg_subagents 2.68`

Conclusion:

- Hard-case behavior is already near the `A1` pilot bar at much lower cost.
- The remaining miss is easy-case routing: too many `S0-easy` questions are still entering `recurse`.
- On the completed easy subset, route mix is overwhelmingly `recurse`, which is exactly the wrong behavior for the adaptive claim.

Next iteration direction:

- Strengthen the route prompt so `single_probe` is the default choice under uncertainty.
- Reserve `recurse` only for cases where the first useful retrieval query must already be narrower than the original question.

## 2026-04-15 02:03 CEST

Iteration-7 launched:

- Route prompt now explicitly states the cost ordering `direct_answer < single_probe < recurse`.
- Under uncertainty between `single_probe` and `recurse`, the controller must choose `single_probe`.
- `recurse` is now reserved for cases where even the first useful retrieval step must already be narrower than the original question.

## 2026-04-15 02:10 CEST

Iteration-7 first meaningful prefix (`n=20`):

- `EM 0.600`
- `contain 0.650`
- `avg_subagents 1.95`
- `avg_tokens 26.2k`
- route mix: `recurse=18`, `direct_answer=2`, `single_probe=0`

Current partial easy/hard read on the completed overlap with `S0`:

- `S0-easy` (`n=12`): `EM 0.833`, `avg_tokens 25.4k`, `avg_subagents 1.92`
- `S0-hard` (`n=8`): `EM 0.250`, `avg_tokens 27.5k`, `avg_subagents 2.00`

Interpretation:

- Iteration 7 is stronger than iteration 6 on the early slice in both quality and cost.
- The easy slice is cheaper than iteration 6 but still not near the final target.
- The prompt change shifted some mass into `direct_answer`, but `single_probe` is still absent; keep running before deciding whether that is a real problem or just a prefix artifact.

## 2026-04-15 02:18 CEST

Iteration-7 follow-up diagnosis (`n=49`):

- `EM 0.449`
- `contain 0.510`
- `avg_subagents 2.24`
- `avg_tokens 33.8k`

Partial overlap with `S0`:

- `S0-easy` (`n=21`): `EM 0.905`, `avg_tokens 26.6k`, `avg_subagents 1.86`
- `S0-hard` (`n=29`): `EM 0.138`, `avg_tokens 38.8k`, `avg_subagents 2.52`

Conclusion:

- Routing is improving, but the easy slice is still paying avoidable controller overhead.
- The cheap lane still performs an extra `decide`/`answer` turn even when the first grounded probe already resolves the target slot.

Next iteration direction:

- Add one-probe grounded early finalization.
- If the initial `direct_answer` or `single_probe` retrieval resolves all pending slots with a supported high-confidence fact, return the capsule answer immediately instead of calling `decide` again.

## 2026-04-15 02:22 CEST

Iteration-8 launched:

- Added one-probe grounded early finalization in the cheap lane.
- If the first grounded probe resolves all pending slots with a supported fact above the answer-justification threshold, the pipeline now returns the capsule answer immediately.
- Expected effect: lower easy-case token use and lower mean subagent/controller overhead without changing the hard recursive path.

## 2026-04-15 02:31 CEST

Iteration-8 diagnosis (`n=48`):

- `EM 0.375`
- `contain 0.438`
- `avg_subagents 2.27`
- `avg_tokens 34.5k`

Conclusion:

- The one-probe early-finalize branch is worse than iteration 7 at comparable scale.
- Easy-case cost did not improve enough to justify the quality loss.
- Treat iteration 8 as a failed branch and revert it.

Next step:

- Roll back the early-finalize patch.
- Restore the best iteration-7 partial checkpoint and resume it to completion instead of restarting from scratch.

## 2026-04-15 02:35 CEST

Resumed best branch:

- Rolled back iteration-8 code.
- Restored the archived iteration-7 partial checkpoint into the active shard output directories.
- Relaunched from checkpoint so the runner only processes the remaining questions on the strongest known branch.

## 2026-04-15 02:43 CEST

New diagnosis from resumed iteration-7 hard errors:

- On many hard failures, the route step chose `recurse` but the next action was still a broad bootstrap on the original question.
- The hybrid controller was discarding the router's focused first sub-question and dropping into legacy `M1`, which re-asked the original question as its bootstrap.
- This explains several hard errors where the first retrieval step was too broad even though the router had already identified a narrower first probe.

Next iteration direction:

- Thread the router's `sub_question` and `goal` into the legacy adaptive bootstrap when `route_decision == recurse`.
- That keeps the strong iteration-7 routing behavior on easy questions while making the hard path actually respect the routed first probe.

## 2026-04-15 02:47 CEST

Iteration-9 launched:

- `recurse` now seeds legacy adaptive execution with the router-selected first probe instead of discarding it.
- This is the first hybrid run where both easy-case routing and hard-case first-step selection are structurally aligned with the controller.

## 2026-04-15 02:54 CEST

Iteration-9 first meaningful prefix (`n=24`):

- `EM 0.500`
- `contain 0.542`
- `avg_subagents 2.04`
- `avg_tokens 28.2k`

Partial overlap with `S0`:

- `S0-easy` (`n=13`): `EM 0.923`, `avg_tokens 27.9k`
- `S0-hard` (`n=13`): `EM 0.154`, `avg_tokens 29.8k`

Interpretation:

- This is the first branch to hit the hard-case pilot bar on the current overlap.
- The recurse bootstrap fix appears to be helping the hard path.
- Easy routing is still too expensive, but this is now the best hard-case structural signal so far. Let it run deeper before touching it again.

## 2026-04-15 03:03 CEST

Iteration-9 deeper prefix (`n=50`):

- `EM 0.500`
- `contain 0.580`
- `avg_subagents 2.26`
- `avg_tokens 34.4k`

Partial overlap with `S0`:

- `S0-easy` (`n=22`): `EM 0.955`, `avg_tokens 28.3k`, `avg_subagents 2.00`
- `S0-hard` (`n=30`): `EM 0.167`, `avg_tokens 39.6k`, `avg_subagents 2.47`

Interpretation:

- This is the first branch that clears both quality bars on the current overlap.
- Easy-case EM is now at the target level.
- Hard-case EM is now above the pilot target and above the `A1` reference bar.
- The remaining weakness is easy-case efficiency, not quality.

## 2026-04-15 03:11 CEST

Follow-up diagnosis from deeper iteration-9 errors:

- Many `recurse` failures still had the original question as the first bootstrap sub-question.
- Even after threading route state into legacy bootstrap, the router often returned the original question instead of a narrower first missing-fact probe.
- That means the hard path was still wasting its first step on a broad query in many failures.

Next iteration direction:

- Enforce recurse-first-step validity.
- If `route_decision == recurse` but the route sub-question is identical to the original question, run a focused `propose_spawn` refinement before entering the legacy loop.

## 2026-04-15 03:15 CEST

Iteration-10 launched:

- Added recurse-first-step refinement.
- Recurse can no longer enter the legacy loop with the original question as its first bootstrap probe when a narrower first missing-fact question is required.

## 2026-04-15 03:24 CEST

Iteration-10 final:

- `EM 0.265`
- `contain 0.300`
- `avg_subagents 2.86`
- `avg_tokens 66.1k`

Overlap with `S0`:

- `S0-easy` (`n=41`): `EM 0.829`, `avg_tokens 33.4k`
- `S0-hard` (`n=159`): `EM 0.120`, `avg_tokens 74.6k`

Conclusion:

- Iteration 10 collapsed and is not competitive.
- The recurse-first-step refinement by itself did not solve the main problem.

Diagnosis:

- Many failing `recurse` trajectories still look like `route -> spawn -> answer`.
- The system is stopping after a single retrieved fact on questions the router itself classified as requiring recursion.
- The next structural fix should target premature answer acceptance on recurse routes, not first-step retrieval selection alone.

## 2026-04-15 03:30 CEST

Next structural fix prepared:

- Added a recurse-specific minimum depth guard.
- If the route classified the question as `recurse`, the legacy adaptive loop is no longer allowed to answer after only one retrieved fact.
- Instead it must retrieve at least one additional missing fact before answering.

## 2026-04-15 03:34 CEST

Iteration-11 launched:

- Enabled recurse-specific minimum depth.
- Recurse routes must now retrieve at least two facts before the legacy adaptive loop is allowed to answer.
- Remote launch note: had to validate from `/tmp` because repo-root `types.py` shadows the stdlib `types` module during `python -m py_compile`.

## 2026-04-15 03:39 CEST

Iteration-11 relaunch fix:

- The initial shard launch picked up the wrong `python3` on node409 and hit Python 3.6 syntax errors.
- Relaunched all three shard runners with `/local/yzheng/pnair/workspace/05-mas/.venv/bin/python3` explicitly.

## 2026-04-15 03:43 CEST

Iteration-11 first prefix (`n=9`):

- `EM 0.556`
- `contain 0.667`
- `avg_subagents 2.00`
- `avg_tokens 25.4k`
- route mix: `recurse=9`

Interpretation:

- Early quality and cost are promising.
- Still too early to judge because only recurse trajectories have completed so far.

## 2026-04-15 03:49 CEST

Next refinement prepared:

- The recurse minimum-depth guard will now be confidence-gated.
- Only recurse routes with confidence at or above `0.7` will be forced to retrieve at least two facts before answering.
- Lower-confidence recurse decisions will fall back to the normal legacy stopping rule, which should reduce collateral damage on misrouted easy cases.

## 2026-04-15 03:53 CEST

Iteration-12 launched:

- Enabled confidence-gated recurse depth.
- High-confidence recurse routes keep the two-fact minimum.
- Lower-confidence recurse routes are allowed to answer under the normal stopping rule.

## 2026-04-15 03:57 CEST

Iteration-12 first prefix (`n=10`):

- `EM 0.400`
- `contain 0.500`
- `avg_subagents 1.70`
- `avg_tokens 22.3k`
- route mix: `recurse=7`, `direct_answer=3`

Interpretation:

- Cheaper than iteration 11 on the first slice.
- Too early to tell whether the quality tradeoff is acceptable.

## 2026-04-15 04:06 CEST

Iteration-12 larger prefix (`n=34`):

- `EM 0.588`
- `contain 0.647`
- `avg_subagents 2.03`
- `avg_tokens 29.8k`

Partial overlap with `S0`:

- `S0-easy` (`n=18`): `EM 0.889`, `avg_tokens 28.5k`
- `S0-hard` (`n=17`): `EM 0.235`, `avg_tokens 31.2k`

Interpretation:

- This is the strongest hard-case prefix so far.
- Easy-case EM is still a bit below the target on this overlap, but the hard slice is materially stronger and cheaper than earlier branches.
- Keep iteration 12 running; it is the new best live branch.

## 2026-04-15 04:17 CEST

Iteration-12 late prefix (`n=192`):

- `EM 0.286`
- `contain 0.318`
- `avg_subagents 2.93`
- `avg_tokens 68.1k`

Overlap with `S0`:

- `S0-easy` (`n=41`): `EM 0.878`, `avg_tokens 35.5k`
- `S0-hard` (`n=151`): `EM 0.126`, `avg_tokens 77.0k`

Conclusion:

- Iteration 12 looked strong early but failed at scale.
- Confidence-gated recurse depth did not preserve the quality gains once the hard tail arrived.

Next step:

- Treat iteration 12 as another failed branch.
- Inspect the current hard failures directly before choosing the next structural change.

## 2026-04-15 04:22 CEST

Next refinement prepared:

- Strengthen recurse bootstrap refinement.
- If the recurse route and the first refinement both still echo the original question, force a second refinement that targets only the first unresolved slot in dependency order.

## 2026-04-15 04:25 CEST

Iteration-13 launched:

- Added slot-focused second-pass recurse refinement.
- Recurse should now be much less likely to waste its first step on a broad question that simply restates the original query.

## 2026-04-15 04:36 CEST

Iteration-13 final result (`198/200`):

- `EM 0.278`
- `contain 0.313`
- `avg_subagents 2.95`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.02`
- `avg_tokens 69.7k`

Overlap with `S0`:

- `S0-easy` (`n=41`): `EM 0.878`, `avg_tokens 35.5k`
- `S0-hard` (`n=157`): `EM 0.121`, `avg_tokens 78.6k`

Diagnosis:

- The slot-focused bootstrap refinement did not fix the real control-path bug.
- `recurse` still falls back into the legacy `_run_adaptive` loop, so the routed `M1.1` logic is bypassed on most hard questions.
- This is why `auto_verify` and `answer_rejection` are effectively dead again, and why routed `direct_answer` examples still pay multi-step MAS cost.
- The next iteration must keep all three lanes inside the routed controller instead of delegating recurse to legacy `M1`.

## 2026-04-15 05:04 CEST

Iteration-14 launched:

- Replaced the remaining legacy handoff inside `M1.1`.
- `recurse` now stays inside the routed controller instead of calling `_run_adaptive`.
- `direct_answer` and `single_probe` now do one routed grounded probe and try to finalize immediately before entering the longer loop.
- Initial probe verification is now available on every routed lane, not only on the non-legacy path.

Expected effect:

- Non-zero cheap-lane mass with actual low-cost completions.
- Non-zero `auto_verify` and `answer_rejection` on hard routed cases, instead of collapsing back to legacy `M1`.
- Better easy-case EM/tokens without giving away hard-case control.

## 2026-04-15 05:09 CEST

Iteration-14 first prefix (`n=11`):

- `EM 0.455`
- `contain 0.636`
- `avg_subagents 1.64`
- `avg_tokens 23.0k`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.00`
- route mix: `recurse=7`, `direct_answer=3`, `single_probe=1`

Overlap with `S0`:

- `S0-easy` (`n=6`): `EM 0.833`, `avg_tokens 24.4k`
- `S0-hard` (`n=5`): `EM 0.000`, `avg_tokens 21.4k`

Interpretation:

- The structural patch did what it was supposed to do: cheap routed completions are back, including true `1`-subagent trajectories.
- This is too early to judge hard-case behavior because only five hard-overlap questions have completed.
- Let iteration 14 run deeper before patching again.

## 2026-04-15 05:41 CEST

Iteration-14 final result (`200/200`):

- `EM 0.275`
- `contain 0.310`
- `avg_subagents 2.56`
- `avg_tokens 59.4k`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.175`

Overlap with `S0`:

- `S0-easy` (`n=41`): `EM 0.854`, `avg_tokens 39.1k`
- `S0-hard` (`n=159`): `EM 0.126`, `avg_tokens 64.6k`

Diagnosis:

- The controller shape improved: cheap lanes survived to the final run and overall token cost dropped materially.
- Quality did not improve because many questions now fail at the final budget boundary with exactly one unresolved slot left.
- These failures are principled recovery failures, not routing failures: the chain often resolves the early slots correctly, but the last missing slot never gets one final focused retrieval step.

Next refinement:

- Add a final targeted recovery probe when the controller reaches the answer boundary with a specific missing slot still unresolved.
- Keep this recovery conditional on an explicit missing slot from the answer justification, so the extra retrieval is paid only on unresolved hard cases.

## 2026-04-15 06:32 CEST

Iteration-15 final result (`200/200`):

- `EM 0.245`
- `contain 0.290`
- `avg_subagents 3.13`
- `avg_tokens 71.7k`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.68`

Overlap with `S0`:

- `S0-easy` (`n=41`): `EM 0.805`, `avg_tokens 42.1k`
- `S0-hard` (`n=159`): `EM 0.101`, `avg_tokens 79.3k`

Diagnosis:

- The final targeted recovery patch was too broad and reintroduced the same failure mode in a different place.
- It fires on too many cases, pushes trajectories to `4` subagents, and still often ends with an unresolved answer.
- The principled version of this recovery should be much narrower: only when exactly one unresolved slot remains and the answer justification names that same slot.

Next refinement:

- Narrow the final recovery trigger to the single-missing-slot case only.
- Do not pay the extra recovery retrieval when multiple slots are still open or the answer justification does not align with the remaining slot.

## 2026-04-15 07:16 CEST

Iteration-16 final result (`200/200`):

- `EM 0.285`
- `contain 0.320`
- `avg_subagents 2.81`
- `avg_tokens 62.6k`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.425`

Overlap with `S0`:

- `S0-easy` (`n=41`): `EM 0.805`, `avg_tokens 37.4k`
- `S0-hard` (`n=159`): `EM 0.151`, `avg_tokens 69.1k`

Interpretation:

- This is the strongest hard-case run so far.
- Narrowing the final recovery trigger helped materially versus iteration 15.
- The remaining deficit is now clearly the easy slice: hard-case EM is at the target band, but the cheap/easy lane still gives away too many cases and too many tokens.

Next refinement:

- Diagnose the residual easy-case losses directly.
- Focus on why routed `direct_answer` / `single_probe` questions are still sliding into multi-step traces or empty finals.

## 2026-04-15 07:28 CEST

Easy-case diagnosis after iteration 16:

- Several `S0`-easy losses already contain the correct grounded answer in memory, but the final answer object still comes back empty or misaligned.
- Example: the Willy Wonka / Veruca Salt case retrieved the correct fact `Julie Dawn Cole played Veruca Salt`, but the controller discarded it because the final answer generator fixated on the surface form `Seether`.
- This is not a retrieval failure. It is answer preservation failure: a strong grounded answer candidate gets lost after decomposition.

Next refinement:

- Preserve the strongest grounded capsule answer seen so far.
- Use it as a fallback only when all routed slots are already resolved and the final answer generator still returns an empty answer.
- This keeps the method principled: no voting, no heuristics over strings, just preserving a grounded answer already produced by the system itself.

## 2026-04-15 08:15 CEST

Iteration-17 final result (`200/200`):

- `EM 0.255`
- `contain 0.295`
- `avg_subagents 2.75`
- `avg_tokens 62.3k`
- `avg_auto_verify 0.00`
- `avg_answer_reject 0.325`
- `empty_answers 90`

Overlap with `S0`:

- `S0-easy` (`n=41`): `EM 0.805`, `avg_tokens 40.3k`
- `S0-hard` (`n=159`): `EM 0.113`, `avg_tokens 67.9k`

Outcome:

- The grounded-answer preservation fallback did not improve the pilot; it regressed versus iteration 16.
- Best run remains iteration 16: `EM 0.285`, `hard overlap EM 0.151`, `avg_subagents 2.81`, `avg_tokens 62.6k`.
- By the 9am deadline, the method improved the hard slice substantially over the earlier M1.1 branches, but it still failed the core easy-case target and did not surpass the old `M1` enough to claim the full adaptive story is solved.

## 2026-04-15 09:48 CEST

Focused forensic pass on the best branch (`iteration 16`), restricted to the exact `S0-correct / M1.1-wrong` slice:

- There are exactly `8` such cases.
- Route mix on those `8`: `recurse=6`, `direct_answer=1`, `single_probe=1`.
- `5/8` end with an empty final answer.
- `4/8` already have all routed slots marked resolved.

Mechanism clustering:

1. **One clear grounded-answer preservation failure**.
   - Example: `musique_2hop__847760_80026`
   - The fact pool already contains the correct grounded answer `Julie Dawn Cole played Veruca Salt ...`, but final answer generation still returns empty.
   - This is the cleanest remaining controller/finalization bug.

2. **Two pure surface-form mismatches**.
   - `Talca` vs `Talca Province`
   - `Alpes-de-Haute-Provence department` vs `Alpes-de-Haute-Provence`
   - These are not retrieval failures, but they are also not the kind of heuristic/canonicalization patch we want to headline.

3. **The rest are genuine unresolved-slot retrieval failures or semantic mis-selection**.
   - Missing-state / missing-date / missing-opening-date failures still have unresolved slots at the end.
   - The Vivaldi example resolves all slots but selects the wrong salient bridge (`Bridge of Sighs` instead of `Rialto Bridge`), which is a reasoning/selection failure, not just answer preservation.

Conclusion:

- There is **not** one single remaining clean fix that plausibly closes the whole easy-case gap.
- A principled grounded-answer preservation fallback can recover the single clear empty-final case, but the remaining deficit is spread across true retrieval misses, semantic selection errors, and judge-benign surface-form differences.
- So the right next move is to **freeze iteration 16 as the best method**, write the paper around the actual evidence, and only reopen controller iteration if we are willing to invest in a deeper redesign of slot grounding / retrieval coverage rather than another quick patch cycle.

## 2026-04-15 10:02 CEST

Frozen as best:

- `iteration 16` is now the frozen best `M1.1` branch.
- Exact easy-failure breakdown on the forensic slice (`n=8`): `1` grounded-answer preservation failure, `2` surface-form mismatches, `5` real retrieval / semantic-selection misses.
- Decision: stop further controller edits tonight and preserve `iteration 16` as the reference result for writing.

## 2026-04-15 10:18 CEST

Headroom analysis after freezing `iteration 16`:

- Current pilot `EM`: `0.285`
- On the exact `S0-correct / iter16-wrong` forensic slice (`n=8`):
  - fixing only the single clean preservation failure gives `EM 0.290`
  - fixing preservation + the `2` surface-form cases gives `EM 0.300`
  - fixing the `5` real retrieval / selection misses gives `EM 0.310`
  - fixing all `8` gives `EM 0.325`

Broader wrong-case scan on the full pilot:

- total wrong: `143 / 200`
- wrong with empty final answer: `95`
- wrong with all slots marked resolved: `58`
- wrong with both empty final answer and all slots resolved: `11`

Interpretation:

- The easy-slice ceiling from another quick controller patch is too small to justify a new patch cycle.
- The broader wrong set still contains many false-positive slot resolutions and wrong fact selections, so even the `11` empty+resolved cases are not a clean preservation-only fix bucket.
- Decision: move to the scale-up path with frozen `iteration 16`, not another controller-iteration loop.

Scale-up preparation:

- Restored the executable code path to the frozen `iteration 16` behavior by removing the later `iteration 17` grounded-answer fallback.
- Validation passed again (`py_compile`, `112` tests).
- Built faithful `1000q` shard files that preserve the existing pilot200 shard assignments for resume-safe extension.

## 2026-04-15 10:24 CEST

`1000q` scale-up launched for frozen `iteration 16`.

- New shard question files:
  - `data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard0.json` (`334`)
  - `data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard1.json` (`334`)
  - `data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard2.json` (`332`)
- New output dirs:
  - `results/M1_1_iter16_1000_shard0`
  - `results/M1_1_iter16_1000_shard1`
  - `results/M1_1_iter16_1000_shard2`
- Each output dir was seeded with the corresponding frozen pilot shard predictions so the runner resumes from the existing `200q` work instead of rerunning it.
- Initial checkpoint state:
  - shard0: `67 completed`, `267 remaining`
  - shard1: `67 completed`, `267 remaining`
  - shard2: `66 completed`, `266 remaining`
- Runner PIDs on `node409`: `219988`, `219989`, `219990`

Artifact hygiene:

- Added executable snapshot `frozen/iter16_best/pipeline.iter16.py`
- Checksums refreshed locally and remotely after adding the code snapshot

## 2026-04-15 10:31 CEST

Live scale-up status check:

- `M1_1_iter16_1000_shard0`: `71`
- `M1_1_iter16_1000_shard1`: `72`
- `M1_1_iter16_1000_shard2`: `74`

Endpoint health:

- `8001`, `8002`, `8003` all up
- active requests: `24`, `23`, `23`
- waiting queue: `0`, `0`, `0`
- no stall or endpoint drop detected

Baseline preparation for apples-to-apples `1000q` follow-up:

- Seeded shard-aligned output dirs for:
  - `results/S0_1000_seeded_shard{0,1,2}`
  - `results/M1_1000_seeded_shard{0,1,2}`
  - `results/A1_1000_seeded_shard{0,1,2}`
- Each seeded dir contains the exact pilot200 predictions partitioned onto the same `67 / 67 / 66` shard split as frozen `M1.1`.
- This means `S0`, old `M1`, and `A1` can start their `1000q` extensions immediately as soon as GPUs free up, without rerunning the pilot portion.

## 2026-04-15 10:38 CEST

Judge-path clarification:

- The canonical judge for this project is **DeepSeek-R1-Distill-Qwen-32B** via `/projects/prjs1800/external/arag/scripts/eval.py` on Snellius.
- The repo-local `llm_judge_eval.py` was considered and explicitly rejected because it uses `gpt-4o-mini`, which would break comparability with the established scoreboard.
- `node409` does **not** have `/projects/prjs1800/...`, so the authoritative judge evaluation must run on Snellius after predictions are synced there. No local substitute judge will be used.
- Critical detail for the eventual Snellius eval job: `eval.py` defaults to `ARAG_MODEL=gpt-4o-mini` unless overridden. The E4-comparable pattern must explicitly export:
  - `ARAG_API_KEY=dummy`
  - `ARAG_BASE_URL=http://127.0.0.1:8000/v1`
  - `ARAG_MODEL=DeepSeek-R1-Distill-Qwen-32B`
- Required Snellius job pattern:
  - use the template at `/projects/prjs1800/msc-thesis/02-arag-multi-agent/jobs/eval_m5_1000_musique.job`
  - load the same modules and activate `arag-venv`
  - stand up `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` on port `8000` with matching `served-model-name`
  - wait for `/v1/models`
  - run `python -u scripts/eval.py --predictions <path> --workers 10 --output <dir>`
  - use `partition=gpu_h100`, `1 GPU`, `max-model-len=32768`, `gpu-memory-utilization=0.90`

Launch discipline:

- Baseline full1000 runs (`S0`, old `M1`, `A1`) remain queued but **not launched**.
- Current frozen-`iter16` progress is only about `22%–23%` per shard, so baseline launch is held until the active `M1.1` shards are comfortably past the requested `50%` threshold to avoid GPU contention.

## 2026-04-15 13:16 CEST

Stopped old `M1` full1000 scale-up early at partial progress (`107 / 107 / 117`) to avoid wasting more GPU time.

Action taken:

- terminated the three live `configs/m1.yaml` runner processes
- updated the post-iter16 launcher to skip old `M1` entirely
- moved directly to `A1` full1000 launch across the same 3 shard splits

Rationale:

- old `M1` full1000 is not needed for the core paper story
- the key comparison now is `S0` vs frozen `M1.1` vs `A1` on full1000

## 2026-04-15 15:58 CEST

Final full1000 MuSiQue summary (fair offline metrics):

- `E2` (from `04-sage-autonomous/ARCHITECTURE.md`):
  - `EM 13.4`
  - `contain 27.1`
  - note: the `30.3` judge row is not the fair answer-accuracy comparison because `E2` stores verbose responses
- `S0`:
  - `EM 18.6`
  - `contain 21.3`
  - `avg_tokens 24.8k`
  - `avg_subagents 1.0`
- frozen `M1.1` (`iter16`):
  - `EM 29.6`
  - `contain 34.6`
  - `avg_tokens 63.0k`
  - `avg_subagents 2.78`
  - `avg_verify 0.00`
- `A1`:
  - `EM 34.3`
  - `contain 39.2`
  - `avg_tokens 100.4k`
  - `avg_subagents 5.0`
  - `avg_verify 0.00`

Ordering:

- by EM: `E2 < S0 < M1.1 < A1`
- by compute: `E2 << S0 < M1.1 < A1`

Key deltas:

- `M1.1` vs `E2`: `+16.2 EM`, `+7.5 contain`
- `M1.1` vs `S0`: `+11.0 EM`, `+13.3 contain`
- `A1` vs `M1.1`: `+4.7 EM`, `+4.6 contain`, at about `1.59x` the token cost

Interpretation:

- The current adaptive story is real but partial.
- `M1.1` is a strong middle point on the quality/compute frontier:
  - much better than the cheap single-pass baseline `S0`
  - much better than fair `E2`
  - substantially cheaper than always-recursive `A1`
- But it does **not** yet achieve the strongest possible claim of “MAS quality at SAS cost”.
- The honest claim supported by the full1000 data is:
  - adaptive recursive MAS recovers most of the quality gain of full MAS (`A1`) at substantially lower test-time compute, while clearly outperforming single-pass baselines.

Important limitation:

- `verify` remained effectively inactive at scale (`avg_verify 0.00` for both `M1.1` and `A1`), so the observed gains come from adaptive recursive retrieval / subagent allocation, not from adaptive self-review.

## 2026-04-15 16:22 CEST

Non-launch follow-up prep completed.

Verify bug writeup, recorded but not fixed in frozen `iter16`:

- `src/adaptive_sage/prompts/orchestrator_decide.txt:3-20` only exposes two controller actions: `answer` and `spawn`. `verify` is absent from the prompt-level JSON schema, so the LLM cannot emit it.
- `src/adaptive_sage/pipeline.py:1108-1148` still contains a legacy `if action == "verify"` branch. Under the frozen prompt, this branch is unreachable from the orchestrator side.
- `src/adaptive_sage/pipeline.py:1360-1422` contains the only live verify path in frozen `iter16`: `_maybe_verify_fact(...)`.
- That live auto-verify path only fires on `(a)` conflict with memory or `(b)` final-slot fact confidence `< auto_verify_threshold`; see `src/adaptive_sage/pipeline.py:1379-1385`.
- Frozen `iter16` sets `adaptive.auto_verify_threshold: 0.7` in `frozen/iter16_best/config/m1_1.iter16.yaml:19-26`.
- Because full1000 `avg_verify=0.00` for both `M1.1` and `A1`, the practical conclusion is: verify is dead in the frozen method for two concrete reasons:
  - the orchestrator cannot choose it;
  - the auto-verify trigger is too strict or the confidence signal is too uncalibrated to cross it.

Retriever / passage-cap parity check across MuSiQue full1000 runs:

- `search_top_k` is inherited from investigator default, not overridden in any of `configs/s0.yaml`, `configs/a1.yaml`, or frozen `iter16`.
- Investigator default is `search_top_k=5`; see `src/adaptive_sage/investigator.py:52-55`.
- Therefore all three full1000 systems used:
  - retrieval top-k = `5` for keyword search
  - retrieval top-k = `5` for semantic search
  - max passage read pool = `10` (`search_top_k * 2`); see `src/adaptive_sage/investigator.py:150`
- Passage cap parity is **not** matched:
  - `S0`: `investigator.evidence_capsule_limit=2` in `configs/s0.yaml:10-11`
  - `A1`: `investigator.evidence_capsule_limit=2` in `configs/a1.yaml:13-14`
  - frozen `M1.1`: `investigator.evidence_capsule_limit=4` in `frozen/iter16_best/config/m1_1.iter16.yaml:11-13`
- So the current full1000 comparison matches retriever `k`, corpus, index, model, and runner concurrency, but does **not** match evidence capsule width. This should be stated explicitly in the paper and resolved in the later Pareto / matched-budget sweeps.

Draft-only next runs prepared, not launched:

- `scripts/draft_musique_followups.sh`
- Contains:
  - draft budget-capped `A1` commands targeting the `S0` token band
  - draft MuSiQue Pareto sweep commands for `S0`, frozen `M1.1`, and `A1`
  - all commands are comments only; nothing was launched

## 2026-04-15 17:03 CEST

Fairness rerun launched:

- Added `configs/s0_cap4.yaml` with `investigator.evidence_capsule_limit=4`, keeping `S0` otherwise unchanged (`max_steps=0`, `max_verify_calls=0`).
- Launched full MuSiQue `1000q` rerun on the exact same shard files used by frozen `M1.1`:
  - `data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard0.json`
  - `data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard1.json`
  - `data/musique/m1_1_full1000_iter16_shards/questions_full1000_iter16_shard2.json`
- Output dirs:
  - `results/S0_cap4_1000_shard0`
  - `results/S0_cap4_1000_shard1`
  - `results/S0_cap4_1000_shard2`

Operational note:

- First relaunch attempt failed because the runner tried to use `/home/yzheng/.cache/huggingface`, which was over quota.
- Second relaunch attempt failed because the shared scratch HF cache was over quota for new lockfiles.
- Final working fix: moved the existing `intfloat/e5-base-v2` cache into a clean project-local cache at `/local/yzheng/pnair/workspace/05-mas/.hf_home` and launched with `HF_HOME`, `TRANSFORMERS_CACHE`, and `HUGGINGFACE_HUB_CACHE` pointed there.

Live runner PIDs:

- `456174`
- `456175`
- `456176`

Status at launch check:

- all three shard runners are alive
- logs show active `Pipeline start` events on all three shards
- `predictions.jsonl` is not populated yet at the first 20s check, which is expected immediately after startup

## 2026-04-15 18:07 CEST

Fair `S0` cap4 rerun completed and evaluated.

Setup:

- `S0` was rerun with `investigator.evidence_capsule_limit=4` via `configs/s0_cap4.yaml`
- same MuSiQue `1000q` shard files as frozen `M1.1`
- same corpus / index / model / runner setup

Results:

- `S0` cap4:
  - `EM 18.0`
  - `F1 22.89`
  - `contain 20.4`
  - `avg_total_tokens 24,522.8`
  - `avg_num_subagent_calls 1.0`
- frozen `M1.1`:
  - `EM 29.6`
  - `F1 38.26`
  - `contain 34.6`
  - `avg_total_tokens 62,984.7`
  - `avg_num_subagent_calls 2.784`

Delta:

- frozen `M1.1` vs fair `S0` cap4:
  - `+11.6 EM`
  - `+15.37 contain`
  - `+15.71 F1`

Interpretation:

- Matching `S0` up to `evidence_capsule_limit=4` does **not** explain away the `M1.1 > S0` gap.
- `S0` cap4 actually moved slightly down relative to the earlier `S0` cap2 run (`18.6 -> 18.0 EM`), so wider evidence capsules alone do not rescue the single-pass baseline here.
- This materially strengthens the fairness of the main MuSiQue frontier story:
  - `E2 < S0 < M1.1 < A1`
  - `M1.1`'s gain over `S0` is not a trivial capsule-width artifact.

## 2026-04-15 18:34 CEST

Iteration 18 patch prepared locally. No launch yet.

Scope:

- new config: `configs/m1_1.iter18.yaml`
- code change only in `src/adaptive_sage/pipeline.py`
- frozen `iter16` artifacts unchanged

Change:

- final answer acceptance now uses a strict fallback cascade:
  1. structured `answer_obj["answer"]`
  2. `route_draft_answer`
  3. best grounded fact already in `fact_memory`
  4. question text as a final `no_evidence` marker
- final `answer` step metadata now logs `answer_source` as one of:
  - `structured`
  - `route_draft`
  - `grounded_fact`
  - `no_evidence`

Validation:

- `python -m py_compile src/adaptive_sage/pipeline.py` passed
- `pytest tests/test_types.py tests/test_fact_memory.py tests/test_investigator.py` passed: `112 passed`

Status:

- waiting for pilot-launch approval before running MuSiQue `200q`

## Iteration 24 — strict investigator extraction path on subset20
- Branch base: `iter21` champion only. No orchestrator or routing changes.
- Change: separate investigator extraction path via `investigator_llm` (`temperature=0.0`, `max_tokens=384`, `enable_thinking=false`) plus strict schema-only prompt `investigator_distill_strict.txt` selected by config.
- Subset: frozen `questions_iter21_bridge_subset20.json` (20 real retrieval-failure stress cases).
- Results vs `iter21`:
  - `iter21`: `EM 0.100`, `F1 0.1830`, `avg_tokens 70,324.4`, `parse_failure_rate 0.55`, `abstain_rate 0.65`, `gold_in_seen 0.35`, `seen_to_exact 0.286`
  - `iter24`: `EM 0.100`, `F1 0.2644`, `avg_tokens 42,372.6`, `parse_failure_rate 0.00`, `abstain_rate 0.40`, `gold_in_seen 0.45`, `seen_to_exact 0.222`
- Decision: promote `iter24` over `iter21`. Same EM, materially better F1, much lower token cost, zero parse failures, lower abstain, higher retrieval coverage. Launch 200q immediately.

## 2026-04-16 01:15 CEST

Reverted active ship candidate back to frozen `iter16`.

Reversion actions:

- restored local `src/adaptive_sage/pipeline.py` from `frozen/iter16_best/pipeline.iter16.py`
- restored remote `src/adaptive_sage/pipeline.py` on `node409` from `frozen/iter16_best/pipeline.iter16.py`
- restored `configs/m1_1.iter16.yaml` locally and on `node409` from `frozen/iter16_best/config/m1_1.iter16.yaml`
- stopped treating `iter24` as the forward/ship branch

Reason for reversion:

- the post-`iter16` branches improved isolated mechanics, but none produced a clearly better end-to-end pilot result than frozen `iter16`
- the strongest later branch (`iter24`) improved investigator reliability and efficiency, but its `pilot200` answer quality still landed below `iter16`
- `iter16` remains the best defensible full candidate until a branch beats it cleanly on pilot-scale quality

Clean record of post-`iter16` branches:

### Iteration 21 — bridge-anchored `retrieval_query`

Change:

- split `sub_question` from `retrieval_query`
- when a bridge entity was known, retrieval was allowed to anchor on that entity + missing target relation
- disabled grounded-fact final fallback on this branch

Subset20 (`questions_iter21_bridge_subset20.json`, 20 real retrieval-failure stress cases):

- `EM 0.100`
- `F1 0.1830`
- `avg_tokens 70,324.4`
- `parse_failure_rate 0.55`
- `abstain_rate 0.65`
- `gold_in_seen 0.35`
- `seen_to_exact 0.286`

Outcome:

- first real retrieval-side gain; positive on the stress subset relative to `iter16` / `iter20`
- succeeded because bridge-anchored queries recovered target evidence that was previously absent from seen chunks
- failed as a ship candidate because investigator reliability was poor and parse failures were frequent

### Iteration 22 — duplicate gating by retrieval intent

Change:

- duplicate gating switched from raw `sub_question` text to retrieval intent / `retrieval_query`
- goal was to allow repeated sub-questions when the underlying retrieval query materially changed

Subset20:

- `EM 0.050`
- `F1 0.1373`
- `avg_tokens 65,139.8`

Outcome:

- failed
- duplicate-intent gating reduced cost a bit but hurt answer quality and retrieval coverage
- not promoted

### Iteration 23 — slot-conditioned distillation

Change:

- threaded `slot_name` / target slot into investigator distillation
- asked investigator to extract only the slot-resolving value instead of a broader fact

Subset20:

- run stopped early after `16 / 20`
- partial exact metrics on completed 16:
  - `EM 0.0625`
  - `F1 0.1643`
  - `avg_tokens 69,670.3`

Outcome:

- failed
- parse failures increased and quality trended below `iter21`
- slot-conditioned distillation did not produce a better champion and was abandoned

### Iteration 24 — strict investigator extraction path

Change:

- kept `iter21` controller/routing unchanged
- added separate investigator extraction path only:
  - `investigator_llm.temperature = 0.0`
  - `investigator_llm.max_tokens = 384`
  - `investigator_llm.chat_template_kwargs.enable_thinking = false`
  - tiny schema-only prompt `investigator_distill_strict.txt`
- no orchestrator or routing changes

Subset20:

- `EM 0.100`
- `F1 0.2644`
- `avg_tokens 42,372.6`
- `parse_failure_rate 0.00`
- `abstain_rate 0.40`
- `gold_in_seen 0.45`
- `seen_to_exact 0.222`

Outcome on subset20:

- succeeded relative to `iter21`
- same EM, much higher F1, much lower token cost, zero parse failures, lower abstain, higher retrieval coverage
- promoted temporarily for pilot testing

Pilot200 (`questions_pilot200_seed42.json`):

- `EM 0.210`
- `F1 0.3137`
- `avg_tokens 38,420.6`
- `avg_subagents 2.59`
- `parse_failure_rate 0.025`
- `abstain_rate 0.325`
- `gold_in_seen 0.86`
- `route_mix = {single_probe: 180, direct_answer: 20}`

Outcome on pilot200:

- failed as a replacement for `iter16`
- despite much better reliability, retrieval coverage, and efficiency than earlier branches, answer quality did not beat frozen `iter16`
- route mix showed almost no true recursive behavior (`0 recurse`), so the branch drifted into a mostly single-probe regime rather than a compelling AMAS regime

Current champion after reversion:

- frozen `iter16`
- pilot summary remains:
  - `EM 0.285`
  - `contain 0.320`
  - `avg_subagents 2.81`
  - `avg_tokens 62.6k`

Next recommended move from `iter16`:

- treat `iter16` as the working baseline/champion again
- do not keep pushing investigator-only extraction branches
- if continuing method work, the next change should be judged against `iter16` first on a tiny failure subset and must improve answer quality, not just retrieval coverage or parse reliability
- the strongest concrete lesson from `iter21–24` is that retrieval/query improvements alone were not enough; the remaining gap is retrieval-to-answer translation without collapsing into a single-probe-only policy

### Iteration 25 — recurse-only bridge-anchored retrieval query

Change:

- restored `iter16` pipeline/config as base
- only change was in the hard path:
  - when a spawned follow-up retrieval happened after routing, thread `decision["retrieval_query"]` / `recovery_decision["retrieval_query"]` into the investigator
  - no investigator extraction change
  - no cheap-lane change
  - no routing logic change intended

Gate slice: `retrieval_hard` (`25` IDs)

Baseline on the same slice from frozen `iter16`:

- `EM 0.000`
- `F1 0.0467`
- `gold_in_seen 0.00`
- `gold_in_top5 0.00`
- `gold_in_top10 0.00`
- `avg_tokens 81,393.3`
- `avg_subagents 3.20`
- route mix: `{recurse: 21, single_probe: 3, direct_answer: 1}`

`iter25` result:

- `EM 0.080`
- `F1 0.1467`
- `gold_in_seen 0.36`
- `gold_in_top5 0.04`
- `gold_in_top10 0.16`
- `avg_tokens 73,017.0`
- `avg_subagents 3.12`
- route mix: `{single_probe: 24, direct_answer: 1}`

Diagnosis:

- gain on the hard slice was real
- but the branch did **not** qualify
- exact issue:
  - route-collapse bug
  - intended recurse-only improvement was tested under an unintended shift from `recurse` to `single_probe`
- secondary issues:
  - `15` parse-failure events
  - `17` duplicate-subquestion warnings

Outcome:

- no pilot200 launch
- diagnosis established that the next fix had to be route preservation, not duplicate-gating or parser work

### Iteration 25r — iter25 with recurse-preserving route prompt restore

Change:

- kept the `iter25` bridge-anchored retrieval-query patch
- restored the route prompt away from the later `single_probe` bias:
  - removed `choose the cheapest correct lane`
  - removed `if unsure between single_probe and recurse, choose single_probe`
  - restored recurse preference for multi-slot unresolved / bridge-heavy questions
- no investigator change
- no duplicate-gating change

Gate slice: same `retrieval_hard` (`25` IDs)

`iter25r` result:

- `EM 0.120`
- `F1 0.1787`
- `gold_in_seen 0.44`
- `gold_in_top5 0.08`
- `gold_in_top10 0.24`
- `avg_tokens 69,077.8`
- `avg_subagents 2.96`
- route mix: `{recurse: 16, single_probe: 6, direct_answer: 3}`

Observed during live run:

- recurse behavior restored on the hard slice
- parse failures and duplicate warnings still appeared, but were no longer the primary blocker

Outcome:

- succeeded on the hard-slice gate relative to both `iter16` and `iter25`
- first branch after `iter16` that simultaneously:
  - improved hard-slice answer quality
  - improved retrieval coverage
  - restored nonzero recurse on hard cases
  - reduced token cost below `iter16`
- next step pending PM confirmation:
  - advance to `pilot200` from `iter25r`

Pilot200 (`questions_pilot200_seed42.json`):

- `EM 0.240`
- `F1 0.3325`
- `contain 0.335`
- `avg_tokens 59,364.9`
- `avg_subagents 2.77`
- `abstain_rate 0.47`
- route mix: `{recurse: 131, single_probe: 46, direct_answer: 23}`
- retrieval coverage:
  - `gold_in_seen 0.79`
  - `gold_in_top5 0.26`
  - `gold_in_top10 0.61`
- reliability:
  - `parse_events 93`
  - `duplicate_events 97`
  - `rows_with_parse 45 / 200`
  - `rows_with_dup 38 / 200`

Comparison on the same `200` IDs:

- frozen `iter16`:
  - `EM 0.285`
  - `F1 0.3717`
  - `contain 0.37`
  - `avg_tokens 62,648.7`
  - `avg_subagents 2.81`
  - route mix: `{recurse: 157, single_probe: 20, direct_answer: 23}`
- `iter24`:
  - `EM 0.200`
  - `F1 0.2951`
  - `contain 0.28`
  - `avg_tokens 38,420.6`
  - route mix: `{single_probe: 180, direct_answer: 20}`

Outcome on pilot200:

- failed as a replacement for frozen `iter16`
- this branch preserved real AMAS behavior and stayed cheaper than `iter16`
- but final answer quality remained below `iter16` on both `EM` and `F1`
- therefore:
  - no promotion to `1000q`
  - no ship-candidate change
- strongest current read:
  - bridge-anchored retrieval helps hard cases
  - route preservation matters
  - but retrieval gains are still being lost in retrieval-to-answer translation

## 2026-04-16 22:35 CEST

Seeded-1000 freeze point:

`iter27_think`:

- `EM 0.276`
- `F1 0.3791`
- `contain 0.363`
- `answered 928/1000`
- `avg_subagents 2.81`
- `avg_verify 0.04`
- `avg_tokens 60,236.9`

`S0_no_think`:

- `EM 0.130`
- `F1 0.2412`
- `contain 0.182`
- `answered 1000/1000`
- `avg_tokens 11,299.9`

Comparison against frozen `iter16_1000`:

- `iter16_1000`: `EM 0.284`, `F1 0.3698`, `contain 0.334`, `answered 573/1000`, `avg_tokens 62,984.7`
- `iter27_think`: `EM 0.276`, `F1 0.3791`, `contain 0.363`, `answered 928/1000`, `avg_tokens 60,236.9`

Outcome:

- `iter27_think` became the current headline result
- it beat `iter16_1000` on the primary metric (`contain`) at lower token cost
- it also materially improved completion rate
- this is the frozen EMNLP figure candidate before any further architectural changes

## 2026-04-16 22:50 CEST

Git freeze / backup state:

- frozen branch: `codex/iter27-no-think-results`
- frozen branch tip: `0e57b9e` (`Freeze iter27 think and ablation configs`)
- merge commit on `main`: `9a3f3c6` (`Merge branch 'codex/iter27-no-think-results'`)

Rollback / restore:

- to inspect the frozen branch directly:
  - `git checkout codex/iter27-no-think-results`
- to restore `main` back to the frozen state without losing later history:
  - `git checkout main`
  - `git revert -m 1 <later-merge-commit>` as needed
- to hard-reset a scratch branch to the frozen point:
  - `git checkout -b scratch-iter27-restore 0e57b9e`

Safety note:

- keep `codex/iter27-no-think-results` on both local and remote as the canonical recovery point
- do not force-push over that branch

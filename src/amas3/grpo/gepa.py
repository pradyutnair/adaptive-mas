"""Canonical constrained Role-aware Prompt Evolution.

Design notes:
1. Hard prompt length cap at 800 chars (was 1500-1700)
2. Strong token penalty in variant reward 
3. Reject evolved prompts that worsen answer rate or increase blank count
4. Incremental rule injection instead of full rewrites
5. Max 5 operational rules per role to prevent bloat
6. Token-cost tracking in contrastive analysis
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import dspy

from ..pipeline import AmasResult
from .experience_library import ExperienceEntry, ExperienceLibrary

logger = logging.getLogger(__name__)

AGENT_ROLES = ("planner", "solver", "synth", "orchestrator")

MAX_PROMPT_CHARS = 800  # Hard cap - was 1500-1700 and caused bloat

VARIANT_AXES = (
    "efficiency",
    "thoroughness",
    "risk_sensitivity",
    "error_correction",
    "heuristic_injection",
    "balanced_grounding",
)


@dataclass
class FailureBuffer:
    """Per-agent recent failure memory for HERA RoPE."""
    max_per_role: int = 50
    entries: dict[str, list[dict]] = field(default_factory=lambda: {role: [] for role in AGENT_ROLES})

    def add(self, trajectory: dict) -> None:
        result = trajectory["result"]
        gold = trajectory["gold_answer"]
        for role, reason in identify_failing_agents(result, gold):
            item = dict(trajectory)
            item["failure_reason"] = reason
            self.entries.setdefault(role, []).append(item)
            if len(self.entries[role]) > self.max_per_role:
                self.entries[role] = self.entries[role][-self.max_per_role:]

    def extend(self, trajectories: list[dict]) -> None:
        for trajectory in trajectories:
            self.add(trajectory)

    def trajectories_for(self, role: str, limit: int = 10) -> list[dict]:
        return list(self.entries.get(role, []))[-limit:]

    def reasons_for(self, role: str) -> list[str]:
        reasons = []
        for trajectory in self.entries.get(role, []):
            reason = trajectory.get("failure_reason", "")
            if reason and reason not in reasons:
                reasons.append(reason)
        return reasons


def _pipeline_with_role_prompt(pipeline, role: str, prompt: str):
    """Clone a pipeline with a role-specific evolved prompt, capped at MAX_PROMPT_CHARS."""
    config = replace(pipeline.config)
    role_prompts = dict(getattr(config, "role_prompts", {}) or {})
    role_prompts[role] = prompt[:MAX_PROMPT_CHARS]
    config.role_prompts = role_prompts
    return pipeline.__class__(
        planner_lm=pipeline.planner_lm,
        worker_lm=pipeline.worker_lm,
        synth_lm=pipeline.synth_lm,
        sas_lm=pipeline.sas_lm,
        retriever=pipeline.retriever,
        config=config,
    )


# ---------------------------------------------------------------------------
# Credit assignment
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return ' '.join(s.split())


def compute_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens or not pred_tokens:
        return 0.0
    pred_counts = {token: pred_tokens.count(token) for token in set(pred_tokens)}
    gold_counts = {token: gold_tokens.count(token) for token in set(gold_tokens)}
    common = sum(min(pred_counts.get(token, 0), gold_counts.get(token, 0)) for token in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
    

def compute_contain(prediction: str, gold: str) -> float:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return 0.0
    return 1.0 if gold_norm in pred_norm else 0.0


def compute_task_reward(em: float, f1: float, contain: float) -> float:
    return 0.45 * contain + 0.35 * f1 + 0.20 * em


def identify_failing_agents(
    result: AmasResult, gold_answer: str
) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    pred = result.answer or ""
    f1 = compute_f1(pred, gold_answer)
    em = int(normalize_answer(pred) == normalize_answer(gold_answer))
    cont = compute_contain(pred, gold_answer)

    if em or cont:
        return []

    if result.sas_collapse and f1 < 0.5 and cont < 0.5:
        failures.append(("orchestrator", "premature_answer"))
    if result.sas_escalated and f1 >= 0.8:
        failures.append(("orchestrator", "unnecessary_escalation"))

    if result.plan_subgoals == 0:
        failures.append(("planner", "bad_decomposition"))
    elif result.plan_subgoals == 1 and result.topology not in ("sas", "orchestrator_answer", "verified_sas"):
        failures.append(("planner", "bad_decomposition"))

    low_conf_count = 0
    hallucinated = False
    for finding in result.findings:
        conf = finding.get("confidence", 0.0)
        status = finding.get("status", "")
        answer = finding.get("answer", "")
        if status in ("no_evidence", "error") or conf < 0.3:
            low_conf_count += 1
        if answer and conf > 0.7 and status == "ok":
            if f1 < 0.3:
                hallucinated = True

    if low_conf_count > 0:
        failures.append(("solver", "low_confidence"))
    if hallucinated:
        failures.append(("solver", "hallucination"))
    if any(f.get("status") == "ok" and f.get("answer") for f in result.findings):
        if f1 < 0.3 and not hallucinated:
            failures.append(("solver", "wrong_extraction"))

    gold_norm = normalize_answer(gold_answer)
    pred_norm = normalize_answer(pred)
    if pred_norm and gold_norm and pred_norm != gold_norm:
        intermediate_answers = [
            normalize_answer(f.get("answer", ""))
            for f in result.findings
            if not f.get("is_final", False) and f.get("answer")
        ]
        if pred_norm in intermediate_answers:
            failures.append(("synth", "bridge_confusion"))
        elif f1 < 0.5:
            failures.append(("synth", "wrong_extraction"))

    if not failures and f1 < 0.5:
        failures.append(("solver", "wrong_extraction"))

    return failures


# ---------------------------------------------------------------------------
# Prompt variant generation (concise)
# ---------------------------------------------------------------------------

VARIANT_TEMPLATES = {
    "efficiency": (
        "Minimize retrieval calls and token usage. Skip unnecessary steps. "
        "Prefer concise answers from clear evidence."
    ),
    "thoroughness": (
        "Verify every answer against source text. Match wh-target type exactly. "
        "Cross-reference chunks before committing."
    ),
    "risk_sensitivity": (
        "Verify or escalate for comparison, temporal, numeric, conflicting, or ambiguous evidence. "
        "Use shortcuts only when the final target is explicit."
    ),
    "error_correction": (
        "Identify the likely prior-step failure, fix bridge/entity drift, and retry only the smallest missing evidence step."
    ),
    "heuristic_injection": (
        "Bridge: final answer is not the intermediate entity. Comparison: retrieve each attribute separately. Intersection: preserve property type."
    ),
    "balanced_grounding": (
        "Prefer directly supported answer spans. Avoid bridge entities as final answers. "
        "Choose best canonical span from evidence."
    ),
}


VARIANT_GENERATION_PROMPT = """Generate role-aware prompt variants for DSPy GEPA-style replay.

Agent role: {agent_role}
Current prompt: {current_prompt}
Observed failure reasons: {failure_reasons}
Failure examples:
{failure_examples}

Create 3 to 5 concise candidate prompts. They must:
- stay under {max_chars} characters each
- preserve the role and tool behavior
- improve Contain/F1 while reducing unnecessary retrieval/tokens
- avoid answer-discouraging language
- be meaningfully different semantic strategies, not fixed templates

Return ONLY JSON:
[{{"axis":"<short semantic axis>","variant_prompt":"<full prompt>","rationale":"<why this variant addresses failures>"}}]"""


def _failure_examples_text(failed_trajectories: list[dict] | None, limit: int = 5) -> str:
    rows = []
    for traj in (failed_trajectories or [])[:limit]:
        result = traj.get("result")
        answer = getattr(result, "answer", "") if result is not None else ""
        topology = getattr(result, "topology", "") if result is not None else ""
        tokens = getattr(result, "total_tokens", "") if result is not None else ""
        rows.append(
            "Q: {q}\nGold: {gold}\nBad answer: {ans}\nTopology: {topo}; tokens={tok}".format(
                q=str(traj.get("question", ""))[:180],
                gold=str(traj.get("gold_answer", ""))[:80],
                ans=str(answer)[:80],
                topo=str(topology)[:80],
                tok=tokens,
            )
        )
    return "\n---\n".join(rows) if rows else "(none)"


def _parse_variant_array(text: str) -> list[dict]:
    import re
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def generate_prompt_variants(
    reflection_lm: dspy.LM | None,
    agent_role: str,
    current_prompt: str,
    failure_reasons: list[str],
    failed_trajectories: list[dict] | None = None,
) -> list[dict]:
    failure_context = ", ".join(failure_reasons) if failure_reasons else "general underperformance"
    if reflection_lm is not None:
        prompt = VARIANT_GENERATION_PROMPT.format(
            agent_role=agent_role,
            current_prompt=current_prompt[:500],
            failure_reasons=failure_context,
            failure_examples=_failure_examples_text(failed_trajectories, limit=5),
            max_chars=MAX_PROMPT_CHARS,
        )
        try:
            response = reflection_lm(prompt)
            text = response[0] if isinstance(response, list) else str(response)
            variants = []
            for idx, raw in enumerate(_parse_variant_array(text)[:5], start=1):
                if not isinstance(raw, dict):
                    continue
                variant_prompt = str(raw.get("variant_prompt", "") or "").strip()[:MAX_PROMPT_CHARS]
                if not variant_prompt:
                    continue
                variants.append({
                    "axis": str(raw.get("axis", f"semantic_{idx}"))[:40],
                    "variant_prompt": variant_prompt,
                    "rationale": str(raw.get("rationale", "reflection-generated variant"))[:160],
                    "generated_by": "reflection_lm",
                })
            if variants:
                return variants
        except Exception as exc:
            logger.warning("GEPA variant generation failed for %s; using fallback variants: %s", agent_role, exc)

    variants = []
    for axis in VARIANT_AXES:
        modifier = VARIANT_TEMPLATES[axis]
        # Prepend directive concisely - keep under MAX_PROMPT_CHARS
        variant_prompt = (
            f"[{axis.upper()}] Fix: {failure_context}\n"
            f"{modifier}\n\n"
            f"{current_prompt}"
        )[:MAX_PROMPT_CHARS]  # Hard cap
        
        variants.append({
            "axis": axis,
            "variant_prompt": variant_prompt,
            "rationale": f"{agent_role}/{axis}: addressing {failure_context}",
            "generated_by": "fallback_template",
        })

    return variants


# ---------------------------------------------------------------------------
# Contrastive analysis (with token-cost emphasis)
# ---------------------------------------------------------------------------

CONTRASTIVE_PROMPT = """Analyze prompt variant results for a multi-hop QA agent.
Focus on quality AND token efficiency. Keep rules concise.

Agent: {agent_role}
Original prompt (abbreviated): {original_prompt_snippet}

Variant results:
{variant_results_json}

Extract:
1. OPERATIONAL RULES: max 3 concrete, short instructions (under 60 chars each)
2. BEHAVIORAL PRINCIPLES: max 2 strategic generalizations (under 80 chars each)
3. Rules MUST address token efficiency (prefer fewer retrievals, shorter outputs)
4. Preserve or improve Contain first, then F1/EM
5. Do NOT add rules encouraging blank answers

Return STRICT JSON:
{{"operational_rules":[{{"rule":"<under 60 chars>","derived_from":"<axis>"}}],"behavioral_principles":[{{"principle":"<under 80 chars>","derived_from":"<axis>"}}],"best_axis":"<axis>","analysis_summary":"<1 sentence>"}}"""


def run_contrastive_analysis(
    reflection_lm: dspy.LM,
    agent_role: str,
    original_prompt: str,
    variant_results: list[dict],
) -> dict:
    prompt_snippet = original_prompt[:300]
    results_json = json.dumps(variant_results, indent=2, ensure_ascii=False)

    user_content = CONTRASTIVE_PROMPT.format(
        agent_role=agent_role,
        original_prompt_snippet=prompt_snippet,
        variant_results_json=results_json,
    )

    response = reflection_lm(user_content)
    text = response[0] if isinstance(response, list) else str(response)

    import re
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            parsed = {}
    else:
        parsed = {}

    # Enforce conciseness limits
    rules = parsed.get("operational_rules", [])[:3]  # Max 3 rules
    for r in rules:
        if isinstance(r, dict) and "rule" in r:
            r["rule"] = r["rule"][:60]
    
    principles = parsed.get("behavioral_principles", [])[:2]  # Max 2 principles
    for p in principles:
        if isinstance(p, dict) and "principle" in p:
            p["principle"] = p["principle"][:80]

    return {
        "operational_rules": rules,
        "behavioral_principles": principles,
        "best_axis": parsed.get("best_axis", ""),
        "analysis_summary": parsed.get("analysis_summary", ""),
    }


# ---------------------------------------------------------------------------
# Prompt consolidation (length-constrained)
# ---------------------------------------------------------------------------

CONSOLIDATION_PROMPT = """Consolidate an evolved prompt for a multi-hop QA agent.

Agent: {agent_role}
Current prompt: {current_prompt}

New rules to integrate (max {max_rules} total):
{rules_json}

New principles:
{principles_json}

HARD REQUIREMENTS:
1. Output prompt MUST be under {max_chars} characters total
2. Preserve core role definition
3. Add rules as numbered bullet points
4. Remove redundant instructions
5. Do NOT add instructions that encourage blank/empty answers
6. Prefer SHORT, DIRECT instructions over long explanations

Output ONLY the consolidated prompt text. No JSON, no explanation."""


class PromptIntegrationSignature(dspy.Signature):
    """Integrate role-specific GEPA feedback into a concise agent prompt.

Preserve the role, keep the prompt actionable, and stay under the character
limit. Do not add answer-discouraging instructions.
"""
    agent_role: str = dspy.InputField()
    current_prompt: str = dspy.InputField()
    rules_json: str = dspy.InputField()
    principles_json: str = dspy.InputField()
    failure_context: str = dspy.InputField()
    max_chars: int = dspy.InputField()
    updated_prompt: str = dspy.OutputField()


class PromptIntegrator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.integrate = dspy.Predict(PromptIntegrationSignature)

    def forward(
        self,
        agent_role: str,
        current_prompt: str,
        rules_json: str,
        principles_json: str,
        failure_context: str,
        max_chars: int,
    ):
        return self.integrate(
            agent_role=agent_role,
            current_prompt=current_prompt,
            rules_json=rules_json,
            principles_json=principles_json,
            failure_context=failure_context,
            max_chars=max_chars,
        )


def _prompt_quality_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    try:
        from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
    except Exception:
        ScoreWithFeedback = None
    prompt = str(getattr(pred, "updated_prompt", "") or "").strip()
    max_chars = int(getattr(example, "max_chars", MAX_PROMPT_CHARS))
    score = 1.0
    feedback = []
    if not prompt:
        score -= 0.8
        feedback.append("Output was empty; produce a usable role prompt.")
    if len(prompt) > max_chars:
        score -= min(0.5, (len(prompt) - max_chars) / max(max_chars, 1))
        feedback.append(f"Prompt exceeds {max_chars} chars; compress it.")
    banned = ["do not answer", "refuse to", "always escalate", "return empty"]
    lower = prompt.lower()
    if any(x in lower for x in banned):
        score -= 0.5
        feedback.append("Remove answer-discouraging language.")
    role = str(getattr(example, "agent_role", "")).lower()
    if role and role not in lower:
        score -= 0.05
        feedback.append("Mention or preserve the agent role.")
    if "rule" not in lower and "must" not in lower and "prefer" not in lower:
        score -= 0.1
        feedback.append("Make the prompt operational and actionable.")
    failure_context = str(getattr(example, "failure_context", "") or "").lower()
    if failure_context and not any(tok in lower for tok in failure_context.replace("_", " ").split()[:8]):
        score -= 0.05
        feedback.append("Tie the prompt to observed failure context.")
    score = max(0.0, min(1.0, score))
    message = " ".join(feedback) or "Prompt is concise, role-preserving, and actionable."
    if ScoreWithFeedback:
        return ScoreWithFeedback(score=score, feedback=message)
    return score


def _consolidate_prompt_dspy_gepa(
    reflection_lm: dspy.LM,
    agent_role: str,
    current_prompt: str,
    rules_json: str,
    principles_json: str,
    failure_examples: list[dict] | None = None,
) -> str:
    student = PromptIntegrator()
    examples: list[dspy.Example] = []
    contexts = []
    for traj in (failure_examples or [])[:5]:
        result = traj.get("result")
        answer = getattr(result, "answer", "") if result is not None else ""
        topology = getattr(result, "topology", "") if result is not None else ""
        contexts.append(
            f"question={str(traj.get('question', ''))[:120]}; "
            f"gold={str(traj.get('gold_answer', ''))[:60]}; "
            f"bad_answer={str(answer)[:60]}; topology={str(topology)[:60]}"
        )
    if not contexts:
        contexts = ["general role failure; preserve answer quality while reducing unnecessary tokens"]
    for context in contexts:
        examples.append(
            dspy.Example(
                agent_role=agent_role,
                current_prompt=current_prompt[:500],
                rules_json=rules_json,
                principles_json=principles_json,
                failure_context=context,
                max_chars=MAX_PROMPT_CHARS,
            ).with_inputs("agent_role", "current_prompt", "rules_json", "principles_json", "failure_context", "max_chars")
        )
    if len(examples) >= 3:
        trainset = examples[:2]
        valset = examples[2:5]
    elif len(examples) == 2:
        trainset = examples[:1]
        valset = examples[1:]
    else:
        trainset = examples
        valset = examples
    optimizer = dspy.GEPA(
        metric=_prompt_quality_feedback,
        reflection_lm=reflection_lm,
        max_metric_calls=int(os.environ.get("AMAS_DSPY_GEPA_MAX_METRIC_CALLS", "6")),
        reflection_minibatch_size=1,
        candidate_selection_strategy="current_best",
        use_merge=True,
        max_merge_invocations=1,
        num_threads=1,
        track_stats=False,
    )
    with dspy.context(lm=reflection_lm):
        optimized = optimizer.compile(student, trainset=trainset, valset=valset)
        pred = optimized(
            agent_role=agent_role,
            current_prompt=current_prompt[:500],
            rules_json=rules_json,
            principles_json=principles_json,
            failure_context=contexts[0],
            max_chars=MAX_PROMPT_CHARS,
        )
    return str(getattr(pred, "updated_prompt", "") or "").strip()[:MAX_PROMPT_CHARS]


def consolidate_prompt(
    reflection_lm: dspy.LM,
    agent_role: str,
    current_prompt: str,
    operational_rules: list,
    behavioral_principles: list,
    max_rules: int = 5,
    failure_examples: list[dict] | None = None,
) -> str:
    rules_trimmed = operational_rules[:max_rules]
    rules_json = json.dumps(rules_trimmed, indent=2, ensure_ascii=False)
    principles_json = json.dumps(behavioral_principles, indent=2, ensure_ascii=False)

    if os.environ.get("AMAS_DISABLE_DSPY_GEPA", "0") != "1":
        try:
            consolidated = _consolidate_prompt_dspy_gepa(
                reflection_lm=reflection_lm,
                agent_role=agent_role,
                current_prompt=current_prompt,
                rules_json=rules_json,
                principles_json=principles_json,
                failure_examples=failure_examples,
            )
            if consolidated:
                return consolidated[:MAX_PROMPT_CHARS]
        except Exception as exc:
            logger.warning("DSPy GEPA consolidation failed for %s; falling back: %s", agent_role, exc)

    user_content = CONSOLIDATION_PROMPT.format(
        agent_role=agent_role,
        current_prompt=current_prompt[:500],  # Cap input too
        max_rules=max_rules,
        max_chars=MAX_PROMPT_CHARS,
        rules_json=rules_json,
        principles_json=principles_json,
    )

    response = reflection_lm(user_content)
    text = response[0] if isinstance(response, list) else str(response)

    consolidated = text.strip()
    if not consolidated:
        return current_prompt[:MAX_PROMPT_CHARS]
    
    # Hard cap enforcement
    return consolidated[:MAX_PROMPT_CHARS]


# ---------------------------------------------------------------------------
# Default prompts (concise versions)
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = {
    "planner": (
        "Decompose multi-hop questions into atomic single-hop sub-questions. "
        "Each answerable by ONE retrieval + ONE extraction. "
        "Preserve the original wh-target category in the final subgoal."
    ),
    "solver": (
        "Extract a SHORT answer span (1-6 words) VERBATIM from evidence chunks. "
        "Match the expected answer type. Return empty if no evidence supports it."
    ),
    "synth": (
        "Emit the SHORT answer span (1-6 words) that DIRECTLY answers the original "
        "question's wh-target. Never return a bridge entity as final answer."
    ),
    "orchestrator": (
        "Route to SAS shortcut for simple factoid queries with clear evidence. "
        "Escalate to full MAS for bridge/comparison/multi-hop questions."
    ),
}


# ---------------------------------------------------------------------------
# Main GEPA loop with quality gates
# ---------------------------------------------------------------------------

def run_gepa_epoch(
    reflection_lm: dspy.LM,
    pipeline,
    retriever,
    failed_trajectories: list[dict],
    experience_library: ExperienceLibrary,
    failure_buffer: FailureBuffer | None = None,
) -> dict:
    """Run one GEPA epoch with quality gates and token constraints."""
    import asyncio

    if failure_buffer is None:
        failure_buffer = FailureBuffer()
        failure_buffer.extend(failed_trajectories)

    evolved_prompts = dict(DEFAULT_PROMPTS)
    all_rules: list[dict] = []
    all_principles: list[dict] = []

    for role in AGENT_ROLES:
        reasons = failure_buffer.reasons_for(role)
        if not reasons:
            continue

        current_prompt = evolved_prompts[role]
        trajs = failure_buffer.trajectories_for(role, limit=10)

        logger.info(
            "GEPA: evolving %s (failures: %s, trajs: %d)",
            role, reasons, len(trajs),
        )

        variants = generate_prompt_variants(reflection_lm, role, current_prompt, reasons, trajs)
        variant_results = []
        replay_cap = max(1, int(os.environ.get("AMAS_GEPA_REPLAY_PER_VARIANT", "5")))

        def evaluate_prompt(prompt: str, axis: str) -> dict:
            em_scores = []
            f1_scores = []
            contain_scores = []
            token_counts = []
            answered_count = 0
            variant_pipeline = _pipeline_with_role_prompt(pipeline, role, prompt)
            for traj in trajs[:replay_cap]:
                question = traj["question"]
                qid = traj.get("qid", "")
                gold = traj["gold_answer"]
                loop = asyncio.new_event_loop()
                try:
                    re_result = loop.run_until_complete(variant_pipeline.run(question, qid))
                except Exception as exc:
                    logger.warning("GEPA replay failed for %s/%s: %s", role, axis, exc)
                    em_scores.append(0)
                    f1_scores.append(0.0)
                    contain_scores.append(0.0)
                    token_counts.append(20000)
                    continue
                finally:
                    loop.close()
                em_scores.append(int(normalize_answer(re_result.answer) == normalize_answer(gold)))
                f1_scores.append(compute_f1(re_result.answer, gold))
                contain_scores.append(compute_contain(re_result.answer, gold))
                token_counts.append(re_result.total_tokens)
                if (re_result.answer or "").strip():
                    answered_count += 1
            avg_em = sum(em_scores) / max(len(em_scores), 1)
            avg_f1 = sum(f1_scores) / max(len(f1_scores), 1)
            avg_contain = sum(contain_scores) / max(len(contain_scores), 1)
            avg_tokens = sum(token_counts) / max(len(token_counts), 1)
            answer_rate = answered_count / max(len(f1_scores), 1)
            blank_count = len(f1_scores) - answered_count
            reward = (compute_task_reward(avg_em, avg_f1, avg_contain) + 0.1 * answer_rate
                      - (avg_tokens / 15000) - 0.12 * blank_count)
            return {
                "axis": axis,
                "em": round(avg_em, 4),
                "f1": round(avg_f1, 4),
                "contain": round(avg_contain, 4),
                "answer_rate": round(answer_rate, 4),
                "blank_count": blank_count,
                "tokens": round(avg_tokens, 1),
                "reward": round(reward, 4),
                "prompt_chars": len(prompt),
            }

        baseline_result = evaluate_prompt(current_prompt, "baseline")
        for variant in variants:
            variant_results.append(evaluate_prompt(variant["variant_prompt"], variant["axis"]))

        # Contrastive analysis
        analysis = run_contrastive_analysis(
            reflection_lm, role, current_prompt, variant_results
        )
        rules = analysis.get("operational_rules", [])
        principles = analysis.get("behavioral_principles", [])
        all_rules.extend(rules)
        all_principles.extend(principles)

        # Consolidate prompt
        new_prompt = consolidate_prompt(
            reflection_lm, role, current_prompt, rules, principles,
            max_rules=5,  # Hard cap on rules per role
            failure_examples=trajs[:5],
        )

        # QUALITY GATE: only accept if prompt is within length limit
        if len(new_prompt) > MAX_PROMPT_CHARS:
            logger.warning("GEPA: %s prompt too long (%d > %d), truncating", 
                          role, len(new_prompt), MAX_PROMPT_CHARS)
            new_prompt = new_prompt[:MAX_PROMPT_CHARS]
        
        # QUALITY GATE: verify the prompt doesn't discourage answering
        prompt_lower = new_prompt.lower()
        ban_phrases = ["do not answer", "refuse to", "return empty", "always escalate"]
        if any(phrase in prompt_lower for phrase in ban_phrases):
            logger.warning("GEPA: %s prompt contains answer-discouraging language, keeping original", role)
            new_prompt = current_prompt[:MAX_PROMPT_CHARS]

        best_variant = max(variant_results, key=lambda r: r.get("reward", -999)) if variant_results else baseline_result
        if best_variant.get("reward", -999) < baseline_result.get("reward", -999) + 0.02:
            logger.info(
                "GEPA: rejecting %s update; best variant reward %.3f did not beat baseline %.3f",
                role, best_variant.get("reward", 0.0), baseline_result.get("reward", 0.0),
            )
            new_prompt = current_prompt[:MAX_PROMPT_CHARS]
            rules = []
            principles = []

        evolved_prompts[role] = new_prompt

        # Add concise experience entries from rules
        for i, rule in enumerate(rules[:2]):  # Max 2 entries per role
            entry = ExperienceEntry(
                id=f"gepa_{role}_{i}",
                profile=f"{role}_rule",
                insight=rule.get("rule", "")[:100],
                utility=0.6,
                target_roles=(role,),
                applies_when=f"Agent {role} shows: {', '.join(reasons[:2])}",
                avoid_when="",
            )
            experience_library.add(entry)

        logger.info("GEPA: evolved %s (%d chars, %d rules, %d principles)", 
                    role, len(new_prompt), len(rules), len(principles))

    return {
        "evolved_prompts": evolved_prompts,
        "operational_rules": all_rules,
        "behavioral_principles": all_principles,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_evolved_prompts(prompts: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")


def load_evolved_prompts(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_PROMPTS)
    return json.loads(p.read_text(encoding="utf-8"))

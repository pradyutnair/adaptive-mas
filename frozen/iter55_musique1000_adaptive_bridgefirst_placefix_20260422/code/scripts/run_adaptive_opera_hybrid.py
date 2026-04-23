#!/usr/bin/env python3
"""Adaptive wrapper: typed direct-probe for easy questions, OPERA for hard ones."""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from arag.core.config import Config  # noqa: E402
from adaptive_sage.fact_memory import FactMemory  # noqa: E402
from adaptive_sage.pipeline import AdaptiveRecursivePipeline  # noqa: E402
from adaptive_sage.types import PipelineResult, StepTrace  # noqa: E402

_NAME_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:[-'][A-Z][A-Za-z]+)?(?:\s+[A-Z][A-Za-z]+(?:[-'][A-Z][A-Za-z]+)?)*\b")


def _strip_rhs(text: str) -> str:
    text = text.strip().strip(".")
    for sep in (" is ", " was ", " are ", " were "):
        if sep in text:
            text = text.split(sep, 1)[1].strip(" .")
            break
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" .")


def _comma_parts(text: str) -> list[str]:
    return [part.strip(" .") for part in text.split(",") if part.strip(" .")]


def _extract_year_or_date(text: str) -> str:
    for pattern in (
        r"\b((?:18|19|20)\d{2})\b",
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
    ):
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return text.strip()


def _extract_numberish(text: str) -> str:
    for pattern in (
        r"\b(\d[\d,]*(?:\s*(?:to|-)\s*\d[\d,]*)?)\b",
        r"\b(\d+\.?\d*)\b",
    ):
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return text.strip()


def _extract_locationish(text: str) -> str:
    rhs = _strip_rhs(text)
    for pattern in (
        r"\bin\s+([^.;]+)",
        r"\bat\s+([^.;]+)",
        r"\bfrom\s+([^.;]+)",
    ):
        m = re.search(pattern, rhs, flags=re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            value = re.sub(r"^\d+\s+", "", value).strip()
            parts = [p.strip() for p in value.split(",") if p.strip()]
            if parts:
                return ", ".join(parts[:2])
            return value
    parts = _comma_parts(rhs)
    if parts:
        return ", ".join(parts[:2])
    caps = _NAME_RE.findall(rhs)
    return caps[0].strip() if caps else rhs


def _extract_entityish(text: str) -> str:
    rhs = _strip_rhs(text)
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", rhs)
    if quoted:
        return quoted[0].strip()
    if " and " in rhs:
        rhs = rhs.split(" and ", 1)[0].strip()
    parts = _comma_parts(rhs)
    if parts:
        rhs = parts[0]
    names = _NAME_RE.findall(rhs)
    if names:
        return names[0].strip()
    return rhs or text.strip().strip(".")


def _extract_placeholder_value(answer: str, info_type: str) -> str:
    kind = info_type.strip().lower()
    if kind in {"date", "year", "time"}:
        return _extract_year_or_date(answer)
    if kind in {"number", "count", "amount"}:
        return _extract_numberish(answer)
    if kind == "city":
        value = _extract_locationish(answer)
        parts = _comma_parts(value)
        return parts[0] if parts else value
    if kind in {"country", "state", "province", "region"}:
        value = _extract_locationish(answer)
        parts = _comma_parts(value)
        return parts[-1] if parts else value
    if kind in {"location", "place"}:
        return _extract_locationish(answer)
    if kind in {"person", "entity", "organization", "thing"}:
        return _extract_entityish(answer)
    if kind in {"answer", "general"}:
        if re.search(r"\b((?:18|19|20)\d{2})\b", answer):
            return _extract_year_or_date(answer)
        if re.search(r"\d", answer):
            return _extract_numberish(answer)
        return _extract_entityish(answer)
    return _extract_entityish(answer)


def _repair_plan(question: str, plan: list[dict], expected_answer_type: str) -> list[dict]:
    repaired = [dict(step) for step in plan]
    i = 0
    while i < len(repaired):
        step = repaired[i]
        sq = str(step.get("sub_question", "")).strip()
        identity = re.match(r"^(who|what)\s+is\s+([A-Z][^?]+)\??$", sq, flags=re.IGNORECASE)
        if identity and i + 1 < len(repaired):
            subject = identity.group(2).strip()
            next_step = dict(repaired[i + 1])
            next_sq = str(next_step.get("sub_question", "")).strip()
            if f"from step {step.get('step_id')}" in next_sq:
                next_sq = re.sub(r"\[[^]]+\s+from\s+step\s+%s\]" % step.get("step_id"), subject, next_sq)
                step["sub_question"] = next_sq
                step["goal"] = next_step.get("goal", step.get("goal", ""))
                repaired[i] = step
                removed_id = next_step.get("step_id")
                repaired.pop(i + 1)
                for later in repaired[i + 1:]:
                    deps = [step.get("step_id") if dep == removed_id else dep for dep in later.get("dependencies", [])]
                    later["dependencies"] = deps
                    later["sub_question"] = re.sub(
                        r"\[[^]]+\s+from\s+step\s+%s\]" % removed_id,
                        f"[entity from step {step.get('step_id')}]",
                        str(later.get("sub_question", "")),
                    )
                continue
        yn_loc = re.match(r"^Is\s+(.+?)\s+the location of\s+(.+?)\??$", sq, flags=re.IGNORECASE)
        if yn_loc:
            subject = yn_loc.group(2).strip()
            step["sub_question"] = f"What is the location of {subject}?"
            step["goal"] = f"Find the location of {subject}."
            repaired[i] = step
        i += 1

    return repaired


def _needs_bridge_first(question: str) -> bool:
    q = f" {question.lower()} "
    relative_markers = (
        " who ",
        " whose ",
        " where ",
        " which ",
        " that ",
    )
    if any(marker in q for marker in relative_markers):
        return True
    if q.count(" of ") >= 2:
        return True
    if " the person " in q or " the city " in q or " the state " in q or " the country " in q:
        return True
    return False


def _build_planner_messages(
    question: str,
    target_profile: str,
    expected_answer_type: str,
    *,
    force_bridge_first: bool,
) -> list[dict]:
    extra_rules = ""
    if force_bridge_first:
        extra_rules = (
            "- This question is multi-hop. A single fused sub-question is invalid.\n"
            "- Step 1 must resolve the hidden bridge entity/location/person/date from the relative clause.\n"
            "- Later steps must use placeholder syntax like [entity from step 1] instead of repeating the whole original question.\n"
            "- Do not ask the final relation until the bridge entity has been resolved in an earlier step.\n"
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a strategic planning agent for multi-hop QA. "
                "Respond with one JSON object only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Decompose the question into a short sequential plan.\n\n"
                "Output ONLY a JSON object:\n"
                "{"
                "\"sub_questions\": ["
                "{\"step_id\": 1, \"sub_question\": \"...\", \"goal\": \"...\", \"dependencies\": [], \"expected_info_type\": \"entity|person|location|date|year|number|other\"}"
                "]"
                "}\n\n"
                "Rules:\n"
                "- Use placeholder syntax for dependencies: [entity from step 1], [location from step 2], [year from step 1].\n"
                "- Preserve the exact unresolved relation from the original question.\n"
                "- Do not swap the asked target for a nearby proxy relation.\n"
                "- The final step must ask for the final target slot, not a bridge entity.\n"
                "- Never ask trivial biography/definition questions about an already named entity, such as 'Who is X?' or 'What is X?'. Ask directly for the missing relation instead.\n"
                "- Never use yes/no verification steps like 'Is X the location of Y?'. Ask directly for the missing fact.\n"
                "- Keep the plan to 2-4 steps.\n"
                "- Each step must be answerable with one retrieval step.\n"
                "- If a step returns a set or multiple candidates, the next step must disambiguate to the exact asked relation.\n"
                f"{extra_rules}\n"
                f"Question: {question}\n"
                f"Target profile: {target_profile}\n"
                f"Expected final answer type: {expected_answer_type}\n"
            ),
        },
    ]


def _parse_plan(parsed: dict, expected_answer_type: str, question: str) -> list[dict]:
    raw = parsed.get("sub_questions", [])
    plan = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue
            sub_question = str(item.get("sub_question", "")).strip()
            if not sub_question:
                continue
            deps = item.get("dependencies", [])
            if not isinstance(deps, list):
                deps = []
            info_type = str(item.get("expected_info_type", "")).strip().lower() or "general"
            plan.append({
                "step_id": idx,
                "sub_question": sub_question,
                "goal": str(item.get("goal", "")).strip() or f"Collect information for: {sub_question}",
                "dependencies": [int(d) for d in deps if str(d).isdigit()],
                "expected_info_type": info_type,
            })
            if len(plan) >= 4:
                break
    if not plan:
        plan = [{
            "step_id": 1,
            "sub_question": question,
            "goal": f"Collect information for: {question}",
            "dependencies": [],
            "expected_info_type": expected_answer_type.replace("/", "_"),
        }]
    return _repair_plan(question, plan, expected_answer_type)


def _patch_opera_placeholder_resolution(opera: object) -> None:
    def _resolve_placeholders(sub_question: str, previous_results: dict[int, str]) -> str:
        resolved = sub_question
        matches = re.findall(r"\[([^]]+) from step (\d+)\]", sub_question)
        for info_type, step_id_text in matches:
            step_id = int(step_id_text)
            answer = previous_results.get(step_id)
            if not answer:
                continue
            placeholder = f"[{info_type} from step {step_id}]"
            resolved = resolved.replace(placeholder, _extract_placeholder_value(answer, info_type))
        return resolved

    opera._resolve_placeholders = _resolve_placeholders  # type: ignore[attr-defined]


def _is_direct_probe_safe(question: str, route: dict, expected_answer_type: str, threshold: float) -> bool:
    if str(route.get("action", "")).strip().lower() != "single_probe":
        return False
    if float(route.get("confidence", 0.0)) < max(threshold, 0.9):
        return False
    q = f" {question.lower()} "
    if q.count(" of ") >= 2:
        return False
    if any(marker in q for marker in (" who ", " whose ", " where ", " after ", " before ", " same ", " one that ", " one who ")):
        return False
    if expected_answer_type not in {"date/year", "number", "yes/no"}:
        return False
    return True


def _result_to_output_dict(
    result: PipelineResult,
    gold_answer: str,
    wallclock_seconds: float,
    mode: str,
) -> dict:
    return {
        "id": result.question_id,
        "question": result.question,
        "answer": result.answer,
        "gold_answer": gold_answer,
        "metadata": {
            "step_trace": [t.to_dict() for t in result.step_trace],
            "num_subagent_calls": result.num_subagent_calls,
            "num_verify_calls": result.num_verify_calls,
            "total_tokens": result.total_tokens,
            "orchestrator_tokens": result.orchestrator_tokens,
            "subagent_tokens": result.subagent_tokens,
            "facts_used": [f.to_dict() for f in result.facts_used],
            "retrieved_doc_ids": result.retrieved_doc_ids,
            "retrieved_docs_total": result.retrieved_docs_total,
            "evidence_capsule_limit": result.evidence_capsule_limit,
            "fact_memory_capacity": result.fact_memory_capacity,
            "duplicate_subquestion_count": result.duplicate_subquestion_count,
            "route_decision": result.route_decision,
            "route_confidence": result.route_confidence,
            "route_draft_answer": result.route_draft_answer,
            "slot_resolution": result.slot_resolution,
            "auto_verify_calls": result.auto_verify_calls,
            "answer_rejection_count": result.answer_rejection_count,
            "wallclock_seconds": round(wallclock_seconds, 3),
            "mode": mode,
        },
    }


def _load_opera(opera_repo: Path):
    if str(opera_repo) not in sys.path:
        sys.path.insert(0, str(opera_repo))
    from run_opera_05mas import OperaCompat  # type: ignore
    return OperaCompat


async def _plan_opera_style(
    pipeline: AdaptiveRecursivePipeline,
    question: str,
) -> tuple[dict, int]:
    expected_answer_type = pipeline._expected_answer_type(question)
    target_profile = pipeline._target_profile(question)
    messages = _build_planner_messages(
        question,
        target_profile,
        expected_answer_type,
        force_bridge_first=False,
    )
    parsed, tokens = await pipeline.orchestrator._call_and_parse_with_usage(messages, temperature=0.1)
    plan = _parse_plan(parsed, expected_answer_type, question)
    if _needs_bridge_first(question) and len(plan) < 2:
        bridge_messages = _build_planner_messages(
            question,
            target_profile,
            expected_answer_type,
            force_bridge_first=True,
        )
        parsed2, tokens2 = await pipeline.orchestrator._call_and_parse_with_usage(bridge_messages, temperature=0.0)
        plan2 = _parse_plan(parsed2, expected_answer_type, question)
        if len(plan2) >= 2:
            plan = plan2
            tokens += tokens2
    return {"reasoning": "Adaptive typed strategic decomposition", "sub_questions": plan}, tokens


async def _try_typed_direct_probe(
    pipeline: AdaptiveRecursivePipeline,
    question: str,
    question_id: str,
) -> PipelineResult | None:
    expected_answer_type = pipeline._expected_answer_type(question)
    target_profile = pipeline._target_profile(question)
    route, route_tokens = await pipeline.orchestrator.route_with_usage(
        question=question,
        target_profile=target_profile,
    )
    if pipeline._should_force_bridge_first_route(question, route):
        route = pipeline._force_bridge_first_route(question, route, target_profile)
    if not _is_direct_probe_safe(question, route, expected_answer_type, pipeline.route_direct_threshold):
        return None

    final_slot = str(route.get("target_slot", "")).strip() or "final_answer"
    slot_state = [{
        "slot_name": final_slot,
        "hint": target_profile,
        "resolved": False,
        "dependency_group": 0,
    }]
    memory = FactMemory.with_strategy(
        capacity=pipeline.fact_memory_capacity,
        strategy=pipeline.fact_memory_strategy,
    )
    step_trace = [
        StepTrace(
            step=0,
            action="route",
            tokens=route_tokens,
            route_decision=route["action"],
            route_confidence=route["confidence"],
            route_draft_answer=route["draft_answer"],
            metadata={
                "expected_answer_type": expected_answer_type,
                "answer_type": route["answer_type"],
                "target_slot": final_slot,
                "retrieval_query": route.get("retrieval_query", ""),
                "adaptive_opera_wrapper": True,
            },
        )
    ]
    capsule, investigate_tokens = await pipeline.investigator.investigate_with_usage(
        sub_question=question,
        goal=str(route.get("goal", "")).strip() or f"Answer the question directly. {target_profile}",
        prior_facts=[],
        retrieval_query=str(route.get("retrieval_query", "")).strip() or question,
        slot_name=final_slot,
        slot_hint=target_profile,
    )
    pipeline._add_fact(memory, capsule, step=1, slot_name=final_slot)
    step_trace.append(
        StepTrace(
            step=1,
            action="spawn",
            sub_question=question,
            fact_added=True,
            tokens=investigate_tokens,
            slot_name=final_slot,
            metadata={
                "goal": str(route.get("goal", "")).strip(),
                "retrieval_query": str(route.get("retrieval_query", "")).strip() or question,
                "slot_expected_answer_type": expected_answer_type,
                "direct_probe": True,
            },
        )
    )
    answer_obj = pipeline._strict_terminal_slot_answer_object(
        facts=memory.get_all(),
        slot_state=slot_state,
        expected_answer_type=expected_answer_type,
    )
    if not answer_obj["answer"]:
        return None
    step_trace.append(
        StepTrace(
            step=2,
            action="answer",
            tokens=0,
            cited_fact_ids=answer_obj["cited_fact_ids"],
            justification_confidence=answer_obj["justification_confidence"],
            metadata={
                "justification": answer_obj["justification"],
                "missing_slot": answer_obj["missing_slot"],
                "fallback_source": answer_obj.get("fallback_source", ""),
                "direct_terminal_slot_answer": True,
            },
        )
    )
    return PipelineResult(
        question_id=question_id,
        question=question,
        answer=answer_obj["answer"],
        step_trace=step_trace,
        num_subagent_calls=1,
        num_verify_calls=0,
        total_tokens=route_tokens + investigate_tokens,
        orchestrator_tokens=route_tokens,
        subagent_tokens=investigate_tokens,
        facts_used=memory.get_all(),
        retrieved_doc_ids=capsule.retrieved_doc_ids,
        retrieved_docs_total=capsule.retrieved_docs_total,
        evidence_capsule_limit=pipeline.investigator.evidence_capsule_limit,
        fact_memory_capacity=pipeline.fact_memory_capacity,
        duplicate_subquestion_count=0,
        route_decision=route["action"],
        route_confidence=route["confidence"],
        route_draft_answer=route["draft_answer"],
        slot_resolution={final_slot: True},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--opera-repo", default="/local/yzheng/pnair/workspace/baseline_repos/OPERA")
    ap.add_argument("--retriever-url", default="http://127.0.0.1:9102")
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    config = Config.from_yaml(args.config)
    config.set("llm.base_url", args.server_url)
    pipeline = AdaptiveRecursivePipeline(config)
    OperaCompat = _load_opera(Path(args.opera_repo))
    opera = OperaCompat(
        model_name=args.model_name,
        base_url=args.server_url,
        api_key=args.api_key,
        retriever_url=args.retriever_url,
        top_k=5,
        thinking=args.thinking,
    )
    _patch_opera_placeholder_resolution(opera)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_file = out_dir / "predictions.jsonl"

    questions = json.loads(Path(args.questions).read_text())
    with preds_file.open("a", encoding="utf-8") as out:
        for item in questions:
            qid = str(item.get("id", ""))
            question = item.get("question", "")
            gold = item.get("answer", "") or (item.get("answers", [""]) or [""])[0]
            start = time.time()

            direct_result = asyncio.run(_try_typed_direct_probe(pipeline, question, qid))
            if direct_result is not None:
                rec = _result_to_output_dict(
                    direct_result,
                    gold_answer=gold,
                    wallclock_seconds=time.time() - start,
                    mode="direct_probe",
                )
            else:
                opera.reset_question_metrics(qid)
                plan, plan_tokens = asyncio.run(_plan_opera_style(pipeline, question))
                opera.question_metrics["total_tokens"] += int(plan_tokens)
                opera.question_metrics["prompt_tokens"] += int(plan_tokens)
                opera.question_metrics["llm_call_count"] += 1
                original_role_plan_agent = opera.role_plan_agent
                opera.role_plan_agent = lambda _q, _plan=plan: _plan  # type: ignore[method-assign]
                answer, traj = opera.answer_question(question)
                opera.role_plan_agent = original_role_plan_agent
                rec = {
                    "id": qid,
                    "question": question,
                    "answer": answer,
                    "gold_answer": gold,
                    "metadata": {
                        "mode": "opera",
                        "wallclock_seconds": round(time.time() - start, 3),
                        "trajectory": traj.to_dict(),
                        **opera.get_question_metrics(),
                    },
                }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()

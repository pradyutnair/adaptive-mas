"""Structured trace formatting for TF-GRPO reflection."""

from __future__ import annotations

import json
from typing import Any

from .contracts import normalize_answer


def build_readable_trace(trajectory: dict[str, Any], findings: list[dict], answer: str, gold: str = "") -> str:
    """Format trajectory into a structured narrative for the TF-GRPO reflection LM."""
    lines: list[str] = []
    step = 0
    while f"tool_name_{step}" in trajectory:
        tool = trajectory[f"tool_name_{step}"]
        args = trajectory.get(f"tool_args_{step}", {})
        obs_raw = str(trajectory.get(f"observation_{step}", ""))
        thought = str(trajectory.get(f"thought_{step}", ""))

        lines.append(f"[Step {step}] THOUGHT: {thought}")

        if tool == "hop":
            q = args.get("question", "")
            eat = args.get("expected_answer_type", "auto")
            lines.append(f"  ACTION: hop(question={q!r}, expected_answer_type={eat!r})")
            parsed = {}
            if obs_raw.startswith("{"):
                parsed = json.loads(obs_raw) if obs_raw else {}
            if parsed:
                lines.append(f"  RESULT: answer={parsed.get('answer','')!r}, confidence={parsed.get('confidence',0)}, "
                             f"chunk={parsed.get('evidence_chunk_id','')}, queries_tried={len(parsed.get('queries_used',[]))}")
            else:
                lines.append(f"  RESULT: {obs_raw[:200]}")

        elif tool == "hop_batch":
            lines.append(f"  ACTION: hop_batch (parallel)")
            results = []
            if obs_raw.startswith("["):
                results = json.loads(obs_raw) if obs_raw else []
            for j, r in enumerate(results):
                lines.append(f"    [{j}] answer={r.get('answer','')!r}, confidence={r.get('confidence',0)}, "
                             f"chunk={r.get('evidence_chunk_id','')}")
            if not results and obs_raw:
                lines.append(f"  ERROR: {obs_raw[:200]}")

        elif tool == "submit":
            accepted = obs_raw == "ACCEPTED"
            lines.append(f"  ACTION: submit(answer={args.get('answer','')!r}, ids={args.get('support_ids','')})")
            lines.append(f"  RESULT: {'ACCEPTED' if accepted else obs_raw[:200]}")

        elif tool == "finish":
            lines.append(f"  ACTION: finish")

        lines.append("")
        step += 1

    finding_answers = [f.get("answer", "") for f in findings]
    confidences = [f.get("confidence", 0) for f in findings]
    low_conf = [f for f in findings if f.get("confidence", 0) < 0.65]
    unique_chunks = {f.get("evidence_chunk_id", "") for f in findings if f.get("evidence_chunk_id")}
    total_retries = sum(max(0, len(f.get("queries_used", [])) - 1) for f in findings)

    lines.append("--- DIAGNOSIS ---")
    lines.append(f"Final answer: {answer!r}")
    if gold:
        em = normalize_answer(answer) == normalize_answer(gold)
        lines.append(f"Gold answer: {gold!r} | Exact match: {em}")
    lines.append(f"Hops: {len(findings)} | Retries: {total_retries} | Unique chunks cited: {len(unique_chunks)}")
    lines.append(f"Confidences: {confidences}")

    if low_conf:
        lines.append(f"LOW-CONFIDENCE hops ({len(low_conf)}): {[f.get('answer','') for f in low_conf]}")
    if len(finding_answers) >= 2 and len(set(finding_answers)) == len(finding_answers):
        lines.append("All hop answers are distinct (no convergence).")
    batch_used = any(trajectory.get(f"tool_name_{i}") == "hop_batch" for i in range(step))
    if not batch_used and len(findings) > 2:
        lines.append("MISSED PARALLELISM: >2 hops done sequentially, hop_batch could have helped.")
    submit_rejections = sum(1 for i in range(step)
                           if trajectory.get(f"tool_name_{i}") == "submit"
                           and trajectory.get(f"observation_{i}", "") != "ACCEPTED")
    if submit_rejections:
        lines.append(f"Submit rejections: {submit_rejections}")

    return "\n".join(lines)


def build_structured_stats(metadata: dict[str, Any], trajectory: dict[str, Any], answer: str, gold: str = "") -> dict[str, Any]:
    """Build enriched stats dict for TF-GRPO."""
    findings = metadata.get("findings", [])
    step = 0
    while f"tool_name_{step}" in trajectory:
        step += 1

    tools_used = [trajectory.get(f"tool_name_{i}", "") for i in range(step)]

    topology = "unknown"
    if "hop_batch" in tools_used:
        topology = "parallel"
    elif tools_used.count("hop") == 1:
        topology = "single_hop"
    elif tools_used.count("hop") == 2:
        topology = "sequential_bridge"
    elif tools_used.count("hop") >= 3:
        topology = "multi_sequential"

    return {
        **metadata,
        "topology": topology,
        "steps": step,
        "tools_sequence": tools_used,
        "low_confidence_hops": sum(1 for f in findings if f.get("confidence", 0) < 0.65),
        "submit_rejections": sum(1 for i in range(step)
                                 if trajectory.get(f"tool_name_{i}") == "submit"
                                 and trajectory.get(f"observation_{i}", "") != "ACCEPTED"),
        "used_hop_batch": "hop_batch" in tools_used,
        "unique_chunks": len({f.get("evidence_chunk_id", "") for f in findings if f.get("evidence_chunk_id")}),
        "answer": answer,
        "gold": gold,
        "em": 1.0 if gold and normalize_answer(answer) == normalize_answer(gold) else 0.0,
    }

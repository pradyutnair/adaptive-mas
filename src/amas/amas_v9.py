#!/usr/bin/env python3
"""
AMAS v9: Adaptive Multi-Agent Search with Router
OPERA-style MAS + difficulty router + answer-type enforcement.
Easy (≤2 hops) → single investigator with iterative retrieval.
Hard (3+ hops) → plan + parallel independent / sequential chained sub-goals.
"""
import argparse
import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import requests
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class SubGoal:
    step_id: int
    question: str
    dependencies: list[int] = field(default_factory=list)
    answer: Optional[str] = None


@dataclass
class Metrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    retrieval_rounds: int = 0
    route_decision: str = ""
    answer_type: str = ""
    num_subgoals: int = 0

    def merge(self, other: "Metrics"):
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.llm_calls += other.llm_calls
        self.retrieval_rounds += other.retrieval_rounds


class AMASv9:
    def __init__(self, model: str, retriever_url: str, top_k: int = 5):
        self.model = model
        self.client = OpenAI()
        self.retriever_url = retriever_url.rstrip("/")
        self.top_k = top_k

    # ── LLM ──────────────────────────────────────────────────────────────

    def _llm(self, prompt: str, metrics: Metrics, max_tokens: int = 512, temperature: float = 0.1) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        u = resp.usage
        metrics.prompt_tokens += u.prompt_tokens
        metrics.completion_tokens += u.completion_tokens
        metrics.total_tokens += u.total_tokens
        metrics.llm_calls += 1
        return (resp.choices[0].message.content or "").strip()

    # ── Retrieval ────────────────────────────────────────────────────────

    def _retrieve(self, query: str, metrics: Metrics, topk_override: int = 0) -> list[dict]:
        k = topk_override or self.top_k
        r = requests.post(
            f"{self.retriever_url}/retrieve",
            json={"queries": [query], "topk": k, "mode": "text"},
            timeout=60,
        )
        r.raise_for_status()
        metrics.retrieval_rounds += 1
        rows = (r.json().get("results") or [[]])[0]
        return [
            {"title": item.get("title", item.get("chunk_id", "")), "content": item.get("text", "")}
            for item in rows
        ]

    def _format_docs(self, docs: list[dict], limit: int = 0) -> str:
        if not docs:
            return "No documents available"
        n = limit or self.top_k
        parts = []
        for i, d in enumerate(docs[:n], 1):
            title = d.get("title", f"Document {i}")
            content = (d.get("content", "") or "").strip()
            parts.append(f"[{i}] {title}: {content}")
        return "\n".join(parts)

    # ── Router ───────────────────────────────────────────────────────────

    def _route(self, question: str, metrics: Metrics) -> tuple[str, str]:
        prompt = f"""Classify this question and determine its expected answer type.

Question: {question}

Return JSON:
{{"difficulty": "easy" or "hard", "answer_type": "<type>", "reasoning": "<1 sentence>"}}

Difficulty rules:
- "easy": The question requires at most 2 hops. It has at most one bridge entity to resolve before finding the answer. Examples: "Where was X born?", "Who directed the film that Y starred in?", "What country is the birthplace of Z in?"
- "hard": The question requires 3 or more hops, with multiple chained bridge entities. Examples: "What is the capital of the country where the director of film X was born?"

answer_type must be one of: date, person, location, count, entity, year, other

Return ONLY the JSON."""
        raw = self._llm(prompt, metrics, max_tokens=100, temperature=0.0)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            obj = json.loads(m.group()) if m else {}
            diff = obj.get("difficulty", "hard")
            atype = obj.get("answer_type", "entity")
        except Exception:
            diff, atype = "hard", "entity"
        return diff, atype

    # ── Planner (OPERA-style) ────────────────────────────────────────────

    def _plan(self, question: str, answer_type: str, metrics: Metrics) -> list[SubGoal]:
        prompt = f"""You are a strategic planning agent. Given a complex multi-hop question, decompose it into a sequence of simpler sub-goals with dependency modeling.

Question: {question}

Please generate a plan with the following JSON format:
[
  {{
    "subgoal_id": 1,
    "subgoal": "First sub-question to answer",
    "dependencies": []
  }},
  {{
    "subgoal_id": 2, 
    "subgoal": "Second sub-question using [entity from step 1]",
    "dependencies": [1]
  }}
]

Requirements:
- Use placeholder mechanism: [entity from step X] for dependencies
- Each subgoal should be answerable with a small set of documents
- Maintain logical flow and clear dependencies
- Sub-goals with NO dependencies on each other should have empty or non-overlapping dependency lists (they can run in parallel)

IMPORTANT: For dependencies, you MUST use placeholders like [entity from step 1], [location from step 2], etc. 
Example: If step 1 finds "Alexander Graham Bell", step 2 should be "Where was [entity from step 1] born?" not "Where was Alexander Graham Bell born?"

The final answer should be of type: {answer_type}

Return ONLY the JSON array, no other text."""
        raw = self._llm(prompt, metrics, max_tokens=512)
        try:
            m = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
            items = json.loads(m.group()) if m else []
            goals = [
                SubGoal(
                    step_id=g.get("subgoal_id", i + 1),
                    question=g.get("subgoal", ""),
                    dependencies=g.get("dependencies", []),
                )
                for i, g in enumerate(items)
            ]
            if goals:
                return goals
        except Exception:
            pass
        return [SubGoal(step_id=1, question=question)]

    # ── Analysis-Answer (OPERA-style) ────────────────────────────────────

    def _analyze(self, sub_question: str, docs: list[dict], metrics: Metrics) -> dict:
        docs_text = self._format_docs(docs)
        prompt = f"""You are an analysis and answering agent. Given a sub-question and retrieved documents, determine if you can answer the question and provide analysis.

Sub-question: {sub_question}

Retrieved Documents: {docs_text}

Please respond in the following JSON format:
{{
  "status": "yes" or "no",
  "answer": "extracted answer if status is yes, empty if no",
  "analysis": "explain why you can/cannot answer based on the provided documents"
}}

Key principles:
- status="yes": Documents contain sufficient information
- status="no": Documents lack necessary information
- analysis: Always explain your reasoning"""
        raw = self._llm(prompt, metrics, max_tokens=512)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(m.group()) if m else {}
            if "status" not in result:
                result["status"] = "no"
            result["answer"] = str(result.get("answer", "") or "")
            if result["status"] == "yes" and not result["answer"]:
                result["status"] = "no"
            if "analysis" not in result:
                result["analysis"] = "No analysis provided"
            return result
        except Exception:
            return {"status": "no", "answer": "", "analysis": "parse error"}

    # ── Rewrite (OPERA-style) ────────────────────────────────────────────

    def _rewrite(self, sub_question: str, failure_info: str, metrics: Metrics) -> str:
        prompt = f"""You are an expert query rewriter for information retrieval.

## Rewrite Task
Original Question: {sub_question}
Failure Reason: {failure_info}

## Instructions
1. Analyze why the current query failed to retrieve relevant information
2. Generate an improved search query using keyword expansion and synonyms
3. Focus on key entities, concepts, and alternative phrasings
4. Keep the rewritten query concise but comprehensive

## Output JSON Format
{{
  "rewritten_query": "improved search query with expanded keywords",
  "strategy": "brief explanation of rewrite approach"
}}

Generate rewrite:"""
        raw = self._llm(prompt, metrics, max_tokens=256)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            obj = json.loads(m.group()) if m else {}
            return obj.get("rewritten_query", sub_question + " detailed information")
        except Exception:
            return sub_question + " more information"

    # ── Placeholder resolution ───────────────────────────────────────────

    @staticmethod
    def _resolve_placeholders(text: str, results: dict[int, str]) -> str:
        resolved = text
        for info_type, step_id in re.findall(r"\[([^\]]+) from step (\d+)\]", text):
            sid = int(step_id)
            if sid in results:
                resolved = resolved.replace(f"[{info_type} from step {step_id}]", results[sid])
        return resolved

    # ── Investigate loop (retrieve → analyze → rewrite) ──────────────────

    def _investigate_loop(self, question: str, metrics: Metrics, max_retries: int = 2) -> dict:
        docs = self._retrieve(question, metrics)
        result = self._analyze(question, docs, metrics)

        retries = 0
        while result["status"] == "no" and retries < max_retries:
            rewritten = self._rewrite(question, result.get("analysis", ""), metrics)
            docs = self._retrieve(rewritten, metrics)
            result = self._analyze(question, docs, metrics)
            retries += 1

        return result

    # ── Easy path: iterative self-directed investigator ──────────────────

    def _easy_investigate(self, question: str, answer_type: str, metrics: Metrics) -> dict:
        """For ≤2-hop questions: iteratively search, accumulate facts, answer."""
        accumulated_facts = []
        current_query = question
        max_rounds = 3

        for rnd in range(max_rounds):
            docs = self._retrieve(current_query, metrics, topk_override=10)
            docs_text = self._format_docs(docs, limit=10)

            facts_str = "; ".join(accumulated_facts) if accumulated_facts else "none yet"
            prompt = f"""You are investigating a question step by step through retrieval.

Original question: {question}
Expected answer type: {answer_type}
Search round: {rnd + 1}/{max_rounds}
Current search query: {current_query}
Facts found so far: {facts_str}

Retrieved Documents:
{docs_text}

If you can answer the original question, return:
{{"done": true, "answer": "<concise answer>", "reasoning": "..."}}

If you found useful info but need to search more, return:
{{"done": false, "new_fact": "<what you learned>", "next_query": "<what to search next>", "reasoning": "..."}}

Return ONLY the JSON."""
            raw = self._llm(prompt, metrics, max_tokens=256)
            try:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                obj = json.loads(m.group()) if m else {}
            except Exception:
                obj = {}

            if obj.get("done"):
                return {"status": "yes", "answer": str(obj.get("answer", "")), "analysis": obj.get("reasoning", "")}

            new_fact = str(obj.get("new_fact", ""))
            if new_fact:
                accumulated_facts.append(new_fact)
            next_q = obj.get("next_query", "")
            current_query = next_q if next_q else question

        if accumulated_facts:
            prompt = f"""Based on these facts, answer the question concisely.

Question: {question}
Expected answer type: {answer_type}
Facts: {"; ".join(accumulated_facts)}

Return JSON: {{"answer": "<concise answer>"}}"""
            raw = self._llm(prompt, metrics, max_tokens=64)
            try:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                obj = json.loads(m.group()) if m else {}
                ans = str(obj.get("answer", ""))
                if ans:
                    return {"status": "yes", "answer": ans, "analysis": "synthesized from accumulated facts"}
            except Exception:
                pass

        return {"status": "no", "answer": "", "analysis": "exhausted rounds"}

    # ── Parallel + sequential sub-goal execution ─────────────────────────

    def _execute_subgoals(self, goals: list[SubGoal], metrics: Metrics) -> dict[int, str]:
        sub_results: dict[int, str] = {}
        executed: set[int] = set()

        while True:
            ready = [
                g for g in goals
                if g.step_id not in executed
                and all(d in sub_results for d in g.dependencies)
            ]
            if not ready:
                break

            if len(ready) == 1:
                goal = ready[0]
                ans = self._run_single_goal(goal, sub_results, metrics)
                if ans:
                    sub_results[goal.step_id] = ans
                executed.add(goal.step_id)
            else:
                log.info(f"Parallel execution: {[g.step_id for g in ready]}")
                with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                    futures = {}
                    for goal in ready:
                        m = Metrics()
                        futures[pool.submit(self._run_single_goal, goal, dict(sub_results), m)] = (goal, m)
                    for future in as_completed(futures):
                        goal, m = futures[future]
                        metrics.merge(m)
                        ans = future.result()
                        if ans:
                            sub_results[goal.step_id] = ans
                        executed.add(goal.step_id)

        return sub_results

    def _run_single_goal(self, goal: SubGoal, sub_results: dict[int, str], metrics: Metrics) -> Optional[str]:
        resolved_q = self._resolve_placeholders(goal.question, sub_results)
        log.info(f"Sub-goal {goal.step_id}: {resolved_q[:80]}")

        result = self._investigate_loop(resolved_q, metrics)
        if result["status"] == "yes":
            log.info(f"  -> {result['answer'][:60]}")
            return result["answer"]
        else:
            log.warning(f"  -> FAILED: {result.get('analysis', '')[:60]}")
            return None

    # ── Main entry ───────────────────────────────────────────────────────

    def answer_question(self, question: str) -> tuple[str, Metrics]:
        metrics = Metrics()

        difficulty, answer_type = self._route(question, metrics)
        metrics.route_decision = difficulty
        metrics.answer_type = answer_type
        log.info(f"Route: {difficulty}, answer_type={answer_type}")

        if difficulty == "easy":
            metrics.num_subgoals = 0
            result = self._easy_investigate(question, answer_type, metrics)
            answer = result.get("answer", "") if result["status"] == "yes" else ""
        else:
            goals = self._plan(question, answer_type, metrics)
            metrics.num_subgoals = len(goals)
            log.info(f"Plan: {len(goals)} sub-goals")

            sub_results = self._execute_subgoals(goals, metrics)

            if sub_results:
                last_id = max(sub_results.keys())
                answer = sub_results[last_id]
            else:
                answer = ""

        # Fallback: if blank, try direct single-shot on original question
        if not answer:
            log.info("Fallback: direct retrieval on original question")
            result = self._investigate_loop(question, metrics, max_retries=1)
            answer = result.get("answer", "") if result["status"] == "yes" else "Unable to find answer"

        return answer, metrics


# ── Runner ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="AMAS v9: Adaptive MAS with Router")
    ap.add_argument("--questions", required=True, help="Path to questions JSON")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--retriever-url", default="http://node408:8003")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]

    pipeline = AMASv9(model=args.model, retriever_url=args.retriever_url, top_k=args.top_k)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line).get("id", ""))

    total = len(questions)
    with out_path.open("a") as out:
        for i, q in enumerate(questions):
            qid = str(q.get("id", ""))
            if qid in done_ids:
                continue
            question_text = q.get("question") or q.get("input") or ""
            gold = q.get("answer") or (q.get("answers", [""]) or [""])[0]

            start = time.time()
            try:
                answer, metrics = pipeline.answer_question(question_text)
                rec = {
                    "id": qid,
                    "question": question_text,
                    "answer": answer,
                    "gold_answer": gold,
                    "wallclock_seconds": round(time.time() - start, 3),
                    "total_tokens": metrics.total_tokens,
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "llm_calls": metrics.llm_calls,
                    "retrieval_rounds": metrics.retrieval_rounds,
                    "route_decision": metrics.route_decision,
                    "answer_type": metrics.answer_type,
                    "num_subgoals": metrics.num_subgoals,
                }
            except Exception as e:
                log.error(f"[{qid}] {e}")
                rec = {
                    "id": qid,
                    "question": question_text,
                    "answer": "",
                    "gold_answer": gold,
                    "error": str(e),
                    "wallclock_seconds": round(time.time() - start, 3),
                    "total_tokens": 0,
                }

            out.write(json.dumps(rec) + "\n")
            out.flush()
            ans_preview = str(rec["answer"])[:50] if rec.get("answer") else "(blank)"
            log.info(f"[{i+1}/{total}] {qid[:40]} | {rec.get('route_decision','')} | tok={rec.get('total_tokens',0)} | {ans_preview}")

    answered = sum(1 for line in out_path.read_text().splitlines() if line.strip() and json.loads(line).get("answer"))
    total_done = sum(1 for line in out_path.read_text().splitlines() if line.strip())
    print(f"Done: {total_done} processed (answered={answered}/{total_done})")


if __name__ == "__main__":
    main()

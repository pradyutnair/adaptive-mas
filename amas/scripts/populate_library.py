"""Stage 6: populate `exp_lib/amas_v1/library.json` via single-q SA extraction.

For each training question we (a) run the pipeline once with `gate=off` and
`lane=MAS` (forced), (b) on F1 >= threshold we ask the orchestrator-LLM (Qwen3-14B
via vLLM) for 0..2 short, generalizable insights from this single trajectory,
and (c) add them to the library tagged with `lane="MAS"`. No group ranking, no
GRPO; that's Stage 9's job. Codex Stage-2 review explicitly upgraded this from
"naive topology copy" because the library is the thesis artifact and seed
quality matters more than 10 minutes of compute.

Usage:
    python scripts/populate_library.py \
        --questions /local/yzheng/pnair/workspace/reproduction/hera/data/train_240_v2.jsonl \
        --concurrency 16 --threshold 0.70
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import yaml
from tqdm.asyncio import tqdm_asyncio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from amas.agents import load_prompts  # noqa: E402
from amas.config import build_probe_client, load_env, validate_probe_config  # noqa: E402
from amas.gates import make_gate  # noqa: E402
from amas.library import ExperienceLibrary, _normalize_lane  # noqa: E402
from amas.lm import OpenAIClient, VLLMClient, parse_json_lenient  # noqa: E402
from amas.orchestrator import Orchestrator  # noqa: E402
from amas.pipeline import AmasResult, run_amas  # noqa: E402
from amas.retriever import RetrieverClient  # noqa: E402


SINGLE_Q_SA_SYSTEM = (
    "You are a library curator distilling reusable strategy from a single successful "
    "multi-agent QA trajectory. Extract 0-2 short, generalizable insights that would help "
    "a future orchestrator answer similar queries. Insights must be operational rules, not "
    "narrative. Avoid quoting the specific question. If nothing generalizable, return an empty list."
)


def build_single_q_sa_user(query: str, profile: str, lane: str,
                            topology_agents: list[str], answer: str,
                            f1: float, tokens: int) -> str:
    return (
        f"Query: {query}\n"
        f"Query profile: {profile}\n"
        f"Routing lane used: {lane}\n"
        f"Topology: {topology_agents}\n"
        f"Final answer: {answer!r}\n"
        f"F1: {f1:.3f}\n"
        f"Tokens used: {tokens}\n\n"
        "Respond ONLY with valid JSON: {\"insights\": [{"
        "\"query_type\": \"<one of: bridge|comparison|temporal|intersection|causal|"
        "ambiguous|verification|any>\", "
        "\"lane\": \"SAS|MAS|any\", "
        "\"insight\": \"<one-sentence operational rule>\""
        "}, ...]}. "
        "Cap at 2 insights. Return {\"insights\": []} if nothing generalizable."
    )


_INSTANCE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-\.]+")


def _looks_instance_specific(insight: str, query: str, answer: str,
                              min_token_len: int = 4) -> bool:
    """Reject insights that quote question/answer-specific strings.

    Filters: any quoted span longer than 1 char; any token from question OR
    answer (length >= 4 chars, alphanumeric) appearing verbatim in the
    insight. Keeps purely operational rules; rejects benchmark-looking
    leakage like 'Use QueryDecomposer for Tom Hanks'.
    """
    s = insight.strip()
    if not s:
        return True
    # Reject if it contains any quoted phrase (likely a verbatim entity).
    if re.search(r'"[^"]{2,}"', s):
        return True
    if re.search(r"'[^']{2,}'", s):
        return True
    insight_low = s.lower()
    haystack_tokens = set()
    for src in (query, answer):
        for m in _INSTANCE_TOKEN_RE.findall(src or ""):
            if len(m) >= min_token_len:
                haystack_tokens.add(m.lower())
    for tok in haystack_tokens:
        if tok in insight_low:
            return True
    # Also reject if the literal answer substring (>= 6 chars) appears.
    a = (answer or "").strip().lower()
    if len(a) >= 6 and a in insight_low:
        return True
    return False


async def extract_single_q_insights(vllm: VLLMClient, *, query: str, profile: str,
                                     lane: str, topology_agents: list[str],
                                     answer: str, f1: float, tokens: int,
                                     filter_counter: Counter | None = None) -> list[dict]:
    user = build_single_q_sa_user(query, profile, lane, topology_agents,
                                   answer, f1, tokens)
    try:
        res = await vllm.chat(SINGLE_Q_SA_SYSTEM, user, temperature=0.3,
                                max_tokens=400, json_mode=True)
        parsed = parse_json_lenient(res.text) or {}
        ins = parsed.get("insights") if isinstance(parsed, dict) else None
        if not isinstance(ins, list):
            return []
        out = []
        for it in ins[:2]:
            if not isinstance(it, dict):
                continue
            text = str(it.get("insight", "")).strip()
            if not text:
                if filter_counter is not None:
                    filter_counter["empty"] += 1
                continue
            if _looks_instance_specific(text, query, answer):
                if filter_counter is not None:
                    filter_counter["instance_specific"] += 1
                logging.info("filtered instance-specific insight: %r", text[:80])
                continue
            out.append({
                "query_type": str(it.get("query_type", profile)),
                "lane": _normalize_lane(it.get("lane", lane)),
                "insight": text,
                "rationale": f"single-q SA from F1={f1:.2f} on training pool",
            })
        return out
    except Exception as e:
        logging.warning("extract_single_q_insights failed: %s", e)
        if filter_counter is not None:
            filter_counter["extraction_error"] += 1
        return []


async def run_one_q(q: dict, *, gate, orch, retriever, openai_client, probe_client,
                     pcfg: dict) -> tuple[AmasResult, list[dict]]:
    """Run a single training question with lane forced to MAS, gate=off. Return
    (result, insights). insights is empty if F1 below threshold or extraction failed.
    """
    res = await run_amas(
        question=q.get("question", ""),
        gold=q.get("answer", ""),
        qid=str(q.get("id", "")),
        gate=gate, orchestrator=orch, retriever=retriever,
        openai_client=openai_client, probe_client=probe_client,
        t_max=pcfg.get("t_max", 2),
        probe_group_size=pcfg.get("probe_group_size", 1),
        probe_topk=pcfg.get("probe_topk", 5),
        rollout_temperature=pcfg.get("rollout_temperature", 0.0),
    )
    return res, []  # caller fills insights after threshold check


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _resolve_outdir(arg_outdir: str | None, force: bool) -> Path:
    """Return a unique output dir. Default is timestamped under
    `results/populate_amas_v1/`. Pass --out-dir to override; if the directory
    already exists and is non-empty, refuse unless --force."""
    if arg_outdir:
        p = Path(arg_outdir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        p = Path("results") / "populate_amas_v1" / ts
    if p.exists() and any(p.iterdir()) and not force:
        print(f"refuse to overwrite non-empty {p} (pass --force)", file=sys.stderr)
        sys.exit(2)
    p.mkdir(parents=True, exist_ok=True)
    return p


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    validate_probe_config(cfg)

    out_dir = _resolve_outdir(args.out_dir, args.force)

    vcfg = cfg["vllm"]; ocfg = cfg["openai"]; rcfg = cfg["retriever"]; pcfg = cfg["pipeline"]
    vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                      max_tokens=vcfg["max_tokens"], temperature=vcfg["temperature"],
                      concurrency=vcfg.get("concurrency", 12))
    openai_client = OpenAIClient(model=ocfg["model"], max_tokens=ocfg["max_tokens"],
                                  temperature=ocfg["temperature"],
                                  concurrency=ocfg.get("concurrency", 24))
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"],
                                 concurrency=rcfg.get("concurrency", 8))
    probe_client, probe_owned = build_probe_client(cfg, vllm=vllm, openai_client=openai_client)

    paths = cfg["paths"]
    library = ExperienceLibrary.load(paths["library"])
    if library.entries and not args.force:
        print(f"refuse to populate non-empty library {paths['library']} "
              f"(entries={len(library.entries)}); pass --force to overwrite",
              file=sys.stderr)
        sys.exit(2)
    if args.force and library.entries:
        print(f"--force: clearing {len(library.entries)} existing entries before population")
        library = ExperienceLibrary(max_entries=library.max_entries)
    prompts = load_prompts(paths["prompts"])

    orch = Orchestrator(vllm=vllm, openai_client=openai_client, retriever=retriever,
                        library=library, prompts=prompts,
                        retriever_topk=rcfg["topk"], library_top_k=5, max_steps=8)
    gate = make_gate("off", openai_client=openai_client, cfg=cfg.get("gate", {}))

    # Load training questions BEFORE entering the try/finally — failure here
    # should bubble up before any clients/monkey-patches are installed.
    qpath = Path(args.questions)
    if qpath.suffix == ".jsonl":
        rows = [json.loads(l) for l in qpath.read_text().splitlines() if l.strip()]
    else:
        rows = json.loads(qpath.read_text())
    if args.n is not None:
        rows = rows[: args.n]

    # Persist a run config + question-id manifest BEFORE running.
    run_config = {
        "git_commit": _git_commit(),
        "config_path": str(args.config),
        "questions_path": str(args.questions),
        "n": len(rows),
        "threshold": args.threshold,
        "concurrency": args.concurrency,
        "force": bool(args.force),
        "forced_lane": "MAS",
        "gate": "off",
        "vllm_model": vcfg.get("model"),
        "vllm_endpoints": list(vcfg.get("endpoints") or []),
        "openai_model": ocfg.get("model"),
        "retriever_url": rcfg.get("url"),
        "retriever_topk": rcfg.get("topk"),
        "library_path": paths.get("library"),
        "prompts_path": paths.get("prompts"),
        "out_dir": str(out_dir),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (out_dir / "question_ids.json").write_text(json.dumps(
        [{"id": q.get("id"), "source": q.get("source"),
           "question_type": q.get("question_type")} for q in rows],
        indent=2,
    ))

    # Force lane=MAS via monkey-patch. MUST be restored even if anything below
    # raises (codex Stage 6 review).
    orig_route = orch.route_lane

    async def force_mas(query, profile, *, probe_answer, probe_agreement,
                         top_passage_score=0.0, temperature=0.0):
        return "MAS", {"reason": "stage6 forced MAS", "tokens": 0}

    orch.route_lane = force_mas  # type: ignore[assignment]

    sem = asyncio.Semaphore(args.concurrency)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"
    fh = open(pred_path, "w")
    results: list[tuple[AmasResult, list[dict]]] = []
    all_insights: list[dict] = []
    filter_counter: Counter[str] = Counter()

    async def step(q: dict) -> tuple[AmasResult, list[dict]]:
        async with sem:
            try:
                res, _ = await run_one_q(
                    q, gate=gate, orch=orch, retriever=retriever,
                    openai_client=openai_client, probe_client=probe_client, pcfg=pcfg,
                )
            except Exception as e:
                logging.warning("run failed for qid=%s: %s", q.get("id"), e)
                return AmasResult(qid=str(q.get("id", "")), question=q.get("question", ""),
                                  gold=q.get("answer", ""), profile="bridge",
                                  gate="off", final_answer="", error=str(e)[:300]), []
            # Force-MAS bookkeeping: route_lane is only called inside run_amas when
            # probe.consensus_answer is non-empty. If probe returned empty, our
            # monkey-patch never fires and res.router_lane stays AUTO. We still
            # *intend* lane=MAS for population provenance, so override here so
            # downstream insight tagging and analytics are honest.
            if res.router_lane != "MAS":
                res.router_lane = "MAS"
                if not res.router_reason:
                    res.router_reason = "stage6 forced MAS (post-hoc, empty probe)"
            insights: list[dict] = []
            if res.f1 >= args.threshold and res.final_answer:
                topology_agents: list[str] = []
                if res.turns:
                    last_topo = res.turns[-1].topology
                    if isinstance(last_topo, dict):
                        topology_agents = [
                            it.get("agent", "")
                            for it in (last_topo.get("execution_order") or [])
                        ]
                insights = await extract_single_q_insights(
                    vllm,
                    query=q.get("question", ""),
                    profile=res.profile,
                    lane="MAS",
                    topology_agents=topology_agents,
                    answer=res.final_answer,
                    f1=res.f1,
                    tokens=res.total_tokens,
                    filter_counter=filter_counter,
                )
            return res, insights

    try:
        coros = [step(q) for q in rows]
        results = await tqdm_asyncio.gather(*coros, desc="populate")
        n_success = 0
        for q, (res, insights) in zip(rows, results):
            d = res.to_dict() if hasattr(res, "to_dict") else asdict(res)
            d["question"] = q.get("question", "")
            d["gold"] = q.get("answer", "")
            d["source"] = q.get("source", "")
            d["question_type"] = q.get("question_type", "")
            d["insights_collected"] = len(insights)
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            if res.f1 >= args.threshold:
                n_success += 1
            all_insights.extend(insights)
        fh.close()
        fh = None  # type: ignore[assignment]

        # Apply filtered insights sequentially.
        added = 0
        for ins in all_insights:
            eid = library.add(profile=ins["query_type"], insight=ins["insight"],
                                rationale=ins.get("rationale", ""), lane=ins["lane"])
            if eid:
                added += 1
        library.save(paths["library"])

        em_list = [r.em for r, _ in results]
        f1_list = [r.f1 for r, _ in results]
        tok_list = [r.total_tokens for r, _ in results]
        lane_dist = Counter((r.router_lane or "AUTO") for r, _ in results)
        summary = {
            "n_total": len(rows),
            "n_success_at_threshold": n_success,
            "threshold": args.threshold,
            "candidate_insights": len(all_insights),
            "filtered_insights_count": dict(filter_counter),
            "library_entries_after": len(library.entries),
            "library_entries_added": added,
            "mean_em": mean(em_list) if em_list else 0.0,
            "mean_f1": mean(f1_list) if f1_list else 0.0,
            "mean_tokens": mean(tok_list) if tok_list else 0.0,
            "tokens_p50": sorted(tok_list)[len(tok_list)//2] if tok_list else 0,
            "tokens_p95": sorted(tok_list)[int(0.95 * len(tok_list))] if tok_list else 0,
            "router_lane_distribution": dict(lane_dist),
            "out_dir": str(out_dir),
            "git_commit": run_config["git_commit"],
        }
        print(json.dumps(summary, indent=2))
        summary_path.write_text(json.dumps(summary, indent=2))
    finally:
        # Restore monkey-patch even on failure.
        orch.route_lane = orig_route  # type: ignore[assignment]
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
        # Close every client exactly once.
        to_close = [retriever, openai_client, vllm]
        if probe_owned:
            to_close.append(probe_client)
        seen: set[int] = set()
        for c in to_close:
            if c is None or id(c) in seen:
                continue
            seen.add(id(c))
            if hasattr(c, "aclose"):
                try:
                    await c.aclose()
                except Exception:
                    pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="Cap number of questions (for smoke runs).")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="Minimum F1 for a trajectory to seed insights "
                         "(codex-tightened from 0.5 -> 0.70 to favour "
                         "fewer, higher-quality library entries).")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for predictions/summary/run_config. "
                         "Defaults to a timestamped path under "
                         "results/populate_amas_v1/.")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting a non-empty library OR an existing "
                         "non-empty out-dir.")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

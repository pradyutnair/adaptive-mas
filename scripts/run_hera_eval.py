#!/usr/bin/env python3
"""Evaluate HERA artifacts on QID-matched baseline question sets."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import string
import sys
import urllib.error
import urllib.request
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from amas3.experience_library import ExperienceEntry, ExperienceLibrary, format_for_prompt
from amas3.lm import make_qwen14b_nothink_lm
from amas3.pipeline import AmasPipeline, AmasPipelineConfig, AmasResult
from amas3.retriever import Retriever
from amas3.tf_grpo import config_from_topology, sample_topology


def check_retriever_health(base_url: str, timeout_seconds: float = 10.0) -> None:
    url = base_url.rstrip("/") + "/retrieve"
    payload = json.dumps({"queries": ["health check"], "topk": 1, "mode": "text"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
            data = json.loads(resp.read().decode("utf-8") or "{}")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Retriever health check failed for {url}: {exc}") from exc
    if not isinstance(data, dict) or "results" not in data:
        raise RuntimeError(f"Retriever health check returned unexpected payload from {url}: {str(data)[:200]}")

log = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "compiled" / "hera_tfgrpo_gepa"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "hera_tfgrpo_gepa"
BALANCED_BASELINE_DIR = REPO_ROOT / "frozen" / "amas-final" / "results" / "balanced"
CLEAN_BASELINE_DIR = PROJECT_ROOT / "final-method" / "results" / "clean" / "base"

DATASETS = {
    "hotpotqa": {
        "aliases": ("hotpotqa", "hotpot"),
        "questions": [
            REPO_ROOT / "data" / "hotpotqa" / "questions_1000_seed42.json",
            REPO_ROOT / "data" / "hotpot" / "questions_1000_seed42.json",
        ],
        "balanced_name": "hotpot",
        "clean_name": "hotpotqa",
    },
    "2wiki": {
        "aliases": ("2wiki", "2wikimultihop"),
        "questions": [REPO_ROOT / "data" / "2wikimultihop" / "questions_1000_seed42.json"],
        "balanced_name": "2wiki",
        "clean_name": "2wiki",
    },
    "musique": {
        "aliases": ("musique",),
        "questions": [REPO_ROOT / "data" / "musique" / "questions_1000_seedfull_combined.json"],
        "balanced_name": "musique",
        "clean_name": "musique",
    },
    "bamboogle": {
        "aliases": ("bamboogle",),
        "questions": [REPO_ROOT / "data" / "bamboogle" / "questions_125.json"],
        "balanced_name": "bamboogle",
        "clean_name": "bamboogle",
    },
}

ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCTUATION = set(string.punctuation)


def parse_args() -> argparse.Namespace:
    """Eval CLI is intentionally minimal.

    HERA's eval-time policy is pi_O(Gamma | q, E, N). The only inputs are:
      - the trained experience library E,
      - the evolved per-agent prompts rho_i,
      - the agent pool N (defined in the pipeline).
    No threshold flags, no per-query-type tables, no ablation overrides:
    every eval-time topology decision must come from pi_O conditioned on E.
    """
    parser = argparse.ArgumentParser(description="Evaluate HERA TF-GRPO GEPA artifacts.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--datasets", default="hotpotqa,2wiki,musique,bamboogle")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--retriever-url", default="http://node408:8003")
    parser.add_argument("--dry-run", action="store_true", help="Validate wiring without LLM calls.")
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = ARTICLES_RE.sub("", text)
    text = "".join(ch for ch in text if ch not in PUNCTUATION)
    return " ".join(text.split()).strip()


def norm_em(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {token: pred_tokens.count(token) for token in set(pred_tokens)}
    gold_counts = {token: gold_tokens.count(token) for token in set(gold_tokens)}
    common = sum(min(pred_counts.get(token, 0), gold_counts.get(token, 0)) for token in pred_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contain(prediction: str, gold: str) -> float:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return 0.0
    return 1.0 if gold_norm in pred_norm else 0.0


def load_json(path: Path) -> list[dict] | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_dataset(name: str) -> str:
    normalized = name.strip().lower()
    for canonical, config in DATASETS.items():
        if normalized == canonical or normalized in config["aliases"]:
            return canonical
    raise ValueError(f"Unknown dataset: {name}")


def resolve_questions_path(dataset: str) -> Path:
    for path in DATASETS[dataset]["questions"]:
        if path.exists():
            return path
    raise FileNotFoundError(f"No question file found for {dataset}")


def load_artifacts(artifacts_dir: Path) -> tuple[ExperienceLibrary, dict]:
    """Load HERA artifacts: experience library E and evolved prompts rho.

    The orchestrator's eval-time decisions are derived ONLY from these.
    Any 'orchestration_config.json' produced by older runs is ignored:
    flat per-type thresholds would turn eval into a grid search.
    """
    library_path = artifacts_dir / "experience_library.json"
    prompts_path = artifacts_dir / "evolved_prompts.json"
    library = ExperienceLibrary.load(library_path) if library_path.exists() else ExperienceLibrary()
    prompts = load_json(prompts_path) if prompts_path.exists() else {}
    return library, prompts


def add_evolved_prompt_entries(library: ExperienceLibrary, prompts: dict) -> None:
    """Skip -- the compiled library already contains gepa_prompt_{role} entries."""
    pass


def selected_experience_text(library: ExperienceLibrary, question: str, limit: int = 4) -> str:
    """Retrieve compact HERA entries for all roles, including orchestrator/planner."""
    selected: list[ExperienceEntry] = []
    seen: set[str] = set()
    for role in ("orchestrator", "planner", "solver", "synthesizer"):
        for entry in library.retrieve(question, role=role, limit=2):
            if entry.id in seen:
                continue
            selected.append(entry)
            seen.add(entry.id)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    return format_for_prompt(selected, max_entries=limit, max_insight_chars=110)


def selected_role_prompts(
    library: ExperienceLibrary,
    question: str,
    evolved_prompts: dict,
    entries_per_role: int = 3,
) -> dict[str, str]:
    """Build role-specific prompt additions from retrieved HERA entries.

    This avoids injecting the same mixed-role library text into every agent and
    keeps each agent conditioned only on relevant ADD/MERGE/PRUNE insights.
    """
    role_prompts: dict[str, str] = {}
    if os.environ.get("AMAS_ROLE_SPECIFIC_EXPERIENCE", "1") != "1":
        return evolved_prompts if isinstance(evolved_prompts, dict) else {}
    for role in ("orchestrator", "planner", "solver", "synthesizer"):
        parts: list[str] = []
        entries = library.retrieve(question, role=role, limit=entries_per_role)
        if entries:
            parts.append(format_for_prompt(entries, max_entries=entries_per_role, max_insight_chars=120))
        evolved = (evolved_prompts or {}).get(role, "") if isinstance(evolved_prompts, dict) else ""
        if evolved:
            parts.append(evolved)
        if parts:
            role_prompts[role] = "\n\n".join(parts)
    return role_prompts


def build_pipeline_config(
    experience_text: str,
    role_prompts: dict | None = None,
) -> AmasPipelineConfig:
    """Build a HERA base config.

    The base only switches on the HERA pipeline functionality required for
    pi_O to operate (orchestrator on, adaptive solver budget, slim synth,
    verifier on). Every topology-shaping knob (retrieval budget, repair,
    routing strategy, orch_min_confidence, etc.) is OVERWRITTEN per question
    by `config_from_topology(base, sampled_Gamma)` where Gamma is sampled
    from pi_O(. | q, E, N). There is no per-query-type threshold table and
    no eval-time grid.
    """
    return AmasPipelineConfig(
        experience_library=experience_text,
        use_orchestrator=True,
        adaptive_solver_budget=True,
        synth_slim=True,
        orch_use_verifier=True,
        role_prompts=role_prompts or {},
    )


def make_shared_lms() -> dict:
    """Build LM instances once; they are stateless HTTP wrappers safe for reuse."""
    max_tokens = int(os.environ.get("AMAS_EVAL_MAX_TOKENS", "640"))
    sas_tokens = int(os.environ.get("AMAS_EVAL_SAS_MAX_TOKENS", "384"))
    return {
        "planner_lm": make_qwen14b_nothink_lm(replica_idx=0, max_tokens=max_tokens),
        "worker_lm": make_qwen14b_nothink_lm(replica_idx=1, max_tokens=max_tokens),
        "synth_lm": make_qwen14b_nothink_lm(replica_idx=2, max_tokens=max_tokens),
        "sas_lm": make_qwen14b_nothink_lm(replica_idx=0, max_tokens=sas_tokens),
    }


def build_pipeline(retriever: Retriever, config: AmasPipelineConfig, lms: dict) -> AmasPipeline:
    return AmasPipeline(
        planner_lm=lms["planner_lm"],
        worker_lm=lms["worker_lm"],
        synth_lm=lms["synth_lm"],
        sas_lm=lms["sas_lm"],
        retriever=retriever,
        config=config,
    )


def prediction_row(question: dict, result: AmasResult, elapsed: float) -> dict:
    metadata = asdict(result)
    metadata["wallclock_seconds"] = round(elapsed, 3)
    return {
        "id": str(question.get("id", "")),
        "question": question.get("question", ""),
        "gold_answer": question.get("answer", ""),
        "answer": result.answer,
        "prediction": result.answer,
        "metadata": metadata,
    }


def dry_run_row(question: dict) -> dict:
    return {
        "id": str(question.get("id", "")),
        "question": question.get("question", ""),
        "gold_answer": question.get("answer", ""),
        "answer": "",
        "prediction": "",
        "metadata": {"dry_run": True, "total_tokens": 0},
    }


CONCURRENCY = int(os.environ.get("AMAS_EVAL_CONCURRENCY", "24"))


async def run_one_question(
    idx: int,
    total: int,
    question: dict,
    library: ExperienceLibrary,
    prompts: dict,
    retriever: Retriever,
    lms: dict,
    sem: asyncio.Semaphore,
    done_counter: list[int],
    dataset: str = "default",
) -> dict:
    """Single eval step: Gamma ~ pi_O(.|q, E, N) then execute the pipeline.

    There is exactly one path: the orchestrator LM samples a topology
    conditioned on the question and the retrieved experiences, and the
    pipeline config is then derived from that sampled topology. No flat
    threshold overrides, no per-query-type tables.
    """
    async with sem:
        qid = str(question.get("id", ""))
        q_text = str(question.get("question", ""))
        use_role_specific = os.environ.get("AMAS_ROLE_SPECIFIC_EXPERIENCE", "1") == "1"
        experience_text = "" if use_role_specific else selected_experience_text(library, q_text)
        role_prompts = (
            selected_role_prompts(library, q_text, prompts if isinstance(prompts, dict) else {})
            if use_role_specific
            else (prompts if isinstance(prompts, dict) else {})
        )
        base_config = build_pipeline_config(experience_text, role_prompts)

        sampler_lm = make_qwen14b_nothink_lm(
            replica_idx=idx % 3,
            max_tokens=int(os.environ.get("AMAS_EVAL_TOPOLOGY_MAX_TOKENS", "512")),
            temperature=float(os.environ.get("AMAS_EVAL_TOPOLOGY_TEMPERATURE", "0.15")),
        )
        sampled_topology = await asyncio.to_thread(
            sample_topology,
            question=q_text,
            qid=qid,
            library=library,
            sampler_lm=sampler_lm,
            sample_index=1,
            dataset=dataset,
        )
        sampler_tokens = int(sampled_topology.get("_sampler_tokens", 0) or 0)
        config = config_from_topology(base_config, sampled_topology)

        pipeline = build_pipeline(retriever, config, lms)
        t0 = time.time()
        result = await pipeline.run(question=q_text, qid=qid)
        pipeline_tokens = int(result.total_tokens)
        result.total_tokens = pipeline_tokens + sampler_tokens
        row = prediction_row(question, result, time.time() - t0)
        row["metadata"]["semantic_policy"] = True
        row["metadata"]["role_specific_experience"] = bool(use_role_specific)
        row["metadata"]["pipeline_tokens"] = pipeline_tokens
        row["metadata"]["topology_sampler_tokens"] = sampler_tokens
        row["metadata"]["sampled_topology"] = sampled_topology
        done_counter[0] += 1
        log.info("done %d/%d qid=%s topo=%s tok=%d",
                 done_counter[0], total, qid,
                 result.topology, result.total_tokens)
        return row


async def run_dataset_predictions(
    questions: list[dict],
    library: ExperienceLibrary,
    prompts: dict,
    retriever_url: str,
    output_path: Path,
    dataset: str = "default",
) -> list[dict]:
    retriever = Retriever(base_url=retriever_url)
    lms = make_shared_lms()
    sem = asyncio.Semaphore(CONCURRENCY)
    done_counter = [0]

    if output_path.exists():
        output_path.unlink()

    tasks = [
        run_one_question(idx, len(questions), q, library, prompts,
                         retriever, lms, sem, done_counter,
                         dataset=dataset)
        for idx, q in enumerate(questions, start=1)
    ]
    rows = await asyncio.gather(*tasks)
    rows = list(rows)

    for row in rows:
        append_jsonl(output_path, row)
    return rows


def evaluate_rows(rows: list[dict], gold_by_id: dict[str, str]) -> dict:
    matched = [row for row in rows if str(row.get("id", "")) in gold_by_id]
    em_scores = []
    f1_scores = []
    contain_scores = []
    token_counts = []
    answered = 0
    for row in matched:
        pred = str(row.get("answer", row.get("prediction", "")))
        gold = gold_by_id[str(row.get("id", ""))]
        if pred.strip():
            answered += 1
        em_scores.append(norm_em(pred, gold))
        f1_scores.append(token_f1(pred, gold))
        contain_scores.append(contain(pred, gold))
        metadata = row.get("metadata", {})
        token_counts.append(float(metadata.get("total_tokens", row.get("total_tokens", 0)) or 0))
    total = len(matched)
    return {
        "norm_em": round(sum(em_scores) / total, 4) if total else 0.0,
        "token_f1": round(sum(f1_scores) / total, 4) if total else 0.0,
        "contain": round(sum(contain_scores) / total, 4) if total else 0.0,
        "avg_tokens": round(sum(token_counts) / total, 1) if total else 0.0,
        "total": total,
        "answered": answered,
    }


def load_baseline_rows(dataset: str, baseline: str) -> list[dict]:
    config = DATASETS[dataset]
    if baseline == "balanced":
        path = BALANCED_BASELINE_DIR / config["balanced_name"] / "predictions.jsonl"
    else:
        path = CLEAN_BASELINE_DIR / config["clean_name"] / "predictions.jsonl"
    if not path.exists():
        return []
    return load_jsonl(path)


def matched_baseline_eval(
    dataset: str,
    baseline: str,
    our_rows: list[dict],
    gold_by_id: dict[str, str],
) -> dict:
    baseline_rows = load_baseline_rows(dataset, baseline)
    baseline_by_id = {str(row.get("id", "")): row for row in baseline_rows}
    our_by_id = {str(row.get("id", "")): row for row in our_rows}
    matched_ids = sorted(set(our_by_id) & set(baseline_by_id) & set(gold_by_id))
    matched_baseline_rows = [baseline_by_id[qid] for qid in matched_ids]
    matched_our_rows = [our_by_id[qid] for qid in matched_ids]
    baseline_summary = evaluate_rows(matched_baseline_rows, gold_by_id)
    ours_summary = evaluate_rows(matched_our_rows, gold_by_id)
    baseline_summary["matched_total"] = len(matched_ids)
    baseline_summary["baseline_rows"] = len(baseline_rows)
    ours_summary["matched_total"] = len(matched_ids)
    delta_norm_em = round(ours_summary["norm_em"] - baseline_summary["norm_em"], 4)
    delta_token_f1 = round(ours_summary["token_f1"] - baseline_summary["token_f1"], 4)
    delta_contain = round(ours_summary["contain"] - baseline_summary["contain"], 4)
    delta_avg_tokens = round(ours_summary["avg_tokens"] - baseline_summary["avg_tokens"], 1)
    token_ratio = (
        round(ours_summary["avg_tokens"] / baseline_summary["avg_tokens"], 4)
        if baseline_summary["avg_tokens"] > 0 else None
    )
    return {
        "ours_matched": ours_summary,
        "baseline": baseline_summary,
        "delta_norm_em": delta_norm_em,
        "delta_token_f1": delta_token_f1,
        "delta_contain": delta_contain,
        "delta_avg_tokens": delta_avg_tokens,
        "token_ratio": token_ratio,
        "beats_quality": delta_norm_em > 0 and delta_token_f1 > 0 and delta_contain > 0,
        "beats_tokens": delta_avg_tokens < 0,
        "tokens_20pct_lower": token_ratio is not None and token_ratio <= 0.8,
        "beats_quality_and_tokens": (delta_norm_em > 0 and delta_token_f1 > 0 and delta_contain > 0 and delta_avg_tokens < 0),
        "qid_matched_total": len(matched_ids),
    }


def run_dry_dataset(questions: list[dict], predictions_path: Path) -> list[dict]:
    if predictions_path.exists():
        predictions_path.unlink()
    rows = [dry_run_row(question) for question in questions]
    for row in rows:
        append_jsonl(predictions_path, row)
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.dry_run:
        try:
            check_retriever_health(args.retriever_url)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Eval needs E (experience library) and the evolved prompts. No flat
    # orchestration config is consumed: pi_O reads E and samples Gamma.
    if not args.dry_run:
        _required = ['experience_library.json', 'evolved_prompts.json']
        _missing = [f for f in _required if not (artifacts_dir / f).exists()]
        if _missing:
            raise SystemExit(
                'Incomplete artifact -- cannot run final eval without: '
                + ', '.join(_missing)
                + f' Artifacts dir: {artifacts_dir}. Training still running?'
            )

    library, prompts = load_artifacts(artifacts_dir)
    add_evolved_prompt_entries(library, prompts if isinstance(prompts, dict) else {})
    write_json(
        output_dir / "eval_config.json",
        {
            "artifacts_dir": str(artifacts_dir),
            "policy": "pi_O(Gamma | q, E, N) with config_from_topology(base, Gamma)",
        },
    )

    requested = [canonical_dataset(name) for name in args.datasets.split(",") if name.strip()]
    comparison: dict[str, dict] = {}

    for dataset in requested:
        questions_path = resolve_questions_path(dataset)
        questions = load_json(questions_path)
        if not isinstance(questions, list):
            raise ValueError(f"Expected list in {questions_path}")
        if args.limit > 0:
            questions = questions[: args.limit]
        gold_by_id = {str(row.get("id", "")): str(row.get("answer", "")) for row in questions}

        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = dataset_dir / "predictions.jsonl"

        if args.dry_run:
            rows = run_dry_dataset(questions, predictions_path)
        else:
            rows = asyncio.run(
                run_dataset_predictions(
                    questions=questions,
                    library=library,
                    prompts=prompts if isinstance(prompts, dict) else {},
                    retriever_url=args.retriever_url,
                    output_path=predictions_path,
                    dataset=dataset,
                )
            )

        ours = evaluate_rows(rows, gold_by_id)
        ours["questions_path"] = str(questions_path)
        ours["dry_run"] = bool(args.dry_run)
        write_json(dataset_dir / "eval.json", ours)

        balanced = matched_baseline_eval(dataset, "balanced", rows, gold_by_id)
        clean = matched_baseline_eval(dataset, "clean", rows, gold_by_id)
        comparison[dataset] = {
            "ours": ours,
            "balanced": balanced,
            "clean_base": clean,
            "qid_match_note": "Deltas are computed only over the exact same IDs present in our run, the baseline, and gold file.",
        }

    write_json(output_dir / "comparison.json", comparison)
    log.info("evaluation complete, wrote %s", output_dir / "comparison.json")


if __name__ == "__main__":
    main()

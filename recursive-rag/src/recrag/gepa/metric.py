from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from recrag.contracts import normalize_answer


@dataclass
class ScoreWithFeedback:
    score: float
    feedback: str


def norm_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def metric(gold: Any, pred: Any, trace: Any = None, pred_name: str | None = None, pred_trace: Any = None) -> ScoreWithFeedback:
    gold_answer = str(getattr(gold, "answer", "") if not isinstance(gold, dict) else gold.get("answer", ""))
    pred_answer = str(getattr(pred, "answer", "") if not isinstance(pred, dict) else pred.get("answer", ""))
    score = norm_em(pred_answer, gold_answer)
    traj = getattr(pred, "trajectory", None) if not isinstance(pred, dict) else pred.get("trajectory")
    meta = getattr(pred, "metadata", None) if not isinstance(pred, dict) else pred.get("metadata", {})
    feedback: list[str] = []
    findings = meta.get("findings", []) if isinstance(meta, dict) else []
    for f in findings:
        try:
            conf = float(f.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf < 0.7:
            feedback.append("Low-confidence hop needs better evidence or a different decomposition.")
            break
    traj_s = json.dumps(traj, ensure_ascii=False) if traj is not None else ""
    if "FAIL:" in traj_s:
        feedback.append("Citation gate rejected a submission; cite hop evidence ids and align answer to cited findings.")
    if traj_s.count("hop_batch") == 0 and traj_s.count("tool_name") > 4:
        feedback.append("Consider hop_batch for independent subquestions.")
    if not pred_answer.strip():
        feedback.append("Final answer was blank; finish only after submit accepts grounded support.")
    if not feedback:
        feedback.append("Use the shortest grounded route: one hop for direct questions, sequential hops for bridges, hop_batch for independent facts.")
    return ScoreWithFeedback(score=score, feedback=" ".join(feedback))

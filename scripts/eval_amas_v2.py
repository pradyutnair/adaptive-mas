import json, re, string, sys, collections, pathlib

def normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def token_f1(pred, gold):
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)

def main():
    base = pathlib.Path("/local/yzheng/pnair/workspace/adaptive-mas")
    pred_path = base / "results/amas_v2_pilot50/predictions.jsonl"
    gold_path = base / "data/musique/opera408_50.json"

    with open(gold_path) as f:
        golds = {item["id"]: item["answer"] for item in json.load(f)}

    preds = {}
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            preds[r["id"]] = r

    em_hits, contain_hits, f1_sum = 0, 0, 0.0
    total_tokens, answered = 0, 0
    wrong = []

    for qid, gold_ans in golds.items():
        rec = preds.get(qid)
        if not rec:
            wrong.append({"id": qid, "pred": "[MISSING]", "gold": gold_ans, "em": 0, "contain": 0, "f1": 0.0})
            continue
        pred_ans = rec.get("answer", "")
        total_tokens += rec.get("metadata", {}).get("total_tokens", 0)
        if pred_ans.strip():
            answered += 1

        n_pred = normalize(pred_ans)
        n_gold = normalize(gold_ans)
        em = int(n_pred == n_gold)
        contain = int(n_gold in n_pred) if n_gold else 0
        f1 = token_f1(pred_ans, gold_ans)

        em_hits += em
        contain_hits += contain
        f1_sum += f1

        if not em:
            wrong.append({"id": qid, "pred": pred_ans, "gold": gold_ans, "em": em, "contain": contain, "f1": round(f1, 4)})

    n = len(golds)
    summary = {
        "n": n,
        "answered": answered,
        "answer_rate": round(answered / n, 4),
        "em": round(em_hits / n, 4),
        "contain": round(contain_hits / n, 4),
        "f1": round(f1_sum / n, 4),
        "mean_tokens": round(total_tokens / n, 1),
    }

    out_path = base / "results/amas_v2_pilot50/eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\n--- Wrong answers ({len(wrong)}/{n}) ---")
    for w in wrong:
        print(f"  {w['id']}: pred={w['pred']!r}  gold={w['gold']!r}  contain={w['contain']}  f1={w['f1']}")

if __name__ == "__main__":
    main()

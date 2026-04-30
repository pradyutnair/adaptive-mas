#!/usr/bin/env python3
"""For each stratified-100 question, retrieve top-50 chunks via original Q +
the planner's sub-questions, and check if the gold answer appears verbatim
in ANY retrieved chunk.

If gold-in-retrieved → question IS answerable from corpus (lower bound).
If gold-NOT-in-retrieved → either retrieval missed (unlikely for top-50
across multiple queries) or corpus doesn't have it.

Cost: ~300 retrieval calls, ~90s. No LLM calls. Doesn't compete with 1000q.
"""
import asyncio, json, re, string, sys
from pathlib import Path

REPO = Path('/local/yzheng/pnair/workspace/adaptive-mas')
sys.path.insert(0, str(REPO / 'src'))

import httpx

RETRIEVER_URL = 'http://node408:8003/retrieve'
TOPK = 5


def normalize(s):
    s = (s or '').lower()
    s = re.sub(r'\b(a|an|the)\b', '', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split()).strip()


def appears_in(answer: str, text: str) -> bool:
    a = normalize(answer)
    t = normalize(text)
    if not a:
        return False
    return a in t


async def retrieve(client, query: str):
    try:
        r = await client.post(RETRIEVER_URL, json={'queries': [query], 'topk': TOPK, 'mode': 'text'})
        r.raise_for_status()
        return r.json()['results'][0]
    except Exception as e:
        return []


async def main():
    qs = json.load(open(REPO / 'data/musique/stratified_100.json'))
    # Load AMAS-PRO 14B predictions to get the planner's sub-questions
    pro_14b_preds = {r['id']: r for r in (json.loads(l) for l in open(REPO / 'results/amas_pro_qwen3_14b_4omini_stratified100/predictions.jsonl') if l.strip())}

    results = []
    async with httpx.AsyncClient(timeout=30) as client:
        sem = asyncio.Semaphore(10)

        async def check_one(q):
            async with sem:
                qid = q['id']
                gold = q['answer']
                queries = [q['question']]
                # Add sub-questions from AMAS-PRO 14B predictions
                pred = pro_14b_preds.get(qid, {})
                for f in pred.get('metadata', {}).get('findings', []):
                    sq = f.get('sub_question', '').strip()
                    if sq and sq not in queries:
                        queries.append(sq)
                # Also try gold answer as a query (sanity check)
                queries.append(gold)
                # Retrieve all
                chunks = []
                for query in queries[:5]:  # cap at 5 queries to keep it fast
                    cs = await retrieve(client, query)
                    chunks.extend(cs)
                # Dedupe by chunk_id
                seen = set()
                unique_chunks = []
                for c in chunks:
                    cid = c.get('chunk_id', '')
                    if cid not in seen:
                        seen.add(cid)
                        unique_chunks.append(c)
                # Check if gold appears in any
                hit = any(appears_in(gold, c.get('text', '')) for c in unique_chunks)
                return {
                    'id': qid,
                    'question': q['question'],
                    'gold': gold,
                    'n_unique_chunks': len(unique_chunks),
                    'gold_in_any_chunk': hit,
                    'queries_tried': queries[:5],
                    'predicted_correct': normalize(pred.get('answer', '')) == normalize(gold) if pred else False,
                }

        results = await asyncio.gather(*[check_one(q) for q in qs])

    # Summary
    n = len(results)
    gold_found = sum(1 for r in results if r['gold_in_any_chunk'])
    correct = sum(1 for r in results if r['predicted_correct'])
    correct_and_found = sum(1 for r in results if r['predicted_correct'] and r['gold_in_any_chunk'])
    correct_but_not_found = sum(1 for r in results if r['predicted_correct'] and not r['gold_in_any_chunk'])
    wrong_and_found = sum(1 for r in results if not r['predicted_correct'] and r['gold_in_any_chunk'])
    wrong_and_not_found = sum(1 for r in results if not r['predicted_correct'] and not r['gold_in_any_chunk'])

    print(f'Total questions: {n}')
    print(f'Gold found in retrieved chunks (top-50 across {len(results[0]["queries_tried"])} queries): {gold_found}/{n} ({100*gold_found/n:.1f}%)')
    print(f'AMAS-PRO 14B EM: {correct}/{n} ({100*correct/n:.1f}%)')
    print()
    print(f'Decomposition of outcomes:')
    print(f'  correct AND gold-found: {correct_and_found:>3}  ({100*correct_and_found/n:.0f}%)  [great]')
    print(f'  correct AND gold-NOT-found: {correct_but_not_found:>3}  ({100*correct_but_not_found/n:.0f}%)  [model knew from training data?]')
    print(f'  wrong AND gold-found: {wrong_and_found:>3}  ({100*wrong_and_found/n:.0f}%)  [retrieval ok, system failed to extract]')
    print(f'  wrong AND gold-NOT-found: {wrong_and_not_found:>3}  ({100*wrong_and_not_found/n:.0f}%)  [unanswerable from this corpus]')

    # Per-hop breakdown
    hop_stats = {2: [0,0,0], 3: [0,0,0], 4: [0,0,0]}  # [total, gold_found, em_correct]
    for r in results:
        m = re.match(r'musique_(\d+)hop', r['id'])
        if m:
            h = int(m.group(1))
            hop_stats[h][0] += 1
            if r['gold_in_any_chunk']: hop_stats[h][1] += 1
            if r['predicted_correct']: hop_stats[h][2] += 1
    print(f'\n  per-hop: hop  total  gold_found  EM_correct')
    for h in sorted(hop_stats):
        t, g, c = hop_stats[h]
        if t:
            print(f'           {h}    {t:<5}  {g:>3} ({100*g/t:.0f}%)   {c:>3} ({100*c/t:.0f}%)')

    # Save
    out = REPO / 'results/diag_unanswerable_stratified100.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nfull results: {out}')


if __name__ == '__main__':
    asyncio.run(main())

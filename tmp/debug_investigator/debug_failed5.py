import asyncio, json, pathlib, sys
sys.path.insert(0, 'src')
from arag.core.config import Config
from adaptive_sage.investigator import Investigator
from arag.core.llm import LLMClient

MODEL='/local/yzheng/pnair/.cache/huggingface/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd'
CFG='configs/_runtime/m1_2.sufficiency_pilot_fast.yaml'
OUT=pathlib.Path('tmp/debug_investigator/failed5_trace.log')
SEL=json.loads(pathlib.Path('tmp/debug_investigator/failed5_212833.json').read_text())

async def main():
    cfg=Config.from_yaml(CFG)
    cfg.set('llm.base_url','http://localhost:8001/v1')
    cfg.set('data.chunks_file','data/musique/chunks.json')
    cfg.set('data.index_dir','data/musique/index_e5_base_v2')
    cfg.set('data.embedding_model',MODEL)
    llm_cfg=cfg.get('llm',{})
    llm=LLMClient(model=llm_cfg.get('model','Qwen/Qwen3-8B'), api_key='EMPTY', base_url='http://localhost:8001/v1', temperature=0.0, max_tokens=2048, chat_template_kwargs=llm_cfg.get('chat_template_kwargs'))
    inv=Investigator(cfg,llm)
    with OUT.open('w', encoding='utf-8') as f:
        for i,row in enumerate(SEL,1):
            q=row['question']
            f.write(f"\n===== CASE {i}: {row['id']} =====\n")
            f.write(f"QUESTION: {q}\nGOLD: {row.get('gold_answer')}\nPREV_PRED: {row.get('answer')}\n")
            cap,toks=await inv.investigate_with_usage(sub_question=q, goal='Answer the original question exactly.', prior_facts=[])
            f.write(f"CAPSULE answer={cap.answer!r} fact={cap.fact.text!r} conf={cap.fact.confidence:.3f} self={cap.fact.confidence_self:.3f} support={cap.fact.support_ids} tokens={toks}\n")
            # Re-run loop internals are not exposed, so call the private loop directly for trajectory with same prompt shape.
            # The actual investigate_with_usage above already ran the real path. This prints current code path details from capsule metadata where available.
            f.write(f"retrieved_doc_ids={cap.retrieved_doc_ids}\n")
            f.write(f"support_snippets_len={len(cap.support_snippets)}\n")
    print(OUT)
asyncio.run(main())

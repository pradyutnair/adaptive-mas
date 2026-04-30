"""Bridge-resolution preprocessor.

When the original-Q probe groundedness is low AND the planner's first sub-Q
has low NE coverage from probe chunks, the bridge entity in the multi-hop
question is likely ambiguous (e.g., 'Steven the Sword Fighter' could refer to
Steven Universe or Adventure Time; 'the person Messi was compared to' is
unresolved in any single chunk that mentions Messi but not Maradona).

The resolver runs ONE LLM call on the original question + probe-original
chunks, asks for the resolved bridge entity, and returns a hint string that
gets injected into the planner's context for downstream planning.
"""
from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any
import dspy
from .types import RetrievedChunk


class ResolveBridgeEntity(dspy.Signature):
    """You are reading a multi-hop question. There is a BRIDGE entity in the
question (e.g. 'the publisher of X', 'the singer of Y', 'the person who Z')
that needs to be resolved to a specific named entity before the question can
be answered. The retrieved chunks are about an entity mentioned in the
question (often the wrong end of the bridge).

Find the BRIDGE phrase in the question. Then identify the resolved entity by
looking through the chunks for a name that satisfies the bridge relation. If
the bridge cannot be resolved from the chunks alone, output an empty string.

Examples:
- Question: 'Who wrote Turn Me On by the singer of Come Away with Me?'
  Bridge: 'the singer of Come Away with Me'
  Chunks mention 'Norah Jones... Come Away with Me debut album'
  Resolved: 'Norah Jones'
- Question: 'When was the person Messi was compared to signed by Barcelona?'
  Bridge: 'the person Messi was compared to'
  Chunks mention 'Messi was often compared to Maradona'
  Resolved: 'Diego Maradona'

Output STRICT JSON: {"bridge_phrase": <str>, "bridge_entity": <str>, "rationale": <brief str>}.
If no clear bridge or chunks don't resolve it, output {"bridge_phrase": "", "bridge_entity": "", "rationale": "no_resolution"}.
"""
    original_question: str = dspy.InputField()
    chunks_json: str = dspy.InputField(desc='Top-K chunks for the original question, as JSON')
    resolution_json: str = dspy.OutputField()


def _format_chunks(chunks: list[RetrievedChunk], excerpt_chars: int = 700) -> str:
    return json.dumps([
        {'chunk_id': c.chunk_id, 'text': c.text[:excerpt_chars]}
        for c in chunks
    ], ensure_ascii=False)


def _parse(raw: str) -> dict[str, Any]:
    text = raw.strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@dataclass
class BridgeResolution:
    bridge_phrase: str
    bridge_entity: str
    rationale: str
    tokens: int


def should_resolve_bridge(
    *,
    probe_original_g: float,
    g_low_threshold: float = 0.45,
    has_bridge_pattern: bool = True,
) -> bool:
    """Trigger condition. Bridge resolution fires when:
    - Probe groundedness on the original Q is low (retrieval doesn't fully
      ground the question), AND
    - The question contains a bridge pattern (default true; we don't gate on
      pattern to keep recall high).
    """
    return probe_original_g < g_low_threshold and has_bridge_pattern


async def run_bridge_resolver(
    *,
    resolver_lm: dspy.LM,
    original_question: str,
    original_probe_chunks: list[RetrievedChunk],
) -> BridgeResolution:
    """One LLM call. Returns the resolved bridge entity (may be empty if no
    resolution found in chunks).
    """
    chunks_json = _format_chunks(original_probe_chunks)
    with dspy.context(lm=resolver_lm):
        mod = dspy.Predict(ResolveBridgeEntity)
        pred = await asyncio.to_thread(
            mod,
            original_question=original_question,
            chunks_json=chunks_json,
        )
    try:
        history = resolver_lm.history[-1] if resolver_lm.history else None
        usage = (history or {}).get('usage') or {}
        tokens = int(usage.get('total_tokens', 0))
    except Exception:
        tokens = 0
    obj = _parse(getattr(pred, 'resolution_json', ''))
    return BridgeResolution(
        bridge_phrase=str(obj.get('bridge_phrase', '')).strip(),
        bridge_entity=str(obj.get('bridge_entity', '')).strip(),
        rationale=str(obj.get('rationale', ''))[:200],
        tokens=tokens,
    )

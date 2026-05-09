#!/usr/bin/env python3
"""Integration test for Investigator against live vLLM server."""

import asyncio
import sys
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arag.core.config import Config
from arag.core.llm import LLMClient
from adaptive_sage.investigator import Investigator
from adaptive_sage.types import EvidenceCapsule


async def main():
    config = Config.from_yaml("configs/m1.yaml")
    # Use whichever model is available on the vLLM server
    # (Qwen3-8B requires vLLM >= 0.7; Qwen2.5-7B-Instruct works with vLLM 0.6.6)
    # Use whichever model is available on the vLLM server
    # (Qwen3-8B requires vLLM >= 0.7; Qwen2.5-7B-Instruct works with vLLM 0.6.6)
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # chat_template_kwargs (enable_thinking) is only supported by Qwen3+ vLLM >= 0.7;
    # omit it for Qwen2.5 on vLLM 0.6.6
    llm = LLMClient(
        model=model_name,
        api_key="dummy",
        base_url="http://localhost:8001/v1",
        temperature=0.6,
        max_tokens=4096,
    )
    inv = Investigator(config, llm)

    capsule = await inv.investigate(
        sub_question="Who developed the theory of relativity?",
        goal="Identify the physicist responsible for the theory of relativity",
        prior_facts=[],
    )

    print("Type:", type(capsule).__name__)
    print("Answer:", capsule.answer)
    print("Fact text:", capsule.fact.text)
    print("Confidence:", capsule.fact.confidence)
    print("Support IDs:", capsule.fact.support_ids)
    print("Support snippets count:", len(capsule.support_snippets))

    assert isinstance(capsule, EvidenceCapsule), "Not an EvidenceCapsule"
    assert len(capsule.fact.support_ids) <= inv.evidence_capsule_limit, (
        f"Too many support IDs: {len(capsule.fact.support_ids)} > {inv.evidence_capsule_limit}"
    )
    print("INTEGRATION TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())

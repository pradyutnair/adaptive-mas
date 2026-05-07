"""Stage 3 smoke: lane plumbing in library + orchestrator schema.

No pipeline behaviour change is verified here. Stage 3 only teaches the library
to store/filter the lane field and the orchestrator to emit/validate it.

Asserts:
  A. ExpEntry.lane defaults to "any" and is normalized on load.
  B. library.add(lane="SAS") persists; retrieve(profile, lane="SAS") returns
     entries whose lane is in {"SAS", "any"} only.
  C. retrieve(profile, lane=None) keeps the lane-agnostic legacy behaviour.
  D. save -> load round-trip preserves lane on every entry.
  E. orchestrator.sample_topology emits a topology dict whose `lane` field
     is one of {"SAS", "MAS", "AUTO"}.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path("/local/yzheng/pnair/workspace/adaptive-mas/amas")
sys.path.insert(0, str(ROOT / "src"))

from amas.agents import load_prompts  # noqa: E402
from amas.config import load_env  # noqa: E402
from amas.library import ExpEntry, ExperienceLibrary, _normalize_lane  # noqa: E402
from amas.lm import OpenAIClient, VLLMClient  # noqa: E402
from amas.orchestrator import Orchestrator  # noqa: E402
from amas.retriever import RetrieverClient  # noqa: E402


def assert_or_die(cond: bool, msg: str) -> None:
    if not cond:
        print("FAIL:", msg, file=sys.stderr)
        sys.exit(1)


def test_lane_normalize() -> None:
    cases = [
        (None, "any"),
        ("", "any"),
        ("any", "any"),
        ("AUTO", "any"),
        ("auto", "any"),
        ("SAS", "SAS"),
        ("sas", "SAS"),
        ("MAS", "MAS"),
        ("mas", "MAS"),
        ("garbage", "any"),
    ]
    for raw, want in cases:
        got = _normalize_lane(raw)
        assert_or_die(got == want, f"_normalize_lane({raw!r}) -> {got!r}, want {want!r}")
    print("OK normalize_lane mapping")


def test_library_lane_roundtrip() -> None:
    lib = ExperienceLibrary(max_entries=10)
    eid_sas = lib.add("comparison", "Probe consensus is reliable on simple comparisons.",
                      lane="SAS")
    eid_mas = lib.add("bridge", "Always run QueryDecomposer for multi-hop bridges.",
                      lane="MAS")
    eid_any = lib.add("temporal", "Resolve relative dates by extracting an anchor entity first.",
                      lane="any")
    assert_or_die(all(e for e in (eid_sas, eid_mas, eid_any)), "all add() returned ids")
    by = {e.id: e for e in lib.entries}
    assert_or_die(by[eid_sas].lane == "SAS", "stored lane=SAS")
    assert_or_die(by[eid_mas].lane == "MAS", "stored lane=MAS")
    assert_or_die(by[eid_any].lane == "any", "stored lane=any")

    # retrieve with lane="SAS" -> include SAS + any, NOT MAS
    sas_only = lib.retrieve("comparison", top_k=10, lane="SAS")
    sas_ids = {e.id for e in sas_only}
    assert_or_die(eid_sas in sas_ids, "retrieve(lane=SAS) includes SAS entry")
    assert_or_die(eid_any in sas_ids, "retrieve(lane=SAS) includes any entry")
    assert_or_die(eid_mas not in sas_ids, "retrieve(lane=SAS) excludes MAS entry")

    # retrieve with lane=None -> all entries (legacy)
    legacy = lib.retrieve("bridge", top_k=10, lane=None)
    legacy_ids = {e.id for e in legacy}
    assert_or_die({eid_sas, eid_mas, eid_any} <= legacy_ids,
                  f"retrieve(lane=None) keeps all, got {legacy_ids}")

    # save / load round-trip
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        lib.save(path)
        loaded = ExperienceLibrary.load(path)
        loaded_by = {e.id: e for e in loaded.entries}
        for eid, lane in [(eid_sas, "SAS"), (eid_mas, "MAS"), (eid_any, "any")]:
            assert_or_die(eid in loaded_by, f"round-trip lost id {eid}")
            assert_or_die(loaded_by[eid].lane == lane,
                          f"round-trip lost lane for {eid}: {loaded_by[eid].lane!r}")
    finally:
        path.unlink(missing_ok=True)
    print("OK library lane round-trip + retrieve filter")


async def test_orchestrator_emits_lane() -> None:
    """Stage-3 contract: validated topologies must always carry a lane in
    {SAS, MAS, AUTO}. We do NOT require the model to actively choose SAS/MAS
    yet — Stage 3 is plumbing, so the AUTO default is sufficient. Stage 4
    introduces the policy that drives SAS/MAS selection.
    """
    cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    load_env()

    vcfg = cfg["vllm"]; ocfg = cfg["openai"]; rcfg = cfg["retriever"]
    vllm = VLLMClient(endpoints=vcfg["endpoints"], model=vcfg["model"],
                      max_tokens=vcfg["max_tokens"], temperature=0.3, concurrency=4)
    openai_client = OpenAIClient(model=ocfg["model"], max_tokens=ocfg["max_tokens"],
                                 temperature=ocfg["temperature"], concurrency=4)
    retriever = RetrieverClient(url=rcfg["url"], topk=rcfg["topk"], concurrency=4)
    library = ExperienceLibrary.load(cfg["paths"]["library"])
    prompts = load_prompts(cfg["paths"]["prompts"])
    orch = Orchestrator(vllm=vllm, openai_client=openai_client, retriever=retriever,
                        library=library, prompts=prompts,
                        retriever_topk=rcfg["topk"], library_top_k=5, max_steps=8)
    try:
        cases = [
            "What is the capital of France?",
            "What is the birthplace of the director of the film The Godfather?",
        ]
        for q in cases:
            topo, ids, _ = await orch.sample_topology(q, profile="bridge", temperature=0.0)
            lane = topo.get("lane")
            assert_or_die(lane in ("SAS", "MAS", "AUTO"),
                          f"validated topology must carry lane in {{SAS,MAS,AUTO}}; "
                          f"got {lane!r} for q={q!r}")
            print(f"  q={q[:40]!r:<42} lane={lane}")
    finally:
        await retriever.aclose()
        await vllm.aclose()
        await openai_client.aclose()
    print("OK orchestrator topology carries valid lane (AUTO default acceptable)")


async def main() -> None:
    test_lane_normalize()
    test_library_lane_roundtrip()
    await test_orchestrator_emits_lane()
    print("STAGE-3 SMOKE PASS")


if __name__ == "__main__":
    asyncio.run(main())

"""Tests for adaptive_sage.pipeline helper behavior."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_sage.pipeline import AdaptiveRecursivePipeline
from adaptive_sage.types import EvidenceCapsule, Fact
from arag.core.config import Config


def test_answer_fallback_prefers_fact_memory() -> None:
    facts = [
        Fact(
            text="Markus Zusak wrote The Book Thief.",
            confidence=0.91,
            answer_span="Markus Zusak",
            support_ids=["1"],
            source_step=1,
        )
    ]

    answer, source, cited_fact_ids, confidence = (
        AdaptiveRecursivePipeline._apply_answer_fallback(
            "",
            facts,
            route_draft_answer="The Book Thief",
        )
    )

    assert answer == "Markus Zusak"
    assert source == "fact_memory"
    assert cited_fact_ids == [1]
    assert confidence == 0.91


def test_answer_fallback_does_not_use_route_draft_when_memory_empty() -> None:
    answer, source, cited_fact_ids, confidence = (
        AdaptiveRecursivePipeline._apply_answer_fallback(
            "",
            [],
            route_draft_answer="Sydney",
        )
    )

    assert answer == ""
    assert source == ""
    assert cited_fact_ids == []
    assert confidence == 0.0


def test_answer_object_fallback_populates_citation() -> None:
    facts = [
        Fact(
            text="Markus Zusak wrote The Book Thief.",
            confidence=0.84,
            answer_span="Markus Zusak",
            support_ids=["1"],
            source_step=1,
        )
    ]

    answer_obj = AdaptiveRecursivePipeline._apply_answer_object_fallback(
        {
            "answer": "",
            "cited_fact_ids": [],
            "justification_confidence": 0.0,
            "justification": "",
            "missing_slot": "",
        },
        facts,
    )

    assert answer_obj["answer"] == "Markus Zusak"
    assert answer_obj["cited_fact_ids"] == [1]
    assert answer_obj["fallback_source"] == "fact_memory"
    assert answer_obj["justification_confidence"] == 0.84


def test_parallel_ready_slots_returns_earliest_group() -> None:
    slot_state = [
        {"slot_name": "book_author", "hint": "author", "resolved": False, "dependency_group": 0},
        {"slot_name": "film_director", "hint": "director", "resolved": False, "dependency_group": 0},
        {"slot_name": "birth_place", "hint": "birth place", "resolved": False, "dependency_group": 1},
    ]

    ready = AdaptiveRecursivePipeline._parallel_ready_slots(slot_state)

    assert [slot["slot_name"] for slot in ready] == ["book_author", "film_director"]


def test_replace_fact_swaps_existing_slot_fact() -> None:
    from adaptive_sage.fact_memory import FactMemory

    memory = FactMemory(capacity=4)
    original = Fact(
        text="Markus Zusak wrote The Book Thief.",
        confidence=0.55,
        slot_name="author",
        answer_span="Markus Zusak",
        support_ids=["1"],
        source_step=1,
    )
    memory.add(original)
    replacement = EvidenceCapsule(
        answer="Markus Zusak",
        fact=Fact(
            text="The Book Thief was written by Markus Zusak.",
            confidence=0.88,
            answer_span="Markus Zusak",
            support_ids=["2"],
            source_step=0,
        ),
    )

    added = AdaptiveRecursivePipeline._replace_fact(
        memory,
        replacement,
        step=2,
        slot_name="author",
    )

    assert added is True
    facts = memory.get_all()
    assert len(facts) == 1
    assert facts[0].text == "The Book Thief was written by Markus Zusak."
    assert facts[0].slot_name == "author"


def test_initialise_slot_state_preserves_expected_info_type() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)

    slot_state = pipeline._initialise_slot_state(
        {
            "required_hops": [
                {
                    "slot_name": "author",
                    "hint": "writer of the book",
                    "expected_info_type": "person",
                    "dependency_group": 0,
                },
                {
                    "slot_name": "birthplace",
                    "hint": "birthplace of the author",
                    "expected_info_type": "location",
                    "dependency_group": 1,
                },
            ]
        },
        "location",
    )

    assert slot_state[0]["expected_info_type"] == "person"
    assert slot_state[1]["expected_info_type"] == "birthplace"


def test_initialise_slot_state_refines_generic_expected_info_type() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)

    slot_state = pipeline._initialise_slot_state(
        {
            "required_hops": [
                {
                    "slot_name": "county",
                    "hint": "resolve the county",
                    "expected_info_type": "location",
                    "dependency_group": 0,
                }
            ]
        },
        "location",
    )

    assert slot_state[0]["expected_info_type"] == "county"


def test_select_sufficiency_probe_uses_bridge_first_slot_for_compositional() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    pipeline.sufficiency_bridge_first_probe = True

    slot_state = [
        {
            "slot_name": "author",
            "hint": "writer of the book",
            "expected_info_type": "person",
            "resolved": False,
            "dependency_group": 0,
        },
        {
            "slot_name": "birthplace",
            "hint": "birthplace of the author",
            "expected_info_type": "location",
            "resolved": False,
            "dependency_group": 1,
        },
        {
            "slot_name": "country",
            "hint": "country of the birthplace",
            "expected_info_type": "country",
            "resolved": False,
            "dependency_group": 2,
        },
    ]

    probe = pipeline._select_sufficiency_probe(
        question="Where was the author of The Book Thief born?",
        route={
            "sub_question": "Who wrote The Book Thief?",
            "retrieval_query": "The Book Thief author",
            "goal": "Resolve the author of The Book Thief.",
        },
        slot_state=slot_state,
        target_profile="location",
    )

    assert probe["strategy"] == "bridge_first_typed"
    assert probe["slot_name"] == "author"
    assert probe["expected_info_type"] == "person"


def test_select_sufficiency_probe_uses_direct_probe_for_two_hop_route() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    pipeline.sufficiency_bridge_first_probe = True

    slot_state = [
        {
            "slot_name": "author",
            "hint": "writer of the book",
            "expected_info_type": "person",
            "resolved": False,
            "dependency_group": 0,
        },
        {
            "slot_name": "birthplace",
            "hint": "birthplace of the author",
            "expected_info_type": "birthplace",
            "resolved": False,
            "dependency_group": 1,
        },
    ]

    probe = pipeline._select_sufficiency_probe(
        question="Where was the author of The Book Thief born?",
        route={
            "action": "recurse",
            "sub_question": "Who wrote The Book Thief?",
            "retrieval_query": "The Book Thief author",
            "goal": "Resolve the author first.",
        },
        slot_state=slot_state,
        target_profile="Answer with the exact span.",
    )

    assert probe["strategy"] == "direct_final_slot"
    assert probe["slot_name"] == "birthplace"


def test_select_sufficiency_probe_prefers_direct_probe_when_router_says_single_probe() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    pipeline.sufficiency_bridge_first_probe = True

    slot_state = [
        {
            "slot_name": "author",
            "hint": "writer of the book",
            "expected_info_type": "person",
            "resolved": False,
            "dependency_group": 0,
        },
        {
            "slot_name": "birthplace",
            "hint": "birthplace of the author",
            "expected_info_type": "birthplace",
            "resolved": False,
            "dependency_group": 1,
        },
    ]

    probe = pipeline._select_sufficiency_probe(
        question="Where was the author of The Book Thief born?",
        route={
            "action": "single_probe",
            "sub_question": "Where was the author of The Book Thief born?",
            "retrieval_query": "author of The Book Thief birthplace",
            "goal": "Resolve the birthplace directly.",
        },
        slot_state=slot_state,
        target_profile="Answer with the exact span.",
    )

    assert probe["strategy"] == "direct_final_slot"
    assert probe["slot_name"] == "birthplace"


def test_select_sufficiency_probe_uses_final_slot_for_single_hop() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    pipeline.sufficiency_bridge_first_probe = True

    slot_state = [
        {
            "slot_name": "birthplace",
            "hint": "birthplace of Markus Zusak",
            "expected_info_type": "location",
            "resolved": False,
            "dependency_group": 0,
        }
    ]

    probe = pipeline._select_sufficiency_probe(
        question="Where was Markus Zusak born?",
        route={
            "sub_question": "Where was Markus Zusak born?",
            "retrieval_query": "Markus Zusak birthplace",
            "goal": "Resolve the birthplace.",
        },
        slot_state=slot_state,
        target_profile="location",
    )

    assert probe["strategy"] == "direct_final_slot"
    assert probe["slot_name"] == "birthplace"


def test_build_sufficiency_result_includes_new_metadata() -> None:
    from adaptive_sage.fact_memory import FactMemory

    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    pipeline.sufficiency_threshold = 0.7
    pipeline.investigator = MagicMock()
    pipeline.investigator.evidence_capsule_limit = 4
    pipeline.fact_memory_capacity = 4

    result = pipeline._build_sufficiency_result(
        question_id="q1",
        question="Where was the author of The Book Thief born?",
        answer="Sydney",
        step_trace=[],
        memory=FactMemory(capacity=4),
        subagent_calls=2,
        total_tokens=100,
        orchestrator_tokens=20,
        subagent_tokens=80,
        retrieved_doc_ids=["1"],
        retrieved_docs_total=1,
        route_label="recurse_after_probe",
        sufficiency=0.81,
        sufficiency_components={"source": "typed_bridge_probe_gate"},
        route_target_slot="birthplace",
        slot_state=[
            {
                "slot_name": "author",
                "hint": "writer",
                "expected_info_type": "person",
                "resolved": True,
                "dependency_group": 0,
            },
            {
                "slot_name": "birthplace",
                "hint": "birthplace",
                "expected_info_type": "location",
                "resolved": False,
                "dependency_group": 1,
            },
        ],
        required_hops=[
            {
                "slot_name": "author",
                "hint": "writer",
                "expected_info_type": "person",
                "dependency_group": 0,
            },
            {
                "slot_name": "birthplace",
                "hint": "birthplace",
                "expected_info_type": "location",
                "dependency_group": 1,
            },
        ],
        recurse_steps_used=1,
        probe_strategy="bridge_first_typed",
        probe_slot_name="author",
        planned_hop_count=2,
        slot_sufficiency_score=0.88,
        answer_sufficiency_score=0.09,
        resolved_slots_after_probe=["author"],
    )

    assert result.extras["probe_strategy"] == "bridge_first_typed"
    assert result.extras["planned_hop_count"] == 2
    assert result.extras["slot_sufficiency_score"] == 0.88
    assert result.extras["resolved_slots_after_probe"] == ["author"]
    assert result.slot_resolution == {"author": True, "birthplace": False}


def test_m1_2_sufficiency_runs_end_to_end_with_probe_answer() -> None:
    config = Config(
        {
            "variant": "m1_2_sufficiency",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 5,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "sufficiency_controller": True,
                "sufficiency_threshold": 0.7,
                "sufficiency_max_recurse_steps": 4,
                "sufficiency_min_recurse_steps": 1,
            },
            "llm": {
                "model": "Qwen/Qwen3-8B",
                "base_url": "http://localhost:8001/v1",
                "temperature": 0.0,
                "max_tokens": 8192,
            },
            "data": {
                "chunks_file": "data/musique/chunks.json",
                "index_dir": "data/musique/index_e5_base_v2",
                "embedding_model": "intfloat/e5-base-v2",
            },
        }
    )

    probe_capsule = EvidenceCapsule(
        answer="Sydney",
        fact=Fact(
            text="Markus Zusak was born in Sydney.",
            confidence=0.91,
            answer_span="Sydney",
            support_ids=["1"],
            source_step=0,
        ),
    )

    mock_orchestrator = MagicMock()
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "action": "single_probe",
                "confidence": 0.9,
                "draft_answer": "",
                "sub_question": "Where was Markus Zusak born?",
                "retrieval_query": "Markus Zusak birthplace",
                "goal": "Resolve the birthplace.",
                "answer_type": "location",
                "target_slot": "birthplace",
                "required_hops": [
                    {
                        "slot_name": "birthplace",
                        "hint": "birthplace of Markus Zusak",
                        "expected_info_type": "location",
                        "dependency_group": 0,
                    }
                ],
            },
            11,
        )
    )
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        return_value=(
            {
                "answer": "Sydney",
                "cited_fact_ids": [1],
                "justification_confidence": 0.91,
                "justification": "Grounded in fact 1.",
                "missing_slot": "",
            },
            17,
        )
    )
    mock_orchestrator.assess_probe_sufficiency_with_usage = AsyncMock(
        return_value=({"sufficient": 0.95, "reason": "Fully resolved."}, 5)
    )

    mock_investigator = MagicMock()
    mock_investigator.evidence_capsule_limit = 4
    mock_investigator.investigate_with_usage = AsyncMock(return_value=(probe_capsule, 40))

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        pipeline = AdaptiveRecursivePipeline(config)
        result = asyncio.run(
            pipeline.run("Where was Markus Zusak born?", "q1")
        )

    assert result.route_decision == "answer_from_probe"
    assert result.answer == "Sydney"
    assert result.extras["probe_strategy"] == "full_question_probe"


def test_m1_3_typed_bridge_sufficiency_recurses_with_bridge_fact() -> None:
    config = Config(
        {
            "variant": "m1_3_typed_bridge_sufficiency",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 5,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "sufficiency_controller": True,
                "sufficiency_threshold": 0.7,
                "sufficiency_max_recurse_steps": 4,
                "sufficiency_min_recurse_steps": 1,
                "sufficiency_bridge_first_probe": True,
                "sufficiency_split_assessment": True,
                "sufficiency_typed_one_shot_followup": False,
            },
            "llm": {
                "model": "Qwen/Qwen3-8B",
                "base_url": "http://localhost:8001/v1",
                "temperature": 0.0,
                "max_tokens": 8192,
            },
            "data": {
                "chunks_file": "data/musique/chunks.json",
                "index_dir": "data/musique/index_e5_base_v2",
                "embedding_model": "intfloat/e5-base-v2",
            },
        }
    )

    author_capsule = EvidenceCapsule(
        answer="Markus Zusak",
        fact=Fact(
            text="Markus Zusak wrote The Book Thief.",
            confidence=0.91,
            slot_filled=True,
            answer_span="Markus Zusak",
            support_ids=["1"],
            source_step=0,
        ),
    )
    birthplace_capsule = EvidenceCapsule(
        answer="Sydney",
        fact=Fact(
            text="Markus Zusak was born in Sydney.",
            confidence=0.93,
            slot_filled=True,
            answer_span="Sydney",
            support_ids=["2"],
            source_step=0,
        ),
    )

    mock_orchestrator = MagicMock()
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "action": "recurse",
                "confidence": 0.92,
                "draft_answer": "",
                "sub_question": "Who wrote The Book Thief?",
                "retrieval_query": "The Book Thief author",
                "goal": "Resolve the author of The Book Thief.",
                "answer_type": "location",
                "target_slot": "birthplace",
                "required_hops": [
                    {
                        "slot_name": "author",
                        "hint": "writer of The Book Thief",
                        "expected_info_type": "person",
                        "dependency_group": 0,
                    },
                    {
                        "slot_name": "birthplace",
                        "hint": "birthplace of the author",
                        "expected_info_type": "location",
                        "dependency_group": 1,
                    },
                ],
            },
            13,
        )
    )
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        side_effect=[
            (
                {
                    "answer": "",
                    "cited_fact_ids": [],
                    "justification_confidence": 0.0,
                    "justification": "",
                    "missing_slot": "birthplace",
                },
                19,
            ),
            (
                {
                    "answer": "Sydney",
                    "cited_fact_ids": [2],
                    "justification_confidence": 0.93,
                    "justification": "Grounded in fact 2.",
                    "missing_slot": "",
                },
                23,
            ),
        ]
    )
    mock_orchestrator.assess_typed_probe_state_with_usage = AsyncMock(
        return_value=(
            {
                "slot_sufficient": 0.95,
                "answer_sufficient": 0.1,
                "slot_reason": "The author slot is grounded.",
                "answer_reason": "The birthplace slot is still missing.",
            },
            7,
        )
    )
    mock_orchestrator.decide_with_usage = AsyncMock(
        return_value=(
            {
                "action": "spawn",
                "sub_question": "Where was Markus Zusak born?",
                "retrieval_query": "Markus Zusak birthplace",
                "goal": "Resolve the birthplace of Markus Zusak.",
                "slot_name": "birthplace",
            },
            9,
        )
    )

    mock_investigator = MagicMock()
    mock_investigator.evidence_capsule_limit = 4
    mock_investigator.investigate_with_usage = AsyncMock(
        side_effect=[(author_capsule, 41), (birthplace_capsule, 43)]
    )

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        pipeline = AdaptiveRecursivePipeline(config)
        pipeline._select_sufficiency_probe = MagicMock(
            return_value={
                "sub_question": "Who wrote The Book Thief?",
                "retrieval_query": "The Book Thief author",
                "goal": "Resolve the author of The Book Thief.",
                "strategy": "bridge_first_typed",
                "slot_name": "author",
                "slot_hint": "writer of The Book Thief",
                "expected_info_type": "person",
            }
        )
        result = asyncio.run(
            pipeline.run(
                "Where was the author of The Book Thief born?",
                "q2",
            )
        )

    assert result.route_decision == "recurse_after_probe"
    assert result.answer == "Sydney"
    assert result.extras["probe_strategy"] == "bridge_first_typed"
    assert result.extras["probe_slot_name"] == "author"
    assert result.extras["resolved_slots_after_probe"] == ["author"]
    assert result.extras["slot_sufficiency_score"] > result.extras["answer_sufficiency_score"]
    assert result.slot_resolution == {"author": True, "birthplace": True}


def test_m1_3_typed_bridge_sufficiency_answers_after_one_shot_followup() -> None:
    config = Config(
        {
            "variant": "m1_3_typed_bridge_sufficiency",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 5,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "sufficiency_controller": True,
                "sufficiency_threshold": 0.7,
                "sufficiency_max_recurse_steps": 4,
                "sufficiency_min_recurse_steps": 1,
                "sufficiency_bridge_first_probe": True,
                "sufficiency_split_assessment": True,
                "sufficiency_typed_one_shot_followup": True,
            },
            "llm": {
                "model": "Qwen/Qwen3-8B",
                "base_url": "http://localhost:8001/v1",
                "temperature": 0.0,
                "max_tokens": 8192,
            },
            "data": {
                "chunks_file": "data/musique/chunks.json",
                "index_dir": "data/musique/index_e5_base_v2",
                "embedding_model": "intfloat/e5-base-v2",
            },
        }
    )

    author_capsule = EvidenceCapsule(
        answer="Markus Zusak",
        fact=Fact(
            text="Markus Zusak wrote The Book Thief.",
            confidence=0.91,
            slot_filled=True,
            answer_span="Markus Zusak",
            support_ids=["1"],
            source_step=0,
        ),
    )
    birthplace_capsule = EvidenceCapsule(
        answer="Sydney",
        fact=Fact(
            text="Markus Zusak was born in Sydney.",
            confidence=0.93,
            slot_filled=True,
            answer_span="Sydney",
            support_ids=["2"],
            source_step=0,
        ),
    )

    mock_orchestrator = MagicMock()
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "action": "recurse",
                "confidence": 0.92,
                "draft_answer": "",
                "sub_question": "Who wrote The Book Thief?",
                "retrieval_query": "The Book Thief author",
                "goal": "Resolve the author of The Book Thief.",
                "answer_type": "location",
                "target_slot": "birthplace",
                "required_hops": [
                    {
                        "slot_name": "author",
                        "hint": "writer of The Book Thief",
                        "expected_info_type": "person",
                        "dependency_group": 0,
                    },
                    {
                        "slot_name": "birthplace",
                        "hint": "birthplace of the author",
                        "expected_info_type": "location",
                        "dependency_group": 1,
                    },
                ],
            },
            13,
        )
    )
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        side_effect=[
            (
                {
                    "answer": "",
                    "cited_fact_ids": [],
                    "justification_confidence": 0.0,
                    "justification": "",
                    "missing_slot": "birthplace",
                },
                19,
            ),
            (
                {
                    "answer": "Sydney",
                    "cited_fact_ids": [2],
                    "justification_confidence": 0.93,
                    "justification": "Grounded in fact 2.",
                    "missing_slot": "",
                },
                23,
            ),
        ]
    )
    mock_orchestrator.assess_typed_probe_state_with_usage = AsyncMock(
        return_value=(
            {
                "slot_sufficient": 0.95,
                "answer_sufficient": 0.1,
                "slot_reason": "The author slot is grounded.",
                "answer_reason": "The birthplace slot is still missing.",
            },
            7,
        )
    )
    mock_orchestrator.propose_spawn = AsyncMock(
        return_value=(
            {
                "action": "spawn",
                "sub_question": "Where was Markus Zusak born?",
                "retrieval_query": "Markus Zusak birthplace",
                "goal": "Resolve the birthplace of Markus Zusak.",
                "slot_name": "birthplace",
            },
            9,
        )
    )

    mock_investigator = MagicMock()
    mock_investigator.evidence_capsule_limit = 4
    mock_investigator.investigate_with_usage = AsyncMock(
        side_effect=[(author_capsule, 41), (birthplace_capsule, 43)]
    )

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        pipeline = AdaptiveRecursivePipeline(config)
        pipeline._select_sufficiency_probe = MagicMock(
            return_value={
                "sub_question": "Who wrote The Book Thief?",
                "retrieval_query": "The Book Thief author",
                "goal": "Resolve the author of The Book Thief.",
                "strategy": "bridge_first_typed",
                "slot_name": "author",
                "slot_hint": "writer of The Book Thief",
                "expected_info_type": "person",
            }
        )
        result = asyncio.run(
            pipeline.run(
                "Where was the author of The Book Thief born?",
                "q3",
            )
        )

    assert result.route_decision == "answer_after_one_shot_followup"
    assert result.answer == "Sydney"
    assert result.slot_resolution == {"author": True, "birthplace": True}

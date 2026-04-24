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


def test_m1_6_hybrid_sufficiency_answers_after_typed_plan_exec() -> None:
    config = Config(
        {
            "variant": "m1_6_hybrid_sufficiency",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 5,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "sufficiency_controller": True,
                "sufficiency_threshold": 0.65,
                "sufficiency_max_recurse_steps": 4,
                "sufficiency_min_recurse_steps": 1,
                "sufficiency_bridge_first_probe": True,
                "sufficiency_split_assessment": True,
                "sufficiency_typed_plan_exec_on_hard": True,
                "sufficiency_recurse_only_after_plan_exec_failure": True,
                "sufficiency_max_plan_exec_steps": 4,
                "sufficiency_max_recovery_steps": 1,
                "sufficiency_enable_slot_rewrite": False,
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
        side_effect=[
            (
                {
                    "slot_sufficient": 0.95,
                    "answer_sufficient": 0.1,
                    "slot_reason": "The author slot is grounded.",
                    "answer_reason": "The birthplace slot is still missing.",
                },
                7,
            ),
            (
                {
                    "slot_sufficient": 0.95,
                    "answer_sufficient": 0.95,
                    "slot_reason": "The birthplace slot is grounded.",
                    "answer_reason": "The full answer is now grounded.",
                },
                7,
            ),
        ]
    )
    mock_orchestrator.propose_spawn = AsyncMock()

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
                "q_m16a",
            )
        )

    assert result.route_decision == "answer_after_typed_plan_exec"
    assert result.answer == "Sydney"
    assert result.slot_resolution == {"author": True, "birthplace": True}
    assert result.extras["typed_plan_exec"] is True
    assert result.extras["num_plan_exec_steps"] == 1
    assert result.extras["num_rewrites"] == 0


def test_m1_6_hybrid_sufficiency_recurses_only_after_plan_exec_failure() -> None:
    config = Config(
        {
            "variant": "m1_6_hybrid_sufficiency",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 5,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "sufficiency_controller": True,
                "sufficiency_threshold": 0.65,
                "sufficiency_max_recurse_steps": 4,
                "sufficiency_min_recurse_steps": 1,
                "sufficiency_bridge_first_probe": True,
                "sufficiency_split_assessment": True,
                "sufficiency_typed_plan_exec_on_hard": True,
                "sufficiency_recurse_only_after_plan_exec_failure": True,
                "sufficiency_max_plan_exec_steps": 4,
                "sufficiency_max_recovery_steps": 1,
                "sufficiency_enable_slot_rewrite": False,
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
    failed_birthplace_capsule = EvidenceCapsule(
        answer="",
        fact=Fact(
            text="",
            confidence=0.0,
            slot_filled=False,
            answer_span="",
            support_ids=[],
            source_step=0,
        ),
    )
    recovered_birthplace_capsule = EvidenceCapsule(
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
                    "answer": "",
                    "cited_fact_ids": [],
                    "justification_confidence": 0.0,
                    "justification": "",
                    "missing_slot": "birthplace",
                },
                13,
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
        side_effect=[
            (
                {
                    "slot_sufficient": 0.95,
                    "answer_sufficient": 0.1,
                    "slot_reason": "The author slot is grounded.",
                    "answer_reason": "The birthplace slot is still missing.",
                },
                7,
            ),
            (
                {
                    "slot_sufficient": 0.0,
                    "answer_sufficient": 0.0,
                    "slot_reason": "The birthplace slot is unresolved.",
                    "answer_reason": "The final answer is still unsupported.",
                },
                7,
            ),
        ]
    )
    mock_orchestrator.propose_spawn = AsyncMock()

    mock_investigator = MagicMock()
    mock_investigator.evidence_capsule_limit = 4
    mock_investigator.investigate_with_usage = AsyncMock(
        side_effect=[
            (author_capsule, 41),
            (failed_birthplace_capsule, 11),
            (recovered_birthplace_capsule, 43),
        ]
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
                "q_m16b",
            )
        )

    assert result.route_decision == "recurse_after_typed_plan_exec"
    assert result.answer == "Sydney"
    assert result.slot_resolution == {"author": True, "birthplace": True}
    assert result.extras["typed_plan_exec"] is True
    assert result.extras["num_plan_exec_steps"] == 1
    assert result.extras["recurse_steps_used"] == 1


def test_select_execution_mode_prefers_direct_for_simple_questions() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    pipeline.enable_recursive_recovery = True
    pipeline.direct_probe_threshold = 0.45
    pipeline.typed_plan_exec_threshold = 0.55
    pipeline.recovery_trigger_threshold = 0.45

    mode = pipeline._select_execution_mode(
        {
            "execution_mode": "direct_probe",
            "compositionality_score": 0.2,
            "bridge_uncertainty_score": 0.1,
            "expected_hop_count": 1,
        },
        [{"slot_name": "birthplace", "resolved": False}],
    )

    assert mode == "direct_probe"


def test_select_execution_mode_blocks_initial_direct_probe_on_multihop() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    pipeline.enable_recursive_recovery = True
    pipeline.direct_probe_threshold = 0.45
    pipeline.typed_plan_exec_threshold = 0.55
    pipeline.recovery_trigger_threshold = 0.45

    mode = pipeline._select_execution_mode(
        {
            "execution_mode": "direct_probe",
            "compositionality_score": 0.5,
            "bridge_uncertainty_score": 0.2,
            "expected_hop_count": 2,
        },
        [
            {"slot_name": "author", "resolved": False},
            {"slot_name": "birthplace", "resolved": False},
        ],
    )

    assert mode == "typed_plan_exec"


def test_conflicting_slot_names_detects_multiple_answers() -> None:
    facts = [
        Fact(
            text="Markus Zusak was born in Sydney.",
            confidence=0.9,
            slot_name="birthplace",
            answer_span="Sydney",
            support_ids=["1"],
            source_step=1,
        ),
        Fact(
            text="Markus Zusak was born in Vienna.",
            confidence=0.8,
            slot_name="birthplace",
            answer_span="Vienna",
            support_ids=["2"],
            source_step=2,
        ),
    ]

    conflicts = AdaptiveRecursivePipeline._conflicting_slot_names(
        facts,
        [{"slot_name": "birthplace", "resolved": False}],
    )

    assert conflicts == ["birthplace"]


def test_slot_guided_plan_injects_anchor_into_placeholder_templates() -> None:
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    slot_state = [
        {
            "slot_name": "university",
            "hint": "Institution affiliated with Daniel Thürer",
            "expected_info_type": "university",
            "resolved": True,
            "dependency_group": 0,
            "sub_question": "Which university is Daniel Thürer affiliated with?",
            "retrieval_query": "Daniel Thürer university",
            "goal": "Find the name of the university",
        },
        {
            "slot_name": "student_count",
            "hint": "Number of students enrolled at the university",
            "expected_info_type": "student_count",
            "resolved": False,
            "dependency_group": 1,
            "sub_question": "How many students attend [university]?",
            "retrieval_query": "[university] student population",
            "goal": "Retrieve enrollment statistics for [university]",
        },
    ]
    facts = [
        Fact(
            text="Daniel Thürer is a professor emeritus at the University of Zurich.",
            confidence=0.91,
            slot_name="university",
            answer_span="University of Zurich",
            support_ids=["1"],
            source_step=0,
        )
    ]

    plan = pipeline._slot_guided_plan(
        question="How many students attend Daniel Thürer's university?",
        slot_state=slot_state,
        slot_name="student_count",
        target_profile="number",
        facts=facts,
    )

    assert plan["sub_question"] == "How many students attend University of Zurich?"
    assert plan["retrieval_query"] == "University of Zurich student population"
    assert plan["goal"] == "Retrieve enrollment statistics for University of Zurich"


def test_m2_1_structure_aware_typed_plan_exec_answers_without_recovery() -> None:
    config = Config(
        {
            "variant": "m2_1_structure_aware_adaptive",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 5,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "execution_mode_controller": True,
                "direct_probe_threshold": 0.45,
                "typed_plan_exec_threshold": 0.55,
                "recovery_trigger_threshold": 0.45,
                "max_plan_exec_steps": 4,
                "max_recovery_steps": 2,
                "enable_slot_rewrite": True,
                "enable_recursive_recovery": True,
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
            slot_name="author",
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
            slot_name="birthplace",
            answer_span="Sydney",
            support_ids=["2"],
            source_step=1,
        ),
    )

    mock_orchestrator = MagicMock()
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "execution_mode": "typed_plan_exec",
                "action": "recurse",
                "compositionality_score": 0.88,
                "bridge_uncertainty_score": 0.22,
                "expected_hop_count": 2,
                "confidence": 0.93,
                "draft_answer": "",
                "sub_question": "Who wrote The Book Thief?",
                "retrieval_query": "The Book Thief author",
                "goal": "Resolve the author.",
                "answer_type": "location",
                "target_slot": "birthplace",
                "required_hops": [
                    {
                        "slot_name": "author",
                        "hint": "writer of The Book Thief",
                        "expected_info_type": "person",
                        "dependency_group": 0,
                        "sub_question": "Who wrote The Book Thief?",
                        "retrieval_query": "The Book Thief author",
                        "goal": "Resolve the author.",
                    },
                    {
                        "slot_name": "birthplace",
                        "hint": "birthplace of the author",
                        "expected_info_type": "location",
                        "dependency_group": 1,
                        "sub_question": "Where was Markus Zusak born?",
                        "retrieval_query": "Markus Zusak birthplace",
                        "goal": "Resolve the birthplace.",
                    },
                ],
            },
            11,
        )
    )
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        return_value=(
            {
                "answer": "Sydney",
                "cited_fact_ids": [2],
                "justification_confidence": 0.94,
                "justification": "Grounded in fact 2.",
                "missing_slot": "",
            },
            17,
        )
    )
    mock_orchestrator.propose_spawn = AsyncMock()

    mock_investigator = MagicMock()
    mock_investigator.evidence_capsule_limit = 4
    mock_investigator.investigate_with_usage = AsyncMock(
        side_effect=[(author_capsule, 31), (birthplace_capsule, 33)]
    )

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        pipeline = AdaptiveRecursivePipeline(config)
        result = asyncio.run(
            pipeline.run("Where was the author of The Book Thief born?", "q_m21")
        )

    assert result.route_decision == "answer_after_typed_plan_exec"
    assert result.answer == "Sydney"
    assert result.extras["execution_mode"] == "typed_plan_exec"
    assert result.extras["num_plan_exec_steps"] == 2
    assert result.extras["num_rewrites"] == 0
    assert result.slot_resolution == {"author": True, "birthplace": True}
    assert mock_orchestrator.propose_spawn.await_count == 0


def test_m2_1_structure_aware_recovery_targets_only_unresolved_slots() -> None:
    config = Config(
        {
            "variant": "m2_1_structure_aware_adaptive",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 5,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "execution_mode_controller": True,
                "direct_probe_threshold": 0.45,
                "typed_plan_exec_threshold": 0.55,
                "recovery_trigger_threshold": 0.45,
                "max_plan_exec_steps": 4,
                "max_recovery_steps": 2,
                "enable_slot_rewrite": False,
                "enable_recursive_recovery": True,
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
            slot_name="author",
            answer_span="Markus Zusak",
            support_ids=["1"],
            source_step=0,
        ),
    )
    unresolved_birthplace = EvidenceCapsule(
        answer="",
        fact=Fact(
            text="No grounded birthplace found.",
            confidence=0.2,
            slot_filled=False,
            slot_name="birthplace",
            answer_span="",
            support_ids=["2"],
            source_step=1,
        ),
    )
    recovered_birthplace = EvidenceCapsule(
        answer="Sydney",
        fact=Fact(
            text="Markus Zusak was born in Sydney.",
            confidence=0.94,
            slot_filled=True,
            slot_name="birthplace",
            answer_span="Sydney",
            support_ids=["3"],
            source_step=2,
        ),
    )

    mock_orchestrator = MagicMock()
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "execution_mode": "typed_plan_exec",
                "action": "recurse",
                "compositionality_score": 0.88,
                "bridge_uncertainty_score": 0.3,
                "expected_hop_count": 2,
                "confidence": 0.93,
                "draft_answer": "",
                "sub_question": "Who wrote The Book Thief?",
                "retrieval_query": "The Book Thief author",
                "goal": "Resolve the author.",
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
            11,
        )
    )
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        side_effect=[
            (
                {
                    "answer": "",
                    "cited_fact_ids": [],
                    "justification_confidence": 0.1,
                    "justification": "",
                    "missing_slot": "birthplace",
                },
                17,
            ),
            (
                {
                    "answer": "Sydney",
                    "cited_fact_ids": [3],
                    "justification_confidence": 0.94,
                    "justification": "Grounded in fact 3.",
                    "missing_slot": "",
                },
                19,
            ),
        ]
    )
    mock_orchestrator.decide_with_usage = AsyncMock(
        return_value=(
            {
                "action": "spawn",
                "sub_question": "Where was Markus Zusak born?",
                "retrieval_query": "Markus Zusak birthplace",
                "goal": "Resolve the birthplace.",
                "slot_name": "birthplace",
            },
            9,
        )
    )

    mock_investigator = MagicMock()
    mock_investigator.evidence_capsule_limit = 4
    mock_investigator.investigate_with_usage = AsyncMock(
        side_effect=[
            (author_capsule, 31),
            (unresolved_birthplace, 33),
            (recovered_birthplace, 35),
        ]
    )

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        pipeline = AdaptiveRecursivePipeline(config)
        result = asyncio.run(
            pipeline.run("Where was the author of The Book Thief born?", "q_m21_recover")
        )

    pending_slots = mock_orchestrator.decide_with_usage.await_args.kwargs["pending_slots"]
    assert [slot["slot_name"] for slot in pending_slots] == ["birthplace"]
    assert result.route_decision == "targeted_recursive_recovery"
    assert result.answer == "Sydney"
    assert result.extras["recovery_trigger"] == "unresolved_slots_after_plan_exec"
    assert result.extras["recovery_input_slots"] == ["birthplace"]
    assert result.extras["unresolved_slots"] == []


def test_m3_1a_structure_slot_exec_stops_after_high_answer_sufficiency() -> None:
    config = Config(
        {
            "variant": "m3_1a_structure_adaptive_slot_exec",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 4,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "execution_mode_controller": True,
                "direct_probe_threshold": 0.45,
                "typed_plan_exec_threshold": 0.55,
                "recovery_trigger_threshold": 0.70,
                "max_plan_exec_steps": 4,
                "max_recovery_steps": 2,
                "enable_slot_rewrite": False,
                "enable_recursive_recovery": False,
                "assess_after_plan_step": True,
                "structure_slot_guided_recovery": True,
                "recover_conflicting_slots": True,
            },
            "llm": {
                "model": "Qwen/Qwen3-8B",
                "base_url": "http://localhost:8001/v1",
                "temperature": 0.0,
                "max_tokens": 768,
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
            text="The Book Thief was written by Markus Zusak.",
            confidence=0.94,
            slot_filled=True,
            slot_name="author",
            answer_span="Markus Zusak",
            support_ids=["1"],
            source_step=0,
        ),
    )

    mock_orchestrator = MagicMock()
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "execution_mode": "typed_plan_exec",
                "action": "recurse",
                "compositionality_score": 0.8,
                "bridge_uncertainty_score": 0.2,
                "expected_hop_count": 2,
                "target_slot": "author",
                "required_hops": [
                    {
                        "slot_name": "author",
                        "hint": "writer of The Book Thief",
                        "expected_info_type": "person",
                        "dependency_group": 0,
                        "sub_question": "Who wrote The Book Thief?",
                        "retrieval_query": "The Book Thief author",
                        "goal": "Resolve the author.",
                    },
                    {
                        "slot_name": "author_alias",
                        "hint": "redundant alias slot",
                        "expected_info_type": "person",
                        "dependency_group": 1,
                        "sub_question": "What is another credited author form?",
                        "retrieval_query": "The Book Thief credited author",
                        "goal": "Resolve the final answer alias.",
                    },
                ],
            },
            11,
        )
    )
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        return_value=(
            {
                "answer": "Markus Zusak",
                "cited_fact_ids": [1],
                "justification_confidence": 0.95,
                "justification": "Grounded in fact 1.",
                "missing_slot": "",
            },
            13,
        )
    )
    mock_orchestrator.assess_typed_probe_state_with_usage = AsyncMock(
        return_value=(
            {
                "slot_sufficient": 0.96,
                "answer_sufficient": 0.92,
                "slot_reason": "The author slot is grounded.",
                "answer_reason": "The final answer is already resolved.",
            },
            7,
        )
    )
    mock_orchestrator.propose_spawn = AsyncMock()

    mock_investigator = MagicMock()
    mock_investigator.evidence_capsule_limit = 4
    mock_investigator.investigate_with_usage = AsyncMock(return_value=(author_capsule, 31))
    mock_orchestrator.get_usage_totals.return_value = {"prompt_tokens": 40, "completion_tokens": 20}
    mock_investigator.get_usage_totals.return_value = {"prompt_tokens": 30, "completion_tokens": 10}

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        pipeline = AdaptiveRecursivePipeline(config)
        result = asyncio.run(pipeline.run("Who wrote The Book Thief?", "q_m31a"))

    assert result.route_decision == "answer_after_typed_plan_exec"
    assert result.answer == "Markus Zusak"
    assert result.extras["controller"] == "structure_adaptive_slot_exec"
    assert result.extras["num_plan_exec_steps"] == 1
    assert result.extras["num_rewrites"] == 0
    assert result.prompt_tokens == 70
    assert result.completion_tokens == 30


def test_m3_1b_structure_slot_exec_rewrites_only_after_slot_failure() -> None:
    config = Config(
        {
            "variant": "m3_1b_structure_adaptive_slot_exec",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {
                "evidence_capsule_limit": 4,
                "search_top_k": 4,
                "min_fact_confidence": 0.5,
            },
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": {
                "execution_mode_controller": True,
                "direct_probe_threshold": 0.45,
                "typed_plan_exec_threshold": 0.55,
                "recovery_trigger_threshold": 0.70,
                "max_plan_exec_steps": 4,
                "max_recovery_steps": 2,
                "enable_slot_rewrite": True,
                "enable_recursive_recovery": False,
                "assess_after_plan_step": True,
                "structure_slot_guided_recovery": True,
                "recover_conflicting_slots": True,
            },
            "llm": {
                "model": "Qwen/Qwen3-8B",
                "base_url": "http://localhost:8001/v1",
                "temperature": 0.0,
                "max_tokens": 768,
            },
            "data": {
                "chunks_file": "data/musique/chunks.json",
                "index_dir": "data/musique/index_e5_base_v2",
                "embedding_model": "intfloat/e5-base-v2",
            },
        }
    )

    weak_capsule = EvidenceCapsule(
        answer="",
        fact=Fact(
            text="",
            confidence=0.0,
            slot_filled=False,
            slot_name="birthplace",
            answer_span="",
            support_ids=[],
            source_step=0,
        ),
    )
    strong_capsule = EvidenceCapsule(
        answer="Sydney",
        fact=Fact(
            text="Markus Zusak was born in Sydney.",
            confidence=0.94,
            slot_filled=True,
            slot_name="birthplace",
            answer_span="Sydney",
            support_ids=["2"],
            source_step=1,
        ),
    )

    mock_orchestrator = MagicMock()
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "execution_mode": "typed_plan_exec",
                "action": "recurse",
                "compositionality_score": 0.75,
                "bridge_uncertainty_score": 0.3,
                "expected_hop_count": 1,
                "target_slot": "birthplace",
                "required_hops": [
                    {
                        "slot_name": "birthplace",
                        "hint": "birthplace of Markus Zusak",
                        "expected_info_type": "location",
                        "dependency_group": 0,
                        "sub_question": "Where was Markus Zusak born?",
                        "retrieval_query": "Markus Zusak birthplace",
                        "goal": "Resolve the birthplace.",
                    }
                ],
            },
            9,
        )
    )
    mock_orchestrator.propose_spawn = AsyncMock(
        return_value=(
            {
                "action": "spawn",
                "sub_question": "In which city was Markus Zusak born?",
                "retrieval_query": "Markus Zusak born city",
                "goal": "Resolve the birthplace city.",
                "slot_name": "birthplace",
            },
            5,
        )
    )
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        return_value=(
            {
                "answer": "Sydney",
                "cited_fact_ids": [1],
                "justification_confidence": 0.94,
                "justification": "Grounded in fact 1.",
                "missing_slot": "",
            },
            11,
        )
    )
    mock_orchestrator.assess_typed_probe_state_with_usage = AsyncMock(
        return_value=(
            {
                "slot_sufficient": 0.95,
                "answer_sufficient": 0.91,
                "slot_reason": "The rewritten slot is grounded.",
                "answer_reason": "The answer is now grounded.",
            },
            6,
        )
    )

    mock_investigator = MagicMock()
    mock_investigator.evidence_capsule_limit = 4
    mock_investigator.investigate_with_usage = AsyncMock(
        side_effect=[(weak_capsule, 25), (strong_capsule, 29)]
    )

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        pipeline = AdaptiveRecursivePipeline(config)
        result = asyncio.run(
            pipeline.run("Where was Markus Zusak born?", "q_m31b")
        )

    assert result.route_decision == "answer_after_typed_plan_exec"
    assert result.answer == "Sydney"
    assert result.extras["num_rewrites"] == 1
    assert result.extras["num_plan_exec_steps"] == 1
    assert mock_orchestrator.propose_spawn.await_count == 1



def _slot_cap(answer: str, *, filled: bool = True, confidence: float = 0.9) -> EvidenceCapsule:
    return EvidenceCapsule(
        answer=answer,
        fact=Fact(
            text=f"{answer} is supported.",
            confidence=confidence,
            slot_filled=filled,
            answer_span=answer,
            support_ids=[f"doc_{answer}"],
        ),
        support_snippets=[f"{answer} snippet"],
        retrieved_doc_ids=[f"doc_{answer}"],
        retrieved_docs_total=1,
    )


class _FakeSlotDagOrchestrator:
    def __init__(
        self,
        route: dict,
        *,
        answer: str = "Sydney",
        answer_sufficient: float = 1.0,
        fallback_hops: list[dict] | None = None,
    ) -> None:
        self.route = route
        self.answer = answer
        self.answer_sufficient = answer_sufficient
        self.fallback_hops = fallback_hops or []
        self.fallback_calls = 0
        self.propose_calls = 0

    async def route_with_usage(self, **kwargs):
        return self.route, 7

    async def decompose_slot_dag_with_usage(self, **kwargs):
        self.fallback_calls += 1
        return self.fallback_hops, 5

    async def generate_answer_object_with_usage(self, **kwargs):
        return {
            "answer": self.answer,
            "cited_fact_ids": [1] if kwargs.get("facts") else [],
            "justification_confidence": self.answer_sufficient,
            "justification": "grounded",
            "missing_slot": "",
        }, 3

    async def assess_typed_probe_state_with_usage(self, **kwargs):
        return {
            "slot_sufficient": 1.0,
            "answer_sufficient": self.answer_sufficient,
            "slot_reason": "ok",
            "answer_reason": "ok",
        }, 2

    async def propose_spawn(self, **kwargs):
        self.propose_calls += 1
        slot = kwargs["pending_slots"][0]
        slot_name = slot["slot_name"]
        return {
            "action": "spawn",
            "sub_question": f"Retry {slot_name}?",
            "retrieval_query": f"retry {slot_name}",
            "goal": f"Resolve {slot_name}.",
            "slot_name": slot_name,
        }, 4


class _FakeSlotDagInvestigator:
    evidence_capsule_limit = 4

    def __init__(self, capsules_by_slot: dict[str, list[EvidenceCapsule]]) -> None:
        self.capsules_by_slot = capsules_by_slot
        self.calls: list[dict] = []

    async def investigate_with_usage(self, **kwargs):
        slot_name = kwargs.get("slot_name", "")
        self.calls.append(
            {
                "slot_name": slot_name,
                "prior_count": len(kwargs.get("prior_facts", [])),
                "sub_question": kwargs.get("sub_question", ""),
            }
        )
        queue = self.capsules_by_slot.setdefault(slot_name, [_slot_cap(slot_name or "answer")])
        capsule = queue.pop(0) if len(queue) > 1 else queue[0]
        return capsule, 11

    def reset_usage_totals(self) -> None:
        pass

    def get_usage_totals(self) -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0}


def _make_slot_dag_pipeline(route: dict, capsules_by_slot: dict[str, list[EvidenceCapsule]], **orch_kwargs):
    pipeline = AdaptiveRecursivePipeline.__new__(AdaptiveRecursivePipeline)
    pipeline.config = Config({})
    pipeline.variant = "m1_7_slot_dag_sufficiency"
    pipeline.orchestrator = _FakeSlotDagOrchestrator(route, **orch_kwargs)
    pipeline.investigator = _FakeSlotDagInvestigator(capsules_by_slot)
    pipeline.max_steps = 4
    pipeline.max_total_tokens = 30000
    pipeline.fact_memory_capacity = 4
    pipeline.fact_memory_strategy = "salience"
    pipeline.direct_probe_threshold = 0.45
    pipeline.direct_answer_threshold = 0.65
    pipeline.sufficiency_threshold = 0.65
    pipeline.slot_dag_max_steps = 4
    pipeline.slot_dag_max_parallel = 2
    pipeline.slot_dag_max_retries_per_slot = 1
    pipeline.slot_dag_search_top_k = 4
    pipeline.slot_dag_max_read = 4
    pipeline.slot_dag_recovery_steps = 0
    pipeline.bootstrap_search_top_k = 4
    pipeline.bootstrap_max_read = 4
    return pipeline


def test_slot_dag_easy_route_collapses_to_one_probe() -> None:
    route = {
        "action": "single_probe",
        "confidence": 0.9,
        "draft_answer": "",
        "sub_question": "Where was Markus Zusak born?",
        "retrieval_query": "Markus Zusak birthplace",
        "goal": "Find the birthplace.",
        "answer_type": "city",
        "target_slot": "birthplace",
        "execution_mode": "direct_probe",
        "compositionality_score": 0.2,
        "expected_hop_count": 1,
        "required_hops": [
            {
                "slot_name": "birthplace",
                "hint": "birthplace of Markus Zusak",
                "expected_info_type": "city",
                "dependency_group": 0,
                "sub_question": "Where was Markus Zusak born?",
                "retrieval_query": "Markus Zusak birthplace",
                "goal": "Find the birthplace.",
            }
        ],
    }
    pipeline = _make_slot_dag_pipeline(route, {"birthplace": [_slot_cap("Sydney")]})

    result = asyncio.run(pipeline._run_slot_dag_sufficiency("Where was Markus Zusak born?", "q1"))

    assert result.num_subagent_calls == 1
    assert pipeline.investigator.calls[0]["slot_name"] == "birthplace"
    assert result.extras["collapsed_to_single_agent"] is True
    assert result.extras["slot_exec_steps"] == 0


def test_slot_dag_sequential_groups_pass_prior_facts_forward() -> None:
    route = {
        "action": "recurse",
        "confidence": 0.8,
        "draft_answer": "",
        "answer_type": "city",
        "target_slot": "birthplace",
        "execution_mode": "typed_plan_exec",
        "compositionality_score": 0.9,
        "expected_hop_count": 2,
        "required_hops": [
            {"slot_name": "author", "hint": "writer of The Book Thief", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who wrote The Book Thief?", "retrieval_query": "The Book Thief author", "goal": "Find the author."},
            {"slot_name": "birthplace", "hint": "birthplace of [author]", "expected_info_type": "city", "dependency_group": 1, "sub_question": "Where was [author] born?", "retrieval_query": "[author] birthplace", "goal": "Find the birthplace."},
        ],
    }
    pipeline = _make_slot_dag_pipeline(route, {"author": [_slot_cap("Markus Zusak")], "birthplace": [_slot_cap("Sydney")]})

    result = asyncio.run(pipeline._run_slot_dag_sufficiency("Where was the author of The Book Thief born?", "q2"))

    assert [call["slot_name"] for call in pipeline.investigator.calls] == ["author", "birthplace"]
    assert pipeline.investigator.calls[1]["prior_count"] == 1
    assert result.extras["slot_exec_steps"] == 2


def test_slot_dag_parallel_group_uses_shared_memory_snapshot() -> None:
    route = {
        "action": "recurse",
        "confidence": 0.8,
        "draft_answer": "",
        "answer_type": "person",
        "target_slot": "comparison",
        "execution_mode": "typed_plan_exec",
        "compositionality_score": 0.9,
        "expected_hop_count": 2,
        "required_hops": [
            {"slot_name": "director_a", "hint": "director of film A", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who directed film A?", "retrieval_query": "film A director", "goal": "Find director A."},
            {"slot_name": "director_b", "hint": "director of film B", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who directed film B?", "retrieval_query": "film B director", "goal": "Find director B."},
        ],
    }
    pipeline = _make_slot_dag_pipeline(route, {"director_a": [_slot_cap("Alice")], "director_b": [_slot_cap("Bob")]})

    result = asyncio.run(pipeline._run_slot_dag_sufficiency("Which directors made the two films?", "q3"))

    assert {call["slot_name"] for call in pipeline.investigator.calls[:2]} == {"director_a", "director_b"}
    assert [call["prior_count"] for call in pipeline.investigator.calls[:2]] == [0, 0]
    assert result.extras["parallel_batches"] == 1


def test_slot_dag_failed_slot_retries_once() -> None:
    route = {
        "action": "recurse",
        "confidence": 0.8,
        "draft_answer": "",
        "answer_type": "person",
        "target_slot": "author",
        "execution_mode": "typed_plan_exec",
        "compositionality_score": 0.9,
        "expected_hop_count": 1,
        "required_hops": [
            {"slot_name": "author", "hint": "writer of The Book Thief", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who wrote The Book Thief?", "retrieval_query": "The Book Thief author", "goal": "Find the author."},
        ],
    }
    pipeline = _make_slot_dag_pipeline(route, {"author": [_slot_cap("", filled=False, confidence=0.0), _slot_cap("Markus Zusak")]})

    result = asyncio.run(pipeline._run_slot_dag_sufficiency("Who wrote The Book Thief?", "q4"))

    assert pipeline.orchestrator.propose_calls == 1
    assert [call["slot_name"] for call in pipeline.investigator.calls] == ["author", "author"]
    assert result.extras["slot_retries"] == 1


def test_slot_dag_invalid_route_uses_fallback_decomposer_once() -> None:
    route = {
        "action": "recurse",
        "confidence": 0.8,
        "draft_answer": "",
        "answer_type": "city",
        "target_slot": "birthplace",
        "execution_mode": "typed_plan_exec",
        "compositionality_score": 0.9,
        "expected_hop_count": 2,
        "required_hops": [
            {"slot_name": "answer", "hint": "bad placeholder", "expected_info_type": "other", "dependency_group": 0, "sub_question": "Where was the author of The Book Thief born?", "retrieval_query": "Where was the author of The Book Thief born?", "goal": "bad"},
        ],
    }
    fallback_hops = [
        {"slot_name": "author", "hint": "writer of The Book Thief", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who wrote The Book Thief?", "retrieval_query": "The Book Thief author", "goal": "Find the author."},
        {"slot_name": "birthplace", "hint": "birthplace of [author]", "expected_info_type": "city", "dependency_group": 1, "sub_question": "Where was [author] born?", "retrieval_query": "[author] birthplace", "goal": "Find the birthplace."},
    ]
    pipeline = _make_slot_dag_pipeline(
        route,
        {"author": [_slot_cap("Markus Zusak")], "birthplace": [_slot_cap("Sydney")]},
        fallback_hops=fallback_hops,
    )

    result = asyncio.run(pipeline._run_slot_dag_sufficiency("Where was the author of The Book Thief born?", "q5"))

    assert pipeline.orchestrator.fallback_calls == 1
    assert result.extras["decomposition_source"] == "fallback_decompose"


def test_active_code_does_not_import_or_call_opera() -> None:
    root = Path(__file__).resolve().parent.parent
    active_files = [
        root / "src" / "adaptive_sage" / "pipeline.py",
        root / "src" / "adaptive_sage" / "orchestrator.py",
        root / "configs" / "m1_7.slot_dag_sufficiency.yaml",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8").lower() for path in active_files)
    assert "opera" not in joined



def _dod_config(extra_adaptive=None):
    adaptive = {
        "dod_controller": True,
        "dod_gate_threshold": 0.85,
        "dod_hop_threshold": 0.80,
        "slot_dag_max_steps": 4,
        "slot_dag_max_parallel": 2,
        "slot_dag_max_retries_per_slot": 1,
        "slot_dag_search_top_k": 4,
        "slot_dag_max_read": 4,
        "bootstrap_max_read": 4,
    }
    if extra_adaptive:
        adaptive.update(extra_adaptive)
    return Config(
        {
            "variant": "m1_8_dod_slot_dag",
            "orchestrator": {"max_steps": 4, "max_verify_calls": 0},
            "investigator": {"evidence_capsule_limit": 2, "search_top_k": 4},
            "fact_memory": {"capacity": 4, "strategy": "salience"},
            "adaptive": adaptive,
            "llm": {"model": "Qwen/Qwen3-8B", "base_url": "http://localhost:8001/v1"},
            "data": {"chunks_file": "x", "index_dir": "x", "embedding_model": "x"},
        }
    )


def _usage_mock_orchestrator():
    mock = MagicMock()
    mock.reset_usage_totals = MagicMock()
    mock.get_usage_totals = MagicMock(return_value={"prompt_tokens": 0, "completion_tokens": 0})
    mock.route_with_usage = AsyncMock(
        return_value=(
            {
                "action": "single_probe",
                "confidence": 0.9,
                "answer_type": "location",
                "target_slot": "final_answer",
                "compositionality_score": 0.0,
                "required_hops": [
                    {
                        "slot_name": "final_answer",
                        "hint": "final answer",
                        "expected_info_type": "location",
                        "dependency_group": 0,
                    }
                ],
            },
            5,
        )
    )
    mock.decompose_slot_dag_with_usage = AsyncMock(return_value=([], 0))
    mock.check_hop_sufficiency_with_usage = AsyncMock(
        return_value=(
            {
                "answerable": False,
                "answer": "",
                "cited_fact_ids": [],
                "justification_confidence": 0.0,
                "missing_slot": "",
            },
            3,
        )
    )
    return mock


def _usage_mock_investigator(*capsules):
    mock = MagicMock()
    mock.evidence_capsule_limit = 2
    mock.reset_usage_totals = MagicMock()
    mock.get_usage_totals = MagicMock(return_value={"prompt_tokens": 0, "completion_tokens": 0})
    mock.investigate_with_usage = AsyncMock(side_effect=[(capsule, 10) for capsule in capsules])
    return mock


def _capsule(answer, text, slot_filled=True, support_ids=None, slot_name=""):
    return EvidenceCapsule(
        answer=answer,
        fact=Fact(
            text=text,
            confidence=0.9 if text else 0.0,
            slot_filled=slot_filled,
            answer_span=answer,
            support_ids=support_ids or (["1"] if text else []),
            slot_name=slot_name,
            source_step=0,
        ),
    )


def test_dod_easy_route_calls_one_investigator_and_never_enters_dag() -> None:
    capsule = _capsule("Sydney", "Markus Zusak was born in Sydney.")
    mock_orchestrator = _usage_mock_orchestrator()
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        return_value=(
            {
                "answer": "Sydney",
                "cited_fact_ids": [1],
                "justification_confidence": 0.91,
                "justification": "Grounded.",
                "missing_slot": "",
            },
            7,
        )
    )
    mock_investigator = _usage_mock_investigator(capsule)

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        result = asyncio.run(AdaptiveRecursivePipeline(_dod_config()).run("Where was Markus Zusak born?", "q1"))

    assert result.route_decision == "answer_after_direct_sas"
    assert result.answer == "Sydney"
    assert result.num_subagent_calls == 1
    assert result.extras["collapsed_to_single_agent"] is True
    mock_orchestrator.route_with_usage.assert_awaited_once()
    mock_investigator.investigate_with_usage.assert_awaited_once()


def test_dod_sequential_dag_runs_group_zero_before_group_one_with_prior_facts() -> None:
    direct = _capsule("", "")
    author = _capsule("Markus Zusak", "Markus Zusak wrote The Book Thief.", slot_name="author")
    birthplace = _capsule("Sydney", "Markus Zusak was born in Sydney.", slot_name="birthplace")
    mock_orchestrator = _usage_mock_orchestrator()
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        side_effect=[
            ({"answer": "", "cited_fact_ids": [], "justification_confidence": 0.0, "justification": "", "missing_slot": ""}, 4),
            ({"answer": "Sydney", "cited_fact_ids": [2], "justification_confidence": 0.9, "justification": "", "missing_slot": ""}, 4),
        ]
    )
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "action": "recurse",
                "confidence": 0.9,
                "answer_type": "location",
                "target_slot": "birthplace",
                "required_hops": [
                    {"slot_name": "author", "hint": "writer", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who wrote The Book Thief?", "retrieval_query": "The Book Thief author", "goal": "Find author."},
                    {"slot_name": "birthplace", "hint": "birthplace", "expected_info_type": "location", "dependency_group": 1, "sub_question": "Where was [author] born?", "retrieval_query": "[author] birthplace", "goal": "Find birthplace."},
                ],
            },
            5,
        )
    )
    mock_investigator = _usage_mock_investigator(direct, author, birthplace)

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        result = asyncio.run(AdaptiveRecursivePipeline(_dod_config({"dod_disable_early_exit": True})).run("Where was the author of The Book Thief born?", "q2"))

    calls = mock_investigator.investigate_with_usage.await_args_list
    assert calls[1].kwargs["slot_name"] == "author"
    assert calls[2].kwargs["slot_name"] == "birthplace"
    assert calls[2].kwargs["prior_facts"][0].answer_span == "Markus Zusak"
    assert result.answer == "Sydney"


def test_dod_parallel_dag_uses_one_shared_memory_snapshot_and_one_gate_check() -> None:
    direct = _capsule("", "")
    director_a = _capsule("A", "Film A was directed by A.", slot_name="director_a")
    director_b = _capsule("B", "Film B was directed by B.", slot_name="director_b")
    mock_orchestrator = _usage_mock_orchestrator()
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        side_effect=[
            ({"answer": "", "cited_fact_ids": [], "justification_confidence": 0.0, "justification": "", "missing_slot": ""}, 4),
            ({"answer": "A and B", "cited_fact_ids": [1, 2], "justification_confidence": 0.9, "justification": "", "missing_slot": ""}, 4),
        ]
    )
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "action": "recurse",
                "confidence": 0.9,
                "answer_type": "entity",
                "target_slot": "directors",
                "required_hops": [
                    {"slot_name": "director_a", "hint": "director A", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who directed Film A?", "retrieval_query": "Film A director", "goal": "Find director A."},
                    {"slot_name": "director_b", "hint": "director B", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who directed Film B?", "retrieval_query": "Film B director", "goal": "Find director B."},
                ],
            },
            5,
        )
    )
    mock_investigator = _usage_mock_investigator(direct, director_a, director_b)

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        result = asyncio.run(AdaptiveRecursivePipeline(_dod_config()).run("Who directed Film A and Film B?", "q3"))

    calls = mock_investigator.investigate_with_usage.await_args_list
    assert calls[1].kwargs["prior_facts"] == calls[2].kwargs["prior_facts"]
    assert mock_orchestrator.check_hop_sufficiency_with_usage.await_count == 1
    assert result.extras["parallel_batches"] == 1


def test_dod_failed_slot_retry_only_reruns_that_slot_once() -> None:
    direct = _capsule("", "")
    failed = _capsule("", "No answer found.", slot_filled=False, support_ids=["1"], slot_name="birthplace")
    retry = _capsule("Sydney", "Markus Zusak was born in Sydney.", slot_name="birthplace")
    mock_orchestrator = _usage_mock_orchestrator()
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        side_effect=[
            ({"answer": "", "cited_fact_ids": [], "justification_confidence": 0.0, "justification": "", "missing_slot": ""}, 4),
            ({"answer": "Sydney", "cited_fact_ids": [1], "justification_confidence": 0.9, "justification": "", "missing_slot": ""}, 4),
        ]
    )
    mock_orchestrator.route_with_usage = AsyncMock(
        return_value=(
            {
                "action": "recurse",
                "confidence": 0.9,
                "answer_type": "location",
                "target_slot": "birthplace",
                "required_hops": [
                    {"slot_name": "birthplace", "hint": "birthplace", "expected_info_type": "location", "dependency_group": 0, "sub_question": "Where was Markus Zusak born?", "retrieval_query": "Markus Zusak birthplace", "goal": "Find birthplace."},
                ],
            },
            5,
        )
    )
    mock_investigator = _usage_mock_investigator(direct, failed, retry)

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        result = asyncio.run(AdaptiveRecursivePipeline(_dod_config()).run("Where was Markus Zusak born?", "q4"))

    assert mock_investigator.investigate_with_usage.await_count == 3
    assert result.extras["slot_retries"] == 1


def test_dod_invalid_dag_fallback_planner_called_once() -> None:
    direct = _capsule("", "")
    author = _capsule("Markus Zusak", "Markus Zusak wrote The Book Thief.", slot_name="author")
    mock_orchestrator = _usage_mock_orchestrator()
    mock_orchestrator.generate_answer_object_with_usage = AsyncMock(
        side_effect=[
            ({"answer": "", "cited_fact_ids": [], "justification_confidence": 0.0, "justification": "", "missing_slot": ""}, 4),
            ({"answer": "Markus Zusak", "cited_fact_ids": [1], "justification_confidence": 0.9, "justification": "", "missing_slot": ""}, 4),
        ]
    )
    mock_orchestrator.route_with_usage = AsyncMock(return_value=({"action": "recurse", "required_hops": []}, 5))
    mock_orchestrator.decompose_slot_dag_with_usage = AsyncMock(
        return_value=(
            [{"slot_name": "author", "hint": "writer", "expected_info_type": "person", "dependency_group": 0, "sub_question": "Who wrote The Book Thief?", "retrieval_query": "The Book Thief author", "goal": "Find author."}],
            6,
        )
    )
    mock_investigator = _usage_mock_investigator(direct, author)

    with patch("adaptive_sage.pipeline.Orchestrator", return_value=mock_orchestrator), patch(
        "adaptive_sage.pipeline.Investigator", return_value=mock_investigator
    ):
        result = asyncio.run(AdaptiveRecursivePipeline(_dod_config({"dod_disable_early_exit": True})).run("Who wrote The Book Thief?", "q5"))

    mock_orchestrator.decompose_slot_dag_with_usage.assert_awaited_once()
    assert result.extras["decomposition_source"] == "fallback_decompose"


def test_dod_gate_blocks_unsupported_high_confidence_answer() -> None:
    facts = [Fact(text="Markus Zusak wrote The Book Thief.", confidence=0.9, answer_span="Markus Zusak", support_ids=["1"])]
    answer_obj = {"answer": "Sydney", "cited_fact_ids": [1], "justification_confidence": 0.95}

    assert AdaptiveRecursivePipeline._is_gate_safe(answer_obj, facts, expected_info_type="location", threshold=0.85) is False

"""Tests for adaptive_sage.orchestrator helper behavior."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_sage.orchestrator import Orchestrator
from adaptive_sage.types import Fact, StepTrace
from arag.core.config import Config
from arag.core.llm import LLMClient


def test_detects_exact_question_echo() -> None:
    question = "Where was the author of The Book Thief born?"
    assert Orchestrator._looks_like_question_echo(question, question) is True


def test_detects_nested_question_echo() -> None:
    original = "Where was the director of the movie written by the author of The Book Thief born?"
    candidate = "Where was the director of the movie written by the author of The Book Thief born"
    assert Orchestrator._looks_like_question_echo(candidate, original) is True


def test_allows_narrow_bridge_question() -> None:
    original = "Where was the director of the movie written by the author of The Book Thief born?"
    candidate = "Who wrote The Book Thief?"
    assert Orchestrator._looks_like_question_echo(candidate, original) is False


def test_normalise_required_hops_keeps_dependency_groups() -> None:
    hops = Orchestrator._normalise_required_hops(
        [
            {
                "slot_name": "author",
                "hint": "writer of the book",
                "expected_info_type": "person",
                "dependency_group": 0,
            },
            {
                "slot_name": "director",
                "hint": "director of the film",
                "expected_info_type": "person",
                "dependency_group": 1,
            },
        ]
    )
    assert hops == [
        {
            "slot_name": "author",
            "hint": "writer of the book",
            "expected_info_type": "person",
            "dependency_group": 0,
            "sub_question": "",
            "retrieval_query": "",
            "goal": "",
        },
        {
            "slot_name": "director",
            "hint": "director of the film",
            "expected_info_type": "person",
            "dependency_group": 1,
            "sub_question": "",
            "retrieval_query": "",
            "goal": "",
        },
    ]


def test_normalise_required_hops_refines_generic_expected_info_type() -> None:
    hops = Orchestrator._normalise_required_hops(
        [
            {
                "slot_name": "county",
                "hint": "resolve the county",
                "expected_info_type": "location",
                "dependency_group": 0,
            }
        ]
    )

    assert hops[0]["expected_info_type"] == "county"


def test_format_pending_slots_includes_expected_info_type() -> None:
    rendered = Orchestrator._format_pending_slots(
        [
            {
                "slot_name": "author",
                "hint": "writer of the book",
                "expected_info_type": "person",
                "resolved": False,
                "dependency_group": 0,
            }
        ]
    )

    assert "[type person]" in rendered


def test_format_hop_chain_uses_grounded_answers() -> None:
    trace = [
        StepTrace(step=1, action="spawn", sub_question="Who wrote The Book Thief?"),
        StepTrace(step=2, action="spawn", sub_question="Where was Markus Zusak born?"),
    ]
    facts = [
        Fact(
            text="Markus Zusak wrote The Book Thief.",
            confidence=0.9,
            answer_span="Markus Zusak",
            support_ids=["1"],
            source_step=1,
        ),
        Fact(
            text="Markus Zusak was born in Sydney.",
            confidence=0.88,
            answer_span="Sydney",
            support_ids=["2"],
            source_step=2,
        ),
    ]

    hop_chain = Orchestrator._format_hop_chain(trace, facts)

    assert "Hop 1: Who wrote The Book Thief? -> found Markus Zusak" in hop_chain
    assert "Hop 2: Where was Markus Zusak born? -> found Sydney" in hop_chain


def test_format_facts_includes_slot_name() -> None:
    facts = [
        Fact(
            text="Markus Zusak wrote The Book Thief.",
            confidence=0.9,
            answer_span="Markus Zusak",
            slot_name="author",
            support_ids=["1"],
            source_step=1,
        )
    ]

    formatted = Orchestrator._format_facts(facts)

    assert "slot: author" in formatted
    assert "answer span: Markus Zusak" in formatted


def test_route_preserves_retrieval_query() -> None:
    llm = MagicMock(spec=LLMClient)
    llm.async_chat = AsyncMock(
        return_value={
            "message": {
                "content": json.dumps(
                    {
                        "action": "recurse",
                        "confidence": 0.9,
                        "draft_answer": "",
                        "sub_question": "Who wrote The Book Thief?",
                        "retrieval_query": "The Book Thief author",
                        "goal": "Resolve the book author.",
                        "answer_type": "person",
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
                    }
                )
            },
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.0,
            "raw_response": {},
        }
    )
    orchestrator = Orchestrator(Config({}), llm)

    route, _ = asyncio.run(
        orchestrator.route_with_usage(
            "Where was the author of The Book Thief born?",
            "location",
        )
    )

    assert route["retrieval_query"] == "The Book Thief author"


def test_route_coerces_direct_answer_to_single_probe() -> None:
    llm = MagicMock(spec=LLMClient)
    llm.async_chat = AsyncMock(
        return_value={
            "message": {
                "content": json.dumps(
                    {
                        "action": "direct_answer",
                        "confidence": 0.95,
                        "draft_answer": "Sydney",
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
                    }
                )
            },
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.0,
            "raw_response": {},
        }
    )
    orchestrator = Orchestrator(Config({}), llm)

    route, _ = asyncio.run(
        orchestrator.route_with_usage(
            "Where was Markus Zusak born?",
            "location",
        )
    )

    assert route["action"] == "single_probe"


def test_duplicate_subquestion_only_blocks_grounded_repeats() -> None:
    llm = MagicMock(spec=LLMClient)
    llm.async_chat = AsyncMock(
        return_value={
            "message": {
                "content": json.dumps(
                    {
                        "action": "spawn",
                        "sub_question": "Who wrote The Book Thief?",
                        "retrieval_query": "The Book Thief author",
                        "goal": "Resolve the author.",
                        "slot_name": "author",
                    }
                )
            },
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.0,
            "raw_response": {},
        }
    )
    orchestrator = Orchestrator(Config({}), llm)

    allowed, _ = asyncio.run(
        orchestrator.decide_with_usage(
            "Where was the author of The Book Thief born?",
            [],
            [
                StepTrace(
                    step=1,
                    action="spawn",
                    sub_question="Who wrote The Book Thief?",
                    fact_added=False,
                    slot_name="author",
                )
            ],
            2,
            pending_slots=[{"slot_name": "author", "hint": "writer"}],
        )
    )
    assert allowed["action"] == "spawn"

    blocked, _ = asyncio.run(
        orchestrator.decide_with_usage(
            "Where was the author of The Book Thief born?",
            [],
            [
                StepTrace(
                    step=1,
                    action="spawn",
                    sub_question="Who wrote The Book Thief?",
                    fact_added=True,
                    slot_name="author",
                )
            ],
            2,
            pending_slots=[{"slot_name": "author", "hint": "writer"}],
        )
    )
    assert blocked["action"] == "answer"


def test_assess_typed_probe_state_parses_scores() -> None:
    llm = MagicMock(spec=LLMClient)
    llm.async_chat = AsyncMock(
        return_value={
            "message": {
                "content": json.dumps(
                    {
                        "slot_sufficient": 0.92,
                        "answer_sufficient": 0.18,
                        "slot_reason": "The author slot is grounded.",
                        "answer_reason": "The birthplace slot is still unresolved.",
                    }
                )
            },
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.0,
            "raw_response": {},
        }
    )
    orchestrator = Orchestrator(Config({}), llm)

    result, _ = asyncio.run(
        orchestrator.assess_typed_probe_state_with_usage(
            question="Where was the author of The Book Thief born?",
            facts=[
                Fact(
                    text="Markus Zusak wrote The Book Thief.",
                    confidence=0.91,
                    slot_name="author",
                    answer_span="Markus Zusak",
                    support_ids=["1"],
                    source_step=0,
                )
            ],
            proposed_answer="Markus Zusak",
            probe_question="Who wrote The Book Thief?",
            probe_strategy="bridge_first_typed",
            probe_slot_name="author",
            probe_slot_hint="writer of The Book Thief",
            probe_expected_info_type="person",
            probe_slot_value="Markus Zusak",
            target_profile="location",
            pending_slots=[
                {
                    "slot_name": "author",
                    "hint": "writer of The Book Thief",
                    "expected_info_type": "person",
                    "resolved": True,
                    "dependency_group": 0,
                },
                {
                    "slot_name": "birthplace",
                    "hint": "birthplace of the author",
                    "expected_info_type": "location",
                    "resolved": False,
                    "dependency_group": 1,
                },
            ],
            resolved_slots=["author"],
            trace=[],
        )
    )

    assert result["slot_sufficient"] == 0.92
    assert result["answer_sufficient"] == 0.18
    assert "grounded" in result["slot_reason"]

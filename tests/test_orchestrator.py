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
            {"slot_name": "author", "hint": "writer of the book", "dependency_group": 0},
            {"slot_name": "director", "hint": "director of the film", "dependency_group": 1},
        ]
    )
    assert hops == [
        {"slot_name": "author", "hint": "writer of the book", "dependency_group": 0},
        {"slot_name": "director", "hint": "director of the film", "dependency_group": 1},
    ]


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


def test_deterministic_innermost_sub_question_narrows_echo() -> None:
    question = "Where was the director of the movie written by the author of The Book Thief born?"
    narrowed = Orchestrator._deterministic_innermost_sub_question(
        question=question,
        facts=[
            Fact(
                text="Markus Zusak wrote The Book Thief.",
                confidence=0.8,
                answer_span="Markus Zusak",
                support_ids=["1"],
                source_step=1,
            )
        ],
        target_profile="location",
        pending_slots=[{"slot_name": "director", "hint": "director"}],
    )

    assert narrowed.endswith("?")
    assert Orchestrator._looks_like_question_echo(narrowed, question) is False


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
                                "dependency_group": 0,
                            },
                            {
                                "slot_name": "birthplace",
                                "hint": "birthplace of the author",
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

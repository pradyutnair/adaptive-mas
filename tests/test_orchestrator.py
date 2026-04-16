"""Tests for adaptive_sage.orchestrator helper behavior."""

import sys
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_sage.orchestrator import Orchestrator
from adaptive_sage.types import Fact, StepTrace


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

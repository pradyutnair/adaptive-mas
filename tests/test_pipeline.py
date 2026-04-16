"""Tests for adaptive_sage.pipeline helper behavior."""

import sys
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_sage.pipeline import AdaptiveRecursivePipeline
from adaptive_sage.types import Fact


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


def test_answer_fallback_uses_route_draft_when_memory_empty() -> None:
    answer, source, cited_fact_ids, confidence = (
        AdaptiveRecursivePipeline._apply_answer_fallback(
            "",
            [],
            route_draft_answer="Sydney",
        )
    )

    assert answer == "Sydney"
    assert source == "route_draft"
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
    from adaptive_sage.types import EvidenceCapsule

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

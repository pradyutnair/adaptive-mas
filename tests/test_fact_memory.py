"""Tests for adaptive_sage.fact_memory.FactMemory — bounded FIFO fact store."""

import sys
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_sage.types import Fact
from adaptive_sage.fact_memory import FactMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fact(text: str, step: int = 0, confidence: float = 0.9) -> Fact:
    """Shorthand to create a Fact with sensible defaults."""
    return Fact(text=text, confidence=confidence, support_ids=[], source_step=step)


# ---------------------------------------------------------------------------
# test_add_within_capacity
# ---------------------------------------------------------------------------

class TestAddWithinCapacity:
    """Adding fewer facts than capacity keeps all of them."""

    def test_add_within_capacity(self):
        fm = FactMemory(capacity=4)
        f1 = _fact("fact1", step=1)
        f2 = _fact("fact2", step=2)
        f3 = _fact("fact3", step=3)

        evicted1 = fm.add(f1)
        evicted2 = fm.add(f2)
        evicted3 = fm.add(f3)

        assert evicted1 is None
        assert evicted2 is None
        assert evicted3 is None
        assert len(fm) == 3

        all_facts = fm.get_all()
        assert all_facts == [f1, f2, f3]

    def test_add_exactly_at_capacity(self):
        fm = FactMemory(capacity=3)
        f1 = _fact("fact1", step=1)
        f2 = _fact("fact2", step=2)
        f3 = _fact("fact3", step=3)

        fm.add(f1)
        fm.add(f2)
        evicted = fm.add(f3)

        # At capacity but not over — no eviction yet
        assert evicted is None
        assert len(fm) == 3
        assert fm.get_all() == [f1, f2, f3]


# ---------------------------------------------------------------------------
# test_fifo_eviction
# ---------------------------------------------------------------------------

class TestFIFOEviction:
    """Oldest facts are evicted first when capacity is exceeded."""

    def test_fifo_eviction(self):
        """Add 5 facts to cap=3, first 2 evicted, last 3 remain."""
        fm = FactMemory(capacity=3)
        facts = [_fact(f"fact{i}", step=i) for i in range(5)]

        evicted0 = fm.add(facts[0])  # len 1
        evicted1 = fm.add(facts[1])  # len 2
        evicted2 = fm.add(facts[2])  # len 3 (at capacity)
        evicted3 = fm.add(facts[3])  # evicts facts[0]
        evicted4 = fm.add(facts[4])  # evicts facts[1]

        assert evicted0 is None
        assert evicted1 is None
        assert evicted2 is None
        assert evicted3 == facts[0]
        assert evicted4 == facts[1]

        assert len(fm) == 3
        remaining = fm.get_all()
        assert remaining == [facts[2], facts[3], facts[4]]

    def test_fifo_eviction_returns_evicted_fact(self):
        fm = FactMemory(capacity=2)
        f1 = _fact("oldest", step=1)
        f2 = _fact("middle", step=2)
        f3 = _fact("newest", step=3)

        fm.add(f1)
        fm.add(f2)
        evicted = fm.add(f3)

        assert evicted == f1
        assert len(fm) == 2
        assert fm.get_all() == [f2, f3]

    def test_capacity_never_exceeded(self):
        fm = FactMemory(capacity=3)
        for i in range(20):
            fm.add(_fact(f"fact{i}", step=i))
            assert len(fm) <= 3

    def test_salience_strategy_evicts_weaker_fact(self):
        fm = FactMemory.with_strategy(capacity=2, strategy="salience")
        weak = _fact("weak fact", step=1, confidence=0.3)
        strong = Fact(
            text="strong fact",
            confidence=0.95,
            support_ids=["c1", "c2"],
            source_step=2,
        )
        newer = Fact(
            text="newer fact",
            confidence=0.8,
            support_ids=["c3"],
            source_step=3,
        )

        fm.add(weak)
        fm.add(strong)
        evicted = fm.add(newer)

        assert evicted == weak
        assert fm.get_all() == [strong, newer]


# ---------------------------------------------------------------------------
# test_deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Adding a fact with the same text as an existing one is a no-op."""

    def test_deduplication(self):
        fm = FactMemory(capacity=4)
        f1 = _fact("Paris is the capital of France", step=1)
        f2 = _fact("Paris is the capital of France", step=2)

        fm.add(f1)
        evicted = fm.add(f2)

        # Duplicate should be skipped
        assert evicted is None
        assert len(fm) == 1
        # The original fact should remain (not replaced)
        assert fm.get_all() == [f1]

    def test_deduplication_different_confidence(self):
        """Dedup is based on text only — different confidence is still a dup."""
        fm = FactMemory(capacity=4)
        f1 = Fact(text="same text", confidence=0.9, support_ids=[], source_step=1)
        f2 = Fact(text="same text", confidence=0.5, support_ids=[], source_step=2)

        fm.add(f1)
        fm.add(f2)

        assert len(fm) == 1
        assert fm.get_all() == [f1]

    def test_deduplication_does_not_evict(self):
        """A skipped duplicate should not cause eviction."""
        fm = FactMemory(capacity=2)
        f1 = _fact("fact1", step=1)
        f2 = _fact("fact2", step=2)
        f3 = _fact("fact1", step=3)  # duplicate of f1

        fm.add(f1)
        fm.add(f2)
        evicted = fm.add(f3)

        assert evicted is None
        assert len(fm) == 2
        assert fm.get_all() == [f1, f2]

    def test_similar_but_not_identical_is_not_dup(self):
        """Only exact string match counts as duplicate."""
        fm = FactMemory(capacity=4)
        f1 = _fact("Paris is the capital", step=1)
        f2 = _fact("Paris is the capital of France", step=2)

        fm.add(f1)
        fm.add(f2)

        assert len(fm) == 2


# ---------------------------------------------------------------------------
# test_get_formatted
# ---------------------------------------------------------------------------

class TestGetFormatted:
    """get_formatted() returns a numbered string suitable for LLM prompts."""

    def test_get_formatted(self):
        fm = FactMemory(capacity=4)
        fm.add(_fact("Paris is the capital of France", step=1))
        fm.add(_fact("The Eiffel Tower was built in 1889", step=2))

        formatted = fm.get_formatted()
        assert "1." in formatted
        assert "2." in formatted
        assert "Paris is the capital of France" in formatted
        assert "The Eiffel Tower was built in 1889" in formatted

    def test_get_formatted_empty(self):
        fm = FactMemory(capacity=4)
        formatted = fm.get_formatted()
        assert formatted == ""

    def test_get_formatted_single(self):
        fm = FactMemory(capacity=4)
        fm.add(_fact("Only fact", step=1))
        formatted = fm.get_formatted()
        assert "1." in formatted
        assert "Only fact" in formatted

    def test_get_formatted_preserves_order(self):
        fm = FactMemory(capacity=4)
        fm.add(_fact("First fact", step=1))
        fm.add(_fact("Second fact", step=2))
        fm.add(_fact("Third fact", step=3))

        formatted = fm.get_formatted()
        lines = [line.strip() for line in formatted.strip().split("\n") if line.strip()]
        assert lines[0].startswith("1.")
        assert "First fact" in lines[0]
        assert lines[1].startswith("2.")
        assert "Second fact" in lines[1]
        assert lines[2].startswith("3.")
        assert "Third fact" in lines[2]


# ---------------------------------------------------------------------------
# test_clear
# ---------------------------------------------------------------------------

class TestClear:
    """clear() resets the memory to empty."""

    def test_clear(self):
        fm = FactMemory(capacity=4)
        fm.add(_fact("fact1", step=1))
        fm.add(_fact("fact2", step=2))
        assert len(fm) == 2

        fm.clear()

        assert len(fm) == 0
        assert fm.get_all() == []
        assert fm.get_formatted() == ""

    def test_clear_then_add(self):
        """After clear, new facts can be added normally."""
        fm = FactMemory(capacity=2)
        fm.add(_fact("fact1", step=1))
        fm.add(_fact("fact2", step=2))
        fm.clear()

        fm.add(_fact("fact3", step=3))
        assert len(fm) == 1
        assert fm.get_all()[0].text == "fact3"


# ---------------------------------------------------------------------------
# test_len
# ---------------------------------------------------------------------------

class TestLen:
    """__len__ correctly tracks the current fact count."""

    def test_len(self):
        fm = FactMemory(capacity=4)
        assert len(fm) == 0

        fm.add(_fact("fact1", step=1))
        assert len(fm) == 1

        fm.add(_fact("fact2", step=2))
        assert len(fm) == 2

        fm.add(_fact("fact3", step=3))
        assert len(fm) == 3

    def test_len_after_eviction(self):
        fm = FactMemory(capacity=2)
        fm.add(_fact("fact1", step=1))
        fm.add(_fact("fact2", step=2))
        fm.add(_fact("fact3", step=3))  # evicts fact1
        assert len(fm) == 2

    def test_len_after_dedup(self):
        fm = FactMemory(capacity=4)
        fm.add(_fact("fact1", step=1))
        fm.add(_fact("fact1", step=2))  # dedup, no change
        assert len(fm) == 1

    def test_len_after_clear(self):
        fm = FactMemory(capacity=4)
        fm.add(_fact("fact1", step=1))
        fm.clear()
        assert len(fm) == 0

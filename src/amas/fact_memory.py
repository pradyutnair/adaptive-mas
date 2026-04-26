"""Bounded fact memory with configurable eviction for AMAS.

Stores distilled facts extracted by investigator subagents.  When the
memory reaches its configured capacity, an eviction policy chooses which
fact to drop.  Duplicate facts (identical text) are silently skipped.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Literal, Optional

from .types import Fact

logger = logging.getLogger(__name__)


class FactMemory:
    """A bounded, FIFO-evicting store of :class:`Fact` objects.

    Parameters
    ----------
    capacity:
        Maximum number of facts the memory can hold.  When full,
        adding a new fact evicts the oldest one (FIFO).

    Examples
    --------
    >>> from adaptive_sage.types import Fact
    >>> fm = FactMemory(capacity=3)
    >>> fm.add(Fact(text="Paris is the capital of France", confidence=0.95, support_ids=[], source_step=1))
    None
    >>> len(fm)
    1
    """

    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._strategy: Literal["fifo", "salience"] = "fifo"
        # We do NOT use deque(maxlen=capacity) because we need to inspect
        # the evicted element before it is removed.  Instead, we manage
        # eviction manually with an unbounded deque.
        self._deque: deque[Fact] = deque()

    @classmethod
    def with_strategy(
        cls,
        capacity: int = 4,
        strategy: Literal["fifo", "salience"] = "fifo",
    ) -> "FactMemory":
        """Construct a memory with the requested eviction strategy."""
        memory = cls(capacity=capacity)
        memory._strategy = strategy
        return memory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, fact: Fact) -> Optional[Fact]:
        """Add a fact to memory.

        If a fact with the same ``text`` already exists, the add is
        skipped and ``None`` is returned (no eviction occurs).

        If the memory is at capacity, the oldest fact is evicted
        (FIFO) and returned.

        Parameters
        ----------
        fact:
            The fact to add.

        Returns
        -------
        Optional[Fact]
            The evicted fact if one was removed due to capacity,
            otherwise ``None``.
        """
        # Deduplication: skip if text already present
        if any(f.text == fact.text for f in self._deque):
            logger.debug("Skipping duplicate fact: %s", fact.text)
            return None

        evicted: Optional[Fact] = None

        # Evict according to the configured strategy when at capacity
        if len(self._deque) >= self._capacity:
            if self._strategy == "salience":
                evicted = self._evict_lowest_salience()
            else:
                evicted = self._deque.popleft()
                logger.debug("Evicted fact (FIFO): %s", evicted.text)

        self._deque.append(fact)
        return evicted

    def get_all(self) -> list[Fact]:
        """Return all facts in insertion order (oldest first)."""
        return list(self._deque)

    def replace(self, slot_name: str, fact: Fact) -> Optional[Fact]:
        """Replace the first fact for *slot_name*, else add as new."""
        cleaned_slot = str(slot_name or "").strip()
        if not cleaned_slot:
            return self.add(fact)
        ranked = list(self._deque)
        for idx, existing in enumerate(ranked):
            if str(existing.slot_name or "").strip() == cleaned_slot:
                evicted = existing
                ranked[idx] = fact
                self._deque = deque(ranked)
                return evicted
        return self.add(fact)

    def get_formatted(self) -> str:
        """Return facts as a numbered string for LLM prompt injection.

        Each fact appears on its own line as ``N. <fact text>``.
        Returns an empty string if no facts are stored.
        """
        if not self._deque:
            return ""
        lines = []
        for i, fact in enumerate(self._deque, start=1):
            lines.append(f"{i}. {fact.text}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset the memory, removing all facts."""
        self._deque.clear()

    def __len__(self) -> int:
        """Current number of facts in memory."""
        return len(self._deque)

    # ------------------------------------------------------------------
    #_repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FactMemory(capacity={self._capacity}, strategy={self._strategy}, "
            f"facts={len(self._deque)})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_lowest_salience(self) -> Fact:
        """Evict the least useful fact under a simple salience heuristic."""
        ranked = list(self._deque)
        evict_idx = min(
            range(len(ranked)),
            key=lambda idx: self._salience_score(ranked[idx], idx),
        )
        evicted = ranked[evict_idx]
        del ranked[evict_idx]
        self._deque = deque(ranked)
        logger.debug("Evicted fact (salience): %s", evicted.text)
        return evicted

    def _salience_score(self, fact: Fact, idx: int) -> tuple[float, int, int, int]:
        """Higher tuple values mean the fact is more worth retaining."""
        recency = fact.source_step
        support_count = len(fact.support_ids)
        text_len = len(fact.text)
        # Confidence dominates, then support count, then recency, then specificity.
        return (fact.confidence, support_count, recency, text_len - idx)

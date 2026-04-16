"""Tests for adaptive_sage.investigator — focused subagent with bounded evidence capsules."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_sage.types import EvidenceCapsule, Fact
from arag.core.config import Config
from arag.core.llm import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(limit: int = 2, top_k: int = 5) -> Config:
    """Build a Config suitable for Investigator without requiring data files."""
    return Config({
        "investigator": {
            "evidence_capsule_limit": limit,
            "search_top_k": top_k,
        },
        "data": {
            "chunks_file": "data/musique/chunks.json",
            "index_dir": "data/musique/index_e5_base_v2",
            "embedding_model": "intfloat/e5-base-v2",
        },
    })


def _make_mock_llm(json_response: dict) -> MagicMock:
    """Create a mock LLMClient that returns the given JSON via async_chat."""
    llm = MagicMock(spec=LLMClient)
    content = json.dumps(json_response)
    llm.async_chat = AsyncMock(return_value={
        "message": {"content": content},
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.0,
        "raw_response": {},
    })
    return llm


def _make_mock_tools():
    """Create mock keyword_search, semantic_search, read_chunk tools.

    Returns a dict with keys 'keyword_search', 'semantic_search', 'read_chunk'.
    """
    kw = MagicMock()
    kw.name = "keyword_search"
    kw.execute.return_value = (
        "Chunk ID: 42, Matched keywords in chunk: ... Einstein ...\n"
        "Chunk ID: 7, Matched keywords in chunk: ... physics ...",
        {"retrieved_tokens": 20, "chunks_found": 2},
    )

    sem = MagicMock()
    sem.name = "semantic_search"
    sem.execute.return_value = (
        "Chunk ID: 42 (Similarity: 0.92)\nMatched: ... Einstein ...\n"
        "Chunk ID: 13 (Similarity: 0.85)\nMatched: ... relativity ...",
        {"retrieved_tokens": 30, "chunks_found": 2},
    )

    read = MagicMock()
    read.name = "read_chunk"
    read.chunks_dict = {
        "42": "Albert Einstein developed the theory of relativity.",
        "7": "Physics is the natural science of matter and energy.",
        "13": "The theory of relativity transformed theoretical physics.",
    }
    read.execute.return_value = (
        "[Chunk 42] Albert Einstein developed the theory of relativity.\n"
        "[Chunk 7] Physics is the natural science of matter and energy.\n"
        "[Chunk 13] The theory of relativity transformed theoretical physics.",
        {"retrieved_tokens": 60, "new_chunks_count": 3},
    )

    return {"keyword_search": kw, "semantic_search": sem, "read_chunk": read}


# Patch targets — must patch where names are *used*, not where defined.
# The investigator module does ``from arag.tools.X import Y``, so we patch
# the name in the ``adaptive_sage.investigator`` namespace.
_TOOL_PATCHES = [
    "adaptive_sage.investigator.KeywordSearchTool",
    "adaptive_sage.investigator.SemanticSearchTool",
    "adaptive_sage.investigator.ReadChunkTool",
]


# ---------------------------------------------------------------------------
# test_investigate_returns_capsule
# ---------------------------------------------------------------------------

class TestInvestigateReturnsCapsule:
    """investigate() returns an EvidenceCapsule with all fields populated."""

    def test_returns_capsule(self):
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config()
            llm = _make_mock_llm({
                "answer": "Albert Einstein",
                "fact": "Albert Einstein developed the theory of relativity",
                "confidence": 0.95,
                "support_ids": ["42", "13"],
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="Who developed the theory of relativity?",
                    goal="Find the scientist responsible for the theory of relativity",
                    prior_facts=[],
                )
            )

            assert isinstance(capsule, EvidenceCapsule)
            assert capsule.answer == "Albert Einstein"
            assert isinstance(capsule.fact, Fact)
            assert capsule.fact.text == "Albert Einstein developed the theory of relativity"
            assert capsule.fact.confidence == pytest.approx(0.9275)
            assert capsule.fact.confidence_self == pytest.approx(0.95)
            assert capsule.fact.confidence_retrieval == pytest.approx(0.885)
            assert capsule.fact.slot_filled is True
            assert isinstance(capsule.support_snippets, list)

    def test_capsule_has_support_ids(self):
        """The fact inside the capsule should carry support_ids."""
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config(limit=2)
            llm = _make_mock_llm({
                "answer": "Albert Einstein",
                "fact": "Einstein developed relativity",
                "confidence": 0.9,
                "support_ids": ["42", "13"],
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="Who developed relativity?",
                    goal="Identify the physicist",
                    prior_facts=[],
                )
            )

            assert capsule.fact.support_ids == ["42", "13"]


# ---------------------------------------------------------------------------
# test_capsule_respects_limit
# ---------------------------------------------------------------------------

class TestCapsuleRespectsLimit:
    """support_ids is truncated to evidence_capsule_limit."""

    def test_capsule_respects_limit(self):
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config(limit=2)
            llm = _make_mock_llm({
                "answer": "Answer",
                "fact": "Some fact",
                "confidence": 0.8,
                "support_ids": ["42", "7", "13", "99"],  # 4 IDs, but limit is 2
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="Test question?",
                    goal="Test goal",
                    prior_facts=[],
                )
            )

            assert len(capsule.fact.support_ids) <= 2

    def test_limit_of_one(self):
        """With limit=1, at most 1 support_id survives."""
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config(limit=1)
            llm = _make_mock_llm({
                "answer": "Answer",
                "fact": "Fact",
                "confidence": 0.7,
                "support_ids": ["42", "7"],
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="Q?",
                    goal="G",
                    prior_facts=[],
                )
            )

            assert len(capsule.fact.support_ids) <= 1

    def test_low_confidence_or_unsupported_answer_is_suppressed(self):
        """Weakly supported answers should not be turned into facts."""
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config(limit=2)
            llm = _make_mock_llm({
                "answer": "Best guess",
                "fact": "A guessed fact",
                "confidence": 0.3,
                "support_ids": [],
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="Test question?",
                    goal="Test goal",
                    prior_facts=[],
                )
            )

            assert capsule.answer == ""
            assert capsule.fact.text == ""
            assert capsule.fact.confidence == 0.0
            assert capsule.fact.support_ids == []


# ---------------------------------------------------------------------------
# test_both_search_types_used
# ---------------------------------------------------------------------------

class TestBothSearchTypesUsed:
    """Both keyword_search and semantic_search are called for each sub-question."""

    def test_both_search_types_used(self):
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config()
            llm = _make_mock_llm({
                "answer": "A",
                "fact": "F",
                "confidence": 0.5,
                "support_ids": ["42"],
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="What is X?",
                    goal="Find X",
                    prior_facts=[],
                )
            )

            # Verify both search tools were called
            mock_tools["keyword_search"].execute.assert_called_once()
            mock_tools["semantic_search"].execute.assert_called_once()


# ---------------------------------------------------------------------------
# test_prior_facts_in_prompt
# ---------------------------------------------------------------------------

class TestPriorFactsInPrompt:
    """Prior facts are included in the distillation prompt context."""

    def test_prior_facts_in_prompt(self):
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config()
            llm = _make_mock_llm({
                "answer": "A",
                "fact": "F",
                "confidence": 0.5,
                "support_ids": ["42"],
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            prior = [Fact(text="The sky is blue", confidence=0.9, support_ids=[], source_step=1)]

            asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="What color is the ocean?",
                    goal="Determine ocean color",
                    prior_facts=prior,
                )
            )

            # Inspect the messages passed to async_chat
            call_args = llm.async_chat.call_args
            messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])

            # Find the user message content
            user_msg = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break

            assert user_msg is not None
            assert "The sky is blue" in user_msg

    def test_no_prior_facts_still_works(self):
        """With no prior facts, investigate still works (prompts shows 'None')."""
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config()
            llm = _make_mock_llm({
                "answer": "A",
                "fact": "F",
                "confidence": 0.5,
                "support_ids": ["42"],
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="What is X?",
                    goal="Find X",
                    prior_facts=[],
                )
            )

            assert isinstance(capsule, EvidenceCapsule)


# ---------------------------------------------------------------------------
# test_malformed_json_fallback
# ---------------------------------------------------------------------------

class TestMalformedJsonFallback:
    """Malformed JSON triggers a retry or produces a fallback capsule."""

    def test_malformed_json_tries_retry(self):
        """When LLM returns bad JSON the first time, it retries."""
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config()
            llm = MagicMock(spec=LLMClient)

            # First call: bad JSON. Second call: good JSON.
            bad_response = {
                "message": {"content": "This is not JSON at all!"},
                "input_tokens": 50,
                "output_tokens": 10,
                "cost": 0.0,
                "raw_response": {},
            }
            good_response = {
                "message": {"content": json.dumps({
                    "answer": "Retried answer",
                    "fact": "Retried fact",
                    "confidence": 0.7,
                    "support_ids": ["42"],
                })},
                "input_tokens": 100,
                "output_tokens": 50,
                "cost": 0.0,
                "raw_response": {},
            }
            llm.async_chat = AsyncMock(side_effect=[bad_response, good_response])

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="What is X?",
                    goal="Find X",
                    prior_facts=[],
                )
            )

            # Should have retried and gotten the good response
            assert llm.async_chat.call_count == 2
            assert isinstance(capsule, EvidenceCapsule)
            assert capsule.answer == "Retried answer"

    def test_all_retries_fail_produces_fallback(self):
        """When all retries fail, a fallback capsule is returned."""
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config()
            llm = MagicMock(spec=LLMClient)
            # Always return bad JSON
            llm.async_chat = AsyncMock(return_value={
                "message": {"content": "Not JSON!"},
                "input_tokens": 50,
                "output_tokens": 10,
                "cost": 0.0,
                "raw_response": {},
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="What is X?",
                    goal="Find X",
                    prior_facts=[],
                )
            )

            # Should return a fallback capsule with empty/default values
            assert isinstance(capsule, EvidenceCapsule)
            assert capsule.answer == ""
            assert capsule.fact.confidence == 0.0


# ---------------------------------------------------------------------------
# test_thinking_tags_stripped
# ---------------------------------------------------------------------------

class TestThinkingTagsStripped:
    """Qwen3 thinking tags (<tool_call>...) are stripped from LLM output."""

    def test_thinking_tags_stripped(self):
        with patch(_TOOL_PATCHES[0]) as MockKW, \
             patch(_TOOL_PATCHES[1]) as MockSem, \
             patch(_TOOL_PATCHES[2]) as MockRead:

            mock_tools = _make_mock_tools()
            MockKW.return_value = mock_tools["keyword_search"]
            MockSem.return_value = mock_tools["semantic_search"]
            MockRead.return_value = mock_tools["read_chunk"]

            config = _make_config()
            llm = MagicMock(spec=LLMClient)

            content_with_thinking = (
                "<think>Let me analyze this step by step...</think>"
                + json.dumps({
                    "answer": "Einstein",
                    "fact": "Einstein developed relativity",
                    "confidence": 0.9,
                    "support_ids": ["42"],
                })
            )
            llm.async_chat = AsyncMock(return_value={
                "message": {"content": content_with_thinking},
                "input_tokens": 100,
                "output_tokens": 50,
                "cost": 0.0,
                "raw_response": {},
            })

            from adaptive_sage.investigator import Investigator
            inv = Investigator(config, llm)

            capsule = asyncio.get_event_loop().run_until_complete(
                inv.investigate(
                    sub_question="Who developed relativity?",
                    goal="Identify physicist",
                    prior_facts=[],
                )
            )

            assert isinstance(capsule, EvidenceCapsule)
            assert capsule.answer == "Einstein"

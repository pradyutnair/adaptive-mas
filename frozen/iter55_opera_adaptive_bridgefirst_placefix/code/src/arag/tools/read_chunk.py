"""Read chunk tool - retrieve full document content."""

import json
import re
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from multi_agent.types import CachedDocument

    HAS_CACHED_DOCUMENT = True
except Exception:
    HAS_CACHED_DOCUMENT = False


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


class ReadChunkTool(BaseTool):
    """Read full content of document chunks."""

    def __init__(
        self,
        chunks_file: str,
        evidence_cache: Any = None,
        chunk_max_chars: int = 0,
        query_snippet_max_chars: int = 0,
    ):
        self.chunks_file = chunks_file
        self.evidence_cache = evidence_cache
        self.chunks = self._load_chunks()
        self.chunks_dict = {c["id"]: c["text"] for c in self.chunks}
        self.chunk_max_chars = int(chunk_max_chars or 0)
        self.query_snippet_max_chars = int(query_snippet_max_chars or 0)

        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data and isinstance(data[0], dict):
            return data

        chunks = []
        for item in data:
            if isinstance(item, str):
                parts = item.split(":", 1)
                if len(parts) == 2:
                    chunks.append({"id": parts[0], "text": parts[1]})
        return chunks

    @property
    def name(self) -> str:
        return "read_chunk"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": """Read the complete content of document chunks by their IDs.

This tool returns the full text of the specified chunks, allowing you to examine the complete context and details that are not visible in search snippets.

IMPORTANT: Search results (keyword_search and semantic_search) only show abbreviated snippets marked with "..." - they are NOT sufficient for answering questions. You MUST use read_chunk to get the full content before formulating your answer.

STRATEGY:
- Always read promising chunks identified by your searches
- Make sure to read the most relevant chunks to gather complete information
- If information seems incomplete or truncated, read adjacent chunks (± 1)
- Reading full text is essential for accurate answers

Note: Previously read chunks will be marked as already seen to avoid redundant information.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chunk IDs to retrieve (e.g., ['0', '24', '172'])",
                        }
                    },
                    "required": ["chunk_ids"],
                },
            },
        }

    def _cache_write_chunk(self, context: "AgentContext", cid: str, content: str) -> bool:
        """Optional write-through to shared evidence cache (no-op if unavailable)."""
        cache_obj = getattr(context, "evidence_cache", None) or self.evidence_cache
        if cache_obj is None:
            return False

        source_agent = int(getattr(context, "source_agent", -1))

        # Preferred path: multi-agent cache object exposing put_sync(CachedDocument).
        if HAS_CACHED_DOCUMENT:
            put_sync = getattr(cache_obj, "put_sync", None)
            if callable(put_sync):
                doc = CachedDocument(
                    doc_id=str(cid),
                    text=content,
                    embedding=None,
                    source_agent=source_agent,
                    retrieval_score=0.5,
                )
                put_sync(doc)
                return True

        # Fallback path: callback hook on context for custom integrations.
        callback = getattr(context, "cache_put_document", None)
        if callable(callback):
            callback(str(cid), content, source_agent)
            return True

        return False

    def execute(
        self,
        context: "AgentContext",
        chunk_ids: List[str] = None,
        chunk_id: str = None,
        query_text: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        """Read chunks by ID(s)."""
        if chunk_ids is None:
            if chunk_id is not None:
                chunk_ids = [str(chunk_id)]
            else:
                return "Error: No chunk IDs provided", {"retrieved_tokens": 0}

        chunk_ids = [str(cid) for cid in chunk_ids]

        result_parts = []
        new_chunks_read = []
        already_read = []
        total_tokens = 0
        cache_writes = 0
        compression_mode = "full"

        for cid in chunk_ids:
            if context.is_chunk_read(cid):
                already_read.append(cid)
                result_parts.append(f"\n{'=' * 80}")
                result_parts.append(f"[Chunk {cid}]")
                result_parts.append("(This chunk has been read before)")
                result_parts.append(f"{'=' * 80}")
                continue

            if cid in self.chunks_dict:
                content = self.chunks_dict[cid]
                if self.query_snippet_max_chars and query_text.strip():
                    content = self._extract_query_snippet(
                        content,
                        query_text=query_text,
                        max_chars=self.query_snippet_max_chars,
                    )
                    compression_mode = "query_snippet"
                elif self.chunk_max_chars and len(content) > self.chunk_max_chars:
                    content = content[: self.chunk_max_chars]
                    compression_mode = "prefix_truncate"
                result_parts.append(f"\n{'=' * 80}")
                result_parts.append(f"[Chunk {cid}]")
                result_parts.append(f"{'-' * 80}")
                result_parts.append(content)
                result_parts.append(f"{'=' * 80}")

                chunk_tokens = len(self.tokenizer.encode(content))
                total_tokens += chunk_tokens

                context.mark_chunk_as_read(cid)
                new_chunks_read.append(cid)

                if self._cache_write_chunk(context, cid, content):
                    cache_writes += 1
            else:
                result_parts.append(f"\n[Chunk {cid}] - Not found")

        tool_result = "\n".join(result_parts)

        context.add_retrieval_log(
            tool_name="read_chunk",
            tokens=total_tokens,
            metadata={
                "chunk_ids_requested": chunk_ids,
                "new_chunks_read": new_chunks_read,
                "already_read": already_read,
                "cache_writes": cache_writes,
                "compression_mode": compression_mode,
            },
        )

        tool_log = {
            "retrieved_tokens": total_tokens,
            "new_chunks_count": len(new_chunks_read),
            "already_read_count": len(already_read),
            "cache_writes": cache_writes,
            "compression_mode": compression_mode,
        }

        return tool_result, tool_log

    def _extract_query_snippet(self, content: str, query_text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(content) <= max_chars:
            return content

        query_terms = self._query_terms(query_text)
        if not query_terms:
            return content[:max_chars]

        sentences = self._split_sentences(content)
        if not sentences:
            return content[:max_chars]

        scores = []
        for idx, sentence in enumerate(sentences):
            sent_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            overlap = len(query_terms & sent_terms)
            if overlap > 0:
                scores.append((overlap, idx))

        if not scores:
            return self._extract_match_window(content, query_terms, max_chars)

        _, best_idx = max(scores)
        chosen = [sentences[best_idx].strip()]
        total_len = len(chosen[0])
        left = best_idx - 1
        right = best_idx + 1
        while total_len < max_chars and (left >= 0 or right < len(sentences)):
            candidates = []
            if left >= 0:
                candidates.append((self._sentence_overlap(sentences[left], query_terms), left))
            if right < len(sentences):
                candidates.append((self._sentence_overlap(sentences[right], query_terms), right))
            if not candidates:
                break
            _, pick = max(candidates)
            sent = sentences[pick].strip()
            if pick < best_idx:
                chosen.insert(0, sent)
                left -= 1
            else:
                chosen.append(sent)
                right += 1
            total_len = len(" ".join(chosen))

        return " ".join(chosen)[:max_chars]

    def _extract_match_window(self, content: str, query_terms: set[str], max_chars: int) -> str:
        lower = content.lower()
        best_pos = -1
        for term in sorted(query_terms, key=len, reverse=True):
            pos = lower.find(term)
            if pos >= 0:
                best_pos = pos
                break
        if best_pos < 0:
            return content[:max_chars]
        start = max(0, best_pos - max_chars // 3)
        end = min(len(content), start + max_chars)
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        return snippet

    def _query_terms(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) >= 4 and token not in _STOPWORDS
        }

    def _split_sentences(self, text: str) -> List[str]:
        return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _sentence_overlap(self, sentence: str, query_terms: set[str]) -> int:
        sent_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        return len(query_terms & sent_terms)

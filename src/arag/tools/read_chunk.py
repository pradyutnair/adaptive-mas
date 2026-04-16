"""Read chunk tool - retrieve full document content."""

import json
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


class ReadChunkTool(BaseTool):
    """Read full content of document chunks."""

    def __init__(self, chunks_file: str, evidence_cache: Any = None):
        self.chunks_file = chunks_file
        self.evidence_cache = evidence_cache
        self.chunks = self._load_chunks()
        self.chunks_dict = {c["id"]: c["text"] for c in self.chunks}

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
            },
        )

        tool_log = {
            "retrieved_tokens": total_tokens,
            "new_chunks_count": len(new_chunks_read),
            "already_read_count": len(already_read),
            "cache_writes": cache_writes,
        }

        return tool_result, tool_log

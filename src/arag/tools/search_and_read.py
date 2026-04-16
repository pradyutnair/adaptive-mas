"""Search-and-Read tool: combines search + auto-read top-k chunks in one call.

Eliminates the reads=0 failure mode by ensuring every search automatically
reads the full text of the top matching chunks.
"""

import re
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class SearchAndReadTool(BaseTool):
    """Combined search + auto-read tool.

    Wraps existing keyword_search/semantic_search tools and automatically
    reads the top-k matching chunks via read_chunk.
    """

    def __init__(self, keyword_tool, semantic_tool, read_tool):
        self.keyword_tool = keyword_tool
        self.semantic_tool = semantic_tool
        self.read_tool = read_tool

    @property
    def name(self) -> str:
        return "search_and_read"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_and_read",
                "description": (
                    "Search for relevant document chunks AND automatically read "
                    "their full text in one step. Combines search + read into a "
                    "single operation.\n\n"
                    "Use 'keyword' method for entity names and specific terms "
                    "(e.g., 'Albert Einstein, birthplace').\n"
                    "Use 'semantic' method for conceptual queries where exact "
                    "wording is unknown.\n\n"
                    "Returns the full text of the top matching chunks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Search query. For keyword: use comma-separated "
                                "entity names/terms. For semantic: use a natural "
                                "language description."
                            ),
                        },
                        "method": {
                            "type": "string",
                            "enum": ["keyword", "semantic"],
                            "description": (
                                "Search method: 'keyword' for exact text matching, "
                                "'semantic' for embedding similarity."
                            ),
                            "default": "keyword",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of chunks to read (default: 3, max: 5)",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    @staticmethod
    def _extract_chunk_ids(search_result: str) -> List[str]:
        """Extract chunk IDs from search result text."""
        raw_ids = re.findall(r"Chunk ID:\s*(\S+)", search_result)
        # Strip trailing commas/punctuation from IDs
        return [cid.rstrip(",.;:") for cid in raw_ids]

    def execute(
        self,
        context: "AgentContext",
        query: str,
        method: str = "keyword",
        top_k: int = 3,
    ) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 5)
        search_top_k = top_k * 3  # Over-retrieve for better coverage

        # Step 1: Search
        if method == "keyword":
            keywords = [k.strip() for k in query.split(",") if k.strip()]
            if not keywords:
                keywords = [query]
            search_result, search_log = self.keyword_tool.execute(
                context, keywords=keywords, top_k=search_top_k,
            )
        elif method == "semantic":
            if self.semantic_tool is None:
                return "Error: semantic search not available", {"retrieved_tokens": 0}
            search_result, search_log = self.semantic_tool.execute(
                context, query=query, top_k=search_top_k,
            )
        else:
            return f"Error: unknown method '{method}'", {"retrieved_tokens": 0}

        # Step 2: Extract chunk IDs
        chunk_ids = self._extract_chunk_ids(search_result)
        if not chunk_ids:
            return f"No results found for: {query}", {
                "retrieved_tokens": 0,
                "chunks_found": 0,
                "method": method,
                "chunk_ids_read": [],
            }

        # Step 3: Read top-k chunks
        read_ids = chunk_ids[:top_k]
        read_result, read_log = self.read_tool.execute(context, chunk_ids=read_ids)

        total_tokens = (
            search_log.get("retrieved_tokens", 0)
            + read_log.get("retrieved_tokens", 0)
        )

        combined_log = {
            "retrieved_tokens": total_tokens,
            "method": method,
            "query": query,
            "search_chunks_found": search_log.get("chunks_found", 0),
            "chunks_read": len(read_ids),
            "chunk_ids_read": read_ids,
        }

        return read_result, combined_log

"""Semantic search tool - embedding-based similarity matching."""

import os
import pickle
import threading
import numpy as np
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SemanticSearchTool(BaseTool):
    """Semantic search using embedding similarity."""

    _embedding_lock = threading.Lock()

    def __init__(
        self,
        chunks_file: str,
        index_dir: str = "index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")

        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")

        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.model_name = model_name
        self.device = device

        self.embedding_model = SentenceTransformer(model_name, device=device)
        self._load_index()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Qwen3 embedding models are instruction-aware for queries.
        # Keep a runtime toggle for fast A/B testing.
        raw_use_prompt = os.getenv("ARAG_USE_QUERY_PROMPT", "1").strip().lower()
        self.use_query_prompt = raw_use_prompt not in {"0", "false", "no", "off"}
        self.query_prompt_name = os.getenv("ARAG_QUERY_PROMPT_NAME", "query")

    @property
    def name(self) -> str:
        return "semantic_search"

    def _load_index(self):
        index_file = os.path.join(self.index_dir, "sentence_index.pkl")

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index not found: {index_file}")

        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)

        self.sentences = index_data['sentences']
        self.embeddings = index_data['embeddings']
        self.sentence_to_chunk = index_data['sentence_to_chunk']
        self.chunks = index_data['chunks']

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": """Semantic search using embedding similarity. Matches your query against sentences in each chunk via vector similarity.

WHEN TO USE:
- When keyword search fails to find relevant information
- When exact wording in documents is unknown
- For conceptual/meaning-based matching

RETURNS: Abbreviated snippets with matched sentences. Use read_chunk to get full text for answering.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what information you're looking for"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of most relevant results to return (default: 5, max: 20)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, context: 'AgentContext', query: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)

        with self._embedding_lock:
            if self.use_query_prompt:
                try:
                    query_embedding = self.embedding_model.encode(
                        [query],
                        prompt_name=self.query_prompt_name,
                        normalize_embeddings=True,
                    )[0]
                except TypeError:
                    query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            else:
                query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]

        chunk_sentences = {}
        for idx in top_indices:
            sentence = self.sentences[idx]
            chunk_id = self.sentence_to_chunk[idx]
            similarity = float(similarities[idx])

            if chunk_id not in chunk_sentences:
                chunk_sentences[chunk_id] = []
            chunk_sentences[chunk_id].append({
                'sentence': sentence,
                'similarity': similarity,
                'position': idx
            })

        chunk_scores = []
        for chunk_id, sents in chunk_sentences.items():
            max_similarity = max(s['similarity'] for s in sents)
            chunk_scores.append((chunk_id, max_similarity, sents))

        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_k]

        if not top_chunks:
            return f"No results for: {query}", {"retrieved_tokens": 0, "chunks_found": 0}

        result_parts = []
        for chunk_id, max_sim, sents in top_chunks:
            chunk_text = self.chunks[chunk_id]['text']
            sents_sorted = sorted(sents, key=lambda x: chunk_text.find(x['sentence']))
            matched_text = "... " + " ... ".join([s['sentence'] for s in sents_sorted]) + " ..."
            result_parts.append(f"Chunk ID: {chunk_id} (Similarity: {max_sim:.3f})\nMatched: {matched_text}")

        tool_result = "\n\n".join(result_parts)

        all_matched = []
        for _, _, sents in top_chunks:
            all_matched.extend([s['sentence'] for s in sents])

        retrieved_tokens = len(self.tokenizer.encode("\n".join(all_matched))) if all_matched else 0

        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={
                "query": query,
                "chunks_found": len(top_chunks),
                "use_query_prompt": self.use_query_prompt,
                "query_prompt_name": self.query_prompt_name,
            },
        )

        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(top_chunks)}

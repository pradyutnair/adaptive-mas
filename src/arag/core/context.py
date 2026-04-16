"""Agent execution context for ARAG."""

from typing import Any, Dict, List, Set
from dataclasses import dataclass, field


@dataclass
class RetrievalLog:
    """Log entry for a retrieval operation."""
    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Context manager for agent execution state."""
    
    def __init__(self):
        # Token statistics
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: List[RetrievalLog] = []
        
        # State management
        self.read_chunk_ids: Set[str] = set()
        self.search_history: List[Dict[str, Any]] = []
    
    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Dict[str, Any] = None
    ):
        """Add a retrieval log entry."""
        log = RetrievalLog(
            tool_name=tool_name,
            tokens=tokens,
            metadata=metadata or {}
        )
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens
    
    def mark_chunk_as_read(self, chunk_id: str):
        """Mark chunk as read."""
        self.read_chunk_ids.add(str(chunk_id))
    
    def is_chunk_read(self, chunk_id: str) -> bool:
        """Check if chunk has been read."""
        return str(chunk_id) in self.read_chunk_ids
    
    # Aliases for backward compatibility
    def add_read_chunk(self, chunk_id: str, content: str = None):
        """Alias for mark_chunk_as_read."""
        self.mark_chunk_as_read(chunk_id)
    
    def has_read_chunk(self, chunk_id: str) -> bool:
        """Alias for is_chunk_read."""
        return self.is_chunk_read(chunk_id)
    
    def get_read_chunk(self, chunk_id: str):
        """Check if chunk was read (returns None, content not stored)."""
        return None if not self.is_chunk_read(chunk_id) else ""
    
    def reset(self):
        """Reset context for new query."""
        self.retrieval_logs = []
        self.read_chunk_ids = set()
        self.search_history = []
        self.total_retrieved_tokens = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        return {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "retrieval_logs": [
                {
                    "tool_name": log.tool_name,
                    "tokens": log.tokens,
                    "metadata": log.metadata
                }
                for log in self.retrieval_logs
            ],
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": list(self.read_chunk_ids)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary."""
        return self.get_summary()

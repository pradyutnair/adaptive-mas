"""Finish tool - agent submits its final answer."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class FinishTool(BaseTool):
    """Explicit answer submission with confidence and supporting evidence."""

    @property
    def name(self) -> str:
        return "finish"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "finish",
                "description": (
                    "Submit your final answer. Call this when you have sufficient "
                    "evidence. Do NOT call any other tools after this."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Your concise final answer.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence from 0.0 to 1.0.",
                        },
                        "supporting_chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Chunk IDs supporting your answer.",
                        },
                    },
                    "required": ["answer"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        answer: str = "",
        confidence: float = 1.0,
        supporting_chunk_ids: List[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        return answer, {
            "retrieved_tokens": 0,
            "confidence": confidence,
            "supporting_chunk_ids": supporting_chunk_ids or [],
            "is_finish": True,
        }
from __future__ import annotations

import re
import string
from typing import Any

from pydantic import BaseModel, Field

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = str(s or "").lower()
    s = _ARTICLES.sub("", s)
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    s = " ".join(s.split())
    return s.strip()


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str = ""
    score: float = 0.0


class HopFinding(BaseModel):
    answer: str = ""
    evidence_chunk_id: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    queries_used: list[str] = Field(default_factory=list)
    expected_answer_type: str = "auto"

    def to_json(self) -> str:
        return self.model_dump_json() if hasattr(self, "model_dump_json") else self.json()

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump() if hasattr(self, "model_dump") else self.dict()


class CitationCheck(BaseModel):
    cited_ids: list[str] = Field(default_factory=list)
    answer_grounded: bool = False
    reason: str = ""

    def to_json(self) -> str:
        return self.model_dump_json() if hasattr(self, "model_dump_json") else self.json()

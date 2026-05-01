from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperienceEntry:
    id: str
    text: str
    rationale: str = ""


@dataclass
class ExperienceLibrary:
    entries: list[ExperienceEntry] = field(default_factory=list)
    next_num: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceLibrary":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        entries = [ExperienceEntry(**x) for x in data.get("entries", [])]
        lib = cls(entries=entries, next_num=int(data.get("next_num", 1)))
        if entries:
            lib.next_num = max(lib.next_num, max(int(e.id.split("-")[-1]) for e in entries) + 1)
        return lib

    def save_json(self, path: str | Path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def save_text(self, path: str | Path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_text(), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {"next_num": self.next_num, "entries": [e.__dict__ for e in self.entries]}

    def to_text(self) -> str:
        return "\n".join(f"{e.id}: {e.text}" for e in self.entries)

    def add(self, text: str, rationale: str = "") -> str:
        text = text.strip()
        if not text:
            return ""
        eid = f"E-{self.next_num:03d}"
        self.next_num += 1
        self.entries.append(ExperienceEntry(id=eid, text=text, rationale=rationale.strip()))
        return eid

    def modify(self, eid: str, text: str, rationale: str = "") -> None:
        for e in self.entries:
            if e.id == eid:
                e.text = text.strip() or e.text
                e.rationale = rationale.strip() or e.rationale
                return

    def delete(self, eid: str) -> None:
        self.entries = [e for e in self.entries if e.id != eid]

    def merge_text(self, text: str) -> None:
        seen = {self._canon(e.text) for e in self.entries}
        for line in text.splitlines():
            clean = re.sub(r"^E-\d{3}:\s*", "", line).strip(" -\t")
            key = self._canon(clean)
            if key and key not in seen:
                self.add(clean)
                seen.add(key)

    def apply_ops(self, ops: list[dict[str, Any]]) -> None:
        for op in ops:
            kind = str(op.get("op", "KEEP")).upper()
            if kind == "ADD":
                self.add(str(op.get("text", "")), str(op.get("rationale", "")))
            elif kind == "MODIFY":
                self.modify(str(op.get("id", "")), str(op.get("text", "")), str(op.get("rationale", "")))
            elif kind == "DELETE":
                self.delete(str(op.get("id", "")))

    @staticmethod
    def _canon(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

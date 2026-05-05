"""WorkingMemory: shared append-only memory across Solvers.

Renamed from FindingsBus. This is the explicit collaboration substrate
referenced in the thesis (Q2: inter-agent communication).

- Solvers append EvidenceCapsules here.
- Downstream Solvers read parent findings via tag interpolation.
- The Synthesizer reads the entire memory.

Append-only. No mutation in place.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from .types import Finding, EvidenceCapsule


@dataclass
class WorkingMemory:
    findings_by_node: dict[int, Finding] = field(default_factory=dict)
    capsules_by_node: dict[int, EvidenceCapsule] = field(default_factory=dict)

    def append(self, node_id: int, finding: Finding) -> None:
        if node_id in self.findings_by_node:
            self.findings_by_node[node_id] = finding
            return
        self.findings_by_node[node_id] = finding

    def append_capsule(self, capsule: EvidenceCapsule) -> None:
        self.capsules_by_node[capsule.node_id] = capsule

    def get(self, node_id: int) -> Finding | None:
        return self.findings_by_node.get(node_id)

    def get_capsule(self, node_id: int) -> EvidenceCapsule | None:
        return self.capsules_by_node.get(node_id)

    def all(self) -> list[Finding]:
        return [self.findings_by_node[k] for k in sorted(self.findings_by_node.keys())]

    def all_capsules(self) -> list[EvidenceCapsule]:
        return [self.capsules_by_node[k] for k in sorted(self.capsules_by_node.keys())]

    def interpolate(self, text: str) -> str:
        """Replace <A.I> tokens with the answer of node I."""
        def repl(m: re.Match[str]) -> str:
            try:
                node_id = int(m.group(1))
            except ValueError:
                return m.group(0)
            f = self.findings_by_node.get(node_id)
            if f is None or not f.answer:
                return m.group(0)
            return f.answer
        return re.sub(r"<A\.(\d+)>", repl, text)


# Backwards compatibility alias
FindingsBus = WorkingMemory

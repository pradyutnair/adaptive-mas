"""FindingsBus: shared append-only memory across Solvers.

This is the explicit collaboration substrate referenced in the thesis (Q2).
- Solvers append Findings here.
- Downstream Solvers read parent Findings via tag interpolation.
- The Synthesizer reads the entire bus.

No mutation in place. Append-only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from .types import Finding


@dataclass
class FindingsBus:
    findings_by_node: dict[int, Finding] = field(default_factory=dict)

    def append(self, node_id: int, finding: Finding) -> None:
        if node_id in self.findings_by_node:
            raise ValueError(f'Finding already exists for node {node_id}')
        self.findings_by_node[node_id] = finding

    def get(self, node_id: int) -> Finding | None:
        return self.findings_by_node.get(node_id)

    def all(self) -> list[Finding]:
        return [self.findings_by_node[k] for k in sorted(self.findings_by_node.keys())]

    def interpolate(self, text: str) -> str:
        """Replace <A.I> tokens with the answer of node I from this bus.

        Tag format borrowed from Plan*RAG (their <A.I.J> notation). We use the
        simpler <A.I> form: replace with findings_by_node[I].answer.
        """
        def repl(m: re.Match[str]) -> str:
            try:
                node_id = int(m.group(1))
            except ValueError:
                return m.group(0)
            f = self.findings_by_node.get(node_id)
            if f is None or not f.answer:
                return m.group(0)
            return f.answer
        return re.sub(r'<A\.(\d+)>', repl, text)

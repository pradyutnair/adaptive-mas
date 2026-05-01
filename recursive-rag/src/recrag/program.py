from __future__ import annotations

import dspy

from .pipeline import ReactRagPipeline


class ReactProgram(dspy.Module):
    """Thin wrapper exposing the ReAct predictor for GEPA."""

    def __init__(self, pipeline: ReactRagPipeline):
        super().__init__()
        self.pipeline = pipeline
        self.react = pipeline.react.react
        self.extract = pipeline.react.extract

    def forward(self, question: str):
        return self.pipeline.react(question=question)

"""Tests for adaptive_sage.types dataclasses and config loading."""

import json
import os
import sys
from pathlib import Path

import pytest

# Ensure src/ is on the import path so we can import src.adaptive_sage and src.arag
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_sage.types import EvidenceCapsule, Fact, PipelineResult, StepTrace
from arag.core.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _config_path(variant: str) -> str:
    """Return absolute path string for a variant YAML config."""
    return str(CONFIGS_DIR / f"{variant}.yaml")


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------

class TestFact:
    def test_construction(self):
        f = Fact(text="Paris is the capital of France", confidence=0.95,
                 support_ids=["c1", "c2"], source_step=1)
        assert f.text == "Paris is the capital of France"
        assert f.confidence == 0.95
        assert f.support_ids == ["c1", "c2"]
        assert f.source_step == 1

    def test_defaults(self):
        f = Fact(text="hello", confidence=0.5)
        assert f.support_ids == []
        assert f.source_step == 0

    def test_to_dict(self):
        f = Fact(text="x", confidence=0.8, support_ids=["a"], source_step=2)
        d = f.to_dict()
        assert d["text"] == "x"
        assert d["confidence"] == 0.8
        assert d["support_ids"] == ["a"]
        assert d["source_step"] == 2
        assert d["confidence_self"] == 0.0
        assert d["confidence_retrieval"] == 0.0
        assert d["slot_filled"] is False
        assert d["slot_name"] == ""

    def test_from_dict(self):
        d = {"text": "y", "confidence": 0.3, "support_ids": [], "source_step": 0}
        f = Fact.from_dict(d)
        assert f.text == "y"
        assert f.confidence == 0.3

    def test_roundtrip(self):
        f = Fact(text="z", confidence=1.0, support_ids=["p", "q"], source_step=5)
        assert Fact.from_dict(f.to_dict()) == f

    def test_json_serializable(self):
        f = Fact(text="test", confidence=0.9, support_ids=["1"], source_step=0)
        s = json.dumps(f.to_dict())
        assert isinstance(s, str)
        assert json.loads(s) == f.to_dict()


# ---------------------------------------------------------------------------
# StepTrace
# ---------------------------------------------------------------------------

class TestStepTrace:
    def test_construction(self):
        t = StepTrace(step=0, action="answer", sub_question=None,
                      claim=None, fact_added=False, tokens=42)
        assert t.step == 0
        assert t.action == "answer"
        assert t.sub_question is None
        assert t.fact_added is False
        assert t.tokens == 42

    def test_spawn_action(self):
        t = StepTrace(step=1, action="spawn", sub_question="Who directed Inception?",
                      claim=None, fact_added=True, tokens=150)
        assert t.action == "spawn"
        assert t.sub_question == "Who directed Inception?"
        assert t.fact_added is True

    def test_verify_action(self):
        t = StepTrace(step=2, action="verify", sub_question=None,
                      claim="Paris is in Germany", fact_added=False, tokens=80)
        assert t.action == "verify"
        assert t.claim == "Paris is in Germany"

    def test_to_dict_keys(self):
        """StepTrace.to_dict() must contain all required keys."""
        t = StepTrace(step=3, action="answer", sub_question=None,
                      claim=None, fact_added=False, tokens=10)
        d = t.to_dict()
        required_keys = {"step", "action", "sub_question", "fact_added", "tokens"}
        assert required_keys.issubset(d.keys())

    def test_from_dict(self):
        d = {"step": 1, "action": "spawn", "sub_question": "q",
             "claim": None, "fact_added": True, "tokens": 100}
        t = StepTrace.from_dict(d)
        assert t.step == 1
        assert t.action == "spawn"

    def test_roundtrip(self):
        t = StepTrace(step=2, action="verify", sub_question="q?", claim="c",
                      fact_added=False, tokens=55)
        assert StepTrace.from_dict(t.to_dict()) == t

    def test_json_serializable(self):
        t = StepTrace(step=0, action="answer", tokens=5)
        s = json.dumps(t.to_dict())
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# EvidenceCapsule
# ---------------------------------------------------------------------------

class TestEvidenceCapsule:
    def test_construction(self):
        f = Fact(text="Eiffel Tower is in Paris", confidence=0.99,
                 support_ids=["c3"], source_step=1)
        cap = EvidenceCapsule(answer="Paris", fact=f,
                              support_snippets=["The Eiffel Tower..."])
        assert cap.answer == "Paris"
        assert cap.fact.text == "Eiffel Tower is in Paris"
        assert len(cap.support_snippets) == 1

    def test_defaults(self):
        f = Fact(text="x", confidence=0.5)
        cap = EvidenceCapsule(answer="a", fact=f)
        assert cap.support_snippets == []

    def test_to_dict(self):
        f = Fact(text="x", confidence=0.7, support_ids=["1"], source_step=0)
        cap = EvidenceCapsule(answer="ans", fact=f, support_snippets=["s1"])
        d = cap.to_dict()
        assert d["answer"] == "ans"
        assert isinstance(d["fact"], dict)
        assert d["support_snippets"] == ["s1"]

    def test_from_dict(self):
        f = Fact(text="x", confidence=0.7, support_ids=["1"], source_step=0)
        cap = EvidenceCapsule(answer="ans", fact=f, support_snippets=["s1"])
        d = cap.to_dict()
        cap2 = EvidenceCapsule.from_dict(d)
        assert cap2.answer == "ans"
        assert cap2.fact.text == "x"
        assert cap2.support_snippets == ["s1"]

    def test_roundtrip(self):
        f = Fact(text="abc", confidence=0.9, support_ids=["i1", "i2"], source_step=3)
        cap = EvidenceCapsule(answer="yes", fact=f, support_snippets=["snip1", "snip2"])
        cap2 = EvidenceCapsule.from_dict(cap.to_dict())
        assert cap2 == cap


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def _make_result(self) -> PipelineResult:
        f1 = Fact(text="f1", confidence=0.9, support_ids=["c1"], source_step=1)
        f2 = Fact(text="f2", confidence=0.8, support_ids=["c2"], source_step=2)
        traces = [
            StepTrace(step=0, action="spawn", sub_question="q1", fact_added=True, tokens=50),
            StepTrace(step=1, action="spawn", sub_question="q2", fact_added=True, tokens=60),
            StepTrace(step=2, action="answer", fact_added=False, tokens=30),
        ]
        return PipelineResult(
            question_id="q_001",
            question="What is the capital of France?",
            answer="Paris",
            step_trace=traces,
            num_subagent_calls=2,
            num_verify_calls=0,
            total_tokens=140,
            facts_used=[f1, f2],
        )

    def test_construction(self):
        r = self._make_result()
        assert r.question_id == "q_001"
        assert r.answer == "Paris"
        assert len(r.step_trace) == 3
        assert r.num_subagent_calls == 2
        assert r.total_tokens == 140
        assert len(r.facts_used) == 2

    def test_defaults(self):
        r = PipelineResult(question_id="q", question="q?", answer="a")
        assert r.step_trace == []
        assert r.num_subagent_calls == 0
        assert r.num_verify_calls == 0
        assert r.total_tokens == 0
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.facts_used == []

    def test_to_dict_metadata(self):
        """PipelineResult.to_dict() produces JSON-serializable output with all metadata."""
        r = self._make_result()
        d = r.to_dict()
        # All top-level keys present
        assert "question_id" in d
        assert "question" in d
        assert "answer" in d
        assert "step_trace" in d
        assert "num_subagent_calls" in d
        assert "num_verify_calls" in d
        assert "total_tokens" in d
        assert "facts_used" in d
        # Nested structures serialized correctly
        assert isinstance(d["step_trace"], list)
        assert len(d["step_trace"]) == 3
        assert all(isinstance(t, dict) for t in d["step_trace"])
        assert isinstance(d["facts_used"], list)
        assert len(d["facts_used"]) == 2
        assert all(isinstance(f, dict) for f in d["facts_used"])

    def test_to_dict_json_serializable(self):
        r = self._make_result()
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)
        loaded = json.loads(s)
        assert loaded["answer"] == "Paris"

    def test_to_json(self):
        r = self._make_result()
        s = r.to_json()
        assert isinstance(s, str)
        loaded = json.loads(s)
        assert loaded["question_id"] == "q_001"

    def test_from_dict(self):
        r = self._make_result()
        d = r.to_dict()
        r2 = PipelineResult.from_dict(d)
        assert r2.question_id == r.question_id
        assert r2.answer == r.answer
        assert len(r2.step_trace) == len(r.step_trace)
        assert len(r2.facts_used) == len(r.facts_used)
        assert r2.total_tokens == r.total_tokens

    def test_roundtrip(self):
        r = self._make_result()
        r2 = PipelineResult.from_dict(r.to_dict())
        assert r2 == r


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Verify YAML configs load correctly and all nested keys accessible via dot-notation."""

    VARIANTS = ["m1", "s0", "s1", "s2", "s3", "s4"]

    EXPECTED_MAX_STEPS = {
        "m1": 4,
        "s0": 0,
        "s1": 1,
        "s2": 2,
        "s3": 3,
        "s4": 4,
    }

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_config_file_exists(self, variant: str):
        path = _config_path(variant)
        assert os.path.isfile(path), f"Config file missing: {path}"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_config_loads(self, variant: str):
        c = Config.from_yaml(_config_path(variant))
        assert c is not None
        assert isinstance(c.to_dict(), dict)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_variant_name(self, variant: str):
        c = Config.from_yaml(_config_path(variant))
        assert c.get("variant") == variant

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_orchestrator_max_steps(self, variant: str):
        c = Config.from_yaml(_config_path(variant))
        expected = self.EXPECTED_MAX_STEPS[variant]
        assert c.get("orchestrator.max_steps") == expected, \
            f"{variant}: expected max_steps={expected}, got {c.get('orchestrator.max_steps')}"

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_fact_memory_cap(self, variant: str):
        c = Config.from_yaml(_config_path(variant))
        assert c.get("fact_memory.capacity") == 4

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_evidence_capsule_limit(self, variant: str):
        c = Config.from_yaml(_config_path(variant))
        assert c.get("investigator.evidence_capsule_limit") == 2

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_llm_params(self, variant: str):
        c = Config.from_yaml(_config_path(variant))
        assert c.get("llm.model") == "Qwen/Qwen3-8B"
        assert c.get("llm.base_url") is not None
        assert isinstance(c.get("llm.temperature"), float)
        assert isinstance(c.get("llm.max_tokens"), int)
        chat_kwargs = c.get("llm.chat_template_kwargs")
        assert chat_kwargs is not None
        assert chat_kwargs.get("enable_thinking") is True

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_data_paths(self, variant: str):
        c = Config.from_yaml(_config_path(variant))
        assert c.get("data.chunks_file") is not None
        assert c.get("data.index_dir") is not None
        assert c.get("data.embedding_model") is not None

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_runner_params(self, variant: str):
        c = Config.from_yaml(_config_path(variant))
        assert isinstance(c.get("runner.concurrency"), int)
        assert c.get("runner.concurrency") > 0
        assert c.get("runner.checkpoint") is True

    def test_m1_max_verify_calls(self):
        c = Config.from_yaml(_config_path("m1"))
        assert c.get("orchestrator.max_verify_calls") == 1

    def test_s0_no_verify(self):
        c = Config.from_yaml(_config_path("s0"))
        assert c.get("orchestrator.max_verify_calls") == 0

    def test_m1_3_typed_bridge_sufficiency_loads(self):
        c = Config.from_yaml(_config_path("m1_3.typed_bridge_sufficiency"))
        assert c.get("variant") == "m1_3_typed_bridge_sufficiency"
        assert c.get("adaptive.sufficiency_bridge_first_probe") is True
        assert c.get("adaptive.sufficiency_split_assessment") is True
        assert c.get("adaptive.sufficiency_typed_one_shot_followup") is True

    def test_m1_4_typed_followup_sufficiency_loads(self):
        c = Config.from_yaml(_config_path("m1_4.typed_followup_sufficiency"))
        assert c.get("variant") == "m1_4_typed_followup_sufficiency"
        assert c.get("adaptive.sufficiency_bridge_first_probe") is False
        assert c.get("adaptive.sufficiency_split_assessment") is True
        assert c.get("adaptive.sufficiency_typed_one_shot_followup") is True

    def test_m1_6_hybrid_sufficiency_loads(self):
        c = Config.from_yaml(_config_path("m1_6.hybrid_sufficiency"))
        assert c.get("variant") == "m1_6_hybrid_sufficiency"
        assert c.get("adaptive.sufficiency_controller") is True
        assert c.get("adaptive.sufficiency_typed_plan_exec_on_hard") is True
        assert c.get("adaptive.sufficiency_recurse_only_after_plan_exec_failure") is True
        assert c.get("adaptive.sufficiency_max_recovery_steps") == 1

    def test_m2_1_structure_aware_adaptive_loads(self):
        c = Config.from_yaml(_config_path("m2_1.structure_aware_adaptive"))
        assert c.get("variant") == "m2_1_structure_aware_adaptive"
        assert c.get("adaptive.execution_mode_controller") is True
        assert c.get("adaptive.enable_slot_rewrite") is True
        assert c.get("adaptive.enable_recursive_recovery") is True

    def test_m3_1a_structure_adaptive_slot_exec_loads(self):
        c = Config.from_yaml(_config_path("m3_1a.structure_adaptive_slot_exec"))
        assert c.get("variant") == "m3_1a_structure_adaptive_slot_exec"
        assert c.get("adaptive.execution_mode_controller") is True
        assert c.get("adaptive.assess_after_plan_step") is True
        assert c.get("adaptive.enable_slot_rewrite") is False
        assert c.get("adaptive.enable_recursive_recovery") is False

    def test_m3_1_main_structure_adaptive_slot_exec_loads(self):
        c = Config.from_yaml(_config_path("m3_1.structure_adaptive_slot_exec"))
        assert c.get("variant") == "m3_1_structure_adaptive_slot_exec"
        assert c.get("adaptive.execution_mode_controller") is True
        assert c.get("adaptive.assess_after_plan_step") is True
        assert c.get("adaptive.enable_slot_rewrite") is True
        assert c.get("adaptive.enable_recursive_recovery") is True
        assert c.get("runner.concurrency") == 16

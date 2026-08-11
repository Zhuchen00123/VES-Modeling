"""R4: LLMRegressionGenerator with a fake client (no network, no API key)."""

from __future__ import annotations

import numpy as np
import pytest
from ves.artifact import ArtifactContract, RawArtifact
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem
from ves.record import Candidate, VerifiedCandidate

from ves_modeling.regression.generator import LLMRegressionGenerator

DUMMY_CODE = "print('candidate')"


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.responses: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("no scripted response left")
        return self.responses.pop(0)


def make_problem() -> VerifiedProblem:
    from ves_modeling.regression.context import RegressionVerificationContext
    from ves_modeling.regression.verifier import RegressionVerifier

    labels = np.array([1.0, 2.0, 3.0])

    def factory() -> RegressionVerificationContext:
        return RegressionVerificationContext(
            labels, dataset_name="unit", expected_count=3
        )

    return VerifiedProblem(
        contract=ArtifactContract(
            filename="predictions.json",
            media_type="application/json",
            required_fields=("predictions",),
        ),
        context_factory=factory,
        verifier=RegressionVerifier(),
        judge_spec=JudgeSpec(
            objectives=(ObjectiveSpec(observation="rmse", direction=Direction.MINIMIZE),),
            gates=(Gate(name="finite", observation="rmse"),),
        ),
    )


def make_anchor(rmse: float, mae: float) -> VerifiedCandidate:
    from ves.evidence import Evidence, Observation
    from ves.record import VerificationRecord

    record = VerificationRecord(
        verifier_version="0.1.0",
        context_id="regression:unit",
        context_fingerprint="f" * 64,
        evidence=Evidence(
            observations=(
                Observation(value=rmse, provenance="host:hidden-test", name="rmse"),
                Observation(value=mae, provenance="host:hidden-test", name="mae"),
            )
        ),
    )
    candidate = Candidate.draft(DUMMY_CODE)
    return VerifiedCandidate(
        candidate=candidate,
        artifact=RawArtifact(name="predictions.json", content="{}", producer="test"),
        record=record,
    )


def test_draft_prompt_contract() -> None:
    llm = FakeLlm()
    llm.responses = [DUMMY_CODE]
    generator = LLMRegressionGenerator(llm)
    problem = make_problem()
    code = generator.draft(problem, 0)
    assert code == DUMMY_CODE
    prompt = llm.prompts[0]
    assert "/data/train.csv" in prompt
    assert "/data/test_features.csv" in prompt
    assert "/output/predictions.json" in prompt
    assert "target" in prompt
    assert "host verifier" in prompt.lower()
    assert "do not use pip install" in prompt.lower()


def test_improve_prompt_uses_host_evidence() -> None:
    llm = FakeLlm()
    llm.responses = [DUMMY_CODE]
    generator = LLMRegressionGenerator(llm)
    problem = make_problem()
    anchor = make_anchor(rmse=42.5, mae=33.25)
    generator.improve(problem, anchor)
    prompt = llm.prompts[0]
    assert DUMMY_CODE in prompt  # previous candidate code
    assert "RMSE: 42.500000" in prompt
    assert "MAE: 33.250000" in prompt
    # Evidence must be host-derived; no self-claimed metric accepted.
    assert "claimed" not in prompt.lower()
    assert "/output/predictions.json" in prompt


def test_improve_prompt_rejects_nonfinite_anchor() -> None:
    llm = FakeLlm()
    llm.responses = [DUMMY_CODE]
    generator = LLMRegressionGenerator(llm)
    problem = make_problem()
    anchor = make_anchor(rmse=float("nan"), mae=1.0)
    with pytest.raises(ValueError, match="rmse"):
        generator.improve(problem, anchor)

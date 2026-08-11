"""R7.3 Batch A: unified search tree, run.json classification, provenance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.regression import capabilities, run_regression_search
from ves_modeling.regression.diagnostics import ClassifyingSearchEngine
from ves_modeling.regression.generator import MockRegressionGenerator
from ves_modeling.regression.problem import build_regression_problem
from ves_modeling.regression.provenance import file_hashes, sanitize_provider
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    LocalRegressionRunner,
    RunResult,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "candidates"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _make_data(root: Path, n: int = 30) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2))
    y = 3.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(scale=0.1, size=n)
    split = int(n * 0.7)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train = pd.DataFrame(
        {**{f"x{i}": x[:split, i] for i in range(2)}, "target": y[:split]}
    )
    test = pd.DataFrame({f"x{i}": x[split:, i] for i in range(2)})
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    pd.DataFrame({"target": y[split:]}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    return public, host


def test_search_unified_tree(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.run_dir.name == result.run_id
    candidate = result.run_dir / "candidates" / "draft0"
    assert candidate.is_dir()
    assert (candidate / "solution.py").is_file()
    assert (candidate / "stdout.log").is_file()
    assert (candidate / "stderr.log").is_file()
    assert (candidate / "predictions.json").is_file()
    assert (candidate / "run.json").is_file()
    assert (result.run_dir / "config.json").is_file()
    assert (result.run_dir / "provenance.json").is_file()
    assert (result.run_dir / "summary.json").is_file()
    assert (result.run_dir / "best_solution.py").is_file()


def test_search_run_json_classifies_execution_failure(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "linear_regression.py").write_text(
        "import sys\nsys.exit(3)\n", encoding="utf-8"
    )
    (fixtures / "random_forest.py").write_text(
        (FIXTURES / "linear_regression.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = run_regression_search(
        public,
        host,
        drafts=2,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=fixtures,
    )
    assert result.status == "verified"
    assert result.rejected == 1
    expected = {"draft0": "execution_failed", "draft1": "verified"}
    for candidate_name, status in expected.items():
        run_json = json.loads(
            (
                result.run_dir
                / "candidates"
                / candidate_name
                / "run.json"
            ).read_text(encoding="utf-8")
        )
        assert run_json["candidate"] == candidate_name
        assert run_json["status"] == status
    summary = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert {
        entry["candidate"]: entry["status"] for entry in summary["candidates"]
    } == expected


def test_classifying_engine_records_all_statuses(tmp_path: Path) -> None:
    """Application-layer diagnostics classify every rejection reason."""
    public, host = _make_data(tmp_path / "data")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "crash.py").write_text(
        "import sys\nsys.exit(3)\n", encoding="utf-8"
    )
    (fixtures / "no_artifact.py").write_text(
        'print("no artifact")\n', encoding="utf-8"
    )
    (fixtures / "bad_json.py").write_text(
        'import os\n'
        'os.makedirs(os.environ["REGRESSION_OUTPUT_DIR"], exist_ok=True)\n'
        'with open(os.path.join(os.environ["REGRESSION_OUTPUT_DIR"], '
        '"predictions.json"), "w") as fh:\n'
        '    fh.write("not-json{")\n',
        encoding="utf-8",
    )
    (fixtures / "good.py").write_text(
        (FIXTURES / "linear_regression.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (fixtures / "slow.py").write_text(
        "import time\ntime.sleep(30)\n", encoding="utf-8"
    )
    problem = build_regression_problem(public, host)
    runner = LocalRegressionRunner(
        workspace=tmp_path / "runs",
        data_dir=public,
        timeout_seconds=2.0,
        run_layout="flat",
    )
    generator = MockRegressionGenerator(
        fixtures,
        drafts=(
            "crash.py",
            "no_artifact.py",
            "bad_json.py",
            "good.py",
            "slow.py",
        ),
        improves=(),
    )
    engine = ClassifyingSearchEngine(
        problem=problem,
        generator=generator,
        runner=runner,
        drafts=5,
        improves=0,
    )
    engine.search()
    assert {
        attempt: outcome.status for attempt, outcome in engine.outcomes.items()
    } == {
        "draft0": "execution_failed",
        "draft1": "artifact_missing",
        "draft2": "artifact_invalid",
        "draft3": "verified",
        "draft4": "timeout",
    }
    assert engine.outcomes["draft3"].core_candidate_id
    assert set(engine.outcomes["draft3"].evidence or {}) == {"rmse", "mae"}


def test_summary_json_round_trip_no_core_objects(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    summary = result.to_summary()
    json.dumps(summary)  # no custom encoder required
    assert summary["run_id"] == result.run_id
    assert summary["status"] == "verified"
    assert "records" not in summary
    assert "run_dir" not in summary
    assert "evidence" not in summary
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "verified"
    assert "run_dir" not in persisted


def test_to_summary_has_candidates_and_artifact_refs(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    summary = result.to_summary()
    assert summary["candidates"] == [
        {"candidate": "draft0", "status": "verified"}
    ]
    assert summary["best_solution"] == "best_solution.py"
    assert summary["summary"] == "summary.json"
    assert summary["provenance"] == "provenance.json"


def test_split_metadata_validated_before_search(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="JSON-serializable"):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
            split_metadata={"seed": {1, 2}},
        )
    for bad in ("api_key", "token", "password", "secret", "API_KEY"):
        with pytest.raises(ValueError, match="sensitive"):
            run_regression_search(
                public,
                host,
                drafts=1,
                improves=0,
                workspace=tmp_path / "workspace",
                fixture_dir=FIXTURES,
                split_metadata={bad: "x"},
            )
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        split_metadata={"seed": 42, "strategy": "kfold"},
    )
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["split_metadata"] == {"seed": 42, "strategy": "kfold"}


def test_split_metadata_requires_dict_and_scans_nested(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data")
    workspace = tmp_path / "workspace"
    for bad in ([1, 2], "label", 42):
        with pytest.raises(ValueError, match="must be a dict"):
            run_regression_search(
                public,
                host,
                drafts=1,
                improves=0,
                workspace=workspace,
                fixture_dir=FIXTURES,
                split_metadata=bad,
            )
    nested_bad = (
        {"model": {"api_key": "x"}},
        {"layers": [{"token": "x"}]},
        {"items": [{"inner": {"password": "x"}}]},
    )
    for bad in nested_bad:
        with pytest.raises(ValueError, match="sensitive"):
            run_regression_search(
                public,
                host,
                drafts=1,
                improves=0,
                workspace=workspace,
                fixture_dir=FIXTURES,
                split_metadata=bad,
            )
    assert not workspace.exists()
    assert not any(Path(tmp_path).rglob("provenance.json"))


def test_split_metadata_rejects_nested_tuple_secret(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data")
    workspace = tmp_path / "workspace"
    with pytest.raises(ValueError, match="sensitive"):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=workspace,
            fixture_dir=FIXTURES,
            split_metadata={"nested": ({"api_key": "secret"},)},
        )
    assert not workspace.exists()
    assert not any(Path(tmp_path).rglob("provenance.json"))


def test_split_metadata_rejects_non_finite_without_workspace(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data")
    workspace = tmp_path / "workspace"
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="JSON-serializable"):
            run_regression_search(
                public,
                host,
                drafts=1,
                improves=0,
                workspace=workspace,
                fixture_dir=FIXTURES,
                split_metadata={"score": value},
            )
    assert not workspace.exists()
    assert not any(Path(tmp_path).rglob("provenance.json"))


def test_split_metadata_canonicalizes_tuples_to_lists(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        split_metadata={"items": (1, 2), "nested": {"tags": ("a", "b")}},
    )
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["split_metadata"] == {
        "items": [1, 2],
        "nested": {"tags": ["a", "b"]},
    }


def test_search_missing_public_file_fail_fast(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    (public / "train.csv").unlink()
    workspace = tmp_path / "workspace"
    with pytest.raises(FileNotFoundError, match="train.csv"):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=workspace,
            fixture_dir=FIXTURES,
        )
    if workspace.exists():
        assert not any(workspace.rglob("provenance.json"))


def test_file_hashes_strict_flag(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("a", encoding="utf-8")
    assert set(file_hashes(data, ("train.csv", "test_features.csv"))) == {
        "train.csv"
    }
    with pytest.raises(FileNotFoundError, match="test_features.csv"):
        file_hashes(
            data, ("train.csv", "test_features.csv"), strict=True
        )


def test_provenance_has_runtime_environment(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["environment"]["python"]
    assert provenance["environment"]["platform"]


def test_search_image_digest_passthrough(tmp_path: Path, monkeypatch) -> None:
    public, host = _make_data(tmp_path / "data")
    digest = "sha256:" + "b" * 64

    class _FakeClient:
        def complete(self, prompt: str) -> str:
            return (FIXTURES / "linear_regression.py").read_text(
                encoding="utf-8"
            )

    def fake_run(self, code: str, run_id: str) -> RunResult:
        run_root = Path(self.config.workspace) / run_id
        code_dir = run_root / "code"
        output_dir = run_root / "output"
        code_dir.mkdir(parents=True)
        output_dir.mkdir()
        (code_dir / "solution.py").write_text(code, encoding="utf-8")
        test = pd.read_csv(self.config.data_dir / "test_features.csv")
        (output_dir / "predictions.json").write_text(
            json.dumps({"predictions": [0.0] * len(test)}), encoding="utf-8"
        )
        (run_root / "stdout.log").write_text("", encoding="utf-8")
        (run_root / "stderr.log").write_text("", encoding="utf-8")
        return RunResult(
            succeeded=True,
            run_dir=output_dir,
            run_root=run_root,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(DockerRegressionRunner, "run", fake_run)
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        generator="llm",
        client=_FakeClient(),
        fallback_code=(FIXTURES / "linear_regression.py").read_text(
            encoding="utf-8"
        ),
        workspace=tmp_path / "workspace",
        image_digest=digest,
    )
    assert result.status == "verified"
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["runner"]["kind"] == "docker"
    assert provenance["runner"]["image_digest"] == digest
    assert provenance["runner"]["image_digest_status"] == "configured"


def test_provenance_hashes_and_versions(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["run_id"] == result.run_id
    assert provenance["generator"] == "mock"
    assert provenance["provider"] is None
    assert provenance["runner"]["kind"] == "local"
    for name in ("train.csv", "test_features.csv"):
        assert _HEX64.fullmatch(provenance["inputs"]["public"][name])
    assert _HEX64.fullmatch(provenance["inputs"]["host"]["hidden_test_labels.csv"])
    assert provenance["best"]["candidate"] == "draft0"
    assert provenance["best"]["candidate_id"]
    assert provenance["best"]["code_sha256"] == hashlib.sha256(
        (result.run_dir / "best_solution.py").read_bytes()
    ).hexdigest()
    assert provenance["versions"]["ves_modeling"]["version"] == "0.1.0"
    assert provenance["versions"]["ves_core"]["version"] == "0.1.0"
    raw = (result.run_dir / "provenance.json").read_text(
        encoding="utf-8"
    )
    assert "api_key" not in raw.lower()
    assert "Bearer" not in raw
    assert "VES_MODELING_LLM_API_KEY" not in raw
    assert "VES_MODELING_LLM_BASE_URL" not in raw


def test_provenance_hashes_change_when_data_changes(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    first = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "w1",
        fixture_dir=FIXTURES,
    )
    first_hash = json.loads(
        (first.run_dir / "provenance.json").read_text(encoding="utf-8")
    )["inputs"]["public"]["train.csv"]

    train = pd.read_csv(public / "train.csv")
    extra = pd.DataFrame({"x0": [0.0], "x1": [0.0], "target": [0.0]})
    pd.concat([train, extra], ignore_index=True).to_csv(
        public / "train.csv", index=False
    )
    second = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "w2",
        fixture_dir=FIXTURES,
    )
    second_hash = json.loads(
        (second.run_dir / "provenance.json").read_text(encoding="utf-8")
    )["inputs"]["public"]["train.csv"]
    assert second_hash != first_hash


def test_sanitize_provider_strips_credentials() -> None:
    provider = sanitize_provider(
        "https://user:sekret@opencode.ai/zen/go/v1", "deepseek-v4-flash"
    )
    assert provider == {"host": "opencode.ai", "model": "deepseek-v4-flash"}
    assert "sekret" not in json.dumps(provider)
    assert sanitize_provider(None, None) is None


def test_capabilities_json_serializable() -> None:
    payload = capabilities()
    json.dumps(payload)
    assert payload["api_schema_version"] == "1.0"
    assert "run_regression_search" in payload["operations"]
    assert "apply_regression_solution" in payload["operations"]
    assert "verified" in payload["candidate_statuses"]
    assert payload["apply_statuses"] == ["produced_unverified"]

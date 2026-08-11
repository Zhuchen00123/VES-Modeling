"""R7.3 Batch A: apply_regression_solution (trusted local, no metrics)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.regression import (
    ApplyRegressionResult,
    apply_regression_solution,
)
from ves_modeling.regression.runner import (
    DockerRegressionRunner,
    LocalRegressionRunner,
    RunResult,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "candidates"
GOOD_CODE = (FIXTURES / "linear_regression.py").read_text(encoding="utf-8")


def _make_data(root: Path, n: int = 30) -> Path:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2))
    y = 3.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(scale=0.1, size=n)
    split = int(n * 0.7)
    public = root / "public"
    public.mkdir(parents=True)
    train = pd.DataFrame(
        {**{f"x{i}": x[:split, i] for i in range(2)}, "target": y[:split]}
    )
    test = pd.DataFrame({f"x{i}": x[split:, i] for i in range(2)})
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    return public


def test_apply_trusted_local_produces_predictions(tmp_path: Path) -> None:
    public = _make_data(tmp_path / "data")
    result = apply_regression_solution(
        GOOD_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyRegressionResult)
    assert result.status == "produced_unverified"
    assert result.runner == "local"
    assert result.predictions_path is not None
    assert result.predictions_path.is_file()
    candidate = result.run_dir / "candidate"
    assert candidate.is_dir()
    assert (candidate / "solution.py").is_file()
    assert (candidate / "stdout.log").is_file()
    assert (candidate / "stderr.log").is_file()
    assert (candidate / "run.json").is_file()
    assert result.stdout_log == candidate / "stdout.log"
    assert result.stderr_log == candidate / "stderr.log"
    payload = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    assert len(payload["predictions"]) == 9
    assert result.code_sha256 == hashlib.sha256(
        GOOD_CODE.encode("utf-8")
    ).hexdigest()
    assert set(result.data_sha256) == {"train.csv", "test_features.csv"}
    assert result.predictions_sha256 == hashlib.sha256(
        result.predictions_path.read_bytes()
    ).hexdigest()


def test_apply_accepts_path_solution(tmp_path: Path) -> None:
    public = _make_data(tmp_path / "data")
    solution_path = tmp_path / "solution.py"
    solution_path.write_text(GOOD_CODE, encoding="utf-8")
    result = apply_regression_solution(
        solution_path,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert result.status == "produced_unverified"
    assert result.predictions_path is not None


def test_apply_result_has_no_metrics(tmp_path: Path) -> None:
    public = _make_data(tmp_path / "data")
    result = apply_regression_solution(
        GOOD_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert not hasattr(result, "best_rmse")
    assert not hasattr(result, "best_mae")
    summary = result.to_summary()
    json.dumps(summary)  # must round-trip without a custom encoder
    assert "rmse" not in summary
    assert "mae" not in summary
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert "rmse" not in persisted
    assert "mae" not in persisted
    assert persisted == summary


def test_apply_default_untrusted_uses_docker_error_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = _make_data(tmp_path / "data")
    monkeypatch.setattr(
        DockerRegressionRunner, "is_available", lambda self: False
    )
    with pytest.raises(RuntimeError, match="Docker"):
        apply_regression_solution(
            GOOD_CODE,
            public,
            workspace=tmp_path / "runs",
        )


def test_apply_reports_execution_failure(tmp_path: Path) -> None:
    public = _make_data(tmp_path / "data")
    with pytest.raises(RuntimeError, match="execution_failed") as excinfo:
        apply_regression_solution(
            "import sys\nsys.exit(3)\n",
            public,
            workspace=tmp_path / "runs",
            run_id="fixed",
            trusted_code=True,
        )
    assert "candidate execution_failed" in str(excinfo.value)
    candidate = tmp_path / "runs" / "fixed" / "candidate"
    run_json = json.loads((candidate / "run.json").read_text(encoding="utf-8"))
    assert run_json["status"] == "execution_failed"
    summary = json.loads(
        (tmp_path / "runs" / "fixed" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "execution_failed"
    assert "rmse" not in summary
    assert "mae" not in summary
    assert (tmp_path / "runs" / "fixed" / "config.json").is_file()
    assert (tmp_path / "runs" / "fixed" / "provenance.json").is_file()


def test_apply_reports_missing_artifact(tmp_path: Path) -> None:
    public = _make_data(tmp_path / "data")
    with pytest.raises(RuntimeError, match="artifact_missing"):
        apply_regression_solution(
            'print("no artifact")\n',
            public,
            workspace=tmp_path / "runs",
            run_id="fixed",
            trusted_code=True,
        )
    run_json = json.loads(
        (
            tmp_path / "runs" / "fixed" / "candidate" / "run.json"
        ).read_text(encoding="utf-8")
    )
    assert run_json["status"] == "artifact_missing"
    summary = json.loads(
        (tmp_path / "runs" / "fixed" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "artifact_missing"


def test_apply_invalid_predictions_write_failure_artifacts(
    tmp_path: Path,
) -> None:
    public = _make_data(tmp_path / "data")
    code = (
        'import json, os\n'
        'os.makedirs(os.environ["REGRESSION_OUTPUT_DIR"], exist_ok=True)\n'
        'with open(os.path.join(os.environ["REGRESSION_OUTPUT_DIR"], '
        '"predictions.json"), "w") as fh:\n'
        '    json.dump({"predictions": [1.0, 2.0]}, fh)\n'
    )
    with pytest.raises(RuntimeError, match="artifact_invalid") as excinfo:
        apply_regression_solution(
            code,
            public,
            workspace=tmp_path / "runs",
            run_id="fixed",
            trusted_code=True,
        )
    assert "prediction count 2 != expected 9" in str(excinfo.value)
    run_json = json.loads(
        (
            tmp_path / "runs" / "fixed" / "candidate" / "run.json"
        ).read_text(encoding="utf-8")
    )
    assert run_json["status"] == "artifact_invalid"
    assert run_json["issues"]
    summary = json.loads(
        (tmp_path / "runs" / "fixed" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "artifact_invalid"
    assert "rmse" not in summary
    assert (tmp_path / "runs" / "fixed" / "config.json").is_file()
    assert (tmp_path / "runs" / "fixed" / "provenance.json").is_file()


def test_apply_rejects_unsafe_run_id(tmp_path: Path) -> None:
    public = _make_data(tmp_path / "data")
    for bad in ("../escaped", "/abs/path", "a/b", "with space"):
        with pytest.raises(ValueError, match="run_id"):
            apply_regression_solution(
                GOOD_CODE,
                public,
                workspace=tmp_path / "runs",
                run_id=bad,
                trusted_code=True,
            )
    assert not (tmp_path / "runs" / "escaped").exists()
    assert not (tmp_path / "runs" / "abs").exists()


def test_apply_duplicate_run_id_fails_without_mixing(
    tmp_path: Path,
) -> None:
    public = _make_data(tmp_path / "data")
    first = apply_regression_solution(
        GOOD_CODE,
        public,
        workspace=tmp_path / "runs",
        run_id="fixed",
        trusted_code=True,
    )
    old_summary = (first.run_dir / "summary.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        apply_regression_solution(
            'print("no artifact")\n',
            public,
            workspace=tmp_path / "runs",
            run_id="fixed",
            trusted_code=True,
        )
    # The original run artifacts stay untouched.
    assert (first.run_dir / "summary.json").read_text(
        encoding="utf-8"
    ) == old_summary


def test_apply_docker_success_with_fake_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = _make_data(tmp_path / "data")
    digest = "sha256:" + "a" * 64

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
    result = apply_regression_solution(
        GOOD_CODE,
        public,
        workspace=tmp_path / "runs",
        image_digest=digest,
    )
    assert result.status == "produced_unverified"
    assert result.runner == "docker"
    assert result.docker_digest == digest
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["runner"]["image_digest"] == digest
    assert provenance["runner"]["image_digest_status"] == "configured"


def test_apply_summary_uses_relative_paths(tmp_path: Path) -> None:
    public = _make_data(tmp_path / "data")
    result = apply_regression_solution(
        GOOD_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    summary = result.to_summary()
    assert summary["candidate_dir"] == "candidate"
    assert summary["stdout_log"] == "candidate/stdout.log"
    assert summary["stderr_log"] == "candidate/stderr.log"
    assert summary["predictions"] == "candidate/predictions.json"


def test_apply_missing_input_files_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = _make_data(tmp_path / "data")
    (public / "test_features.csv").unlink()

    def _forbid_run(*args, **kwargs):
        raise AssertionError("runner must not be called without valid inputs")

    monkeypatch.setattr(LocalRegressionRunner, "run", _forbid_run)
    with pytest.raises(FileNotFoundError, match="test_features.csv"):
        apply_regression_solution(
            GOOD_CODE,
            public,
            workspace=tmp_path / "runs",
            run_id="fixed",
            trusted_code=True,
        )
    assert not (tmp_path / "runs" / "fixed").exists()


@pytest.mark.parametrize(
    "payload_code",
    [
        'fh.write("not-json{")',
        'json.dump({"other": [1.0]}, fh)',
        'json.dump({"predictions": "nope"}, fh)',
        'json.dump({"predictions": [True] * 9}, fh)',
        'json.dump({"predictions": [float("nan")] * 9}, fh)',
        'json.dump({"predictions": [float("inf")] * 9}, fh)',
        'json.dump({"predictions": [1.0] * 2}, fh)',
    ],
)
def test_apply_rejects_attack_predictions(
    tmp_path: Path, payload_code: str
) -> None:
    public = _make_data(tmp_path / "data")
    code = (
        'import json, os\n'
        'os.makedirs(os.environ["REGRESSION_OUTPUT_DIR"], exist_ok=True)\n'
        'with open(os.path.join(os.environ["REGRESSION_OUTPUT_DIR"], '
        '"predictions.json"), "w") as fh:\n'
        f"    {payload_code}\n"
    )
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_regression_solution(
            code,
            public,
            workspace=tmp_path / "runs",
            run_id="fixed",
            trusted_code=True,
        )
    run_json = json.loads(
        (
            tmp_path / "runs" / "fixed" / "candidate" / "run.json"
        ).read_text(encoding="utf-8")
    )
    assert run_json["status"] == "artifact_invalid"
    summary = json.loads(
        (tmp_path / "runs" / "fixed" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "artifact_invalid"
    assert "rmse" not in summary
    assert "mae" not in summary


def test_apply_timeout_writes_run_json(tmp_path: Path) -> None:
    public = _make_data(tmp_path / "data")
    with pytest.raises(RuntimeError, match="timeout"):
        apply_regression_solution(
            "import time\ntime.sleep(5)\n",
            public,
            workspace=tmp_path / "runs",
            run_id="fixed",
            trusted_code=True,
            timeout_seconds=0.5,
        )
    run_json = json.loads(
        (
            tmp_path / "runs" / "fixed" / "candidate" / "run.json"
        ).read_text(encoding="utf-8")
    )
    assert run_json["status"] == "timeout"
    summary = json.loads(
        (tmp_path / "runs" / "fixed" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "timeout"

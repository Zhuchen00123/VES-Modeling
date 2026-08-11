"""B-002 regression: runners are reusable across runs (no FileExistsError)."""

from __future__ import annotations

from pathlib import Path

from ves_modeling.regression.runner import LocalRegressionRunner

CODE = """\
import json, os
os.makedirs(os.environ["REGRESSION_OUTPUT_DIR"], exist_ok=True)
with open(os.path.join(os.environ["REGRESSION_OUTPUT_DIR"], "predictions.json"), "w") as f:
    json.dump({"predictions": [1.0, 2.0, 3.0]}, f)
"""


def test_local_runner_reusable_workspace(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runner = LocalRegressionRunner(workspace=tmp_path / "runs", data_dir=data_dir)
    first = runner.run(CODE, "draft0")
    second = runner.run(CODE, "draft0")
    assert first.succeeded and second.succeeded
    assert (first.run_dir / "predictions.json").is_file()
    assert (second.run_dir / "predictions.json").is_file()

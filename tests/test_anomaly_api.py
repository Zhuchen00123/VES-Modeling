"""R13: anomaly API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.anomaly import (
    AnomalySearchResult,
    ApplyAnomalyResult,
    apply_anomaly_solution,
    capabilities,
    run_anomaly_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "anomaly"
SCORE_CODE = (FIXTURES / "isolation_forest_score.py").read_text(
    encoding="utf-8"
)
LABEL_CODE = (FIXTURES / "zscore_threshold.py").read_text(encoding="utf-8")


def _make_anomaly_data(
    root: Path,
    *,
    n_train: int = 100,
    n_test_normal: int = 30,
    n_test_anomaly: int = 10,
    seed: int = 23,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    train = rng.normal(0.0, 1.0, size=(n_train, 4))
    test_normal = rng.normal(0.0, 1.0, size=(n_test_normal, 4))
    test_anomaly = rng.normal(6.0, 1.0, size=(n_test_anomaly, 4))
    test = np.vstack([test_normal, test_anomaly])
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(train, columns=[f"f{i}" for i in range(4)]).to_csv(
        public / "train.csv", index=False
    )
    pd.DataFrame(test, columns=[f"f{i}" for i in range(4)]).to_csv(
        public / "test_features.csv", index=False
    )
    pd.DataFrame(
        {"label": ["normal"] * n_test_normal + ["anomaly"] * n_test_anomaly}
    ).to_csv(host / "hidden_test_labels.csv", index=False)
    return public, host


def test_run_anomaly_search_score_mode_verified(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    result = run_anomaly_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        output_mode="score",
    )
    assert isinstance(result, AnomalySearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_auroc is not None and result.best_auroc > 0.5
    assert result.best_average_precision is not None
    assert result.best_f1 is None
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_run_anomaly_search_label_mode_verified(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    result = run_anomaly_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        output_mode="label",
    )
    assert result.status == "verified"
    assert result.best_f1 is not None and np.isfinite(result.best_f1)
    assert result.best_balanced_accuracy is not None
    assert result.best_auroc is None
    assert result.output_mode == "label"


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    result = run_anomaly_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == result.to_summary()
    assert persisted["task"] == "anomaly"
    assert persisted["output_mode"] == "score"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["host"]["hidden_test_labels.csv"],
    )


def test_apply_score_and_label_produced_unverified(tmp_path: Path) -> None:
    public, _host = _make_anomaly_data(tmp_path / "data")
    score_result = apply_anomaly_solution(
        SCORE_CODE,
        public,
        workspace=tmp_path / "runs_score",
        trusted_code=True,
        output_mode="score",
    )
    assert isinstance(score_result, ApplyAnomalyResult)
    assert score_result.status == "produced_unverified"
    payload = json.loads(
        score_result.predictions_path.read_text(encoding="utf-8")
    )
    assert len(payload["scores"]) == 40
    assert not hasattr(score_result, "best_auroc")
    summary = score_result.to_summary()
    json.dumps(summary)
    assert "auroc" not in summary
    persisted = json.loads(
        (score_result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary

    label_result = apply_anomaly_solution(
        LABEL_CODE,
        public,
        workspace=tmp_path / "runs_label",
        trusted_code=True,
        output_mode="label",
    )
    assert label_result.status == "produced_unverified"
    payload = json.loads(
        label_result.predictions_path.read_text(encoding="utf-8")
    )
    assert len(payload["labels"]) == 40


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _host = _make_anomaly_data(tmp_path / "data")
    bad_code = SCORE_CODE.replace(
        'json.dump({"scores": [float(value) for value in scores]}, fh)',
        'json.dump({"scores": [float(value) for value in scores][:-1]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_anomaly_solution(
            bad_code,
            public,
            workspace=tmp_path / "runs",
            trusted_code=True,
            output_mode="score",
            run_id=run_id,
        )
    run_dir = tmp_path / "runs" / run_id
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "artifact_invalid"
    assert summary["error"] is not None


def test_run_anomaly_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _make_anomaly_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_anomaly_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_anomaly() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_anomaly_search",
        "apply_anomaly_solution",
    ]
    assert declaration["verified_metrics"]["score"] == [
        "auroc",
        "average_precision",
    ]
    assert declaration["data_contract"]["output_modes"] == ["score", "label"]

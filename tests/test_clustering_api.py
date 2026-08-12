"""R12: clustering API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from ves_modeling.clustering import (
    ApplyClusteringResult,
    ClusteringSearchResult,
    apply_clustering_solution,
    capabilities,
    run_clustering_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "clustering"
KMEANS_CODE = (FIXTURES / "kmeans_fit.py").read_text(encoding="utf-8")


def _make_clustering_data(
    root: Path,
    *,
    n_clusters: int = 3,
    id_col: str | None = None,
    seed: int = 23,
) -> tuple[Path, Path]:
    samples_per_cluster = 40
    test_per_cluster = 10
    X, y = make_blobs(
        n_samples=(samples_per_cluster + test_per_cluster) * n_clusters,
        centers=n_clusters,
        n_features=4,
        cluster_std=1.0,
        random_state=seed,
    )
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cluster in range(n_clusters):
        indices = np.where(y == cluster)[0]
        train_idx.extend(indices[:samples_per_cluster].tolist())
        test_idx.extend(indices[samples_per_cluster:].tolist())
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train = pd.DataFrame(X[train_idx], columns=[f"f{i}" for i in range(4)])
    test = pd.DataFrame(X[test_idx], columns=[f"f{i}" for i in range(4)])
    host_frame = pd.DataFrame(
        {"label": [int(value) for value in y[test_idx]]}
    )
    if id_col:
        train[id_col] = np.arange(1, len(train) + 1)
        test[id_col] = np.arange(1, len(test) + 1)
        host_frame[id_col] = np.arange(1, len(test) + 1)
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    host_frame.to_csv(host / "hidden_test_labels.csv", index=False)
    return public, host


def test_run_clustering_search_mock_verified(tmp_path: Path) -> None:
    public, host = _make_clustering_data(tmp_path / "data")
    result = run_clustering_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, ClusteringSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_ari is not None and result.best_ari > 0.5
    assert result.best_nmi is not None and np.isfinite(result.best_nmi)
    assert result.best_v_measure is not None
    assert result.best_silhouette is not None
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()
    candidates = list((result.run_dir / "candidates").iterdir())
    assert len(candidates) == 3


def test_search_and_apply_id_mode_e2e(tmp_path: Path) -> None:
    public, host = _make_clustering_data(
        tmp_path / "data", id_col="id_col"
    )
    result = run_clustering_search(
        public,
        host,
        drafts=2,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        row_order="id",
        id_column="id_col",
    )
    assert result.status == "verified"
    assert result.best_ari is not None and result.best_ari > 0.5
    applied = apply_clustering_solution(
        result.best_code,
        public,
        workspace=tmp_path / "apply_workspace",
        trusted_code=True,
        row_order="id",
        id_column="id_col",
    )
    assert applied.status == "produced_unverified"
    payload = json.loads(applied.predictions_path.read_text(encoding="utf-8"))
    assert all(
        set(item.keys()) == {"id", "label"}
        for item in payload["predictions"]
    )


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_clustering_data(tmp_path / "data")
    result = run_clustering_search(
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
    assert persisted["task"] == "clustering"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["host"]["hidden_test_labels.csv"],
    )


def test_apply_produced_unverified_no_metrics(tmp_path: Path) -> None:
    public, _host = _make_clustering_data(tmp_path / "data")
    result = apply_clustering_solution(
        KMEANS_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyClusteringResult)
    assert result.status == "produced_unverified"
    assert result.predictions_path is not None
    payload = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    assert len(payload["labels"]) == 30
    assert not hasattr(result, "best_ari")
    summary = result.to_summary()
    json.dumps(summary)
    assert "ari" not in summary
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _host = _make_clustering_data(tmp_path / "data")
    bad_code = KMEANS_CODE.replace(
        'payload = {"labels": labels}',
        'payload = {"labels": labels[:-1]}',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_clustering_solution(
            bad_code,
            public,
            workspace=tmp_path / "runs",
            trusted_code=True,
            run_id=run_id,
        )
    run_dir = tmp_path / "runs" / run_id
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "artifact_invalid"
    assert summary["error"] is not None


def test_run_clustering_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _make_clustering_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_clustering_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_clustering() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_clustering_search",
        "apply_clustering_solution",
    ]
    assert "ari" in declaration["verified_metrics"]
    assert declaration["data_contract"]["row_order"] == ["input", "id"]

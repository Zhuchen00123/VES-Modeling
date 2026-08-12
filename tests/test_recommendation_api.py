"""R17: recommendation API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.recommendation import (
    ApplyRecommendationResult,
    RecommendationSearchResult,
    apply_recommendation_solution,
    capabilities,
    run_recommendation_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "recommendation"
BIAS_CODE = (FIXTURES / "bias_baseline.py").read_text(encoding="utf-8")


def _make_recommendation_data(
    root: Path, *, seed: int = 23
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    users = [f"u{i}" for i in range(1, 6)]
    items = [f"i{i}" for i in range(1, 8)]
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    host_rows: list[dict] = []
    for user in users:
        base = float(rng.integers(1, 5))
        train_items = rng.choice(items, size=4, replace=False)
        for item in train_items:
            train_rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "rating": base + float(rng.integers(0, 2)),
                }
            )
        pool = [item for item in items if item not in set(train_items)]
        test_items = rng.choice(pool, size=2, replace=False)
        for item in test_items:
            rating = base + float(rng.integers(0, 3))
            test_rows.append({"user_id": user, "item_id": item})
            host_rows.append(
                {"user_id": user, "item_id": item, "rating": rating}
            )
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(
        public / "test_features.csv", index=False
    )
    pd.DataFrame(host_rows).to_csv(
        host / "hidden_test_ratings.csv", index=False
    )
    return public, host


def test_run_recommendation_search_verified(tmp_path: Path) -> None:
    public, host = _make_recommendation_data(tmp_path / "data")
    result = run_recommendation_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, RecommendationSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_rmse is not None and np.isfinite(result.best_rmse)
    assert result.best_mae is not None
    assert result.best_ndcg is not None and np.isfinite(result.best_ndcg)
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_run_recommendation_search_input_mode_verified(tmp_path: Path) -> None:
    public, host = _make_recommendation_data(tmp_path / "data")
    result = run_recommendation_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        row_order="input",
    )
    assert result.status == "verified"
    assert result.best_rmse is not None and np.isfinite(result.best_rmse)
    assert result.best_ndcg == 1.0  # input mode has no user keys
    assert result.data_contract["row_order"] == "input"


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_recommendation_data(tmp_path / "data")
    result = run_recommendation_search(
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
    assert persisted["task"] == "recommendation"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["host"]["hidden_test_ratings.csv"],
    )


def test_apply_produced_unverified_no_metrics(tmp_path: Path) -> None:
    public, _host = _make_recommendation_data(tmp_path / "data")
    result = apply_recommendation_solution(
        BIAS_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyRecommendationResult)
    assert result.status == "produced_unverified"
    assert result.predictions_path is not None
    payload = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    assert len(payload["predictions"]) == 10
    assert all(
        set(item.keys()) == {"user_id", "item_id", "prediction"}
        for item in payload["predictions"]
    )
    assert not hasattr(result, "best_rmse")
    summary = result.to_summary()
    json.dumps(summary)
    assert "rmse" not in summary
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _host = _make_recommendation_data(tmp_path / "data")
    bad_code = BIAS_CODE.replace(
        '    json.dump(payload, fh)',
        '    payload["predictions"] = payload["predictions"][:-1]\n'
        '    json.dump(payload, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_recommendation_solution(
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


def test_run_recommendation_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _make_recommendation_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_recommendation_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_recommendation() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_recommendation_search",
        "apply_recommendation_solution",
    ]
    assert "ndcg@5" in declaration["verified_metrics"]
    assert declaration["data_contract"]["row_order"] == ["input", "key"]

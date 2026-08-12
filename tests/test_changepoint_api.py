"""R25: change-point API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from ves_modeling.changepoint import (
    ApplyChangepointResult,
    ChangepointSearchResult,
    apply_changepoint_solution,
    capabilities,
    run_changepoint_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "changepoint"
CUSUM_CODE = (FIXTURES / "changepoint_cusum.py").read_text(encoding="utf-8")


def _write_data(root: Path) -> tuple[Path, Path]:
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(
        {"t": [float(i) for i in range(40)], "y": [0.0] * 40}
    ).to_csv(public / "train.csv", index=False)
    y = [0.0] * 30 + [10.0] * 30
    pd.DataFrame(
        {"t": [float(i) for i in range(60)], "y": y}
    ).to_csv(public / "test_features.csv", index=False)
    pd.DataFrame({"changepoint": [30]}).to_csv(
        host / "hidden_test_changepoints.csv", index=False
    )
    return public, host


def test_run_changepoint_search_verified(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "data")
    result = run_changepoint_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, ChangepointSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_f1 == pytest.approx(1.0)
    assert result.best_mean_distance == 0.0
    assert result.best_precision == pytest.approx(1.0)
    assert result.best_recall == pytest.approx(1.0)
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _write_data(tmp_path / "data")
    result = run_changepoint_search(
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
    assert persisted["task"] == "changepoint"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["train.csv"],
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["public"]["test_features.csv"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public, _ = _write_data(tmp_path / "data")
    result = apply_changepoint_solution(
        CUSUM_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyChangepointResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert payload["changepoints"] == [30]
    assert not hasattr(result, "best_f1")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _ = _write_data(tmp_path / "data")
    bad_code = CUSUM_CODE.replace(
        'json.dump({"changepoints": [index]}, fh)',
        'json.dump({"changepoints": [0]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_changepoint_solution(
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


def test_run_changepoint_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _write_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_changepoint_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_changepoint() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_changepoint_search",
        "apply_changepoint_solution",
    ]
    assert "f1" in declaration["verified_metrics"]
    assert declaration["data_contract"]["tolerance_window"] == {
        "default": 3,
        "customizable": True,
    }

"""R21: survival API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.survival import (
    ApplySurvivalResult,
    SurvivalSearchResult,
    apply_survival_solution,
    capabilities,
    run_survival_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "survival"
RISK_CODE = (FIXTURES / "cox_linear_risk.py").read_text(encoding="utf-8")


def _make_data(root: Path, *, seed: int = 23) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    n = 80
    x = rng.normal(size=(n, 2))
    times = 15.0 + 8.0 * np.abs(x[:, 0]) + rng.exponential(
        scale=5.0, size=n
    )
    events = (rng.random(n) < 0.6).astype(int)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train = pd.DataFrame(x[:60], columns=["f0", "f1"])
    train["time"] = times[:60]
    train["event"] = events[:60]
    test = pd.DataFrame(x[60:], columns=["f0", "f1"])
    pd.DataFrame({"time": times[60:], "event": events[60:]}).to_csv(
        host / "hidden_test_outcomes.csv", index=False
    )
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    return public, host


def test_run_survival_search_risk_verified(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_survival_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, SurvivalSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.output_kind == "risk_score"
    assert result.best_c_index is not None and np.isfinite(result.best_c_index)
    assert result.best_mae is None
    assert result.rejected == 0


def test_run_survival_search_time_verified(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_survival_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        output_kind="time",
    )
    assert result.status == "verified"
    assert result.output_kind == "time"
    assert result.best_c_index is not None
    assert result.best_mae is not None and np.isfinite(result.best_mae)


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_survival_search(
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
    assert persisted["task"] == "survival"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["host"]["hidden_test_outcomes.csv"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    result = apply_survival_solution(
        RISK_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplySurvivalResult)
    assert result.status == "produced_unverified"
    assert result.predictions_path is not None
    payload = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    assert len(payload["predictions"]) == 20
    assert not hasattr(result, "best_c_index")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _host = _make_data(tmp_path / "data")
    bad_code = RISK_CODE.replace(
        'json.dump({"predictions": [float(value) for value in risk]}, fh)',
        'json.dump({"predictions": [float(value) for value in risk][:-1]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_survival_solution(
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


def test_run_survival_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_survival_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_survival() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_survival_search",
        "apply_survival_solution",
    ]
    assert declaration["data_contract"]["output_kinds"] == [
        "risk_score",
        "time",
    ]

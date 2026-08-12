"""R18: probabilistic API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.probabilistic import (
    ApplyProbabilisticResult,
    ProbabilisticSearchResult,
    apply_probabilistic_solution,
    capabilities,
    run_probabilistic_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "probabilistic"
MEAN_CODE = (FIXTURES / "mle_mean_variance.py").read_text(encoding="utf-8")


def _make_data(
    root: Path,
    *,
    family: str = "normal",
    quantity: str = "mean",
    params: dict | None = None,
    q: float | None = None,
    threshold: float | None = None,
    n: int = 300,
    seed: int = 23,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    if family == "normal":
        params = params or {"mean": 3.0, "std": 2.0}
        samples = rng.normal(params["mean"], params["std"], size=n)
    elif family == "gamma":
        params = params or {"shape": 2.0, "scale": 3.0}
        samples = rng.gamma(params["shape"], scale=params["scale"], size=n)
    else:
        params = params or {"alpha": 2.0, "beta": 5.0}
        samples = rng.beta(params["alpha"], params["beta"], size=n)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    problem: dict = {
        "version": 1,
        "family": family,
        "quantity": quantity,
    }
    if q is not None:
        problem["q"] = q
    if threshold is not None:
        problem["threshold"] = threshold
    (public / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    pd.DataFrame({"value": samples}).to_csv(
        public / "train.csv", index=False
    )
    (host / "hidden_parameters.json").write_text(
        json.dumps(params), encoding="utf-8"
    )
    return public, host


def test_run_probabilistic_search_mean_verified(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data", quantity="mean")
    result = run_probabilistic_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, ProbabilisticSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.family == "normal"
    assert result.quantity == "mean"
    assert result.best_relative_error is not None
    assert result.best_relative_error < 0.1
    assert result.best_ci_coverage == 1.0
    assert result.rejected == 0


def test_run_probabilistic_search_quantile_verified(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data", quantity="quantile", q=0.9)
    result = run_probabilistic_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    assert result.quantity == "quantile"
    assert result.best_relative_error is not None
    assert result.best_relative_error < 0.1


def test_run_probabilistic_search_gamma_probability_verified(
    tmp_path: Path,
) -> None:
    public, host = _make_data(
        tmp_path / "data",
        family="gamma",
        quantity="probability_ge",
        threshold=5.0,
    )
    result = run_probabilistic_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    assert result.family == "gamma"
    assert result.quantity == "probability_ge"
    assert result.best_relative_error is not None
    assert result.best_relative_error < 0.2


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data", quantity="mean")
    result = run_probabilistic_search(
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
    assert persisted["task"] == "probabilistic"
    assert "reference" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["host"]["hidden_parameters.json"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data", quantity="mean")
    result = apply_probabilistic_solution(
        MEAN_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyProbabilisticResult)
    assert result.status == "produced_unverified"
    assert result.solutions_path is not None
    payload = json.loads(result.solutions_path.read_text(encoding="utf-8"))
    assert "estimate" in payload
    assert not hasattr(result, "best_absolute_error")
    summary = result.to_summary()
    json.dumps(summary)
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _host = _make_data(tmp_path / "data", quantity="mean")
    bad_code = MEAN_CODE.replace(
        '"estimate": estimate,',
        '"estimate": estimate + float("nan"),',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_probabilistic_solution(
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


def test_run_probabilistic_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data", quantity="mean")
    with pytest.raises(ValueError, match="unknown generator"):
        run_probabilistic_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_probabilistic() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_probabilistic_search",
        "apply_probabilistic_solution",
    ]
    assert declaration["data_contract"]["families"] == [
        "normal",
        "gamma",
        "beta",
    ]

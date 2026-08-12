"""R23: markov API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.markov import (
    ApplyMarkovResult,
    MarkovSearchResult,
    apply_markov_solution,
    capabilities,
    run_markov_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "markov"
FREQ_CODE = (FIXTURES / "markov_frequency.py").read_text(encoding="utf-8")


def _make_data(root: Path, *, seed: int = 23) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    p = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=float)
    current = 0
    rows = []
    for _ in range(300):
        rows.append({"state": ["a", "b"][current]})
        current = int(rng.choice(2, p=p[current]))
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    (public / "problem.json").write_text(
        json.dumps(
            {
                "version": 1,
                "quantity": "transition_probability",
                "states": ["a", "b"],
                "from_state": "a",
                "to_state": "b",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(public / "train.csv", index=False)
    (host / "hidden_parameters.json").write_text(
        json.dumps({"transition_matrix": p.tolist()}), encoding="utf-8"
    )
    return public, host


def test_run_markov_search_transition_verified(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_markov_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, MarkovSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.quantity == "transition_probability"
    assert result.best_relative_error is not None
    assert result.best_relative_error < 0.1
    assert result.best_ci_coverage in (0.0, 1.0)
    assert result.rejected == 0


def test_run_markov_search_steady_verified(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    problem_path = public / "problem.json"
    problem = json.loads(problem_path.read_text())
    problem["quantity"] = "steady_state"
    problem.pop("from_state")
    problem.pop("to_state")
    problem["state"] = "a"
    problem_path.write_text(json.dumps(problem))
    result = run_markov_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    assert result.quantity == "steady_state"
    assert result.best_relative_error is not None
    assert result.best_relative_error < 0.1


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_markov_search(
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
    assert persisted["task"] == "markov"
    assert "reference" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["host"]["hidden_parameters.json"],
    )


def test_apply_produced_unverified(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    result = apply_markov_solution(
        FREQ_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyMarkovResult)
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
    public, _host = _make_data(tmp_path / "data")
    bad_code = FREQ_CODE.replace(
        '"estimate": estimate,',
        '"estimate": estimate + float("nan"),',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_markov_solution(
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


def test_run_markov_search_rejects_unknown_generator(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_markov_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_markov() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_markov_search",
        "apply_markov_solution",
    ]
    assert declaration["data_contract"]["quantities"] == [
        "transition_probability",
        "steady_state",
        "expected_recurrence_time",
    ]

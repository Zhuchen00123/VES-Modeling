"""R20: association API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.association import (
    ApplyAssociationResult,
    AssociationSearchResult,
    apply_association_solution,
    capabilities,
    run_association_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "association"
APRIORI_CODE = (FIXTURES / "apriori_rules.py").read_text(encoding="utf-8")


def _make_data(root: Path, *, seed: int = 23) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    items = ["a", "b", "c", "d", "e", "f"]
    train_rows: list[dict] = []
    hidden_rows: list[dict] = []
    for tid in range(1, 15):
        chosen = rng.choice(items, size=3, replace=False)
        for item in chosen:
            train_rows.append({"transaction_id": tid, "item": item})
    for tid in range(1, 10):
        chosen = rng.choice(items, size=2, replace=False)
        for item in chosen:
            hidden_rows.append({"transaction_id": tid, "item": item})
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    pd.DataFrame(train_rows).to_csv(public / "train.csv", index=False)
    pd.DataFrame(hidden_rows).to_csv(
        host / "hidden_test_transactions.csv", index=False
    )
    return public, host


def test_run_association_search_verified(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_association_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, AssociationSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_mean_lift is not None and np.isfinite(
        result.best_mean_lift
    )
    assert result.best_mean_confidence is not None
    assert result.best_evaluable_rule_count is not None
    assert result.best_evaluable_rule_count >= 1.0
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_association_search(
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
    assert persisted["task"] == "association"
    assert "hidden" not in json.dumps(persisted)
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        provenance["inputs"]["host"]["hidden_test_transactions.csv"],
    )


def test_apply_produced_unverified_no_metrics(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    result = apply_association_solution(
        APRIORI_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyAssociationResult)
    assert result.status == "produced_unverified"
    assert result.rules_path is not None
    payload = json.loads(result.rules_path.read_text(encoding="utf-8"))
    assert len(payload["rules"]) >= 1
    assert not hasattr(result, "best_mean_lift")
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
    bad_code = APRIORI_CODE.replace(
        'json.dump({"rules": rules}, fh)',
        'json.dump({"rules": []}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_association_solution(
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


def test_run_association_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_association_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_association() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_association_search",
        "apply_association_solution",
    ]
    assert "mean_lift" in declaration["verified_metrics"]
    assert declaration["data_contract"]["lift_cap"]["default"] == "1e6"

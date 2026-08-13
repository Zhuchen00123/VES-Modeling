"""VES Modeling — real-world regression closed loop (T-034).

Constructs a public regression dataset (scikit-learn bundled / downloaded
data, CSV copies under /tmp/realworld-data), runs
``run_regression_search`` (mock or real LLM + Docker sandbox), applies the
best verified solution with ``apply_regression_solution`` and re-verifies the
apply artifact with the same host verifier to confirm the metrics agree.

Usage:
  python examples/regression_realworld.py --generator mock
  python examples/regression_realworld.py --generator llm --drafts 3 --improves 3

The LLM path reads ``VES_MODELING_LLM_BASE_URL`` / ``VES_MODELING_LLM_API_KEY``
/ ``VES_MODELING_LLM_MODEL`` and runs candidates in the Docker sandbox
(``--network none``, ``--read-only``, hidden labels never mounted).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from ves.artifact import SafeArtifactLoader

from ves_modeling.regression import (
    apply_regression_solution,
    run_regression_search,
)
from ves_modeling.regression.context import RegressionVerificationContext
from ves_modeling.regression.data_contract import (
    load_host_labels,
    validate_regression_data,
)
from ves_modeling.regression.verifier import RegressionVerifier

logger = logging.getLogger(__name__)

DATASETS = ("california", "diabetes")
TARGET_COLUMN = "target"
SEED = 42
TEST_FRACTION = 0.2
DEFAULT_DATA_ROOT = Path("/tmp/realworld-data")
REPORT_SCHEMA = "realworld-regression-1.0"


def _frame_for_dataset(
    dataset: str, data_root: Path
) -> pd.DataFrame:
    """Return a features + target frame for a public dataset."""
    if dataset == "california":
        bundle = fetch_california_housing(
            data_home=str(data_root / "sklearn-cache"),
            download_if_missing=True,
        )
        frame = pd.DataFrame(bundle.data, columns=list(bundle.feature_names))
        frame[TARGET_COLUMN] = bundle.target
        return frame
    if dataset == "diabetes":
        bundle = load_diabetes()
        frame = pd.DataFrame(bundle.data, columns=list(bundle.feature_names))
        frame[TARGET_COLUMN] = bundle.target
        return frame
    raise ValueError(f"unknown regression dataset: {dataset!r}")


def ensure_data(
    data_root: Path, dataset: str, *, force: bool = False
) -> tuple[Path, Path, dict[str, Any]]:
    """Prepare deterministic train/test/host CSVs under ``data_root``.

    Returns ``(public_dir, host_dir, manifest)``.  Host labels are written
    only under ``host/`` and are never part of the candidate-visible
    ``public/`` directory.
    """
    data_dir = data_root / "regression" / dataset
    public_dir = data_dir / "public"
    host_dir = data_dir / "host"
    manifest_path = data_dir / "manifest.json"
    required = (
        public_dir / "train.csv",
        public_dir / "test_features.csv",
        host_dir / "hidden_test_labels.csv",
        manifest_path,
    )
    if not force and all(path.is_file() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return public_dir, host_dir, manifest

    frame = _frame_for_dataset(dataset, data_root)
    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    train, test = train_test_split(
        frame,
        test_size=TEST_FRACTION,
        random_state=SEED,
        shuffle=True,
    )
    public_dir.mkdir(parents=True, exist_ok=True)
    host_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(public_dir / "train.csv", index=False)
    test[feature_columns].to_csv(public_dir / "test_features.csv", index=False)
    test[[TARGET_COLUMN]].to_csv(host_dir / "hidden_test_labels.csv", index=False)

    manifest = {
        "schema_version": REPORT_SCHEMA,
        "dataset": dataset,
        "source": {
            "california": "sklearn.datasets.fetch_california_housing",
            "diabetes": "sklearn.datasets.load_diabetes",
        }[dataset],
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "train_rows": len(train),
        "test_rows": len(test),
        "data_dir": str(data_dir),
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return public_dir, host_dir, manifest


def _observation(evidence: Any, name: str) -> float:
    for item in evidence:
        if item.name == name:
            return float(item.value)
    raise ValueError(f"missing observation {name}")


def _load_payload(path: Path) -> dict[str, Any]:
    artifact = SafeArtifactLoader(root=path.parent).load(path.name)
    return json.loads(artifact.content)


def _search_best_payload(
    run_dir: Path, best_candidate_id: str | None
) -> dict[str, Any] | None:
    """Best-effort locate the search artifact of the best candidate.

    The best candidate id is not necessarily the attempt directory name, so
    match attempt run.json entries by the sha256 of ``best_solution.py``.
    """
    del best_candidate_id
    best_path = run_dir / "best_solution.py"
    if not best_path.is_file():
        return None
    best_sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    candidates_dir = run_dir / "candidates"
    if not candidates_dir.is_dir():
        return None
    for attempt_dir in sorted(candidates_dir.iterdir()):
        run_json = attempt_dir / "run.json"
        if not run_json.is_file():
            continue
        record = json.loads(run_json.read_text(encoding="utf-8"))
        if record.get("code_sha256") != best_sha:
            continue
        for path in (
            attempt_dir / "output" / "predictions.json",
            attempt_dir / "predictions.json",
        ):
            if path.is_file():
                return _load_payload(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator", choices=("mock", "llm"), default="mock"
    )
    parser.add_argument("--dataset", choices=DATASETS, default="california")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--drafts", type=int, default=3)
    parser.add_argument("--improves", type=int, default=3)
    parser.add_argument("--image", default="ves-modeling-runner:0.1")
    parser.add_argument("--force", action="store_true", help="regenerate data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    root = args.root.resolve()
    runs_dir = (args.runs_dir or root / "runs" / "realworld" / "regression").resolve()
    public_dir, host_dir, manifest = ensure_data(
        args.data_dir.resolve(), args.dataset, force=args.force
    )

    split_metadata = {
        "source": manifest["source"],
        "seed": manifest["seed"],
        "test_fraction": manifest["test_fraction"],
        "train_rows": manifest["train_rows"],
        "test_rows": manifest["test_rows"],
    }
    dataset_name = f"realworld_{args.dataset}"
    result = run_regression_search(
        public_dir,
        host_dir,
        drafts=args.drafts,
        improves=args.improves,
        workspace=runs_dir / "search",
        generator=args.generator,
        fixture_dir=root / "fixtures" / "candidates",
        image=args.image,
        dataset_name=dataset_name,
        split_metadata=split_metadata,
        target_column=TARGET_COLUMN,
    )
    search_payload = {
        "run_id": result.run_id,
        "status": result.status,
        "drafts": args.drafts,
        "improves": args.improves,
        "best_candidate_id": result.best_candidate_id,
        "best_rmse": result.best_rmse,
        "best_mae": result.best_mae,
        "rejected": result.rejected,
    }
    if result.best_code is None:
        report = {
            "schema_version": REPORT_SCHEMA,
            "task": "regression",
            "generator": args.generator,
            "dataset": manifest,
            "search": search_payload,
            "apply": None,
            "consistency": None,
        }
        (runs_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise SystemExit(f"no verified solution for {args.dataset}; see report")

    applied = apply_regression_solution(
        result.best_code,
        public_dir,
        workspace=runs_dir / "apply",
        image=args.image,
        target_column=TARGET_COLUMN,
    )
    contract = validate_regression_data(public_dir, target_column=TARGET_COLUMN)
    labels = load_host_labels(host_dir, contract)
    context = RegressionVerificationContext(
        labels,
        dataset_name=dataset_name,
        expected_count=contract.test_rows,
        row_order="input",
    )
    apply_payload_path = applied.predictions_path
    if apply_payload_path is None:
        raise RuntimeError(f"apply did not produce predictions: {applied.status}")
    apply_artifact = SafeArtifactLoader(root=apply_payload_path.parent).load(
        apply_payload_path.name
    )
    evidence = RegressionVerifier().verify(apply_artifact, context)
    apply_rmse = _observation(evidence, "rmse")
    apply_mae = _observation(evidence, "mae")

    best_payload = _search_best_payload(result.run_dir, result.best_candidate_id)
    predictions_identical = False
    predictions_close = False
    if best_payload is not None:
        apply_values = json.loads(
            apply_payload_path.read_text(encoding="utf-8")
        )["predictions"]
        predictions_identical = best_payload["predictions"] == apply_values
        predictions_close = bool(
            np.allclose(
                np.asarray(apply_values, dtype=np.float64),
                np.asarray(best_payload["predictions"], dtype=np.float64),
                rtol=1e-9,
                atol=1e-12,
            )
        )

    consistency = {
        "apply_rmse": apply_rmse,
        "apply_mae": apply_mae,
        "search_best_rmse": result.best_rmse,
        "search_best_mae": result.best_mae,
        "rmse_abs_diff": abs(apply_rmse - (result.best_rmse or float("nan"))),
        "mae_abs_diff": abs(apply_mae - (result.best_mae or float("nan"))),
        "predictions_identical_to_search_best": predictions_identical,
        "predictions_close_to_search_best": predictions_close,
        "matches": bool(
            applied.status == "produced_unverified"
            and abs(apply_rmse - (result.best_rmse or float("inf"))) <= 1e-6
            and abs(apply_mae - (result.best_mae or float("inf"))) <= 1e-6
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "task": "regression",
        "generator": args.generator,
        "dataset": manifest,
        "search": search_payload,
        "apply": {
            "run_id": applied.run_id,
            "status": applied.status,
            "runner": applied.runner,
            "docker_image": applied.docker_image,
            "docker_digest": applied.docker_digest,
            "predictions_sha256": applied.predictions_sha256,
        },
        "consistency": consistency,
    }
    (runs_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("VES Modeling — real-world regression closed loop")
    print(f"dataset: {args.dataset} ({manifest['source']})")
    print(f"rows: train={manifest['train_rows']} test={manifest['test_rows']}")
    print(f"search: {args.generator} {args.drafts}d+{args.improves}i "
          f"status={result.status}")
    print(f"best verified: rmse={result.best_rmse:.6f} mae={result.best_mae:.6f} "
          f"rejected={result.rejected}")
    print(f"apply: status={applied.status} runner={applied.runner}")
    print(f"apply host-verified: rmse={apply_rmse:.6f} mae={apply_mae:.6f}")
    print(f"consistency: matches={consistency['matches']} "
          f"identical_predictions={predictions_identical} "
          f"close_predictions={predictions_close}")
    print(f"report: {runs_dir / 'report.json'}")


if __name__ == "__main__":
    main()

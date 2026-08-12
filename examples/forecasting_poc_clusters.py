"""VES Modeling — forecasting method-cluster POC (mock default, LLM optional).

Runs the forecasting POC benchmark (data/forecasting-poc) in three method
families (statistical / ml / mechanistic) with a fixed per-family budget,
then reports per-family bests, the global best under the same judge
ordering, a per-series winner table, and a same-total-budget single-space
baseline.

Usage:
  python examples/forecasting_poc_clusters.py
  python examples/forecasting_poc_clusters.py --generator llm
  python examples/forecasting_poc_clusters.py --drafts-per-family 3

The LLM path reads VES_MODELING_LLM_BASE_URL / VES_MODELING_LLM_API_KEY /
VES_MODELING_LLM_MODEL and runs candidates in the Docker sandbox.
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

from ves_modeling.forecasting import run_forecasting_search

logging.basicConfig(level=logging.WARNING)

FAMILIES = ("statistical", "ml", "mechanistic")
DEFAULT_DATA_DIR = "data/forecasting-poc"
OUTPUT_DIR = "runs/forecasting-poc-clusters"


def _family_seed(family: str) -> int:
    # 为报告标识，非控制性随机种子：mock 候选确定性、LLM 模式不控制采样。
    return int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:8], 16)


def _normalize_data(data_dir: Path) -> None:
    """Canonicalize POC CSVs (timestamp, series_id[, target]) in place.

    The provided POC files carry the same content but in a different column
    order and under the host filename ``hidden_test_values.csv``; the
    forecasting contract requires ``hidden_test_labels.csv`` with matching
    column order.  This host-side preparation rewrites the files so the
    existing public API can consume them unchanged.
    """
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test_features.csv"
    hidden_path = data_dir / "hidden_test_values.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    hidden = pd.read_csv(hidden_path)
    train = train[["timestamp", "series_id", "target"]]
    test = test[["timestamp", "series_id"]]
    hidden = hidden[["timestamp", "series_id", "target"]]
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    hidden.to_csv(data_dir / "hidden_test_labels.csv", index=False)


def _best_predictions_path(
    run_dir: Path,
    best_candidate_id: str | None,
    best_rmse: float | None,
) -> Path | None:
    """Locate the best candidate's predictions.json in the run tree."""
    candidates = run_dir / "candidates"
    if not candidates.is_dir():
        return None
    candidate_dirs = sorted(candidates.iterdir())
    if best_candidate_id:
        for candidate_dir in candidate_dirs:
            if candidate_dir.name != best_candidate_id:
                continue
            for name in ("predictions.json", "output/predictions.json"):
                candidate_path = candidate_dir / name
                if candidate_path.is_file():
                    return candidate_path
    if best_rmse is not None:
        for candidate_dir in candidate_dirs:
            run_json = candidate_dir / "run.json"
            if not run_json.is_file():
                continue
            try:
                evidence = json.loads(
                    run_json.read_text(encoding="utf-8")
                )["evidence"]
                if abs(float(evidence["rmse"]) - best_rmse) < 1e-9:
                    candidate_path = candidate_dir / "predictions.json"
                    if candidate_path.is_file():
                        return candidate_path
            except (KeyError, TypeError, ValueError):
                continue
    for candidate_dir in candidate_dirs:
        candidate_path = candidate_dir / "predictions.json"
        if candidate_path.is_file():
            return candidate_path
    fallback = sorted(candidates.glob("*/predictions.json"))
    return fallback[0] if fallback else None


def _load_keyed_predictions(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], float] = {}
    for row in payload["predictions"]:
        series_id = str(row["series_id"])
        timestamp = str(row["timestamp"])
        result[(series_id, timestamp)] = float(row["prediction"])
    return result


def _per_series_metrics(
    predictions: dict[tuple[str, str], float],
    hidden: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Host-side per-series RMSE/MAE/SMAPE over hidden test rows."""
    metrics: dict[str, dict[str, float]] = {}
    for series_id, group in hidden.groupby("series_id", sort=False):
        errors: list[float] = []
        absolute_errors: list[float] = []
        smapes: list[float] = []
        for _, row in group.iterrows():
            prediction = predictions.get(
                (str(row["series_id"]), str(row["timestamp"]))
            )
            if prediction is None:
                continue
            actual = float(row["target"])
            error = prediction - actual
            errors.append(error * error)
            absolute_errors.append(abs(error))
            denominator = abs(actual) + abs(prediction)
            smapes.append(
                2.0 * abs(error) / denominator if denominator > 0 else 0.0
            )
        if errors and len(errors) == len(group):
            metrics[str(series_id)] = {
                "rmse": float(np.sqrt(np.mean(errors))),
                "mae": float(np.mean(absolute_errors)),
                "smape": float(np.mean(smapes)),
            }
    return metrics


def _run_search(
    *,
    public_dir: Path,
    host_dir: Path,
    fixture_dir: Path,
    workspace: Path,
    generator: str,
    dataset_name: str,
    drafts: int,
    improves: int,
    method_family: str | None,
    seed: int | None,
) -> Any:
    split_metadata: dict[str, Any] = {}
    if method_family is not None:
        split_metadata["method_family"] = method_family
    if seed is not None:
        split_metadata["poc_seed"] = seed
    return run_forecasting_search(
        public_dir,
        host_dir,
        drafts=drafts,
        improves=improves,
        workspace=workspace,
        generator=generator,
        dataset_name=dataset_name,
        fixture_dir=fixture_dir,
        method_family=method_family,
        split_metadata=split_metadata,
        frequency="MS",
        row_order="key",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="directory with train/test_features/hidden_test_values CSVs",
    )
    parser.add_argument("--generator", choices=("mock", "llm"), default="mock")
    parser.add_argument("--drafts-per-family", type=int, default=2)
    parser.add_argument("--improves-per-family", type=int, default=2)
    parser.add_argument(
        "--families",
        default=",".join(FAMILIES),
        help="comma-separated method families",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    data_dir = (args.data_dir or root / DEFAULT_DATA_DIR).resolve()
    public_dir = data_dir
    host_dir = data_dir
    for name in ("train.csv", "test_features.csv", "hidden_test_values.csv"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"missing POC data file: {data_dir / name}")
    _normalize_data(data_dir)
    fixture_dir = root / "fixtures" / "forecasting"
    output_dir = (root / OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    families = tuple(
        family.strip() for family in args.families.split(",") if family.strip()
    )
    if not families:
        raise ValueError("--families must be non-empty")

    hidden = pd.read_csv(host_dir / "hidden_test_values.csv")
    results: list[dict[str, Any]] = []
    for family in families:
        seed = _family_seed(family)
        workspace = output_dir / family
        result = _run_search(
            public_dir=public_dir,
            host_dir=host_dir,
            fixture_dir=fixture_dir,
            workspace=workspace,
            generator=args.generator,
            dataset_name=f"forecasting-poc-{family}",
            drafts=args.drafts_per_family,
            improves=args.improves_per_family,
            method_family=family,
            seed=seed,
        )
        summary = result.to_summary()
        best_path = _best_predictions_path(
            result.run_dir, result.best_candidate_id, result.best_rmse
        )
        per_series: dict[str, dict[str, float]] = {}
        if best_path is not None:
            per_series = _per_series_metrics(
                _load_keyed_predictions(best_path), hidden
            )
        results.append(
            {
                "family": family,
                "seed": seed,
                "status": result.status,
                "best_candidate_id": result.best_candidate_id,
                "best_rmse": result.best_rmse,
                "best_mae": result.best_mae,
                "best_smape": result.best_smape,
                "rejected": result.rejected,
                "drafts_used": args.drafts_per_family,
                "improves_used": args.improves_per_family,
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "per_series": per_series,
                "summary": summary,
            }
        )
        print(
            f"[{family}] status={result.status} "
            f"rmse={result.best_rmse} mae={result.best_mae} "
            f"smape={result.best_smape} rejected={result.rejected}"
        )

    verified = [
        entry
        for entry in results
        if entry["status"] == "verified" and entry["best_rmse"] is not None
    ]
    global_best: dict[str, Any] | None = None
    if verified:
        # POC 报告排序=Judge 顺序(rmse→mae)+smape 附加 tie-break。
        global_best = min(
            verified,
            key=lambda entry: (
                entry["best_rmse"],
                entry["best_mae"],
                entry["best_smape"],
            ),
        )
        print(
            f"[global] family={global_best['family']} "
            f"rmse={global_best['best_rmse']} mae={global_best['best_mae']} "
            f"smape={global_best['best_smape']}"
        )

    baseline: dict[str, Any] | None = None
    total_drafts = args.drafts_per_family * len(families)
    total_improves = args.improves_per_family * len(families)
    baseline_result = _run_search(
        public_dir=public_dir,
        host_dir=host_dir,
        fixture_dir=fixture_dir,
        workspace=output_dir / "baseline",
        generator=args.generator,
        dataset_name="forecasting-poc-baseline",
        drafts=total_drafts,
        improves=total_improves,
        method_family=None,
        seed=None,
    )
    baseline = {
        "status": baseline_result.status,
        "best_candidate_id": baseline_result.best_candidate_id,
        "best_rmse": baseline_result.best_rmse,
        "best_mae": baseline_result.best_mae,
        "best_smape": baseline_result.best_smape,
        "rejected": baseline_result.rejected,
        "drafts_used": total_drafts,
        "improves_used": total_improves,
        "run_id": baseline_result.run_id,
        "run_dir": str(baseline_result.run_dir),
    }
    print(
        f"[baseline] status={baseline_result.status} "
        f"rmse={baseline_result.best_rmse} mae={baseline_result.best_mae} "
        f"smape={baseline_result.best_smape} rejected={baseline_result.rejected}"
    )

    per_series_table: dict[str, dict[str, dict[str, float]]] = {}
    winner_by_series: dict[str, str] = {}
    for series in sorted(hidden["series_id"].astype(str).unique()):
        per_series_table[series] = {}
        for entry in results:
            if series in entry["per_series"]:
                per_series_table[series][entry["family"]] = entry["per_series"][
                    series
                ]
        candidates = per_series_table[series]
        if candidates:
            winner_by_series[series] = min(
                candidates, key=lambda family: candidates[family]["rmse"]
            )

    report = {
        "schema_version": "poc-1.0",
        "data_dir": str(data_dir),
        "generator": args.generator,
        "drafts_per_family": args.drafts_per_family,
        "improves_per_family": args.improves_per_family,
        "families": list(families),
        "output_dir": str(output_dir),
        "family_results": results,
        "global_best": global_best,
        "baseline": baseline,
        "per_series_table": per_series_table,
        "winner_by_series": winner_by_series,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("report:", report_path)


if __name__ == "__main__":
    main()

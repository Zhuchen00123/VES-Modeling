"""VES Modeling - forecasting method-cluster POC, three-arm experiment (poc-2.0).

Runs the forecasting POC benchmark (``data/forecasting-poc``) as a
reproducible three-arm comparison where every arm spends exactly 6 generator
calls per repeat:

- ``long-single``: one long search (3 drafts + 3 improves), no family prompt;
- ``short-single``: three independent short searches (1 draft + 1 improve)
  without family prompts, global best across the three runs;
- ``short-cluster``: one short search per method family (1 draft + 1 improve)
  with family prompts.

All metrics come from the host verifier (candidate claims are never used).

Usage::

  python examples/forecasting_poc_clusters.py
  python examples/forecasting_poc_clusters.py --arm long-single
  python examples/forecasting_poc_clusters.py --arm short-cluster --repeats 2
  python examples/forecasting_poc_clusters.py --generator llm

The LLM path reads ``VES_MODELING_LLM_BASE_URL`` / ``VES_MODELING_LLM_API_KEY``
/ ``VES_MODELING_LLM_MODEL`` and runs candidates in the Docker sandbox.
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
ARMS = ("long-single", "short-single", "short-cluster")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _family_seed(family: str) -> int:
    # 为报告标识，非控制性随机种子：mock 候选确定性、LLM 模式不控制采样。
    return int(hashlib.sha256(family.encode()).hexdigest()[:8], 16)


def _short_seed(index: int) -> int:
    # 同上：报告/运行标识，不控制 mock 确定性或 LLM 采样。
    return int(
        hashlib.sha256(f"short-single-{index}".encode())
        .hexdigest()[:8],
        16,
    )


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


def _candidate_artifact(candidate_dir: Path) -> Path | None:
    for name in ("predictions.json", "output/predictions.json"):
        path = candidate_dir / name
        if path.is_file():
            return path
    return None


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
    """Host-side per-series RMSE/MAE/SMAPE over hidden test rows.

    A series is only reported when predictions cover every hidden row for it
    (conservative ``len(errors) == len(group)`` gate); missing rows exclude
    the series rather than silently lowering the error.
    """
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


def _collect_candidates(
    *,
    run_dir: Path,
    family: str | None,
    fallback_sha256: str | None,
    generator: str,
    hidden: pd.DataFrame,
) -> tuple[list[dict[str, Any]], int]:
    """Read every candidate attempt from the run tree.

    ``is_fallback`` is True only in LLM mode when the candidate code is
    byte-identical to ``fixtures/forecasting/linear_forecast.py`` (the trusted
    fallback); mock candidates always report False.
    """
    candidates: list[dict[str, Any]] = []
    fallback_count = 0
    candidate_root = run_dir / "candidates"
    if not candidate_root.is_dir():
        return candidates, fallback_count
    for candidate_dir in sorted(candidate_root.iterdir()):
        run_json_path = candidate_dir / "run.json"
        if not run_json_path.is_file():
            continue
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        code_sha256_full = payload.get("code_sha256")
        code_sha256 = (
            code_sha256_full[:12] if code_sha256_full else None
        )
        is_fallback = bool(
            generator == "llm"
            and code_sha256_full
            and fallback_sha256
            and code_sha256_full == fallback_sha256
        )
        evidence = payload.get("evidence") or {}
        per_series: dict[str, dict[str, float]] = {}
        artifact = _candidate_artifact(candidate_dir)
        if artifact is not None:
            predictions = _load_keyed_predictions(artifact)
            per_series = _per_series_metrics(predictions, hidden)
        candidates.append(
            {
                "attempt": payload.get("candidate", candidate_dir.name),
                "family": family,
                "status": payload.get("status", "unknown"),
                "is_fallback": is_fallback,
                "code_sha256": code_sha256,
                "rmse": evidence.get("rmse"),
                "mae": evidence.get("mae"),
                "smape": evidence.get("smape"),
                "per_series": per_series,
            }
        )
        if is_fallback:
            fallback_count += 1
    return candidates, fallback_count


def _best_candidate(
    candidates: list[dict[str, Any]], best_candidate_id: str | None
) -> dict[str, Any] | None:
    for candidate in candidates:
        if best_candidate_id and candidate["attempt"] == best_candidate_id:
            return candidate
    verified = [
        candidate
        for candidate in candidates
        if candidate["status"] == "verified" and candidate["rmse"] is not None
    ]
    if verified:
        # POC 报告排序=Judge 顺序(rmse→mae)+smape 附加 tie-break。
        return min(
            verified,
            key=lambda candidate: (
                candidate["rmse"],
                candidate["mae"],
                candidate["smape"],
            ),
        )
    return None


def _collect_run(
    *,
    result: Any,
    label: str,
    family: str | None,
    seed: int | None,
    fallback_sha256: str | None,
    generator: str,
    hidden: pd.DataFrame,
) -> dict[str, Any]:
    candidates, fallback_count = _collect_candidates(
        run_dir=result.run_dir,
        family=family,
        fallback_sha256=fallback_sha256,
        generator=generator,
        hidden=hidden,
    )
    best = _best_candidate(candidates, result.best_candidate_id)
    return {
        "label": label,
        "family": family,
        "seed": seed,
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "status": result.status,
        "best_candidate_id": result.best_candidate_id,
        "best_rmse": result.best_rmse,
        "best_mae": result.best_mae,
        "best_smape": result.best_smape,
        "rejected": result.rejected,
        "fallback_count": fallback_count,
        "candidate_count": len(candidates),
        "best_is_fallback": bool(best and best["is_fallback"]),
        "candidates": candidates,
        "best": best,
    }


def _judge_key(run: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(run["best_rmse"]),
        float(run["best_mae"]),
        float(run["best_smape"]),
    )


def _arm_best(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    verified = [
        run
        for run in runs
        if run["status"] == "verified" and run["best_rmse"] is not None
    ]
    if not verified:
        return None
    # POC 报告排序=Judge 顺序(rmse→mae)+smape 附加 tie-break。
    return min(verified, key=_judge_key)


def _winner_by_series(
    runs: list[dict[str, Any]], hidden: pd.DataFrame
) -> dict[str, str]:
    """Per-series winner label: the run with the lowest host RMSE on that
    series among runs that fully cover it."""
    series_ids = sorted(hidden["series_id"].astype(str).unique())
    winners: dict[str, str] = {}
    for series in series_ids:
        contenders: list[tuple[float, str]] = []
        for run in runs:
            best = run.get("best")
            if not best:
                continue
            per_series = best.get("per_series") or {}
            if series in per_series:
                contenders.append(
                    (float(per_series[series]["rmse"]), run["label"])
                )
        if contenders:
            winners[series] = min(contenders)[1]
    return winners


def _run_repeat(
    *,
    arm: str,
    repeat: int,
    families: tuple[str, ...],
    public_dir: Path,
    host_dir: Path,
    fixture_dir: Path,
    output_dir: Path,
    generator: str,
    hidden: pd.DataFrame,
    fallback_sha256: str | None,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    workspace_root = output_dir / arm / f"repeat-{repeat}"
    if arm == "long-single":
        specs = [
            {
                "label": "long-single",
                "family": None,
                "seed": None,
                "drafts": 3,
                "improves": 3,
                "method_family": None,
            }
        ]
    elif arm == "short-single":
        specs = [
            {
                "label": f"short-single-{index}",
                "family": None,
                "seed": _short_seed(index),
                "drafts": 1,
                "improves": 1,
                "method_family": None,
            }
            for index in range(3)
        ]
    else:
        specs = [
            {
                "label": family,
                "family": family,
                "seed": _family_seed(family),
                "drafts": 1,
                "improves": 1,
                "method_family": family,
            }
            for family in families
        ]
    for spec in specs:
        workspace = workspace_root / spec["label"]
        result = _run_search(
            public_dir=public_dir,
            host_dir=host_dir,
            fixture_dir=fixture_dir,
            workspace=workspace,
            generator=generator,
            dataset_name=f"poc-{arm}-r{repeat}-{spec['label']}",
            drafts=spec["drafts"],
            improves=spec["improves"],
            method_family=spec["method_family"],
            seed=spec["seed"],
        )
        run = _collect_run(
            result=result,
            label=spec["label"],
            family=spec["family"],
            seed=spec["seed"],
            fallback_sha256=fallback_sha256,
            generator=generator,
            hidden=hidden,
        )
        runs.append(run)
        print(
            f"[{arm} r{repeat} {spec['label']}] status={run['status']} "
            f"rmse={run['best_rmse']} mae={run['best_mae']} "
            f"smape={run['best_smape']} rejected={run['rejected']} "
            f"fallback={run['fallback_count']}/{run['candidate_count']}"
        )
    arm_best = _arm_best(runs)
    if arm_best is not None:
        print(
            f"[{arm} r{repeat} best] {arm_best['label']} "
            f"rmse={arm_best['best_rmse']} mae={arm_best['best_mae']} "
            f"smape={arm_best['best_smape']}"
        )
    return {
        "repeat": repeat,
        "runs": runs,
        "arm_best": arm_best,
        "winner_by_series": _winner_by_series(runs, hidden),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else float("nan")


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
    parser.add_argument(
        "--arm",
        choices=ARMS,
        default="short-cluster",
        help="experiment arm (default: short-cluster, current behavior)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="independent repeats per arm (default: 1)",
    )
    parser.add_argument(
        "--families",
        default=",".join(FAMILIES),
        help="comma-separated method families",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

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
    if args.arm == "short-cluster" and len(families) != 3:
        raise ValueError(
            "--arm short-cluster requires exactly 3 families "
            f"(got {len(families)}) to keep the 6-call budget"
        )

    hidden = pd.read_csv(host_dir / "hidden_test_labels.csv")
    fallback_sha256: str | None = None
    if args.generator == "llm":
        fallback_path = fixture_dir / "linear_forecast.py"
        if not fallback_path.is_file():
            raise FileNotFoundError(f"fallback fixture missing: {fallback_path}")
        fallback_sha256 = _sha256_text(fallback_path.read_text(encoding="utf-8"))

    repeat_results: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        repeat_results.append(
            _run_repeat(
                arm=args.arm,
                repeat=repeat,
                families=families,
                public_dir=public_dir,
                host_dir=host_dir,
                fixture_dir=fixture_dir,
                output_dir=output_dir,
                generator=args.generator,
                hidden=hidden,
                fallback_sha256=fallback_sha256,
            )
        )

    arm_bests = [
        entry["arm_best"]
        for entry in repeat_results
        if entry["arm_best"] is not None
    ]
    arm_stats: dict[str, Any] = {
        "count": len(arm_bests),
        "mean_rmse": _mean([entry["best_rmse"] for entry in arm_bests]),
        "std_rmse": _std([entry["best_rmse"] for entry in arm_bests]),
        "mean_mae": _mean([entry["best_mae"] for entry in arm_bests]),
        "std_mae": _std([entry["best_mae"] for entry in arm_bests]),
        "mean_smape": _mean([entry["best_smape"] for entry in arm_bests]),
        "std_smape": _std([entry["best_smape"] for entry in arm_bests]),
    }
    total_calls_per_repeat = (
        6
        if args.arm in ("long-single", "short-single")
        else 2 * len(families)
    )
    report = {
        "schema_version": "poc-2.0",
        "arm": args.arm,
        "repeats": args.repeats,
        "generator": args.generator,
        "total_calls_per_repeat": total_calls_per_repeat,
        "families": list(families),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "repeat_results": repeat_results,
        "arm_stats": arm_stats,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("report:", report_path)


if __name__ == "__main__":
    main()

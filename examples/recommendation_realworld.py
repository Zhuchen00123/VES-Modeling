"""VES Modeling — real-world recommendation closed loop (T-034 batch 2).

Builds a deterministic synthetic user-item rating dataset (latent-factor
ratings, CSV copies under /tmp/realworld-data; hidden ratings only in host/),
runs ``run_recommendation_search`` (mock or real LLM + Docker sandbox),
applies the best verified solution with ``apply_recommendation_solution`` and
re-verifies the apply artifact with the same host verifier (RMSE/MAE +
NDCG@5 audit) to confirm the metrics agree.

Usage:
  python examples/recommendation_realworld.py --generator mock
  python examples/recommendation_realworld.py --generator llm --drafts 3 --improves 3

The LLM path reads ``VES_MODELING_LLM_BASE_URL`` / ``VES_MODELING_LLM_API_KEY``
/ ``VES_MODELING_LLM_MODEL`` and runs candidates in the Docker sandbox
(``--network none``, ``--read-only``, hidden ratings never mounted).
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
from ves.artifact import SafeArtifactLoader

from ves_modeling.recommendation import (
    apply_recommendation_solution,
    run_recommendation_search,
)
from ves_modeling.recommendation.context import (
    RecommendationVerificationContext,
)
from ves_modeling.recommendation.data_contract import (
    load_host_ratings,
    validate_recommendation_data,
)
from ves_modeling.recommendation.verifier import RecommendationVerifier

logger = logging.getLogger(__name__)

SEED = 42
N_USERS = 80
N_ITEMS = 120
N_FACTORS = 8
TRAIN_FRACTION = 0.8
TEST_PAIRS_PER_USER = 3
MIN_RATING = 1
MAX_RATING = 5
DEFAULT_DATA_ROOT = Path("/tmp/realworld-data")
REPORT_SCHEMA = "realworld-recommendation-1.0"
METRIC_FIELDS = {
    "rmse": "best_rmse",
    "mae": "best_mae",
    "ndcg@5": "best_ndcg",
}


def _rating_matrix(rng: np.random.RandomState) -> np.ndarray:
    """Deterministic latent-factor rating matrix (ints in [1, 5])."""
    users = rng.normal(0.0, 1.0, size=(N_USERS, N_FACTORS))
    items = rng.normal(0.0, 1.0, size=(N_ITEMS, N_FACTORS))
    noise = rng.normal(0.0, 0.6, size=(N_USERS, N_ITEMS))
    raw = 3.0 + users @ items.T + noise
    return np.clip(np.rint(raw), MIN_RATING, MAX_RATING).astype(np.int64)


def ensure_data(
    data_root: Path, *, force: bool = False
) -> tuple[Path, Path, dict[str, Any]]:
    """Prepare deterministic train/test/host rating CSVs under ``data_root``.

    Returns ``(public_dir, host_dir, manifest)``.  Hidden ratings are written
    only under ``host/`` and are never part of the candidate-visible
    ``public/`` directory.
    """
    data_dir = data_root / "recommendation" / "synthetic"
    public_dir = data_dir / "public"
    host_dir = data_dir / "host"
    manifest_path = data_dir / "manifest.json"
    required = (
        public_dir / "train.csv",
        public_dir / "test_features.csv",
        host_dir / "hidden_test_ratings.csv",
        manifest_path,
    )
    if not force and all(path.is_file() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return public_dir, host_dir, manifest

    rng = np.random.RandomState(SEED)
    ratings = _rating_matrix(rng)
    train_rows: list[tuple[int, int, int]] = []
    test_rows: list[tuple[int, int, int]] = []
    for user in range(1, N_USERS + 1):
        items = list(range(1, N_ITEMS + 1))
        rng.shuffle(items)
        train_count = int(N_ITEMS * TRAIN_FRACTION)
        for item in items[:train_count]:
            train_rows.append((user, item, int(ratings[user - 1, item - 1])))
        for item in items[
            train_count : train_count + TEST_PAIRS_PER_USER
        ]:
            test_rows.append((user, item, int(ratings[user - 1, item - 1])))

    train = pd.DataFrame(
        train_rows, columns=["user_id", "item_id", "rating"]
    )
    test = pd.DataFrame(
        [(user, item) for user, item, _rating in test_rows],
        columns=["user_id", "item_id"],
    )
    host = pd.DataFrame(
        test_rows, columns=["user_id", "item_id", "rating"]
    )
    public_dir.mkdir(parents=True, exist_ok=True)
    host_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(public_dir / "train.csv", index=False)
    test.to_csv(public_dir / "test_features.csv", index=False)
    host.to_csv(host_dir / "hidden_test_ratings.csv", index=False)

    manifest = {
        "schema_version": REPORT_SCHEMA,
        "dataset": "synthetic_ratings",
        "source": "synthetic latent-factor ratings (fixed seed)",
        "seed": SEED,
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "n_factors": N_FACTORS,
        "train_fraction": TRAIN_FRACTION,
        "test_pairs_per_user": TEST_PAIRS_PER_USER,
        "rating_range": [MIN_RATING, MAX_RATING],
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


def _search_best_payload(run_dir: Path) -> dict[str, Any] | None:
    """Best-effort locate the search artifact of the best candidate."""
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


def _keyed_predictions(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for row in payload["predictions"]:
        result[(str(row["user_id"]), str(row["item_id"]))] = float(
            row["prediction"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator", choices=("mock", "llm"), default="mock"
    )
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
    logging.basicConfig(
        level=logging.INFO if args.generator == "llm" else logging.WARNING,
        force=True,
    )

    root = args.root.resolve()
    runs_dir = (
        args.runs_dir or root / "runs" / "realworld" / "recommendation"
    ).resolve()
    public_dir, host_dir, manifest = ensure_data(
        args.data_dir.resolve(), force=args.force
    )
    dataset_name = "realworld_synthetic_ratings"
    split_metadata = {
        "source": manifest["source"],
        "seed": manifest["seed"],
        "n_users": manifest["n_users"],
        "n_items": manifest["n_items"],
        "n_factors": manifest["n_factors"],
        "train_rows": manifest["train_rows"],
        "test_rows": manifest["test_rows"],
    }
    result = run_recommendation_search(
        public_dir,
        host_dir,
        drafts=args.drafts,
        improves=args.improves,
        workspace=runs_dir / "search",
        generator=args.generator,
        fixture_dir=root / "fixtures" / "recommendation",
        image=args.image,
        dataset_name=dataset_name,
        split_metadata=split_metadata,
        row_order="key",
    )
    search_payload = {
        "run_id": result.run_id,
        "status": result.status,
        "drafts": args.drafts,
        "improves": args.improves,
        "best_candidate_id": result.best_candidate_id,
        "best_rmse": result.best_rmse,
        "best_mae": result.best_mae,
        "best_ndcg": result.best_ndcg,
        "rejected": result.rejected,
    }
    if result.best_code is None:
        report = {
            "schema_version": REPORT_SCHEMA,
            "task": "recommendation",
            "generator": args.generator,
            "dataset": manifest,
            "search": search_payload,
            "apply": None,
            "consistency": None,
        }
        (runs_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise SystemExit("no verified solution; see report")

    applied = apply_recommendation_solution(
        result.best_code,
        public_dir,
        workspace=runs_dir / "apply",
        image=args.image,
        row_order="key",
    )
    contract = validate_recommendation_data(
        public_dir, row_order="key"
    )
    ratings = load_host_ratings(host_dir, contract)
    context = RecommendationVerificationContext(
        ratings,
        dataset_name=dataset_name,
        expected_count=contract.test_rows,
        user_keys=tuple(key[0] for key in contract.test_keys),
        item_keys=tuple(key[1] for key in contract.test_keys),
        row_order="key",
    )
    apply_payload_path = applied.predictions_path
    if apply_payload_path is None:
        raise RuntimeError(f"apply did not produce predictions: {applied.status}")
    apply_artifact = SafeArtifactLoader(root=apply_payload_path.parent).load(
        apply_payload_path.name
    )
    evidence = RecommendationVerifier().verify(apply_artifact, context)
    apply_metrics = {
        name: _observation(evidence, name)
        for name in ("rmse", "mae", "ndcg@5")
    }

    best_payload = _search_best_payload(result.run_dir)
    predictions_identical = False
    predictions_close = False
    if best_payload is not None:
        apply_values = _keyed_predictions(
            json.loads(apply_payload_path.read_text(encoding="utf-8"))
        )
        best_values = _keyed_predictions(best_payload)
        predictions_identical = best_payload["predictions"] == json.loads(
            apply_payload_path.read_text(encoding="utf-8")
        )["predictions"]
        if apply_values and best_values and set(apply_values) == set(best_values):
            ordered_apply = [apply_values[key] for key in sorted(apply_values)]
            ordered_best = [best_values[key] for key in sorted(apply_values)]
            predictions_close = bool(
                np.allclose(
                    np.asarray(ordered_apply, dtype=np.float64),
                    np.asarray(ordered_best, dtype=np.float64),
                    rtol=1e-9,
                    atol=1e-12,
                )
            )
    diffs = {
        f"{name}_abs_diff": abs(
            apply_metrics[name]
            - (getattr(result, METRIC_FIELDS[name]) or float("nan"))
        )
        for name in ("rmse", "mae", "ndcg@5")
    }
    matches = bool(
        applied.status == "produced_unverified"
        and all(
            abs(
                apply_metrics[name]
                - (getattr(result, METRIC_FIELDS[name]) or float("inf"))
            )
            <= 1e-6
            for name in ("rmse", "mae", "ndcg@5")
        )
    )
    consistency = {
        "apply_metrics": apply_metrics,
        "search_best_metrics": {
            "rmse": result.best_rmse,
            "mae": result.best_mae,
            "ndcg@5": result.best_ndcg,
        },
        **diffs,
        "predictions_identical_to_search_best": predictions_identical,
        "predictions_close_to_search_best": predictions_close,
        "matches": matches,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "task": "recommendation",
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

    print("VES Modeling — real-world recommendation closed loop", flush=True)
    print(f"dataset: synthetic ratings ({N_USERS} users x {N_ITEMS} items, "
          f"factors={N_FACTORS})", flush=True)
    print(f"rows: train={manifest['train_rows']} test={manifest['test_rows']}",
          flush=True)
    print(f"search: {args.generator} {args.drafts}d+{args.improves}i "
          f"status={result.status}", flush=True)
    print(f"best verified: rmse={result.best_rmse:.6f} "
          f"mae={result.best_mae:.6f} ndcg@5={result.best_ndcg:.6f} "
          f"rejected={result.rejected}", flush=True)
    print(f"apply: status={applied.status} runner={applied.runner}", flush=True)
    print("apply host-verified metrics: "
          + ", ".join(f"{name}={value:.6f}" for name, value in apply_metrics.items()),
          flush=True)
    print(f"consistency: matches={matches} "
          f"identical_predictions={predictions_identical} "
          f"close_predictions={predictions_close}", flush=True)
    print(f"report: {runs_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

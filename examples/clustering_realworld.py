"""VES Modeling — real-world clustering closed loop (T-034 batch 2).

Builds a public unlabeled clustering dataset (scikit-learn bundled iris/wine,
CSV copies under /tmp/realworld-data; reference partition only in host/),
runs ``run_clustering_search`` (mock or real LLM + Docker sandbox), applies
the best verified solution with ``apply_clustering_solution`` and re-verifies
the apply artifact with the same host verifier (ARI/NMI/V-measure +
silhouette) to confirm the metrics agree.

Usage:
  python examples/clustering_realworld.py --generator mock
  python examples/clustering_realworld.py --generator llm --drafts 3 --improves 3

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
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from ves.artifact import SafeArtifactLoader

from ves_modeling.clustering import (
    apply_clustering_solution,
    run_clustering_search,
)
from ves_modeling.clustering.context import ClusteringVerificationContext
from ves_modeling.clustering.data_contract import (
    load_host_labels,
    validate_clustering_data,
)
from ves_modeling.clustering.verifier import ClusteringVerifier

logger = logging.getLogger(__name__)

DATASETS = ("iris", "wine")
LABEL_COLUMN = "label"
SEED = 42
TEST_FRACTION = 0.3
DEFAULT_DATA_ROOT = Path("/tmp/realworld-data")
REPORT_SCHEMA = "realworld-clustering-1.0"


def _frame_for_dataset(
    dataset: str,
) -> tuple[pd.DataFrame, np.ndarray, list[Any]]:
    """Return (feature frame, reference labels, label values)."""
    if dataset == "iris":
        bundle = load_iris()
        frame = pd.DataFrame(bundle.data, columns=list(bundle.feature_names))
        labels = np.asarray(bundle.target, dtype=np.int64)
        return frame, labels, [str(value) for value in bundle.target_names]
    if dataset == "wine":
        bundle = load_wine()
        frame = pd.DataFrame(bundle.data, columns=list(bundle.feature_names))
        labels = np.asarray(bundle.target, dtype=np.int64)
        return frame, labels, [str(value) for value in bundle.target_names]
    raise ValueError(f"unknown clustering dataset: {dataset!r}")


def ensure_data(
    data_root: Path, dataset: str, *, force: bool = False
) -> tuple[Path, Path, dict[str, Any]]:
    """Prepare deterministic unlabeled train/test + host reference CSVs.

    Returns ``(public_dir, host_dir, manifest)``.  Reference labels are
    written only under ``host/`` and are never part of the candidate-visible
    ``public/`` directory.
    """
    data_dir = data_root / "clustering" / dataset
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

    frame, labels, label_values = _frame_for_dataset(dataset)
    feature_columns = list(frame.columns)
    train, test, _train_labels, test_labels = train_test_split(
        frame,
        labels,
        test_size=TEST_FRACTION,
        random_state=SEED,
        shuffle=True,
        stratify=labels,
    )
    public_dir.mkdir(parents=True, exist_ok=True)
    host_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(public_dir / "train.csv", index=False)
    test.to_csv(public_dir / "test_features.csv", index=False)
    host_labels = pd.DataFrame(
        {LABEL_COLUMN: [label_values[int(index)] for index in test_labels]}
    )
    host_labels.to_csv(host_dir / "hidden_test_labels.csv", index=False)

    manifest = {
        "schema_version": REPORT_SCHEMA,
        "dataset": dataset,
        "source": {
            "iris": "sklearn.datasets.load_iris",
            "wine": "sklearn.datasets.load_wine",
        }[dataset],
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "label_column": LABEL_COLUMN,
        "classes": label_values,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator", choices=("mock", "llm"), default="mock"
    )
    parser.add_argument("--dataset", choices=DATASETS, default="iris")
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
        args.runs_dir or root / "runs" / "realworld" / "clustering"
    ).resolve()
    public_dir, host_dir, manifest = ensure_data(
        args.data_dir.resolve(), args.dataset, force=args.force
    )
    dataset_name = f"realworld_{args.dataset}"
    split_metadata = {
        "source": manifest["source"],
        "seed": manifest["seed"],
        "test_fraction": manifest["test_fraction"],
        "train_rows": manifest["train_rows"],
        "test_rows": manifest["test_rows"],
    }
    result = run_clustering_search(
        public_dir,
        host_dir,
        drafts=args.drafts,
        improves=args.improves,
        workspace=runs_dir / "search",
        generator=args.generator,
        fixture_dir=root / "fixtures" / "clustering",
        image=args.image,
        dataset_name=dataset_name,
        split_metadata=split_metadata,
        label_column=LABEL_COLUMN,
        row_order="input",
    )
    metric_names = (
        "best_ari",
        "best_nmi",
        "best_v_measure",
        "best_silhouette",
    )
    search_payload = {
        "run_id": result.run_id,
        "status": result.status,
        "drafts": args.drafts,
        "improves": args.improves,
        "best_candidate_id": result.best_candidate_id,
        **{name: getattr(result, name) for name in metric_names},
        "rejected": result.rejected,
    }
    if result.best_code is None:
        report = {
            "schema_version": REPORT_SCHEMA,
            "task": "clustering",
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

    applied = apply_clustering_solution(
        result.best_code,
        public_dir,
        workspace=runs_dir / "apply",
        image=args.image,
        label_column=LABEL_COLUMN,
        row_order="input",
    )
    contract = validate_clustering_data(
        public_dir, label_column=LABEL_COLUMN, row_order="input"
    )
    host_keys, _distinct = load_host_labels(host_dir, contract)
    test_features = pd.read_csv(public_dir / "test_features.csv").to_numpy(
        dtype=np.float64
    )
    context = ClusteringVerificationContext(
        host_keys,
        dataset_name=dataset_name,
        expected_count=contract.test_rows,
        row_order="input",
        test_features=test_features,
    )
    apply_payload_path = applied.predictions_path
    if apply_payload_path is None:
        raise RuntimeError(f"apply did not produce predictions: {applied.status}")
    apply_artifact = SafeArtifactLoader(root=apply_payload_path.parent).load(
        apply_payload_path.name
    )
    evidence = ClusteringVerifier().verify(apply_artifact, context)
    apply_metrics = {
        name: _observation(evidence, name)
        for name in ("ari", "nmi", "v_measure", "silhouette")
    }

    best_payload = _search_best_payload(result.run_dir)
    predictions_identical = False
    if best_payload is not None:
        apply_labels = json.loads(
            apply_payload_path.read_text(encoding="utf-8")
        )["labels"]
        predictions_identical = best_payload["labels"] == apply_labels
    diffs = {
        f"{name}_abs_diff": abs(
            apply_metrics[name] - (getattr(result, f"best_{name}") or float("nan"))
        )
        for name in apply_metrics
    }
    matches = bool(
        applied.status == "produced_unverified"
        and all(
            abs(apply_metrics[name] - (getattr(result, f"best_{name}") or float("inf")))
            <= 1e-6
            for name in apply_metrics
        )
    )
    consistency = {
        "apply_metrics": apply_metrics,
        "search_best_metrics": {
            name: getattr(result, f"best_{name}") for name in apply_metrics
        },
        **diffs,
        "predictions_identical_to_search_best": predictions_identical,
        "matches": matches,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "task": "clustering",
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

    print("VES Modeling — real-world clustering closed loop", flush=True)
    print(f"dataset: {args.dataset} ({manifest['source']})", flush=True)
    print(f"rows: train={manifest['train_rows']} test={manifest['test_rows']}",
          flush=True)
    print(f"search: {args.generator} {args.drafts}d+{args.improves}i "
          f"status={result.status}", flush=True)
    print(f"best verified: ari={result.best_ari:.6f} "
          f"nmi={result.best_nmi:.6f} v_measure={result.best_v_measure:.6f} "
          f"silhouette={result.best_silhouette:.6f} rejected={result.rejected}",
          flush=True)
    print(f"apply: status={applied.status} runner={applied.runner}", flush=True)
    print("apply host-verified metrics: "
          + ", ".join(f"{name}={value:.6f}" for name, value in apply_metrics.items()),
          flush=True)
    print(f"consistency: matches={matches} "
          f"identical_predictions={predictions_identical}", flush=True)
    print(f"report: {runs_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

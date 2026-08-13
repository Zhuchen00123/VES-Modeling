"""VES Modeling — real-world ODE closed loop (T-034 batch 2).

Builds a deterministic logistic-growth ODE instance (true solution computed
with ``scipy.integrate.solve_ivp``, CSV copies under /tmp/realworld-data),
runs ``run_ode_search`` (mock or real LLM + Docker sandbox), applies the best
verified solution with ``apply_ode_solution`` and re-verifies the apply
artifact with the same host verifier to confirm the metrics agree.

Usage:
  python examples/ode_realworld.py --generator mock
  python examples/ode_realworld.py --generator llm --drafts 3 --improves 3

The LLM path reads ``VES_MODELING_LLM_BASE_URL`` / ``VES_MODELING_LLM_API_KEY``
/ ``VES_MODELING_LLM_MODEL`` and runs candidates in the Docker sandbox
(``--network none``, ``--read-only``, hidden values never mounted).
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
from scipy.integrate import solve_ivp
from ves.artifact import SafeArtifactLoader

from ves_modeling.ode import apply_ode_solution, run_ode_search
from ves_modeling.ode.context import OdeVerificationContext
from ves_modeling.ode.data_contract import (
    load_host_values,
    validate_ode_data,
)
from ves_modeling.ode.verifier import OdeVerifier

logger = logging.getLogger(__name__)

TIME_COLUMN = "t"
VALUE_COLUMN = "y"
SEED = 42
DEFAULT_DATA_ROOT = Path("/tmp/realworld-data")
REPORT_SCHEMA = "realworld-ode-1.0"

# Logistic growth y' = r*y*(1 - y/K), integrated densely with a strict solver.
ODE_PARAMS = {"r": 0.9, "K": 1000.0, "y0": 10.0, "t_max": 24.0}
TRAIN_POINTS = 18


def _true_solution(t_values: np.ndarray) -> np.ndarray:
    """Numeric reference solution of the logistic ODE at given t values."""
    r = ODE_PARAMS["r"]
    k = ODE_PARAMS["K"]
    y0 = ODE_PARAMS["y0"]

    def fun(_t: float, y: np.ndarray) -> np.ndarray:
        value = float(y[0])
        return np.asarray([r * value * (1.0 - value / k)])

    result = solve_ivp(
        fun,
        (0.0, float(t_values[-1])),
        [y0],
        t_eval=t_values,
        rtol=1e-10,
        atol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"ODE integration failed: {result.message}")
    return np.asarray(result.y[0], dtype=np.float64)


def ensure_data(
    data_root: Path, *, force: bool = False
) -> tuple[Path, Path, dict[str, Any]]:
    """Prepare deterministic train/test/host CSVs under ``data_root``.

    Returns ``(public_dir, host_dir, manifest)``.  Host values are written
    only under ``host/`` and are never part of the candidate-visible
    ``public/`` directory.
    """
    data_dir = data_root / "ode" / "logistic_growth"
    public_dir = data_dir / "public"
    host_dir = data_dir / "host"
    manifest_path = data_dir / "manifest.json"
    required = (
        public_dir / "train.csv",
        public_dir / "test_features.csv",
        host_dir / "hidden_test_values.csv",
        manifest_path,
    )
    if not force and all(path.is_file() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return public_dir, host_dir, manifest

    t_values = np.arange(0.0, ODE_PARAMS["t_max"] + 1.0, 1.0)
    y_values = _true_solution(t_values)
    train = pd.DataFrame(
        {TIME_COLUMN: t_values[:TRAIN_POINTS], VALUE_COLUMN: y_values[:TRAIN_POINTS]}
    )
    test = pd.DataFrame({TIME_COLUMN: t_values[TRAIN_POINTS:]})
    host = pd.DataFrame(
        {
            TIME_COLUMN: t_values[TRAIN_POINTS:],
            VALUE_COLUMN: y_values[TRAIN_POINTS:],
        }
    )
    public_dir.mkdir(parents=True, exist_ok=True)
    host_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(public_dir / "train.csv", index=False)
    test.to_csv(public_dir / "test_features.csv", index=False)
    host.to_csv(host_dir / "hidden_test_values.csv", index=False)

    manifest = {
        "schema_version": REPORT_SCHEMA,
        "dataset": "logistic_growth",
        "source": "scipy.integrate.solve_ivp (logistic growth numeric solution)",
        "seed": SEED,
        "ode_params": ODE_PARAMS,
        "solver": {"rtol": 1e-10, "atol": 1e-12},
        "time_column": TIME_COLUMN,
        "value_column": VALUE_COLUMN,
        "row_order": "input",
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
    runs_dir = (args.runs_dir or root / "runs" / "realworld" / "ode").resolve()
    public_dir, host_dir, manifest = ensure_data(
        args.data_dir.resolve(), force=args.force
    )
    dataset_name = "realworld_logistic_growth"
    split_metadata = {
        "source": manifest["source"],
        "seed": manifest["seed"],
        "ode_params": manifest["ode_params"],
        "train_rows": manifest["train_rows"],
        "test_rows": manifest["test_rows"],
    }
    result = run_ode_search(
        public_dir,
        host_dir,
        drafts=args.drafts,
        improves=args.improves,
        workspace=runs_dir / "search",
        generator=args.generator,
        fixture_dir=root / "fixtures" / "ode",
        image=args.image,
        dataset_name=dataset_name,
        split_metadata=split_metadata,
        time_column=TIME_COLUMN,
        value_column=VALUE_COLUMN,
        row_order="input",
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
            "task": "ode",
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

    applied = apply_ode_solution(
        result.best_code,
        public_dir,
        workspace=runs_dir / "apply",
        image=args.image,
        time_column=TIME_COLUMN,
        value_column=VALUE_COLUMN,
        row_order="input",
    )
    contract = validate_ode_data(
        public_dir,
        time_column=TIME_COLUMN,
        value_column=VALUE_COLUMN,
        row_order="input",
    )
    values = load_host_values(host_dir, contract)
    context = OdeVerificationContext(
        values,
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
    evidence = OdeVerifier().verify(apply_artifact, context)
    apply_rmse = _observation(evidence, "rmse")
    apply_mae = _observation(evidence, "mae")

    best_payload = _search_best_payload(result.run_dir)
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
        "task": "ode",
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

    print("VES Modeling — real-world ODE closed loop", flush=True)
    print(f"dataset: logistic_growth r={ODE_PARAMS['r']} K={ODE_PARAMS['K']} "
          f"y0={ODE_PARAMS['y0']}", flush=True)
    print(f"rows: train={manifest['train_rows']} test={manifest['test_rows']}",
          flush=True)
    print(f"search: {args.generator} {args.drafts}d+{args.improves}i "
          f"status={result.status}", flush=True)
    print(f"best verified: rmse={result.best_rmse:.6f} mae={result.best_mae:.6f} "
          f"rejected={result.rejected}", flush=True)
    print(f"apply: status={applied.status} runner={applied.runner}", flush=True)
    print(f"apply host-verified: rmse={apply_rmse:.6f} mae={apply_mae:.6f}",
          flush=True)
    print(f"consistency: matches={consistency['matches']} "
          f"identical_predictions={predictions_identical} "
          f"close_predictions={predictions_close}", flush=True)
    print(f"report: {runs_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

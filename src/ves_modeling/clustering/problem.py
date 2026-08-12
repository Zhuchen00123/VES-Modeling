"""Clustering VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.clustering.context import ClusteringVerificationContext
from ves_modeling.clustering.data_contract import (
    load_host_labels,
    validate_clustering_data,
)
from ves_modeling.clustering.verifier import ClusteringVerifier

verifier = ClusteringVerifier()

SCALAR_METRICS = ("ari", "nmi", "v_measure", "silhouette")


def build_clustering_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "clustering",
    label_column: str = "label",
    id_column: str | None = None,
    row_order: str = "input",
) -> VerifiedProblem:
    """Assemble the clustering VerifiedProblem.

    Reference labels are loaded from ``host_dir``; ``host_dir`` must never
    be exposed to candidates.
    """
    data_contract = validate_clustering_data(
        public_dir,
        label_column=label_column,
        id_column=id_column,
        row_order=row_order,
    )
    host_labels, _distinct = load_host_labels(host_dir, data_contract)
    test_features = _load_test_features(public_dir, data_contract)

    def make_context() -> ClusteringVerificationContext:
        return ClusteringVerificationContext(
            host_labels,
            dataset_name=dataset_name,
            expected_count=len(host_labels),
            id_column=id_column,
            prediction_ids=(
                data_contract.test_ids if row_order == "id" else None
            ),
            row_order=row_order,
            test_features=test_features,
        )

    contract = ArtifactContract(
        filename="predictions.json",
        media_type="application/json",
        required_fields=(),
    )

    return VerifiedProblem(
        contract=contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=(
                ObjectiveSpec(
                    observation="ari", direction=Direction.MAXIMIZE
                ),
                ObjectiveSpec(
                    observation="nmi", direction=Direction.MAXIMIZE
                ),
            ),
            gates=tuple(
                Gate(
                    name=f"{name}_finite",
                    observation=name,
                    finite=True,
                )
                for name in SCALAR_METRICS
            ),
        ),
        name=f"clustering:{dataset_name}",
        problem_ref=(
            "ves_modeling.clustering.problem:build_clustering_problem"
        ),
        verifier_module="ves_modeling.clustering.problem",
        verifier_attr="verifier",
        context_factory_ref=(
            "ves_modeling.clustering.problem:context_factory"
        ),
    )


def _load_test_features(
    public_dir: Path, contract
) -> np.ndarray:
    test = pd.read_csv(Path(public_dir) / "test_features.csv")
    columns = list(contract.input_columns)
    return test[columns].to_numpy(dtype=np.float64)


def context_factory() -> ClusteringVerificationContext:
    """Module-level factory used by ``ves replay`` (requires env config).

    Env: ``VES_MODELING_HOST_DIR`` (required), ``VES_MODELING_DATASET``,
    ``VES_MODELING_LABEL_COLUMN``, ``VES_MODELING_ID_COLUMN``,
    ``VES_MODELING_ROW_ORDER``.  ``VES_MODELING_PUBLIC_DIR`` is required for
    id-mode replay and to attach public test features.
    """
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not host_dir:
        raise RuntimeError(
            "VES_MODELING_HOST_DIR must be set to replay a clustering record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "clustering")
    label_column = os.environ.get("VES_MODELING_LABEL_COLUMN", "label")
    id_column = os.environ.get("VES_MODELING_ID_COLUMN") or None
    row_order = os.environ.get("VES_MODELING_ROW_ORDER", "input")
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay a clustering "
            "record"
        )
    contract = validate_clustering_data(
        Path(public_dir),
        label_column=label_column,
        id_column=id_column,
        row_order=row_order,
    )
    host_labels, _distinct = load_host_labels(Path(host_dir), contract)
    test_features = _load_test_features(Path(public_dir), contract)
    return ClusteringVerificationContext(
        host_labels,
        dataset_name=dataset_name,
        expected_count=len(host_labels),
        id_column=id_column,
        prediction_ids=(
            contract.test_ids if row_order == "id" else None
        ),
        row_order=row_order,
        test_features=test_features,
    )

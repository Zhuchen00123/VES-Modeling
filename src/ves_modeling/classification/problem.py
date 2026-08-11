"""Classification VerifiedProblem assembly (contract + context + verifier + judge)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from ves.artifact import ArtifactContract
from ves.judge import Direction, Gate, JudgeSpec, ObjectiveSpec
from ves.problem import VerifiedProblem

from ves_modeling.classification.context import (
    ClassificationVerificationContext,
)
from ves_modeling.classification.data_contract import (
    load_host_labels,
    validate_classification_data,
)
from ves_modeling.classification.verifier import ClassificationVerifier

verifier = ClassificationVerifier()

SCALAR_METRICS = (
    "accuracy",
    "macro_f1",
    "log_loss",
    "auroc",
    "multiclass_brier",
    "calibration_ece",
)


def build_classification_problem(
    public_dir: Path,
    host_dir: Path,
    *,
    dataset_name: str = "classification",
    labels: np.ndarray | None = None,
    label_column: str = "target",
    id_column: str | None = None,
    row_order: str = "input",
    classes: list[Any] | tuple[Any, ...] | None = None,
) -> VerifiedProblem:
    """Assemble the classification VerifiedProblem.

    ``labels`` may be injected for tests (input mode only); otherwise loaded
    from ``host_dir`` and aligned to public test order.  ``host_dir`` must
    never be exposed to candidates.
    """
    data_contract = validate_classification_data(
        public_dir,
        label_column=label_column,
        id_column=id_column,
        row_order=row_order,
        classes=classes,
    )
    hidden: np.ndarray
    if labels is not None:
        if row_order == "id":
            raise ValueError(
                "labels injection is not supported for row_order='id': "
                "provide hidden_test_labels.csv so labels can be aligned "
                "by id"
            )
        hidden = np.asarray(labels, dtype=np.int64).reshape(-1)
        if hidden.size == 0:
            raise ValueError("hidden labels must be non-empty")
        if hidden.min() < 0 or hidden.max() >= data_contract.n_classes:
            raise ValueError(
                "hidden label indices must be within [0, n_classes)"
            )
        if set(hidden.tolist()) != set(range(data_contract.n_classes)):
            raise ValueError(
                "hidden labels must cover every declared class"
            )
    else:
        hidden, _class_keys = load_host_labels(host_dir, data_contract)
    if hidden.size != data_contract.test_rows:
        raise ValueError(
            f"hidden labels count {hidden.size} != test rows "
            f"{data_contract.test_rows}"
        )
    expected_count = int(hidden.size)

    def make_context() -> ClassificationVerificationContext:
        return ClassificationVerificationContext(
            hidden,
            dataset_name=dataset_name,
            expected_count=expected_count,
            classes=data_contract.classes,
            class_keys=data_contract.class_keys,
            id_column=id_column,
            prediction_ids=(
                data_contract.test_ids if row_order == "id" else None
            ),
            row_order=row_order,
        )

    contract = ArtifactContract(
        filename="predictions.json",
        media_type="application/json",
        required_fields=("predictions",),
    )

    return VerifiedProblem(
        contract=contract,
        context_factory=make_context,
        verifier=verifier,
        judge_spec=JudgeSpec(
            objectives=(
                ObjectiveSpec(
                    observation="macro_f1",
                    direction=Direction.MAXIMIZE,
                ),
                ObjectiveSpec(
                    observation="log_loss",
                    direction=Direction.MINIMIZE,
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
        name=f"classification:{dataset_name}",
        problem_ref=(
            "ves_modeling.classification.problem:build_classification_problem"
        ),
        verifier_module="ves_modeling.classification.problem",
        verifier_attr="verifier",
        context_factory_ref=(
            "ves_modeling.classification.problem:context_factory"
        ),
    )


def context_factory() -> ClassificationVerificationContext:
    """Module-level factory used by ``ves replay`` (requires env config).

    Env: ``VES_MODELING_HOST_DIR`` (required), ``VES_MODELING_DATASET``,
    ``VES_MODELING_LABEL_COLUMN``, ``VES_MODELING_ID_COLUMN``,
    ``VES_MODELING_ROW_ORDER``, ``VES_MODELING_CLASSES`` (JSON array,
    optional; otherwise derived from train first appearance).
    ``VES_MODELING_PUBLIC_DIR`` is required for id-mode replay.
    """
    host_dir = os.environ.get("VES_MODELING_HOST_DIR")
    if not host_dir:
        raise RuntimeError(
            "VES_MODELING_HOST_DIR must be set to replay a "
            "classification record"
        )
    dataset_name = os.environ.get("VES_MODELING_DATASET", "classification")
    label_column = os.environ.get("VES_MODELING_LABEL_COLUMN", "target")
    id_column = os.environ.get("VES_MODELING_ID_COLUMN") or None
    row_order = os.environ.get("VES_MODELING_ROW_ORDER", "input")
    classes_env = os.environ.get("VES_MODELING_CLASSES")
    classes = json.loads(classes_env) if classes_env else None
    public_dir = os.environ.get("VES_MODELING_PUBLIC_DIR")
    if row_order == "id" and not public_dir:
        raise RuntimeError(
            "VES_MODELING_PUBLIC_DIR must be set to replay "
            "row_order='id' records"
        )
    if public_dir:
        contract = validate_classification_data(
            Path(public_dir),
            label_column=label_column,
            id_column=id_column,
            row_order=row_order,
            classes=classes,
        )
        labels, _class_keys = load_host_labels(Path(host_dir), contract)
        return ClassificationVerificationContext(
            labels,
            dataset_name=dataset_name,
            expected_count=int(labels.size),
            classes=contract.classes,
            class_keys=contract.class_keys,
            id_column=id_column,
            prediction_ids=(
                contract.test_ids if row_order == "id" else None
            ),
            row_order=row_order,
        )
    # No public dir: derive classes from a raw host CSV (input mode replay).
    import pandas as pd

    frame = pd.read_csv(Path(host_dir) / "hidden_test_labels.csv")
    host_labels = frame[label_column]
    seen: list[Any] = []
    seen_keys: set[str] = set()
    for value in host_labels:
        from ves_modeling.classification.data_contract import _label_key

        key = _label_key(value)
        if key not in seen_keys:
            seen_keys.add(key)
            seen.append(value)
    classes_derived = tuple(seen)
    class_keys = tuple(
        _label_key(value) for value in classes_derived
    )
    key_to_index = {
        key: index for index, key in enumerate(class_keys)
    }
    indices = np.asarray(
        [key_to_index[_label_key(value)] for value in host_labels],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise ValueError("hidden labels must be non-empty")
    if set(indices.tolist()) != set(range(len(class_keys))):
        raise ValueError(
            "hidden labels must cover every declared class"
        )
    return ClassificationVerificationContext(
        indices,
        dataset_name=dataset_name,
        expected_count=int(indices.size),
        classes=classes_derived,
        class_keys=class_keys,
        id_column=None,
        prediction_ids=None,
        row_order="input",
    )

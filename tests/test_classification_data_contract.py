"""R9: classification data contract (classes/labels/probabilities/id)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ves.artifact import RawArtifact
from ves.problem import VerificationPipeline

from ves_modeling.classification.context import (
    ClassificationVerificationContext,
)
from ves_modeling.classification.data_contract import (
    _label_key,
    load_host_labels,
    validate_classification_data,
    validate_predictions,
)
from ves_modeling.classification.problem import build_classification_problem
from ves_modeling.classification.verifier import ClassificationVerifier


def _make_data(
    root: Path,
    *,
    n_classes: int = 3,
    class_values: list | None = None,
    train_per_class: int = 40,
    test_per_class: int = 10,
    id_col: str | None = None,
    host_order: str = "same",
    drop_host_class: int | None = None,
    seed: int = 7,
) -> tuple[Path, Path]:
    if class_values is None:
        class_values = list(range(n_classes))
    n_train = train_per_class * n_classes
    n_test = test_per_class * n_classes
    X, y = make_classification(
        n_samples=n_train + n_test,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, stratify=y, random_state=seed
    )
    feature_names = [f"f{i}" for i in range(4)]
    train_labels = [class_values[int(index)] for index in y_train]
    test_labels = [class_values[int(index)] for index in y_test]
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train = pd.DataFrame(X_train, columns=feature_names)
    train["target"] = train_labels
    # Sort train rows by the underlying class index so first appearance
    # matches ``class_values`` order (the host default when classes=None).
    train["_class_index"] = y_train
    train = train.sort_values("_class_index").drop(
        columns="_class_index"
    ).reset_index(drop=True)
    test = pd.DataFrame(X_test, columns=feature_names)
    host_frame = pd.DataFrame({"target": test_labels})
    if id_col:
        train_ids = np.arange(1, len(train) + 1)
        test_ids = np.arange(1, len(test) + 1)
        train[id_col] = train_ids
        test[id_col] = test_ids
        host_frame[id_col] = test_ids
        if host_order == "reversed":
            host_frame = host_frame.iloc[::-1].reset_index(drop=True)
    if drop_host_class is not None:
        dropped_value = class_values[drop_host_class]
        host_frame = host_frame[
            host_frame["target"] != dropped_value
        ].reset_index(drop=True)
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    host_frame.to_csv(host / "hidden_test_labels.csv", index=False)
    return public, host


def _artifact(payload: dict) -> RawArtifact:
    return RawArtifact(
        name="predictions.json",
        content=json.dumps(payload),
        producer="test",
    )


def _perfect_predictions(
    class_values: list,
    n_test_rows: int,
    *,
    start_index: int = 0,
) -> dict:
    rows = []
    for index in range(n_test_rows):
        label = class_values[(start_index + index) % len(class_values)]
        label_index = class_values.index(label)
        probabilities = [0.0] * len(class_values)
        probabilities[label_index] = 1.0
        rows.append({"label": label, "probabilities": probabilities})
    return {"predictions": rows}


def test_valid_contract_first_appearance_classes(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    contract = validate_classification_data(public)
    assert contract.n_classes == 3
    assert contract.classes == (0, 1, 2)
    assert contract.class_keys == ("n:0", "n:1", "n:2")
    assert contract.class_counts == (40, 40, 40)
    assert contract.feature_columns == ("f0", "f1", "f2", "f3")
    labels, class_keys = load_host_labels(host, contract)
    assert labels.shape == (30,)
    assert class_keys == contract.class_keys


def test_explicit_classes_must_be_list_or_tuple(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="list or tuple"):
        validate_classification_data(public, classes="012")
    with pytest.raises(ValueError, match="at least two"):
        validate_classification_data(public, classes=[0])
    with pytest.raises(ValueError, match="unique"):
        validate_classification_data(public, classes=[0, 0])


def test_explicit_reversed_classes_order_and_counts(tmp_path: Path) -> None:
    public, _host = _make_data(
        tmp_path / "data", n_classes=2, class_values=[1, 0]
    )
    contract = validate_classification_data(public, classes=[1, 0])
    assert contract.classes == (1, 0)
    assert contract.class_keys == ("n:1", "n:0")
    assert contract.class_counts[0] == contract.class_counts[1]
    payload = json.dumps(contract.to_dict())  # JSON-serializable
    assert '"classes": [1, 0]' in payload
    assert '"class_counts"' in payload


def test_declared_class_must_appear_in_train(tmp_path: Path) -> None:
    public, _host = _make_data(
        tmp_path / "data", n_classes=2, class_values=[0, 1]
    )
    with pytest.raises(ValueError, match="every declared class must appear"):
        validate_classification_data(public, classes=[0, 1, 2])


def test_train_label_outside_declared_classes_rejected(tmp_path: Path) -> None:
    public, _host = _make_data(
        tmp_path / "data", n_classes=2, class_values=[0, 1]
    )
    train = pd.read_csv(public / "train.csv")
    train.loc[0, "target"] = 99
    train.to_csv(public / "train.csv", index=False)
    with pytest.raises(ValueError, match="outside the declared classes"):
        validate_classification_data(public, classes=[0, 1])


def test_host_must_cover_all_classes(tmp_path: Path) -> None:
    public, host = _make_data(
        tmp_path / "data", n_classes=3, drop_host_class=2
    )
    contract = validate_classification_data(public)
    with pytest.raises(ValueError, match="cover every declared class"):
        load_host_labels(host, contract)


def test_label_key_rejects_bad_values() -> None:
    assert _label_key("a") == "s:a"
    assert _label_key(1) == _label_key(1.0) == "n:1"
    assert _label_key("1") == "s:1"
    assert _label_key(1) != _label_key("1")
    assert _label_key(1.5) == "n:1.5"
    for bad in (True, False, None, float("nan"), float("inf"), ""):
        with pytest.raises(ValueError, match="class labels"):
            _label_key(bad)


def test_numeric_and_string_classes_json_safe(tmp_path: Path) -> None:
    public, _host = _make_data(
        tmp_path / "data", n_classes=3, class_values=["a", "b", "c"]
    )
    contract = validate_classification_data(public)
    assert contract.classes == ("a", "b", "c")
    json.dumps(contract.to_dict())
    public2, _host2 = _make_data(
        tmp_path / "data2", n_classes=3, class_values=[10, 11, 12]
    )
    contract2 = validate_classification_data(public2)
    json.dumps(contract2.to_dict())


def test_id_mode_validation_and_alignment(tmp_path: Path) -> None:
    public, host = _make_data(
        tmp_path / "data", n_classes=3, id_col="id_col", host_order="reversed"
    )
    contract = validate_classification_data(
        public, id_column="id_col", row_order="id"
    )
    assert contract.test_ids is not None
    labels, _class_keys = load_host_labels(host, contract)
    assert labels.shape == (30,)
    # Reversed host order still aligns to public test id order.
    assert set(labels.tolist()) == {0, 1, 2}


def test_id_attacks_rejected(tmp_path: Path) -> None:
    public, _host = _make_data(
        tmp_path / "data", n_classes=3, id_col="id_col"
    )
    train = pd.read_csv(public / "train.csv")
    for bad in ("", float("nan")):
        frame = train.copy()
        frame["id_col"] = frame["id_col"].astype(object)
        frame.loc[0, "id_col"] = bad
        frame.to_csv(public / "train.csv", index=False)
        with pytest.raises(ValueError, match="id"):
            validate_classification_data(
                public, id_column="id_col", row_order="id"
            )


def test_host_labels_count_mismatch(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    frame = pd.read_csv(host / "hidden_test_labels.csv").head(29)
    frame.to_csv(host / "hidden_test_labels.csv", index=False)
    contract = validate_classification_data(public)
    with pytest.raises(ValueError, match="hidden labels count"):
        load_host_labels(host, contract)


def test_validate_predictions_input_mode(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    contract = validate_classification_data(public)
    good = _perfect_predictions([0, 1, 2], 30)
    indices, probabilities = validate_predictions(
        good,
        expected_count=contract.test_rows,
        n_classes=contract.n_classes,
        class_keys=contract.class_keys,
    )
    assert indices.shape == (30,)
    assert probabilities.shape == (30, 3)
    with pytest.raises(ValueError, match="prediction count"):
        validate_predictions(
            _perfect_predictions([0, 1, 2], 29),
            expected_count=contract.test_rows,
            n_classes=3,
            class_keys=contract.class_keys,
        )
    bad_shape = {
        "predictions": [
            {"label": 0, "probabilities": [1.0, 0.0]}
            for _ in range(30)
        ]
    }
    with pytest.raises(ValueError, match="n_classes"):
        validate_predictions(
            bad_shape,
            expected_count=30,
            n_classes=3,
            class_keys=contract.class_keys,
        )
    bad_sum = {
        "predictions": [
            {"label": 0, "probabilities": [0.9, 0.0, 0.0]}
            for _ in range(30)
        ]
    }
    with pytest.raises(ValueError, match="sum to 1"):
        validate_predictions(
            bad_sum,
            expected_count=30,
            n_classes=3,
            class_keys=contract.class_keys,
        )
    bad_argmax = {
        "predictions": [
            {"label": 1, "probabilities": [1.0, 0.0, 0.0]}
            for _ in range(30)
        ]
    }
    with pytest.raises(ValueError, match="argmax"):
        validate_predictions(
            bad_argmax,
            expected_count=30,
            n_classes=3,
            class_keys=contract.class_keys,
        )
    outside = {
        "predictions": [
            {"label": 99, "probabilities": [1.0, 0.0, 0.0]}
            for _ in range(30)
        ]
    }
    with pytest.raises(ValueError, match="outside the declared classes"):
        validate_predictions(
            outside,
            expected_count=30,
            n_classes=3,
            class_keys=contract.class_keys,
        )


def test_validate_predictions_id_mode_attacks(tmp_path: Path) -> None:
    public, _host = _make_data(
        tmp_path / "data", n_classes=3, id_col="id_col"
    )
    contract = validate_classification_data(
        public, id_column="id_col", row_order="id"
    )
    test_ids = contract.test_ids
    assert test_ids is not None
    rows = [
        {
            "id": test_id,
            "label": 0,
            "probabilities": [1.0, 0.0, 0.0],
        }
        for test_id in test_ids
    ]
    validate_predictions(
        {"predictions": rows},
        expected_count=30,
        n_classes=3,
        class_keys=contract.class_keys,
        test_ids=test_ids,
        id_column="id_col",
    )
    with pytest.raises(ValueError, match="missing="):
        validate_predictions(
            {"predictions": rows[:-1]},
            expected_count=30,
            n_classes=3,
            class_keys=contract.class_keys,
            test_ids=test_ids,
            id_column="id_col",
        )
    with pytest.raises(ValueError, match="duplicate id"):
        validate_predictions(
            {"predictions": [rows[0], *rows]},
            expected_count=30,
            n_classes=3,
            class_keys=contract.class_keys,
            test_ids=test_ids,
            id_column="id_col",
        )
    with pytest.raises(ValueError, match="exactly"):
        validate_predictions(
            {"predictions": [{**rows[0], "extra": 1}, *rows[1:]]},
            expected_count=30,
            n_classes=3,
            class_keys=contract.class_keys,
            test_ids=test_ids,
            id_column="id_col",
        )
    bad_id = [dict(row) for row in rows]
    bad_id[0]["id"] = True
    with pytest.raises(ValueError, match="id must"):
        validate_predictions(
            {"predictions": bad_id},
            expected_count=30,
            n_classes=3,
            class_keys=contract.class_keys,
            test_ids=test_ids,
            id_column="id_col",
        )


def test_claims_ignored_and_metrics_finite(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    problem = build_classification_problem(public, host)
    host_labels = pd.read_csv(
        host / "hidden_test_labels.csv"
    )["target"].tolist()
    payload = {
        "predictions": [
            {
                "label": label,
                "probabilities": [
                    1.0 if index == label else 0.0
                    for index in range(3)
                ],
            }
            for label in host_labels
        ],
        "claimed_accuracy": 0.0001,
        "claimed_macro_f1": 0.0001,
    }
    payload["score"] = 0.999
    pipeline = VerificationPipeline(problem)
    verification = pipeline.verify(_artifact(payload))
    assert verification.status.value == "verified"
    values = {o.name: o.value for o in verification.evidence}
    for name in (
        "accuracy",
        "macro_f1",
        "log_loss",
        "auroc",
        "multiclass_brier",
        "calibration_ece",
    ):
        assert np.isfinite(values[name])
    assert values["accuracy"] == 1.0  # host value, not the tiny claim


def test_verifier_confusion_all_cells_and_binary(tmp_path: Path) -> None:
    public, host = _make_data(
        tmp_path / "data", n_classes=2, class_values=[0, 1]
    )
    contract = validate_classification_data(public)
    labels, _class_keys = load_host_labels(host, contract)
    context = ClassificationVerificationContext(
        labels,
        expected_count=int(labels.size),
        classes=contract.classes,
        class_keys=contract.class_keys,
        row_order="input",
    )
    verifier = ClassificationVerifier()
    payload = _perfect_predictions([0, 1], 20)
    evidence = verifier.verify(_artifact(payload), context)
    values = {o.name: o.value for o in evidence.observations}
    assert {"confusion_0_0", "confusion_0_1", "confusion_1_0", "confusion_1_1"} <= set(
        values
    )
    assert sum(values[f"confusion_{i}_{j}"] for i in range(2) for j in range(2)) == 20
    assert np.isfinite(values["auroc"])


def test_constant_probability_auroc_is_half(tmp_path: Path) -> None:
    public, host = _make_data(
        tmp_path / "data", n_classes=2, class_values=[0, 1]
    )
    contract = validate_classification_data(public)
    labels, _class_keys = load_host_labels(host, contract)
    context = ClassificationVerificationContext(
        labels,
        expected_count=int(labels.size),
        classes=contract.classes,
        class_keys=contract.class_keys,
        row_order="input",
    )
    verifier = ClassificationVerifier()
    payload = {
        "predictions": [
            {"label": 0, "probabilities": [0.5, 0.5]}
            for _ in range(20)
        ]
    }
    evidence = verifier.verify(_artifact(payload), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["auroc"] == 0.5


def test_ece_includes_confidence_one(tmp_path: Path) -> None:
    # All wrong with confidence 1.0 -> ECE == 1.0 (last bin holds 1.0).
    public, host = _make_data(
        tmp_path / "data", n_classes=2, class_values=[0, 1]
    )
    contract = validate_classification_data(public)
    labels, _class_keys = load_host_labels(host, contract)
    context = ClassificationVerificationContext(
        labels,
        expected_count=int(labels.size),
        classes=contract.classes,
        class_keys=contract.class_keys,
        row_order="input",
    )
    verifier = ClassificationVerifier()
    wrong = {
        "predictions": [
            {
                "label": 1 - int(label),
                "probabilities": (
                    [0.0, 1.0] if int(label) == 0 else [1.0, 0.0]
                ),
            }
            for label in labels
        ]
    }
    evidence = verifier.verify(_artifact(wrong), context)
    values = {o.name: o.value for o in evidence.observations}
    assert values["calibration_ece"] == 1.0
    assert values["accuracy"] == 0.0


def test_context_invariant() -> None:
    labels = np.array([0, 1, 2, 0])
    with pytest.raises(ValueError, match="expected_count"):
        ClassificationVerificationContext(
            labels,
            expected_count=3,
            classes=(0, 1, 2),
            class_keys=("n:0", "n:1", "n:2"),
        )
    with pytest.raises(ValueError, match="at least two"):
        ClassificationVerificationContext(
            labels, classes=(0,), class_keys=("n:0",)
        )
    with pytest.raises(ValueError, match="equal length"):
        ClassificationVerificationContext(
            labels,
            classes=(0, 1, 2),
            class_keys=("n:0", "n:1"),
        )
    with pytest.raises(ValueError, match="within \\[0, n_classes\\)"):
        ClassificationVerificationContext(
            np.array([0, 1, 5]),
            classes=(0, 1, 2),
            class_keys=("n:0", "n:1", "n:2"),
        )
    with pytest.raises(ValueError, match="cover every declared class"):
        ClassificationVerificationContext(
            np.array([0, 1, 0, 1]),
            classes=(0, 1, 2),
            class_keys=("n:0", "n:1", "n:2"),
        )
    with pytest.raises(ValueError, match="prediction_ids"):
        ClassificationVerificationContext(
            labels,
            classes=(0, 1, 2),
            class_keys=("n:0", "n:1", "n:2"),
            row_order="id",
            id_column="id_col",
        )
    context = ClassificationVerificationContext(
        labels,
        classes=(0, 1, 2),
        class_keys=("n:0", "n:1", "n:2"),
    )
    assert context.n_classes == 3
    assert len(context.fingerprint()) == 64

"""R9: classification API end-to-end (mock generator + local runner)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ves_modeling.classification import (
    ApplyClassificationResult,
    ClassificationSearchResult,
    apply_classification_solution,
    capabilities,
    run_classification_search,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "classification"

TEMP_FIXTURE = '''\
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.environ.get("VES_DATA_DIR", "/data")
OUT_DIR = os.environ.get("VES_OUTPUT_DIR", "/output")

def to_scalar(value):
    return value.item() if hasattr(value, "item") else value

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test_features.csv")
id_col = "id_col" if "id_col" in test.columns else None
features = [c for c in test.columns if c != id_col]
classes = [to_scalar(v) for v in pd.unique(train["target"])]
class_to_index = {v: i for i, v in enumerate(classes)}
y = np.asarray(
    [class_to_index[to_scalar(v)] for v in train["target"]], dtype=np.int64
)
model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
    train[features].to_numpy(dtype=np.float64), y
)
probs = model.predict_proba(test[features].to_numpy(dtype=np.float64))
col_of_index = {int(v): c for c, v in enumerate(model.classes_)}
ordered = np.zeros((probs.shape[0], len(classes)))
for index, cls in enumerate(classes):
    ordered[:, index] = probs[:, col_of_index[class_to_index[cls]]]
labels = np.argmax(ordered, axis=1)
rows = []
for row_index, (_, row) in enumerate(test.iterrows()):
    record = {
        "label": to_scalar(classes[int(labels[row_index])]),
        "probabilities": [float(v) for v in ordered[row_index]],
    }
    if id_col:
        record["id"] = to_scalar(row[id_col])
    rows.append(record)
os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": rows}, fh)
'''


def _make_data(
    root: Path,
    *,
    n_classes: int = 3,
    class_values: list | None = None,
    id_col: str | None = None,
    seed: int = 23,
) -> tuple[Path, Path]:
    if class_values is None:
        class_values = list(range(n_classes))
    per_class = 40
    n_test = 10 * n_classes
    X, y = make_classification(
        n_samples=per_class * n_classes + n_test,
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
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train = pd.DataFrame(X_train, columns=feature_names)
    train["target"] = [class_values[int(i)] for i in y_train]
    train["_class_index"] = y_train
    train = train.sort_values("_class_index").drop(
        columns="_class_index"
    ).reset_index(drop=True)
    test = pd.DataFrame(X_test, columns=feature_names)
    host_frame = pd.DataFrame(
        {"target": [class_values[int(i)] for i in y_test]}
    )
    if id_col:
        train[id_col] = np.arange(1, len(train) + 1)
        test[id_col] = np.arange(1, len(test) + 1)
        host_frame[id_col] = np.arange(1, len(test) + 1)
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test_features.csv", index=False)
    host_frame.to_csv(host / "hidden_test_labels.csv", index=False)
    return public, host


def _write_temp_fixture(root: Path) -> Path:
    fixture_dir = root / "fixtures"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "logistic_balanced.py").write_text(
        TEMP_FIXTURE, encoding="utf-8"
    )
    (fixture_dir / "random_forest_balanced.py").write_text(
        TEMP_FIXTURE, encoding="utf-8"
    )
    return fixture_dir


def test_run_classification_search_mock_verified(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_classification_search(
        public,
        host,
        drafts=2,
        improves=1,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert isinstance(result, ClassificationSearchResult)
    assert result.status == "verified"
    assert result.best_code is not None
    assert result.best_macro_f1 is not None and result.best_macro_f1 > 0.0
    assert result.best_accuracy is not None
    assert result.best_log_loss is not None
    assert result.best_auroc is not None
    assert result.best_multiclass_brier is not None
    assert result.best_calibration_ece is not None
    assert result.best_confusion_matrix is not None
    assert len(result.best_confusion_matrix) == 3
    assert result.classes == (0, 1, 2)
    assert result.rejected == 0
    assert (result.run_dir / "best_solution.py").is_file()
    candidates = list((result.run_dir / "candidates").iterdir())
    assert len(candidates) == 3
    assert all((candidate / "run.json").is_file() for candidate in candidates)


def test_search_explicit_reversed_classes_e2e(tmp_path: Path) -> None:
    public, host = _make_data(
        tmp_path / "data", n_classes=2, class_values=[1, 0]
    )
    fixture_dir = _write_temp_fixture(tmp_path)
    result = run_classification_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=fixture_dir,
        classes=[1, 0],
    )
    assert result.status == "verified"
    assert result.classes == (1, 0)
    assert result.data_contract["classes"] == [1, 0]
    assert result.data_contract["class_counts"] == [40, 40]


def test_search_and_apply_id_mode_e2e(tmp_path: Path) -> None:
    public, host = _make_data(
        tmp_path / "data", n_classes=3, id_col="id_col"
    )
    fixture_dir = _write_temp_fixture(tmp_path)
    result = run_classification_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=fixture_dir,
        row_order="id",
        id_column="id_col",
    )
    assert result.status == "verified"
    assert result.data_contract["row_order"] == "id"
    assert result.data_contract["id_column"] == "id_col"
    applied = apply_classification_solution(
        result.best_code,
        public,
        workspace=tmp_path / "apply_workspace",
        trusted_code=True,
        row_order="id",
        id_column="id_col",
    )
    assert applied.status == "produced_unverified"
    assert applied.predictions_path is not None
    payload = json.loads(applied.predictions_path.read_text(encoding="utf-8"))
    assert all(
        set(item.keys()) == {"id", "label", "probabilities"}
        for item in payload["predictions"]
    )


def test_summary_parity_and_no_hidden(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    result = run_classification_search(
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
    assert persisted["task"] == "classification"
    assert persisted["status"] == "verified"
    assert persisted["classes"] == [0, 1, 2]
    summary_text = json.dumps(persisted)
    assert "hidden" not in summary_text
    provenance = json.loads(
        (result.run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    host_hash = provenance["inputs"]["host"]["hidden_test_labels.csv"]
    assert re.fullmatch(r"[0-9a-f]{64}", host_hash)


def test_apply_produced_unverified_no_metrics(tmp_path: Path) -> None:
    public, _host = _make_data(tmp_path / "data")
    code = (FIXTURES / "logistic_balanced.py").read_text(encoding="utf-8")
    result = apply_classification_solution(
        code,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
    )
    assert isinstance(result, ApplyClassificationResult)
    assert result.status == "produced_unverified"
    assert result.runner == "local"
    assert result.predictions_path is not None
    payload = json.loads(result.predictions_path.read_text(encoding="utf-8"))
    assert len(payload["predictions"]) == 30
    assert all(
        set(item.keys()) == {"label", "probabilities"}
        for item in payload["predictions"]
    )
    summary = result.to_summary()
    json.dumps(summary)
    for metric in ("accuracy", "macro_f1", "auroc", "log_loss"):
        assert metric not in summary
    persisted = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_apply_artifact_invalid_raises_and_summary_status(
    tmp_path: Path,
) -> None:
    public, _host = _make_data(tmp_path / "data")
    good = (FIXTURES / "logistic_balanced.py").read_text(encoding="utf-8")
    bad_code = good.replace(
        'json.dump({"predictions": rows}, fh)',
        'json.dump({"predictions": rows[:-1]}, fh)',
    )
    run_id = "applyinvalid0"
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_classification_solution(
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
    run_json = json.loads(
        (run_dir / "candidate" / "run.json").read_text(encoding="utf-8")
    )
    assert run_json["status"] == "artifact_invalid"


def test_run_classification_search_rejects_unknown_generator(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data")
    with pytest.raises(ValueError, match="unknown generator"):
        run_classification_search(
            public,
            host,
            generator="unknown",
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_capabilities_classification() -> None:
    declaration = capabilities()
    assert declaration["operations"] == [
        "run_classification_search",
        "apply_classification_solution",
    ]
    assert declaration["apply_statuses"] == ["produced_unverified"]
    assert "confusion_*" in declaration["verified_metrics"]
    assert declaration["data_contract"]["row_order"] == ["input", "id"]

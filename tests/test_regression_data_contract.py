"""R7.3 Batch B: regression data contract (target/id/row_order)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ves_modeling.regression import (
    apply_regression_solution,
    capabilities,
    run_regression_search,
)
from ves_modeling.regression.context import RegressionVerificationContext
from ves_modeling.regression.data_contract import (
    validate_predictions,
    validate_regression_data,
)
from ves_modeling.regression.problem import build_regression_problem

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "candidates"
LINEAR = (FIXTURES / "linear_regression.py").read_text(encoding="utf-8")


def _make_data(
    root: Path,
    n: int = 30,
    *,
    target: str = "target",
    id_col: str | None = None,
    host_order: str = "same",
) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2))
    y = 3.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(scale=0.1, size=n)
    split = int(n * 0.7)
    public = root / "public"
    host = root / "host"
    public.mkdir(parents=True)
    host.mkdir(parents=True)
    train_cols = {f"x{i}": x[:split, i] for i in range(2)}
    test_cols = {f"x{i}": x[split:, i] for i in range(2)}
    if id_col:
        train_cols[id_col] = np.arange(1, split + 1)
        test_cols[id_col] = np.arange(1, n - split + 1)
    train_cols[target] = y[:split]
    pd.DataFrame(train_cols).to_csv(public / "train.csv", index=False)
    pd.DataFrame(test_cols).to_csv(
        public / "test_features.csv", index=False
    )
    host_ids = np.arange(1, n - split + 1)
    host_labels = y[split:]
    if id_col and host_order == "reversed":
        host_ids = host_ids[::-1]
        host_labels = host_labels[::-1]
    host_cols: dict[str, object] = {}
    if id_col:
        host_cols[id_col] = host_ids
    host_cols[target] = host_labels
    pd.DataFrame(host_cols).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    return public, host


ID_CODE = '''\
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = os.environ["REGRESSION_DATA_DIR"]
OUT = os.environ["REGRESSION_OUTPUT_DIR"]
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test_features.csv")
features = [c for c in test.columns if c != "id_col"]
model = LinearRegression().fit(
    train[features].to_numpy(dtype=np.float64),
    train["y"].to_numpy(dtype=np.float64),
)
predictions = model.predict(test[features].to_numpy(dtype=np.float64))
rows = [
    {"id": int(raw_id), "prediction": float(value)}
    for raw_id, value in zip(test["id_col"], predictions)
]
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/predictions.json", "w", encoding="utf-8") as fh:
    json.dump({"predictions": rows}, fh)
'''


def _id_fixtures(tmp_path: Path, name: str = "linear_regression.py") -> Path:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / name).write_text(ID_CODE, encoding="utf-8")
    return fixtures


def test_default_contract_matches_previous_behavior(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    contract = validate_regression_data(public)
    assert contract.target_column == "target"
    assert contract.id_column is None
    assert contract.row_order == "input"
    assert contract.test_ids is None
    assert contract.to_dict()["feature_columns"] == ["x0", "x1"]
    assert contract.to_dict()["train_rows"] == 21
    assert contract.to_dict()["test_rows"] == 9
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
    )
    assert result.status == "verified"
    summary = json.loads(
        (result.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["data_contract"]["row_order"] == "input"
    assert capabilities()["data_contract"]["row_order"] == ["input", "id"]


def test_custom_target_apply_and_problem(tmp_path: Path) -> None:
    public, _ = _make_data(tmp_path / "data", target="y")
    custom_code = LINEAR.replace('drop(columns=["target"])', 'drop(columns=["y"])')
    custom_code = custom_code.replace('train["target"]', 'train["y"]')
    result = apply_regression_solution(
        custom_code,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
        target_column="y",
    )
    assert result.status == "produced_unverified"
    assert result.data_contract["target_column"] == "y"
    labels = np.full(9, 1.0)
    problem = build_regression_problem(
        public,
        tmp_path / "host",
        labels=labels,
        target_column="y",
    )
    assert problem.make_context().expected_count == 9


@pytest.mark.parametrize(
    "mutate, match",
    [
        ("drop-feature", "columns must match train features exactly"),
        ("extra-feature", "columns must match train features exactly"),
        ("reorder", "columns must match train features exactly"),
        ("dup-header", "duplicate column names"),
    ],
)
def test_feature_contract_violations(
    tmp_path: Path, mutate: str, match: str
) -> None:
    public, _ = _make_data(tmp_path / "data")
    test_path = public / "test_features.csv"
    if mutate == "drop-feature":
        frame = pd.read_csv(test_path).drop(columns=["x1"])
        frame.to_csv(test_path, index=False)
    elif mutate == "extra-feature":
        frame = pd.read_csv(test_path)
        frame["z"] = 0.0
        frame.to_csv(test_path, index=False)
    elif mutate == "reorder":
        frame = pd.read_csv(test_path)[["x1", "x0"]]
        frame.to_csv(test_path, index=False)
    elif mutate == "dup-header":
        (public / "train.csv").write_text(
            "x0,x0,target\n1,2,3\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match=match):
        validate_regression_data(public)


def test_labels_count_must_match_test_rows(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    pd.DataFrame({"target": [1.0, 2.0, 3.0]}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    with pytest.raises(ValueError, match="hidden labels count"):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


@pytest.mark.parametrize(
    "mutate, match",
    [
        ("train-dup", "duplicate ids"),
        ("test-null", "empty ids"),
        ("host-dup", "duplicate ids"),
    ],
)
def test_id_column_contract_violations(
    tmp_path: Path, mutate: str, match: str
) -> None:
    public, host = _make_data(tmp_path / "data", id_col="id_col")
    if mutate == "train-dup":
        frame = pd.read_csv(public / "train.csv")
        frame.loc[1, "id_col"] = frame.loc[0, "id_col"]
        frame.to_csv(public / "train.csv", index=False)
    elif mutate == "test-null":
        frame = pd.read_csv(public / "test_features.csv")
        frame.loc[0, "id_col"] = None
        frame.to_csv(public / "test_features.csv", index=False)
    elif mutate == "host-dup":
        frame = pd.read_csv(host / "hidden_test_labels.csv")
        frame.loc[1, "id_col"] = frame.loc[0, "id_col"]
        frame.to_csv(host / "hidden_test_labels.csv", index=False)
    with pytest.raises(ValueError, match=match):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
            target_column="target",
            id_column="id_col",
            row_order="id",
        )


def test_host_id_mismatch_rejected(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data", id_col="id_col")
    frame = pd.read_csv(host / "hidden_test_labels.csv")
    frame.loc[0, "id_col"] = 999
    frame.to_csv(host / "hidden_test_labels.csv", index=False)
    with pytest.raises(ValueError, match="ids must match public test ids"):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
            target_column="target",
            id_column="id_col",
            row_order="id",
        )


def test_host_id_reorder_aligns_labels(tmp_path: Path) -> None:
    public_a, host_a = _make_data(
        tmp_path / "a", target="y", id_col="id_col", host_order="same"
    )
    public_b, host_b = _make_data(
        tmp_path / "b", target="y", id_col="id_col", host_order="reversed"
    )
    result_a = run_regression_search(
        public_a,
        host_a,
        drafts=1,
        improves=0,
        workspace=tmp_path / "w1",
        fixture_dir=_id_fixtures(tmp_path / "f1"),
        target_column="y",
        id_column="id_col",
        row_order="id",
    )
    result_b = run_regression_search(
        public_b,
        host_b,
        drafts=1,
        improves=0,
        workspace=tmp_path / "w2",
        fixture_dir=_id_fixtures(tmp_path / "f2"),
        target_column="y",
        id_column="id_col",
        row_order="id",
    )
    assert result_a.status == "verified"
    assert result_b.status == "verified"
    assert result_a.best_rmse == result_b.best_rmse


def test_id_predictions_validator_direct() -> None:
    test_ids = ("1", "2", "3")
    aligned = validate_predictions(
        {
            "predictions": [
                {"id": 1, "prediction": 1.5},
                {"id": "2", "prediction": 2.5},
                {"id": 3.0, "prediction": 3.5},
            ]
        },
        expected_count=3,
        test_ids=test_ids,
    )
    assert list(aligned) == [1.5, 2.5, 3.5]
    bad_payloads = [
        [{"prediction": 1.0}, {"id": 2, "prediction": 2.0}, {"id": 3, "prediction": 3.0}],
        [{"id": 1, "prediction": 1.0}, {"id": 2, "prediction": 2.0}, {"id": 99, "prediction": 3.0}],
        [{"id": 1, "prediction": 1.0}, {"id": 1, "prediction": 2.0}, {"id": 3, "prediction": 3.0}],
        [{"id": True, "prediction": 1.0}, {"id": 2, "prediction": 2.0}, {"id": 3, "prediction": 3.0}],
        [{"id": [1], "prediction": 1.0}, {"id": 2, "prediction": 2.0}, {"id": 3, "prediction": 3.0}],
        [{"id": 1, "prediction": True}, {"id": 2, "prediction": 2.0}, {"id": 3, "prediction": 3.0}],
        [{"id": 1, "prediction": float("nan")}, {"id": 2, "prediction": 2.0}, {"id": 3, "prediction": 3.0}],
    ]
    for payload in bad_payloads:
        with pytest.raises(ValueError):
            validate_predictions(
                {"predictions": payload},
                expected_count=3,
                test_ids=test_ids,
            )


def test_apply_id_success_and_failure(tmp_path: Path) -> None:
    public, _ = _make_data(tmp_path / "data", target="y", id_col="id_col")
    success = apply_regression_solution(
        ID_CODE,
        public,
        workspace=tmp_path / "runs",
        trusted_code=True,
        target_column="y",
        id_column="id_col",
        row_order="id",
    )
    assert success.status == "produced_unverified"
    assert success.data_contract["row_order"] == "id"
    payload = json.loads(success.predictions_path.read_text(encoding="utf-8"))
    assert all(isinstance(row, dict) for row in payload["predictions"])

    bad_code = ID_CODE.replace(
        'os.makedirs(OUT, exist_ok=True)',
        'rows[0]["id"] = rows[1]["id"]\nos.makedirs(OUT, exist_ok=True)',
    )
    with pytest.raises(RuntimeError, match="artifact_invalid"):
        apply_regression_solution(
            bad_code,
            public,
            workspace=tmp_path / "runs2",
            run_id="fixed",
            trusted_code=True,
            target_column="y",
            id_column="id_col",
            row_order="id",
        )
    run_json = json.loads(
        (
            tmp_path / "runs2" / "fixed" / "candidate" / "run.json"
        ).read_text(encoding="utf-8")
    )
    assert run_json["status"] == "artifact_invalid"


def test_llm_prompt_follows_data_contract() -> None:
    from ves_modeling.regression.generator import _draft_prompt, _improve_prompt

    draft = _draft_prompt("y", "id_col", "id")
    improve = _improve_prompt(
        LINEAR, 1.0, 2.0, target_column="y", id_column="id_col", row_order="id"
    )
    assert "y" in draft and "id_col" in draft
    assert "must NOT be used as a model feature" in draft
    assert '"id": <test row id>' in draft
    assert "must NOT be used as a model feature" in improve
    assert "row order" not in draft
    input_prompt = _draft_prompt("target", None, "input")
    assert "aligned to test_features.csv row order" in input_prompt
    assert "model feature" not in input_prompt


def test_id_column_with_input_row_order_uses_array_artifact(
    tmp_path: Path,
) -> None:
    public, host = _make_data(tmp_path / "data", id_col="id_col")
    result = run_regression_search(
        public,
        host,
        drafts=1,
        improves=0,
        workspace=tmp_path / "workspace",
        fixture_dir=FIXTURES,
        id_column="id_col",
        row_order="input",
    )
    assert result.status == "verified"
    run_json = json.loads(
        (
            result.run_dir / "candidates" / "draft0" / "run.json"
        ).read_text(encoding="utf-8")
    )
    assert run_json["status"] == "verified"
    apply_result = apply_regression_solution(
        LINEAR,
        public,
        workspace=tmp_path / "apply",
        trusted_code=True,
        id_column="id_col",
        row_order="input",
    )
    assert apply_result.status == "produced_unverified"
    assert apply_result.data_contract["row_order"] == "input"
    assert apply_result.data_contract["input_columns"] == ["x0", "x1", "id_col"]
    assert apply_result.data_contract["feature_columns"] == ["x0", "x1"]


def test_host_duplicate_target_header_rejected(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    (host / "hidden_test_labels.csv").write_text(
        "target,target\n1,2\n3,4\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate column names"):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
        )


def test_context_fingerprint_includes_row_order_and_id_hash() -> None:
    labels = np.array([1.0, 2.0, 3.0])
    input_ctx = RegressionVerificationContext(
        labels, dataset_name="unit", expected_count=3
    )
    id_ctx = RegressionVerificationContext(
        labels,
        dataset_name="unit",
        expected_count=3,
        id_column="id_col",
        prediction_ids=("1", "2", "3"),
        row_order="id",
    )
    id_ctx_other = RegressionVerificationContext(
        labels,
        dataset_name="unit",
        expected_count=3,
        id_column="id_col",
        prediction_ids=("3", "2", "1"),
        row_order="id",
    )
    assert id_ctx.fingerprint() != input_ctx.fingerprint()
    assert id_ctx.fingerprint() != id_ctx_other.fingerprint()
    assert id_ctx.fingerprint() == id_ctx.fingerprint()


def test_labels_injection_rejected_for_id_row_order(tmp_path: Path) -> None:
    public, _ = _make_data(tmp_path / "data", target="y", id_col="id_col")
    with pytest.raises(ValueError, match="labels injection"):
        build_regression_problem(
            public,
            tmp_path / "host",
            labels=np.full(9, 1.0),
            target_column="y",
            id_column="id_col",
            row_order="id",
        )


def test_id_artifact_extra_key_rejected() -> None:
    test_ids = ("1", "2", "3")
    with pytest.raises(ValueError, match="exactly"):
        validate_predictions(
            {
                "predictions": [
                    {"id": 1, "prediction": 1.0, "score": 9},
                    {"id": 2, "prediction": 2.0},
                    {"id": 3, "prediction": 3.0},
                ]
            },
            expected_count=3,
            test_ids=test_ids,
        )


def test_context_factory_env_id_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ves_modeling.regression.problem import context_factory

    public, host = _make_data(tmp_path / "data", target="y", id_col="id_col")
    monkeypatch.setenv("VES_MODELING_HOST_DIR", str(host))
    monkeypatch.setenv("VES_MODELING_PUBLIC_DIR", str(public))
    monkeypatch.setenv("VES_MODELING_TARGET_COLUMN", "y")
    monkeypatch.setenv("VES_MODELING_ID_COLUMN", "id_col")
    monkeypatch.setenv("VES_MODELING_ROW_ORDER", "id")
    context = context_factory()
    assert context.row_order == "id"
    assert context.id_column == "id_col"
    assert context.expected_count == 9
    assert tuple(context.prediction_ids or ()) == tuple(
        str(i) for i in range(1, 10)
    )

    monkeypatch.delenv("VES_MODELING_PUBLIC_DIR")
    with pytest.raises(RuntimeError, match="PUBLIC_DIR"):
        context_factory()

    monkeypatch.setenv("VES_MODELING_ROW_ORDER", "input")
    monkeypatch.delenv("VES_MODELING_ID_COLUMN")
    monkeypatch.setenv("VES_MODELING_TARGET_COLUMN", "y")
    monkeypatch.delenv("VES_MODELING_PUBLIC_DIR", raising=False)
    input_context = context_factory()
    assert input_context.row_order == "input"
    assert input_context.expected_count == 9


def test_host_contract_error_leaves_no_run_dir(tmp_path: Path) -> None:
    public, host = _make_data(tmp_path / "data")
    pd.DataFrame({"target": [1.0, 2.0]}).to_csv(
        host / "hidden_test_labels.csv", index=False
    )
    workspace = tmp_path / "workspace"
    with pytest.raises(ValueError, match="hidden labels count"):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=workspace,
            fixture_dir=FIXTURES,
        )
    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize(
    "mutate, match",
    [
        ("empty-train", "train.csv must have at least one row"),
        ("empty-test", "test_features.csv must have at least one row"),
        ("target-text", "target column 'target' must be numeric"),
        ("target-nan", "target column must be non-empty and finite"),
        ("id-bool", "ids must not be booleans"),
        ("id-inf", "ids must be finite"),
        ("host-text", "hidden labels target column 'target' must be numeric"),
        ("no-feature", "at least one model feature column"),
    ],
)
def test_data_quality_violations(
    tmp_path: Path, mutate: str, match: str
) -> None:
    public, host = _make_data(tmp_path / "data", id_col="id_col")
    if mutate == "empty-train":
        (public / "train.csv").write_text(
            "x0,x1,id_col,target\n", encoding="utf-8"
        )
    elif mutate == "empty-test":
        (public / "test_features.csv").write_text(
            "x0,x1,id_col\n", encoding="utf-8"
        )
    elif mutate == "target-text":
        frame = pd.read_csv(public / "train.csv")
        frame["target"] = frame["target"].astype(object)
        frame.loc[0, "target"] = "not-a-number"
        frame.to_csv(public / "train.csv", index=False)
    elif mutate == "target-nan":
        frame = pd.read_csv(public / "train.csv")
        frame.loc[0, "target"] = None
        frame.to_csv(public / "train.csv", index=False)
    elif mutate == "id-bool":
        frame = pd.read_csv(public / "test_features.csv")
        frame["id_col"] = ["True", "False", "True", "False", "True", "False", "True", "False", "True"]
        frame.to_csv(public / "test_features.csv", index=False)
    elif mutate == "id-inf":
        frame = pd.read_csv(public / "test_features.csv")
        frame["id_col"] = frame["id_col"].astype(np.float64)
        frame.loc[0, "id_col"] = float("inf")
        frame.to_csv(public / "test_features.csv", index=False)
    elif mutate == "host-text":
        frame = pd.read_csv(host / "hidden_test_labels.csv")
        frame["target"] = frame["target"].astype(object)
        frame.loc[0, "target"] = "not-a-number"
        frame.to_csv(host / "hidden_test_labels.csv", index=False)
    elif mutate == "no-feature":
        (public / "train.csv").write_text(
            "id_col,target\n1,1.0\n2,2.0\n", encoding="utf-8"
        )
        (public / "test_features.csv").write_text(
            "id_col\n1\n2\n", encoding="utf-8"
        )
    with pytest.raises(ValueError, match=match):
        run_regression_search(
            public,
            host,
            drafts=1,
            improves=0,
            workspace=tmp_path / "workspace",
            fixture_dir=FIXTURES,
            target_column="target",
            id_column="id_col",
            row_order="id",
        )


def test_id_column_input_host_reorder_aligned_and_mismatch(
    tmp_path: Path,
) -> None:
    public_a, host_a = _make_data(
        tmp_path / "a", id_col="id_col", host_order="same"
    )
    public_b, host_b = _make_data(
        tmp_path / "b", id_col="id_col", host_order="reversed"
    )
    result_a = run_regression_search(
        public_a,
        host_a,
        drafts=1,
        improves=0,
        workspace=tmp_path / "w1",
        fixture_dir=FIXTURES,
        id_column="id_col",
        row_order="input",
    )
    result_b = run_regression_search(
        public_b,
        host_b,
        drafts=1,
        improves=0,
        workspace=tmp_path / "w2",
        fixture_dir=FIXTURES,
        id_column="id_col",
        row_order="input",
    )
    assert result_a.status == "verified"
    assert result_b.status == "verified"
    assert result_a.best_rmse == result_b.best_rmse

    frame = pd.read_csv(host_a / "hidden_test_labels.csv")
    frame.loc[0, "id_col"] = 999
    frame.to_csv(host_a / "hidden_test_labels.csv", index=False)
    with pytest.raises(ValueError, match="ids must match public test ids"):
        run_regression_search(
            public_a,
            host_a,
            drafts=1,
            improves=0,
            workspace=tmp_path / "w3",
            fixture_dir=FIXTURES,
            id_column="id_col",
            row_order="input",
        )


def test_context_factory_id_input_replay_requires_public_and_aligns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ves_modeling.regression.problem import context_factory

    public, host = _make_data(tmp_path / "data", id_col="id_col")
    monkeypatch.setenv("VES_MODELING_HOST_DIR", str(host))
    monkeypatch.setenv("VES_MODELING_ID_COLUMN", "id_col")
    monkeypatch.setenv("VES_MODELING_ROW_ORDER", "input")
    with pytest.raises(RuntimeError, match="PUBLIC_DIR"):
        context_factory()

    monkeypatch.setenv("VES_MODELING_PUBLIC_DIR", str(public))
    replayed = context_factory()
    assert replayed.id_column == "id_col"
    assert replayed.row_order == "input"
    assert replayed.prediction_ids is None
    assert replayed.expected_count == 9
    direct = build_regression_problem(
        public,
        host,
        id_column="id_col",
        row_order="input",
    ).make_context()
    assert replayed.fingerprint() == direct.fingerprint()

    monkeypatch.setenv("VES_MODELING_ROW_ORDER", "id")
    replayed_id = context_factory()
    direct_id = build_regression_problem(
        public,
        host,
        id_column="id_col",
        row_order="id",
    ).make_context()
    assert replayed_id.prediction_ids == direct_id.prediction_ids
    assert replayed_id.fingerprint() == direct_id.fingerprint()


def test_fingerprint_id_hash_is_ambiguous_safe() -> None:
    labels = np.array([1.0, 2.0])
    first = RegressionVerificationContext(
        labels,
        dataset_name="unit",
        expected_count=2,
        id_column="id_col",
        prediction_ids=("ab", "c"),
        row_order="id",
    )
    second = RegressionVerificationContext(
        labels,
        dataset_name="unit",
        expected_count=2,
        id_column="id_col",
        prediction_ids=("a", "bc"),
        row_order="id",
    )
    assert first.fingerprint() != second.fingerprint()

"""Candidate generators for the forecasting slice.

- MockForecastingGenerator: returns trusted fixture code (no LLM).
- LLMForecastingGenerator: provider-neutral adapter over an OpenAI-compatible
  client, with draft/improve prompts that follow the active forecasting data
  contract (series/time columns, frequency, horizon, exog and keyed artifact
  format).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Protocol

from ves.problem import VerifiedProblem
from ves.record import VerifiedCandidate

logger = logging.getLogger(__name__)

MOCK_FIXTURES = (
    "naive_forecast.py",
    "linear_forecast.py",
)

FAMILY_FIXTURES = {
    "statistical": (
        "family_statistical_seasonal_naive.py",
        "family_statistical_linear_seasonal.py",
    ),
    "ml": (
        "family_ml_ridge_lag.py",
        "family_ml_ridge_lag.py",
    ),
    "mechanistic": (
        "family_mechanistic_decomp_sine.py",
        "family_mechanistic_decomp_sine.py",
    ),
}

FAMILY_PROMPT_LINES = {
    "statistical": (
        "本次聚焦方法族 statistical：统计方法（季节朴素、线性+季节哑元、"
        "Holt-Winters 类），强调趋势与季节分解。"
    ),
    "ml": (
        "本次聚焦方法族 ml：机器学习方法（滞后特征 + 日历特征的岭回归等），"
        "强调特征工程与正则化。"
    ),
    "mechanistic": (
        "本次聚焦方法族 mechanistic：机理/结构方法（趋势分解 + 正弦周期结构、"
        "阻尼趋势），强调可解释结构。"
    ),
}


class LlmClient(Protocol):
    """Minimal OpenAI-compatible client used by generators."""

    def complete(self, prompt: str) -> str: ...


def _read_fixture(fixture_dir: Path, name: str) -> str:
    path = fixture_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"fixture candidate missing: {path}")
    return path.read_text(encoding="utf-8")


class MockForecastingGenerator:
    """Returns preset trusted candidates for draft/improve steps.

    draft(0) -> naive_forecast, draft(1) -> linear_forecast,
    improve -> linear_forecast (cycled for later improves).
    """

    def __init__(
        self,
        fixture_dir: Path,
        *,
        drafts: tuple[str, ...] = MOCK_FIXTURES[:2],
        improves: tuple[str, ...] = MOCK_FIXTURES[1:],
        method_family: str | None = None,
    ) -> None:
        if method_family is not None:
            if method_family not in FAMILY_FIXTURES:
                raise ValueError(
                    f"method_family must be one of {tuple(FAMILY_FIXTURES)}"
                )
            drafts = FAMILY_FIXTURES[method_family]
            improves = FAMILY_FIXTURES[method_family][1:]
            if not improves:
                improves = FAMILY_FIXTURES[method_family]
        self._fixture_dir = Path(fixture_dir)
        self._drafts = tuple(drafts)
        self._improves = tuple(improves)
        self._method_family = method_family

    def draft(self, problem: VerifiedProblem, index: int) -> str:
        name = self._drafts[index % len(self._drafts)]
        return _read_fixture(self._fixture_dir, name)

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        generation = anchor.candidate.generation
        name = self._improves[generation % len(self._improves)]
        return _read_fixture(self._fixture_dir, name)


def _artifact_format_prompt(
    series_id_column: str, time_column: str, row_order: str
) -> str:
    if row_order == "key":
        return (
            "{{\n"
            '  "predictions": [\n'
            '    {{"<series id column>": <series id>, '
            '"<time column>": "<strict ISO 8601 timestamp>", '
            '"prediction": <number>}},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Use the actual column names from test_features.csv for "
            "<series id column> and <time column>. Every (series, timestamp) "
            "row of test_features.csv must appear exactly once, copying the "
            "timestamp as the strict ISO 8601 string from the CSV. "
            "Series ids and timestamps are row keys, not model features."
        )
    return (
        "{{\n"
        '  "predictions": [...]\n'
        "}}\n\n"
        "Predictions must be aligned to test_features.csv row order."
    )


def _draft_prompt(
    *,
    time_column: str,
    series_id_column: str,
    target_column: str,
    frequency: str,
    horizon: int,
    feature_columns: tuple[str, ...],
    row_order: str,
    method_family: str | None = None,
) -> str:
    lines = [
        "You are solving a time-series forecasting task.",
        "",
    ]
    if method_family is not None:
        lines.append(FAMILY_PROMPT_LINES[method_family])
        lines.append("")
    lines += [
        "Available files:",
        "/data/train.csv",
        "/data/test_features.csv",
        "",
        (
            "train.csv is long-format history: one row per "
            f"({series_id_column}, {time_column}). It contains the target "
            f"column: {target_column}."
        ),
        f"Series id column: {series_id_column}.",
        f"Time column: {time_column} (strict ISO 8601 strings).",
        f"Frequency: {frequency}.",
        f"Forecast horizon: {horizon} future steps per series.",
        "",
    ]
    if feature_columns:
        lines += [
            "Exogenous/feature columns (may be used as model features):",
            ", ".join(feature_columns),
            "",
        ]
    else:
        lines += [
            (
                "There are no exogenous feature columns; forecast from the "
                "target history only."
            ),
            "",
        ]
    lines += [
        "You must create one complete Python program. The program must:",
        "1. Load /data/train.csv.",
        "2. Load /data/test_features.csv.",
        (
            "3. Forecast every row of test_features.csv (each series appears "
            f"{horizon} times, one step per frequency {frequency})."
        ),
        "4. Write exactly one artifact:",
        "   /output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(series_id_column, time_column, row_order),
        "",
        "You cannot access hidden evaluation labels.",
        "Do not report or rely on a self-computed test score.",
        "The host verifier will independently evaluate the predictions.",
        "",
        "Allowed libraries: numpy, pandas, scikit-learn.",
        "Do not use pip install, curl, wget, or network access.",
        "",
        "Your response must contain the complete solution.py only.",
        "Do not wrap the code in markdown fences.",
    ]
    return "\n".join(lines)


def _improve_prompt(
    code: str,
    rmse: float,
    mae: float,
    smape: float,
    *,
    time_column: str,
    series_id_column: str,
    target_column: str,
    frequency: str,
    horizon: int,
    feature_columns: tuple[str, ...],
    row_order: str,
    method_family: str | None = None,
) -> str:
    lines = [
        "Previous candidate:",
        "",
        code,
        "",
    ]
    if method_family is not None:
        lines.append(FAMILY_PROMPT_LINES[method_family])
        lines.append("")
    lines += [
        "Host-verified evidence (independently recomputed by the host):",
        "",
        f"RMSE: {rmse:.6f}",
        f"MAE: {mae:.6f}",
        f"SMAPE: {smape:.6f}",
        "",
        (
            "Improve the executable solution. You may change preprocessing, "
            "trend/seasonality handling, estimator, hyperparameters, or "
            "ensembling."
        ),
        "",
        (
            "You must still load /data/train.csv and /data/test_features.csv, "
            "forecast every row, and write exactly one artifact:"
        ),
        "/output/predictions.json",
        "",
        "Artifact format:",
        _artifact_format_prompt(series_id_column, time_column, row_order),
        "",
        f"Series id column: {series_id_column}.",
        f"Time column: {time_column} (strict ISO 8601 strings).",
        f"Frequency: {frequency}. Forecast horizon: {horizon} per series.",
        f"Target column: {target_column}.",
    ]
    if feature_columns:
        lines += [
            "Exogenous/feature columns: " + ", ".join(feature_columns) + ".",
        ]
    else:
        lines += [
            (
                "There are no exogenous feature columns; forecast from target "
                "history only."
            ),
        ]
    lines += [
        "",
        (
            "You cannot access hidden evaluation labels. Do not report a "
            "self-computed score; the host verifier independently evaluates "
            "the predictions."
        ),
        "",
        "Allowed libraries: numpy, pandas, scikit-learn.",
        "Do not use pip install, curl, wget, or network access.",
        "",
        (
            "Return the complete solution.py only. Do not wrap the code in "
            "markdown fences."
        ),
    ]
    return "\n".join(lines)


class LLMForecastingGenerator:
    """Draft/improve prompts driven entirely by host-verified evidence.

    ``fallback_code`` (optional) is used when the LLM client fails after its
    internal retries, so an infrastructure outage degrades to a real
    executable candidate instead of crashing the search.  The fallback is
    still verified by the host; no score is ever fabricated.
    """

    def __init__(
        self,
        llm: LlmClient,
        fallback_code: str | None = None,
        *,
        time_column: str = "timestamp",
        series_id_column: str = "series_id",
        target_column: str = "target",
        frequency: str = "D",
        horizon: int = 1,
        feature_columns: tuple[str, ...] = (),
        row_order: str = "key",
        method_family: str | None = None,
    ) -> None:
        self._llm = llm
        self._fallback_code = fallback_code
        if row_order not in ("input", "key"):
            raise ValueError("row_order must be 'input' or 'key'")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self._time_column = time_column
        self._series_id_column = series_id_column
        self._target_column = target_column
        self._frequency = frequency
        self._horizon = horizon
        self._feature_columns = tuple(feature_columns)
        self._row_order = row_order
        if method_family is not None and method_family not in FAMILY_PROMPT_LINES:
            raise ValueError(
                f"method_family must be one of {tuple(FAMILY_PROMPT_LINES)}"
            )
        self._method_family = method_family

    def _complete(self, prompt: str) -> str:
        try:
            return self._llm.complete(prompt)
        except Exception as exc:
            if self._fallback_code is None:
                raise
            logger.warning(
                "LLM call failed (%s: %s); using fallback candidate",
                type(exc).__name__,
                exc,
            )
            return self._fallback_code

    def draft(self, problem: VerifiedProblem, index: int) -> str:
        return self._complete(
            _draft_prompt(
                time_column=self._time_column,
                series_id_column=self._series_id_column,
                target_column=self._target_column,
                frequency=self._frequency,
                horizon=self._horizon,
                feature_columns=self._feature_columns,
                row_order=self._row_order,
                method_family=self._method_family,
            )
        )

    def improve(
        self, problem: VerifiedProblem, anchor: VerifiedCandidate
    ) -> str:
        evidence = anchor.record.evidence
        rmse = _observation(evidence, "rmse")
        mae = _observation(evidence, "mae")
        smape = _observation(evidence, "smape")
        return self._complete(
            _improve_prompt(
                anchor.program,
                rmse,
                mae,
                smape,
                time_column=self._time_column,
                series_id_column=self._series_id_column,
                target_column=self._target_column,
                frequency=self._frequency,
                horizon=self._horizon,
                feature_columns=self._feature_columns,
                row_order=self._row_order,
                method_family=self._method_family,
            )
        )


def _observation(evidence, name: str) -> float:
    for observation in evidence:
        if observation.name == name:
            if not math.isfinite(observation.value):
                raise ValueError(
                    f"non-finite anchor observation {name!r} cannot be used"
                )
            return observation.value
    raise ValueError(f"missing observation {name!r} in anchor evidence")

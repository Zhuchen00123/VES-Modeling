"""Sequential pattern mining data contract (R27).

Public file ``train.csv``: event sequences with columns ``sequence_id``,
``step`` (strictly increasing int per sequence) and ``event`` (canonical);
at least 10 sequences with at least 3 steps each.  Host-only file
``hidden_test_sequences.csv`` has the same shape and is never mounted.

Artifact ``patterns.json``: ``{"patterns": [{"prefix": [e, ...],
"suffix": [e, ...]}, ...]}`` — at least one pattern; prefix/suffix are
non-empty, disjoint, use train events only and are deduplicated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ves_modeling.regression.data_contract import (
    _check_no_duplicate_headers,
    _raw_headers,
)

COLUMNS = ("sequence_id", "step", "event")
LIFT_CAP = 1_000_000.0


@dataclass(frozen=True)
class SeqPatternDataContract:
    """Canonical public sequential-pattern input (never hidden truth)."""

    train_sequences: tuple[tuple[str, ...], ...] = field(repr=False)
    event_set: tuple[str, ...] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_sequences": len(self.train_sequences),
            "min_steps_per_sequence": 3,
            "event_set_size": len(self.event_set),
        }


def _canonical_scalar(value: Any, what: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} must not be boolean")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{what} must be non-empty")
        return text
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{what} must be an integer or string")
        return str(int(number))
    raise ValueError(f"{what} must be an integer or string")


def _read_sequences(
    path: Path, source: str
) -> tuple[tuple[str, ...], ...]:
    """Read event sequences (sequence_id, step, event) in row order."""
    _check_no_duplicate_headers(path, _raw_headers(path))
    frame = pd.read_csv(path)
    if list(frame.columns) != list(COLUMNS):
        raise ValueError(
            f"{source} must contain exactly columns "
            f"{COLUMNS[0]!r}, {COLUMNS[1]!r}, {COLUMNS[2]!r}"
        )
    if len(frame) == 0:
        raise ValueError(f"{source} must not be empty")
    sids = frame["sequence_id"].to_numpy()
    step_values = frame["step"].to_numpy()
    events = frame["event"].to_numpy()
    groups: dict[str, list[tuple[int, str]]] = {}
    order: list[str] = []
    for row_index in range(len(frame)):
        sid = _canonical_scalar(
            sids[row_index], f"{source} sequence_id"
        )
        step = step_values[row_index]
        if isinstance(step, (bool, np.bool_)) or not isinstance(
            step, (int, np.integer)
        ):
            raise ValueError(f"{source} 'step' must be an integer")
        event = _canonical_scalar(
            events[row_index], f"{source} event"
        )
        step = int(step)
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        steps = groups[sid]
        if steps and step <= steps[-1][0]:
            raise ValueError(
                f"{source} 'step' must be strictly increasing within "
                f"sequence {sid!r}"
            )
        steps.append((step, event))
    sequences: list[tuple[str, ...]] = []
    for sid in order:
        events = tuple(event for _, event in groups[sid])
        if not events:
            raise ValueError(f"{source} sequence {sid!r} is empty")
        sequences.append(events)
    return tuple(sequences)


def validate_seqpattern_data(public_dir: Path) -> SeqPatternDataContract:
    """Validate candidate-visible train.csv and return the contract."""
    train = _read_sequences(
        Path(public_dir) / "train.csv", "train.csv"
    )
    if len(train) < 10:
        raise ValueError("train.csv must have at least 10 sequences")
    for sequence in train:
        if len(sequence) < 3:
            raise ValueError(
                "train.csv sequences must have at least 3 steps each"
            )
    event_set = tuple(sorted({event for sequence in train for event in sequence}))
    if not event_set:
        raise ValueError("train.csv must contain at least one event")
    return SeqPatternDataContract(
        train_sequences=train, event_set=event_set
    )


def load_hidden_sequences(
    host_dir: Path,
) -> tuple[tuple[str, ...], ...]:
    """Load hidden test sequences (same shape as train.csv)."""
    hidden = _read_sequences(
        Path(host_dir) / "hidden_test_sequences.csv",
        "hidden_test_sequences.csv",
    )
    if not hidden:
        raise ValueError("hidden_test_sequences.csv must have sequences")
    return hidden


def _event_list(value: Any, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{what} must be a non-empty list")
    events: list[str] = []
    for index, entry in enumerate(value):
        events.append(_canonical_scalar(entry, f"{what}[{index}]"))
    return tuple(events)


def validate_patterns(
    payload: dict[str, Any], *, event_set: tuple[str, ...]
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Validate a patterns artifact; returns (prefix, suffix) pairs."""
    if "patterns" not in payload:
        raise ValueError("missing required field 'patterns'")
    raw = payload["patterns"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, list):
        raise ValueError("'patterns' must be a JSON array")
    if not raw:
        raise ValueError("'patterns' must contain at least one pattern")
    allowed = set(event_set)
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    patterns: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"patterns[{index}] must be an object")
        unknown = set(item) - {"prefix", "suffix"}
        if unknown:
            raise ValueError(
                f"patterns[{index}] has unknown fields: {sorted(unknown)}"
            )
        if "prefix" not in item or "suffix" not in item:
            raise ValueError(
                f"patterns[{index}] must contain 'prefix' and 'suffix'"
            )
        prefix = _event_list(item["prefix"], f"patterns[{index}].prefix")
        suffix = _event_list(item["suffix"], f"patterns[{index}].suffix")
        if set(prefix) & set(suffix):
            raise ValueError(
                f"patterns[{index}] prefix and suffix must be disjoint"
            )
        for event in prefix + suffix:
            if event not in allowed:
                raise ValueError(
                    f"patterns[{index}] event {event!r} is not in the "
                    "train event set"
                )
        key = (prefix, suffix)
        if key in seen:
            raise ValueError(f"patterns[{index}] duplicates an earlier pattern")
        seen.add(key)
        patterns.append(key)
    return patterns


def _contains_contiguous(
    sequence: tuple[str, ...], pattern: tuple[str, ...]
) -> bool:
    width = len(pattern)
    return any(
        sequence[index:index + width] == pattern
        for index in range(len(sequence) - width + 1)
    )


def evaluate_patterns(
    patterns: list[tuple[tuple[str, ...], tuple[str, ...]]],
    hidden_sequences: Iterable[tuple[str, ...]],
) -> dict[str, float]:
    """Evaluate patterns on hidden sequences; all results are finite."""
    sequences = tuple(hidden_sequences)
    total = len(sequences)
    confidences: list[float] = []
    lifts: list[float] = []
    for prefix, suffix in patterns:
        combined = prefix + suffix
        prefix_occurrences = 0
        follow_occurrences = 0
        suffix_occurrences = 0
        for sequence in sequences:
            if _contains_contiguous(sequence, prefix):
                prefix_occurrences += 1
            if _contains_contiguous(sequence, combined):
                follow_occurrences += 1
            if _contains_contiguous(sequence, suffix):
                suffix_occurrences += 1
        if prefix_occurrences == 0:
            continue
        confidence = follow_occurrences / prefix_occurrences
        p_suffix = suffix_occurrences / total if total else 0.0
        if p_suffix > 0.0:
            lift = min(confidence / p_suffix, LIFT_CAP)
        else:
            lift = 0.0
        confidences.append(float(confidence))
        lifts.append(float(lift))
    return {
        "mean_lift": float(np.mean(lifts)) if lifts else 0.0,
        "mean_confidence": (
            float(np.mean(confidences)) if confidences else 0.0
        ),
        "evaluable_pattern_count": float(len(lifts)),
        "pattern_count": float(len(patterns)),
    }

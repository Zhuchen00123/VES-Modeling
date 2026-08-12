"""Association rule mining data contract (R20).

Public file: ``train.csv`` (transaction long format: transaction_id, item;
one row per transaction-item).  Host-only file:
``hidden_test_transactions.csv`` (same long format; never mounted).

The artifact is ``rules.json``: ``{"rules": [{"antecedent": [item, ...],
"consequent": [item, ...]}, ...]}`` with at least one rule; antecedent and
consequent are non-empty, disjoint, use items declared in the train item
set, and rules are deduplicated by sorted item lists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ves_modeling.regression.data_contract import (
    _check_no_duplicate_headers,
    _id_key,
    _raw_headers,
)

DEFAULT_LIFT_CAP = 1e6


@dataclass(frozen=True)
class AssociationDataContract:
    """Public association input contract (never hidden values)."""

    transaction_id_column: str
    item_column: str
    train_rows: int
    n_transactions: int
    n_items: int
    item_set: tuple[str, ...] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id_column": self.transaction_id_column,
            "item_column": self.item_column,
            "train_rows": self.train_rows,
            "n_transactions": self.n_transactions,
            "n_items": self.n_items,
        }


def _key(value: Any) -> str:
    """Canonical item key (1, 1.0 and '1' are the same)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, str)
    ):
        raise ValueError(
            "item keys must be a scalar string or finite number, "
            f"got {type(value).__name__}"
        )
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("item keys must not be empty")
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("item keys must be finite")
    return _id_key(value)


def _validate_long_table(
    frame: pd.DataFrame,
    *,
    transaction_id_column: str,
    item_column: str,
    source: str,
) -> tuple[list[frozenset[str]], set[str]]:
    if transaction_id_column not in frame.columns:
        raise ValueError(
            f"{source} must contain transaction id column "
            f"{transaction_id_column!r}"
        )
    if item_column not in frame.columns:
        raise ValueError(
            f"{source} must contain item column {item_column!r}"
        )
    transaction_ids = frame[transaction_id_column]
    if transaction_ids.isna().any() or (
        transaction_ids.astype(str).str.strip() == ""
    ).any():
        raise ValueError(f"{source} contains empty transaction ids")
    transactions: dict[str, set[str]] = {}
    items: set[str] = set()
    for raw_tid, raw_item in zip(
        frame[transaction_id_column], frame[item_column]
    ):
        tid = _key(raw_tid)
        item = _key(raw_item)
        transactions.setdefault(tid, set()).add(item)
        items.add(item)
    if not transactions:
        raise ValueError(f"{source} must have at least one transaction")
    if any(len(item_set) < 1 for item_set in transactions.values()):
        raise ValueError(
            f"{source} every transaction must have at least one item"
        )
    return [frozenset(item_set) for item_set in transactions.values()], items


def validate_association_data(
    public_dir: Path,
    *,
    transaction_id_column: str = "transaction_id",
    item_column: str = "item",
) -> AssociationDataContract:
    """Validate the public train.csv and return the contract."""
    if not transaction_id_column.strip():
        raise ValueError("transaction_id_column must be non-empty")
    if not item_column.strip():
        raise ValueError("item_column must be non-empty")
    train_path = Path(public_dir) / "train.csv"
    _check_no_duplicate_headers(train_path, _raw_headers(train_path))
    train = pd.read_csv(train_path)
    if len(train) == 0:
        raise ValueError("train.csv must have at least one row")
    _transactions, items = _validate_long_table(
        train,
        transaction_id_column=transaction_id_column,
        item_column=item_column,
        source="train.csv",
    )
    if len(_transactions) < 2:
        raise ValueError("train.csv needs at least two transactions")
    return AssociationDataContract(
        transaction_id_column=transaction_id_column,
        item_column=item_column,
        train_rows=len(train),
        n_transactions=len(_transactions),
        n_items=len(items),
        item_set=tuple(sorted(items)),
    )


def load_hidden_transactions(
    host_dir: Path, contract: AssociationDataContract
) -> list[frozenset[str]]:
    """Load hidden test transactions (host-only, never exposed)."""
    path = Path(host_dir) / "hidden_test_transactions.csv"
    _check_no_duplicate_headers(path, _raw_headers(path))
    hidden = pd.read_csv(path)
    if len(hidden) == 0:
        raise ValueError("hidden_test_transactions.csv must have rows")
    transactions, _items = _validate_long_table(
        hidden,
        transaction_id_column=contract.transaction_id_column,
        item_column=contract.item_column,
        source="hidden_test_transactions.csv",
    )
    if len(transactions) < 1:
        raise ValueError("hidden test needs at least one transaction")
    return transactions


def _canonical_rule(antecedent: list[str], consequent: list[str]):
    return tuple(sorted(antecedent)), tuple(sorted(consequent))


def validate_rules(
    payload: dict[str, Any], contract: AssociationDataContract
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Validate a rules artifact; returns canonical (ante, conseq) rules."""
    if "rules" not in payload:
        raise ValueError("missing required field 'rules'")
    raw_rules = payload["rules"]
    if isinstance(raw_rules, (str, bytes)) or not isinstance(raw_rules, list):
        raise ValueError("'rules' must be a JSON array")
    if not raw_rules:
        raise ValueError("'rules' must contain at least one rule")
    item_set = set(contract.item_set)
    rules: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for index, entry in enumerate(raw_rules):
        if not isinstance(entry, dict):
            raise ValueError(f"rule {index} must be an object")
        if set(entry) != {"antecedent", "consequent"}:
            raise ValueError(
                f"rule {index} must contain exactly 'antecedent' and "
                "'consequent'"
            )
        antecedent_raw = entry["antecedent"]
        consequent_raw = entry["consequent"]
        if (
            not isinstance(antecedent_raw, list)
            or not isinstance(consequent_raw, list)
            or not antecedent_raw
            or not consequent_raw
        ):
            raise ValueError(
                f"rule {index} antecedent/consequent must be non-empty lists"
            )
        antecedent = [_key(value) for value in antecedent_raw]
        consequent = [_key(value) for value in consequent_raw]
        if set(antecedent) & set(consequent):
            raise ValueError(
                f"rule {index} antecedent and consequent must be disjoint"
            )
        if not set(antecedent) <= item_set or not set(consequent) <= item_set:
            raise ValueError(
                f"rule {index} references items outside the train item set"
            )
        canonical = _canonical_rule(antecedent, consequent)
        if canonical in seen:
            raise ValueError(f"rule {index} duplicates an earlier rule")
        seen.add(canonical)
        rules.append(canonical)
    return rules

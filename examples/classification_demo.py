"""VES Modeling — Classification demo (mock generator + trusted local fixture).

Usage:
  python examples/classification_demo.py --drafts 2 --improves 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ves_modeling.classification import run_classification_search

logging.basicConfig(level=logging.WARNING)


def ensure_data(root: Path) -> tuple[Path, Path]:
    public_dir = root / "data" / "classification" / "public"
    host_dir = root / "data" / "classification" / "host"
    if (public_dir / "train.csv").is_file():
        return public_dir, host_dir
    public_dir.mkdir(parents=True)
    host_dir.mkdir(parents=True)
    X, y = make_classification(
        n_samples=160,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=23,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=40, stratify=y, random_state=23
    )
    features = [f"f{i}" for i in range(4)]
    train = pd.DataFrame(X_train, columns=features)
    train["target"] = ["no" if i == 0 else "yes" for i in y_train]
    # Keep first-appearance class order == ["no", "yes"] so the explicit
    # host classes match the fixture's first-seen ordering.
    train = train.sort_values("target").reset_index(drop=True)
    test = pd.DataFrame(X_test, columns=features)
    train.to_csv(public_dir / "train.csv", index=False)
    test.to_csv(public_dir / "test_features.csv", index=False)
    pd.DataFrame(
        {"target": ["no" if i == 0 else "yes" for i in y_test]}
    ).to_csv(host_dir / "hidden_test_labels.csv", index=False)
    return public_dir, host_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--drafts", type=int, default=2)
    parser.add_argument("--improves", type=int, default=3)
    args = parser.parse_args()

    root = args.root.resolve()
    public_dir, host_dir = ensure_data(root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    result = run_classification_search(
        public_dir,
        host_dir,
        drafts=args.drafts,
        improves=args.improves,
        workspace=runs_dir,
        generator="mock",
        classes=["no", "yes"],
    )
    print("status:", result.status)
    print("best_accuracy:", result.best_accuracy)
    print("best_macro_f1:", result.best_macro_f1)
    print("best_log_loss:", result.best_log_loss)
    print("best_auroc:", result.best_auroc)
    print("run_dir:", result.run_dir)


if __name__ == "__main__":
    main()

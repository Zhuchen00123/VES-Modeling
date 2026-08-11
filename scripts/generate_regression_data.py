"""Generate the public regression dataset (fixed seed, reproducible).

Writes:
  data/regression/public/train.csv
  data/regression/public/test_features.csv
  data/regression/host/hidden_test_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


def generate(
    public_dir: Path,
    host_dir: Path,
    *,
    n_samples: int = 1500,
    n_features: int = 12,
    noise: float = 30.0,
    test_frac: float = 0.2,
    random_state: int = 42,
) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    host_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_samples)
    n_test = round(n_samples * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    feature_names = [f"feature_{i}" for i in range(n_features)]
    train = pd.DataFrame(X[train_idx], columns=feature_names)
    train["target"] = y[train_idx]
    test_features = pd.DataFrame(X[test_idx], columns=feature_names)
    hidden = pd.DataFrame({"target": y[test_idx]})

    train.to_csv(public_dir / "train.csv", index=False)
    test_features.to_csv(public_dir / "test_features.csv", index=False)
    hidden.to_csv(host_dir / "hidden_test_labels.csv", index=False)
    print(
        f"train={len(train)} test={len(test_features)} "
        f"features={n_features} noise={noise} seed={random_state}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root (default: repo root)",
    )
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--n-features", type=int, default=12)
    parser.add_argument("--noise", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    generate(
        args.root / "data" / "regression" / "public",
        args.root / "data" / "regression" / "host",
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

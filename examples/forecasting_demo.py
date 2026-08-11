"""VES Modeling — Forecasting demo (mock generator + trusted local fixture).

Usage:
  python examples/forecasting_demo.py --drafts 2 --improves 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from ves_modeling.forecasting import run_forecasting_search

logging.basicConfig(level=logging.WARNING)


def ensure_data(root: Path) -> tuple[Path, Path]:
    public_dir = root / "data" / "forecasting" / "public"
    host_dir = root / "data" / "forecasting" / "host"
    if (public_dir / "train.csv").is_file():
        return public_dir, host_dir
    public_dir.mkdir(parents=True)
    host_dir.mkdir(parents=True)
    rng = np.random.default_rng(11)
    offset = to_offset("D")
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    host_rows: list[dict] = []
    for series_id in ("a", "b"):
        slope = float(rng.normal(1.0, 0.3))
        intercept = float(rng.normal(0.0, 1.0))
        times = pd.date_range("2024-01-01", periods=20, freq=offset)
        for step, timestamp in enumerate(times):
            train_rows.append(
                {
                    "series_id": series_id,
                    "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                    "target": intercept + slope * step + rng.normal(0.0, 0.1),
                }
            )
        for index, timestamp in enumerate(
            pd.date_range(times[-1] + offset, periods=5, freq=offset)
        ):
            step = 20 + index
            value = intercept + slope * step + rng.normal(0.0, 0.1)
            iso = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            test_rows.append({"series_id": series_id, "timestamp": iso})
            host_rows.append(
                {"series_id": series_id, "timestamp": iso, "target": value}
            )
    pd.DataFrame(train_rows).to_csv(public_dir / "train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(public_dir / "test_features.csv", index=False)
    pd.DataFrame(host_rows).to_csv(host_dir / "hidden_test_labels.csv", index=False)
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
    result = run_forecasting_search(
        public_dir,
        host_dir,
        drafts=args.drafts,
        improves=args.improves,
        workspace=runs_dir,
        generator="mock",
        frequency="D",
        row_order="key",
    )
    print("status:", result.status)
    print("best_rmse:", result.best_rmse)
    print("best_mae:", result.best_mae)
    print("best_smape:", result.best_smape)
    print("run_dir:", result.run_dir)


if __name__ == "__main__":
    main()

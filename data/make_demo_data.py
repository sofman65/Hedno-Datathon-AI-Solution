"""Generate small synthetic demo files matching the HEDNO dataset schema.

The real datathon data cannot be redistributed. This script creates
`clean_data.pkl` and `clean_data_test.pkl` in the repository root so the
notebooks can be executed end-to-end as a structural smoke test.

The data is random: metrics computed on it are meaningless.

Usage:
    python data/make_demo_data.py [--train N] [--test N] [--seed S]
"""

import argparse
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES_LEN = 24  # monthly-style consumption readings per account


def make_frame(n_rows: int, theft_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    labels = (rng.random(n_rows) < theft_rate).astype(int)

    def consumption(label: int) -> list:
        base = rng.uniform(50, 400)
        series = base + rng.normal(0, base * 0.15, SERIES_LEN)
        if label == 1:
            # crude theft signature: consumption drops sharply partway through
            drop_at = rng.integers(SERIES_LEN // 3, SERIES_LEN)
            series[drop_at:] *= rng.uniform(0.05, 0.4)
        series = np.clip(series, 0, None)
        # 0 encodes a missing reading (as in the real data)
        missing = rng.random(SERIES_LEN) < 0.1
        series[missing] = 0.0
        return [round(float(v), 2) for v in series]

    # Coordinates roughly inside Greece, comma decimal separator as in the raw data
    lon = rng.uniform(21.0, 26.5, n_rows)
    lat = rng.uniform(35.0, 41.5, n_rows)

    return pd.DataFrame(
        {
            "ACCT_NBR": [f"DEMO{idx:012d}" for idx in range(n_rows)],
            "SUCCESSOR": rng.integers(1, 4, n_rows),
            "MS_METER_NBR": [f"MTR{idx:09d}" for idx in range(n_rows)],
            "BS_RATE": rng.choice(["21", "22", "31"], n_rows),
            "time_series": [consumption(lbl) for lbl in labels],
            "label": labels,
            "XRHSH": rng.choice([1.0, 2.0, 3.0], n_rows),
            "VOLTAGE": rng.choice(["LOW", "MEDIUM"], n_rows, p=[0.95, 0.05]),
            "PARNO": rng.choice([0.0, 1.0], n_rows),
            "CONTRACT_CAPACITY": rng.choice([8.0, 12.0, 15.0, 25.0], n_rows),
            "ACCT_CONTROL": rng.choice([0.0, 1.0], n_rows, p=[0.9, 0.1]),
            "ACCT_WGS84_X": [f"{v:.5f}".replace(".", ",") for v in lon],
            "ACCT_WGS84_Y": [f"{v:.5f}".replace(".", ",") for v in lat],
            "SUPPLIER": rng.choice(["S1", "S2", "S3"], n_rows),
            "SUPPLIER_TO": rng.choice(["S1", "S2", "S3"], n_rows),
            "REQUEST_TYPE": rng.choice(["A", "B", "C"], n_rows),
            "COMPL_REQUEST_STATUS": rng.choice(["OPEN", "CLOSED"], n_rows),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=2000, help="train rows")
    parser.add_argument("--test", type=int, default=400, help="test rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    theft_rate = 0.015  # rare positive class, as in the real problem

    train_path = os.path.join(REPO_ROOT, "clean_data.pkl")
    test_path = os.path.join(REPO_ROOT, "clean_data_test.pkl")

    make_frame(args.train, theft_rate, rng).to_pickle(train_path)
    make_frame(args.test, theft_rate, rng).to_pickle(test_path)

    os.makedirs(os.path.join(REPO_ROOT, "Models"), exist_ok=True)

    print(f"Wrote {train_path} ({args.train} rows)")
    print(f"Wrote {test_path} ({args.test} rows)")
    print("Synthetic data only — metrics on it are meaningless.")


if __name__ == "__main__":
    main()

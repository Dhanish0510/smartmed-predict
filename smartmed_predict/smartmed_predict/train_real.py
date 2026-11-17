"""Train the LIVE model from real baseline readings (normal operation).

Steps: run the rig normally -> collector.py saves rows -> stop it ->
       python train_real.py
"""
import argparse
import sys

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (LIVE_CONTAMINATION, LIVE_CSV_PATH, LIVE_FEATURES,
                    LIVE_MIN_ROWS, LIVE_MODEL_PATH, RANDOM_SEED)


def train_live(df):
    clean = df.dropna(subset=LIVE_FEATURES)
    model = IsolationForest(n_estimators=200,
                            contamination=LIVE_CONTAMINATION,
                            random_state=RANDOM_SEED)
    model.fit(clean[LIVE_FEATURES])
    return model, len(clean)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-simulated", action="store_true",
                    help="also use rows from collector --simulate (testing)")
    args = ap.parse_args()

    if not LIVE_CSV_PATH.exists():
        sys.exit("No data/live_readings.csv yet. Run collector.py first.")
    df = pd.read_csv(LIVE_CSV_PATH, on_bad_lines="skip")
    if not args.include_simulated:
        df = df[df["data_source"] == "esp32_live"]
    df = df.dropna(subset=LIVE_FEATURES)
    if len(df) < LIVE_MIN_ROWS:
        sys.exit(f"Only {len(df)} usable rows; need at least "
                 f"{LIVE_MIN_ROWS}. Keep the collector running longer.")

    model, n = train_live(df)
    LIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": LIVE_FEATURES,
                 "version": "v0.2-live-baseline", "n_train": n},
                LIVE_MODEL_PATH)
    print(f"Trained on {n} baseline rows -> {LIVE_MODEL_PATH}")
    print(df[LIVE_FEATURES].describe().loc[["mean", "std", "min", "max"]]
          .round(3))
    flagged = int((model.predict(df[LIVE_FEATURES]) == -1).sum())
    print(f"Baseline rows flagged: {flagged}/{n} (expected ~"
          f"{LIVE_CONTAMINATION:.0%} by design)")

"""Show the training data (for viva / external demo).

    python show_data.py                              -> project dataset (data/synthetic_sensor_data.csv)
    python show_data.py --equipment "Ventilator"     -> synthetic dataset of one hospital equipment
    python show_data.py --equipment "Ventilator" --save   -> also save it as data/<equipment>.csv (open in Excel)
"""
import argparse
import re
from pathlib import Path

import pandas as pd

import equipment as EQ
from config import DATA_PATH

pd.set_option("display.width", 160)
ap = argparse.ArgumentParser()
ap.add_argument("--equipment", help="equipment name; default = project dataset")
ap.add_argument("--save", action="store_true", help="save the equipment dataset as CSV")
args = ap.parse_args()

if args.equipment:
    if args.equipment not in EQ.NAMES:
        raise SystemExit(f"Unknown equipment '{args.equipment}'. Choose from: {', '.join(EQ.NAMES)}")
    df = EQ.synthetic_df(EQ.NAMES.index(args.equipment)); title = args.equipment
else:
    if not DATA_PATH.exists():
        raise SystemExit("Run first:  python generate_data.py")
    df = pd.read_csv(DATA_PATH); title = "Project dataset (TEST_RIG_001)"

F = ["temperature_c", "vibration_rms_g", "current_a"]
n_anom = int((df["synthetic_label"] == "synthetic_anomaly").sum())
print(f"\nTRAINING DATA: {title}")
print(f"Rows: {len(df)} | columns: {', '.join(df.columns)}")
print(f"Normal: {len(df) - n_anom} | synthetic anomalies: {n_anom} ({n_anom / len(df):.0%})")
print(f"Split 80/20 -> train {int(len(df) * 0.8)} (normal rows used for training about {int((len(df) - n_anom) * 0.8)}), test {int(len(df) * 0.2)}")
print("\nFirst 5 rows:")
print(df.head(5).to_string(index=False))
print("\nNormal readings (mean / min / max):")
print(df[df.synthetic_label == "normal"][F].agg(["mean", "min", "max"]).round(3).to_string())
print("\nInjected anomalies (mean / min / max):")
print(df[df.synthetic_label != "normal"][F].agg(["mean", "min", "max"]).round(3).to_string())
print("\nMissing values:", int(df[F].isna().sum().sum()), "| data source:", df["data_source"].iloc[0])
if args.equipment and args.save:
    Path("data").mkdir(exist_ok=True)
    out = Path("data") / (re.sub(r"[^A-Za-z0-9]+", "_", args.equipment).strip("_") + ".csv")
    df.to_csv(out, index=False); print("Saved:", out)

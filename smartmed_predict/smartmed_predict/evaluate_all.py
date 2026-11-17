"""Train and test the Isolation Forest for every hospital equipment (synthetic data).

    python evaluate_all.py                      -> table for all equipment + results/model_results.csv
    python evaluate_all.py --equipment Ventilator   -> detailed output for one equipment
"""
import argparse
from pathlib import Path

import pandas as pd

import equipment as EQ

ap = argparse.ArgumentParser()
ap.add_argument("--equipment", help="equipment name (see equipment.py), default: all")
args = ap.parse_args()

names = [args.equipment] if args.equipment else EQ.NAMES
rows = []
for name in names:
    if name not in EQ.NAMES:
        raise SystemExit(f"Unknown equipment '{name}'. Choose from: {', '.join(EQ.NAMES)}")
    idx = EQ.NAMES.index(name)
    d, m = EQ.synthetic_bundle(idx)
    n_anom = int((d["synthetic_label"] == "synthetic_anomaly").sum())
    tn = m["test_rows"] - m["tp"] - m["fp"] - m["fn"]
    rows.append({"Equipment": name, "Rows": len(d), "Anomalies": n_anom, "Train": len(d) - m["test_rows"], "Test": m["test_rows"],
                 "TP": m["tp"], "FN": m["fn"], "FP": m["fp"], "TN": tn,
                 "Precision": round(m["precision"], 3), "Recall": round(m["recall"], 3), "F1": round(m["f1"], 3)})
    if args.equipment:
        print(f"\nEquipment: {name}  (ID {EQ.CATALOG[idx]['id']}, {EQ.CATALOG[idx]['dept']})")
        print(f"Total rows: {len(d)} | train: {len(d) - m['test_rows']} | test: {m['test_rows']}")
        print("\nTEST SET RESULTS (synthetic data)")
        print(f"  Caught anomalies (TP): {m['tp']}   Missed (FN): {m['fn']}")
        print(f"  False alarms (FP):     {m['fp']}   Correct normal (TN): {tn}")
        print(f"  Precision: {m['precision']:.3f}  Recall: {m['recall']:.3f}  F1: {m['f1']:.3f}")

table = pd.DataFrame(rows)
if not args.equipment:
    print("\nTRAIN AND TEST RESULTS FOR ALL EQUIPMENT (synthetic data, 80/20 split, Isolation Forest)\n")
    print(table.to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/model_results.csv", index=False)
    print("\nSaved: results/model_results.csv")

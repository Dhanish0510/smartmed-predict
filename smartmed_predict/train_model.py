"""Train an Isolation Forest on rows labelled 'normal' (synthetic data)."""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from config import (CONTAMINATION, DATA_PATH, FEATURES, MODEL_PATH,
                    MODEL_VERSION, RANDOM_SEED)


def train(df: pd.DataFrame):
    normal = df[df["synthetic_label"] == "normal"][FEATURES].dropna()
    model = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                            random_state=RANDOM_SEED)
    model.fit(normal)
    return model, len(normal)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    # 80% train / 20% test (test rows are never seen during training)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED,
        stratify=df["synthetic_label"])

    model, n_train = train(train_df)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES,
                 "version": MODEL_VERSION}, MODEL_PATH)
    print(f"Total rows: {len(df)} | train: {len(train_df)} | "
          f"test: {len(test_df)}")
    print(f"Trained on {n_train} normal rows -> {MODEL_PATH}")

    # Evaluate on the held-out test set
    pred = model.predict(test_df[FEATURES]) == -1
    truth = (test_df["synthetic_label"] == "synthetic_anomaly").to_numpy()
    p, r, f1, _ = precision_recall_fscore_support(
        truth, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
    print("\nTEST SET RESULTS (synthetic data)")
    print(f"  Caught anomalies (TP): {tp}   Missed (FN): {fn}")
    print(f"  False alarms (FP):     {fp}   Correct normal (TN): {tn}")
    print(f"  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")

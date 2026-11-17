"""Helpers for live (ESP32) monitoring - no Streamlit needed, so testable."""
import joblib
import pandas as pd

from config import (LIVE_CSV_PATH, LIVE_FEATURES, LIVE_MODEL_PATH,
                    STALE_SECONDS)


def load_live(n=100):
    """Last n valid rows of data/live_readings.csv, or None."""
    if not LIVE_CSV_PATH.exists():
        return None
    try:
        df = pd.read_csv(LIVE_CSV_PATH, on_bad_lines="skip")
    except Exception:           # empty file / file being written
        return None
    if df.empty or "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True,
                                     errors="coerce")
    df = df.dropna(subset=["timestamp", *LIVE_FEATURES])
    if df.empty:
        return None
    return df.tail(n).reset_index(drop=True)


def is_stale(df, seconds=STALE_SECONDS, now=None):
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return (now - df["timestamp"].iloc[-1]).total_seconds() > seconds


def load_bundle():
    if not LIVE_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(LIVE_MODEL_PATH)
    except Exception:
        return None


def score(df, bundle):
    out = df.copy()
    out["anomaly_flag"] = bundle["model"].predict(out[bundle["features"]]) == -1
    out["anomaly_score"] = bundle["model"].decision_function(
        out[bundle["features"]])
    return out

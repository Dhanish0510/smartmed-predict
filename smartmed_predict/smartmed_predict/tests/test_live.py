import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LIVE_FEATURES  # noqa: E402
from live import is_stale, score  # noqa: E402
from train_real import train_live  # noqa: E402


def fake(n=400, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="s", tz="UTC")
    return pd.DataFrame({"timestamp": ts,
                         "temperature_c": rng.normal(30, 0.3, n),
                         "vibration_rms_g": np.abs(rng.normal(0.1, 0.01, n))})


def test_train_and_flag_live():
    df = fake()
    model, n = train_live(df)
    bundle = {"model": model, "features": LIVE_FEATURES}
    odd = df.tail(1).copy()
    odd["temperature_c"], odd["vibration_rms_g"] = 45.0, 0.9
    assert score(odd, bundle)["anomaly_flag"].iloc[0]


def test_stale_detection():
    df = fake(10)
    last = df["timestamp"].iloc[-1]
    assert not is_stale(df, 10, now=last + pd.Timedelta(seconds=3))
    assert is_stale(df, 10, now=last + pd.Timedelta(seconds=30))

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FEATURES  # noqa: E402
from generate_data import generate  # noqa: E402
from train_model import train  # noqa: E402

EXPECTED_COLS = ["timestamp", "equipment_id", "temperature_c",
                 "vibration_rms_g", "current_a", "synthetic_label",
                 "data_source"]


def test_schema():                      # T01
    df = generate(200)
    assert list(df.columns) == EXPECTED_COLS


def test_finite_values():               # T02
    df = generate(200)
    assert np.isfinite(df[FEATURES].to_numpy()).all()


def test_train_and_predict():           # T03
    df = generate(500)
    model, n = train(df)
    assert n > 0
    assert set(model.predict(df[FEATURES])) <= {-1, 1}


def test_reproducible():                # T09
    assert generate(100, seed=1).equals(generate(100, seed=1))


def test_obvious_anomaly_flagged():
    df = generate(1000)
    model, _ = train(df)
    import pandas as pd
    odd = pd.DataFrame([[48.0, 0.55, 0.70]], columns=FEATURES)
    assert model.predict(odd)[0] == -1

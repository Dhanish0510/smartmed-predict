"""Create reproducible SYNTHETIC sensor readings (1-minute interval)."""
import numpy as np
import pandas as pd

from config import (ANOMALY_FRACTION, DATA_PATH, EQUIPMENT_ID, N_ROWS,
                    RANDOM_SEED)


def generate(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="min")

    temperature = rng.normal(37.0, 0.8, n_rows)
    vibration = np.abs(rng.normal(0.12, 0.02, n_rows))
    current = rng.normal(0.31, 0.03, n_rows)

    label = np.array(["normal"] * n_rows, dtype=object)
    n_anom = int(n_rows * ANOMALY_FRACTION)
    idx = rng.choice(n_rows, size=n_anom, replace=False)

    # Intentionally artificial abnormal patterns
    temperature[idx] += rng.uniform(4, 9, n_anom)
    vibration[idx] += rng.uniform(0.15, 0.40, n_anom)
    current[idx] += rng.uniform(0.15, 0.40, n_anom)
    label[idx] = "synthetic_anomaly"

    return pd.DataFrame({
        "timestamp": timestamps,
        "equipment_id": EQUIPMENT_ID,
        "temperature_c": temperature.round(3),
        "vibration_rms_g": vibration.round(4),
        "current_a": current.round(4),
        "synthetic_label": label,
        "data_source": "synthetic",
    })


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} rows -> {DATA_PATH}")
    print(df["synthetic_label"].value_counts())

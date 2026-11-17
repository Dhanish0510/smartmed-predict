"""Shared settings so every script uses the same names and paths."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "synthetic_sensor_data.csv"
MODEL_PATH = BASE_DIR / "models" / "isolation_forest.joblib"

FEATURES = ["temperature_c", "vibration_rms_g", "current_a"]
RANDOM_SEED = 42
N_ROWS = 10000
ANOMALY_FRACTION = 0.05
EQUIPMENT_ID = "TEST_RIG_001"
CONTAMINATION = 0.05  # expected outlier share; NOT a real failure rate
MODEL_VERSION = "v0.1-synthetic"

# ---- Live (real hardware) mode ----
LIVE_CSV_PATH = BASE_DIR / "data" / "live_readings.csv"
LIVE_MODEL_PATH = BASE_DIR / "models" / "live_model.joblib"
LIVE_FEATURES = ["temperature_c", "vibration_rms_g"]  # no current sensor yet
LIVE_CONTAMINATION = 0.02
LIVE_MIN_ROWS = 300
STALE_SECONDS = 10

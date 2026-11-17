"""Validate one JSON line from the ESP32 (no hardware needed to test)."""
import json
import math

# Plausible physical limits (DS18B20 / ADXL345 +-16 g class sensors)
LIMITS = {
    "temperature_c": (-20.0, 100.0),
    "vibration_rms_g": (0.0, 16.0),
    "current_a": (0.0, 30.0),
}
REQUIRED = ["equipment_id", "temperature_c", "vibration_rms_g",
            "sensor_status"]


def _num(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def validate_line(line):
    """Return (record, error). Exactly one of them is None."""
    try:
        data = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None, "malformed_json"
    if not isinstance(data, dict):
        return None, "not_an_object"
    if "info" in data:
        return None, "info_message"

    for field in REQUIRED:
        if field not in data:
            return None, f"missing_field:{field}"

    status = data["sensor_status"]
    if status != "ok":
        return None, f"sensor_fault:{status}"   # data-quality event

    for field in ("temperature_c", "vibration_rms_g"):
        if not _num(data[field]):
            return None, f"non_numeric:{field}"
    current = data.get("current_a")
    if current is not None and not _num(current):
        return None, "non_numeric:current_a"

    for field, (lo, hi) in LIMITS.items():
        v = data.get(field)
        if v is not None and not (lo <= v <= hi):
            return None, f"out_of_range:{field}"

    return {
        "seq": data.get("seq"),
        "equipment_id": str(data["equipment_id"]),
        "temperature_c": float(data["temperature_c"]),
        "vibration_rms_g": float(data["vibration_rms_g"]),
        "current_a": None if current is None else float(current),
    }, None

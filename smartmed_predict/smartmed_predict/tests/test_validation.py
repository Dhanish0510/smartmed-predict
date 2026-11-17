import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import validate_line  # noqa: E402


def good(**over):
    d = {"seq": 1, "equipment_id": "TEST_RIG_001", "temperature_c": 37.4,
         "vibration_rms_g": 0.12, "current_a": None, "sensor_status": "ok"}
    d.update(over)
    return json.dumps(d)


def test_valid_record():
    rec, err = validate_line(good())
    assert err is None and rec["temperature_c"] == 37.4


def test_malformed_json():                 # T07
    assert validate_line("{oops")[1] == "malformed_json"


def test_missing_field():                  # T07
    d = json.loads(good()); del d["vibration_rms_g"]
    assert validate_line(json.dumps(d))[1] == "missing_field:vibration_rms_g"


def test_sensor_fault_is_data_quality_event():
    assert validate_line(good(sensor_status="temp_error",
                              temperature_c=None))[1] == "sensor_fault:temp_error"


def test_out_of_range():
    assert validate_line(good(temperature_c=-127))[1] == "out_of_range:temperature_c"


def test_non_numeric():
    assert validate_line(good(vibration_rms_g="abc"))[1] == "non_numeric:vibration_rms_g"

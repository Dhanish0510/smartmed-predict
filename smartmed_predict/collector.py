"""Read ESP32 JSON lines over USB serial, validate, append to CSV.

  python collector.py --port COM5          # real ESP32
  python collector.py --simulate           # no hardware (tests the pipeline)
"""
import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from validation import validate_line

BASE = Path(__file__).parent
LIVE_CSV = BASE / "data" / "live_readings.csv"
DQ_LOG = BASE / "logs" / "data_quality.log"
COLUMNS = ["timestamp", "equipment_id", "seq", "temperature_c",
           "vibration_rms_g", "current_a", "data_source"]
STALE_AFTER_S = 5


def simulated_lines(interval=1.0):
    """Fake ESP32 output, including bad lines to exercise error handling."""
    seq = 0
    while True:
        r = random.random()
        if r < 0.05:
            yield "{bad json"
        elif r < 0.10:
            yield json.dumps({"seq": seq, "equipment_id": "TEST_RIG_001",
                              "temperature_c": None, "vibration_rms_g": 0.1,
                              "current_a": None, "sensor_status": "temp_error"})
        else:
            yield json.dumps({
                "seq": seq, "equipment_id": "TEST_RIG_001",
                "temperature_c": round(random.gauss(37, 0.5), 2),
                "vibration_rms_g": round(abs(random.gauss(0.12, 0.02)), 4),
                "current_a": None, "sensor_status": "ok"})
        seq += 1
        time.sleep(interval)


def serial_lines(port, baud):
    import serial  # pip install pyserial
    while True:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                print(f"Connected to {port} @ {baud}")
                while True:
                    raw = ser.readline()
                    yield raw.decode("utf-8", errors="replace").strip() or None
        except serial.SerialException as exc:
            print(f"Serial problem: {exc}. Retrying in 3 s ...")
            time.sleep(3)


def log_dq(reason, line):
    DQ_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(DQ_LOG, "a", encoding="utf-8") as f:
        f.write(f"{stamp}\t{reason}\t{line}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between simulated readings")
    args = ap.parse_args()
    if not args.simulate and not args.port:
        ap.error("give --port COMx or use --simulate")

    source = "simulated" if args.simulate else "esp32_live"
    lines = simulated_lines(args.interval) if args.simulate else serial_lines(args.port, args.baud)

    LIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LIVE_CSV.exists()
    last_ok, stale_shown, accepted, rejected = time.time(), False, 0, 0

    with open(LIVE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        try:
            for line in lines:
                if line is None:                       # serial timeout
                    if time.time() - last_ok > STALE_AFTER_S and not stale_shown:
                        print("SENSOR OFFLINE / STALE: no data received")
                        stale_shown = True
                    continue
                record, error = validate_line(line)
                if error:
                    rejected += 1
                    log_dq(error, line)
                    print(f"[rejected] {error}")
                    continue
                record["timestamp"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds")
                record["data_source"] = source
                writer.writerow({k: record.get(k) for k in COLUMNS})
                f.flush()
                last_ok, stale_shown, accepted = time.time(), False, accepted + 1
                print(f"[ok #{accepted}] T={record['temperature_c']} C  "
                      f"vib={record['vibration_rms_g']} g")
        except KeyboardInterrupt:
            print(f"\nStopped. Accepted {accepted}, rejected {rejected}.")


if __name__ == "__main__":
    main()

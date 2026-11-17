"""Simulated hospital-equipment profiles for the dashboard demo.

Every number here is an ILLUSTRATIVE demo profile, not a measurement of a real
device. Real readings arrive only from the ESP32 test rig (Live mode).
"""
import io
import wave

import numpy as np
from sklearn.ensemble import IsolationForest

# name, department, id, normal temp (C), vibration RMS (g), current RMS (A), motor/fan frequency (Hz)
CATALOG = [
    dict(name="Patient Monitoring System", dept="ICU / Ward", id="PM-01", temp=36.0, vib=0.020, cur=0.55, f0=48.0),
    dict(name="Infusion Pump", dept="Ward", id="IP-02", temp=33.0, vib=0.060, cur=0.25, f0=22.0),
    dict(name="Syringe Pump", dept="ICU", id="SP-03", temp=32.0, vib=0.050, cur=0.20, f0=18.0),
    dict(name="Ventilator", dept="ICU", id="VT-04", temp=40.0, vib=0.150, cur=1.50, f0=35.0),
    dict(name="Anaesthesia Machine", dept="Operation Theatre", id="AM-05", temp=35.0, vib=0.060, cur=1.00, f0=30.0),
    dict(name="Defibrillator", dept="Emergency", id="DF-06", temp=30.0, vib=0.030, cur=0.40, f0=40.0),
    dict(name="ECG Machine", dept="Cardiology", id="EC-07", temp=31.0, vib=0.020, cur=0.30, f0=45.0),
    dict(name="Ultrasound Scanner", dept="Radiology", id="US-08", temp=38.0, vib=0.080, cur=2.00, f0=42.0),
    dict(name="X-Ray Machine", dept="Radiology", id="XR-09", temp=42.0, vib=0.100, cur=4.00, f0=28.0),
    dict(name="Suction Machine", dept="Operation Theatre", id="SM-10", temp=41.0, vib=0.300, cur=1.80, f0=25.0),
    dict(name="Oxygen Concentrator", dept="Ward", id="OC-11", temp=45.0, vib=0.350, cur=1.90, f0=29.5),
    dict(name="Dialysis Machine", dept="Nephrology", id="DM-12", temp=39.0, vib=0.180, cur=3.00, f0=33.0),
    dict(name="Laboratory Centrifuge", dept="Laboratory", id="LC-13", temp=36.0, vib=0.450, cur=2.50, f0=50.0),
    dict(name="Autoclave (Sterilizer)", dept="CSSD", id="AC-14", temp=55.0, vib=0.050, cur=5.00, f0=20.0),
]
SIGNATURE = {'PM-01': {'vh': [1.0, 0.25, 0.1], 'vn': 0.35, 'vp': 0.0, 'vi': 0.0, 'ct': 'smps'}, 'IP-02': {'vh': [1.0, 0.5, 0.3], 'vn': 0.15, 'vp': 4.0, 'vi': 0.0, 'ct': 'smps'}, 'SP-03': {'vh': [1.0, 0.4, 0.2], 'vn': 0.15, 'vp': 2.0, 'vi': 0.0, 'ct': 'smps'}, 'VT-04': {'vh': [1.0, 0.15, 0.08, 0.05, 0.35], 'vn': 0.1, 'vp': 0.0, 'vi': 0.0, 'ct': 'motor'}, 'AM-05': {'vh': [1.0, 0.3, 0.1], 'vn': 0.2, 'vp': 0.0, 'vi': 0.0, 'ct': 'smps'}, 'DF-06': {'vh': [1.0, 0.1], 'vn': 0.55, 'vp': 0.0, 'vi': 0.0, 'ct': 'smps'}, 'EC-07': {'vh': [1.0, 0.1], 'vn': 0.5, 'vp': 0.0, 'vi': 0.0, 'ct': 'smps'}, 'US-08': {'vh': [1.0, 0.2, 0.1], 'vn': 0.25, 'vp': 0.0, 'vi': 0.0, 'ct': 'smps'}, 'XR-09': {'vh': [0.2, 1.0, 0.1, 0.45], 'vn': 0.08, 'vp': 0.0, 'vi': 0.0, 'ct': 'motor'}, 'SM-10': {'vh': [0.6, 1.0, 0.5], 'vn': 0.15, 'vp': 0.0, 'vi': 1.2, 'ct': 'motor'}, 'OC-11': {'vh': [0.7, 1.0, 0.4], 'vn': 0.15, 'vp': 1.5, 'vi': 0.6, 'ct': 'motor'}, 'DM-12': {'vh': [1.0, 0.6, 0.3], 'vn': 0.15, 'vp': 3.0, 'vi': 0.0, 'ct': 'motor'}, 'LC-13': {'vh': [1.0, 0.05], 'vn': 0.04, 'vp': 0.0, 'vi': 0.0, 'ct': 'motor'}, 'AC-14': {'vh': [1.0, 0.3], 'vn': 0.4, 'vp': 0.0, 'vi': 0.0, 'ct': 'resistive'}}
NAMES = [c["name"] for c in CATALOG]
FAULTS = ["None", "Overheating", "Bearing wear / imbalance", "Overload (high current)", "Combined fault"]
# multipliers / offsets at full fault: (temp offset C, vibration x, current x, bearing-impulse strength)
FAULT_EFFECT = {
    "None": (0.0, 1.0, 1.0, 0.0),
    "Overheating": (12.0, 1.60, 1.25, 0.0),
    "Bearing wear / imbalance": (3.0, 3.5, 1.10, 1.0),
    "Overload (high current)": (5.0, 1.30, 1.70, 0.0),
    "Combined fault": (12.0, 3.5, 1.70, 1.0),
}
N = 3000          # length of the repeating normal series
TRAIN_N = 1500    # normal samples used to train each equipment model
FS_VIB, N_VIB = 800, 400       # vibration waveform: 0.5 s at 800 Hz
FS_CUR, N_CUR = 10000, 600     # current waveform: 60 ms at 10 kHz (3 cycles of 50 Hz)

_cache = {}
def base_series(idx):
    if idx in _cache: return _cache[idx]
    c = CATALOG[idx]; rng = np.random.default_rng(1000 + idx); t = np.arange(N)
    temp = c["temp"] + 0.6 * np.sin(2 * np.pi * t / 90) + 0.15 * rng.standard_normal(N)
    vib = np.abs(c["vib"] * (1 + 0.05 * np.sin(2 * np.pi * t / 40)) + 0.06 * c["vib"] * rng.standard_normal(N))
    cur = c["cur"] * (1 + 0.03 * np.sin(2 * np.pi * t / 70)) + 0.02 * c["cur"] * rng.standard_normal(N)
    _cache[idx] = (temp, vib, cur); return _cache[idx]

def train_model(idx):
    temp, vib, cur = base_series(idx)
    X = np.column_stack([temp, vib, cur])[:TRAIN_N]
    return IsolationForest(n_estimators=100, contamination=0.02, random_state=42).fit(X)

def ramp_at(tick, fault_start, length=15):
    return float(np.clip((tick - fault_start) / length, 0.0, 1.0)) if fault_start is not None else 0.0

def window(idx, tick, fault="None", fault_start=None, n=120):
    """Last n readings ending at `tick`, with the chosen fault ramped in."""
    temp, vib, cur = base_series(idx); ix = (np.arange(tick - n + 1, tick + 1)) % N
    dt, vm, cm, _ = FAULT_EFFECT[fault]
    r = np.array([ramp_at(t, fault_start) for t in range(tick - n + 1, tick + 1)]) if fault != "None" else np.zeros(n)
    return dict(temp=temp[ix] + dt * r, vib=vib[ix] * (1 + (vm - 1) * r), cur=cur[ix] * (1 + (cm - 1) * r), ramp=r)

def flags(model, w):
    return model.predict(np.column_stack([w["temp"], w["vib"], w["cur"]])) == -1

def status(fl):
    """Unusual only when the last 3 readings are all flagged (avoids false flicker)."""
    return "Unusual" if len(fl) >= 3 and bool(np.all(fl[-3:])) else "Normal"

def vib_wave(idx, tick, vrms, bearing=0.0):
    """Vibration waveform with this equipment's own signature (harmonics, noise, pulsing, impulses)."""
    c = CATALOG[idx]; g = SIGNATURE[c["id"]]; rng = np.random.default_rng(tick * 7 + idx); t = np.arange(N_VIB) / FS_VIB
    x = sum(a * np.sin(2 * np.pi * (k + 1) * c["f0"] * t + 0.7 * k) for k, a in enumerate(g["vh"])) + g["vn"] * rng.standard_normal(N_VIB)
    if g["vp"] > 0:
        x = x * (0.35 + 0.65 * (np.sin(2 * np.pi * g["vp"] * t) > 0))
    if g["vi"] > 0:
        x = x + g["vi"] * (np.sin(2 * np.pi * c["f0"] * t + 0.3) > 0.96)
    if bearing > 0:
        x = x + bearing * (2.4 * (np.sin(2 * np.pi * 97 * t) > 0.985) + 0.3 * np.sin(2 * np.pi * 160 * t)) * np.sqrt(np.mean(x ** 2)) / 1.0
    x = x / np.sqrt(np.mean(x ** 2)) * vrms
    return t * 1000.0, x

def cur_wave(idx, tick, irms, overload=0.0):
    """Mains current waveform shaped by the load type: motor, switch-mode supply (peaky) or resistive heater."""
    c = CATALOG[idx]; kind = SIGNATURE[c["id"]]["ct"]; rng = np.random.default_rng(tick * 11 + idx); t = np.arange(N_CUR) / FS_CUR
    base = np.sin(2 * np.pi * 50 * t)
    if kind == "smps":
        i = np.where(np.abs(base) > 0.72, base, 0.10 * base)
    elif kind == "motor":
        i = base + 0.10 * np.sin(2 * np.pi * 150 * t + 0.3) + 0.05 * np.sin(2 * np.pi * 250 * t)
    else:
        i = base
    i = i + overload * 0.18 * np.sin(2 * np.pi * 150 * t + 0.5) + 0.02 * rng.standard_normal(N_CUR)
    i = i / np.sqrt(np.mean(i ** 2)) * irms
    return t * 1000.0, i

def step_events(events, sched, rng, tick, n_equipment):
    """Random demo faults: one equipment at a time gets a fault for 25-40 s, then recovers."""
    for j in [j for j, ev in events.items() if tick > ev["end"]]:
        del events[j]
    if tick >= sched["next"] and not events:
        j = int(rng.integers(n_equipment)); f = FAULTS[1 + int(rng.integers(len(FAULTS) - 1))]
        dur = int(rng.integers(25, 41))
        events[j] = dict(fault=f, start=tick, end=tick + dur)
        sched["next"] = tick + dur + int(rng.integers(25, 46))

def make_beep_wav(tick=0, seconds=0.7, fs=8000):
    """Two-tone alarm beep as WAV bytes (tick changes 1 sample so the browser re-plays it)."""
    t = np.arange(int(fs * seconds)) / fs
    tone = np.where((t // 0.175) % 2 == 0, np.sin(2 * np.pi * 1200 * t), np.sin(2 * np.pi * 900 * t))
    pcm = (tone * 0.6 * 32767).astype("<i2"); pcm[0] = tick % 7
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs); w.writeframes(pcm.tobytes())
    return buf.getvalue()


SYN_N = 10000
def synthetic_df(idx, n=SYN_N, seed=42):
    """Synthetic dataset for one equipment: normal readings around its profile + 5% injected anomalies."""
    import pandas as pd
    c = CATALOG[idx]; rng = np.random.default_rng(seed + idx)
    temp = rng.normal(c["temp"], 0.8, n); vib = np.abs(rng.normal(c["vib"], 0.17 * c["vib"], n)); cur = rng.normal(c["cur"], 0.10 * c["cur"], n)
    label = np.array(["normal"] * n, dtype=object); k = int(n * 0.05); ix = rng.choice(n, k, replace=False)
    temp[ix] += rng.uniform(4, 9, k); vib[ix] += c["vib"] * rng.uniform(1.2, 3.3, k); cur[ix] += c["cur"] * rng.uniform(0.5, 1.3, k)
    label[ix] = "synthetic_anomaly"
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="min"), "equipment_id": c["id"],
                         "temperature_c": temp.round(3), "vibration_rms_g": vib.round(4), "current_a": cur.round(4),
                         "synthetic_label": label, "data_source": "synthetic"})

def synthetic_bundle(idx):
    """Train an Isolation Forest on normal rows (80/20 split), score all rows, return (dataframe, test metrics)."""
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    d = synthetic_df(idx); cols = ["temperature_c", "vibration_rms_g", "current_a"]
    tr, te = train_test_split(d, test_size=0.2, random_state=42, stratify=d["synthetic_label"])
    m = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(tr[tr.synthetic_label == "normal"][cols])
    pred = m.predict(te[cols]) == -1; truth = (te["synthetic_label"] == "synthetic_anomaly").to_numpy()
    p, r, f1, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
    d["anomaly_flag"] = m.predict(d[cols]) == -1; d["anomaly_score"] = m.decision_function(d[cols])
    return d, dict(precision=float(p), recall=float(r), f1=float(f1), test_rows=len(te), tp=int((pred & truth).sum()),
                   fp=int((pred & ~truth).sum()), fn=int((~pred & truth).sum()))


STRONG_SCORE = -0.10   # synthetic replay: alarm sound only for strong anomalies (decision score below this)

def alarm_due(active, new, armed, tick, last_beep, repeat=4):
    """Beep for a NEW alarm at once, then repeat every `repeat` readings while it stays active.
    No sound while 'armed' is False (just after selecting another equipment)."""
    return bool(active) and armed and (bool(new) or tick - last_beep >= repeat)

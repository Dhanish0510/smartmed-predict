import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import equipment as E


def test_each_equipment_has_own_values():
    assert len({(c["vib"], c["cur"]) for c in E.CATALOG}) == len(E.CATALOG)


def test_normal_window_mostly_normal():
    for i in (0, 3, 12):
        m = E.train_model(i); w = E.window(i, 2000)
        assert E.flags(m, w).mean() < 0.08


def test_every_fault_is_flagged():
    for fault in E.FAULTS[1:]:
        for i in (0, 9):
            m = E.train_model(i); w = E.window(i, 2000, fault, fault_start=1960)
            assert E.status(E.flags(m, w)) == "Unusual", (fault, i)


def test_waveform_rms_matches_value():
    t, x = E.vib_wave(0, 5, 0.25, bearing=1.0)
    assert abs(np.sqrt(np.mean(x ** 2)) - 0.25) < 1e-9
    t, i = E.cur_wave(0, 5, 1.2)
    assert abs(np.sqrt(np.mean(i ** 2)) - 1.2) < 1e-9


def test_random_events_start_and_end():
    rng = np.random.default_rng(1); events = {}; sched = {"next": 5}; started = 0
    for tick in range(0, 600):
        before = dict(events)
        E.step_events(events, sched, rng, tick, len(E.CATALOG))
        assert len(events) <= 1
        started += sum(1 for j in events if j not in before)
    assert started >= 5


def test_beep_is_valid_wav():
    import io, wave
    w = wave.open(io.BytesIO(E.make_beep_wav(3)), "rb")
    assert w.getnchannels() == 1 and w.getnframes() > 4000


def test_synthetic_bundle_per_equipment():
    for i in (0, 3, 13):
        d, met = E.synthetic_bundle(i)
        assert len(d) == E.SYN_N and met["test_rows"] == 2000
        assert met["recall"] > 0.95 and met["precision"] > 0.4
        c = E.CATALOG[i]; assert abs(d[d.synthetic_label == "normal"].temperature_c.mean() - c["temp"]) < 0.2


def test_each_equipment_has_its_own_waveform_signature():
    sigs = {repr(sorted(E.SIGNATURE[c["id"]].items())) for c in E.CATALOG}
    assert len(sigs) == len(E.CATALOG)
    shapes = []
    for i in range(len(E.CATALOG)):
        tt, x = E.vib_wave(i, 10, 1.0); assert abs(np.sqrt(np.mean(x ** 2)) - 1.0) < 1e-9
        tc, y = E.cur_wave(i, 10, 1.0); assert abs(np.sqrt(np.mean(y ** 2)) - 1.0) < 1e-9
        shapes.append((round(float(np.max(np.abs(x))), 2), round(float(np.max(np.abs(y))), 2)))
    assert len(set(shapes)) >= 12


def test_alarm_due_rules():
    assert not E.alarm_due([], [], True, 10, 0)                 # nothing active
    assert not E.alarm_due([3], [3], False, 10, 0)              # not armed (just selected)
    assert E.alarm_due([3], [3], True, 10, 9)                   # new alarm: beep at once
    assert not E.alarm_due([3], [], True, 10, 8)                # already active, too soon
    assert E.alarm_due([3], [], True, 12, 8)                    # repeat after 4 readings

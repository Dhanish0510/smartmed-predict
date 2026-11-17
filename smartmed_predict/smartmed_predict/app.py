"""SmartMed Predict - dashboard (synthetic data prototype)."""
import sys
import threading
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import equipment as EQ
import live
from config import DATA_PATH, FEATURES, LIVE_FEATURES, MODEL_PATH

try:
    import winsound
except ImportError:
    winsound = None

st.set_page_config(page_title="SmartMed Predict", page_icon="🩺",
                   layout="wide")

NAVY, GREEN, RED = "#0b3b7a", "#1f9d55", "#d64545"
WINDOW = 100  # readings shown in the trend chart

st.markdown(f"""
<style>
.block-container {{ padding-top: 4.5rem; max-width: 1100px; }}
.sm-header {{ background:{NAVY}; color:#fff; border-radius:12px;
  padding:14px 20px; display:flex; align-items:center; gap:14px; }}
.sm-header .t {{ font-size:1.35rem; font-weight:700; line-height:1.2; }}
.sm-header .s {{ font-size:.85rem; opacity:.85; }}
.sm-card {{ background:#fff; border:1px solid #d5deeb; border-radius:10px;
  padding:12px 16px; height:92px; }}
.sm-label {{ font-size:.8rem; color:#5b6b82; }}
.sm-value {{ font-size:1.6rem; font-weight:600; color:#10223d; }}
.sm-status {{ display:flex; align-items:center; gap:12px; }}
.sm-dot {{ width:38px; height:38px; border-radius:50%; color:#fff;
  display:flex; align-items:center; justify-content:center;
  font-size:1.3rem; font-weight:700; }}
.sm-panel-title {{ font-weight:700; color:{NAVY}; margin-bottom:6px; }}
.sm-ok {{ text-align:center; color:#5b6b82; padding:36px 0; }}
.sm-ok .c {{ width:44px; height:44px; border-radius:50%;
  border:2px solid {GREEN}; color:{GREEN}; margin:0 auto 8px auto;
  display:flex; align-items:center; justify-content:center;
  font-size:1.4rem; font-weight:700; }}
.sm-eq {{ background:#fff; border:1px solid #d5deeb; border-left:6px solid {NAVY}; border-radius:10px;
  padding:12px 18px; display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
.sm-eq .n {{ font-size:1.35rem; font-weight:700; color:{NAVY}; }}
.sm-eq .d {{ font-size:.85rem; color:#5b6b82; }}
.sm-eq .b {{ background:#fff4d6; color:#8a6100; font-size:.7rem; font-weight:700; padding:4px 10px; border-radius:20px; }}
.sm-alarm {{ background:#d64545; color:#fff; border-radius:10px; padding:12px 18px; font-weight:700; font-size:1.05rem;
  margin-bottom:12px; animation: smblink 1s infinite; }}
.sm-allok {{ background:#d8f0df; color:#17653a; border-radius:10px; padding:8px 18px; font-weight:600; margin-bottom:12px; }}
.sm-eq .live {{ background:#d8f0df; color:#17653a; margin-right:8px; animation: smpulse 1.4s infinite; }}
.sm-eq .paused {{ background:#e6e9ee; color:#5b6b82; margin-right:8px; }}
@keyframes smblink {{ 50% {{ opacity:.55; }} }}
@keyframes smpulse {{ 50% {{ opacity:.45; }} }}
.sm-note {{ background:#e4eefb; border-radius:10px; padding:10px 16px;
  color:{NAVY}; font-size:.85rem; margin-top:10px; }}
</style>
""", unsafe_allow_html=True)

HEART = ("<svg width='40' height='40' viewBox='0 0 24 24' fill='none' "
         "stroke='white' stroke-width='1.8' stroke-linecap='round' "
         "stroke-linejoin='round'><path d='M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 "
         "5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12 21l7.8-7.5 1-1.1a5.5 "
         "5.5 0 0 0 0-7.8z'/><path d='M5 12h3l2-3 3 6 2-3h4'/></svg>")
st.markdown(f"<div class='sm-header'>{HEART}<div><div class='t'>SmartMed "
            "Predict</div><div class='s'>Predictive Maintenance Dashboard"
            "</div></div></div>", unsafe_allow_html=True)
st.write("")

HAVE_SYN = DATA_PATH.exists() and MODEL_PATH.exists()


@st.cache_data
def load_scored():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    df["anomaly_flag"] = model.predict(df[bundle["features"]]) == -1
    df["anomaly_score"] = model.decision_function(df[bundle["features"]])
    return df, bundle


if HAVE_SYN:
    df, bundle = load_scored()
    n = len(df)
else:
    df = bundle = None
    n = 0

# ---- sidebar controls ----
st.sidebar.header("Controls")
def _reset_fault():
    st.session_state["fault_sel"] = "None"   # a new equipment starts healthy


EQ_MODE, LIVE_MODE, SYN_MODE = "Hospital equipment (demo)", "Live ESP32 (test rig)", "Synthetic replay"
mode = st.sidebar.radio("Data source", [EQ_MODE, LIVE_MODE, SYN_MODE])
SYN_PROJECT = "Test rig (project dataset)"
syn_choice = SYN_PROJECT
replay, pos = False, n
eq_name, fault, eq_live, rand_events, alarm_on, test_alarm = EQ.NAMES[0], "None", False, False, False, False
if mode == EQ_MODE:
    eq_name = st.sidebar.selectbox("Select equipment", EQ.NAMES, key="eq_sel", on_change=_reset_fault)
    fault = st.sidebar.selectbox("Simulate fault", EQ.FAULTS, key="fault_sel")
    eq_live = st.sidebar.toggle("Live monitoring (1 reading / sec)", value=True)
    rand_events = st.sidebar.toggle("Random fault events (demo)", value=True)
    alarm_on = st.sidebar.toggle("Alarm sound", value=True)
    test_alarm = st.sidebar.button("Test alarm sound")
elif mode == SYN_MODE:
    syn_choice = st.sidebar.selectbox("Select equipment", [SYN_PROJECT] + EQ.NAMES, index=1)
    replay = st.sidebar.toggle("Auto replay (1 reading / sec)", value=True)
    alarm_on = st.sidebar.toggle("Alarm sound", value=True)
    test_alarm = st.sidebar.button("Test alarm sound")
    if not replay:
        pos = st.sidebar.slider("Reading position", WINDOW, n if syn_choice == SYN_PROJECT else EQ.SYN_N, n if syn_choice == SYN_PROJECT else EQ.SYN_N)
if mode == SYN_MODE and syn_choice == SYN_PROJECT and not HAVE_SYN:
    st.error("Missing data/model. Run: python generate_data.py  then  python train_model.py")
    st.stop()
if test_alarm:
    st.session_state["test_pending"] = True
st.sidebar.caption("Alarm sound: new alarms beep at once, then every 4 s while active (synthetic replay: strong anomalies only).")
st.sidebar.caption("Hospital equipment = simulated demo profiles. Live = real readings "
                   "from collector.py. Synthetic = generated dataset replay for the selected equipment.")
if "replay_idx" not in st.session_state:
    st.session_state.replay_idx = WINDOW

STATES = {
    "normal":   (GREEN, "✓", "Normal"),
    "unusual":  (RED, "!", "Unusual"),
    "offline":  ("#7a8699", "–", "Sensor offline"),
    "baseline": ("#2e6fd8", "…", "No model yet"),
}


def card(label, value):
    return (f"<div class='sm-card'><div class='sm-label'>{label}</div>"
            f"<div class='sm-value'>{value}</div></div>")


def board(w, feats, state):
    last = w.iloc[-1]
    has_current = "current_a" in feats

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    c1.markdown(card("Temperature", f"{last['temperature_c']:.1f} °C"),
                unsafe_allow_html=True)
    c2.markdown(card("Vibration", f"{last['vibration_rms_g']:.2f} g"),
                unsafe_allow_html=True)
    c3.markdown(card("Current", f"{last['current_a']:.2f} A" if has_current
                     else "N/A"), unsafe_allow_html=True)
    color, mark, text = STATES[state]
    c4.markdown(
        f"<div class='sm-card'><div class='sm-label'>Status</div>"
        f"<div class='sm-status'><div class='sm-dot' style='background:{color}'>"
        f"{mark}</div><div class='sm-value' style='font-size:1.15rem'>{text}"
        f"</div></div></div>", unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([2, 1])
    with left:
        with st.container(border=True):
            st.markdown("<div class='sm-panel-title'>Sensor Trends</div>",
                        unsafe_allow_html=True)
            x = list(range(len(w)))
            fig = go.Figure()
            fig.add_scatter(x=x, y=w["temperature_c"], name="Temperature (°C)",
                            line=dict(color="#e0408a", width=1.6))
            fig.add_scatter(x=x, y=w["vibration_rms_g"], name="Vibration (g)",
                            yaxis="y2", line=dict(color="#17a589", width=1.6))
            if has_current:
                fig.add_scatter(x=x, y=w["current_a"], name="Current (A)",
                                yaxis="y2", line=dict(color="#2e6fd8", width=1.6))
            fl = w.reset_index(drop=True)
            fl = fl[fl["anomaly_flag"]]
            fig.add_scatter(x=fl.index, y=fl["temperature_c"], mode="markers",
                            name="Flagged", marker=dict(color=RED, size=8))
            fig.update_layout(
                height=300, template="plotly_white",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.15),
                yaxis=dict(title="°C"),
                yaxis2=dict(title="g / A", overlaying="y", side="right"),
                xaxis=dict(title="Last readings"))
            st.plotly_chart(fig, use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("<div class='sm-panel-title'>Anomalies Detected "
                        "(last 100)</div>", unsafe_allow_html=True)
            flagged = w[w["anomaly_flag"]]
            if state == "baseline":
                st.markdown("<div class='sm-ok'>Collect normal data, then run"
                            "<br><b>python train_real.py</b></div>",
                            unsafe_allow_html=True)
            elif flagged.empty:
                st.markdown("<div class='sm-ok'><div class='c'>✓</div>"
                            "No anomalies detected</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:{RED};font-size:1.6rem;"
                            f"font-weight:700'>{len(flagged)} flagged</div>",
                            unsafe_allow_html=True)
                st.dataframe(
                    flagged.tail(6)[["timestamp", *feats]]
                    .assign(timestamp=lambda d: d["timestamp"].dt.strftime("%H:%M:%S"))
                    .rename(columns={"timestamp": "Time", "temperature_c": "Temp (°C)", "vibration_rms_g": "Vib (g)", "current_a": "Current (A)"}),
                    hide_index=True, use_container_width=True, height=215)


def synthetic_board(end, data):
    w = data.iloc[max(0, end - WINDOW):end]
    state = "unusual" if bool(w.iloc[-1]["anomaly_flag"]) else "normal"
    ss = st.session_state
    strong = state == "unusual" and float(w.iloc[-1]["anomaly_score"]) <= EQ.STRONG_SCORE
    if ss.pop("test_pending", False) or (alarm_on and replay and strong and end >= ss.get("syn_arm_until", 0)):
        play_alarm(end)
    board(w, FEATURES, state)
    if syn_choice != SYN_PROJECT:
        i = EQ.NAMES.index(syn_choice); c = EQ.CATALOG[i]; last = w.iloc[-1]
        bearing = float(np.clip((last["vibration_rms_g"] / c["vib"] - 1.5) / 1.5, 0, 1))
        overload = float(np.clip((last["current_a"] / c["cur"] - 1.2) / 0.6, 0, 1))
        show_waves(i, end, last["vibration_rms_g"], last["current_a"], bearing=bearing, overload=overload)


@st.cache_resource
def eq_synth(idx):
    return EQ.synthetic_bundle(idx)


def syn_source():
    """(dataframe, banner html, metrics caption) for the selected synthetic-replay equipment."""
    if syn_choice == SYN_PROJECT:
        return df, ("Test rig", "Project synthetic dataset", "TEST_RIG_001"), None
    idx = EQ.NAMES.index(syn_choice); d, met = eq_synth(idx); c = EQ.CATALOG[idx]
    cap = (f"Model: Isolation Forest trained on the normal rows only. Test on {met['test_rows']:,} unseen readings: "
           f"anomalies caught {met['tp']} of {met['tp'] + met['fn']}, false alarms {met['fp']}, recall {met['recall']:.2f}, "
           f"precision {met['precision']:.2f}, F1 {met['f1']:.2f}. Synthetic data, not a real device.")
    return d, (c["name"], c["dept"], c["id"]), cap


def syn_banner(info, cap):
    badge = "<div class='b live'>● LIVE REPLAY</div>" if replay else "<div class='b paused'>PAUSED</div>"
    st.markdown(f"<div class='sm-eq'><div><div class='n'>{info[0]}</div><div class='d'>{info[1]}  |  Equipment ID {info[2]}</div></div>"
                f"<div style='display:flex'>{badge}<div class='b'>SYNTHETIC DATA</div></div></div>", unsafe_allow_html=True)
    if cap:
        st.caption(cap)


@st.fragment(run_every=2.0)
def live_board():
    w = live.load_live(WINDOW)
    if w is None:
        st.info("Waiting for data from the ESP32. In another terminal run: "
                "python collector.py --port COM5")
        return
    lb = live.load_bundle()
    if lb is not None:
        w = live.score(w, lb)
    else:
        w["anomaly_flag"] = False
    if live.is_stale(w):
        state = "offline"
    elif lb is None:
        state = "baseline"
    else:
        state = "unusual" if bool(w.iloc[-1]["anomaly_flag"]) else "normal"
    last_ts = w["timestamp"].iloc[-1].strftime("%H:%M:%S UTC")
    st.caption(f"Last reading: {last_ts}  |  source: "
               f"{w['data_source'].iloc[-1]}  |  refreshes every 2 s")
    board(w, LIVE_FEATURES, state)



def show_waves(idx, tick, vib, cur, bearing=0.0, overload=0.0):
    """Vibration and current waveform panels, drawn with the selected equipment's own signature."""
    def wave_fig(x, y, color, xt, yt):
        f = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=1.4)))
        f.update_layout(height=250, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(title=xt, showgrid=True, gridcolor="#e8edf4"),
                        yaxis=dict(title=yt, showgrid=True, gridcolor="#e8edf4", zeroline=True, zerolinecolor="#9aa7b8"))
        return f
    a_, b_ = st.columns(2)
    with a_:
        with st.container(border=True):
            st.markdown("<div class='sm-panel-title'>Vibration waveform (accelerometer, 0.5 s)</div>", unsafe_allow_html=True)
            tx, vx = EQ.vib_wave(idx, tick, float(vib), bearing=bearing)
            st.plotly_chart(wave_fig(tx, vx, "#17a589", "Time (ms)", "Acceleration (g)"), use_container_width=True)
    with b_:
        with st.container(border=True):
            st.markdown("<div class='sm-panel-title'>Current waveform (60 ms, 50 Hz supply)</div>", unsafe_allow_html=True)
            tc, ic = EQ.cur_wave(idx, tick, float(cur), overload=overload)
            st.plotly_chart(wave_fig(tc, ic, "#2e6fd8", "Time (ms)", "Current (A)"), use_container_width=True)


@st.cache_resource
def eq_models():
    return [EQ.train_model(i) for i in range(len(EQ.CATALOG))]


def _style_status(v):
    return "background-color:#fde2e2;color:#a12b2b;font-weight:700" if v == "Unusual" else "background-color:#d8f0df;color:#17653a;font-weight:700"


_beep_lock = threading.Lock()


def _beep_worker():
    if not _beep_lock.acquire(blocking=False):
        return
    try:
        for f in (1200, 900, 1200, 900):
            winsound.Beep(f, 140)
    finally:
        _beep_lock.release()


def play_alarm(tick):
    """Windows laptop: beep through the speakers. Other systems: browser audio."""
    if winsound is not None:
        threading.Thread(target=_beep_worker, daemon=True).start()
    else:
        try:
            st.audio(EQ.make_beep_wav(tick), format="audio/wav", autoplay=True)
        except TypeError:
            st.audio(EQ.make_beep_wav(tick), format="audio/wav")


def _eq_body():
    ss = st.session_state
    ss.setdefault("tick", 800); ss.setdefault("fault_key", None); ss.setdefault("fault_start", None)
    ss.setdefault("events", {}); ss.setdefault("sched", {"next": 835}); ss.setdefault("last_beep", -99); ss.setdefault("rng", np.random.default_rng(7))
    ss.setdefault("active", []); ss.setdefault("log", [])
    key = (eq_name, fault)
    if ss.fault_key != key:
        ss.fault_key = key
        ss.fault_start = None if fault == "None" else ss.tick - (0 if eq_live else 20)
    just_changed = ss.get("arm_key") != (mode, eq_name)
    ss.arm_key = (mode, eq_name)
    if just_changed:
        ss.arm_until = ss.tick + 3          # no alarm sound for 3 readings after selecting an equipment
    if eq_live:
        ss.tick += 1
        if rand_events:
            EQ.step_events(ss.events, ss.sched, ss.rng, ss.tick, len(EQ.CATALOG))
    tick = ss.tick
    idx = EQ.NAMES.index(eq_name); c = EQ.CATALOG[idx]; models = eq_models()

    def fault_of(j):
        if j == idx and fault != "None":
            return fault, ss.fault_start
        ev = ss.events.get(j)
        return (ev["fault"], ev["start"]) if ev and ev["start"] <= tick <= ev["end"] else ("None", None)

    # ---- status of every equipment (also drives the alarm) ----
    rows, active = [], {}
    for j, cj in enumerate(EQ.CATALOG):
        fj, sj = fault_of(j)
        wj = EQ.window(j, tick, fj, sj, n=10)
        stt = EQ.status(EQ.flags(models[j], wj))
        tj, vj, ij = float(wj["temp"][-1]), float(wj["vib"][-1]), float(wj["cur"][-1])
        rows.append({"Equipment": cj["name"], "Department": cj["dept"], "ID": cj["id"], "Temperature (°C)": round(tj, 1),
                     "Vibration (g)": round(vj, 3), "Current (A)": round(ij, 2), "Status": stt})
        if stt == "Unusual":
            active[j] = f"{cj['name']} ({cj['id']}): {tj:.1f} °C, {vj:.3f} g, {ij:.2f} A"
    now_s = datetime.now().strftime("%H:%M:%S")
    new = [j for j in active if j not in ss.active]
    for j in new:
        ss.log.insert(0, {"Time": now_s, "Event": "ALARM", "Equipment": EQ.CATALOG[j]["name"], "Reading": active[j].split(": ", 1)[1]})
    for j in [j for j in ss.active if j not in active]:
        ss.log.insert(0, {"Time": now_s, "Event": "Cleared", "Equipment": EQ.CATALOG[j]["name"], "Reading": "back to normal"})
    ss.active = list(active); ss.log = ss.log[:30]
    if ss.pop("test_pending", False):
        play_alarm(tick)
    elif alarm_on and EQ.alarm_due(active, new, (ss.tick >= ss.arm_until) and not just_changed, tick, ss.last_beep):
        ss.last_beep = tick
        play_alarm(tick)

    if active:
        names = ", ".join(EQ.CATALOG[j]["name"] for j in active)
        st.markdown(f"<div class='sm-alarm'>🚨 ALARM: {names}: unusual pattern detected. Inspect the equipment.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='sm-allok'>✓ All {len(EQ.CATALOG)} equipment are normal</div>", unsafe_allow_html=True)

    w = EQ.window(idx, tick, *fault_of(idx))
    fl = EQ.flags(models[idx], w); state = "unusual" if EQ.status(fl) == "Unusual" else "normal"
    cur_fault = fault_of(idx)[0]
    ramp = float(w["ramp"][-1]); eff = EQ.FAULT_EFFECT[cur_fault]
    live_badge = "<div class='b live'>● LIVE</div>" if eq_live else "<div class='b paused'>PAUSED</div>"
    st.markdown(f"<div class='sm-eq'><div><div class='n'>{c['name']}</div>"
                f"<div class='d'>{c['dept']}  |  Equipment ID {c['id']}  |  fault now: {cur_fault}  |  {now_s}</div></div>"
                f"<div style='display:flex'>{live_badge}<div class='b'>SIMULATED DEMO PROFILE</div></div></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    c1.markdown(card("Temperature", f"{w['temp'][-1]:.1f} °C"), unsafe_allow_html=True)
    c2.markdown(card("Vibration (RMS)", f"{w['vib'][-1]:.3f} g"), unsafe_allow_html=True)
    c3.markdown(card("Current (RMS)", f"{w['cur'][-1]:.2f} A"), unsafe_allow_html=True)
    color, mark, text = STATES[state]
    c4.markdown(f"<div class='sm-card'><div class='sm-label'>Status</div><div class='sm-status'>"
                f"<div class='sm-dot' style='background:{color}'>{mark}</div>"
                f"<div class='sm-value' style='font-size:1.15rem'>{text}</div></div></div>", unsafe_allow_html=True)
    st.write("")

    show_waves(idx, tick, w["vib"][-1], w["cur"][-1], bearing=eff[3] * ramp, overload=ramp if eff[2] > 1.2 else 0.0)

    n_pts = len(w["temp"]); t_now = datetime.now()
    x = [t_now - timedelta(seconds=n_pts - 1 - k) for k in range(n_pts)]
    a_, b_ = st.columns(2)
    with a_:
        with st.container(border=True):
            st.markdown("<div class='sm-panel-title'>Temperature trend (last 2 minutes)</div>", unsafe_allow_html=True)
            f = go.Figure(go.Scatter(x=x, y=w["temp"], mode="lines", name="Temperature (°C)", line=dict(color="#e0408a", width=1.6)))
            f.add_scatter(x=[xx for xx, z in zip(x, fl) if z], y=[yy for yy, z in zip(w["temp"], fl) if z], mode="markers",
                          name="Flagged", marker=dict(color=RED, size=7))
            f.update_layout(height=260, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15),
                            xaxis=dict(title="Time", tickformat="%H:%M:%S"), yaxis=dict(title="°C"))
            st.plotly_chart(f, use_container_width=True)
    with b_:
        with st.container(border=True):
            st.markdown("<div class='sm-panel-title'>Vibration and current trend</div>", unsafe_allow_html=True)
            f = go.Figure()
            f.add_scatter(x=x, y=w["vib"], name="Vibration (g)", line=dict(color="#17a589", width=1.6))
            f.add_scatter(x=x, y=w["cur"], name="Current (A)", yaxis="y2", line=dict(color="#2e6fd8", width=1.6))
            f.update_layout(height=260, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15),
                            xaxis=dict(title="Time", tickformat="%H:%M:%S"), yaxis=dict(title="g"), yaxis2=dict(title="A", overlaying="y", side="right"))
            st.plotly_chart(f, use_container_width=True)

    st.markdown("<div class='sm-panel-title' style='margin-top:6px'>Hospital equipment overview</div>", unsafe_allow_html=True)
    tbl = pd.DataFrame(rows)
    try:
        styled = tbl.style.map(_style_status, subset=["Status"])
    except AttributeError:
        styled = tbl.style.applymap(_style_status, subset=["Status"])
    st.dataframe(styled, hide_index=True, use_container_width=True, height=400)
    st.markdown("<div class='sm-panel-title' style='margin-top:6px'>Alarm log (latest events)</div>", unsafe_allow_html=True)
    if ss.log:
        st.dataframe(pd.DataFrame(ss.log[:10]), hide_index=True, use_container_width=True)
    else:
        st.caption("No alarms yet. Faults appear automatically in live mode, or choose Simulate fault in the sidebar.")
    st.caption("All values on this screen are simulated demo profiles for illustration. A separate Isolation Forest is trained "
               "for each equipment on its own normal profile; real readings come only from the ESP32 test rig (Live mode).")


def equipment_view():
    if eq_live:
        st.fragment(run_every=1.0)(_eq_body)()
    else:
        _eq_body()


if mode == EQ_MODE:
    equipment_view()
elif mode == LIVE_MODE:
    live_board()
else:
    sdata, sinfo, scap = syn_source()
    syn_banner(sinfo, scap)
    if replay:
        @st.fragment(run_every=1.0)
        def replay_board():
            ss = st.session_state
            if ss.get("replay_eq") != syn_choice:
                ss.replay_eq = syn_choice; ss.replay_idx = WINDOW; ss.syn_arm_until = WINDOW + 3
            ss.replay_idx += 1
            if ss.replay_idx > len(sdata):
                ss.replay_idx = WINDOW
            synthetic_board(ss.replay_idx, sdata)
        replay_board()
    else:
        synthetic_board(pos, sdata)

if HAVE_SYN and mode != EQ_MODE:
  with st.expander("Tools: manual reading check and CSV download"):
      st.download_button("Download scored synthetic CSV",
                         df.to_csv(index=False).encode("utf-8"),
                         "scored_readings.csv", "text/csv")
      m1, m2, m3 = st.columns(3)
      t = m1.number_input("Temperature (°C)", value=37.0, step=0.1)
      v = m2.number_input("Vibration RMS (g)", value=0.12, step=0.01, format="%.3f")
      a = m3.number_input("Current (A)", value=0.31, step=0.01, format="%.3f")
      if st.button("Check reading (synthetic model)"):
          row = pd.DataFrame([[t, v, a]], columns=FEATURES)
          if bundle["model"].predict(row)[0] == -1:
              st.error("Unusual pattern flagged - review by a person needed.")
          else:
              st.success("Within learned baseline (synthetic). Not a safety guarantee.")

st.markdown("<div class='sm-note'>🛡️ Educational prototype. Flags show unusual "
            "patterns for human review. They are NOT failure probabilities, "
            "clinical advice or safety certification. Use a low-voltage, "
            "guarded test rig only.</div>", unsafe_allow_html=True)

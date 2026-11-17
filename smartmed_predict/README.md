# SmartMed Predict (educational prototype)
AI/ML based predictive maintenance for a low-voltage motor/pump test rig.
Synthetic data only until real hardware data is collected. Not for clinical use.

## Install (Windows, VS Code terminal)
    pip install -r requirements.txt

## 0. Hospital equipment demo (default screen)
Sidebar > Data source > "Hospital equipment (demo)":
- Select equipment (Patient Monitoring System, Infusion Pump, Ventilator, Suction Machine, ...).
- Each equipment has its own temperature, vibration and current values, its own waveform signature (fan, pump pulses, compressor impulses, switch-mode current peaks, resistive heater, ...) and trends.
- "Simulate fault" (Overheating / Bearing wear / Overload / Combined) shows how the status turns Unusual.
- "Live monitoring" (default ON) moves one reading per second with a real clock, like a live feed.
- "Random fault events" makes the SELECTED equipment develop a fault for 25-40 s, then recover. The alarm banner, beep and log always show the equipment you selected; selecting another equipment starts it healthy.
- ALARM rules: a new alarm beeps at once, then repeats every 4 s while it stays active; no sound for 3 s after you select another equipment; selecting another equipment resets "Simulate fault" to None; in Synthetic replay only strong anomalies (decision score below -0.10) sound the alarm.
- ALARM: when any equipment turns Unusual, a red blinking banner appears, a beep sounds on the
  laptop speakers (Windows: winsound; other systems: browser audio) and the event goes to the Alarm log.
  Use "Alarm sound" to mute and "Test alarm sound" to check the speakers.
- All values are SIMULATED demo profiles.

## 0b. Synthetic replay for one equipment (to explain monitoring to examiners)
Sidebar > Data source > "Synthetic replay" > Select equipment (e.g. Ventilator).
A 10,000-row synthetic dataset is generated for that equipment's normal profile (5% injected anomalies),
an Isolation Forest is trained on normal rows only (80/20 split), and the readings replay at 1 per second.
Cards, trends, flagged readings, status and the alarm sound work as in live monitoring.
The model's test result (anomalies caught, false alarms, recall, precision, F1) is shown under the equipment name.
"Test rig (project dataset)" replays the dataset created by generate_data.py (needs generate_data.py + train_model.py first).

## Model train and test output (for the report)
    python generate_data.py && python train_model.py      -> project dataset: train/test results (screenshot the terminal)
    python -m pytest -q                                    -> automated tests output (screenshot the terminal)
    python evaluate_all.py                                 -> table for all 14 equipment + results/model_results.csv
    python evaluate_all.py --equipment "Ventilator"        -> detailed train/test output for one equipment

## Viva / external demo order
    python show_data.py --equipment "Ventilator" --save     1. training data (also data\Ventilator.csv for Excel)
    python evaluate_all.py --equipment "Ventilator"          2. train/test result
    python -m pytest -q                                       3. automated tests
    python collector.py --simulate --interval 0.05            4. data collection + validation (Ctrl+C after 15 s)
    python -m streamlit run app.py                            5. dashboard: Hospital equipment / Synthetic replay / Live

## 1. Demo (no hardware)
    python generate_data.py
    python train_model.py
    python -m pytest -q
    python -m streamlit run app.py        (Data source = Synthetic replay)

## 2. Live mode test without hardware
    python collector.py --simulate --interval 0.05     (run ~20 s, then Ctrl+C)
    python train_real.py --include-simulated
    Terminal 1: python collector.py --simulate
    Terminal 2: python -m streamlit run app.py          (Data source = Live ESP32)

## 3. With hardware (ESP32 + DS18B20 + ADXL345)
1. Open esp32_firmware/smartmed_esp32/smartmed_esp32.ino in Arduino IDE
   (libraries: OneWire, DallasTemperature; board: ESP32 Dev Module) and upload.
2. Close Serial Monitor. Find the COM port in Device Manager.
3. Run the rig normally for 30-60 min:  python collector.py --port COM5
4. Ctrl+C, then:  python train_real.py
5. Terminal 1: python collector.py --port COM5
   Terminal 2: python -m streamlit run app.py        (Data source = Live ESP32)

Wiring: DS18B20 DATA->GPIO4 (4.7k pull-up to 3.3V); ADXL345 SDA->GPIO21, SCL->GPIO22; both on 3.3V.
Motor/pump uses its own 12 V supply. Never connect mains voltage or clinical equipment.

/*
  SmartMed Predict - ESP32 firmware (educational prototype)
  Reads DS18B20 temperature + ADXL345 vibration, prints ONE JSON line / second.
  Safe low-voltage test rig only. Do NOT attach to clinical equipment.

  Wiring (ESP32 DevKit):
    DS18B20 : VDD->3.3V, GND->GND, DATA->GPIO4 (+4.7k pull-up DATA to 3.3V)
    ADXL345 : VCC->3.3V, GND->GND, SDA->GPIO21, SCL->GPIO22
              (CS -> 3.3V for I2C; SDO -> GND gives address 0x53)

  Libraries (Arduino IDE > Library Manager): OneWire, DallasTemperature
  Board: "ESP32 Dev Module"   Serial Monitor: 115200 baud
*/
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ---------- settings ----------
const char*    EQUIPMENT_ID = "TEST_RIG_001";
const uint32_t BAUD         = 115200;
const uint8_t  ONE_WIRE_PIN = 4;
const uint8_t  ADXL_ADDR    = 0x53;
const uint16_t N_SAMPLES    = 256;     // RMS window length
const uint32_t SAMPLE_US    = 1250;    // 800 Hz sampling (matches ODR below)
const uint32_t REPORT_MS    = 1000;    // one record per second
const float    G_PER_LSB    = 0.0039f; // full-resolution mode
// ------------------------------

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature tempSensor(&oneWire);

int16_t ax[N_SAMPLES], ay[N_SAMPLES], az[N_SAMPLES];
uint32_t seqNo = 0;

bool adxlWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADXL_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool adxlInit() {
  bool ok = adxlWrite(0x2C, 0x0D);  // BW_RATE: 800 Hz output data rate
  ok &= adxlWrite(0x31, 0x08);      // DATA_FORMAT: full resolution, +/-2 g
  ok &= adxlWrite(0x2D, 0x08);      // POWER_CTL: measurement mode
  return ok;
}

bool adxlRead(int16_t &x, int16_t &y, int16_t &z) {
  Wire.beginTransmission(ADXL_ADDR);
  Wire.write(0x32);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)ADXL_ADDR, 6) != 6) return false;
  x = Wire.read() | (Wire.read() << 8);
  y = Wire.read() | (Wire.read() << 8);
  z = Wire.read() | (Wire.read() << 8);
  return true;
}

// Dynamic RMS (g): gravity/offset removed by subtracting per-axis mean.
bool vibrationRms(float &rmsG) {
  for (uint16_t i = 0; i < N_SAMPLES; i++) {
    uint32_t t0 = micros();
    if (!adxlRead(ax[i], ay[i], az[i])) return false;
    while (micros() - t0 < SAMPLE_US) { }
  }
  double mx = 0, my = 0, mz = 0;
  for (uint16_t i = 0; i < N_SAMPLES; i++) { mx += ax[i]; my += ay[i]; mz += az[i]; }
  mx /= N_SAMPLES; my /= N_SAMPLES; mz /= N_SAMPLES;
  double acc = 0;
  for (uint16_t i = 0; i < N_SAMPLES; i++) {
    double dx = ax[i] - mx, dy = ay[i] - my, dz = az[i] - mz;
    acc += dx * dx + dy * dy + dz * dz;
  }
  rmsG = sqrt(acc / N_SAMPLES) * G_PER_LSB;
  return true;
}

void setup() {
  Serial.begin(BAUD);
  Wire.begin(21, 22);
  Wire.setClock(400000);
  tempSensor.begin();
  tempSensor.setResolution(10);     // ~190 ms conversion
  if (!adxlInit()) Serial.println("{\"info\":\"adxl345 init failed\"}");
  delay(500);
}

void loop() {
  uint32_t start = millis();

  tempSensor.requestTemperatures();
  float t = tempSensor.getTempCByIndex(0);
  bool tempOk = (t != DEVICE_DISCONNECTED_C) && (t > -50.0f) && (t < 125.0f);

  float rms = 0;
  bool accelOk = vibrationRms(rms);

  const char* status = "ok";
  if (!tempOk && !accelOk) status = "multi_error";
  else if (!tempOk)        status = "temp_error";
  else if (!accelOk)       status = "accel_error";

  Serial.print("{\"seq\":");            Serial.print(seqNo++);
  Serial.print(",\"uptime_ms\":");      Serial.print(millis());
  Serial.print(",\"equipment_id\":\""); Serial.print(EQUIPMENT_ID); Serial.print("\"");
  Serial.print(",\"temperature_c\":");  if (tempOk)  Serial.print(t, 2);   else Serial.print("null");
  Serial.print(",\"vibration_rms_g\":");if (accelOk) Serial.print(rms, 4); else Serial.print("null");
  Serial.print(",\"current_a\":null");  // current sensor not implemented yet
  Serial.print(",\"sensor_status\":\""); Serial.print(status); Serial.println("\"}");

  uint32_t used = millis() - start;
  if (used < REPORT_MS) delay(REPORT_MS - used);
}

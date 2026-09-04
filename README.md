# Adaptive Edge–Cloud TinyML for Gesture Recognition

Canonical implementation repository for the undergraduate project:

**سامانه هوشمند تطبیقی لبه–ابر مبتنی بر TinyML**

The gesture-recognition pipeline is the testbed. The primary contribution is adaptive
Edge–Cloud inference with learned offloading and split-point selection.

## Current project phase

**Phase 2 — Data collection / pre-M3**

Current checkpoint:

- hardware bring-up is complete,
- 100 Hz calibrated IMU streaming is stable,
- boot gyroscope calibration is active,
- `orientation-v1` is frozen for dataset-v1,
- `accel-cal-v1` is applied and six-position validated,
- dataset-v1 gesture collection has not started yet.

M3 is reached only when dataset-v1 is fully collected with session-based
train/validation/test separation.

Do not start Adaptive Cloud work before Local Gesture Recognition on ESP32 is stable.

## Fixed v1 choices

- Board: ESP32-S3-WROOM-1-N8R2
- Current IMU unit: GY-521 with MPU6050-compatible register interface
- Current IMU identity: non-standard `WHO_AM_I=0x74`, register compatibility verified
- Sampling: 100 Hz
- Window: 1 s / 100 samples
- Runtime overlap: 50%
- Accelerometer: ±4 g
- Gyroscope: ±500 °/s
- I2C: 400 kHz
- Verified SDA/SCL: GPIO8/GPIO9
- Gesture classes: IDLE, SWIPE_LEFT, SWIPE_RIGHT, ROTATE_CW, SHAKE
- Feature version: features-v1
- Dataset version: dataset-v1
- Firmware version: 0.1.0
- Accelerometer calibration: accel-cal-v1
- Orientation protocol: orientation-v1

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pytest -q
```

The main source of truth is `docs/PROJECT_ARCHITECTURE.md`.

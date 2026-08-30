# Adaptive Edge–Cloud TinyML for Gesture Recognition

Canonical implementation repository for the undergraduate project:

**سامانه هوشمند تطبیقی لبه–ابر مبتنی بر TinyML**

The gesture-recognition pipeline is the testbed. The primary contribution is adaptive
Edge–Cloud inference with learned offloading and split-point selection.

## Current project phase

**Phase 0 — Project skeleton**

Phase 0 is complete when:
- the repository exists,
- configuration files parse correctly,
- basic Python tests pass.

Do not start Adaptive Cloud work before Local Gesture Recognition on ESP32 is stable.

## Fixed v1 choices

- Board: ESP32-S3-WROOM-1-N8R2
- IMU: MPU6050 GY-521
- Sampling: 100 Hz
- Window: 1 s / 100 samples
- Runtime overlap: 50%
- Accelerometer: ±4 g
- Gyroscope: ±500 °/s
- I2C: 400 kHz
- Planned SDA/SCL: GPIO8/GPIO9
- Gesture classes: IDLE, SWIPE_LEFT, SWIPE_RIGHT, ROTATE_CW, SHAKE
- Feature version: features-v1
- Dataset version: dataset-v1

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

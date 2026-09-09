# Adaptive Edge–Cloud TinyML for Gesture Recognition

Canonical implementation repository for the undergraduate project:

**سامانه هوشمند تطبیقی لبه–ابر مبتنی بر TinyML**

The gesture-recognition pipeline is the testbed for a learned binary LOCAL/CLOUD
policy under Architecture Revision R1. CLOUD sends exactly 10 features-v1 values.

## Current project phase

Canonical Phase 11 / M11: Database + Dashboard, **IN PROGRESS**.
Phase 9 / M9 learned binary LOCAL/CLOUD policy is closed and Phase 10 / M10
failover is closed with controlled ESP32 hardware evidence. The current Phase-11
checkpoint is the versioned decision-telemetry → PostgreSQL persistence path; the
dashboard UI follows only after the reboot-safe persistence path is hardware-validated.

Split1/2/3 artifacts and regression suites remain fixed experimental baselines.
Production adaptive action space remains exactly `LOCAL` / `CLOUD`; CLOUD sends
exactly ten `features-v1` values. Historical four-action datasets/configurations are
immutable evidence, not production policy contracts.

See [Phase 10 completion](docs/phase10_m10_completion.md),
[Phase 11 plan](docs/phase11_database_dashboard.md), and
[R1 migration audit](docs/phase9_r1_migration_audit.md).

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
- Firmware version: 0.3.1-r1
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

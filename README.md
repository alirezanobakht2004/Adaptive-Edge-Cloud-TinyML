# Adaptive Edge–Cloud TinyML for Gesture Recognition

Canonical implementation repository for the undergraduate project:

**سامانه هوشمند تطبیقی لبه–ابر مبتنی بر TinyML**

The gesture-recognition pipeline is the testbed for a learned binary LOCAL/CLOUD
policy under Architecture Revision R1. CLOUD sends exactly 10 features-v1 values.

## Current project phase

Canonical Phase 9 / M9: learned binary adaptive policy, CLOSED (controlled MVP).
Phase 7 split baselines and Phase 8 policy studies are complete. M25?M34 names in
older reports are historical study milestones, not the canonical phase sequence.
Phase 10 / M10 failover has not started.

Split1/2/3 artifacts and regression suites remain fixed experimental baselines.
Production firmware integrates meta-policy-v1.0.0: LOCAL reuses the validated
local uncertainty result; CLOUD sends exactly ten features. Historical four-action datasets
and configurations are immutable evidence, not production policy contracts.

See [canonical Phase 9 status](docs/phase9_learned_binary_policy.md) and
[R1 migration audit](docs/phase9_r1_migration_audit.md). The reuse-LOCAL blocker is
resolved. Both learned actions pass controlled hardware E2E; a live run passes
50 LOCAL decisions. Policy generalization is limited by one CLOUD-benefit window
and zero CLOUD-optimal holdout support. Next: Phase 10 / M10 - Failover.

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
- Firmware version: 0.2.0-r1
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

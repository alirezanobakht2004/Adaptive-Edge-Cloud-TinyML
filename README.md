# Adaptive Edge–Cloud TinyML for Gesture Recognition

Canonical implementation repository for the undergraduate project:

**سامانه هوشمند تطبیقی لبه–ابر مبتنی بر TinyML**

The gesture-recognition pipeline is the testbed for a learned binary LOCAL/CLOUD
policy under Architecture Revision R1. CLOUD sends exactly 10 features-v1 values.

## Current project phase

Canonical Phase 9 / M9: learned binary adaptive policy, in progress and gated.
Phase 7 split baselines and Phase 8 policy studies are complete. M25?M34 names in
older reports are historical study milestones, not the canonical phase sequence.
Phase 10 / M10 failover has not started.

Split1/2/3 artifacts and regression suites remain fixed experimental baselines.
The production firmware currently retains the validated local uncertainty path;
a learned binary policy is not yet integrated. Historical four-action datasets
and configurations are immutable evidence, not production policy contracts.

See [canonical Phase 9 status](docs/phase9_learned_binary_policy.md) and
[R1 migration audit](docs/phase9_r1_migration_audit.md). Training is blocked because
historical LOCAL candidates rerun inference, while R1 must reuse the local result.

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

# Phase 11 / M11 Dashboard Quality v3

Checkpoint: M11 Dashboard Quality Closure — 3D reference-rig redesign + pose responsiveness tuning.

## Why this patch exists

Measured/user-observed browser behavior after v2 still showed two quality issues:

1. the 3D representation did not visually match the physical ESP32/IMU orientation reference well enough and the large white carrier was visually distracting;
2. motion remained visibly step-like / delayed. The prior pose path used a 200 ms (5 Hz) MQTT publish target plus an 80 ms database-backed pose WebSocket poll interval. Screenshots also showed live pose ages in the approximate 0.1–0.46 s range during the test session.

## Changes

- Reworked `DeviceTwin` into a dark industrial reference rig with an asymmetric ESP32-S3 + GY-521 layout, USB/front marker, gold pin rails, and sensor-axis marker.
- Kept the canonical orientation mapping unchanged:
  - sensor +X -> scene +X
  - sensor +Y -> scene -Z
  - sensor +Z -> scene +Y
  - Home Pose = sensor +Z upward.
- Increased pose publish target from 5 Hz to 10 Hz (`200 ms -> 100 ms`) while retaining:
  - latest-value overwrite queue,
  - non-blocking 100 Hz sampling path,
  - RTT probe / CLOUD response priority,
  - visualization-only provenance.
- Reduced database-backed pose WebSocket polling from 80 ms to 50 ms.
- Retuned quaternion interpolation for the higher pose update rate.
- Bumped firmware version to `0.3.3-r1`.
- Bumped dashboard UI version to `dashboard-ui-r1-v4`.
- Preserved the original frozen 5 Hz config in `config/r1_pose_v1.json`; added `config/r1_pose_v1_1.json` for the tuned 10 Hz target.

## Unchanged contracts

- Production action space remains LOCAL/CLOUD only.
- CLOUD still sends only features-v1 (10 values).
- pose schema remains `pose-v1`.
- estimator remains `attitude-complementary-v1`.
- roll/pitch remain estimated.
- yaw remains boot-relative and drift-prone.
- pose is not a policy input.
- no raw 100x6 IMU window is added to the production dashboard path.

## Validation required on project machine

1. `npm run typecheck`
2. `npm run build`
3. Phase 11 pytest regression suite
4. `pio run -e esp32-s3-n8r2`
5. upload firmware `0.3.3-r1`
6. validate live pose stream and browser motion
7. repeat Home/+X/-X/+Y/-Y/-Z visual orientation checks
8. confirm no inference/sampling/network regression

Do not close M11 until the hardware/browser checks pass.

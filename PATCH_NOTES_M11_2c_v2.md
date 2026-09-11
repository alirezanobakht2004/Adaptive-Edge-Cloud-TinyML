# Phase 11 / M11.2c — Sensor-Driven 3D Device Twin — Final v2

This archive supersedes the previous M11.2c archive.

## Firmware compile hotfix

The previous `attitude_estimator.cpp` declared a local constant named `RAD_TO_DEG`.
Arduino-ESP32 already defines `RAD_TO_DEG` as a preprocessor macro in `Arduino.h`, so the identifier was expanded before C++ parsing and PlatformIO failed with:

`error: expected unqualified-id before numeric constant`

The local constant is now named `RADIANS_TO_DEGREES`. No estimator math, pose schema, MQTT topic, policy input, or database contract changed.

Firmware version remains `0.3.2-r1` because the earlier build never completed and that version was not deployed.

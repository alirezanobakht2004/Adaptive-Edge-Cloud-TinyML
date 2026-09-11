# Phase 11 / M11 Dashboard Quality v2 — 3D Orientation & Response Patch

## Scope

This is a focused follow-up to Dashboard Quality v1. It does **not** change the learned LOCAL/CLOUD policy, firmware pose estimator, pose-v1 schema, MQTT topics, database schema, or cloud inference path.

## Changes

- Replaces the generic green PCB 3D model with a representative physical rig based on `Orientations.png`:
  - white breadboard body,
  - centrally aligned ESP32-S3 development board,
  - distinct blue GY-521/MPU6050 module,
  - USB connector/cable marker,
  - sensor-axis triad.
- Keeps the canonical orientation mapping:
  - Home Pose = sensor `+Z` upward,
  - sensor `+X` -> scene `+X`,
  - sensor `+Y` -> scene `-Z`,
  - sensor `+Z` -> scene `+Y`.
- Adds an English-only orientation reference overlay (`HOME +Z`, `+X`, `+Y`, `+Z`).
- Keeps camera rotation locked so physical device movement cannot be confused with mouse rotation.
- Changes dashboard-only quaternion interpolation coefficient from `k=10` to `k=28` to reduce visual settling delay while retaining smooth motion.

## Intentionally unchanged

- Pose source sampling: 100 Hz.
- Pose MQTT publication target: 200 ms / ~5 Hz.
- Server pose WebSocket DB poll: 80 ms.
- `pose-v1` semantics and estimator.
- Firmware version `0.3.2-r1`.
- `meta-policy-v1.0.0` and all Phase 9 policy artifacts.

## Why firmware pose rate is not changed yet

The current screenshot showed pose ages around 0.1–0.2 s while the live WebSocket was healthy. The first low-risk correction is therefore to remove excess frontend smoothing and improve the visual model. If the physical-to-screen lag is still materially high after this patch, the next measured experiment is to compare the current 5 Hz pose publish rate against a 10 Hz candidate before changing firmware/config versions.

## Policy note

A 100% LOCAL dashboard window is not evidence of a dashboard bug. The repository's Phase 9 evidence documents a strongly LOCAL-biased learned policy under the qualified dataset. The existing ESP32 policy parity evidence also proves the deployed model can output both LOCAL and CLOUD on known vectors. Do not force CLOUD or change the learned policy merely to make the dashboard donut look balanced.

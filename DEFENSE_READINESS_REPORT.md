# Defense Readiness Report

Audit date: 2026-09-12  
Final state: **Defense-ready with frozen undergraduate scope through Phase 11**

## Audit basis and precedence

This audit covered the outer defense package (`Instruction/PROJECT_ARCHITECTURE.md`,
`Project Document/ProjectDocument.docx`, `ProjectDefinitionAlirezaNobakht.pdf`, hardware
datasheets and implementation photographs) and the versioned implementation repository:
firmware, server, dashboard, configuration, documentation, experiments/evidence, model
artifacts, tests, Docker Compose, dependency manifests, and the defense launcher.

Conflicts were resolved in this order: measured hardware/experiment evidence, official
datasheets, canonical architecture, versioned artifacts, implementation notes, and old
assumptions. Historical Phase 7–9 records were retained as evidence and were not rewritten
as if they were current production contracts. The original project-definition PDF is an
origin/scope artifact; the R1/R2 architecture and measured implementation evidence govern
the final implementation claim.

## Current architecture summary

The final system is Adaptive Edge–Cloud TinyML on ESP32-S3-WROOM-1-N8R2 with a
GY-521/MPU6050-compatible IMU. Five-class gesture/movement recognition is the testbed,
not the main contribution.

The device samples six IMU channels at 100 Hz, forms 100-sample/one-second windows with
50% runtime overlap, extracts and normalizes exactly ten `features-v1` values, runs the
validated local inference and five-pass uncertainty path, and evaluates a learned binary
policy from uncertainty, network state, and device/resource state.

Production actions are exactly:

- `LOCAL`: use the local result already computed for the window.
- `CLOUD`: send exactly ten normalized `features-v1` values over MQTT to the server's
  full-cloud model.

Production CLOUD does not send a raw 100x6 IMU window. Raw windows are limited to dataset,
debugging, or future-research paths. The observability path is ESP32 → MQTT → server →
PostgreSQL → FastAPI/WebSocket → dashboard. The auxiliary `pose-v1` visualization stream
does not change the policy or inference payload.

## Completed implementation scope

Final implementation scope completed through Phase 11:

- Phases 0–4: repository, hardware, dataset, base model, and local TinyML
- Phase 5: uncertainty estimation
- Phase 6: server-assisted inference and MQTT
- Phase 7: validated fixed split-inference baselines
- Phase 8: action-space benchmarking and policy-data infrastructure
- Phase 9: learned binary LOCAL/CLOUD policy
- Phase 10: failover
- Phase 11: PostgreSQL persistence, FastAPI/WebSocket API, live dashboard, and device twin

The defense contribution list is:

- Adaptive Edge–Cloud Inference
- Local TinyML inference
- Uncertainty Estimation
- Learned Binary LOCAL/CLOUD Policy
- MQTT / Server-Assisted Inference
- Failover
- PostgreSQL telemetry persistence
- Live Dashboard / Observability
- Validated Split-Inference baselines

## Split-inference conclusion

Split1, Split2, and Split3 remain validated experimental baselines with 64-D, 48-D, and
32-D representations. They are fixed-split comparisons and negative-result evidence—not
failures and not production policy actions. Direct CLOUD sends only 10 feature values;
each split sends a larger representation and adds edge-prefix computation. The evidence
therefore supports retiring adaptive split selection from production without deleting the
validated split implementation.

## Failover contract

If Wi-Fi, MQTT, publish, or server response is unavailable, the effective action becomes
`LOCAL`, inference continues, and the transition/failure is logged. Because the local
result already exists before policy selection, failover reuses it and does not run a
duplicate local inference. Firmware implementation and Phase 10 tests enforce this state
transition.

## Canonical parameters and versions

| Item | Defense value |
|---|---|
| Hardware | ESP32-S3-WROOM-1-N8R2; GY-521/MPU6050-compatible interface |
| Sampling/window/step | 100 Hz; 100 samples / 1 s; 50 samples / 50% overlap |
| Sensor ranges / I2C | ±4 g; ±500 dps; 400 kHz; GPIO8/GPIO9 |
| Inference input / uncertainty | features-v1, 10 values; 5 stochastic passes |
| Local model | gesture-model-v1.1.0 |
| Cloud model | gesture-full-cloud-v1.0.0 |
| Policy | meta-policy-v1.0.0 |
| Current firmware | 0.3.5-r1 |
| Dashboard package | 1.0.0 |
| Production schema | inference-r1-v1; decision-r1-v1; pose-v1 |

Older version strings in immutable evidence, experiment manifests, and historical closure
documents identify the artifact that produced that evidence; they are not current-version
conflicts. `server/app/r1_schema.py` uses `0.2.0-r1` only as the historical request-builder
fixture default, while production parsing accepts the versioned firmware identifier.

## Verification performed

- Python suite: **343 passed**, 7 dependency deprecation warnings.
- Dashboard: TypeScript/Vite production build passed; Vite reported a non-blocking
  JavaScript chunk-size warning.
- Docker: `docker compose config --quiet` parsed successfully.
- Firmware source contract: static assertions confirm 100 samples, 50-sample step,
  six channels, ten input features, and five uncertainty passes.
- Firmware compile: not rerun during this audit because PlatformIO is not installed in
  the current PATH. Prior hardware build/upload and Phase 11 evidence are preserved.
- Runtime hardware/database: not re-executed during this documentation audit; existing
  measured Phase 10/11 evidence and the final report are the claim basis.

## Metrics and claim discipline

Only values with recorded provenance may be presented. The report's 98.5% accuracy and
98.49% macro F1 are held-out test results for 200 `session_03` samples, not validation
metrics. Server compute duration, device-observed request elapsed/E2E duration, and RTT
are separate quantities. Energy values in policy studies are estimated or simulated
dimensionless proxies; battery energy and joules were not physically measured.

## Known limitations

- The primary gesture dataset is dominated by one initial user.
- Naturally observed CLOUD-optimal states are limited; controlled/simulated state
  variation must remain labeled as such.
- The short normal Phase 9 production run selected LOCAL; controlled vectors demonstrate
  both policy actions but do not prove natural action diversity.
- No physical battery-energy measurement exists.
- MPU6050-only yaw is boot-relative and drift-prone, not absolute heading.
- PlatformIO and live hardware dependencies should be checked on the defense laptop before
  presentation; the dashboard bundle warning may modestly affect first-load time.
- Local defense startup depends on Docker Desktop/PostgreSQL and a reachable Mosquitto
  broker. Wi-Fi credentials remain intentionally outside version control.

## Future work (not implemented in this closure)

- Phase 12: continual learning with EWC
- Phase 13: model OTA
- Phase 14: extended final evaluation

These are future research/engineering paths, not unfinished requirements of the final
undergraduate implementation.

## Demo checklist

- [ ] Use the defense laptop and disable disruptive updates/notifications.
- [ ] Confirm ESP32 firmware reports `0.3.5-r1` and the intended serial port.
- [ ] Confirm Mosquitto is listening on port 1883 and firmware broker addressing is valid.
- [ ] Run `powershell -ExecutionPolicy Bypass -File .\start-defense.ps1` from repository root.
- [ ] Confirm PostgreSQL health, MQTT persistence, dashboard API, and WebSocket status.
- [ ] Open `http://127.0.0.1:8000` and verify fresh decision and pose telemetry.
- [ ] Demonstrate a clear LOCAL decision and explain cached-result reuse.
- [ ] Demonstrate/describe a controlled CLOUD decision and show exactly ten features-v1.
- [ ] Interrupt Wi-Fi/MQTT/server access and show effective LOCAL plus logged failover.
- [ ] Restore the service and show telemetry/dashboard recovery.
- [ ] Show Split1/2/3 artifacts and explain 64/48/32 versus direct CLOUD's 10 values.
- [ ] Keep screenshots/logs available as fallback if venue networking or hardware fails.
- [ ] Stop owned processes afterward with `.\start-defense.ps1 -Stop`.

## Defense claims checklist

- [x] Gesture recognition is a case study, not the main contribution.
- [x] The learned production policy is binary LOCAL/CLOUD.
- [x] Policy inputs combine uncertainty, network state, and device/resource state.
- [x] Production CLOUD sends exactly ten normalized features-v1 values.
- [x] Production CLOUD does not send raw 100x6 windows.
- [x] Local TinyML and five-pass uncertainty run on ESP32.
- [x] Failover continues with the cached local result and no duplicate inference.
- [x] PostgreSQL, FastAPI/WebSocket, and the implemented dashboard form observability.
- [x] Split paths are validated fixed baselines and negative-result evidence.
- [x] Measured, simulated, estimated, validation, and held-out test values stay distinct.
- [x] Phases 12–14 are future work only.

## Final traffic-light assessment

### GREEN — ready for defense

- Frozen binary R1 production architecture and Phase 11/R2 scope state
- Firmware local inference, uncertainty, CLOUD feature contract, and failover logic
- Server/MQTT schemas, PostgreSQL persistence, API/WebSocket, and dashboard implementation
- Split baselines and evidence-driven production simplification
- Final thesis/report framing and metric provenance
- Python regression suite, dashboard build, and Docker Compose definition

### YELLOW — minor issue / pre-demo check

- PlatformIO is unavailable in the current PATH, so rerun a firmware build on the defense
  laptop before presentation.
- Run the live stack with the actual ESP32, broker, and PostgreSQL once after any laptop or
  network change.
- The dashboard production bundle is large enough to trigger Vite's size warning; preload
  it before the defense. No late refactor is recommended.
- Keep the original project-definition PDF framed as the initial proposal when discussing
  differences from the final evidence-driven R1/R2 architecture.

### RED — must fix before defense

- **None identified after closure.** If the pre-demo live-stack check fails, that specific
  runtime dependency becomes RED until restored or the recorded-evidence fallback is ready.

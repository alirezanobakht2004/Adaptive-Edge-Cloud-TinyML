# Phase 11 / M11 — Database + Dashboard

## Canonical status

**IN PROGRESS — Checkpoint 11.2: FastAPI read API + WebSocket + live dashboard UI.**

Phase 10 / M10 is closed by `docs/phase10_m10_completion.md` and its controlled
ESP32 hardware evidence. This phase does not reopen learned-policy or failover work.

## Checkpoint 11.1 scope

The first Phase-11 checkpoint deliberately implements only the minimum persistent
path needed before a dashboard can be trusted:

```text
ESP32 final decision
    ↓
gesture/{device_id}/telemetry
    ↓
decision-r1-v1 validation
    ↓
R1 MQTT service
    ↓
PostgreSQL inference_events
```

Checkpoint 11.1/11.1b is closed. Hardware persistence is stable and production
`esp32-r1` rows are available in PostgreSQL. Checkpoint 11.2 now adds the read-side
API, live WebSocket stream and dashboard UI without changing the production telemetry
contract or the learned LOCAL/CLOUD policy.

## Production telemetry contract

Version: `decision-r1-v1`

Topic:

```text
gesture/{device_id}/telemetry
```

The event carries final decision/result telemetry only. It does **not** send:

- raw 100×6 IMU windows,
- Split1/2/3 embeddings,
- features-v1 values.

The canonical uncertainty field is the **normalized predictive entropy** in `[0,1]`.
The exact learned-policy state contract is not changed by Phase 11.

Stored fields include:

- request/window/device identity,
- requested and effective `LOCAL/CLOUD` action,
- failover state/reason/stage,
- final class, confidence and uncertainty,
- Wi-Fi/MQTT state,
- last RTT value/source/age when available,
- free heap and measured local prefix+uncertainty inference time,
- application-level CLOUD request elapsed time for issued requests,
- server compute latency when a CLOUD response succeeds,
- CLOUD inference request/response TX/RX bytes (telemetry overhead is not folded into these fields),
- model/policy/firmware versions,
- controlled-vs-production provenance.

Fields that are not actually measured in the current runtime remain `NULL` in the
database. In particular this checkpoint does **not** populate measured energy,
battery, RSSI, free-heap ratio, pure network latency, a full edge-compute total, or
a fabricated total E2E latency. `request_elapsed_ms` includes the server turn and is
therefore not mislabeled as `network_ms`.

## Database schema

Configuration: `config/r1_database_v1.json`

Checkpoint table:

```text
inference_events
```

`split_point` is nullable and stays `NULL` for production R1 rows. Fixed-split
benchmark storage is outside this production decision stream.

The database layer uses SQLAlchemy 2.x and is PostgreSQL-compatible. SQLite is used
only by automated unit tests; it is not the production database claim.

## Local PostgreSQL startup

```powershell
docker compose up -d postgres
$env:TINYML_DATABASE_URL="postgresql+psycopg://tinyml:tinyml_dev@127.0.0.1:5432/tinyml"
```

Install/update Python dependencies:

```powershell
pip install -r requirements.txt
```

Start the existing R1 server with persistence enabled:

```powershell
python -m server.app.r1_mqtt --database-url $env:TINYML_DATABASE_URL
```

The server remains backward-compatible with Phase 10: if no database URL is supplied
and `TINYML_DATABASE_URL` is absent, R1 inference/failover service runs without
persistence.

## Validation commands

Desktop contract/database tests:

```powershell
pytest -q tests/test_phase11_telemetry_schema.py tests/test_phase11_database.py tests/test_phase11_dashboard_api.py tests/test_phase11_pose.py tests/test_r1_mqtt_routing.py tests/test_r1_schema.py
```

After flashing the updated production firmware and allowing inference windows to run:

```powershell
python tools/phase11/validate_database.py --min-events 5
```

For a controlled run where both actions are deliberately exercised:

```powershell
python tools/phase11/validate_database.py --min-events 2 --require-actions LOCAL CLOUD
```

Do not claim both actions were observed unless the actual run produces both.

## Checkpoint 11.1 Definition of Done

Implementation-side:

- [x] versioned `decision-r1-v1` telemetry contract
- [x] production firmware publishes final decision telemetry when MQTT is available
- [x] no raw window / split embedding / features-v1 in telemetry
- [x] PostgreSQL-compatible `inference_events` model
- [x] idempotent `(device_id, request_id)` persistence
- [x] existing R1 service remains usable with database disabled
- [x] desktop telemetry/database tests

Hardware-side (must be measured before checkpoint closure):

- [x] updated firmware builds and uploads on ESP32-S3
- [x] at least several live decision events persist to PostgreSQL
- [x] persisted row fields match serial `R1_DECISION` evidence for sampled windows
- [x] no Phase-9 policy or Phase-10 failover regression

## Known limitation

When Wi-Fi/MQTT is unavailable, the ESP32 cannot publish live dashboard telemetry.
The final decision is still emitted on serial and failover still completes locally.
Offline-event buffering/replay is not added in this checkpoint because it would add
state/memory complexity and is not required to validate the basic Phase-11 path.

## Checkpoint 11.1b — reboot-safe request identity

Live hardware persistence exposed an identity issue that desktop-only tests could not
show: continuous runtime request IDs were derived from `millis()` plus `window_id`,
both of which restart after an ESP32 reboot. Because PostgreSQL correctly enforces
`(device_id, request_id)` uniqueness, a later boot could collide with rows from an
earlier boot and be rejected as `conflicting duplicate decision telemetry`.

The firmware now prefixes continuous-runtime and probe request IDs with a per-boot
`esp_random()` nonce. This is an engineering collision-avoidance identifier, not a
cryptographic-randomness claim. The database uniqueness constraint and conflicting-
duplicate protection remain unchanged. Firmware version is bumped to `0.3.1-r1`.

For hardware closure, validate only the production device stream rather than allowing
a synthetic test row to satisfy the event-count gate:

```powershell
$env:TINYML_DATABASE_URL="postgresql+psycopg://tinyml:tinyml_dev@127.0.0.1:5432/tinyml"
python tools/phase11/validate_database.py --min-events 5 --device-id esp32-r1 --production-only
```

A successful run should have no new `conflicting duplicate decision telemetry` after
a normal device reboot. Historical conflicting-duplicate log lines from the pre-fix
firmware are retained as measured evidence for why this correction was made.

## Checkpoint 11.2 — FastAPI read API + WebSocket + live dashboard

### Technology choice

The canonical architecture says to avoid React unless there is a clear need. This
checkpoint has that clear need: the requested dashboard combines a live WebSocket
stream, several synchronized charts, filters, an event inspector, and an interactive
3D representation of the ESP32-S3 + GY-521 device. The implementation therefore uses:

```text
FastAPI + PostgreSQL
React + TypeScript + Vite
React Three Fiber / Three.js
Recharts
WebSocket
```

This is an implementation-level choice inside Phase 11, not a new policy or split
architecture revision. The repository keeps the canonical `dashboard/` directory.

### Data integrity rules

The dashboard only visualizes fields actually persisted by `decision-r1-v1`. In
particular it does not synthesize RSSI, energy, pure network latency, edge-compute
total, or total E2E latency. CLOUD request elapsed and server compute remain separate
channels. Production R1 remains binary LOCAL/CLOUD and live split-point distribution
is intentionally absent because Split1/2/3 are experimental fixed-split baselines,
not production actions.

The 3D device twin is representative geometry for ESP32-S3 + GY-521/MPU6050 and now
uses a separate auxiliary `pose-v1` stream. The stream is derived from the same already-
acquired 100 Hz IMU samples; it does not add a second sensor read, does not enter the
learned policy, and does not contain raw windows/features-v1. Roll/pitch are lightweight
complementary-filter estimates. Yaw is explicitly boot-relative and drift-prone because
the current MPU6050 path has no magnetometer. These values are therefore never labeled
as absolute/measured heading.

### Read API

Endpoints:

```text
GET /api/dashboard/health
GET /api/dashboard/devices
GET /api/dashboard/events
GET /api/dashboard/events/{id}
GET /api/dashboard/latest
GET /api/dashboard/summary
WS  /ws/dashboard/events
GET /api/dashboard/pose/latest
GET /api/dashboard/pose
WS  /ws/dashboard/pose
```

The WebSocket is database-backed so the MQTT persistence service and FastAPI dashboard
can remain separate laptop-side processes. New rows are streamed without coupling the
MQTT callback directly to UI clients.

### Dashboard panels

The English-only dark command UI includes:

- latest gesture / confidence / uncertainty / execution mode,
- Wi-Fi/MQTT and telemetry freshness,
- sensor-driven interactive 3D device twin with estimated roll/pitch and boot-relative yaw,
- LOCAL/CLOUD distribution,
- confidence vs uncertainty history,
- measured latency channels kept semantically separate,
- RTT and free-heap history,
- observed gesture distribution,
- failover/success summary,
- firmware/model/policy versions,
- searchable production event table and CSV export,
- per-event inspector with measurement provenance.

### Checkpoint 11.2 Definition of Done

- [x] PostgreSQL-backed read API
- [x] production-only/device/action filters
- [x] database-backed live WebSocket stream
- [x] English dark dashboard implementation
- [x] interactive representative 3D device twin
- [x] charts do not relabel unavailable metrics
- [x] backend API/database tests
- [x] frontend dependencies install on the project machine
- [x] `npm run build` succeeds on the project machine
- [x] live browser dashboard receives ESP32 decision events through WebSocket
- [ ] sampled dashboard values match PostgreSQL rows

The decision-dashboard portion of Checkpoint 11.2 passed project-machine build and live-WebSocket checks. The checkpoint remains open for the M11.2c sensor-driven 3D twin hardware/browser validation below and the sampled dashboard-vs-database value check.


## Checkpoint 11.2c — sensor-driven 3D device twin

The earlier M11.2 UI correctly updated decision telemetry but the 3D board itself had a fixed transform. Hardware diagnosis showed this was not a WebSocket rendering defect: the decision stream did not contain orientation data and `DeviceTwin.tsx` intentionally used a constant rotation.

The correction adds a separate, versioned visualization path:

```text
existing MPU6050 100 Hz sample
  -> attitude-complementary-v1
  -> latest-value non-blocking queue
  -> gesture/{device_id}/pose (pose-v1, 10 Hz publish target)
  -> R1 MQTT validation/persistence
  -> PostgreSQL device_pose_events
  -> FastAPI pose read API / pose WebSocket
  -> React Three Fiber quaternion interpolation
```

Integrity constraints:

- production inference remains binary `LOCAL` / `CLOUD`;
- the exact learned-policy state is unchanged;
- no second MPU6050 read is introduced;
- pose MQTT publishing is lower priority than active RTT probes/CLOUD response waits so visualization traffic does not intentionally contaminate policy-network timing;
- no raw 100x6 window, split embedding, or features-v1 is sent in `pose-v1`;
- roll/pitch/yaw are explicitly estimates;
- yaw reference is `boot-relative`, not absolute heading;
- when pose telemetry becomes stale the 3D twin is marked stale/offline instead of pretending the last orientation is live.

Firmware version for the responsiveness-tuned checkpoint: `0.3.3-r1`. The original 5 Hz pose contract remains preserved in `config/r1_pose_v1.json`; the active tuned configuration is versioned separately as `config/r1_pose_v1_1.json` with a 100 ms / 10 Hz publish target. Pose schema and estimator semantics are unchanged.

Hardware Definition of Done remains measured rather than assumed:

- [ ] firmware builds/uploads on ESP32-S3;
- [ ] serial emits periodic finite `R1_POSE` estimates while normal 100 Hz inference continues;
- [ ] `device_pose_events` receives at least 10 real rows;
- [ ] `python tools/phase11/validate_pose.py --device-id esp32-r1 --min-events 10` passes;
- [ ] browser reports `LIVE ATTITUDE`;
- [ ] physical tilt/rotation produces a smooth corresponding change in the 3D twin;
- [ ] stationary home pose remains visually stable;
- [ ] no sampling/inference regression is observed in the existing runtime diagnostics.

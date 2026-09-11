# Phase 11 / M11 — Dashboard Quality Patch v1

## Scope

This patch is intentionally limited to Phase 11 dashboard quality closure. It does **not** change:

- the binary production action space (`LOCAL` / `CLOUD`),
- the learned policy,
- the `features-v1` CLOUD payload,
- MQTT inference contracts,
- PostgreSQL decision/pose schemas,
- the pose-v1 estimator or firmware,
- Split1/2/3 experimental baselines,
- Phase 12 continual-learning work.

## UI version

`dashboard-ui-r1-v2` -> `dashboard-ui-r1-v3`

The matching server health metadata constant is updated in `server/app/dashboard_metrics.py`.

## Changes

1. **Stale/current semantics**
   - Decision freshness is no longer allowed to look current after telemetry becomes stale.
   - Wi-Fi/MQTT status becomes `STALE · LAST ...` instead of continuing to claim a current connection.
   - Device Twin action badge becomes `DECISION STALE` when the latest decision is not fresh.
   - Fresh pose alone no longer paints the twin as current `LOCAL`/`CLOUD`; it uses the neutral tracking state when decision telemetry is stale.

2. **3D twin behavior**
   - Removes the fixed model rotation offset so orientation-v1 zero pose maps to the visual home pose.
   - Keeps the existing roll/pitch/yaw mapping and quaternion smoothing unchanged.
   - Locks camera rotation so mouse orbit cannot be confused with sensor-driven board motion; zoom remains available.
   - Tightens the camera/framing and reduces unused canvas height.

3. **No-data provenance**
   - Summary rates are shown as `—` when there is no summary window instead of fabricating `0.0%`.
   - Action donut center says `no events` when empty.
   - Integrity event count is `—` if summary is unavailable.

4. **RTT provenance**
   - RTT card shows `rtt_source` plus `rtt_age_ms`.
   - Current-decision panel and Event Inspector expose RTT source/measurement age so reused last-successful probes are not mistaken for per-inference RTT measurements.

5. **Charts**
   - Confidence vs. Uncertainty gets an explicit legend.
   - Gesture Distribution always renders the canonical five classes:
     `IDLE`, `SWIPE_LEFT`, `SWIPE_RIGHT`, `ROTATE_CW`, `SHAKE`.
   - Latency panel explicitly states when the current event window contains no CLOUD timing samples.
   - No new latency or energy metric is synthesized.

6. **Decision panel / information hierarchy**
   - Removes the oversized duplicated confidence hero/bars.
   - Keeps latest result, effective action, confidence, uncertainty, failover, measured timing, traffic and RTT provenance in a more compact panel.

7. **Event history**
   - Replaces the nested tall vertical table scrollbar with client-side pagination (12 rows/page).
   - Search/action filters remain intact.
   - CSV export still exports all filtered rows, not only the current page.

8. **Responsiveness**
   - Removes the fixed 1180/1024px body minimum width.
   - Adds controlled desktop/laptop/tablet breakpoints without redesigning the dashboard as a mobile app.

## Local validation performed while preparing the patch

- TypeScript/TSX syntax parse: PASS for all five modified TSX files.
- Phase 11 backend/API/schema subset: `12 passed`.
- A broader local subset reached `22 passed, 1 failed`, where the only failure was an environment dependency error (`ModuleNotFoundError: paho`) while importing `server.app.r1_mqtt`; this was not a product assertion failure.
- Full `npm run typecheck` / `npm run build` were **not** claimed in the patch-preparation environment because npm dependencies were not available there. They must be run in the project environment after applying this overlay.

## Required local validation after applying

From `dashboard/`:

```powershell
npm run typecheck
npm run build
```

From repository root with the venv active:

```powershell
pytest -q `
  tests/test_phase11_telemetry_schema.py `
  tests/test_phase11_database.py `
  tests/test_phase11_dashboard_api.py `
  tests/test_phase11_pose.py `
  tests/test_r1_mqtt_routing.py `
  tests/test_r1_schema.py
```

Then restart FastAPI so `/api/dashboard/health` reports `dashboard-ui-r1-v3`, keep the MQTT service running, open the Vite dashboard, and perform the live hardware regression before declaring M11 closed.

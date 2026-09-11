# Phase 11 Live Dashboard

Technology: **React + TypeScript + Vite + React Three Fiber + Recharts**, backed by the existing **FastAPI + PostgreSQL** server.

React is intentionally used here because the requested UI contains a live WebSocket event stream, several coordinated charts, an interactive 3D device twin, filters, an event inspector, and a production-ready dark command-center layout. This is the explicit "clear need" exception allowed by `PROJECT_ARCHITECTURE.md`.

The 3D device twin is representative geometry for ESP32-S3 + GY-521/MPU6050. It reflects operational state (LOCAL/CLOUD/failover/connectivity) and consumes the separate `pose-v1` stream on `gesture/{device_id}/pose`. Roll/pitch are complementary-filter **estimates**; yaw is explicitly **boot-relative and drift-prone** because no magnetometer is used. `pose-v1` is visualization telemetry only: it is not a learned-policy input and does not contain raw IMU windows or features-v1.

## Development

Run the existing MQTT persistence service in one terminal:

```powershell
$env:TINYML_DATABASE_URL="postgresql+psycopg://tinyml:tinyml_dev@127.0.0.1:5432/tinyml"
python -m server.app.r1_mqtt --database-url $env:TINYML_DATABASE_URL
```

Run the dashboard API in a second terminal:

```powershell
$env:TINYML_DATABASE_URL="postgresql+psycopg://tinyml:tinyml_dev@127.0.0.1:5432/tinyml"
python -m server.app.main --database-url $env:TINYML_DATABASE_URL
```

Run the React dashboard in a third terminal:

```powershell
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Production-like local build

```powershell
cd dashboard
npm install
npm run build
cd ..
python -m server.app.main --database-url $env:TINYML_DATABASE_URL
```

When `dashboard/dist` exists, FastAPI serves it at `http://127.0.0.1:8000`.


## Sensor-driven pose validation

Flash firmware `0.3.2-r1` or later, then keep the MQTT persistence service and dashboard API running. The server creates `device_pose_events` automatically.

```powershell
python tools\phase11\validate_pose.py --device-id esp32-r1 --min-events 10
```

The dashboard should show `LIVE ATTITUDE`, non-empty Roll/Pitch/Yaw readouts, and the 3D board should rotate smoothly as the physical assembly is tilted/rotated. A stale pose stream intentionally changes the twin to a stale/offline state rather than pretending the last orientation is live.

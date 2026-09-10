# Phase 11 Live Dashboard

Technology: **React + TypeScript + Vite + React Three Fiber + Recharts**, backed by the existing **FastAPI + PostgreSQL** server.

React is intentionally used here because the requested UI contains a live WebSocket event stream, several coordinated charts, an interactive 3D device twin, filters, an event inspector, and a production-ready dark command-center layout. This is the explicit "clear need" exception allowed by `PROJECT_ARCHITECTURE.md`.

The 3D device twin is representative geometry for ESP32-S3 + GY-521/MPU6050 and reflects operational state (LOCAL/CLOUD/failover/connectivity). **Physical roll/pitch/yaw is not part of `decision-r1-v1`, so the dashboard does not claim live physical pose.**

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

# Final Defense Verification

Date: 2026-09-12  
Scope: Final implementation scope completed through Phase 11.  
Result: **GREEN — defense package validated and frozen**

## Validation checklist

- [x] repository clean at validation start
- [x] tests passed
- [x] dashboard build passed
- [x] Docker Compose passed
- [x] defense stack passed
- [x] firmware build passed
- [x] ESP32 connected
- [x] MQTT verified
- [x] dashboard verified
- [x] failover verified

## Recorded results

| Check | Result |
|---|---|
| Git baseline | `main` at `5c7f418`, initially clean and synchronized with `origin/main` |
| Python | `343 passed, 7 warnings in 62.19s`; warnings are third-party deprecations |
| Dashboard | `npm run build` passed; Vite reported a non-blocking large-chunk warning |
| Docker | `docker compose config` passed |
| PostgreSQL | launcher health check passed on `127.0.0.1:5432` |
| MQTT | broker reachable on `127.0.0.1:1883`; hotspot address `192.168.137.1` present |
| FastAPI/dashboard | HTTP 200; `/api/dashboard/health` returned `status=ok`, Phase 11 / M11 |
| Firmware build | `esp32-s3-n8r2` passed; 130,268/327,680 RAM bytes (39.8%), 839,421/3,342,336 flash bytes (25.1%) |
| Firmware upload | passed on CH343 `COM10`; flash hashes verified; production `0.3.5-r1` restored after tests |
| Sensor/runtime | MPU6050-compatible `WHO_AM_I=0x74`; 400 kHz I2C, ±4 g, ±500 dps, 100 Hz, 100-sample window, 50-sample step |
| Live policy | learned `meta-policy-v1.0.0`; live LOCAL results observed; production contract states CLOUD sends 10 features-v1 values |
| Live persistence | in a five-second post-restore sample, decision rows increased by 10 and pose rows by 51 |
| Latest production event | `decision-r1-v1`, device `esp32-r1`, firmware `0.3.5-r1`, model `gesture-model-v1.1.0`, connected Wi-Fi/MQTT, successful LOCAL |
| Failover hardware cases | 6/6 Unity cases passed: local, CLOUD success, Wi-Fi fallback, MQTT fallback, server timeout/next window, recovery |
| Failover invariants | test reported 7 windows / 7 local calls; ten-feature CLOUD MQTT requests and server-stop interval independently observed |

## Live chain verified

```text
ESP32
  -> MQTT
  -> R1 inference/persistence service
  -> PostgreSQL
  -> FastAPI/WebSocket
  -> dashboard
```

The dashboard and API served live data at `http://127.0.0.1:8000`. During validation,
launcher-owned service logs were located under `.defense/logs`; these runtime logs are
ignored and removed during final cleanup.

## Failover validation note

The failover test must be coordinated with
`python -m tools.failover.validate_hardware`; plain `pio test` cannot react to the
firmware's `M10_SERVER_STOP_REQUEST`. In the coordinated run, the on-device Unity suite
passed all six cases. The validator's legacy post-processing returned a nonzero exit
afterward because firmware `0.3.5-r1` intentionally suppresses verbose `R1_DECISION`
serial JSON to protect pose cadence, while the wrapper still expects seven serial JSON
records. Its independent MQTT checks (`broker_received_during_service_stop`,
`no_timeout_response`, and `ten_feature_payloads`) passed. Existing canonical Phase 10
evidence was preserved rather than replaced by this incomplete wrapper report.

## Architecture freeze confirmation

- Production actions remain exactly `LOCAL` and `CLOUD`.
- CLOUD sends exactly ten normalized `features-v1` values.
- No production raw 100x6 IMU window transport was introduced.
- Split1/2/3 remain 64-D/48-D/32-D validated fixed experimental baselines and
  negative-result evidence.
- No model, policy, dataset, or production action-space artifact changed.
- Phases 12–14 remain future work only.

## Service state after verification

The launcher was stopped and ignored runtime/build outputs were removed for packaging.
For the defense, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-defense.ps1
```

The production firmware remains flashed on the connected ESP32-S3.

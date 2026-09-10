# Phase 11 / M11.2 — WebSocket Runtime + Polling Fallback Fix

Changes:
- Adds the `websockets` runtime dependency required by Uvicorn for WebSocket upgrades.
- Runs Uvicorn with the `websockets-sansio` protocol explicitly.
- Adds a 5-second REST event polling fallback while WebSocket is disconnected, so the dashboard remains live-capable even if WS is temporarily unavailable.

Validation after applying:
1. `pip install -r requirements.txt`
2. `cd dashboard && npm run typecheck && npm run build`
3. Restart `python -m server.app.main ...`
4. Confirm server log shows `WebSocket /ws/dashboard/events ... [accepted]` / `101 Switching Protocols` rather than HTTP 404.
5. Confirm UI status changes from `Reconnecting` to `WebSocket live` and latest event advances without manual refresh.

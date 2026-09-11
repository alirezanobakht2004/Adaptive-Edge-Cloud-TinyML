"""Phase 11 / M11.2 FastAPI read API + live WebSocket dashboard service."""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .dashboard_metrics import DASHBOARD_API_VERSION, DASHBOARD_UI_VERSION, build_summary, event_to_dict, pose_to_dict
from .database import DATABASE_URL_ENV, Database, database_url_from_env

APP_VERSION = "0.3.0"
PHASE = "11"
MILESTONE = "M11"
CHECKPOINT = "M11.2"
DEFAULT_WS_POLL_MS = 750
DEFAULT_POSE_WS_POLL_MS = 50


def create_app(database_url: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        url = database_url or database_url_from_env()
        db: Database | None = None
        if url:
            db = Database(url)
            db.create_schema()
        app.state.database = db
        yield
        if db is not None:
            db.close()

    app = FastAPI(
        title="Adaptive Edge-Cloud TinyML Dashboard API",
        version=APP_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def require_db() -> Database:
        db = getattr(app.state, "database", None)
        if db is None:
            raise HTTPException(
                status_code=503,
                detail=f"Database is not configured. Set {DATABASE_URL_ENV} or pass --database-url.",
            )
        return db

    @app.get("/health")
    @app.get("/api/dashboard/health")
    def health() -> dict[str, Any]:
        db = getattr(app.state, "database", None)
        return {
            "status": "ok" if db is not None else "degraded",
            "phase": PHASE,
            "milestone": MILESTONE,
            "checkpoint": CHECKPOINT,
            "server_version": APP_VERSION,
            "api_version": DASHBOARD_API_VERSION,
            "ui_version": DASHBOARD_UI_VERSION,
            "database_configured": db is not None,
            "event_count": db.count_inference_events() if db is not None else None,
            "pose_event_count": db.count_pose_events() if db is not None else None,
        }

    @app.get("/api/dashboard/devices")
    def devices(production_only: bool = True) -> dict[str, Any]:
        db = require_db()
        return {"devices": db.list_devices(production_only=production_only)}

    @app.get("/api/dashboard/events")
    def events(
        limit: int = Query(default=240, ge=1, le=2000),
        device_id: str | None = None,
        action: str | None = Query(default=None, pattern="^(LOCAL|CLOUD)$"),
        gesture: str | None = None,
        production_only: bool = True,
        before_id: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        db = require_db()
        rows = db.query_inference_events(
            limit=limit,
            device_id=device_id,
            action=action,
            gesture=gesture,
            production_only=production_only,
            before_id=before_id,
        )
        return {
            "events": [event_to_dict(row) for row in rows],
            "count": len(rows),
            "next_before_id": rows[-1].id if rows else None,
        }

    @app.get("/api/dashboard/events/{event_id}")
    def event_detail(event_id: int) -> dict[str, Any]:
        db = require_db()
        rows = db.query_inference_events(limit=1, production_only=False, before_id=event_id + 1)
        row = next((item for item in rows if item.id == event_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return event_to_dict(row, include_raw=True)

    @app.get("/api/dashboard/latest")
    def latest(device_id: str | None = None, production_only: bool = True):
        db = require_db()
        row = db.latest_inference_event(device_id=device_id, production_only=production_only)
        if row is None:
            return JSONResponse(status_code=404, content={"detail": "No matching events"})
        return event_to_dict(row, include_raw=True)

    @app.get("/api/dashboard/pose/latest")
    def latest_pose(device_id: str | None = None):
        db = require_db()
        row = db.latest_pose_event(device_id=device_id)
        if row is None:
            return JSONResponse(status_code=404, content={"detail": "No matching pose telemetry"})
        return pose_to_dict(row, include_raw=True)

    @app.get("/api/dashboard/pose")
    def pose_history(
        limit: int = Query(default=120, ge=1, le=2000),
        device_id: str | None = None,
        before_id: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        db = require_db()
        rows = db.query_pose_events(limit=limit, device_id=device_id, before_id=before_id)
        return {
            "poses": [pose_to_dict(row) for row in rows],
            "count": len(rows),
            "next_before_id": rows[-1].id if rows else None,
        }

    @app.get("/api/dashboard/summary")
    def summary(
        device_id: str | None = None,
        production_only: bool = True,
        window_events: int = Query(default=500, ge=1, le=2000),
    ) -> dict[str, Any]:
        db = require_db()
        rows = db.query_inference_events(
            limit=window_events,
            device_id=device_id,
            production_only=production_only,
        )
        result = build_summary(rows)
        result.update({
            "device_id": device_id,
            "production_only": production_only,
            "requested_window_events": window_events,
        })
        return result

    @app.websocket("/ws/dashboard/events")
    async def websocket_events(
        websocket: WebSocket,
        device_id: str | None = None,
        production_only: bool = True,
        after_id: int = 0,
    ) -> None:
        db = getattr(app.state, "database", None)
        await websocket.accept()
        if db is None:
            await websocket.send_json({"type": "error", "detail": "database_not_configured"})
            await websocket.close(code=1011)
            return

        cursor = max(0, after_id)
        await websocket.send_json({
            "type": "ready",
            "api_version": DASHBOARD_API_VERSION,
            "device_id": device_id,
            "production_only": production_only,
            "after_id": cursor,
        })
        try:
            while True:
                rows = await asyncio.to_thread(
                    db.query_inference_events,
                    limit=200,
                    device_id=device_id,
                    production_only=production_only,
                    after_id=cursor,
                    ascending=True,
                )
                for row in rows:
                    cursor = max(cursor, row.id)
                    await websocket.send_json({"type": "event", "event": event_to_dict(row)})
                await asyncio.sleep(DEFAULT_WS_POLL_MS / 1000.0)
        except WebSocketDisconnect:
            return

    @app.websocket("/ws/dashboard/pose")
    async def websocket_pose(
        websocket: WebSocket,
        device_id: str | None = None,
        after_id: int = 0,
    ) -> None:
        db = getattr(app.state, "database", None)
        await websocket.accept()
        if db is None:
            await websocket.send_json({"type": "error", "detail": "database_not_configured"})
            await websocket.close(code=1011)
            return

        cursor = max(0, after_id)
        await websocket.send_json({
            "type": "ready",
            "api_version": DASHBOARD_API_VERSION,
            "device_id": device_id,
            "after_id": cursor,
        })
        try:
            while True:
                rows = await asyncio.to_thread(
                    db.query_pose_events,
                    limit=100,
                    device_id=device_id,
                    after_id=cursor,
                    ascending=True,
                )
                for row in rows:
                    cursor = max(cursor, row.id)
                    await websocket.send_json({"type": "pose", "pose": pose_to_dict(row)})
                await asyncio.sleep(DEFAULT_POSE_WS_POLL_MS / 1000.0)
        except WebSocketDisconnect:
            return

    repo_root = Path(__file__).resolve().parents[2]
    dashboard_dist = repo_root / "dashboard" / "dist"
    if dashboard_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dashboard_dist), html=True), name="dashboard")
    else:
        @app.get("/")
        def root() -> dict[str, Any]:
            return {
                "service": "Adaptive Edge-Cloud TinyML Dashboard API",
                "dashboard_build": "not_found",
                "hint": "Run npm install && npm run build inside dashboard/, or npm run dev for Vite development mode.",
            }

    return app


app = create_app()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 11 dashboard API")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.database_url:
        os.environ[DATABASE_URL_ENV] = args.database_url
    uvicorn.run(
        create_app(args.database_url),
        host=args.host,
        port=args.port,
        log_level="info",
        ws="websockets-sansio",
    )


if __name__ == "__main__":
    _cli()

"""Minimal FastAPI application for Phase 6 / M7."""

from __future__ import annotations

from fastapi import FastAPI


APP_VERSION = "0.1.0"

app = FastAPI(
    title="Adaptive Edge-Cloud TinyML Server",
    version=APP_VERSION,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "phase": "6",
        "milestone": "M7",
        "server_version": APP_VERSION,
    }

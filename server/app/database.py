"""Minimal PostgreSQL-compatible persistence layer for Phase 11 / M11."""

from __future__ import annotations

import os
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base, InferenceEvent
from .telemetry_schema import class_name, parse_decision_event

DATABASE_URL_ENV = "TINYML_DATABASE_URL"
DEFAULT_POSTGRES_URL = "postgresql+psycopg://tinyml:tinyml_dev@127.0.0.1:5432/tinyml"


class Database:
    """Small synchronous store suitable for the laptop-side MVP server."""

    def __init__(self, url: str):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("database URL must be non-empty")
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite") and ":memory:" in url:
            kwargs.update({"connect_args": {"check_same_thread": False}, "poolclass": StaticPool})
        self.url = url
        self.engine: Engine = create_engine(url, **kwargs)
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def save_decision_event(self, payload) -> InferenceEvent:
        data = parse_decision_event(payload)
        with self.sessions() as session:
            existing = session.scalar(
                select(InferenceEvent).where(
                    InferenceEvent.device_id == data["device_id"],
                    InferenceEvent.request_id == data["request_id"],
                )
            )
            if existing is not None:
                if existing.raw_event != data:
                    raise ValueError("conflicting duplicate decision telemetry")
                return existing

            event = InferenceEvent(
                telemetry_schema_version=data["schema_version"],
                device_id=data["device_id"],
                request_id=data["request_id"],
                timestamp_ms=data["timestamp_ms"],
                window_id=data["window_id"],
                predicted_class=class_name(data["predicted_class_id"]),
                predicted_class_id=data["predicted_class_id"],
                confidence=data["confidence"],
                uncertainty=data["uncertainty"],
                requested_action=data["requested_action"],
                execution_mode=data["effective_action"],
                split_point=None,
                failover=data["failover"],
                failure_reason=data["failover_reason"],
                failure_stage=data["failure_stage"],
                wifi_connected=data["wifi_connected"],
                mqtt_connected=data["mqtt_connected"],
                rssi=None,
                rtt_ms=data["rtt_ms"],
                rtt_source=data["rtt_source"],
                rtt_age_ms=data["rtt_age_ms"],
                free_heap_bytes=data["free_heap_bytes"],
                free_heap_ratio=None,
                energy_budget=None,
                local_inference_ms=data["local_inference_ms"],
                request_elapsed_ms=data["request_elapsed_ms"],
                edge_compute_ms=None,
                network_ms=None,
                server_compute_ms=data["server_compute_ms"],
                total_latency_ms=None,
                bytes_tx=data["bytes_tx"],
                bytes_rx=data["bytes_rx"],
                model_version=data["model_version"],
                policy_version=data["policy_version"],
                firmware_version=data["firmware_version"],
                success=data["success"],
                controlled=data["controlled"],
                raw_event=dict(data),
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def count_inference_events(self) -> int:
        with self.sessions() as session:
            return int(session.scalar(select(func.count()).select_from(InferenceEvent)) or 0)

    def recent_inference_events(self, limit: int = 50) -> list[InferenceEvent]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be in [1, 1000]")
        with self.sessions() as session:
            return list(
                session.scalars(
                    select(InferenceEvent).order_by(InferenceEvent.id.desc()).limit(limit)
                )
            )

    def close(self) -> None:
        self.engine.dispose()


def database_url_from_env() -> str | None:
    value = os.getenv(DATABASE_URL_ENV)
    return value.strip() if value and value.strip() else None

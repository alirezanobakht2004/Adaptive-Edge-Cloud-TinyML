"""PostgreSQL-compatible persistence/read layer for Phase 11 / M11."""

from __future__ import annotations

import os
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base, DevicePoseEvent, InferenceEvent
from .telemetry_schema import class_name, parse_decision_event
from .pose_schema import parse_pose_event

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


    def save_pose_event(self, payload) -> DevicePoseEvent:
        data = parse_pose_event(payload)
        with self.sessions() as session:
            existing = session.scalar(
                select(DevicePoseEvent).where(
                    DevicePoseEvent.device_id == data["device_id"],
                    DevicePoseEvent.pose_id == data["pose_id"],
                )
            )
            if existing is not None:
                if existing.raw_event != data:
                    raise ValueError("conflicting duplicate pose telemetry")
                return existing

            event = DevicePoseEvent(
                pose_schema_version=data["schema_version"],
                pose_id=data["pose_id"],
                device_id=data["device_id"],
                timestamp_ms=data["timestamp_ms"],
                sequence=data["sequence"],
                roll_deg_est=data["roll_deg_est"],
                pitch_deg_est=data["pitch_deg_est"],
                yaw_rel_deg_est=data["yaw_rel_deg_est"],
                estimator_version=data["estimator_version"],
                orientation_version=data["orientation_version"],
                firmware_version=data["firmware_version"],
                source=data["source"],
                yaw_reference=data["yaw_reference"],
                raw_event=dict(data),
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def count_pose_events(self, *, device_id: str | None = None) -> int:
        statement = select(func.count()).select_from(DevicePoseEvent)
        if device_id:
            statement = statement.where(DevicePoseEvent.device_id == device_id)
        with self.sessions() as session:
            return int(session.scalar(statement) or 0)

    def query_pose_events(
        self,
        *,
        limit: int = 100,
        device_id: str | None = None,
        after_id: int | None = None,
        before_id: int | None = None,
        ascending: bool = False,
    ) -> list[DevicePoseEvent]:
        if type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError("limit must be in [1, 2000]")
        statement = select(DevicePoseEvent)
        if device_id:
            statement = statement.where(DevicePoseEvent.device_id == device_id)
        if after_id is not None:
            statement = statement.where(DevicePoseEvent.id > after_id)
        if before_id is not None:
            statement = statement.where(DevicePoseEvent.id < before_id)
        order = DevicePoseEvent.id.asc() if ascending else DevicePoseEvent.id.desc()
        statement = statement.order_by(order).limit(limit)
        with self.sessions() as session:
            return list(session.scalars(statement))

    def latest_pose_event(self, *, device_id: str | None = None) -> DevicePoseEvent | None:
        rows = self.query_pose_events(limit=1, device_id=device_id)
        return rows[0] if rows else None

    def count_inference_events(self) -> int:
        with self.sessions() as session:
            return int(session.scalar(select(func.count()).select_from(InferenceEvent)) or 0)

    def recent_inference_events(self, limit: int = 50) -> list[InferenceEvent]:
        return self.query_inference_events(limit=limit)

    def query_inference_events(
        self,
        *,
        limit: int = 100,
        device_id: str | None = None,
        action: str | None = None,
        gesture: str | None = None,
        production_only: bool = True,
        after_id: int | None = None,
        before_id: int | None = None,
        ascending: bool = False,
    ) -> list[InferenceEvent]:
        if type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError("limit must be in [1, 2000]")
        if action is not None and action not in {"LOCAL", "CLOUD"}:
            raise ValueError("action must be LOCAL or CLOUD")

        statement = select(InferenceEvent)
        if device_id:
            statement = statement.where(InferenceEvent.device_id == device_id)
        if action:
            statement = statement.where(InferenceEvent.execution_mode == action)
        if gesture:
            statement = statement.where(InferenceEvent.predicted_class == gesture)
        if production_only:
            statement = statement.where(InferenceEvent.controlled.is_(False))
        if after_id is not None:
            statement = statement.where(InferenceEvent.id > after_id)
        if before_id is not None:
            statement = statement.where(InferenceEvent.id < before_id)

        order = InferenceEvent.id.asc() if ascending else InferenceEvent.id.desc()
        statement = statement.order_by(order).limit(limit)
        with self.sessions() as session:
            return list(session.scalars(statement))

    def latest_inference_event(
        self, *, device_id: str | None = None, production_only: bool = True
    ) -> InferenceEvent | None:
        rows = self.query_inference_events(
            limit=1,
            device_id=device_id,
            production_only=production_only,
        )
        return rows[0] if rows else None

    def list_devices(self, *, production_only: bool = True) -> list[dict[str, object]]:
        statement = select(
            InferenceEvent.device_id,
            func.count(InferenceEvent.id),
            func.max(InferenceEvent.id),
            func.max(InferenceEvent.received_at),
        )
        if production_only:
            statement = statement.where(InferenceEvent.controlled.is_(False))
        statement = statement.group_by(InferenceEvent.device_id).order_by(func.max(InferenceEvent.id).desc())
        with self.sessions() as session:
            rows = session.execute(statement).all()
        return [
            {
                "device_id": device_id,
                "event_count": int(event_count),
                "latest_event_id": int(latest_event_id),
                "last_seen": last_seen.isoformat() if last_seen is not None else None,
            }
            for device_id, event_count, latest_event_id, last_seen in rows
        ]

    def close(self) -> None:
        self.engine.dispose()


def database_url_from_env() -> str | None:
    value = os.getenv(DATABASE_URL_ENV)
    return value.strip() if value and value.strip() else None

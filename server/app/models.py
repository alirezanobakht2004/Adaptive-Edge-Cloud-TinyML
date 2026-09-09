"""SQLAlchemy models for Phase 11 / M11.

Checkpoint 11.1 intentionally starts with the production inference event stream.
The remaining canonical Phase-11 tables can be added after this path is hardware
validated, without coupling database work to the later continual-learning phase.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class InferenceEvent(Base):
    __tablename__ = "inference_events"
    __table_args__ = (
        UniqueConstraint("device_id", "request_id", name="uq_inference_events_device_request"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    telemetry_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    device_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(96), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    window_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    predicted_class: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    true_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    uncertainty: Mapped[float] = mapped_column(Float, nullable=False)

    requested_action: Mapped[str] = mapped_column(String(24), nullable=False)
    execution_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    split_point: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failover: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failure_reason: Mapped[str] = mapped_column(String(48), nullable=False, default="NONE")
    failure_stage: Mapped[str] = mapped_column(String(48), nullable=False, default="NONE")

    wifi_connected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    mqtt_connected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rssi: Mapped[float | None] = mapped_column(Float, nullable=True)
    rtt_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rtt_source: Mapped[str] = mapped_column(String(48), nullable=False)
    rtt_age_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    free_heap_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    free_heap_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_budget: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Current firmware measures local model/uncertainty time and application-level
    # CLOUD request elapsed time. These are kept distinct from canonical edge/network
    # totals so Phase 11 does not relabel partial timings as E2E/network latency.
    local_inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    request_elapsed_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge_compute_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    network_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    server_compute_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    bytes_tx: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bytes_rx: Mapped[int] = mapped_column(BigInteger, nullable=False)

    model_version: Mapped[str] = mapped_column(String(96), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(96), nullable=False)
    firmware_version: Mapped[str] = mapped_column(String(96), nullable=False)

    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    controlled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw_event: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

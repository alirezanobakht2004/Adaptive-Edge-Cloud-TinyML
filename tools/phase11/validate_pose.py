"""Validate persisted Phase-11 sensor-driven pose telemetry."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app.database import DATABASE_URL_ENV, Database
from server.app.pose_schema import ESTIMATOR_VERSION, ORIENTATION_VERSION, POSE_SCHEMA_VERSION, POSE_SOURCE, YAW_REFERENCE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv(DATABASE_URL_ENV))
    parser.add_argument("--device-id", default="esp32-r1")
    parser.add_argument("--min-events", type=int, default=10)
    args = parser.parse_args()
    if not args.database_url:
        parser.error(f"--database-url or {DATABASE_URL_ENV} is required")
    if args.min_events < 1:
        parser.error("--min-events must be >= 1")

    db = Database(args.database_url)
    try:
        db.create_schema()
        total = db.count_pose_events(device_id=args.device_id)
        rows = db.query_pose_events(limit=min(max(total, 1), 1000), device_id=args.device_id)
        checks = {
            "min_pose_events": len(rows) >= args.min_events,
            "pose_schema": all(row.pose_schema_version == POSE_SCHEMA_VERSION for row in rows),
            "estimator_version": all(row.estimator_version == ESTIMATOR_VERSION for row in rows),
            "orientation_version": all(row.orientation_version == ORIENTATION_VERSION for row in rows),
            "pose_source": all(row.source == POSE_SOURCE for row in rows),
            "yaw_is_boot_relative": all(row.yaw_reference == YAW_REFERENCE for row in rows),
            "finite_pose": all(
                math.isfinite(row.roll_deg_est)
                and math.isfinite(row.pitch_deg_est)
                and math.isfinite(row.yaw_rel_deg_est)
                for row in rows
            ),
        }
        print(f"PHASE11_POSE_DEVICE={args.device_id}")
        print(f"PHASE11_POSE_TOTAL_EVENTS={total}")
        print(f"PHASE11_POSE_EVENTS_CHECKED={len(rows)}")
        if rows:
            latest = rows[0]
            print(f"PHASE11_POSE_LATEST_ID={latest.id}")
            print(f"PHASE11_POSE_LATEST_SEQUENCE={latest.sequence}")
            print(f"PHASE11_POSE_FIRMWARE={latest.firmware_version}")
        for key, value in checks.items():
            print(f"{key}={'PASS' if value else 'FAIL'}")
        return 0 if all(checks.values()) else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

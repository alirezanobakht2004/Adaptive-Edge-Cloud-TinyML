"""Validate the Phase-11 inference-event persistence after a live/controlled run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app.database import DATABASE_URL_ENV, Database


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv(DATABASE_URL_ENV))
    parser.add_argument("--min-events", type=int, default=1)
    parser.add_argument("--require-actions", nargs="*", choices=("LOCAL", "CLOUD"), default=[])
    parser.add_argument("--device-id", default=None, help="Only validate rows for this device_id")
    parser.add_argument("--production-only", action="store_true",
                        help="Exclude rows explicitly marked controlled=True")
    args = parser.parse_args()
    if not args.database_url:
        parser.error(f"--database-url or {DATABASE_URL_ENV} is required")
    if args.min_events < 1:
        parser.error("--min-events must be >= 1")

    db = Database(args.database_url)
    try:
        db.create_schema()
        total_count = db.count_inference_events()
        rows = db.recent_inference_events(min(max(total_count, 1), 1000))
        if args.device_id is not None:
            rows = [row for row in rows if row.device_id == args.device_id]
        if args.production_only:
            rows = [row for row in rows if not row.controlled]
        count = len(rows)
        actions = {row.execution_mode for row in rows}
        checks = {
            "min_events": count >= args.min_events,
            "required_actions": set(args.require_actions).issubset(actions),
            "binary_execution_only": all(row.execution_mode in {"LOCAL", "CLOUD"} for row in rows),
            "no_split_in_production_rows": all(row.split_point is None for row in rows),
            "no_measured_energy_claim": all(row.energy_budget is None for row in rows),
        }
        print(f"PHASE11_DB_TOTAL_EVENTS={total_count}")
        print(f"PHASE11_DB_EVENTS={count}")
        print(f"PHASE11_DB_DEVICE_FILTER={args.device_id or 'ALL'}")
        print(f"PHASE11_DB_PRODUCTION_ONLY={'YES' if args.production_only else 'NO'}")
        print(f"PHASE11_DB_ACTIONS={','.join(sorted(actions)) or 'NONE'}")
        for key, value in checks.items():
            print(f"{key}={'PASS' if value else 'FAIL'}")
        return 0 if all(checks.values()) else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

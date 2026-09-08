"""Validate JSONL inputs and write a reproducible collection-quality report."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .schema import VERSION, ACTION_NAMES
from .serializer import deserialize


def generate_report(paths, output):
    seen, records, errors, inputs = set(), [], [], []
    duplicate_count = 0
    for path in map(Path, paths):
        inputs.append({"path": path.as_posix(),
                       "sha256_lf_utf8": hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()})
        if (path.parent / "configuration.json").exists():
            from .validator import validate_campaign
            try:
                validate_campaign(path.parent)
            except (ValueError, KeyError, OSError) as exc:
                errors.append({"path": path.parent.as_posix(), "error": str(exc)})
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = deserialize(line)
                if record["metadata"]["record_kind"] != "observation":
                    raise ValueError("Example cannot enter a collected dataset")
                key = record["metadata"]["run_id"], record["metadata"]["sample_id"]
                if key in seen:
                    duplicate_count += 1
                    raise ValueError("Duplicate run/sample identity")
                seen.add(key)
                records.append(record)
            except (ValueError, TypeError) as exc:
                errors.append({"path": path.as_posix(), "line": number, "error": str(exc)})
    actions = Counter(record["action"] for record in records)
    report = {"dataset_version": VERSION, "valid": not errors, "inputs": inputs,
              "valid_observations": len(records), "duplicate_samples": duplicate_count, "errors": errors,
              "actions": {ACTION_NAMES[k]: actions[k] for k in ACTION_NAMES},
              "scopes": dict(Counter(r["metadata"]["measurement_scope"] for r in records)),
              "outcomes": dict(Counter(r["outcome"]["status"] for r in records)),
              "labeled_observations": sum(r["outcome"]["true_class"] is not None for r in records),
              "missing_measurements": {key: sum(r["measurements"][key] is None for r in records)
                                       for key in (records[0]["measurements"] if records else [])},
              "energy": "estimated/simulated only; null means unconfigured", "training_ready": False,
              "limitations": ["Schema validity does not establish policy-training readiness.",
                              "Repeated windows in different runs are allowed; split future training by session/window."]}
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", default="dataset_report.json")
    args = parser.parse_args()
    report = generate_report(args.input, args.output)
    print(json.dumps({"valid": report["valid"], "observations": report["valid_observations"]}))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()

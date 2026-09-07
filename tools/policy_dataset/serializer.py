"""Strict JSON serialization; nonfinite values and duplicate keys are rejected."""

import json

from .schema import validate_record


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def deserialize(text):
    record = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    validate_record(record)
    return record


def serialize(record):
    validate_record(record)
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"

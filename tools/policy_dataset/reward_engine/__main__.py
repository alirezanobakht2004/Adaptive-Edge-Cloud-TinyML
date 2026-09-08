"""python -m tools.policy_dataset.reward_engine --input groups.jsonl --config calibration.json --output rewards.jsonl"""

import argparse
import json
from pathlib import Path

from .engine import evaluate_group, validate_config


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def calibrate_file(source, config_path, output):
    config = strict_json(Path(config_path).read_text(encoding="utf-8"))
    validate_config(config)
    results, seen = [], set()
    for line_number, line in enumerate(Path(source).read_text(encoding="utf-8").splitlines(), 1):
        try:
            result = evaluate_group(strict_json(line), config)
            if result["decision_id"] in seen:
                raise ValueError("Duplicate decision_id")
            seen.add(result["decision_id"])
            results.append(result)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc
    if not results:
        raise ValueError("No candidate groups; no labels generated")
    # Validate the entire batch before creating anything; never overwrite inputs.
    with Path(output).open("x", encoding="utf-8", newline="\n") as stream:
        for result in results:
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    return len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Candidate groups calibrated: {calibrate_file(args.input, args.config, args.output)}")


if __name__ == "__main__":
    main()

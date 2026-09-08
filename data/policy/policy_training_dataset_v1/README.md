# policy_training_dataset_v1

Phase9 now provides **100 physical ESP32 observations**, 25 each for ALL_LOCAL,
SPLIT1, SPLIT2 and ALL_CLOUD. `policy_training_dataset_v1.jsonl` is a snapshot of
the same records in `observations.jsonl`; do not count both. Per-run originals,
raw device traces, MQTT server events and configurations are under `campaigns/`.
`campaigns/local/` retains a failed serial startup with zero observations;
the successful local campaign is `campaigns/local_run01/`.

See `dataset_report.json`, `campaign_report.json`, `benchmark_report.json` and
`docs/phase9_dataset_collection.md`. These are 25 unique existing validation
windows replayed across four actions, not new live sensor windows. Rule-based
execution is unmeasured, energy is null, and policy training readiness is false.

Versioned collection directory. sample_000001.json is an explicitly unmeasured example; observations.jsonl began empty in Phase8 and accepts observations only. Use the validator for its current count. Collection alone does not establish policy-training readiness.

Actions: 0 ALL_LOCAL, 1 SPLIT1, 2 SPLIT2, 3 ALL_CLOUD. Action 3 is NOT MQTT split=3.

Unknown metrics are null. Energy is an estimated/simulated proxy, never battery measured. Timestamps name their clock domain; host measurements are not ESP32 measurements.

Validate: `python -m tools.policy_dataset.collector validate`. Import validated JSONL: `python -m tools.policy_dataset.collector collect --input records.jsonl`. Use --directory to select another dataset directory. One writer only.

See docs/phase8_policy_dataset.md for the contract and instrumentation limits.

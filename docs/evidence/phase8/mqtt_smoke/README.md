# policy_training_dataset_v1

Versioned collection directory. sample_000001.json is an explicitly unmeasured example; observations.jsonl starts empty and accepts observations only. Use the validator for its current count. Collection alone does not establish benchmark results or policy-training readiness.

Actions: 0 ALL_LOCAL, 1 SPLIT1, 2 SPLIT2, 3 ALL_CLOUD. Action 3 is NOT MQTT split=3.

Unknown metrics are null. Energy is an estimated/simulated proxy, never battery measured. Timestamps name their clock domain; host measurements are not ESP32 measurements.

Validate: `python -m tools.policy_dataset.collector validate`. Import validated JSONL: `python -m tools.policy_dataset.collector collect --input records.jsonl`. Use --directory to select another dataset directory. One writer only.

See docs/phase8_policy_dataset.md for the contract and instrumentation limits.

# Phase9 first dataset collection completion

Completed the requested M14-M18 first collection and report infrastructure on
2026-09-08. This closes the initial collection scope, not learned-policy training
or a comprehensive five-baseline benchmark study.

| Milestone | Delivered |
|---|---|
| M14 | Physical ESP32 state/action traces, immutable v1 projection and raw sidecars |
| M15 | Executed local, Split1, Split2 and genuine ten-feature ALL_CLOUD paths |
| M16 | Repeatable CLI campaigns with configuration, versions, hashes and partial-run preservation |
| M17 | Schema/version/action/duplicate checks, manifest/trace consistency, dataset reports |
| M18 | Measured four-action summary; rule-based baseline explicitly missing, energy null |

## Evidence

- **100 real device decision records**, 25 per action; **100 successful executions**.
- **25 unique validation windows**, five per class, replayed under each action.
- `data/policy/policy_training_dataset_v1/dataset_report.json`: valid, zero
  duplicate identities, 100 labeled device observations. `training_ready=false`.
- `campaign_report.json`: all four run manifests and raw projections validated.
- Full `pytest -q`: **170 passed**, four existing TensorFlow Lite deprecation warnings.
- `python -m py_compile`: passed for repository Python files.
- `git diff --check` and staged whitespace checks passed before commits.
- Existing Phase7 hardware regression: **six suites passed**, covering all three
  parity tests and all three MQTT E2E tests, five vectors each. See
  `docs/evidence/phase9/phase7_regression.log`.
- Isolated campaign prefix parity passed. The initial serial startup attempt
  collected zero records because Unity closes Serial; the isolated image now
  reopens it. The corrected hardware run is recorded in
  `docs/evidence/phase9/campaign_serial_hardware.log`.
- Physical MQTT campaigns validated **75 remote round trips**, including 25
  full-cloud executions. Raw traces and server events are retained per campaign.

No validated production firmware sources, split models, dataset-v1, features-v1,
sampling rate, gesture classes or production server routing were changed. The
connected board is left running the isolated campaign test image.

## Changed implementation and artifacts

- `firmware/test/test_phase9_campaign/test_main.cpp`
- `server/app/cloud_full.py`
- `tools/policy_dataset/device_trace.py`, `campaign_server.py`, `run_campaign.py`
- `tools/policy_dataset/validator.py`, `dataset_report.py`, `benchmark_report.py`
- `tests/test_policy_device_trace.py`, `test_policy_campaign.py`,
  `test_policy_dataset_report.py`, `test_policy_benchmark_report.py`
- `data/policy/policy_training_dataset_v1/`: assembled JSONL, per-run records,
  raw traces, server events, configurations and validation/benchmark reports
- `docs/evidence/phase9/`, `docs/phase9_dataset_collection.md`, this completion
  document, and the dataset README

## Commands executed

The repository `.venv\Scripts\python.exe` and existing PlatformIO executable
were used. Python validation ran before each requested incremental commit.

```powershell
python -m py_compile <repository Python files>
python -m pytest -q
python -u -m tests.run_phase7_hardware --port COM10 --filter "test_phase7_*"
pio test -d firmware -e esp32-s3-n8r2 -f test_phase9_campaign --upload-port COM10 --test-port COM10 -v
python -m tools.policy_dataset.run_campaign --samples 25 --mode local --output data/policy/policy_training_dataset_v1/campaigns/local_run01
python -m tools.policy_dataset.run_campaign --samples 25 --mode split1 --output data/policy/policy_training_dataset_v1/campaigns/split1
python -m tools.policy_dataset.run_campaign --samples 25 --mode split2 --output data/policy/policy_training_dataset_v1/campaigns/split2
python -m tools.policy_dataset.run_campaign --samples 25 --mode cloud --output data/policy/policy_training_dataset_v1/campaigns/cloud
python -m tools.policy_dataset.dataset_report --input data/policy/policy_training_dataset_v1/policy_training_dataset_v1.jsonl --output data/policy/policy_training_dataset_v1/dataset_report.json
python -m tools.policy_dataset.benchmark_report --input data/policy/policy_training_dataset_v1/policy_training_dataset_v1.jsonl --output data/policy/policy_training_dataset_v1/benchmark_report.json
git diff --check
git status
git log -1
git push origin main
```

The collector and validation APIs also assembled and checked all four campaigns
before writing the aggregate. No fabricated benchmark values, energy measurements,
learned model, reinforcement learning or learned action selection were introduced.

Limitations and metric definitions are documented in
[phase9_dataset_collection.md](phase9_dataset_collection.md). In particular,
normalization-only preprocessing, instrumentation/probe overhead, payload-only
byte counts, differing heads, one network condition and an unexecuted rule baseline
prevent treating this small first dataset as training-qualified evidence.

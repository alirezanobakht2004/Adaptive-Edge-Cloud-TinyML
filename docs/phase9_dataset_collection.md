# Phase9 decision traces and baseline campaigns

This implements the user's M14-M18 **collection and benchmarking** scope.
The external `Instruction/PROJECT_ARCHITECTURE.md` remains the architectural
reference. Its canonical Phase9 learned-policy work is deferred by the current
request. No policy is trained and no adaptive selection is added here.

## Executed actions

| Action | Device execution | Server execution |
|---|---|---|
| 0 ALL_LOCAL | Existing B3 prefix and five-pass stochastic edge head | None for the selected action |
| 1 SPLIT1 | Existing float32 10 -> 64 prefix | Existing Split1 tail: 64 -> 48 -> 32 -> 5 |
| 2 SPLIT2 | Existing float32 10 -> 48 prefix | Existing Split2 tail: 48 -> 32 -> 64 -> 32 -> 5 |
| 3 ALL_CLOUD | Normalize ten features, send all ten | Frozen B1/B2/B3 followed by B4/B5/cloud head: 10 -> 64 -> 48 -> 32 -> 64 -> 32 -> 5 |

ALL_CLOUD is `gesture-full-cloud-v1.0.0`, composed in memory from hash-checked
existing model weights. It accepts ten normalized features, rejects 32-D
embeddings, and executes all neural blocks on the host. Reusing the frozen B4/B5
weights does not make this an alias for a Split3 request. Production
`server/app/inference.py`, MQTT topics and validated inference sources are unchanged.

## Instrumentation and clocks

The isolated `firmware/test/test_phase9_campaign/` image receives raw features-v1
vectors over serial. It reuses the existing normalizer, B3 runtime and stochastic
uncertainty implementation. Split1/2 use their existing model assets and a shared
64 KiB test arena. The test does not change the production inference path.

Before every fixed action it measures B3/MC confidence, predictive entropy in
nats, top-two margin, class, heap, PSRAM, CPU frequency, B3 arena usage, RSSI and a
correlated MQTT probe. The stochastic seed is `42000 + validation window index`.
The local selected action runs B3/MC again with the same seed. ALL_CLOUD still
executes all neural layers remotely for its selected action; the pre-decision
diagnostic state pass is separately measured and included in total cost.

The frozen v1 schema is preserved. `raw_traces.jsonl` retains PSRAM, RSSI, seed,
request ID, probe byte counts and microsecond timestamps not present in that
schema. The v1 records contain uncertainty, device/network state, action,
prediction/confidence, label, timings, payload sizes, versions and artifact hashes.
Correctness is derived from the recorded prediction and validation label.

- `preprocessing_latency_ms`: device normalization only. Feature extraction and
  serial transfer are outside the measured decision interval.
- `total_latency_ms`: normalization, pre-decision inference, probe and selected
  action, measured by device `micros()` differences. Includes cold starts.
- Split prefix timing includes interpreter allocation in this isolated executor.
  It is not a production steady-state latency measurement.
- Selected-action communication bytes count JSON request/response payloads.
  Probe traffic is retained separately, including for ALL_LOCAL. MQTT headers,
  Wi-Fi overhead and retransmissions are not counted.
- MQTT RTT state comes from the probe; selected-action round trip includes host
  processing and device polling. Neither is a pure network RTT measurement.
- Device send/receive timestamps remain boot-relative in raw traces. They are
  not converted into fabricated UTC values. Metadata timestamps are host UTC
  receipt times. Server receive/publish timestamps are host UTC; publish means
  the call to enqueue the QoS-0 response, not broker delivery acknowledgement.
- B3 arena usage is not total reserved test memory. Heap/PSRAM measurements
  describe this instrumentation image, which has an additional test arena.
- Bandwidth estimates, energy and reward remain null/unconfigured. Energy may
  only become an explicitly estimated/simulated proxy; no battery measurement
  or measured-energy claim is made.

## Reproduce a campaign

Use the repository root, its existing Python environment, the connected ESP32
and the existing broker. No dependencies were added. The isolated image uses the
existing Wi-Fi helper and broker address `192.168.137.1:1883`. Credentials remain
in the ignored firmware secrets file. Only one campaign responder/device may use
the isolated `phase9/esp32-policy/request` and `/response` topics at a time.

```powershell
pio test -d firmware -e esp32-s3-n8r2 -f test_phase9_campaign --upload-port COM10 --test-port COM10 -v
python -m tools.policy_dataset.run_campaign --samples 25 --mode local --output <new-local-directory>
python -m tools.policy_dataset.run_campaign --samples 25 --mode split1 --output <new-split1-directory>
python -m tools.policy_dataset.run_campaign --samples 25 --mode split2 --output <new-split2-directory>
python -m tools.policy_dataset.run_campaign --samples 25 --mode cloud --output <new-cloud-directory>
```

Each directory must be new. `--port`, `--broker-host` and `--broker-port` are
optional. The runner uses the existing validation session and interleaves its
five class groups. `--samples` cannot exceed available unique validation windows;
it never silently repeats windows. It does not load held-out test data.

Each run stores `configuration.json`, `policy_training_dataset_v1.jsonl`,
`raw_traces.jsonl` and `server_events.jsonl`. Configuration captures source/model
versions, window indices, seed rule, feature matrix hash, weight hashes, execution
source hashes, locally built firmware hash when available, dependency versions,
Git HEAD/dirty state and start/end timestamps. A locally available binary hash is
provenance, not a device attestation; flash the matching isolated image first.
Partial runs preserve already collected records and an explicit failed manifest.
The first serial startup failure in `campaigns/local/` is retained with zero
samples and is excluded from the assembled dataset.

## Validation and assembly

```powershell
python -m tools.policy_dataset.dataset_report --input <campaign>/policy_training_dataset_v1.jsonl --output <campaign>/dataset_report.json
python -m tools.policy_dataset.collector collect --input <campaign>/policy_training_dataset_v1.jsonl
python -m tools.policy_dataset.collector validate
python -m tools.policy_dataset.dataset_report --input data/policy/policy_training_dataset_v1/policy_training_dataset_v1.jsonl --output data/policy/policy_training_dataset_v1/dataset_report.json
python -m tools.policy_dataset.benchmark_report --input data/policy/policy_training_dataset_v1/policy_training_dataset_v1.jsonl --output data/policy/policy_training_dataset_v1/benchmark_report.json
```

Import each campaign once. The collector rejects repeated run/sample identities.
The assembled named JSONL is a byte-for-byte snapshot of `observations.jsonl`;
refresh that snapshot after intentional imports. These are the same observations,
not two independent datasets. Campaign files retain their original per-run records.

The report checks required fields, fixed actions, versions, duplicate identities,
finite values, explicit energy provenance, and, when alongside configuration,
raw-trace/manifest consistency. Report input hashes normalize line endings to LF
UTF-8 for reproducibility across Git checkouts. Repeated source windows across
different action runs are intentional; future train/validation partitioning must
group by source session/window to avoid leakage.

## First collected dataset and limits

The first dataset contains **100 observations**, 25 per fixed action, using the
same 25 unique `session_02` validation windows (five per class). All executions
succeeded. This is measured feature replay on physical hardware, not a new
live-IMU dataset or a representative accuracy study.

The benchmark JSON computes metrics only from these observations and records
RULE_BASED_ADAPTIVE as missing. No rule-based execution campaign was run. All
energy values are null. Fixed modes ran sequentially in one uncontrolled local
network setup, and host load/cold starts affect their costs. Local and cloud heads
also differ, so comparisons include model differences as well as placement.

This first campaign enables subsequent collection work. Broader sessions,
uncertainty levels, network conditions, repeated/randomized action order,
failure cases, an executed rule baseline and a justified energy proxy are still
needed before qualifying a learned-policy training dataset.

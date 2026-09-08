# Phase10.1 M30: matched-condition measurement campaign

M30 is complete as a first real, configuration-dependent labeling pilot.
The canonical `policy_training_dataset_v2` contains **24 accepted matched
decision samples and 384 candidate executions**. Training remains blocked.
Dataset expansion and final reward calibration are not complete.

## Architecture and scope

This follows the architecture's benchmark-derived action-cost supervision and
M25/M29 formulation. The current explicit policy mapping is fixed:

| Action | Execution |
|---|---|
| 0 ALL_LOCAL | ESP32 B3 prefix and existing stochastic local head |
| 1 SPLIT1 | ESP32 64-dimensional prefix, MQTT, Split1 cloud tail |
| 2 SPLIT2 | ESP32 48-dimensional prefix, MQTT, Split2 cloud tail |
| 3 ALL_CLOUD | ESP32 sends 10 normalized features; server executes full cloud network |

ALL_CLOUD is not Split3. Local and cloud heads retain their existing distinct
architectures. No production runtime, split model, feature extraction,
sampling rate, class definition, dataset-v1 or policy dataset v1 was changed.
No learned model, RL, policy training or deployment was implemented.

## Why an isolated firmware image was necessary

The existing Phase9 test image recomputes uncertainty/device/network state for
each RUN command. Regrouping those traces would not satisfy matched-state
collection. The new `firmware/test/test_phase10_matched_campaign/test_main.cpp`
is derived from that isolated harness, reuses existing inference functions and
model assets, and adds a single-snapshot MATCH command. Production sources are
unchanged. The board is left running this isolated experiment image.

For each existing labeled validation feature window:

1. Normalize features with the existing features-v1 preprocessor.
2. Seed the existing stochastic head with `42000 + window_index`; execute the
   pre-decision inference once and measure uncertainty and state inference time.
3. Perform a real MQTT probe and capture heap, PSRAM, CPU, RSSI and connection.
4. Freeze that physical snapshot on the ESP32. All candidate records refer to
   this same snapshot; the host does not overwrite per-action states to make
   mismatched observations appear equal.
5. Execute four repeats, rotating the four-action order. Every action appears
   once in every execution position per window. The starting rotation also
   varies across windows. Local action uncertainty uses the same fixed seed.
6. Before each action, measure a fresh guard probe, heap/PSRAM, CPU, RSSI,
   connection and snapshot age. Retain these separately from the frozen state.
7. Retain raw device traces and correlate successful remote responses with
   actual server events, predictions, confidence, model version, latency and
   byte counts. Ground truth comes from the original validation window.

This is **one identical decision-state snapshot with bounded physical drift**.
Actions cannot execute simultaneously on one ESP32, and the network and heap
cannot be held mathematically identical. Guards make that limitation observable.
Cyclic order balances position effects; it does not eliminate all carryover,
cache warm-up, stochastic network effects or instrumentation overhead.

## Prespecified connected-lab condition

The condition file is
`tools/policy_dataset/conditions/connected_lab_v1.json`.
The host/ESP32 used the existing broker on port 1883, ESP32 COM10 and the
existing Wi-Fi/hotspot setup. No traffic shaping or disconnected campaign was
introduced. Configuration is a declared condition and acceptance envelope,
not an assertion that the collector controls every external process.

| Guard | Configured acceptance limit | Largest accepted observation |
|---|---:|---:|
| Snapshot/guard MQTT RTT | 50 ms | 27.653 ms |
| Absolute guard versus snapshot RTT difference | 20 ms | 17.251 ms |
| Heap difference | 4,096 bytes | 420 bytes |
| Free PSRAM difference | 4,096 bytes | 0 bytes |
| RSSI difference | 8 dB | 3 dB |
| Snapshot age before action | 10,000 ms | 1,615.192 ms |

CPU frequency and connected status must agree. Invalid/missing guard values,
failed probes, incomplete remote evidence or failed execution reject a window
from this connected-success pilot. Failures and rejected traces remain evidence,
not zero-filled candidate measurements. Future failure-condition campaigns need
an explicit appropriate condition and complete failure-cost accounting.

## Reward and label methodology

This pilot uses explicit **provisional** research parameters, not final weights:

```text
R(a) = 1.0 * correct(a)
       - 0.1 * total_latency_ms(a) / 100
       - 0.1 * selected_payload_bytes(a) / 1024
       - 0.1 * estimated_proxy(a)

estimated_proxy(a) = (preprocessing_ms + state_inference_ms
                     + selected_edge_compute_ms(a)) / 10
                     + selected_payload_bytes(a) / 1024
```

Selected edge compute is the local action duration for ALL_LOCAL, prefix
duration for SPLIT1/2, and zero for full-cloud model compute on the edge.
Shared state inference/preprocessing are included for every candidate.
The proxy is dimensionless and estimated only: **no measured energy, joules or
battery-life claim**. It excludes radio waiting, cloud energy and probe traffic.
It is not a power model; its payload term overlaps the communication penalty.
Its coefficients and version (`m30-compute-payload-proxy-v1`) are retained so
the assumption can be changed and sensitivity inspected without retraining.

Latency is shared snapshot preparation/state/probe overhead plus the actual
selected action duration, measured on the ESP32. It excludes between-action
waiting, serial output and experimental guard probes. Thus each candidate is
charged the same decision overhead, not cumulative time spent testing earlier
actions. Communication is selected request plus response JSON payload bytes,
excluding probe traffic and MQTT/TCP/Wi-Fi headers. ALL_LOCAL has zero selected
payload bytes although this instrumented campaign performs common probes.

Each repeat is validated and scored by the unchanged M29 `evaluate_group`.
The action with maximum **mean observed reward across repeats** receives the
label. All actions within absolute tolerance `1e-6` are retained; a nonunique
winner produces null `optimal_action`. Repeat-level winners are also retained.
These are observed empirical labels, not proof of the expected optimum under
all future conditions or statistically estimated confidence intervals.

## Dataset and evidence

Canonical directory: `data/policy/policy_training_dataset_v2/`.

- `policy_training_dataset_v2.jsonl`: 24 labeled samples, with frozen state,
  four action summaries, repeated candidate measurements, source records,
  raw traces/guards, calibration, hashes and reproducible labels.
- `metadata.json`, `schema_description.json`, `README.md`: source versions,
  schema, action mapping, qualification and canonical/excluded campaign mapping.
- `dataset_report.json`: full raw/source/server validation and measured summary.
- `reward_sensitivity_report.json`: offline reweighting of retained measurements.
- `campaigns/connected_lab_run02/`: canonical source, configuration, raw traces,
  rejected entries, server events and identical campaign JSONL/report copies.
- `campaigns/connected_lab_run01/`: excluded diagnostic run. It yielded 25
  candidate sets but overlapped host pytest, so it is not canonical evidence.

The final campaign requested 25 windows with four repeats of all four actions
(400 executions). Window 124 exceeded RTT drift acceptance: **20.258 ms > 20 ms**.
It was rejected without relaxing the threshold. The accepted set therefore
contains 24 distinct windows, class counts 5/4/5/5/5, from `session_02`.
All 384 accepted executions succeeded, including 288 matched MQTT responses.
There are 96 accepted executions per action. Do not double-count the canonical
file and its campaign copy, or merge the diagnostic run.

| Action | Correct / evaluated | Mean accounted latency | Mean selected payload bytes | Mean estimated proxy |
|---|---:|---:|---:|---:|
| ALL_LOCAL | 96 / 96 | 16.0634 ms | 0 | 0.3411 |
| SPLIT1 | 96 / 96 | 36.3201 ms | 765.8646 | 1.0127 |
| SPLIT2 | 96 / 96 | 38.6415 ms | 663.7500 | 0.9894 |
| ALL_CLOUD | 96 / 96 | 38.7451 ms | 391.9688 | 0.5440 |

These are measurements on the accepted replay subset, not generalization
accuracy, deployment latency or evidence that a learned policy improves results.
The labels are **24 ALL_LOCAL, zero SPLIT1/SPLIT2/ALL_CLOUD**, with agreement
across all four repeats in every accepted sample. Halving/doubling each cost
weight independently leaves these labels unchanged. Accuracy-only weighting
makes all four actions tie for all 24 samples. No weights were fitted to create
action diversity, and these sensitivity checks do not alter canonical labels.

The root dataset version is v2. Embedded source metadata/records and the M29
input contract intentionally retain v1 identifiers; existing v1 placeholder
rewards are never overwritten. A separate top-level label/candidate structure
holds the accuracy-inclusive objective.

## Tools, commands and validation

The repository `.venv` and existing PlatformIO installation were used.

```powershell
# Regression before using the isolated image:
.venv\Scripts\python.exe -u -m tests.run_phase7_hardware --port COM10 --filter 'test_phase7_*'

# Isolated firmware parity and upload:
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" test -d firmware -e esp32-s3-n8r2 -f test_phase10_matched_campaign --upload-port COM10 --test-port COM10 -v

# Final real campaign (output must not already exist):
.venv\Scripts\python.exe -u -m tools.policy_dataset.run_matched_campaign --samples 25 --repeats 4 --condition tools/policy_dataset/conditions/connected_lab_v1.json --output data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02 --port COM10

# Validate source labels, raw traces, server evidence and deterministic M29 rewards:
.venv\Scripts\python.exe -m tools.policy_dataset.validate_matched_dataset --directory data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02 --output data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02/dataset_report.json

.venv\Scripts\python.exe -m py_compile tools/policy_dataset/matched_dataset.py tools/policy_dataset/run_matched_campaign.py tools/policy_dataset/validate_matched_dataset.py tests/test_matched_campaign.py
.venv\Scripts\python.exe -m pytest -q
git diff --check
```

`--samples` counts requested distinct validation windows; no silent input
duplication or retries replace rejected windows. `--repeats` supports 4, 8, 12
or 16 for balanced positions. `--condition` is a versioned JSON path and is
validated before connecting to hardware. Port/broker arguments are available.
Raw evidence is flushed incrementally; failed runs retain manifests and traces.
Copying the accepted campaign JSONL/report to the root creates the canonical
snapshot; tests check exact equality to the metadata-declared campaign.

Validation results:

- **220 Python tests passed**, four existing TensorFlow Lite deprecation warnings.
- Compilation and whitespace checks passed.
- All **six Phase7 hardware suites passed**, covering Split1/2/3 parity and
  MQTT E2E, five vectors per suite.
- The isolated matched-campaign prefix parity test passed on ESP32-S3.
- Dataset source labels, 384 accepted raw observations, remote server evidence,
  action coverage, schema/type checks and deterministic reward/label rebuilding passed.
- `git diff` confirms v1, production firmware sources/includes, server runtime
  and existing policy formulation remain unchanged.

Hardware logs: `docs/evidence/phase10_1_m30/phase7_regression.log` and
`matched_hardware.log`. The regression log contains PowerShell stderr wrapping
around TensorFlow startup diagnostics; all six hardware suite results are PASS.

## Readiness decision

**Policy training is not allowed yet.** M30 supplies real candidate measurements
and labels, but this pilot has one optimal-action class, one acquisition session,
one device and one connected condition. Four repeated executions are correlated
measurements of a window, not four independent training samples. The original
gesture validation windows must remain grouped when planning later policy
partitions; fresh independent sessions and conditions are needed for holdouts.

Next collect uncertainty/error, network and resource conditions where action
tradeoffs can genuinely differ; specify failure-cost campaigns; calibrate reward
and energy assumptions; analyze repeated-measurement variability; then establish
independent policy train/validation/test partitions. Do not manufacture nonlocal
labels by changing weights solely to balance classes. Reassess readiness after
that evidence exists. No training was started.

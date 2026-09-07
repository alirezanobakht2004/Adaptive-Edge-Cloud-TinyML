# Phase8 policy dataset and benchmark infrastructure

## Scope and motivation

This work prepares decision records, opt-in measurement adapters, collection
tools and baseline interfaces for future learned offloading. It does not train
an offloading model or change production inference. The task's M9–M13 labels
describe this preparation work; they do not renumber the canonical architecture's
later learned-policy, failover, dashboard and continual-learning milestones.

The architecture's eventual Phase8 gate requires a dataset from measured costs.
The infrastructure here is ready to collect that campaign; an empty skeleton
and host smoke observations do not satisfy policy-training data requirements.

## Version and fixed actions

The executable contract is `tools/policy_dataset/schema.py`. Its human/machine
readable description is saved in `data/policy/policy_training_dataset_v1/schema.json`.
This is a custom schema description checked by the bundled validator, not a
claim of compatibility with a third-party JSON Schema engine.

| Action | Name | Meaning |
|---:|---|---|
| 0 | ALL_LOCAL | Classification remains local |
| 1 | SPLIT1 | B1 embedding offloaded through the validated Split1 path |
| 2 | SPLIT2 | B1+B2 embedding offloaded through the validated Split2 path |
| 3 | ALL_CLOUD | Entire model computation performed in the cloud; requires an explicit future executor |

This explicit current request supersedes the earlier proposed Phase8 plan's
action-3 meaning. **Action 3 is not MQTT `split=3`.** The existing validated
Split3 runtime/protocol remains unchanged. Its results cannot be imported as
ALL_CLOUD observations. No ALL_CLOUD wire protocol or executor is invented here;
the baseline selects action 3 and the benchmark runner rejects execution unless
the caller supplies its genuine executor.

## Record structure

One `PolicySample` represents one decision opportunity with a frozen state
snapshot, selected action, post-action observations and a separate reward.
Required keys are present even when their value is null. Unknown values are not
silently converted to zero. Extra structured fields are rejected so semantics
cannot drift unnoticed.

| Section | Required content |
|---|---|
| `metadata` | dataset_version, sample_id, timestamp, device_id, model_version, feature_version, source_dataset_version, record_kind, measurement_scope, run_id, session_id |
| `state.uncertainty` | confidence, entropy, margin |
| `state.device` | free_heap, tensor_arena_usage, cpu_frequency, inference_latency_estimate |
| `state.network` | mqtt_rtt_ms, packet_size_estimate, connection_state, estimated_bandwidth |
| `state.application` | predicted_class, window_id |
| `action` | Integer 0, 1, 2 or 3; booleans are invalid |
| `reward` | latency_component, communication_cost_component, energy_component, energy_kind, energy_proxy_version, final_reward, calculation |
| `measurements` | Stage durations, client/server timestamps, bytes, local outcome uncertainty |
| `outcome` | status, predicted_class, confidence, true_class, error |
| `provenance` | state_source, artifact_hashes, clock_domains, notes |

Versions are fixed to `policy_training_dataset_v1`, source `dataset-v1`,
`features-v1`, and source model `gesture-model-v1.1.0`. Artifact hashes can record
the selected tail's identity separately. Class IDs remain the existing 0–4 mapping.
`timestamp` is timezone-aware ISO-8601. `record_kind` is `example` or `observation`;
`measurement_scope` is `unmeasured`, `host`, `device` or `replay`.

Entropy is `-sum(p * ln(p))` in nats, bounded by ln(5). Margin is the largest
minus second-largest class probability. Confidence and margin are in [0,1].
Heap/arena and packet estimates use bytes, CPU frequency uses MHz, bandwidth
uses bytes/second, and duration fields use milliseconds. A later policy must
use only state available before action selection. It must not use outcome
confidence, true labels or measured post-action latency as decision inputs.

## Opt-in instrumentation

`tools/policy_dataset/instrumentation.py` provides:

- `PolicySample.snapshot_state`: copy a caller-supplied pre-decision state.
- `PolicySample.measure`: time an explicitly named stage and preserve failures.
- `PolicySample.local`: measure caller-supplied preprocessing and local inference;
  record output confidence, entropy and margin separately from decision state.
- `PolicySample.prefix`: measure Split1/2 prefix calls and embedding sizes.
- `RoundTrip`: record exact request/response bytes, match request IDs, validate
  split/model/class, measure elapsed time and preserve timeouts.
- `PolicySample.cloud`: observe request-receive, tail-call and publish boundaries
  around caller-supplied functions without changing inference or MQTT payloads.

Round-trip duration uses the client's monotonic clock. `mqtt_send_timestamp`
is client UTC before publish; `server_response_timestamp` is client UTC when the
server response is received. Server `request_receive_time` and
`response_publish_time` use server UTC; the latter marks return from publish,
not broker ACK or delivery. Server and client wall clocks must not be subtracted
to calculate network latency. Recorded server inference duration comes from
its own timer/response field. Provenance identifies clock domains.

No firmware change was needed. The existing production serial `W=...` line
already provides feature, normalization, prefix and MC timings, confidence and
normalized entropy. `production_serial_record` imports a captured line with a
caller-supplied receipt timestamp and run/session IDs. It converts microseconds
to milliseconds and normalized entropy to nats. It leaves margin, heap, arena,
CPU frequency, bandwidth, RTT and pre-decision state null when absent. It does
not invent a historical timestamp or reinterpret post-inference output as state.

Full device-state collection and a controlled network campaign still require
explicit adapters or a separately validated isolated firmware experiment. Do
not flash production firmware merely to run the host smoke tests.

## Collector, serializer and validator

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.collector validate
.venv\Scripts\python.exe -m tools.policy_dataset.collector init --directory path/to/new/dataset
.venv\Scripts\python.exe -m tools.policy_dataset.collector collect --input records.jsonl --directory path/to/dataset
```

The repository skeleton contains:

```text
data/policy/policy_training_dataset_v1/
    metadata.json
    schema.json
    sample_000001.json     # explicitly unmeasured example
    observations.jsonl    # empty until a real collection campaign
    README.md
```

Serialization rejects nonfinite JSON and duplicate object keys. Validation checks
required structure, action/type/range constraints, versions, timestamps, energy
labels and manifest/schema consistency. Collection rejects examples and duplicate
`(run_id, sample_id)` identities. All records in an import batch are checked
before appending. This MVP supports one writer; it does not claim transactional
recovery from disk failure or concurrent-writer safety. Preserve session IDs to
avoid train/test leakage across repeated windows and action comparisons.

## Baselines and reward placeholder

`ml/policy/benchmark.py` defines `PolicyStrategy`, immutable `Selection`, fixed
ALL_LOCAL, ALL_CLOUD, FIXED_SPLIT1 and FIXED_SPLIT2 strategies, and
RULE_BASED_ADAPTIVE. Strategies only return `selected_action`; explicit executors
perform measurements. Missing executors fail clearly. Executors may not change
the saved decision state or selected action.

The rule-based comparator uses configurable confidence and RTT thresholds. Its
default confidence threshold is 0.8, fast RTT is 50 ms and maximum RTT is 200 ms.
These are uncalibrated benchmark assumptions, not measured optimal thresholds.
Disconnected, unknown or high-RTT state falls back to local. Otherwise uncertain
samples use Split1 on a fast link or Split2 on a slower acceptable link. It is
not a learned model and is not installed into production firmware.

The optional reward function requires supplied scales and an explicitly versioned
estimated energy score:

```text
latency_component = total_latency_ms / configured_latency_scale_ms
communication_cost_component = (request_bytes + response_bytes) / configured_byte_scale
energy_component = caller-supplied estimated proxy score
final_reward = -(latency_component + communication_cost_component + energy_component)
calculation = placeholder-negative-sum-v1
```

Without measurements and configured assumptions, components/final reward remain
null and `calculation` is `placeholder-unconfigured`. This is a dimensionless
placeholder, not a validated physical reward model. `energy_kind` can only be
`estimated` or `simulated`; measured-energy labels are rejected. **No battery
energy, current, joules or energy savings are measured or claimed.**

Summaries separate host/device/replay scopes and actions, exclude examples and
unexecuted decisions, retain failed attempts, and report coverage counts.
Accuracy is null without labels; failed labeled attempts count as incorrect.
Missing durations/bytes/proxy scores produce null metrics, not fabricated zeros.
Incompatible energy-proxy assumptions cannot be averaged together.

## Reproducible smoke checks and limitations

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.host_smoke --output path/to/new/host-smoke
.venv\Scripts\python.exe -m tools.policy_dataset.mqtt_smoke --output path/to/new/mqtt-smoke
```

The first command records five real host Keras calls on existing parity inputs.
Its preprocessing means ndarray preparation; model execution includes frozen
normalization. This deterministic smoke is not the production five-pass local
algorithm. The second records ten real host-prefix/broker/cloud round trips on
dedicated smoke topics while reusing the frozen server models and response
builder. These topics are test-only; production topics/protocol are unchanged.
The existing broker must be reachable (default localhost:1883). Both commands
refuse to overwrite output directories and store actual observations separately
from the repository's empty policy dataset skeleton.

Saved evidence is under `docs/evidence/phase8/host_smoke/` and `mqtt_smoke/`.
These records exercise the collection path, carry source/tail hashes and have
no ground-truth labels or energy values. They are not a statistically designed
benchmark, do not establish comparative accuracy/latency, and are not sufficient
to train a policy. Future collection must freeze artifact/runtime versions,
capture complete decision state, compare matched inputs/actions, preserve
session partitions, and validate coverage before any learned-policy training.

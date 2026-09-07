# Phase 8 — benchmark the action space

Prepared 2026-09-08 after the Phase7 float32 split-path gate closed. This is a
plan and future data contract, not an implemented collector, benchmark result,
Meta Learner or Split Controller. See `phase7_m8_completion.md` for validation
evidence and the two outstanding legacy INT8 diagnostic failures.

## Objective and frozen actions

Measure the quality and cost of each available action for the same input under
recorded network conditions. Keep dataset-v1, features-v1, the five gesture
classes, 100 Hz sampling and existing preprocessing fixed.

| Action ID | Name | Execution | Offloaded embedding |
|---:|---|---|---:|
| 0 | All Local | Existing B1+B2+B3 and local uncertainty/head runtime | None |
| 1 | Split1 | ESP32 B1, then `gesture-cloud-tail-split1-v1.0.0` | 64 float32 values |
| 2 | Split2 | ESP32 B1+B2, then `gesture-cloud-tail-split2-v1.0.0` | 48 float32 values |
| 3 | Split3 | Existing ESP32 B1+B2+B3, then `gesture-cloud-tail-v1.0.0` | 32 float32 values |

Record exact model hashes, firmware Git commit, server Git commit, dependency
versions and normalization hash in a run manifest. Split1's shorter tail and
the independently trained tails may produce different predictions; compare
each action's observed result rather than assuming cross-split equivalence.

Action 0 should initially use the current production five-pass uncertainty
behavior; record pass count and PRNG seed. Do not silently substitute a cheaper
deterministic head. Any additional local baseline must be separately identified
and must not reuse action 0's meaning within this dataset version.

## Measurement procedure

1. Validate the artifacts and run the six Phase7 hardware suites before a campaign.
2. Freeze one input window and its feature vector. Record sample/session identity
   and an input hash. Replay that same input for all four actions.
3. Use a seeded, balanced action order within each sample/condition/repetition
   group. Record the order to expose thermal, cache, and network-order effects.
4. Run explicit warm-up iterations and mark them; report cold-start observations
   separately. Specify repetition counts in the manifest before collection.
5. Test baseline connectivity, deliberately degraded connectivity and unavailable
   connectivity. Record the imposed condition separately from actual RSSI/RTT.
   Do not infer a controlled link rate from RSSI alone.
6. Execute one action at a time on the ESP32. Correlate offload results using the
   existing MQTT request ID and response envelope. Record timeouts and rejected
   requests as outcomes, including elapsed time and attempted bytes.
7. Persist raw measurements and a run manifest before deriving aggregate metrics.
   Report per-action distributions and sample counts, not only averages.

Existing train/validation/test session boundaries remain unchanged. Initial
instrumentation checks can replay the five Phase7 validation vectors, but they
cannot establish general recognition accuracy. Define the evaluation protocol
before opening held-out TEST data; never fit preprocessing, cost scaling, or
future policies on TEST. Keep all four actions and repeated measurements from
the same source session together in any future policy split.

## Timing, communication and state definitions

Use the ESP32 monotonic timer for both ends of a device duration. Server latency
uses the server's own timer. Do not subtract a server timestamp from an ESP32
timestamp without clock synchronization.

| Field | Measurement boundary or meaning |
|---|---|
| `feature_latency_ms` | Frozen raw window → extracted features; record normalization separately |
| `normalization_latency_ms` | Frozen features → normalized features |
| `prefix_latency_ms` | Normalized input → selected prefix output |
| `local_head_latency_ms` | Action 0's head/uncertainty computation; includes its recorded pass count |
| `offload_roundtrip_ms` | Before MQTT request serialization/publish → validated matching result on ESP32 |
| `server_latency_ms` | Existing response field; server model execution only |
| `action_latency_ms` | Normalized input ready → validated classification or terminal failure; includes prefix compute |
| `pipeline_latency_ms` | Frozen raw window ready → classification or terminal failure; excludes acquisition of the window |
| `rtt_probe_ms` | Separately measured pre-action network probe; not renamed inference round-trip latency |
| `request_payload_bytes` | UTF-8 bytes actually serialized for the inference request |
| `response_payload_bytes` | UTF-8 bytes actually received for the matching inference response |
| `mqtt_packet_bytes_estimate` | Explicitly calculated MQTT framing estimate, with method in manifest |
| `probe_bytes` | Separate traffic generated to measure network state |
| `wire_bytes_measured` | Optional captured TCP/IP/link bytes; null when not measured |
| `rssi_dbm` | ESP32 Wi-Fi reading at the recorded pre-action state snapshot |
| `heap_before_bytes`, `heap_after_bytes` | Free heap around the action; same memory API for every action |
| `minimum_free_heap_bytes` | Lifetime minimum if that is what the API exposes; not claimed as a per-action minimum |

Network state includes connected/disconnected, broker connectivity, network
condition ID and probe timestamp/age. For action 0, inference request/response
bytes are zero and offload/server durations are null. Probe traffic remains
separate if probing occurs. Missing responses have null class/confidence; the
elapsed timeout duration remains recorded. Do not turn missing measurements
into zero-cost successful samples.

Raw float payload sizes are 256/192/128 bytes for splits 1/2/3. These are not JSON
or wire-byte measurements. Count the real serialized payloads and validate the
existing MQTT buffer before running a campaign; do not redesign the protocol.

If later policy experiments compute B3 and uncertainty before deciding to use
Split1, that earlier computation is not free. Record decision/preparation cost
separately and include it in the later policy's total latency and energy proxy.
This phase benchmarks fixed actions, without implementing that policy.

## Energy proxy — estimated, not battery measured

No battery-measurement hardware is assumed. Use a clearly named, versioned proxy:

```text
estimated_energy_proxy =
    compute_weight * device_compute_ms
  + network_wait_weight * device_network_wait_ms
  + transmit_weight * request_payload_bytes
  + receive_weight * response_payload_bytes
```

The result is a dimensionless score unless a separately documented physical
calibration supplies units. Coefficients are assumptions stored in the manifest;
do not invent measured current, joules, battery life or energy savings. If a
future simulation produces joules, use a separate `simulated_energy_j` field
with model provenance and continue labeling it simulated. Retain the component
measurements so alternative proxy coefficients can be evaluated later.

## Future dataset: `policy_training_dataset_v1`

One row represents one attempted action for one input, network condition and
repetition. The four rows sharing `comparison_group_id` represent a matched
action comparison. Failed and timed-out attempts remain present.

Future directory structure, not created by this planning step:

```text
experiments/action_space/policy_training_dataset_v1/
    schema.json
    runs/<run_id>/manifest.json
    runs/<run_id>/measurements.csv
    runs/<run_id>/events.jsonl
    derived/action_summary.csv
```

Required future row schema:

| Columns | Type and constraints |
|---|---|
| `schema_version` | String, exactly `policy_training_dataset_v1` |
| `run_id`, `comparison_group_id`, `sample_id`, `session_id` | Nonempty stable IDs |
| `dataset_version`, `feature_version` | Exactly `dataset-v1`, `features-v1` |
| `input_sha256`, `artifact_manifest_sha256` | Hashes linking input and immutable run configuration |
| `action_id` | Integer 0–3 |
| `repetition`, `action_order_index` | Nonnegative integers |
| `warmup` | Boolean |
| `device_id`, `network_condition_id` | Nonempty strings |
| `request_id` | Unique offload request ID; null for action 0 |
| `true_class_id` | Existing class ID 0–4 when available; otherwise null |
| `predicted_class_id` | Existing class ID 0–4 on successful classification; otherwise null |
| `confidence` | Finite float in [0,1] on success; otherwise null |
| `outcome` | `success`, `timeout`, `publish_failed`, `invalid_response`, `inference_failed`, or `network_unavailable` |
| `error_detail` | Optional diagnostic string |
| `edge_model_version`, `cloud_model_version` | Actual deployed versions; cloud null for action 0 |
| `mc_passes`, `prng_seed` | Actual local uncertainty settings, nullable when unused |
| `feature_latency_ms`, `normalization_latency_ms`, `prefix_latency_ms` | Finite nonnegative durations when completed; otherwise null |
| `local_head_latency_ms`, `offload_roundtrip_ms`, `server_latency_ms` | Applicable finite nonnegative durations, otherwise null |
| `action_latency_ms`, `pipeline_latency_ms` | Finite nonnegative elapsed duration to outcome |
| `request_payload_bytes`, `response_payload_bytes`, `probe_bytes` | Nonnegative integer observed counts |
| `mqtt_packet_bytes_estimate`, `wire_bytes_measured` | Nonnegative integer or null; never conflate estimate with measurement |
| `wifi_connected`, `mqtt_connected` | Pre-action booleans |
| `rssi_dbm`, `rtt_probe_ms`, `state_age_ms` | Finite state observations, nullable if unavailable |
| `heap_before_bytes`, `heap_after_bytes`, `minimum_free_heap_bytes` | Nonnegative integer bytes or null |
| `device_compute_ms`, `device_network_wait_ms` | Nonnegative durations used by the proxy |
| `estimated_energy_proxy`, `energy_proxy_version` | Finite nonnegative score and assumption version; null if unavailable |

Example CSV header only; no synthetic measurement rows are supplied:

```csv
schema_version,run_id,comparison_group_id,sample_id,session_id,dataset_version,feature_version,input_sha256,artifact_manifest_sha256,action_id,repetition,action_order_index,warmup,device_id,network_condition_id,request_id,true_class_id,predicted_class_id,confidence,outcome,error_detail,edge_model_version,cloud_model_version,mc_passes,prng_seed,feature_latency_ms,normalization_latency_ms,prefix_latency_ms,local_head_latency_ms,offload_roundtrip_ms,server_latency_ms,action_latency_ms,pipeline_latency_ms,request_payload_bytes,response_payload_bytes,probe_bytes,mqtt_packet_bytes_estimate,wire_bytes_measured,wifi_connected,mqtt_connected,rssi_dbm,rtt_probe_ms,state_age_ms,heap_before_bytes,heap_after_bytes,minimum_free_heap_bytes,device_compute_ms,device_network_wait_ms,estimated_energy_proxy,energy_proxy_version
```

The manifest must define units, null encoding, timers, energy assumptions,
software/artifact hashes, action order seed, repetition/warm-up counts, network
impairment settings, QoS, timeout settings and sample/session partition rules.
Keep post-action observations and true labels separate from any future
pre-decision policy inputs to prevent label leakage.

## Future derivation and remaining work

Once real measurements exist, derive per-action success rate, classification
metrics where ground truth is available, latency distributions, payload counts
and proxy-score distributions. Use the same matched groups for comparisons.
Do not discard unavailable-network failures to make offloading look cheaper.

The existing `config/experiment.yaml` cost weights are error 0.50, latency 0.25,
communication 0.15 and energy 0.10. A future derived policy dataset must record
the scaling method, missing-data/failure penalties and weight version. Fit any
scaling only on training sessions. An `oracle_action_id` may later be derived
from complete matched groups; it is an offline target, not an implemented policy
and not an input available at decision time.

Remaining Phase8 implementation:

- Add reversible fixed-action benchmark instrumentation and a host collector.
- Validate timer boundaries, byte accounting, heap/state snapshots and timeout records.
- Freeze the campaign manifest and energy-proxy assumptions before collection.
- Collect the matched action measurements without changing dataset/feature versions.
- Validate the dataset schema and session grouping; produce measured summaries.

Meta Learner and Split Controller training remain a later phase. OTA, continual
learning and dashboard work are outside this plan.

# Canonical Phase 9 / M9 - learned binary adaptive policy

Status: **Phase 9 / M9 - CLOSED** under the explicitly controlled undergraduate MVP
qualification. Both learned actions pass the shared production decision path; the
normal production image also passes a live sensor run.
Phase 10 / M10 Failover is not started.

## Original blocker and correction

The historical M30 LOCAL action reran the network after state/UQ computation.
R1 instead reuses that completed prediction. The old binary comparison and its
blocked qualification remain immutable historical evidence. New accounting is
`binary-reward-experiment-v2-r1-reuse`: compare incremental post-decision costs.
Shared preprocessing, local MC/UQ, state acquisition and policy work are excluded
identically. LOCAL performs zero additional inference and sends zero bytes.
Its zero incremental cost is an accounting convention, not a physical timing claim.
CLOUD uses action time plus application payload cost. Energy is a dimensionless
estimated/simulated endpoint-payload proxy; no measured energy claim.

M30 raw evidence retained the cached prediction/UQ and separate cloud action
measurements. All 24 accepted windows were salvageable and choose LOCAL after
correction. Their old records, labels and split artifacts were not edited.

## Oracle and bounded cloud experiment

`docs/phase9_local_cloud_oracle_audit.md` and its JSON record every development
prediction. No final TEST data was used. TRAIN has A/B/C/D = 592/0/1/7;
VALIDATION has 197/0/0/3, where C means LOCAL wrong / CLOUD correct. Both existing
paths score 98.5% on validation. These are development results, not final test
accuracy. Server resources alone establish no accuracy advantage.

One private end-to-end fine-tune of the unchanged full-cloud architecture was
performed using TRAIN only and validation-loss early stopping (seed 42, Adam
1e-4, batch 32, maximum 200 epochs, patience 20). The initial frozen-variable
clone was an invalid no-training diagnostic, retained separately. The valid run
completed 26 epochs and retained 98.5% validation accuracy with zero C cases.
Candidate gesture-full-cloud-v1.1.0 was rejected; the selected server model
remains gesture-full-cloud-v1.0.0. No further search or gesture collection occurred.

## Qualified dataset and limitations

`policy_training_dataset_binary_r1_v2`: 2424 records, 800 underlying windows,
sessions 01 and 02. Measured: 24 LOCAL / 0 CLOUD. Simulated: 2398 LOCAL / 2 CLOUD.
Both CLOUD labels arise from one actual model-disagreement window in TRAIN under
baseline/+150ms controlled networking. A 1500ms stress RTT instead favors LOCAL.
Only RTT varies as a controlled label-generating parameter, and it is an ESP32
observable. Synthetic hidden pressure/queue/cloud multipliers were excluded.

The singleton positive source group is reserved for fitting. Group-stratified
holdout uses seed 20260908: 600 training groups / 1816 records, 200 held-out groups
/ 608 records, zero overlap. Holdout contains zero CLOUD-optimal examples. It can
measure false cloud selections and achieved reward, but cannot establish CLOUD
recall or generalization to new cloud-benefit states. Replays are not independent
hardware evidence. These limitations do not fail the user-authorized MVP gate.

## Frozen policy and reward contracts

`policy-config-v3`, binary-policy-contract-v2-r1-reuse, keeps exact input order:
confidence, entropy (nats), margin, free_heap (bytes), inference_latency_estimate
(ms), mqtt_rtt_ms (ms). Each has an existing pre-decision ESP32 source. Population
mean/std normalization is fitted on training groups only and frozen in the config.
No synthetic-only state is used. Weights remain accuracy 1.0 and latency,
communication, proxy energy 0.1 each; scales remain 100ms, 1024 bytes, proxy 1.

## Learned policy and comparison

Exactly one Input(6) -> Dense(8, ReLU) -> Dense(4, ReLU) -> Dense(2, softmax)
model was fitted: `meta-policy-v1.0.0`, seed 42, Adam 0.001, 200 fixed epochs,
batch 32. Inverse-class-frequency weights were computed only on training rows.
There was no architecture search, holdout tuning, forced label balancing or RL.
Artifacts, environment, history, dataset/config references and hashes are under
`data/policy/models/meta-policy-v1.0.0/`. Training batch history is not holdout accuracy.

On the 608 group-held-out rows the learned policy scores 606/608 (99.6711%) against
derived action labels, confusion matrix [[606,2],[0,0]]. LOCAL precision is 1,
recall 0.9967105; CLOUD precision is 0 and recall is undefined (zero support).
Macro F1 is 0.4991763 using the report's explicit zero-division convention.
LOCAL/CLOUD selection rates are 99.6711%/0.3289%. Mean reward is 0.994103508,
oracle reward 0.995065789, mean regret 0.000962281. These are policy-label metrics,
not gesture classification accuracy and not a final real-world test.

| Same holdout baseline | Policy-label accuracy | Mean reward | Mean regret |
|---|---:|---:|---:|
| All LOCAL | 1.000000 | 0.995065789 | 0 |
| All CLOUD | 0.000000 | 0.240213440 | 0.754852350 |
| Rule-based LOCAL/CLOUD | 1.000000 | 0.995065789 | 0 |
| Learned LOCAL/CLOUD | 0.996711 | 0.994103508 | 0.000962281 |

The learned policy does not beat All LOCAL or the rule baseline on this holdout.
Rule thresholds were calibrated on 1816 training rows only: entropy > 0.988824613
nats and MQTT RTT <= 10.7475ms, with connected/available cloud; otherwise LOCAL.
`rule_based_config.json` records the versioned search and tie rule. The eight
measured holdout rows all select LOCAL correctly; the remaining 600 are explicitly
simulated. Profile/provenance metrics are in `docs/evidence/phase9_policy_comparison.json`
and `phase9_learned_policy_evaluation.json`. Neither policy's CLOUD recall is
established by this holdout. The only underlying CLOUD-optimal development window
is session_01/481 (SWIPE_RIGHT), not independent evidence of broad cloud superiority.

## Export and isolated hardware parity

Float32 TFLite: 2660 bytes, SHA-256
`2e18cb33f3ac9fa9867675be141d811f5126ad7d6d31aa67f65e3a85e29e23c9`.
Desktop framework/TFLite maximum absolute difference is 1.49011612e-8, action parity
10/10, tolerance 1e-5 inherited from Phase 7 float32 parity. Fixed vectors were
chosen by learned output to exercise five LOCAL and five CLOUD actions, not to
estimate accuracy. No gesture INT8 work was reopened.

Isolated ESP32-S3 test on COM10 passed all ten vectors: maximum difference
2.98023224e-8, action parity 10/10. The one diagnostic run measured 372us summed
policy invoke time, 960 bytes of used tensor arena and 359716 bytes free heap.
These are one-run diagnostic observations, not general runtime performance.
`docs/evidence/phase9_esp32_policy_parity.json` links raw serial evidence.

## Production integration and E2E evidence

The production decision API accepts the completed local result; it contains no
gesture inference call. A dedicated core-0 worker handles MQTT/state acquisition
while the existing core-1 sensor loop retains 100Hz, 100-sample windows and 50-sample
steps. Queue capacity is two; missing state/queue overflow is reported explicitly.
Formal failover is not implemented. All split runtime/model files remain intact.

The normal server entry point is `python -m server.app.r1_mqtt`. R1 requests use
`inference-r1-v1`, mode CLOUD, exactly ten normalized features-v1 values and explicit
model/policy/firmware versions. Legacy fixed-split requests retain their old routes.
The application RTT probe is a shared pre-decision cost. Logs distinguish raw state,
normalized policy input, prediction reuse, measured application bytes, E2E latency
and server compute latency. Application bytes do not include TCP/Wi-Fi framing.

Both controlled E2E and normal production sampling validation now pass.
An initial E2E diagnostic reached CLOUD but had the wrong test expectation for
SHAKE window 332 (expected class 2 instead of the recorded class 4). It is retained
as `phase9_r1_e2e_initial_expectation_failure.log`; neither model nor output was
changed to satisfy that assertion. The corrected controlled test uses actual
SWIPE_RIGHT window 481 and its recorded cloud prediction.


`docs/evidence/test_phase9_r1_e2e_report.json` passes every correlation/contract
check. Its two controlled states use the same `processCachedDecision` function as
the production worker. LOCAL returns cached class 0 with zero request bytes.
CLOUD sends exactly ten normalized features and returns class 2 while cached LOCAL
was class 1 for development window 481. Both carry the learned policy version and
second_inference_count=0. The policy output was never forced. The controlled CLOUD
run measured 606 TX / 267 RX application bytes, 149.223ms E2E and 120.4535ms server
compute time. This is one diagnostic observation, not a network benchmark.

The normal `0.2.0-r1` firmware built and uploaded successfully to COM10. A 30-second
actual sensor capture produced 50 successful LOCAL decisions, zero second
inferences, zero queue-overflow/state-unavailable diagnostics, and final sampling
counters ov=0/rf=0. This run did **not** produce a live CLOUD decision. The CLOUD
branch evidence is the controlled production-core test, as allowed by the MVP
scope. Evidence: `phase9_r1_production_report.json`, its serial/server logs and
`phase9_r1_production_build.json`. No labeled live accuracy result is claimed.

Live free heap ranged 201368..201840 bytes, compared with the almost-constant
training anchor near 164592 bytes. Live RTT ranged 9.354..225.204ms in this short
capture. The heap shift demonstrates extrapolation outside training coverage;
normalization and weights were not changed to conceal it. The system is a
functional learned-policy MVP, not proof of robust deployment generalization.
Training uncertainty also used fixed MC seeds for reproducibility, whereas normal
firmware uses its existing device-seeded masks. Broader policy validity remains
limited by the single CLOUD-benefit source window and controlled replay assumptions.

## Regression and validation record

- Initial Python baseline: 282 passed, four deprecation warnings.
- Final complete Python suite: 309 passed, five TFLite deprecation warnings.
- Python compilation: 21 new/changed Python files pass.
- Dataset CLI: all 12 qualification checks pass; 2424 rows and zero group overlap.
- Desktop policy parity: 10/10; isolated ESP32 parity: 10/10 vectors, one Unity case.
- Controlled production-core E2E: one Unity case covering LOCAL and CLOUD passes.
- All six unchanged Phase7 split parity/E2E suites pass; each uses five vectors.
- Production build/upload and 50 live decisions pass. `git diff --check` passes.

Commands: `python -m ml.policy.r1_dataset --validate`, `python -m pytest -q`,
`python -m py_compile ...` (exact file list in `phase9_r1_compile_check.json`),
`pio test -e esp32-s3-n8r2 -f test_phase9_meta_policy_parity --upload-port COM10 --test-port COM10`,
`python -m tools.policy_dataset.validate_r1_hardware`, the same CLI with
`--suite 'test_phase7_*'`, `pio run -e esp32-s3-n8r2 -t upload --upload-port COM10`
from firmware, and `python -m tools.policy_dataset.capture_r1_runtime --seconds 30`.
The project virtual environment supplies Python; the user-local PlatformIO virtual
environment supplies pio. No new dependency was added.

Resolved diagnostics are retained: the initial generated INPUT/OUTPUT symbols
collided with Arduino macros (renamed); the first E2E expectation referenced the
wrong gesture class; and the split wrapper could not write a wildcard-containing
Windows filename after all six tests passed. The wrapper now sanitizes filenames;
its report recovers the completed run without redundant flashing. The old unrelated
internal-tensor introspection and rejected INT8 logits diagnostics were not rerun
or repaired. See `phase9_r1_split_regression_report.json` and preserved logs.

## M9 Definition of Done

All required MVP gates pass: qualified LOCAL/CLOUD labels; frozen reward/input
contract and group holdout; one learned policy; evaluation and rule comparison;
desktop TFLite parity; isolated ESP32 parity; both learned production-core branches;
exactly ten CLOUD features; versioned decision logs; preserved split artifacts and
passing regressions; explicit measured/simulated limitations. The production
integration adds a queue/worker around the validated local result without changing
feature extraction, sampling parameters, gesture weights or split inference code.
Historical four-action records/configurations and old blocked qualification reports
remain unchanged; they are superseded for deployment by the new R1-reuse contract.

No failover milestone, dashboard, OTA, RL or learned Split Controller was started.
The local R1 server can run with `python -m server.app.r1_mqtt`; uncurated service
telemetry belongs under ignored `data/policy/runtime/`. Stop another subscriber to
the same inference topics before starting it. The lab broker remains unchanged.

**Phase 9 / M9 - CLOSED.**

Next canonical phase: **Phase 10 / M10 - Failover**. This is the next task, not work
performed or claimed complete in this milestone.

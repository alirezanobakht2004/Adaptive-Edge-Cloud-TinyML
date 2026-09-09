# Canonical Phase 9 / M9 ? learned binary adaptive policy

Status: IN PROGRESS. R1 reuse-LOCAL qualification now passes; model/export/device
and production integration gates are being completed. M9 is not yet closed.
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

## Pending deployment evidence

One compact Input(6)?8 ReLU?4 ReLU?2 softmax policy uses fixed training settings,
training-only inverse-class-frequency loss weights, and untouched group holdout.
Evaluation, rule-based calibration, export, isolated parity and production E2E
results will be recorded here as each gate completes. No claimed result may be
inferred merely from the planned architecture. Split1/2/3 models, runtime and
regression suites remain experimental baselines.

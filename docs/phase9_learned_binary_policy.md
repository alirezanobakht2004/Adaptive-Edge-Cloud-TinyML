# Canonical Phase 9 / M9 — learned binary adaptive policy

**Status: BLOCKED at the training qualification gate. Not complete.**

Architecture R1 requires `0 LOCAL / 1 CLOUD`. LOCAL reuses the validated local
prediction and MC uncertainty result; CLOUD sends exactly 10 normalized features-v1
values to the full-cloud model through MQTT. No policy was trained or deployed.

## Migration and production status

The [migration audit](phase9_r1_migration_audit.md) records the clean baseline,
reference classification and external canonical cleanup. README, canonical
architecture, current protocol documentation, configuration annotation and
retired Split Controller placeholders now distinguish R1 production from history.

Production inference behavior is unchanged. The local ESP32 B3/MC path remains
validated; existing server routes remain fixed-split services. The documented R1
request schema is a target, not a deployed endpoint. `cloud_full.py` already
provides real 10-feature full-cloud inference for the historical benchmark and
can be reused after gates pass. No raw IMU or split embedding is proposed for the
normal production CLOUD request.

Split1/2/3 artifacts, models, prefix runtime, cloud-tail routing, fixed-split
benchmarks and parity/E2E regressions remain intact. Historical four-action
datasets, configs and reports retain their original labels and phase names.
No learned Split Controller, split redesign, RL, EWC, OTA, database, dashboard or
failover work was performed.

## Dataset and frozen contracts

See [binary qualification](phase9_binary_dataset_qualification.md) and its
[machine-readable report](evidence/phase9_binary_dataset_qualification.json).

- Dataset: `policy_training_dataset_binary_v1`, derived from accepted M30 v2,
  M31 controlled v2 and eligible M33 v3 records; all four original outcomes retained.
- 168 comparison records, 24 original windows, one session/acquisition run.
- 24 measured records with estimated proxy energy; 144 simulated records.
- 96 LOCAL / 72 CLOUD historical binary labels; all CLOUD labels are simulated.
- 24 unavailable-LOCAL M33 rows excluded. Unmatched v1 observations and duplicate
  canonical mirrors are not imported as counterfactual evidence.
- Contract: `policy-config-v2` / `binary-policy-contract-v1`. Frozen order:
  confidence, entropy, margin, free heap, local inference estimate, MQTT RTT.
- Normalization: `binary-state-normalization-v1`; explicit fixed unit divisors,
  six inputs with documented pre-decision ESP32 sources. No synthetic-only input.
- Reward: `binary-reward-experiment-v1`, exact provisional M30
  `reward-calibration-v1` weights/scales, immutable provenance and SHA-256.
- Group holdout: seed 20260908; 18 train windows / 126 records and six holdout
  windows / 42 records. No underlying window appears in both partitions.

These are reserved partitions, not completed training or test evaluations.

## Gate result and exact blocker

Binary comparison schema, reward arithmetic, runtime feature sources, provenance
and leakage checks pass. **Production action-cost compatibility fails.** M30
executes local inference once for policy state and again for action 0; its
controlled descendants penalize the second execution. R1 LOCAL must reuse the
first result. There are zero measurements of that result-reuse candidate in the
source campaigns. The new historical labels cannot be claimed to optimize the
required production branch. No labels or costs were silently corrected.

The existing 96 local repeats measure an extra pass averaging 1.7986875 ms.
All source pre-decision local predictions are correct. Merely using simulated
edge pressure to increase this redundant pass's penalty is not evidence of a
production cloud advantage. Energy remains estimated/simulated, never measured.

## Learned model and rule-based baseline

No learned model version exists. The architectural candidate remains
`Input(6) → Dense(8, ReLU) → Dense(4, ReLU) → Dense(2, softmax)`, **not trained**.
No seed-driven training run, fitted weights, history, model hash or evaluation
metrics exist. No rule-based binary thresholds were invented or fitted; its
same-holdout result is unavailable. Historical four-action rule-based study
results are not binary baseline results.

All-LOCAL, all-CLOUD, rule-based and learned comparisons on a qualified production
dataset remain pending. No accuracy, F1, regret, oracle reward or selection-rate
claim for a learned policy is made.

## Export, device parity and production E2E

| Required result | Status |
|---|---|
| Learned TFLite size / SHA-256 / desktop parity | Not generated; training gate blocked |
| Isolated ESP32 policy output/action parity | Not run; no learned artifact |
| Learned LOCAL branch with no cloud request | Not run |
| Learned CLOUD branch transmitting 10 features | Not run |
| Production correlated response and policy-version decision logging | Not integrated |
| Existing split regressions | Retained; Python suite exercises desktop parity/routing; historical six-suite hardware evidence retained |

No ESP32 was flashed for this migration. A comment marking a retired placeholder
does not alter firmware execution. Known old INT8/introspection diagnostics remain
separate in the migration audit and are not new regressions.

## Validation executed

| Check | Result |
|---|---|
| Initial full Python baseline | 255 passed, 0 failed, four deprecation warnings |
| R1 documentation/configuration migration regression | 255 passed, 0 failed |
| Binary contract regression | 270 passed, 0 failed |
| Dataset qualification regression | 282 passed, 0 failed |
| Final full Python regression | **282 passed, 0 failed**, four deprecation warnings, 39.40 s |
| `python -m py_compile` on both new utilities, both new test files and retired Python placeholder | Passed |
| Binary dataset reconstruction/qualification CLI | Passed structurally; training eligibility correctly false |
| `git diff --check` | Passed |
| Historical model/dataset/runtime preservation diff | Only retired controller placeholder comment changed under firmware; no runtime, model or historical dataset changes |

Commands used `.venv\Scripts\python.exe`; new tests account for 27 additional
cases. The full suite includes existing split desktop Keras/TFLite parity, server
routing and MQTT service tests. No new policy TFLite, hardware or production E2E
tests ran because the training gate stopped artifact creation. Historical hardware
evidence is explicitly not counted as a new passing run. Validation details are in
`docs/evidence/phase9_r1/validation.json`.

## Definition of Done and next work

R1 documentation and versioned contract: done. Historical binary derivation,
provenance, reward freeze, group isolation and qualification: done. Production
training qualification: **blocked**. Training, baseline calibration, export,
isolated parity, production integration and adaptive E2E: not started.

The device does not yet choose LOCAL/CLOUD through a learned policy, so **Phase 9
/ M9 is not closed**. The next task remains in Phase 9: obtain R1-compatible
matched outcomes for reuse-LOCAL versus full-cloud, including actual request-byte
overhead and explicit measured/simulated provenance; reassess useful action
diversity without forcing labels. Only then revisit training and deployment gates.
Canonical Phase 10 / M10 — Failover remains **not started**.

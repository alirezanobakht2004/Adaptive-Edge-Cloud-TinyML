# Phase 9.5 — External dataset evaluation

M19-M24 evaluation completed on 2026-09-08. Phase10 has not started. The external
resource is useful as an **isolated, conditional robustness stress test**, but it
is not directly compatible with dataset-v1 and does not establish a pretraining
benefit. No production model, feature extractor, firmware or inference runtime
was changed or trained.

## Dataset and provenance

The supplied [Kaggle dataset](https://www.kaggle.com/datasets/dilharajayawardhane/6-axis-motion-gesture-dataset-hand-waves-and-flicks)
is attributed to Dilhara Jayawardhane. Its linked repository now resolves to
`Dilharajay/gesture-cmd`; this evaluation records revision
`0f1e63bad76366c7d415e1e42649841b72635dbc` in `source_context.json`.

The associated [acquisition code](https://github.com/Dilharajay/gesture-cmd/blob/0f1e63bad76366c7d415e1e42649841b72635dbc/firmware/src/data_collection.cpp)
supports 50 Hz acquisition, 50-sample windows, acceleration in g, and gyro in
degrees/s. The exact capture revision is not present in the CSV. Kaggle describes
ESP32 hardware while the current linked code describes ESP8266, so those units
are supported by associated code rather than capture-level attestation.
Mounting relative to `orientation-v1` remains unknown. No axes were rotated.

| Property | Measured CSV result |
|---|---|
| Sensor rows | 55,000 |
| Recordings | 1,100 |
| Recording identity | `(label, sample_id)` |
| Samples per recording | 50 in every recording |
| Within-recording spacing | 20 ms in all 53,900 intervals: 50 Hz |
| First-to-last sample span | 980 ms; nominal window period is 1 second |
| Missing / invalid numeric values | 0 / 0 |
| Duplicate complete rows | 0 |
| Duplicate sensor recordings, ignoring label/timestamps | 0 |
| Nonpositive within-recording timestamp deltas | 0 |
| Timestamp field | `timestamp_ms`; not UTC |
| Sensor columns | `ax, ay, az, gx, gy, gz` |

| Original label | Recordings | Sensor rows |
|---|---:|---:|
| idle | 300 | 15,000 |
| wave_left | 200 | 10,000 |
| wave_right | 200 | 10,000 |
| flick_up | 200 | 10,000 |
| flick_down | 200 | 10,000 |

There are 300 distinct `sample_id` strings; 200 are reused across labels. They
must not be interpreted as independent participant sessions. The associated
[logger](https://github.com/Dilharajay/gesture-cmd/blob/0f1e63bad76366c7d415e1e42649841b72635dbc/server/src/training/gesture_logger.py)
also scopes recording IDs by gesture. Participant identity, independent session
boundaries, calibration and mounting are missing from the supplied CSV.

The raw file SHA-256 is
`839fbe1de07f180f0434288f73fe0ef42b5e5a3a23dd6548efd54bdab2bd575b`.
It remained unchanged. Source material and generated NPZ arrays stay local under
`data/external/kaggle_6axis_motion_v1/`; scoped ignore rules exclude them from Git.
Scripts, provenance, conversion metadata, mapping and measured reports are committed.
The source page exposes inconsistent license labels; its applicable version terms
remain a provenance question for future reuse, not an asserted license conclusion.

## Conversion and feature compatibility

`converted/recordings.npz` preserves parsed numeric values as float64, timestamps
as int64, original labels and IDs. Offsets delimit each original recording.
`external_dataset_metadata.json` documents identity axis mapping, unknown mounting,
units evidence, source hash and conversion version `external-recordings-v1`.
Conversion performs no interpolation, unit scaling, relabeling or row dropping.
Output outside `data/external` is rejected.

The unchanged `features-v1` extractor requires `(100, 6)` at 100 Hz with ten output
features. **Zero external recordings meet its native window contract.** A separate,
explicit `--diagnostic-resample` experiment used linear interpolation from observed
times 0..980 ms onto a 0..990 ms grid at 10 ms intervals. The final 990 ms value
holds the observed endpoint. That value and interpolated intermediate values are
derived, not additional measurements. The adapter cannot restore missing bandwidth.

The adapted experiment generated **1,100 finite ten-feature vectors** through the
existing extractor. Frozen model normalization was applied without refitting and
checked against the actual Keras normalization layer. The largest absolute
normalized feature value was **42.350536** in training-standard-deviation units.
Thus numerical dimensional compatibility does not establish distribution compatibility.
Unknown mounting and capture calibration remain unresolved; native compatibility
is still reported as false after the diagnostic experiment.

## Frozen-model evaluation

Only `gesture-model-v1.1.0` was evaluated, with its existing SHA-256
`4891b4b6d453d96852ddcaea35d847ff8eea4ef1349ae999d7594925dabc5d6c`.
Weights were checked unchanged before/after inference. No optimizer, fitting,
normalization adaptation, production artifact export or policy training was used.

`external_label_mapping.json` maps only `idle -> IDLE`. The source
[README](https://github.com/Dilharajay/gesture-cmd/blob/0f1e63bad76366c7d415e1e42649841b72635dbc/README.md)
describes resting hand positions, while `docs/dataset_protocol.md` defines IDLE
as held still without an intended gesture. This supports broad non-motion semantics,
not matched orientation or calibration. Waves are not assumed to be swipes;
flicks have no corresponding production label. All four remain explicitly unmapped.

| Mode | Internal validation accuracy (200 windows) | External mapped idle accuracy (300 windows) | Overall external accuracy |
|---|---:|---:|---|
| Deterministic | 98.5% | 0.0% | Undefined: 800 recordings have unmapped labels |
| Five-pass MC dropout | 98.5% | 0.0% | Undefined: 800 recordings have unmapped labels |

All 1,100 external recordings received predictions and uncertainty estimates.
Only 300 entered correctness/confusion calculations. The deterministic external
idle confusion row, in `[IDLE, SWIPE_LEFT, SWIPE_RIGHT, ROTATE_CW, SHAKE]` order,
is `[0, 250, 38, 2, 10]`; the MC5 row is `[0, 209, 80, 2, 9]`. Other true-class
rows are empty because no external labels were assigned to those classes.
These are conditional mapped-subset results, not overall gesture accuracy or
proof of a particular causal explanation for the errors.

## Uncertainty findings

The internal reference is the existing validation session `session_02`, not the
training or held-out test split. Deterministic mode disables dropout. MC5 uses
five seeded stochastic masks at the existing head dropout rate 0.2 and computes
predictive entropy from their mean probabilities. Learned layers remain frozen.

| Mode / cohort | Mean confidence | Mean entropy (nats) | Mean prediction margin |
|---|---:|---:|---:|
| Deterministic / internal | 0.993417 | 0.029375 | 0.987796 |
| Deterministic / external | 0.910486 | 0.182357 | 0.823201 |
| MC5 / internal | 0.985874 | 0.059807 | 0.974035 |
| MC5 / external | 0.848933 | 0.315001 | 0.706843 |

Histograms, per-label distributions and a label-controlled idle comparison are in
[external_uncertainty_report.md](evidence/external_uncertainty_report.md) and its
[standalone plot](evidence/external_uncertainty_histograms.png). Histogram fractions
account for the differing cohort sizes; exact counts are retained in the report.

Despite higher average external entropy, **46/300 deterministic idle errors** and
**36/300 MC5 idle errors** had confidence at least 0.9. That threshold is a
descriptive count, not a selected adaptive-policy threshold. The experiment shows
why confidence alone needs stress testing; it does not establish a calibrated
out-of-distribution detector. Pooled cohorts have different gestures and class
balance, and unknown mounting, sampling adaptation and acquisition differences
confound the comparison. The wave classes can also produce very high confidence,
but their predictions cannot be labeled correct or incorrect under this mapping.

## Decision

| Question | Evidence-based decision |
|---|---|
| Can it be used for robustness evaluation? | Yes, as a separate diagnostic shift/stress resource with explicit adaptation and label limits. Current results expose mapped idle failures and high-confidence errors. It is not a native five-class benchmark. |
| Can it be used for pretraining? | Not yet justified for production. Six sensor channels make it a possible research resource, but pretraining benefit is unmeasured. A separately approved experiment needs verified units/mounting/provenance, participant/session-aware evaluation and comparison against an unchanged baseline. |
| Can it be merged into dataset-v1? | No. Dataset-v1 is frozen, gestures differ, sampling differs, and orientation/session contracts are missing. No merge was performed. |
| What changes are required? | For stronger external claims, obtain capture revision, mounting, calibration and independent session/participant information; validate adaptation and any additional semantic mapping. Any future combined dataset requires its own version and an explicit protocol, not changes to dataset-v1. |

For Phase10 planning, retain this as a separate robustness resource and keep the
validated production system unchanged. Do not add these gesture rows to the
policy decision dataset or infer offloading rewards from them. No measured
pretraining gain, model improvement, latency, energy benefit or policy readiness
is established. This evaluation does not start Phase10.

## Reproduction and validation

Run from the repository root using the existing Python environment. Supply the
original CSV at its documented path; its hash must match `source_context.json`.
On a fresh Git checkout the tracked conversion metadata already exists, so use
`--rebuild` to recreate only derived conversion files for that same source/version.

```powershell
python -m tools.external_dataset.inspect_kaggle_dataset --source-context data/external/kaggle_6axis_motion_v1/source_context.json
python -m tools.external_dataset.convert_kaggle_dataset --source-context data/external/kaggle_6axis_motion_v1/source_context.json --rebuild
python -m tools.external_dataset.run_feature_compatibility --diagnostic-resample
python -m tools.external_dataset.evaluate_existing_model
python -m ml.evaluation.external_uncertainty_analysis
python -m py_compile <new external-evaluation scripts and tests>
python -m pytest -q
git diff --check
```

The initial conversion ran without `--rebuild`; tests also verify controlled
rebuilding. Evaluation captures TensorFlow/NumPy versions, source/model/artifact
hashes, seed, pass count and the transformation version. Conversion timestamps
are run metadata. Recreating reports under different numerical library versions
may introduce small numerical differences; pinned data/weights do not make every
environment bit-identical.

Validation: **179 Python tests passed**, with four existing TensorFlow Lite
deprecation warnings; compilation and whitespace checks passed. New tests cover
label-scoped IDs, missing/nonfinite values, lossless conversion, forbidden output
paths, native window rejection, interpolation/endpoints, irregular timestamps,
unmapped-label exclusion, histogram boundary counts, repeatable MC inference and
unchanged model weights. The measured report and plot were generated and checked.
No ESP32 tests or firmware flashes were run.
The final [Python test log](evidence/external_validation_tests.log) is retained.
The original CSV hash and all converted/feature/probability archive hashes were
checked against their reports; protected production paths have no Git diff.

Deliverables are the four command-line tools under `tools/external_dataset/`,
`ml/evaluation/external_uncertainty_analysis.py`, four external test modules,
isolated conversion/mapping/provenance metadata, the three required JSON evidence
reports, uncertainty Markdown/PNG, and this decision document.

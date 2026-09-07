# Phase 7 / M8 â€” split inference validation

Validation ran on 2026-09-07; closure recorded on 2026-09-08.
**M8's float32 split-inference gate is closed:** all three prefixes, tails,
parity paths and ESP32/MQTT round trips passed. The unfiltered firmware suite
is **not fully green** because of two unchanged legacy INT8 diagnostics,
detailed below. This closure does not claim those diagnostics were repaired.

## Architecture and tensor contracts

`Instruction/PROJECT_ARCHITECTURE.md` was read first. Source model
`gesture-model-v1.1.0`, dataset-v1, features-v1, class IDs, sampling rate and
normalization remain fixed. No adaptive policy, OTA, continual learning or
dashboard changes were made.

| Split | ESP32 prefix | Embedding | Server continuation | Output |
|---|---|---:|---|---:|
| 1 | B1: 10 â†’ 64 | 64 | 48 ReLU â†’ 32 ReLU â†’ 5 softmax | 5 |
| 2 | B1+B2: 10 â†’ 64 â†’ 48 | 48 | 32 ReLU â†’ 64 ReLU â†’ 32 ReLU â†’ 5 softmax | 5 |
| 3 | Existing B1+B2+B3: 10 â†’ 64 â†’ 48 â†’ 32 | 32 | Existing 64 ReLU â†’ 32 ReLU â†’ 5 softmax | 5 |

The shorter Split1 tail was explicitly requested and completed in the prior
milestone. It is preserved here. Split2 and Split3 follow the canonical deeper
cloud continuation. The tails are separate trained artifacts, so identical
predictions across all split choices are not a mathematical requirement. Parity
compares each deployed path with its corresponding Python reference.

All prefix inputs are externally normalized float32 features-v1 vectors of size
10. The server receives the embedding directly without additional normalization.
Keras tails accept `(None, embedding_dimension)` and return `(None, 5)`.
TFLite tests use batch size one; all new exports are float32, without quantization.

## Artifacts and implementation

All model paths below are relative to
`data/processed/dataset-v1/features-v1/models/`.

| Artifact | Path | Bytes |
|---|---|---:|
| Split1 prefix, unchanged | `gesture-model-v1.1.0/tflite/splits/gesture-model-v1.1.0-split1-prefix-float32-normalized-input.tflite` | 3,996 |
| Split2 prefix, new | `gesture-model-v1.1.0/tflite/splits/gesture-model-v1.1.0-split2-prefix-float32-normalized-input.tflite` | 16,984 |
| Split3 prefix, reused | `gesture-model-v1.1.0/tflite/gesture-model-v1.1.0-prefix-b3-float32-normalized-input.tflite` | 23,868 |
| Split1 cloud, unchanged | `gesture-cloud-tail-split1-v1.0.0/gesture-cloud-tail-split1-v1.0.0.tflite` | 21,972 |
| Split2 cloud, new | `gesture-cloud-tail-split2-v1.0.0/gesture-cloud-tail-split2-v1.0.0.tflite` | 26,820 |
| Split3 cloud, newly exported from unchanged Keras | `gesture-cloud-tail-v1.0.0/gesture-cloud-tail-v1.0.0.tflite` | 19,876 |

Split2's model directory also contains its Keras model, `metadata.json`,
`training_history.json`, `split2_export_report.json`, and
`split2_cloud_parity_vectors.npz`. Its prefix directory contains
`split2_export_report.json` and `split2_parity_vectors.npz`. Split3's existing
model directory gains only its TFLite export, `split3_export_report.json`, and
`split3_cloud_parity_vectors.npz`; its Keras model and original metadata are unchanged.

Split2 training freezes the source prefix, extracts B2 embeddings from the
existing TRAIN/session_01 (600 samples) and VALIDATION/session_02 (200 samples),
copies source B3 and existing Split3 cloud weights as initialization, then trains
only the separate B3/B4/B5/head continuation. Early stopping restored the best
validation-loss weights after 22 epochs. TEST/session_03 was not loaded. Source
weights are verified unchanged and artifacts record SHA-256 provenance.

New Python source:

- `ml/models/cloud_tail_split2.py`
- `ml/training/train_cloud_tail_split2.py`
- `ml/export/export_split2_cloud_tail.py`
- `ml/export/generate_split2_firmware.py`
- `ml/export/split_validation.py`
- `ml/export/validate_split3.py`
- `ml/export/generate_phase7_e2e.py`
- `tests/test_split2_cloud_tail.py`
- `tests/test_split3_cloud_tail.py`
- `tests/test_split_routing.py`
- `tests/run_phase7_hardware.py`

Firmware additions are `split2_model_data.h/.cpp`, generated expectation/vector
headers, and isolated `test_phase7_split2_parity`, `test_phase7_split2_e2e`,
`test_phase7_split3_parity`, and `test_phase7_split3_e2e` suites. Split3 tests use
the existing `initPrefixRunner`/`runPrefixB3` API. No production runtime was rewritten.

`server/app/inference.py`, `schemas.py`, and `mqtt.py` now route splits 1, 2 and 3.
Existing request topics, envelopes, source-version validation and response fields
are retained. Embeddings must contain exactly 64, 48 or 32 finite numeric values,
respectively. Invalid splits and cross-split dimensions are rejected.

## Validation method and results

The five established validation vectors are reused. Tests compare source Keras
prefixes, desktop TFLite prefixes, cloud Keras, cloud TFLite and the actual ESP32
path. E2E firmware sends the computed embedding and checks request ID, class,
confidence, split ID and selected cloud version in the MQTT response.

| Check | Split1 | Split2 | Split3 |
|---|---:|---:|---:|
| Cloud Keras/TFLite maximum absolute difference | 1.16e-10 | 2.24e-8 | 3.73e-9 |
| Required cloud tolerance | 2e-5 (established) | 1e-5 | 1e-5 |
| Desktop prefix maximum absolute difference | Prior verified evidence | 4.77e-7 | 7.15e-7 |

- `python -m py_compile` on all new/modified Python implementation and test files: passed.
- `pytest -q`: **127 passed**, four TensorFlow Lite interpreter deprecation warnings.
- Isolated Split2 ESP32 prefix test: **passed**, five vectors,
  `SPLIT2_MAX_ABS_DIFF=0.000000000`.
- Full unfiltered PlatformIO hardware suite: **23 of 25 suites passed**;
  **28 test cases passed**, one test case failed, and one suite failed to compile.
- All six Phase7 suites passed: Split1/2/3 parity and Split1/2/3 MQTT E2E.
  Each parity suite reported `MAX_ABS_DIFF=0.000000000` on five vectors.
  All 15 cloud responses were received and checked on the ESP32.

| Phase7 hardware suite | Result |
|---|---|
| `test_phase7_split1_parity` | PASS, 5 vectors |
| `test_phase7_split1_e2e` | PASS, 5 round trips |
| `test_phase7_split2_parity` | PASS, 5 vectors |
| `test_phase7_split2_e2e` | PASS, 5 round trips |
| `test_phase7_split3_parity` | PASS, 5 vectors |
| `test_phase7_split3_e2e` | PASS, 5 round trips |

The returned class sequence for every split was IDLE, SWIPE_LEFT, SWIPE_RIGHT,
ROTATE_CW, SHAKE. Per-vector confidence, split and model version are preserved in
`docs/evidence/phase7/phase7_validation_summary.json`, extracted and checked
against the model parity NPZs from `full_hardware.log`.

The full-suite exceptions are left visible and unmodified:

1. `test_internal_tensor_diagnostic` does not compile against Chirale TFLM 2.0.0:
   it uses an unsupported seven-argument constructor, `preserve_all_tensors`,
   and `GetTensor` APIs.
2. `test_tflm_logits_diagnostic` runs the legacy INT8 path and fails its desktop
   class-agreement gate (`Expected 0 Was 1`; maximum INT8 logits difference 18 LSB).

These are not failures in the float32 split artifacts. Fixing INT8 deployment or
changing the TFLM dependency is outside this milestone. The full hardware command
therefore returned a nonzero result, despite all Phase7 suites passing. The
production local float32, feature, preprocessing, uncertainty, Wi-Fi, MQTT,
timeout and window-buffer regression suites also passed.

Two earlier harness attempts were incomplete: one was restarted to disable
subprocess stdout buffering, and one hit Windows console encoding on a boot byte.
Both harness issues were corrected before the complete run; their logs are
retained with `_attempt` filenames and do not count as successful full runs.

Raw hardware evidence is saved in `docs/evidence/phase7/`. The runner supplies
the existing receive-only MQTT test's fixed transport message and simulates an
unavailable server only during the existing timeout test. Actual E2E requests
use the production server callback and all three real model runtimes.

## Commands

Use the existing `.venv` Python from the repository root:

```powershell
.venv\Scripts\python.exe -m ml.training.train_cloud_tail_split2
.venv\Scripts\python.exe -m ml.export.export_split2_cloud_tail
.venv\Scripts\python.exe -m ml.export.generate_split2_firmware
.venv\Scripts\python.exe -m ml.export.validate_split3
.venv\Scripts\python.exe -m ml.export.generate_phase7_e2e --split 2
.venv\Scripts\python.exe -m ml.export.generate_phase7_e2e --split 3
.venv\Scripts\pytest.exe -q
.venv\Scripts\python.exe -u -m tests.run_phase7_hardware --port COM10
```

The hardware runner executes unfiltered:

```text
pio test -d firmware -e esp32-s3-n8r2 --upload-port COM10 --test-port COM10 -v
```

Compilation was run with `python -m py_compile` and explicit paths for all changed
Python files. Export/generation scripts refuse to overwrite existing versioned
evidence. No new dependencies were installed. `git status` and `git diff --check`
were run after each logical commit.

## Known limitations

Validation uses fixed feature vectors, not a live-sensor recognition experiment.
No held-out accuracy, latency benchmark, battery measurement or energy claim is
made. Individual server timestamps/confidences in logs are diagnostic observations.
The established JSON buffers fit the validated payloads; arbitrary envelopes and
decimal formatting require separate size validation.

The board is flashed with isolated test images during validation. Production
application selection among the actions and learned policies are not implemented
by this milestone. All models, dataset and feature definitions remain versioned
and fixed for the next benchmark phase.


## Definition of Done

- [x] Split1 retained and revalidated on ESP32 and MQTT.
- [x] Split2 Keras, float32 tail, prefix, reports and parity artifacts saved.
- [x] Split2 prefix and five-sample hardware MQTT round trip passed.
- [x] Existing Split3 Keras and B3 runtime retained; float32/parity evidence added.
- [x] Split3 prefix and five-sample hardware MQTT round trip passed.
- [x] All three server routes validated with real models.
- [x] Python and per-path TFLite validation passed.

The temporary test server disconnected after the suite. The existing broker was
left running. The ESP32 was left with the final `test_window_buffer` test image;
no production application source was changed or automatically restored.

## Commit record

The previously completed, uncommitted Split1 work was recorded separately as
`f7ef266` before starting the requested six-commit sequence:

1. `81c04f8` ? `feat(split2): add split2 cloud tail training and export`
2. `f02cadf` ? `feat(split2): add esp32 prefix and parity validation`
3. `ab12bd7` ? `feat(split-routing): add mqtt server routing for all splits`
4. `1680a8e` ? `test(phase7): validate split2 and split3 e2e`
5. `docs(phase7): close M8 split inference milestone` ? this closure document.
6. `docs(phase8): add benchmark action space plan` ? subsequent planning only.

Phase8 can now plan measurements for actions 0?3. No policy training data,
measured comparative benchmark, learned policy or adaptive firmware is claimed
by this closure.

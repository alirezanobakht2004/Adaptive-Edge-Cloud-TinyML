# Phase 7 / M8: Split1 completion

Validated on 2026-09-07. The Split1 milestone is complete for the fixed-vector
ESP32 → MQTT → server → ESP32 path. This does not complete all of M8.

## Architecture and scope

Read `Instruction/PROJECT_ARCHITECTURE.md` before implementation. Sections 10–11
describe a proposed B2+B3+B4+B5+head continuation. The explicit current milestone
request specifies a separate, shorter Split1 tail; this implementation follows
that requested contract:

```text
features-v1 (10) → frozen normalization → frozen B1 Dense(64, ReLU)
                 ESP32 | Split1 embedding: float32[64] | server
                 B2 Dense(48, ReLU) → B3 Dense(32, ReLU)
                 → Cloud Head Dense(5, softmax)
```

Model name: `gesture_cloud_tail_split1`.
Version: `gesture-cloud-tail-split1-v1.0.0`.
Keras shapes: `(None, 64)` → `(None, 5)`.
TFLite invocation shapes: `(1, 64)` → `(1, 5)`, float32, no quantization.

The existing `gesture-cloud-tail-v1.0.0.keras` remains the independent 32-input
Split3 tail. These tails have different architectures and independently trained
weights; cross-split equality is not asserted.

No production firmware source, prefix artifact, local B3 runtime, feature
extractor, dataset, class mapping, sampling configuration, or dependency changed.
No Meta Learner, Split Controller, OTA, continual learning, or dashboard work.

## Completed files

New source files:

- `ml/models/cloud_tail_split1.py`: named model, metadata contract, shape validation.
- `ml/training/train_cloud_tail_split1.py`: source hash verification, frozen B1
  embedding extraction, copied B2/B3/edge-head initialization and tail-only training.
- `ml/export/export_split1_cloud_tail.py`: float32 export, parity validation,
  report update, generated firmware test expectations.
- `tests/test_split1_cloud_tail.py`: deterministic inference, real prefix/tail
  parity, both MQTT routes, malformed inputs, metadata and artifact corruption.
- `firmware/test/test_phase7_split1_e2e/test_main.cpp`: isolated hardware test
  using the existing prefix, parity inputs, Wi-Fi and MQTT helpers.
- `firmware/test/test_phase7_split1_e2e/cloud_expectations.h`: generated class and
  confidence expectations tied to the cloud Keras SHA-256.
- `docs/phase7_split1_completion.md`: this record.

Modified files:

- `server/app/inference.py`: validated Split1 runtime and fixed split routing,
  sharing the existing inference implementation with Split3.
- `server/app/schemas.py`: accept split 1 with 64 values or split 3 with 32 values.
- `server/app/mqtt.py`: route the selected split and include `split` in responses.
- `tests/test_mqtt_inference.py`: updated unsupported-split and response assertions.

Saved under
`data/processed/dataset-v1/features-v1/models/gesture-cloud-tail-split1-v1.0.0/`:

- `gesture-cloud-tail-split1-v1.0.0.keras`
- `gesture-cloud-tail-split1-v1.0.0.tflite` (21,972 bytes)
- `metadata.json`
- `training_history.csv`
- `split1_export_report.json`
- `split1_cloud_parity_vectors.npz`

Training used 600 TRAIN/session_01 and 200 VALIDATION/session_02 samples.
Early stopping ran 37 epochs and restored the best validation-loss weights.
The source weights were checked unchanged in memory and the source file's hash
was verified unchanged. TEST/session_03 was not loaded.

## MQTT contract

Existing topics and envelope fields remain required. A payload containing only
`split` and `embedding` is shorthand, not the complete wire request.

Request topic: `gesture/<device_id>/inference/request`.

```json
{
  "request_id": "split1-e2e-0",
  "device_id": "esp32-split1-test",
  "timestamp_ms": 123456,
  "split": 1,
  "embedding": ["64 finite numeric values; replace this explanatory string"],
  "model_version": "gesture-model-v1.1.0"
}
```

The request model version identifies the frozen **edge source**. The embedding
is B1's output; do not normalize it again. Boolean, string, nonfinite, wrong-size,
unsupported-split, and incompatible-version requests are rejected.

Response topic: `gesture/<device_id>/inference/response`. Responses contain
`request_id`, `predicted_class`, `confidence`, `split`, `model_version`, and the
existing `server_latency_ms` field. The response model version identifies the
**selected cloud tail**. Adding `split` preserves all existing response fields.

The firmware test serializes float values with nine significant digits and
checks that the complete packet fits the existing 1,024-byte MQTT buffer.

## Validation and results

Five established validation vectors from the existing prefix parity NPZ are
continued through the new cloud model. The new cloud NPZ records normalized
inputs, embeddings, Keras probabilities, TFLite probabilities, and original labels.

- Keras repeated inference: deterministic.
- Keras vs. float32 TFLite maximum absolute probability difference:
  `1.1641532182693481e-10`, tolerance `2e-5`; class IDs agree.
- Desktop test executes the frozen prefix TFLite from 10 normalized features,
  compares the 64-value embedding, and checks both tail TFLite and server Keras.
- Python compilation: passed.
- Full pytest suite: **113 passed, 1 warning** (existing TensorFlow Lite interpreter
  deprecation). No skipped tests.
- ESP32-S3 on **COM10**, broker `192.168.137.1:1883`: **1 hardware test passed**.
  It invokes B1 for each of five fixed vectors, verifies embedding parity, sends
  the actual output through MQTT, and checks request correlation, class,
  confidence within `2e-5`, split 1, and cloud version on the returned response.
- Server observed `split1-e2e-0` through `split1-e2e-4`, respectively returning
  IDLE, SWIPE_LEFT, SWIPE_RIGHT, ROTATE_CW, and SHAKE.

Hardware result:

```text
test/test_phase7_split1_e2e/test_main.cpp:92: testSplit1EndToEnd [PASSED]
esp32-s3-n8r2:test_phase7_split1_e2e [PASSED]
1 test cases: 1 succeeded
```

The initial hardware build failed on a generated `1f` literal. The exporter now
generates scientific-notation C++ floats, and the subsequent build/upload/test
passed. Production firmware source was untouched; the board was flashed with
the isolated test image for this validation.

## Commands executed

Run from the repository root in PowerShell using the existing `.venv` (the default
system Python has no TensorFlow). No dependency installation was needed.

```powershell
.venv\Scripts\python.exe -m ml.training.train_cloud_tail_split1
.venv\Scripts\python.exe -m ml.export.export_split1_cloud_tail
.venv\Scripts\python.exe -m py_compile ml/models/cloud_tail_split1.py ml/training/train_cloud_tail_split1.py ml/export/export_split1_cloud_tail.py server/app/inference.py server/app/mqtt.py server/app/schemas.py tests/test_split1_cloud_tail.py tests/test_mqtt_inference.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\pytest.exe -q
.venv\Scripts\python.exe -u -m server.app.mqtt
& 'C:\Users\alire\.platformio\penv\Scripts\platformio.exe' test -d firmware -e esp32-s3-n8r2 -f test_phase7_split1_e2e --upload-port COM10 --test-port COM10
git diff --check
```

The generated expectation header was regenerated after correcting its float
format using `write_firmware_expectations` with the saved NPZ and metadata hash;
the trained model was not retrained or overwritten. Training/export scripts
refuse to overwrite existing versioned evidence. The temporary server process
used for validation is stopped after testing; the existing broker is left running.

## Definition of Done

- [x] Split1 cloud tail model exists.
- [x] Keras artifact saved.
- [x] TFLite float32 artifact exported.
- [x] Server accepts split=1 and preserves split=3.
- [x] Actual ESP32-computed embedding reaches the server over MQTT.
- [x] End-to-end response is verified on the ESP32 for all five parity vectors.
- [x] Python and isolated hardware tests pass.

## Limitations and remaining M8 work

This validates fixed-vector split inference, not live sensor classification,
adaptive selection, held-out accuracy, or a latency benchmark. The individual
server timing field is diagnostic only; no latency or accuracy claim is made.
The five validated packets fit the current MQTT buffer; unrestricted decimal
serialization or larger envelopes may require a separately scoped buffer change.

Split2 still needs its 48-value prefix/tail artifacts, parity tests, server route,
and hardware round trip. Split3's existing 32-value cloud tail and Python route
remain operational; complete Phase7 split-prefix export/parity and an actual
B3-embedding hardware round trip should be verified separately. Payload and
latency measurements across the three splits remain future M8 work.

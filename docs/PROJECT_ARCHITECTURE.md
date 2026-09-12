# Adaptive Edge–Cloud TinyML for Gesture Recognition
## Canonical Project Architecture & Execution Plan

> **Status:** Canonical architecture document for the project  
> **Purpose:** This file is the main technical source of truth for the project. Any implementation, code organization, experiment, report section, or design decision should be checked against this document first.  
> **Core project title:** سامانه هوشمند تطبیقی لبه–ابر مبتنی بر TinyML  
> **Case study:** Handheld Movement / Gesture Recognition using ESP32-S3 + MPU6050

---

## Architecture Revision R2 — 2026-09-12

**Status:** Final undergraduate scope frozen; defense-ready.

Final implementation scope completed through Phase 11. Phases 0–11 are completed;
Phases 12–14 are future work and are not completion requirements for the undergraduate
project. This revision does not change the R1 production architecture or add features.

The completed contribution set is:

- Adaptive Edge–Cloud Inference
- Local TinyML inference
- Uncertainty Estimation
- Learned Binary LOCAL/CLOUD Policy
- MQTT / Server-Assisted Inference
- Failover
- PostgreSQL telemetry persistence
- Live Dashboard / Observability
- Validated Split-Inference baselines

Continual learning with EWC, model OTA, and extended final evaluation remain future
work only. They must not be presented as implemented contributions or as missing work
that makes the current project incomplete.

---

## Architecture Revision R1 — 2026-09-08

**Status:** Evidence-driven scope revision after completed Split-Inference and policy-action studies.

The original architecture required a learned split-point controller choosing among Split 1/2/3 after an OFFLOAD decision. The split-inference execution paths (not a learned Split Controller) were implemented and validated end-to-end during Phase 7, then evaluated during the subsequent policy-dataset and action-space studies.

Measured and controlled studies established that, for the current `features-v1` / `gesture-model-v1.1.0` design:

- `ALL_CLOUD` sends the 10-feature vector, while the validated split embeddings are 64, 48, and 32 float32 values.
- Therefore all current split embeddings are larger than the 10-feature cloud input before serialization overhead is considered.
- Split actions also add edge-prefix compute before communication and server-tail compute afterward.
- Matched-condition pilot measurements and controlled replay studies did not produce a state in which Split 1 or Split 2 was optimal.
- Reward-sensitivity analysis found both Split 1 and Split 2 dominated under the current measured/simulated action space; non-negative reward reweighting could not make either a unique winner.
- Candidate bottlenecks smaller than the current B3 boundary would require new model parameters and new accuracy/deployment validation, which would materially expand scope.

Per the source-of-truth priority in Section 39, measured project behavior has higher priority than the original architecture assumptions. Therefore the **production adaptive policy is revised to a binary learned decision: `LOCAL` vs `CLOUD`**.

The completed split-inference implementation is **not deleted**. Split 1/2/3 remain:

- validated research artifacts,
- fixed-split experimental baselines,
- evidence supporting the architecture revision,
- optional future work if a new compact bottleneck architecture is investigated.

For the undergraduate MVP, **adaptive split-point selection and the Split Controller are retired from the production decision path**. No further split-architecture optimization is required unless new measured evidence justifies reopening it.

Canonical production policy:

```text
LOCAL
or
CLOUD
```

Canonical server-heavy path:

```text
features-v1 (10 values)
→ MQTT
→ server full-cloud model
→ result
```

This revision narrows scope while preserving the main project contribution: **uncertainty-aware learned Edge–Cloud offloading on a resource-constrained TinyML device**.

---

## 1. Project Identity

### 1.1 Core idea
The project is **not primarily a gesture-recognition project**. Gesture recognition is the practical testbed used to evaluate the main research/engineering contribution:

> **A TinyML edge device that learns whether to finish inference locally or offload to a server based on uncertainty, network conditions, and device state.**

The system combines:

- TinyML inference on ESP32-S3
- IMU-based gesture recognition
- uncertainty estimation
- learned `LOCAL` vs `CLOUD` decision making
- MQTT communication
- server-side cloud inference
- failover to full local inference when Wi-Fi/server access is unavailable
- experiment logging
- dashboard visualization
- comparison against simpler baselines

Continual learning with EWC and model update through OTA are documented future work
outside the frozen undergraduate implementation scope.

Split inference was implemented and validated as an experimental branch of the project. However, measured and controlled action-space studies showed that the current 64/48/32-dimensional split embeddings are not competitive with directly offloading the 10-feature vector. Therefore adaptive split-point selection is no longer part of the production MVP policy.

### 1.2 Case-study framing
Gesture recognition remains the case study because it is:

- inexpensive to build
- easy to demo repeatedly
- sufficiently non-trivial for TinyML
- based on multivariate time-series patterns rather than a single threshold
- suitable for uncertainty estimation
- suitable for different users / distribution shift
- appropriate for local-vs-cloud adaptive inference
- compatible with continual learning and model personalization

The final presentation should frame the work as:

> **Adaptive Edge–Cloud TinyML architecture with a learned uncertainty-aware LOCAL/CLOUD policy, evaluated through gesture recognition.**

Avoid presenting the project merely as:

> “An intelligent gesture detector.”

Also avoid claiming that adaptive split selection is part of the final production policy. The split work should be presented as a validated experimental baseline and an evidence-driven negative result that motivated the binary production policy.

## 2. Research / Engineering Question

The revised main question is:

> Can a resource-constrained TinyML device learn, based on prediction uncertainty, network conditions, and device state, whether to finish inference locally or offload the feature vector to a server, while maintaining useful classification quality and reducing unnecessary communication/latency?

The original split-point question was investigated experimentally. For the current network architecture, the validated split actions were dominated by `LOCAL` and/or `CLOUD`, so adaptive split selection is retained as an evaluated baseline rather than a production requirement.

Secondary questions:

1. Does learned adaptive LOCAL/CLOUD inference reduce unnecessary communication compared with always-cloud inference?
2. Does it preserve useful classification quality compared with fully local inference?
3. Does the learned LOCAL/CLOUD policy outperform a simple rule-based adaptive baseline under the chosen cost function?
4. Can the system remain functional when Wi-Fi or the server is unavailable?
5. How does the validated fixed-split implementation compare with LOCAL/CLOUD, and why was it excluded from the final adaptive action space?
6. Can continual learning improve performance on new-user motion patterns without catastrophic forgetting?
7. Can new models be safely distributed to the device through OTA?

## 3. System-Level Architecture

### 3.1 Production adaptive path

```text
                       ┌──────────────────────┐
                       │      MPU6050         │
                       │ ax ay az gx gy gz    │
                       └──────────┬───────────┘
                                  │ 100 Hz
                                  ▼
                       ┌──────────────────────┐
                       │    Window Buffer     │
                       │ 1 s / 100 samples    │
                       │ 50% overlap runtime  │
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │  Feature Extractor   │
                       │     10 features      │
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ Local NN + MC path   │
                       │ prediction + UQ      │
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ Learned Meta Policy  │
                       │   LOCAL or CLOUD     │
                       └──────────┬───────────┘
                              ┌───┴────┐
                        LOCAL │        │ CLOUD
                              ▼        ▼
                        Edge Result   MQTT
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ FastAPI / Server │
                              │ full-cloud model │
                              └────────┬─────────┘
                                       ▼
                                  Cloud Result
                                       │
                   ┌───────────────────┼───────────────────┐
                   ▼                   ▼
              PostgreSQL          Dashboard
```

### 3.2 Experimental split branch

The validated Split 1/2/3 implementation remains in the repository for benchmarking and documentation:

```text
features-v1
→ prefix B1 / B1+B2 / B1+B2+B3
→ 64 / 48 / 32-D embedding
→ MQTT
→ matching server tail
→ result
```

This branch is **not selected by the final adaptive production policy** unless a future architecture revision explicitly reintroduces split selection.

## 4. Hardware Baseline

### 4.1 Main MCU
**Board / module:** ESP32-S3-WROOM-1-N8R2 development board

Target hardware characteristics:

- dual-core Xtensa LX7
- up to 240 MHz
- 512 KB internal SRAM
- 8 MB Flash
- 2 MB PSRAM
- Wi-Fi 2.4 GHz 802.11 b/g/n
- Bluetooth LE 5
- I2C support
- USB programming/debugging
- 3.3 V GPIO logic

### 4.2 IMU
**Sensor module:** MPU6050 GY-521

Relevant characteristics:

- 3-axis accelerometer
- 3-axis gyroscope
- 16-bit digital output
- I2C interface
- accelerometer ranges:
  - ±2 g
  - ±4 g
  - ±8 g
  - ±16 g
- gyroscope ranges:
  - ±250 °/s
  - ±500 °/s
  - ±1000 °/s
  - ±2000 °/s

### 4.3 Initial electrical connection
Planned connection:

```text
GY-521              ESP32-S3
--------------------------------
VCC       --------> 3.3V
GND       --------> GND
SDA       --------> GPIO8
SCL       --------> GPIO9
```

Important:

- GPIO8/GPIO9 are the **planned project pins**.
- Verify the exact dev-board silkscreen/pinout after the physical board arrives.
- Avoid using ESP32-S3 strapping pins for the IMU unless necessary:
  - GPIO0
  - GPIO3
  - GPIO45
  - GPIO46
- Avoid GPIO19/GPIO20 for the IMU because they may be used by USB functionality.
- The development board is powered through 5 V USB.
- The GY-521 should be powered from the board’s 3.3 V pin for clean 3.3 V logic compatibility.

### 4.4 Initial sensor configuration
Project default:

```text
I2C frequency          400 kHz
Sampling rate          100 Hz
Accelerometer range    ±4 g
Gyroscope range        ±500 °/s
```

These are initial engineering choices and may be changed only after actual data inspection.

### 4.5 Hardware not required initially
Do not expand hardware unless the project later proves it necessary.

Not currently required:

- display
- SD card
- battery-management module
- additional sensors
- Raspberry Pi
- external server
- microphone
- motor
- camera

The development laptop acts as the server during implementation and defense.

---

## 5. Gesture Recognition Scope

### 5.1 Version-1 gesture classes
Freeze the first version to exactly five classes:

| ID | Class | Description |
|---:|---|---|
| 0 | `IDLE` | device held still / no intended gesture |
| 1 | `SWIPE_LEFT` | clear leftward handheld movement |
| 2 | `SWIPE_RIGHT` | clear rightward handheld movement |
| 3 | `ROTATE_CW` | clockwise rotation |
| 4 | `SHAKE` | repeated short back-and-forth shaking |

Do **not** add more classes until the 5-class pipeline works end-to-end.

Possible later additions:

- `ROTATE_CCW`
- `SWIPE_UP`
- `SWIPE_DOWN`
- `IMPACT`
- `UNKNOWN`

These are optional extensions, not initial requirements.

### 5.2 Device orientation protocol
Dataset consistency requires a fixed handheld orientation.

Before collection begins, define a physical convention, for example:

- USB connectors toward the user
- board front side upward
- same grip orientation for all planned sessions

Document this orientation with a photo once hardware arrives.

---

## 6. Sampling, Windowing, and Runtime Timing

### 6.1 Raw sensor channels

Each sensor sample:

```text
timestamp_ms
ax
ay
az
gx
gy
gz
```

### 6.2 Sampling rate

```text
100 samples / second
```

Therefore:

```text
sample period = 10 ms
```

### 6.3 Window size

Initial window:

```text
1 second
100 IMU samples
6 channels
```

One raw window therefore contains:

```text
100 × 6 = 600 sensor values
```

### 6.4 Runtime overlap

For continuous inference:

```text
window length = 1.0 s
step size     = 0.5 s
overlap       = 50%
```

This allows approximately two gesture predictions per second after the initial buffer fills.

### 6.5 Timing requirements
Sampling must not be implemented by imprecise long blocking delays in the final firmware.

During early tests `delay()` may be temporarily acceptable, but the final sensor loop should use:

- timestamp scheduling, or
- ESP timer / FreeRTOS timing, or
- interrupt-based scheduling if later required

The system must log actual timestamps so real sample timing can be checked.

---

## 7. Calibration

At boot:

1. keep the board still for approximately 2 seconds
2. collect gyro samples
3. estimate gyro bias
4. subtract bias from future gyro readings

Optional later calibration:

- accelerometer offset correction
- per-axis scale correction

Calibration parameters should be stored in configuration and logged with datasets when relevant.

---

## 8. Dataset Design

### 8.1 Principle
Use **real IMU data** recorded from the project hardware.

Do not begin with synthetic gesture data as the main dataset.

### 8.2 Initial dataset target

Target:

```text
200 labeled windows per class
5 classes
----------------------
1000 labeled windows
```

This is an initial target, not a hard scientific requirement.

### 8.3 Session-based split
Do not randomly shuffle highly similar windows from one recording session across train/test.

Preferred split:

| Session | Purpose | Approx. samples/class |
|---|---|---:|
| A | Training | 120 |
| B | Validation | 40 |
| C | Test | 40 |

Thus each class initially contributes about 200 examples.

### 8.4 Raw directory structure

```text
data/
└── raw/
    └── user_01/
        ├── session_01/
        │   ├── idle_001.csv
        │   ├── idle_002.csv
        │   ├── swipe_left_001.csv
        │   ├── swipe_right_001.csv
        │   ├── rotate_cw_001.csv
        │   └── shake_001.csv
        ├── session_02/
        └── session_03/
```

### 8.5 Raw CSV format

```csv
timestamp_ms,ax,ay,az,gx,gy,gz
0,...
10,...
20,...
...
990,...
```

### 8.6 Metadata example

```json
{
  "gesture": "SWIPE_LEFT",
  "user": "user_01",
  "session": "session_01",
  "sample_rate_hz": 100,
  "window_ms": 1000,
  "accel_range_g": 4,
  "gyro_range_dps": 500,
  "firmware_version": "0.1.0",
  "notes": ""
}
```

### 8.7 Dataset versioning

Examples:

```text
dataset-v1
dataset-v1.1
dataset-v2
```

A new dataset version is required when:

- gesture definition changes
- orientation convention changes
- sampling rate changes
- major preprocessing changes
- new users are added for continual-learning experiments

---

## 9. Feature Extraction

### 9.1 Requirement
The edge device should convert each raw sensor window into a small feature vector.

Project target:

```text
600 raw values
      ↓
10 meaningful features
```

### 9.2 Initial `features-v1`
Start with:

| # | Feature |
|---:|---|
| 1 | `std(ax)` |
| 2 | `max(abs(ax))` |
| 3 | `mean(ax_first_half) - mean(ax_second_half)` |
| 4 | `std(ay)` |
| 5 | `std(az)` |
| 6 | RMS deviation of acceleration magnitude |
| 7 | `mean(gz)` |
| 8 | `std(gz)` |
| 9 | RMS gyroscope magnitude |
| 10 | maximum gyroscope magnitude |

Definitions:

```text
acc_mag = sqrt(ax² + ay² + az²)
gyro_mag = sqrt(gx² + gy² + gz²)
```

### 9.3 Feature parity requirement
The Python feature extractor and ESP32 feature extractor must produce equivalent values.

Create automated parity tests:

```text
same raw window
    ↓
Python feature vector
ESP32 feature vector
    ↓
compare within tolerance
```

No ML result is trusted until feature parity is verified.

### 9.4 Feature changes
`features-v1` is a starting point.

Features may be modified after exploratory data analysis if:

- two gestures are poorly separated
- a feature carries almost no information
- a feature is too expensive on ESP32
- the left/right direction feature is unstable

Any change creates a new feature version.

---

## 10. Base Neural-Network Architecture

### 10.1 Why not a trivial one-head model?
If edge and cloud run identical classifiers, cloud execution may have little accuracy advantage while adding network delay.

Therefore the architecture should explicitly support:

- a lighter local exit
- a deeper server-side continuation

### 10.2 Proposed network

```text
Input(10)
   ↓
Block 1
Dense(64)
ReLU
   ↓
Split Point 1
   ↓
Block 2
Dense(48)
ReLU
   ↓
Split Point 2
   ↓
Block 3
Dense(32)
ReLU
   ↓
Split Point 3
   │
   ├───────────────> Edge Exit Head
   │                    ↓
   │                  5 classes
   │
   ↓
Cloud Block 4
Dense(64)
ReLU
   ↓
Cloud Block 5
Dense(32)
ReLU
   ↓
Cloud Head
   ↓
5 classes
```

This is the initial architecture and may be tuned after real data becomes available.

---

## 11. Split Inference — Validated Experimental Baseline

### 11.1 Validated split points

| Split | ESP32 executes | Server executes | Embedding dimension |
|---|---|---|---:|
| 1 | B1 | B2+B3+B4+B5+Cloud Head | 64 |
| 2 | B1+B2 | B3+B4+B5+Cloud Head | 48 |
| 3 | B1+B2+B3 | B4+B5+Cloud Head | 32 |

All three split paths were implemented and validated end-to-end during Phase 7.

### 11.2 Observed architectural limitation

The production feature representation contains only 10 values. Under the current float32 representation:

```text
ALL_CLOUD input: 10 values
Split 1:         64 values
Split 2:         48 values
Split 3:         32 values
```

Therefore the existing split embeddings are larger than the direct feature-vector payload, while also requiring edge-prefix compute before transmission.

Subsequent matched-condition, controlled-replay, reward-sensitivity, state-expansion, and candidate-architecture studies did not establish a condition in which the current Split 1 or Split 2 action became optimal. Payload reduction alone also did not justify a new bottleneck without retraining and revalidating the model.

### 11.3 Revised role of split inference

For the undergraduate MVP:

- Split 1/2/3 remain implemented and testable.
- Fixed-split execution remains an experimental baseline.
- Split results are retained as evidence for the design decision.
- The learned production policy does not choose a split point.
- No Split Controller is deployed in the production decision path.

A future model version may revisit split inference using a deliberately compact bottleneck if there is sufficient time and measured benefit.

## 12. TFLite / TinyML Deployment

### 12.1 Edge components
Expected on-device ML components:

- validated complete local inference (B3 prefix + Edge Exit Head)
- uncertainty mechanism
- learned binary Meta Policy: 0 LOCAL / 1 CLOUD
- features-v1 MQTT client for full-cloud inference

Split prefixes remain isolated fixed-split experimental baselines. The Split Controller is retired from production.

### 12.2 Quantization
Prefer int8 quantization for deployment once the float baseline is validated.

Suggested workflow:

```text
float Keras model
→ evaluate
→ export TFLite float
→ verify parity
→ int8 quantization
→ evaluate accuracy delta
→ deploy
```

### 12.3 Model parity
For fixed test vectors, compare:

```text
Python/Keras output
TFLite desktop output
ESP32 output
```

Set acceptable numeric tolerance based on quantization.

---

## 13. Uncertainty Estimation

### 13.1 Project approach
Use an MC-Dropout-style uncertainty mechanism with five stochastic passes.

```text
Pass 1
Pass 2
Pass 3
Pass 4
Pass 5
```

### 13.2 Deployment caveat
Standard TFLite inference normally disables ordinary training-time Dropout.

Therefore the implementation should not blindly assume a Keras Dropout layer will remain stochastic on ESP32.

Planned approach:

- train with dropout
- reproduce stochastic masking explicitly in the edge uncertainty path if required
- run the classifier 5 times
- aggregate predictions

### 13.3 Outputs

Compute:

- mean class probability
- predictive entropy
- variance / disagreement
- maximum mean confidence

Example:

```text
mean:
IDLE         0.02
SWIPE_LEFT   0.48
SWIPE_RIGHT  0.42
ROTATE_CW    0.05
SHAKE        0.03

confidence  = 0.48
uncertainty = high
```

Such a sample is a good candidate for offloading.

---

## 14. Learned Meta Policy — LOCAL vs CLOUD

### 14.1 Responsibility

The production learned policy answers one question:

```text
LOCAL
or
CLOUD
```

`LOCAL` means the ESP32 completes inference using the validated local path.

`CLOUD` means the ESP32 sends the current `features-v1` vector to the server, where the full-cloud inference path returns the result.

### 14.2 Candidate inputs

The exact versioned training feature contract must be taken from the policy configuration artifact (for example `ml/policy/policy_config_v2.json`).

Candidate state information includes:

```text
uncertainty / entropy
confidence
prediction margin
RSSI
estimated network RTT
free heap / resource state
connection / cloud availability
controlled device/network pressure features when explicitly marked
energy budget / energy proxy
```

Measured and simulated fields must retain explicit provenance.

### 14.3 Prototype energy constraint

The current hardware setup has no battery/energy measurement hardware.

Therefore energy-related policy inputs and reward components are **estimated/simulated proxies**, not measured battery percentage or measured Joules.

This distinction is mandatory in code, datasets, reports, and the defense.

### 14.4 Initial deployment architecture

Keep the policy deliberately small. A compact binary classifier is preferred, for example:

```text
State vector
   ↓
Dense(8)
ReLU
   ↓
Dense(4)
ReLU
   ↓
Output(2)
LOCAL / CLOUD
```

The exact architecture may be adjusted based on the qualified training dataset, but it must remain small enough for ESP32 deployment and easy to defend.

## 15. Split Controller — Retired from Production MVP

The original architecture proposed a second learned controller that selected:

```text
Split 1
Split 2
Split 3
```

after an `OFFLOAD` decision.

That component is **retired from the production MVP** because completed action-space studies showed the current split actions to be dominated by `LOCAL` and/or direct `CLOUD` execution.

Rules:

- Do not train or deploy a Split Controller for the current MVP.
- Keep existing Split 1/2/3 artifacts and tests as experimental baselines.
- Do not delete validated split code merely because it is no longer in the production policy.
- Reopen split-controller work only if a new compact bottleneck architecture is implemented and measured evidence shows a competitive split action.

This is an evidence-driven architecture revision, not an implementation failure.

## 16. Training the Adaptive Policy

### 16.1 Final policy must be learned

The proposed production method must use a learned binary policy.

A rule-based LOCAL/CLOUD method is allowed only as a baseline.

### 16.2 Policy-training data

For many combinations of:

```text
gesture sample
uncertainty
confidence
RSSI / RTT
resource state
cloud availability
energy proxy
```

evaluate the two production candidate actions:

```text
Action 0 = LOCAL
Action 1 = CLOUD
```

The historical four-action datasets (`LOCAL`, `Split1`, `Split2`, `ALL_CLOUD`) remain valid research evidence but must not be silently relabeled as binary training data. If reused, conversion/provenance must be explicit and versioned.

For each production action record:

- classification correctness / error
- end-to-end latency
- bytes transmitted
- bytes received
- estimated energy / computation proxy
- failure status
- measurement provenance

### 16.3 Cost / reward function

Use the versioned reward engine and configuration.

Conceptually:

```text
J =
w_error * classification_error
+
w_latency * normalized_latency
+
w_comm * normalized_communication
+
w_energy * normalized_energy_proxy
```

or the equivalent reward-maximization form.

Weights and normalization scales must be versioned experimental parameters, not hidden constants.

### 16.4 Label generation

For matched or explicitly controlled state conditions:

```text
state_i
    ↓
evaluate LOCAL and CLOUD
    ↓
compute comparable reward/cost
    ↓
best binary action label
```

Do not force class balance by falsifying labels.

Controlled/simulated network or device perturbations may be used for training-data diversity only when their provenance is explicit. Final claims must distinguish real measurements from controlled simulation.

### 16.5 Training gate

Binary policy training may begin when:

- both LOCAL and CLOUD labels are present,
- schema/version checks pass,
- reward configuration is frozen for the experiment,
- provenance is retained,
- an evaluation/holdout strategy is defined,
- no fabricated metrics are introduced.

The training report must state any remaining limitation in real-condition diversity.

## 17. Rule-Based Baseline

Implement a simple binary policy baseline such as:

```text
if Wi-Fi/server unavailable:
    LOCAL
else if uncertainty > threshold and network_quality is acceptable:
    CLOUD
else:
    LOCAL
```

This baseline is essential for answering:

> “Why not just use if/else?”

The final adaptive-policy comparison should include:

```text
Rule-Based LOCAL/CLOUD
vs
Learned LOCAL/CLOUD
```

Fixed Split 1/2/3 strategies remain separate experimental baselines; they are not part of the rule-based production policy.

## 18. Wi-Fi and MQTT Communication

### 18.1 Broker
Use:

```text
Mosquitto MQTT
```

during local development.

### 18.2 Topic scheme

```text
gesture/{device_id}/telemetry
gesture/{device_id}/inference/request
gesture/{device_id}/inference/response
gesture/{device_id}/status
gesture/{device_id}/model/update
```

### 18.3 Production R1 inference request example

This is the versioned production contract to implement after policy gates pass.
Values below are illustrative, not measured evidence. `features` contains exactly
10 normalized features-v1 values using the gesture-model-v1.1.0 scaler.

```json
{
  "schema_version": "inference-r1-v1",
  "request_id": "abc123",
  "device_id": "esp32-01",
  "timestamp_ms": 123456,
  "mode": "CLOUD",
  "features": [0.1, -0.2, 0.4, 0.0, 0.1, 0.2, -0.1, 0.3, 0.0, 0.5],
  "feature_version": "features-v1",
  "feature_encoding": "normalized-float32",
  "model_version": "gesture-full-cloud-v1.0.0",
  "policy_version": "meta-policy-v1.0.0",
  "firmware_version": "example-not-deployed"
}
```

### 18.4 Production R1 inference response example

```json
{
  "schema_version": "inference-r1-v1",
  "request_id": "abc123",
  "mode": "CLOUD",
  "predicted_class": "SWIPE_RIGHT",
  "confidence": 0.94,
  "server_latency_ms": 3.8,
  "model_version": "gesture-full-cloud-v1.0.0",
  "policy_version": "meta-policy-v1.0.0"
}
```

### 18.5 Serialization strategy

- Production R1: JSON with exactly 10 normalized features-v1 values.
- Retained fixed-split regression requests: separately identifiable `split` and
  `embedding` (64/48/32 values); these are not production CLOUD requests.
- No raw 100x6 IMU payload in normal CLOUD inference.
- Binary serialization is future work; no new quantization is required.

---

## 19. Network Quality Measurement

The adaptive system needs actual network-state inputs.

Collect:

- Wi-Fi RSSI
- MQTT / application-level round-trip time
- reconnect count
- timeout count

RTT can be measured using a lightweight ping/request-response message or inference request timestamps.

Do not use only RSSI as a proxy for all network quality.

---

## 20. Failover

### 20.1 Requirement
The system must remain operational without Wi-Fi.

```text
Wi-Fi connected
      ↓
adaptive mode

Wi-Fi disconnected / server timeout
      ↓
force full local inference
      ↓
B1 → B2 → B3 → Edge Head
```

### 20.2 No dropped decision
A network failure must not make the system incapable of producing a gesture result.

### 20.3 Log failover
Each fallback event should be logged with:

- timestamp
- reason
- previous intended action
- local fallback latency
- result

---

## 21. Server Architecture

### 21.1 Stack

```text
Python
FastAPI
Paho MQTT
TensorFlow / TFLite utilities
SQLAlchemy
PostgreSQL
Uvicorn
```

### 21.2 Production responsibilities

```text
MQTT Subscriber
      ↓
Request Validation
      ↓
LOCAL/CLOUD protocol handling
      ↓
Full-cloud model for CLOUD action
      ↓
Prediction
      ↓
MQTT Response
      ↓
Database Logging
      ↓
WebSocket / Dashboard
```

### 21.3 Retained split support

The server may retain the already validated Split-1 / Split-2 / Split-3 tail routing for:

- fixed-split benchmarks,
- regression testing,
- reproducibility of Phase-7 evidence.

These routes are not selected by the final learned production policy.

### 21.4 Server input contract

Normal `CLOUD` inference operates on the 10-value `features-v1` representation rather than raw sensor windows.

Raw sensor data may still be uploaded separately for:

- debugging
- labeled-sample collection
- continual-learning datasets

Keep those paths conceptually separate.

## 22. Database Design

### 22.1 `inference_events`

Suggested fields:

```text
id
timestamp
device_id
request_id

predicted_class
true_label
confidence
uncertainty

execution_mode
split_point (nullable; fixed-split experiments only)

rssi
rtt_ms
free_heap_bytes
free_heap_ratio
energy_budget

edge_compute_ms
network_ms
server_compute_ms
total_latency_ms

bytes_tx
bytes_rx

model_version
policy_version
firmware_version

success
failure_reason
```

### 22.2 `devices`

```text
device_id
hardware_revision
firmware_version
active_model_version
last_seen
notes
```

### 22.3 `model_versions`

```text
model_version
created_at
dataset_version
feature_version
metrics_json
artifact_path
sha256
active
```

### 22.4 `labeled_samples`

```text
sample_id
timestamp
device_id
raw_data_path
predicted_label
true_label
confirmed_by
model_version
used_in_training
```

### 22.5 `training_runs`

```text
training_run_id
started_at
finished_at
base_model_version
new_model_version
dataset_version
method
metrics_before_json
metrics_after_json
notes
```

---

## 23. Dashboard

### 23.1 Keep the UI simple
Use:

```text
FastAPI
HTML
JavaScript
Chart.js
WebSocket
```

Do not introduce React unless there is a clear need.

### 23.2 Live panel

Example:

```text
Detected Gesture: SWIPE_LEFT
Confidence:       96%
Uncertainty:      0.08
Execution:        LOCAL
Policy:           example-not-deployed
RSSI:             -62 dBm
RTT:              19 ms
Total Latency:    5.1 ms
Model:            v1.0.0
```

### 23.3 Charts

Required / useful:

- local exit rate
- cloud-offload rate
- LOCAL/CLOUD action distribution (fixed-split distribution only in experimental views)
- latency over time
- mean latency by strategy
- P95 latency by strategy
- bytes transferred
- uncertainty distribution
- confidence / calibration
- classification accuracy
- confusion matrix
- continual-learning history
- model version history

### 23.4 Sensor-driven 3D device twin

Phase 11 may use a separate auxiliary visualization stream for the 3D device twin:

```text
gesture/{device_id}/pose
pose-v1
```

This stream is **not** part of the learned policy input contract and does not change the
R1 `LOCAL`/`CLOUD` production action space. It must reuse the already-acquired IMU
samples and must not transmit raw 100x6 windows, split embeddings, or features-v1.

For the current MPU6050-only hardware, dashboard attitude values must be labeled as
estimates. Roll/pitch may be gravity-corrected with a lightweight complementary filter.
Without a magnetometer, yaw must be reported only as a boot-relative/drift-prone
estimate and must not be described as absolute heading.

---

## 24. Continual Learning with EWC

### 24.1 Motivation
Gesture style changes across users.

Example:

```text
User A
→ initial model

User B
→ different gesture dynamics
→ distribution shift
```

This gives a natural continual-learning scenario.

### 24.2 Experiment
Compare:

```text
Model v1 before adaptation
Naive fine-tuning
EWC fine-tuning
```

Measure performance on:

- old-user test set
- new-user test set

Goal:

- naive fine-tuning may improve new data but forget old data
- EWC should reduce catastrophic forgetting

### 24.3 Report table

```text
Model          Old-data Acc    New-data Acc
------------------------------------------------
v1             ...
Naive FT       ...
EWC            ...
```

Never invent numbers. Use real measured results only.

---

## 25. OTA Model Update

### 25.1 Initial scope
Prefer OTA for **model artifacts** rather than unnecessarily implementing full firmware OTA first.

### 25.2 Manifest example

```json
{
  "version": "1.1.0",
  "feature_version": "features-v1",
  "files": {
    "model": "gesture_model_v1.1.0.tflite",
    "meta": "meta_policy_v1.1.0.tflite"
  },
  "sha256": {
    "model": "...",
    "meta": "..."
  }
}
```

### 25.3 OTA flow

```text
Server publishes model-update metadata
      ↓
ESP32 checks version
      ↓
HTTP download
      ↓
SHA-256 validation
      ↓
store new artifact
      ↓
load / activate
      ↓
health check
      ↓
mark successful
```

### 25.4 Rollback
If verification or activation fails:

```text
keep previous model
```

Never overwrite the only working model before integrity is verified.

---

## 26. Evaluation Modes

The final system should compare at least:

1. **All Local**
2. **All Cloud**
3. **Fixed Split** — one or more validated split baselines
4. **Rule-Based Adaptive LOCAL/CLOUD**
5. **Learned Adaptive LOCAL/CLOUD** — proposed production system

### 26.1 All Local

```text
always local B1 → B2 → B3 → Edge Head
```

### 26.2 All Cloud

```text
features-v1 (10 values)
→ network
→ server full-cloud model
```

### 26.3 Fixed Split

Use the already validated split implementation as a baseline, for example:

```text
always Split 2
```

This mode is retained to demonstrate why split selection was not included in the final adaptive action space.

### 26.4 Rule-Based Adaptive

Use binary threshold logic:

```text
LOCAL or CLOUD
```

### 26.5 Learned Adaptive

Use the learned binary Meta Policy:

```text
state
→ learned policy
→ LOCAL or CLOUD
```

The production system does not require a learned Split Controller.

## 27. Evaluation Metrics

Collect:

### Classification
- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix

### Latency
- mean latency
- median latency
- P95 latency
- local compute latency
- network latency
- server latency
- total end-to-end latency

### Communication
- bytes transmitted/sample
- bytes received/sample
- total traffic
- cloud-offload percentage

### Adaptive behavior
- LOCAL rate
- CLOUD rate
- policy action distribution
- policy disagreement vs rule-based baseline

For fixed-split baseline experiments, also report split-specific payload/latency separately.

### Reliability
- success under Wi-Fi loss
- server timeout handling
- fallback latency

### Uncertainty
- entropy distribution
- confidence distribution
- prediction margin
- calibration curve
- expected calibration error if implemented

### Resource behavior
- free heap
- model size
- tensor arena size
- Flash usage
- PSRAM usage
- approximate compute / energy proxy

Never present estimated energy as a physical measurement.

## 28. Experiment Scenarios

### Scenario A — Clear gesture, normal network

Expected tendency:

```text
high confidence
low uncertainty
→ likely LOCAL
```

### Scenario B — Ambiguous gesture, cloud available

```text
higher uncertainty
acceptable network state
→ CLOUD may be selected
```

### Scenario C — Ambiguous gesture, poor network

```text
high uncertainty
high network cost
→ policy may still prefer LOCAL
```

### Scenario D — Wi-Fi/server unavailable

```text
→ force LOCAL fallback
```

### Scenario E — Controlled network/device perturbation

```text
explicitly marked controlled/simulated state
→ evaluate whether learned LOCAL/CLOUD policy responds appropriately
```

Such data must not be described as an unmodified real-world network measurement.

### Scenario F — Different user

```text
performance shift
→ collect labels
→ continual learning
→ compare naive FT vs EWC
```

### Scenario G — Model update

```text
server model N+1
→ OTA
→ ESP32 verifies and activates N+1
```

Fixed-split scenarios may be replayed as benchmark evidence, not as the production adaptive path.

## 29. Repository Structure

```text
adaptive-edge-cloud-gesture/
│
├── README.md
├── .gitignore
├── requirements.txt
├── docker-compose.yml
│
├── config/
│   ├── gestures.yaml
│   ├── hardware.yaml
│   ├── model.yaml
│   └── experiment.yaml
│
├── docs/
│   ├── PROJECT_ARCHITECTURE.md
│   ├── architecture.md
│   ├── hardware.md
│   ├── dataset_protocol.md
│   ├── mqtt_protocol.md
│   ├── experiments.md
│   └── diagrams/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── metadata/
│
├── ml/
│   ├── dataset/
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── split_dataset.py
│   │
│   ├── features/
│   │   ├── extractor.py
│   │   └── features_v1.py
│   │
│   ├── models/
│   │   ├── base_model.py
│   │   ├── edge_exit.py
│   │   ├── cloud_model.py
│   │   └── split_models.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   │
│   ├── uncertainty/
│   │   ├── mc_dropout.py
│   │   └── calibration.py
│   │
│   ├── policy/
│   │   ├── build_policy_dataset.py
│   │   ├── train_meta.py
│   │   └── train_split_controller.py  # retired historical placeholder; not production
│   │
│   ├── continual/
│   │   ├── ewc.py
│   │   └── continual_train.py
│   │
│   └── export/
│       ├── tflite_export.py
│       ├── quantize.py
│       └── manifest.py
│
├── firmware/
│   ├── platformio.ini
│   ├── include/
│   │   ├── config.h
│   │   ├── pins.h
│   │   └── version.h
│   │
│   └── src/
│       ├── main.cpp
│       │
│       ├── sensors/
│       │   └── mpu6050.cpp
│       │
│       ├── sampling/
│       │   └── window_buffer.cpp
│       │
│       ├── features/
│       │   └── feature_extractor.cpp
│       │
│       ├── inference/
│       │   ├── prefix_runner.cpp
│       │   ├── edge_head.cpp
│       │   └── uncertainty.cpp
│       │
│       ├── policy/
│       │   ├── meta_learner.cpp
│       │   └── split_controller.cpp  # retired historical placeholder; not production
│       │
│       ├── network/
│       │   ├── wifi_manager.cpp
│       │   └── mqtt_client.cpp
│       │
│       ├── ota/
│       │   └── model_updater.cpp
│       │
│       └── diagnostics/
│           └── metrics.cpp
│
├── collector/
│   ├── serial_collector.py
│   ├── record_gesture.py
│   └── validate_capture.py
│
├── server/
│   ├── app/
│   │   ├── main.py
│   │   ├── mqtt.py
│   │   ├── inference.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   └── ota.py
│   │
│   └── model_registry/
│
├── dashboard/
│   ├── templates/
│   └── static/
│
├── experiments/
│   ├── all_local/
│   ├── all_cloud/
│   ├── fixed_split/
│   ├── rule_based/
│   └── adaptive/
│
└── tests/
    ├── test_features.py
    ├── test_dataset.py
    ├── test_model_parity.py
    ├── test_mqtt.py
    └── test_server.py
```

---

## 30. Configuration Files

### 30.1 `config/gestures.yaml`

Example:

```yaml
gestures:
  - id: 0
    name: IDLE
  - id: 1
    name: SWIPE_LEFT
  - id: 2
    name: SWIPE_RIGHT
  - id: 3
    name: ROTATE_CW
  - id: 4
    name: SHAKE
```

### 30.2 `config/hardware.yaml`

```yaml
board:
  family: ESP32-S3
  module: ESP32-S3-WROOM-1-N8R2
  flash_mb: 8
  psram_mb: 2

imu:
  model: MPU6050_GY521
  i2c_frequency_hz: 400000
  sampling_rate_hz: 100
  accel_range_g: 4
  gyro_range_dps: 500

pins:
  sda: 8
  scl: 9
```

### 30.3 `config/model.yaml`

```yaml
input_features: 10

blocks:
  b1: 64
  b2: 48
  b3: 32
  cloud_b4: 64
  cloud_b5: 32

classes: 5

mc_dropout:
  passes: 5
  dropout_rate: 0.2
```

### 30.4 `config/experiment.yaml`

```yaml
cost_weights:
  error: 0.50
  latency: 0.25
  communication: 0.15
  energy: 0.10

runtime:
  window_ms: 1000
  step_ms: 500
```

---

## 31. Versioning Rules

Track:

```text
dataset version
feature version
model version
policy version
firmware version
server version
```

Examples:

```text
dataset-v1
features-v1
gesture-model-v1.0.0
meta-policy-v1.0.0
split-policy-v1.0.0 (historical naming only; no production Split Controller)
firmware-v0.1.0
server-v0.1.0
```

Every inference record should include model/policy/firmware versions where practical.

---

## 32. Tests

### 32.1 Dataset validation
Check:

- exact column names
- no NaN
- timestamps monotonic
- approximate 100 Hz timing
- expected sample count
- class label valid

### 32.2 Feature tests
Check:

- no divide-by-zero
- no NaN/inf
- Python/C++ parity
- correct feature order

### 32.3 Model parity
Check the same fixed vector across:

- Keras
- desktop TFLite
- ESP32

### 32.4 MQTT tests
Check:

- request schema
- response schema
- timeout
- duplicate request ID
- malformed payload
- reconnect

### 32.5 OTA tests
Check:

- valid manifest
- bad hash
- interrupted download
- rollback

---

## 33. Implementation Phases

### Phase 0 — Project skeleton
Complete before hardware is required.

Definition of done:

```text
repository exists
configs parse correctly
basic Python tests run
```

### Phase 1 — Hardware bring-up

Definition of done:

```text
stable ax ay az gx gy gz stream
```

### Phase 2 — Data collection

Definition of done:

```text
dataset-v1 complete
train / validation / test sessions separated
```

### Phase 3 — Base ML model

Definition of done:

```text
acceptable validation and held-out test performance
```

### Phase 4 — Local TinyML

Definition of done:

```text
gesture recognized locally on ESP32
```

**Hard rule:** adaptive cloud work begins only after this phase is stable.

### Phase 5 — Uncertainty

Definition of done:

```text
uncertainty score available per inference
```

### Phase 6 — Server + MQTT

Definition of done:

```text
ESP32 sends request
server finishes inference
ESP32 receives result
```

### Phase 7 — Split inference feasibility / baseline

Tasks:

- export Split 1/2/3 artifacts
- validate Python/TFLite/ESP32 parity
- validate MQTT E2E for all split paths
- record payload/latency behavior

Definition of done:

```text
all three split points produce correct end-to-end predictions
```

**Status:** completed. These split paths are retained as baselines; they are no longer production adaptive actions.

### Phase 8 — Action-space benchmarking and policy-data infrastructure

Tasks:

- benchmark LOCAL / CLOUD / fixed split actions
- vary network/state conditions
- build versioned policy datasets
- calibrate reward/cost infrastructure
- analyze action dominance

Definition of done:

```text
action-space evidence and policy-data infrastructure available
```

**Architecture revision trigger:** completed studies showed current split actions dominated; production action space narrowed to `LOCAL` vs `CLOUD`.

### Phase 9 — Learned binary adaptive policy

Tasks:

- freeze binary policy dataset/configuration
- train compact learned Meta Policy
- implement rule-based binary baseline
- evaluate learned vs rule-based behavior
- export/deploy policy to ESP32 if deployment artifact passes parity/resource checks
- log LOCAL/CLOUD decisions

Definition of done:

```text
device adaptively selects LOCAL or CLOUD using a learned policy
```

No learned Split Controller is required for the current MVP.

### Phase 10 — Failover

Tasks:

- disconnect Wi-Fi
- simulate server timeout
- force LOCAL
- log transition

Definition of done:

```text
system continues gesture recognition without network
```

### Phase 11 — Database + Dashboard

Definition of done:

```text
live dashboard reflects device decisions
```

### Phase 12 — Continual learning

**Status:** future work; outside the frozen undergraduate implementation scope.

Definition of done:

```text
measured continual-learning experiment completed
```

### Phase 13 — OTA

**Status:** future work; outside the frozen undergraduate implementation scope.

Definition of done:

```text
ESP32 moves from model version N to N+1 safely
```

### Phase 14 — Final evaluation

**Status:** future work; outside the frozen undergraduate implementation scope.

Compare:

```text
All Local
All Cloud
Fixed Split baseline
Rule-Based Adaptive LOCAL/CLOUD
Learned Adaptive LOCAL/CLOUD
```

Generate final measured tables and plots.

## 34. Milestones

```text
M1  ESP32 boots and reports correctly
M2  MPU6050 raw stream works
M3  dataset-v1 collected
M4  Python gesture classifier works
M5  ESP32 local gesture classification works
M6  uncertainty works
M7  server-assisted inference works
M8  all split points work as validated experimental baselines
M9  learned adaptive LOCAL/CLOUD policy works
M10 failover works
M11 dashboard works
M12 continual-learning experiment works
M13 OTA works
M14 final benchmark completed
```

### Policy-study sub-milestones

The M25–M34 labels used in implementation notes are analysis sub-milestones inside the policy-development work. They do not renumber the canonical Phase 9/10 sequence above.

The key conclusion from those studies is:

```text
Current adaptive production action space:
LOCAL
CLOUD
```

Split 1/2/3 remain fixed experimental baselines.

## 35. Scope-Control Rules

These rules are mandatory unless there is a strong technical reason to change them.

1. Do not add new sensors before the base project works.
2. Do not add more gesture classes before the 5-class model works.
3. Do not build cloud/adaptive components before local TinyML is stable.
4. Do not optimize binary MQTT payloads before JSON works.
5. Do not build a complex frontend.
6. Do not introduce reinforcement learning unless the supervised policy method clearly fails.
7. Do not claim battery/energy measurements without measurement hardware.
8. Do not invent experiment results.
9. Do not change feature definitions without versioning them.
10. Do not change sampling settings without creating a new dataset version.
11. Every new feature must justify its implementation cost.
12. Prefer a working end-to-end system over a larger unfinished system.
13. Do not force Split 1/2/3 labels merely to obtain action balance.
14. Do not reopen split-architecture optimization in the MVP unless new measured evidence shows a credible benefit.
15. Preserve the completed split implementation as baseline evidence rather than deleting it.
16. The learned production policy is binary `LOCAL`/`CLOUD` unless a later documented architecture revision changes this decision.

## 36. Known Risks

### Risk 1 — MC Dropout on TFLite Micro
Ordinary Dropout may be disabled at inference.

Mitigation:

- implement explicit stochastic masking if needed
- verify actual variation across the 5 passes

### Risk 2 — Split inference is not economically competitive
The validated split paths increase representation size relative to the 10-feature all-cloud input and add prefix compute.

Measured/controlled project studies found the current split actions dominated.

Mitigation:

- keep split paths as fixed experimental baselines
- use binary LOCAL/CLOUD adaptive policy for the MVP
- revisit compact bottlenecks only as future work

### Risk 3 — Dataset leakage
Random windows from one gesture sequence may leak across train/test.

Mitigation:

- session-based split

### Risk 4 — Gesture inconsistency
Different grip/orientation may hurt accuracy.

Mitigation:

- freeze physical orientation protocol for v1

### Risk 5 — Adaptive policy collapses to one action
A learned policy is not useful if the training/evaluation state space always favors one action.

Mitigation:

- preserve matched-condition measurements
- use explicitly controlled state perturbations where appropriate
- keep provenance
- report limitations honestly
- never force labels

### Risk 6 — Simulated-vs-measured state confusion
Controlled perturbation is useful for policy development but can be mistaken for real-world measurement.

Mitigation:

- retain explicit provenance fields
- separate measured and simulated analyses
- make final claims only at the level supported by evidence

### Risk 7 — Project scope
EWC + OTA + dashboard + policy deployment can expand rapidly.

Mitigation:

- follow phase gates
- stop further split optimization for the MVP
- prioritize a working binary adaptive system and failover before late-stage polish

## 37. Final Defense Story

The final demo should show the **adaptive Edge–Cloud decision**, not merely gesture classification.

### Demo moment 1 — confident local result

```text
clear gesture
→ confidence high
→ uncertainty low
→ learned policy selects LOCAL
→ no cloud request
```

### Demo moment 2 — uncertain + acceptable network

```text
ambiguous movement
→ uncertainty higher
→ learned policy may select CLOUD
→ 10-feature vector sent
→ server result returned
```

### Demo moment 3 — network/state change

```text
network or resource state changes
→ LOCAL/CLOUD decision changes
```

### Demo moment 4 — Wi-Fi/server loss

```text
network unavailable
→ force LOCAL fallback
→ gesture recognition continues
```

### Demo moment 5 — explain the split experiment

Show that Split 1/2/3 were actually implemented and validated, then present the measured design finding:

```text
current split embeddings > 10-feature cloud input
+
split prefix compute
→ split actions dominated in current action-space studies
→ production policy simplified to LOCAL/CLOUD
```

This is an evidence-driven engineering decision and should be presented as such.

### Future-work illustration — model evolution (not part of the defense demo)

```text
new-user labeled data
→ continual learning / EWC
→ model N+1
→ OTA
→ device activates verified update
```

## 38. Expected Final Report Comparisons

Suggested central result table:

| Strategy | Accuracy | Macro F1 | Mean Latency | P95 Latency | Bytes/Sample | Local Rate | Failure Resilience |
|---|---:|---:|---:|---:|---:|---:|---:|
| All Local | | | | | | | |
| All Cloud | | | | | | | |
| Fixed Split baseline | | | | | | | |
| Rule-Based LOCAL/CLOUD | | | | | | | |
| Learned LOCAL/CLOUD | | | | | | | |

Do not require the proposed method to be best in every column.

A reasonable final claim is:

> **The learned uncertainty-aware LOCAL/CLOUD policy provides an evidence-based trade-off among classification quality, latency, communication cost, and resilience, while the validated fixed-split experiments explain why adaptive split selection was excluded from the final MVP.**

Do not claim that the learned policy optimizes split points.

## 39. Source-of-Truth Priority

When project decisions conflict, use this priority:

1. **Physical hardware behavior and measured experiments**
2. **Official vendor datasheets**
3. **This architecture document**
4. **Repository configuration files**
5. **Implementation notes / comments**
6. **Old conversation assumptions**

If this document is changed, record the reason.

### Current recorded architecture changes

Revision R1 narrows the production adaptive action space from:

```text
LOCAL / Split1 / Split2 / Split3 (or LOCAL + learned split selection)
```

to:

```text
LOCAL / CLOUD
```

Reason:

- the split implementation was completed successfully,
- subsequent policy-action studies found current split actions dominated,
- current split embeddings are larger than the direct 10-feature cloud input,
- further compact-bottleneck redesign would require retraining/revalidation and expand project scope.

This revision follows Priority 1: measured project behavior overrides the earlier architectural expectation.

Revision R2 freezes the completed undergraduate implementation scope at Phase 11.
Phases 12–14 remain documented future work. This is a scope-status correction only;
it does not alter measured behavior or the R1 binary production architecture.

## 40. Current Fixed Decisions

As of Revision R1:

```text
Core project:
Adaptive Edge–Cloud TinyML

Case study:
Handheld Gesture Recognition

Board:
ESP32-S3-WROOM-1-N8R2 development board

IMU:
MPU6050 GY-521

Gesture classes:
IDLE
SWIPE_LEFT
SWIPE_RIGHT
ROTATE_CW
SHAKE

Sampling:
100 Hz

Window:
1 second

Runtime overlap:
50%

Accelerometer:
±4 g

Gyroscope:
±500 °/s

I2C:
400 kHz

Planned I2C pins:
SDA = GPIO8
SCL = GPIO9

Feature count:
10

MC stochastic passes:
5

Production adaptive actions:
LOCAL
CLOUD

Cloud offload payload:
features-v1 (10 values)

Learned production policy:
binary Meta Policy (LOCAL / CLOUD)

Split Controller:
retired from production MVP

Validated split points:
Split1 = 64-D
Split2 = 48-D
Split3 = 32-D

Role of split points:
fixed experimental baselines / negative-result evidence

Server:
FastAPI + MQTT

Broker:
Mosquitto

Database:
PostgreSQL

Dashboard:
FastAPI + HTML/JS + Chart.js + WebSocket

Continual learning:
Future work: EWC

Update:
Future work: Model OTA

Energy:
estimated/simulated proxy unless measurement hardware is added

Final comparison:
All Local
All Cloud
Fixed Split baseline
Rule-Based LOCAL/CLOUD
Learned LOCAL/CLOUD
```

The project must not silently revert to adaptive split selection without a new documented architecture revision supported by measured evidence.

## 41. References / Project Sources

The project source collection should include, at minimum:

1. Original undergraduate project-definition document
2. `GY-521.pdf`
3. `esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf`
4. This file: `PROJECT_ARCHITECTURE.md`

The original project definition describes the Edge–Cloud TinyML concept, learned local/server policy, split inference, uncertainty, MQTT, continual learning, OTA, dashboard, and evaluation structure. The gesture-recognition case study replaces the original fire/smoke testbed while keeping the main project architecture intact.

---

# End of Canonical Architecture Document

R1 example cleanup (2026-09-08): edge components, policy config reference, production MQTT examples, future OTA example and directory/version annotations reconciled with R1. Historical split discussion and measured reports are retained. See docs/phase9_r1_migration_audit.md.

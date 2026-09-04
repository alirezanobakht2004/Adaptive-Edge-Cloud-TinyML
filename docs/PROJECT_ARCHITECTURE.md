# Adaptive Edge–Cloud TinyML for Gesture Recognition
## Canonical Project Architecture & Execution Plan

> **Status:** Canonical architecture document for the project  
> **Current implementation checkpoint:** Phase 2 / pre-M3 — calibrated IMU stream stable; dataset-v1 not yet collected  
> **Purpose:** This file is the main technical source of truth for the project. Any implementation, code organization, experiment, report section, or design decision should be checked against this document first.  
> **Core project title:** سامانه هوشمند تطبیقی لبه–ابر مبتنی بر TinyML  
> **Case study:** Handheld Movement / Gesture Recognition using ESP32-S3 + GY-521 IMU

---

## 1. Project Identity

### 1.1 Core idea
The project is **not primarily a gesture-recognition project**. Gesture recognition is the practical testbed used to evaluate the main research/engineering contribution:

> **A TinyML edge device that adaptively decides how much inference to perform locally and when/where to offload intermediate representations to a server, using learned decision policies rather than only hand-written rules.**

The system must combine:

- TinyML inference on ESP32-S3
- IMU-based gesture recognition
- uncertainty estimation
- learned local-vs-cloud decision making
- learned split-point selection
- MQTT communication
- server-side continuation of a split neural network
- failover to full local inference when Wi-Fi is unavailable
- experiment logging
- dashboard visualization
- continual learning with EWC
- model update through OTA
- comparison against simpler baselines

### 1.2 Case-study framing
Gesture recognition is selected because it is:

- inexpensive to build
- easy to demo repeatedly
- sufficiently non-trivial for TinyML
- based on multivariate time-series patterns rather than a single threshold
- suitable for uncertainty estimation
- suitable for different users / distribution shift
- appropriate for local-vs-cloud adaptive inference
- compatible with continual learning and model personalization

The final presentation should always frame it as:

> **Adaptive Edge–Cloud TinyML architecture, evaluated through a gesture-recognition case study.**

Avoid presenting the project merely as:

> “An intelligent gesture detector.”

---

## 2. Research / Engineering Question

The main question is:

> Can a resource-constrained TinyML device learn, based on prediction uncertainty, network conditions, and device state, whether to finish inference locally or offload computation to a server, and if offloading is selected, which neural-network split point should be used?

Secondary questions:

1. Does adaptive inference reduce communication cost compared with all-cloud inference?
2. Does it preserve useful accuracy compared with fully local inference?
3. Does learned policy selection outperform a simple rule-based adaptive policy?
4. Can the system remain functional when Wi-Fi is lost?
5. Can continual learning improve performance on new user motion patterns without catastrophic forgetting?
6. Can new models be safely distributed to the device through OTA?

---

## 3. System-Level Architecture

```text
                       ┌──────────────────────┐
                       │      MPU6050         │
                       │ ax ay az gx gy gz    │
                       └──────────┬───────────┘
                                  │
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
                              NN Block 1
                                  │
                              Split #1
                                  ▼
                              NN Block 2
                                  │
                              Split #2
                                  ▼
                              NN Block 3
                                  │
                              Split #3
                                  │
                      ┌───────────┴────────────┐
                      │                        │
                      ▼                        ▼
               Edge Exit Head          MC-Dropout Path
                      │                        │
                      │                        ▼
                      │                  Uncertainty
                      │                        │
                      └─────────────┬──────────┘
                                    ▼
                              Meta Learner
                          LOCAL or OFFLOAD?
                               │          │
                         LOCAL │          │ OFFLOAD
                               ▼          ▼
                            Result   Split Controller
                                          │
                                     1 / 2 / 3
                                          │
                                          ▼
                                        MQTT
                                          │
                                          ▼
                                   FastAPI Server
                                          │
                                  NN Tail Continuation
                                          │
                                          ▼
                                     Cloud Result
                                          │
                 ┌────────────────────────┼───────────────────────┐
                 ▼                        ▼                       ▼
            PostgreSQL                Dashboard             Continual
                                                             Learning
                                                                │
                                                               EWC
                                                                │
                                                             TFLite
                                                                │
                                                               OTA
                                                                │
                                                              ESP32
```

---

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
**Sensor module:** GY-521 with an MPU6050-compatible register interface

Current project hardware reports a non-standard `WHO_AM_I=0x74`. The register behavior required by this project has been verified on the current unit, but the report and firmware should not claim a confirmed standard MPU6050 identity.

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

### 4.3 Verified electrical connection
Current project connection:

```text
GY-521              ESP32-S3
--------------------------------
VCC       --------> 3.3V
GND       --------> GND
SDA       --------> GPIO8
SCL       --------> GPIO9
```

Important:

- GPIO8/GPIO9 are the **verified project I2C pins** on the current physical development board.
- Avoid using ESP32-S3 strapping pins for the IMU unless necessary:
  - GPIO0
  - GPIO3
  - GPIO45
  - GPIO46
- Avoid GPIO19/GPIO20 for the IMU because they may be used by USB functionality.
- The development board is powered through 5 V USB.
- The GY-521 should be powered from the board’s 3.3 V pin for clean 3.3 V logic compatibility.

### 4.4 Current sensor configuration
Dataset-v1 baseline:

```text
I2C frequency          400 kHz
Sampling rate          100 Hz
Accelerometer range    ±4 g
Gyroscope range        ±500 °/s
```

These settings are verified on the current hardware and are frozen for dataset-v1. Any intentional change requires documentation and, when it affects recorded data, a new dataset version.

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

The frozen dataset-v1 convention is `orientation-v1`:

- component side upward
- ESP32 / USB end toward the user
- GY-521 end away from the user
- same grip orientation for all planned sessions

Static gravity measurements define the authoritative sign reference: `+Z` gives `az≈+1 g`, `-Z` gives `az≈-1 g`, `+X` gives `ax≈+1 g`, `-X` gives `ax≈-1 g`, `+Y` gives `ay≈+1 g`, and `-Y` gives `ay≈-1 g`.

Document this orientation with the project reference image/photo and keep it unchanged while collecting dataset-v1.

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

Current pre-M3 status: `accel-cal-v1` is applied in firmware and has been validated with six static orientations after correction. The active-axis means were `X+=+1.001801 g`, `X-=-1.008400 g`, `Y+=+1.000596 g`, `Y-=-0.998428 g`, `Z+=+0.999007 g`, and `Z-=-1.007722 g`. These validation measurements are not raw calibration coefficients.

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

## 11. Split Inference

### 11.1 Split points

| Split | ESP32 executes | Server executes | Embedding dimension |
|---|---|---|---:|
| 1 | B1 | B2+B3+B4+B5+Cloud Head | 64 |
| 2 | B1+B2 | B3+B4+B5+Cloud Head | 48 |
| 3 | B1+B2+B3 | B4+B5+Cloud Head | 32 |

### 11.2 Intended trade-off

```text
Split 1:
less edge compute
larger communication payload

Split 3:
more edge compute
smaller communication payload
```

### 11.3 Local mode
When the device chooses full local inference:

```text
Features
→ B1
→ B2
→ B3
→ Edge Head
→ Result
```

### 11.4 Offload mode
When the device chooses cloud assistance:

```text
Features
→ execute selected prefix
→ extract embedding
→ MQTT
→ server executes remaining tail
→ cloud result
```

The project should send intermediate embeddings rather than raw IMU windows during split inference.

---

## 12. TFLite / TinyML Deployment

### 12.1 Edge components
Expected on-device ML components:

- prefix / split network execution
- Edge Exit Head
- uncertainty mechanism
- Meta Learner
- Split Controller

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

## 14. Meta Learner

### 14.1 Responsibility
The Meta Learner answers only:

```text
LOCAL
or
OFFLOAD
```

### 14.2 Candidate inputs

```text
uncertainty
confidence
RSSI
estimated network RTT
free heap ratio
energy budget
```

### 14.3 Important prototype constraint
The current hardware setup has no battery sensor.

Therefore:

```text
energy_budget
```

is initially a **software-simulated or estimated state variable**, not a directly measured battery percentage.

This must be stated honestly in the report.

### 14.4 Initial architecture

```text
Input(6)
   ↓
Dense(8)
ReLU
   ↓
Dense(4)
ReLU
   ↓
Output(2)
LOCAL / OFFLOAD
```

Keep it deliberately small.

---

## 15. Split Controller

### 15.1 Responsibility
The Split Controller executes only when the Meta Learner selects `OFFLOAD`.

It chooses:

```text
Split 1
Split 2
Split 3
```

### 15.2 Candidate inputs

```text
RSSI
RTT
free heap
uncertainty
energy budget
```

### 15.3 Initial architecture

```text
Input(5)
   ↓
Dense(8)
ReLU
   ↓
Output(3)
```

---

## 16. Training the Adaptive Policy

### 16.1 No purely hand-written final policy
The proposed method must be learned.

A rule-based method is allowed only as a baseline.

### 16.2 Generate policy-training data
For many combinations of:

```text
gesture sample
uncertainty
RSSI
RTT
free memory
energy budget
```

benchmark candidate actions:

```text
Action 0 = Full Local
Action 1 = Split 1
Action 2 = Split 2
Action 3 = Split 3
```

Record for each action:

- classification correctness / error
- end-to-end latency
- bytes transmitted
- bytes received
- estimated energy / computation cost
- failure status

### 16.3 Cost function
Initial concept:

```text
J =
w_error * classification_error
+
w_latency * normalized_latency
+
w_comm * normalized_communication
+
w_energy * normalized_energy
```

Initial experimental weights can start around:

```text
w_error   = 0.50
w_latency = 0.25
w_comm    = 0.15
w_energy  = 0.10
```

These are **initial experiment parameters**, not guaranteed final values.

### 16.4 Label generation
The action with minimum cost becomes the supervision label.

Example:

```text
state_i
    ↓
benchmark all actions
    ↓
argmin(J)
    ↓
best_action label
```

Use these labels to train:

- Meta Learner: Local vs Offload
- Split Controller: 1 vs 2 vs 3

---

## 17. Rule-Based Baseline

Implement a simple policy baseline such as:

```text
if Wi-Fi unavailable:
    LOCAL
else if uncertainty > threshold and RSSI > threshold:
    OFFLOAD
else:
    LOCAL
```

And a fixed split selection such as:

```text
if OFFLOAD:
    Split 2
```

This baseline is essential for answering:

> “Why not just use if/else?”

The project should experimentally compare learned adaptive policy against this rule-based version.

---

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

### 18.3 Inference request example

```json
{
  "request_id": "abc123",
  "device_id": "esp32-01",
  "timestamp_ms": 123456,
  "split": 2,
  "embedding": [0.1, -0.2, 0.4],
  "rssi": -67,
  "rtt_ms": 23,
  "uncertainty": 0.31,
  "confidence": 0.62,
  "model_version": "1.0.0"
}
```

### 18.4 Inference response example

```json
{
  "request_id": "abc123",
  "predicted_class": "SWIPE_RIGHT",
  "confidence": 0.94,
  "server_latency_ms": 3.8,
  "model_version": "1.0.0"
}
```

### 18.5 Serialization strategy
Version 1:

- JSON payload
- embedding as numeric array
- easy debugging

Later optimization:

- quantized int8 embedding
- binary serialization if needed

Do not optimize communication before the basic system works.

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

### 21.2 Responsibilities

```text
MQTT Subscriber
      ↓
Request Validation
      ↓
Split Router
      ↓
Split-1 Tail / Split-2 Tail / Split-3 Tail
      ↓
Prediction
      ↓
MQTT Response
      ↓
Database Logging
      ↓
WebSocket / Dashboard
```

### 21.3 Server must not assume raw sensor input
Normal split inference should operate on intermediate embeddings.

Raw sensor data may still be uploaded separately for:

- debugging
- labeled-sample collection
- continual-learning datasets

Keep those paths conceptually separate.

---

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
split_point

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
Execution:        EDGE
Split:            -
RSSI:             -62 dBm
RTT:              19 ms
Total Latency:    5.1 ms
Model:            v1.0.0
```

### 23.3 Charts

Required / useful:

- local exit rate
- cloud-offload rate
- split-point distribution
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
    "meta": "meta_v1.1.0.tflite",
    "split": "split_controller_v1.1.0.tflite"
  },
  "sha256": {
    "model": "...",
    "meta": "...",
    "split": "..."
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
3. **Fixed Split**
4. **Rule-Based Adaptive**
5. **Learned Adaptive** — proposed system

### 26.1 All Local

```text
always B1 → B2 → B3 → Edge Head
```

### 26.2 All Cloud
Always offload using a predefined early split or equivalent server-heavy configuration.

### 26.3 Fixed Split
Always use one split point, e.g.:

```text
Split 2
```

### 26.4 Rule-Based Adaptive
Use threshold logic.

### 26.5 Learned Adaptive
Use:

```text
Meta Learner
+
Split Controller
```

---

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
- edge compute latency
- network latency
- server latency

### Communication
- bytes transmitted/sample
- bytes received/sample
- total traffic
- cloud-offload percentage

### Adaptive behavior
- local exit rate
- Split 1 rate
- Split 2 rate
- Split 3 rate
- policy action distribution

### Reliability
- success under Wi-Fi loss
- server timeout handling
- fallback latency

### Uncertainty
- entropy distribution
- confidence distribution
- calibration curve
- expected calibration error if implemented

### Resource behavior
- free heap
- model size
- tensor arena size
- Flash usage
- PSRAM usage
- approximate compute / energy proxy

---

## 28. Experiment Scenarios

### Scenario A — Clear gesture, good network

Expected behavior:

```text
high confidence
low uncertainty
→ likely LOCAL
```

### Scenario B — Ambiguous gesture, good network

```text
higher uncertainty
good RSSI / RTT
→ OFFLOAD
```

### Scenario C — Ambiguous gesture, poor network

```text
high uncertainty
high network cost
→ learned policy may prefer deeper local computation
```

### Scenario D — Wi-Fi disconnected

```text
→ force local fallback
```

### Scenario E — Different user

```text
performance shift
→ collect labels
→ continual learning
→ compare naive FT vs EWC
```

### Scenario F — Model update

```text
server model v1.1.0
→ OTA
→ ESP32 activates v1.1.0
```

---

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
│   │   └── train_split_controller.py
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
│       │   └── split_controller.cpp
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
split-policy-v1.0.0
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

Deliverables:

- repository
- configuration files
- architecture document
- dataset protocol
- feature-extractor Python skeleton
- experiment naming convention

Definition of done:

```text
repository exists
configs parse correctly
basic Python tests run
```

---

### Phase 1 — Hardware bring-up

Tasks:

1. identify correct USB port
2. flash Hello World / serial test
3. confirm board identity
4. verify Flash/PSRAM configuration
5. solder MPU6050 header
6. connect I2C
7. run I2C scanner
8. confirm MPU address
9. read raw sensor values

Definition of done:

```text
stable ax ay az gx gy gz stream
```

---

### Phase 2 — Data collection

Tasks:

- implement 100 Hz acquisition
- implement timestamps
- implement serial collector
- define recording command
- capture 5 gesture classes
- validate recordings
- build session-based split

Definition of done:

```text
dataset-v1 complete
train / validation / test sessions separated
```

---

### Phase 3 — Base ML model

Tasks:

- exploratory analysis
- compute features-v1
- plot feature distributions
- train classifier
- confusion matrix
- tune only if necessary
- save preprocessing/model metadata

Definition of done:

```text
acceptable validation and held-out test performance
```

---

### Phase 4 — Local TinyML

Tasks:

- export TFLite
- quantize if acceptable
- implement C++ feature extractor
- verify feature parity
- execute model on ESP32
- benchmark latency/memory

Definition of done:

```text
gesture recognized locally on ESP32
```

**Hard rule:** do not begin adaptive cloud work until this phase is stable.

---

### Phase 5 — Uncertainty

Tasks:

- train dropout-enabled path
- implement 5 stochastic passes
- compute entropy/variance
- test ambiguous gestures
- evaluate calibration

Definition of done:

```text
uncertainty score available per inference
```

---

### Phase 6 — Server + MQTT

Tasks:

- install Mosquitto
- FastAPI server
- MQTT request/response
- server tail inference
- timeout handling

Definition of done:

```text
ESP32 sends request
server finishes inference
ESP32 receives result
```

---

### Phase 7 — Split inference

Tasks:

- export split artifacts
- implement split 1
- implement split 2
- implement split 3
- measure payload/latency

Definition of done:

```text
all three split points produce correct end-to-end predictions
```

---

### Phase 8 — Benchmark action space

Tasks:

- test local
- test split 1
- test split 2
- test split 3
- vary RSSI / RTT / state
- build policy dataset

Definition of done:

```text
policy-training dataset generated from measured costs
```

---

### Phase 9 — Learned adaptive policy

Tasks:

- train Meta Learner
- train Split Controller
- deploy policies
- implement rule-based baseline
- compare actions

Definition of done:

```text
device adaptively selects local/offload/split
```

---

### Phase 10 — Failover

Tasks:

- disconnect Wi-Fi
- simulate server timeout
- force local
- log transition

Definition of done:

```text
system continues gesture recognition without network
```

---

### Phase 11 — Database + Dashboard

Tasks:

- PostgreSQL
- persist inference events
- WebSocket live view
- charts
- experiment summaries

Definition of done:

```text
live dashboard reflects device decisions
```

---

### Phase 12 — Continual learning

Tasks:

- collect second-user data
- evaluate model v1
- naive fine-tuning
- EWC fine-tuning
- compare forgetting

Definition of done:

```text
measured continual-learning experiment completed
```

---

### Phase 13 — OTA

Tasks:

- model registry
- manifest
- version check
- download
- SHA-256
- activation
- rollback

Definition of done:

```text
ESP32 moves from model version N to N+1 safely
```

---

### Phase 14 — Final evaluation

Compare:

```text
All Local
All Cloud
Fixed Split
Rule-Based Adaptive
Learned Adaptive
```

Generate final tables and plots.

---

## 34. Milestones

```text
M1  ESP32 boots and reports correctly
M2  MPU6050 raw stream works
M3  dataset-v1 collected
M4  Python gesture classifier works
M5  ESP32 local gesture classification works
M6  uncertainty works
M7  server-assisted inference works
M8  all split points work
M9  learned adaptive policy works
M10 failover works
M11 dashboard works
M12 continual-learning experiment works
M13 OTA works
M14 final benchmark completed
```

---

## 35. Scope-Control Rules

These rules are mandatory unless there is a strong technical reason to change them.

1. Do not add new sensors before the base project works.
2. Do not add more gesture classes before the 5-class model works.
3. Do not build cloud/adaptive components before local TinyML is stable.
4. Do not optimize binary MQTT payloads before JSON works.
5. Do not build a complex frontend.
6. Do not introduce reinforcement learning unless the supervised policy method clearly fails.
7. Do not claim battery measurements without battery-measurement hardware.
8. Do not invent experiment results.
9. Do not change feature definitions without versioning them.
10. Do not change sampling settings without creating a new dataset version.
11. Every new feature must justify its implementation cost.
12. Prefer a working end-to-end system over a larger unfinished system.

---

## 36. Known Risks

### Risk 1 — MC Dropout on TFLite Micro
Ordinary Dropout may be disabled at inference.

Mitigation:

- implement explicit stochastic masking if needed
- verify actual variation across the 5 passes

### Risk 2 — Split-model complexity
Exporting arbitrary graph splits may become time-consuming.

Mitigation:

- design the network as clearly separable sequential blocks
- keep split interfaces simple dense vectors

### Risk 3 — Dataset leakage
Random windows from one gesture sequence may leak across train/test.

Mitigation:

- session-based split

### Risk 4 — Gesture inconsistency
Different grip/orientation may hurt accuracy.

Mitigation:

- freeze physical orientation protocol for v1

### Risk 5 — Adaptive policy becomes effectively always-local
If cloud model adds no accuracy/value, offloading is irrational.

Mitigation:

- use a meaningful deeper cloud continuation
- create ambiguous samples
- benchmark real trade-offs

### Risk 6 — Project scope
EWC + OTA + dashboard + split inference can expand rapidly.

Mitigation:

- follow phase gates
- prioritize M1–M10 before polishing late-stage features

---

## 37. Final Defense Story

The final demo should not be:

```text
perform gesture
→ classifier names gesture
```

Instead it should show the adaptive system.

### Demo moment 1 — confident local result

```text
clear Swipe Right
→ confidence high
→ uncertainty low
→ LOCAL
→ no cloud payload
```

### Demo moment 2 — uncertain + network available

```text
ambiguous movement
→ uncertainty high
→ OFFLOAD
→ selected split
→ embedding sent
→ server result returned
```

### Demo moment 3 — degraded network

```text
network quality changes
→ split selection changes
```

### Demo moment 4 — Wi-Fi loss

```text
Wi-Fi disconnected
→ automatic local fallback
→ gesture recognition continues
```

### Demo moment 5 — model evolution

```text
new-user labeled data
→ continual learning
→ EWC
→ model v2
→ OTA
→ device activates new model
```

---

## 38. Expected Final Report Comparisons

Suggested central result table:

| Strategy | Accuracy | Macro F1 | Mean Latency | P95 Latency | Bytes/Sample | Local Rate | Failure Resilience |
|---|---:|---:|---:|---:|---:|---:|---:|
| All Local | | | | | | | |
| All Cloud | | | | | | | |
| Fixed Split | | | | | | | |
| Rule-Based | | | | | | | |
| Learned Adaptive | | | | | | | |

Do not require the proposed method to be best in every single column.

The intended research claim is more reasonably:

> **The learned adaptive system provides a better overall trade-off among classification quality, latency, communication cost, and network resilience.**

---

## 39. Source-of-Truth Priority

When project decisions conflict, use this priority:

1. **Physical hardware behavior and measured experiments**
2. **Official vendor datasheets**
3. **This architecture document**
4. **Repository configuration files**
5. **Implementation notes / comments**
6. **Old conversation assumptions**

If this document is changed, record the reason.

---

## 40. Current Fixed Decisions

As of the creation of this document:

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

Split points:
3

Server:
FastAPI + MQTT

Broker:
Mosquitto

Database:
PostgreSQL

Dashboard:
FastAPI + HTML/JS + Chart.js + WebSocket

Continual learning:
EWC

Update:
Model OTA

Final comparison:
All Local
All Cloud
Fixed Split
Rule-Based Adaptive
Learned Adaptive
```

---

## 41. References / Project Sources

The project source collection should include, at minimum:

1. Original undergraduate project-definition document
2. `GY-521.pdf`
3. `esp32-s3-wroom-1_wroom-1u_datasheet_en.pdf`
4. This file: `PROJECT_ARCHITECTURE.md`

The original project definition describes the Edge–Cloud TinyML concept, learned local/server policy, split inference, uncertainty, MQTT, continual learning, OTA, dashboard, and evaluation structure. The gesture-recognition case study replaces the original fire/smoke testbed while keeping the main project architecture intact.

---

# End of Canonical Architecture Document

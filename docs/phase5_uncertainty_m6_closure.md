# Phase 5 / M6 — Uncertainty Closure Report

**Project:** Adaptive Edge–Cloud TinyML for Gesture Recognition  
**Checkpoint before closure:** `de6074b feat: integrate continuous on-device uncertainty`  
**Date:** 2026-09-06  
**Dataset:** `dataset-v1`  
**Feature version:** `features-v1`  
**Uncertainty model:** `gesture-model-v1.1.0`  
**Firmware version:** `0.1.0`  
**Hardware:** ESP32-S3-WROOM-1-N8R2 + GY-521 / MPU6050-compatible IMU  
**Sampling:** 100 Hz  
**Window:** 100 samples / 1 s  
**Runtime step:** 50 samples / 50% overlap  
**MC-Dropout passes:** 5  
**Canonical scalar uncertainty score:** normalized predictive entropy  
**Offload threshold / policy:** NOT DEFINED in Phase 5  
**TEST split:** NOT USED in the physical sanity test

---

## 1. Phase 5 Definition of Done

`PROJECT_ARCHITECTURE.md` defines Phase 5 / M6 as:

```text
Tasks:
- train dropout-enabled path
- implement 5 stochastic passes
- compute entropy/variance
- test ambiguous gestures
- evaluate calibration

Definition of done:
uncertainty score available per inference
```

All five tasks have now been exercised, including the ESP32 deployment path.

---

## 2. Completed Phase-5 Evidence

The following Phase-5 checkpoints were completed before the physical ambiguity sanity test:

- dropout-enabled uncertainty model trained;
- exactly 5 stochastic passes validated in Python;
- predictive entropy, normalized entropy, confidence and variance computed;
- controlled feature-space ambiguity probes evaluated;
- calibration evaluated on VALIDATION only;
- a single-inference Python uncertainty API added;
- B1/B2/B3 Float32 TFLite prefix exported;
- explicit post-B3 Dropout masking selected instead of relying on ordinary TFLite Dropout;
- manual Float32 Dense(32→5) Edge Head exported;
- Python/TFLite deployment parity validated;
- fixed-mask Python↔ESP32 parity validated;
- device-seeded stochastic-mask variation validated on ESP32;
- continuous 100 Hz / 50%-overlap on-device uncertainty runtime validated.

The deployed Phase-5 edge path is:

```text
MPU6050
→ 100 Hz sampling
→ 100-sample window / 50-sample step
→ features-v1
→ frozen external normalization
→ B1/B2/B3 Float32 TFLM
→ 5 explicit stochastic Dropout masks
→ manual Dense(32→5) + Softmax
→ mean probabilities
→ confidence
→ normalized predictive entropy
→ variance
```

No LOCAL/OFFLOAD decision is made in this phase.

---

## 3. Physical Live Ambiguity Sanity Protocol

Five physical motion groups were executed on the continuous ESP32 runtime:

```text
A) CLEAR SWIPE_LEFT
B) CLEAR SWIPE_RIGHT
C) AMBIGUOUS LEFT↔RIGHT
D) CLEAR ROTATE_CW
E) AMBIGUOUS ROTATE↔SHAKE
```

Each requested motion was repeated five times with pauses between executions.

This was a **sanity test**, not a new accuracy benchmark, calibration experiment, or threshold-selection experiment.

Because the firmware did not record explicit gesture-start / gesture-stop markers, exact per-execution segmentation cannot be claimed from the serial stream. The interpretation below therefore uses observed motion bursts and idle-separated regions only.

---

## 4. Physical Ambiguity Observations

### 4.1 Clear ROTATE_CW behavior

The clear ROTATE_CW run produced many windows with very high confidence and very low normalized predictive entropy.

Examples measured on-device:

```text
W=28  ROTATE_CW  conf=0.999999  unc=0.000012
W=36  ROTATE_CW  conf=0.999995  unc=0.000042
W=37  ROTATE_CW  conf=0.999988  unc=0.000094
W=45  ROTATE_CW  conf=0.999974  unc=0.000195
W=27  ROTATE_CW  conf=0.999931  unc=0.000476
```

Some transition / overlap windows in the clear-motion run had noticeably higher uncertainty. Therefore a clear physical gesture does **not** imply that every overlapping 1-second runtime window must have low uncertainty.

### 4.2 Ambiguous ROTATE↔SHAKE behavior

The intentionally ambiguous ROTATE↔SHAKE run produced multiple strongly uncertain windows with lower confidence and class instability.

Measured examples:

```text
W=12  SWIPE_LEFT   conf=0.553479  unc=0.708007
W=37  ROTATE_CW    conf=0.292991  unc=0.915287
W=38  ROTATE_CW    conf=0.577141  unc=0.660100
W=52  SWIPE_RIGHT  conf=0.477510  unc=0.753899
W=61  SWIPE_RIGHT  conf=0.643133  unc=0.631412
W=70  SWIPE_RIGHT  conf=0.699483  unc=0.581985
W=71  SWIPE_RIGHT  conf=0.580200  unc=0.573706
```

This is direct real-hardware evidence that physically ambiguous movement can produce:

```text
higher uncertainty
lower confidence
class disagreement / class switching
```

which is the qualitative behavior required for the later adaptive offloading policy.

### 4.3 Ambiguous LEFT↔RIGHT behavior

The ambiguous LEFT↔RIGHT run also produced repeated LEFT/RIGHT switching and several high-uncertainty windows.

Measured examples:

```text
W=28  SWIPE_RIGHT  conf=0.574405  unc=0.687254
W=59  SWIPE_RIGHT  conf=0.637784  unc=0.601766
W=61  SWIPE_LEFT   conf=0.686052  unc=0.518007
W=67  SWIPE_LEFT   conf=0.580553  unc=0.443083
W=88  SWIPE_LEFT   conf=0.542208  unc=0.491984
W=92  SWIPE_RIGHT  conf=0.633433  unc=0.410074
```

However, the clear SWIPE_LEFT / SWIPE_RIGHT runs also contained some high-uncertainty transition windows and occasional left/right class flips.

Therefore the physical sanity evidence **does not justify selecting a fixed entropy threshold in Phase 5**.

This observation is consistent with the project architecture: a rule-based threshold is only a baseline, while the proposed system later uses a learned policy with uncertainty plus network/device state.

---

## 5. Illustrative Manual Segmentation — Not a Formal Benchmark

Using idle-separated motion regions as a manual approximation:

```text
Clear ROTATE_CW motion-region windows:
  normalized entropy median ≈ 0.009
  mean confidence           ≈ 0.937

Ambiguous ROTATE↔SHAKE motion-region windows:
  normalized entropy median ≈ 0.206
  mean confidence           ≈ 0.866
```

These values are useful as a sanity summary only.

They must **not** be reported as a formal accuracy/calibration result because:

- physical gesture boundaries were not timestamp-marked;
- runtime windows overlap by 50%;
- transition windows can contain both motion and idle content;
- the sample size is intentionally small;
- this test was not designed as a statistical benchmark.

---

## 6. Runtime Stability During Physical Tests

During the successful runtime portions of the physical tests:

```text
masks = 5
ov    = 0
rf    = 0
```

throughout the observed inference lines.

The runtime continued producing finite confidence and normalized predictive entropy values on-device.

No offload threshold or policy was used.

---

## 7. Observed Hardware Anomaly

At the beginning of the AMBIGUOUS ROTATE↔SHAKE test, one boot produced:

```text
i2cWriteReadNonStop returned Error -1
WHO_AM_I: 0xFF
FATAL: unsupported sensor identity.
```

After reset, the same hardware initialized successfully with:

```text
WHO_AM_I: 0x74
```

and the ambiguity test proceeded normally.

This is recorded as a **single observed transient I2C / sensor-initialization failure**.

No root cause is claimed from this one occurrence.

It does not invalidate the Phase-5 uncertainty results, but it should remain a known hardware observation. If it becomes reproducible, sensor-startup retry / robustness should be investigated before final system validation.

---

## 8. Interpretation

The physical test supports the intended Phase-5 behavior:

```text
clear gesture
→ can produce high-confidence / low-uncertainty windows

ambiguous gesture
→ can produce substantially higher uncertainty,
  lower confidence,
  and unstable class predictions
```

The separation is not perfect at every overlapping window, which is expected from a 1-second window with 50% overlap and physical transition regions.

This is also an important design result:

> uncertainty is informative, but a single hand-picked entropy threshold should not be treated as the proposed adaptive method.

The later learned policy should use uncertainty together with network state and device state, as defined in `PROJECT_ARCHITECTURE.md`.

---

## 9. Phase 5 / M6 Closure

### Tasks

```text
[x] train dropout-enabled path
[x] implement exactly 5 stochastic passes
[x] compute entropy / variance
[x] test ambiguous gestures
[x] evaluate calibration
```

### Definition of Done

```text
uncertainty score available per inference
```

**Status: PASS**

The scalar normalized predictive entropy is available for every live ESP32 inference window.

### Milestone

```text
M6 — uncertainty works
```

**Status: COMPLETE**

---

## 10. Next Phase Gate

The next architecture-defined phase is:

```text
Phase 6 — Server + MQTT
```

with tasks:

```text
- install Mosquitto
- FastAPI server
- MQTT request/response
- server tail inference
- timeout handling
```

Phase-6 work should start only after this M6 closure evidence is committed.

No uncertainty threshold, split selection, learned offloading policy, or adaptive decision logic should be introduced before the relevant later phases.

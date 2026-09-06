# Phase 6 / M7 — Server + MQTT Closure Report

Project: Adaptive Edge–Cloud TinyML for Gesture Recognition  
Checkpoint before closure: `3a36350 test: validate server response timeout`  
Date: 2026-09-06  
Dataset: `dataset-v1`  
Feature version: `features-v1`  
Edge uncertainty model: `gesture-model-v1.1.0`  
Cloud-tail model: `gesture-cloud-tail-v1.0.0`  
Server stack used in Phase 6: Python + FastAPI + Paho MQTT  
Broker: Mosquitto  
Phase-6 fixed offload path: Split 3 only  
Serialization: JSON  
Production adaptive policy: NOT IMPLEMENTED in Phase 6  
Split 1 / Split 2: NOT IMPLEMENTED in Phase 6  

---

## 1. Phase 6 Definition of Done

`PROJECT_ARCHITECTURE.md` defines Phase 6 / M7 as:

```text
Tasks:
- install Mosquitto
- FastAPI server
- MQTT request/response
- server tail inference
- timeout handling

Definition of done:

ESP32 sends request
server finishes inference
ESP32 receives result
```

Milestone:

```text
M7 — server-assisted inference works
```

---

## 2. Implemented Phase-6 Architecture

The validated Phase-6 server-assisted path is:

```text
ESP32
→ Wi-Fi
→ Mosquitto MQTT broker
→ gesture/{device_id}/inference/request
→ Python MQTT server
→ fixed Split-3 cloud-tail inference
→ gesture/{device_id}/inference/response
→ ESP32
```

Phase 6 intentionally uses only a fixed Split-3 path to close the first working server-assisted inference loop.

This does **not** represent the final adaptive split-selection system.

Split 1, Split 2, full split-point validation, and split-point comparison remain Phase 7 / M8 responsibilities.

---

## 3. MQTT Contract Used

Request topic:

```text
gesture/{device_id}/inference/request
```

Response topic:

```text
gesture/{device_id}/inference/response
```

The Phase-6 server-side request parser validates the following required fields:

```text
request_id
device_id
timestamp_ms
split = 3
embedding = exactly 32 finite numeric values
model_version = gesture-model-v1.1.0
```

The response contains:

```text
request_id
predicted_class
confidence
server_latency_ms
model_version
```

Phase 6 uses JSON with a numeric embedding array.

No binary MQTT serialization optimization was introduced.

---

## 4. Fixed Split-3 Cloud Tail

For the Phase-6 fixed offload path:

```text
Edge representation used by server:
B3 output = 32-D embedding

Server continuation:
B4 Dense(64)
→ B5 Dense(32)
→ Cloud Head Dense(5)
```

Cloud-tail model:

```text
gesture-cloud-tail-v1.0.0
```

Source model:

```text
gesture-model-v1.1.0
```

Cloud-tail artifact SHA-256:

```text
9a834eecf909374488fe346c89b34310bd6c7fdb58570ed2ffb0c885535866d1
```

Measured cloud-tail validation accuracy during training:

```text
0.985000
```

The TEST split was not used during cloud-tail training or model selection.

The fixed Split-3 path in Phase 6 is an intentional MVP path and must not be interpreted as completion of Phase 7 split inference.

---

## 5. Server Runtime Validation

The Phase-6 server runtime validates the expected cloud-tail artifact and request shape before inference.

Measured startup evidence:

```text
PHASE6_MQTT_SERVER_START broker=127.0.0.1:1883
CLOUD_TAIL_READY model=gesture-cloud-tail-v1.0.0 split=3 embedding_dim=32 sha256=9a834eecf909374488fe346c89b34310bd6c7fdb58570ed2ffb0c885535866d1
MQTT_CONNECTED broker=127.0.0.1:1883 subscribe=gesture/+/inference/request
```

The cloud-tail model is loaded once at server startup and reused for requests.

FastAPI health support was also added during this phase as the minimal HTTP server surface; MQTT remains the inference request/response transport for M7.

---

## 6. ESP32 Broker Reachability

Initial direct LAN testing allowed ESP32 Wi-Fi association and DHCP, but ESP32 TCP/1883 traffic did not successfully reach the development PC in that topology.

No exact root cause was proven, so no claim such as definite AP isolation is made.

The development topology was moved to a Windows Mobile Hotspot, after which measured ESP32-to-broker communication succeeded.

Measured hardware reachability evidence included:

```text
BROKER_TCP_CONNECT=PASS
```

The working Phase-6 development topology is:

```text
ESP32
→ Windows Mobile Hotspot
→ development laptop
→ Mosquitto
→ Python server
```

---

## 7. Production Wi-Fi Manager

The Wi-Fi behavior first validated in a hardware test was moved into the production firmware network module.

Measured production Wi-Fi manager result:

```text
WIFI_MANAGER_CONNECT=PASS
WIFI_MANAGER_RSSI_DBM=-17
PHASE6_WIFI_MANAGER_PASS
```

Unity result:

```text
1 Tests 0 Failures
```

The `-17 dBm` value is one measured observation from that run and is not treated as a network benchmark.

Wi-Fi credentials remain in local ignored `firmware/include/secrets.h` and are not committed to the repository.

---

## 8. Production MQTT Publish

`PubSubClient 2.8` was added to the firmware dependencies.

The production MQTT client validated:

```text
Wi-Fi connect
→ MQTT configure
→ MQTT CONNECT
→ MQTT publish
```

Measured hardware test result:

```text
MQTT_CLIENT_CONNECT=PASS state=0
MQTT_CLIENT_PUBLISH=PASS
PHASE6_MQTT_CLIENT_PASS
```

Unity result:

```text
1 Tests 0 Failures
```

An independent PC subscriber received:

```text
gesture/esp32-01/status {"device_id":"esp32-01","status":"phase6_mqtt_client_pass"}
```

This confirms real ESP32 → broker delivery rather than only a successful local API return.

---

## 9. Production MQTT Receive

The production MQTT module was extended with:

- subscription support;
- MQTT message callback dispatch;
- payload delivery to firmware code.

Measured receive test:

```text
MQTT_SUBSCRIBE=PASS
MQTT_RECEIVE=PASS
MQTT_RECEIVED_TOPIC=gesture/esp32-01/inference/response
MQTT_RECEIVED_PAYLOAD={"transport_test":"phase6_mqtt_receive_pass"}
PHASE6_MQTT_RECEIVE_PASS
```

Unity result:

```text
1 Tests 0 Failures
```

A retained transport-probe message was used only to make the receive test deterministic and was removed after validation.

An earlier failed receive test was caused by command-line quoting of the test payload, which produced:

```text
{transport_test:phase6_mqtt_receive_pass}
```

instead of valid JSON.

The broker payload was then republished from a file and independently verified before the final passing ESP32 receive test.

---

## 10. Real ESP32 → Server → ESP32 End-to-End Test

A real Phase-6 E2E request was executed using:

```text
device_id = esp32-01
request_id = m7-esp32-e2e-001
split = 3
embedding dimension = 32
model_version = gesture-model-v1.1.0
```

For this Phase-6 transport/inference test, the embedding consisted of 32 zero values.

This was intentional: the purpose was to validate the complete server-assisted inference path, not real Split-3 feature correctness.

Measured ESP32 output:

```text
E2E_SUBSCRIBE=PASS
E2E_REQUEST_PUBLISH=PASS
E2E_RESPONSE_RECEIVE=PASS
E2E_RESPONSE_TOPIC=gesture/esp32-01/inference/response
E2E_RESPONSE_PAYLOAD={"request_id":"m7-esp32-e2e-001","predicted_class":"SWIPE_RIGHT","confidence":0.22002758085727692,"server_latency_ms":173.5505,"model_version":"gesture-cloud-tail-v1.0.0"}
PHASE6_SERVER_E2E_PASS
```

Unity result:

```text
1 Tests 0 Failures
```

Simultaneously, the server logged:

```text
MQTT_INFERENCE_OK request_id=m7-esp32-e2e-001 device_id=esp32-01 split=3 class=SWIPE_RIGHT confidence=0.220028 server_ms=173.550500 response_topic=gesture/esp32-01/inference/response
```

This directly satisfies the Phase-6 Definition of Done:

```text
ESP32 sends request
server finishes inference
ESP32 receives result
```

---

## 11. Interpretation of the E2E Prediction

The returned prediction in the all-zero embedding test was:

```text
predicted_class = SWIPE_RIGHT
confidence = 0.22002758085727692
```

This is **not** interpreted as a meaningful gesture-classification result because the embedding was synthetic.

The test proves:

```text
ESP32 request
→ server-side cloud-tail model execution
→ MQTT response
→ ESP32 reception
```

It does not prove that a real ESP32 B3 embedding produces the correct end-to-end gesture prediction.

That remains a Phase-7 responsibility.

---

## 12. Server Latency Observation

For the successful hardware E2E run, the server reported:

```text
server_latency_ms = 173.5505
```

This value is one measured sample from one execution.

It is **not**:

- mean latency;
- P95 latency;
- MQTT round-trip latency;
- network RTT;
- full ESP32-to-server-to-ESP32 latency;
- a benchmark result.

Formal split payload and latency measurement belongs to Phase 7 and later benchmarking phases.

---

## 13. Timeout Handling

A negative-path test was executed with:

```text
Mosquitto broker = running
Python inference server = stopped
ESP32 Wi-Fi = connected
ESP32 MQTT = connected
```

The ESP32 successfully published the inference request:

```text
TIMEOUT_REQUEST_PUBLISH=PASS
```

No matching response arrived:

```text
SERVER_RESPONSE_RECEIVED=NO
```

The bounded wait expired:

```text
SERVER_TIMEOUT_DETECTED=PASS
TIMEOUT_ELAPSED_MS=2001
PHASE6_SERVER_TIMEOUT_PASS
```

Unity result:

```text
1 Tests 0 Failures
```

The test-only timeout constant was:

```text
2000 ms
```

The `2001 ms` elapsed value is the measured result from this execution.

The `2000 ms` constant is **not** a selected production/adaptive threshold. It was used only to prove bounded waiting in Phase 6 / M7.

---

## 14. Timeout Scope Boundary

Phase 6 validates:

```text
request published
→ no server response
→ bounded timeout detected
→ firmware does not wait indefinitely
```

Phase 6 does **not** implement:

```text
timeout
→ automatic force-local inference
```

The architecture places complete failover behavior in Phase 10 / M10.

Therefore no Phase-10 failover functionality is claimed in this closure.

---

## 15. Development Security Scope

The current Mosquitto configuration is a local development configuration with anonymous access on explicitly configured local interfaces.

Phase 6 does not claim:

- MQTT authentication;
- TLS;
- production Internet exposure;
- hardened broker deployment.

These were intentionally kept outside the current undergraduate MVP scope.

---

## 16. MQTT Buffer Scope

The production MQTT client currently configures a 1024-byte PubSubClient buffer.

This is an implementation budget used to support the JSON-first M7 path.

No formal communication-cost claim is made from this value.

Actual serialized request sizes and payload/latency measurements must be measured during Phase 7 rather than inferred from the configured buffer capacity.

---

## 17. Relevant Phase-6 Commits

Key Phase-6 commits:

```text
0b6f2f5 feat: add minimal mqtt request response server
9e09ceb feat: add minimal FastAPI health server
8fb0d23 feat: train fixed split3 cloud tail
6116c14 feat: add fixed split3 server inference runtime
2abcddc feat: connect mqtt to split3 cloud inference
6fd831e test: validate ESP32 broker reachability
5a675ed test: validate ESP32 MQTT publish
a849658 feat: add production WiFi manager
de401d8 feat: add production MQTT client
7e63902 feat: add MQTT response receive support
808dd01 test: validate ESP32 server inference end to end
3a36350 test: validate server response timeout
```

---

## 18. Phase 6 / M7 Closure

### Tasks

```text
[x] install Mosquitto
[x] FastAPI server
[x] MQTT request/response
[x] server tail inference
[x] timeout handling
```

### Definition of Done

```text
ESP32 sends request
server finishes inference
ESP32 receives result
```

Status:

```text
PASS
```

### Milestone

```text
M7 — server-assisted inference works
```

Status:

```text
COMPLETE
```

---

## 19. Explicit Non-Claims

Phase 6 does not claim that:

- Split 1 works end to end;
- Split 2 works end to end;
- a real ESP32 Split-3 embedding has been validated end to end;
- all three split points are correct;
- split selection exists;
- an adaptive offloading policy exists;
- a rule-based adaptive policy exists;
- a learned adaptive policy exists;
- a production timeout threshold has been selected;
- failover to local inference has been implemented;
- communication latency has been benchmarked;
- payload cost has been formally measured;
- energy consumption has been measured;
- binary MQTT serialization has been implemented.

These remain later-phase responsibilities.

---

## 20. Phase 7 Gate

The next architecture-defined phase is:

```text
Phase 7 — Split inference
```

Tasks:

```text
- export split artifacts
- implement split 1
- implement split 2
- implement split 3
- measure payload/latency
```

Definition of Done:

```text
all three split points produce correct end-to-end predictions
```

Corresponding milestone:

```text
M8 — all split points work
```

Phase 7 must begin only after this closure report is reviewed and committed.

No learned adaptive policy, adaptive threshold selection, continual learning, OTA, or dashboard work should be started at this point.

---

## 21. Closure Decision

Based on the measured evidence above:

```text
Phase 6 / M7 = COMPLETE
```

The project may proceed to Phase 7 / M8 only after this document is committed as the formal M7 closure record.

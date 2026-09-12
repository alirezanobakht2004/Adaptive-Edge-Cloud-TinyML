# MQTT contracts under Architecture R1

Production service: `python -m server.app.r1_mqtt --events data/policy/runtime/r1_server_events.jsonl`.
Start the existing broker first. Firmware currently uses the existing lab broker
192.168.137.1:1883 and device `esp32-r1`; Wi-Fi credentials remain in ignored secrets.
Do not run two inference subscribers for the same production request topic.

Request topic: `gesture/{device_id}/inference/request`.
Response topic: `gesture/{device_id}/inference/response`.
Schema: `inference-r1-v1`, mode `CLOUD`, feature encoding `normalized-float32`.
Exactly ten finite normalized features-v1 values belong under `features`.
`server/app/r1_schema.py` validates identifiers, versions, values and dimensions.
Requests include device/request IDs, timestamp_ms (device uptime), full-cloud model,
policy and firmware versions, confidence, entropy in nats, margin and the ordered
six-value raw policy state. The six policy inputs are not the ten gesture features.
Unknown modes, missing/extra keys and malformed feature payloads are rejected.

Responses echo request_id, schema, mode and policy version and include success,
predicted_class_id, confidence, model_version and server_latency_ms. The latter is
server compute duration; the ESP32 separately measures E2E request-response time.
The selected full-cloud model is gesture-full-cloud-v1.0.0. Its complete ten-input
network reuses frozen validated weights; CLOUD is never aliased to a split request.

Application RTT probes use `gesture/{device_id}/policy/probe/request` and
`.../response`, schema `r1-probe-v1`. This is shared pre-decision state acquisition,
not a CLOUD inference request. LOCAL sends no inference request and reuses the
already-computed five-pass local result. Probe overhead is not charged a second
time as LOCAL action cost. In the completed production runtime, Wi-Fi, MQTT,
publish, or server-response failure changes the effective action to LOCAL, reuses
the already-computed local result, and records the transition/failure. It does not
execute duplicate local inference for that window.

The ESP32 logs `R1_DECISION` JSON with raw and normalized policy state, selected
action, cached/final predictions, versions, result status and zero second inference
count. CLOUD application payload TX/RX bytes exclude MQTT/TCP/Wi-Fi framing.
Payload publish success and a correlated response are independently observable;
failed/unavailable state and queue overflow have explicit diagnostic lines.

Fixed-split requests without the R1 schema retain `split` and 64/48/32-value
`embedding` routing through `server/app/mqtt.py`. Their action/field meanings are
unchanged historical baselines. The historical four-action full-cloud campaign
used action 3 and `values`; those immutable records are not the binary contract.
No raw 100x6 IMU is transmitted for normal production inference.

Controlled hardware evidence: `docs/evidence/test_phase9_r1_e2e_report.json` and
its server-event JSONL show both learned actions, correlation and exactly ten
features. Controlled vectors do not establish natural physical label diversity.

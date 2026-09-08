# MQTT contracts under Architecture R1

Production target: `inference-r1-v1`, mode `CLOUD`, exactly 10 normalized
features-v1 values under `features`. Include request/device/time identifiers,
feature encoding/version, full-cloud model version, policy and firmware versions.
Responses echo request ID, mode and policy version. The full-cloud implementation
is `server/app/cloud_full.py`; it does not alias CLOUD to Split3.

This target contract is not yet wired into production. Integration follows the
training, export and isolated device parity gates. Existing `server/app/schemas.py`
and `server/app/mqtt.py` serve the retained Phase 7 fixed-split contract: `split`
plus a 64/48/32-value `embedding`. Preserve those routes and tests.

The Phase 9 historical campaign harness uses action 3 and `values` for its full-cloud
benchmark. Its four-action numbering is historical and must not be reinterpreted
as the binary production contract (0 LOCAL, 1 CLOUD). Raw 100x6 IMU is not a normal
production request. No protocol rollout or hardware validation is claimed here.

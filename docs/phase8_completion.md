# Phase8 dataset preparation completion

Completed 2026-09-08. The requested M9–M13 preparation infrastructure is ready
for future learned-policy work. No policy model was trained and no reward,
energy or benchmark performance was invented.

This closes **dataset preparation**, not the canonical architecture's eventual
measured-cost campaign. The main policy dataset intentionally contains zero
observations. A complete device-state campaign and real ALL_CLOUD executor are
still required before training an offloading model.

## Delivered

| Work item | Result |
|---|---|
| M9 data model | Versioned decision record, strict validator, fixed 0/1/2/3 actions |
| M10 instrumentation | Local/prefix/server timing adapters, correlated MQTT round trips, uncertainty calculations, explicit clock provenance |
| M11 collection | JSONL serializer, single-writer collector, manifest/schema checks, duplicate rejection, existing production serial-log importer |
| M12 baselines | ALL_LOCAL, ALL_CLOUD, FIXED_SPLIT1, FIXED_SPLIT2, RULE_BASED_ADAPTIVE; explicit executor boundary and coverage-aware summaries |
| M13 documentation | `phase8_policy_dataset.md`, this completion record, and a supersession notice on the earlier proposed benchmark plan |

The action mapping is **0 ALL_LOCAL, 1 SPLIT1, 2 SPLIT2, 3 ALL_CLOUD**.
MQTT split 3 continues to mean the validated B3 continuation, and is not relabeled
as policy ALL_CLOUD. ALL_CLOUD has a strategy interface but no implicit transport
executor. This preserves both the explicit new request and the existing runtime.

## Files

New implementation:

- `tools/policy_dataset/__init__.py`
- `tools/policy_dataset/schema.py`
- `tools/policy_dataset/instrumentation.py`
- `tools/policy_dataset/serializer.py`
- `tools/policy_dataset/validator.py`
- `tools/policy_dataset/collector.py`
- `tools/policy_dataset/host_smoke.py`
- `tools/policy_dataset/mqtt_smoke.py`
- `ml/policy/benchmark.py`

New tests:

- `tests/test_policy_schema.py`
- `tests/test_policy_instrumentation.py`
- `tests/test_policy_collector.py`
- `tests/test_policy_benchmark.py`

The skeleton in `data/policy/policy_training_dataset_v1/` contains `metadata.json`,
`schema.json`, `sample_000001.json`, empty `observations.jsonl`, and `README.md`.
Smoke evidence uses the same file structure under
`docs/evidence/phase8/host_smoke/` and `docs/evidence/phase8/mqtt_smoke/`.

## Validation

- `python -m py_compile` passed for all new implementation and test modules.
- Full `pytest -q`: **152 passed**, four existing TensorFlow Lite interpreter
  deprecation warnings. Existing Split1/2/3 Python regression tests passed.
- Schema/collector tests cover required keys, action types, versions, timestamp
  timezones, nonfinite values, measured-energy rejection, duplicate JSON keys,
  duplicate sample identities, batch validation and example/observation separation.
- Instrumentation tests cover entropy/margin, timed failures, request correlation,
  timeout preservation, valid responses and the ALL_CLOUD/Split3 distinction.
- Benchmark tests cover fixed/rule selections, unavailable executors, empty
  summaries, measured metrics and explicitly estimated placeholder rewards.
- Skeleton validation: **0 observations, 1 labeled example**.
- Real host Keras smoke: **5 observations** on existing parity inputs.
- Real host/broker/cloud smoke: **10 observations**, five each for Split1/2.
  Records contain actual request/response bytes, prefix/round-trip/server timing
  and outcomes. The host executes the prefixes; these are not ESP32 measurements.
- `git diff --check` passed before commits.

Validation commands, using the existing environment:

```powershell
.venv\Scripts\python.exe -m py_compile tools/policy_dataset/__init__.py tools/policy_dataset/schema.py tools/policy_dataset/instrumentation.py tools/policy_dataset/collector.py tools/policy_dataset/serializer.py tools/policy_dataset/validator.py tools/policy_dataset/host_smoke.py tools/policy_dataset/mqtt_smoke.py ml/policy/benchmark.py tests/test_policy_schema.py tests/test_policy_instrumentation.py tests/test_policy_collector.py tests/test_policy_benchmark.py
.venv\Scripts\pytest.exe -q
.venv\Scripts\python.exe -m tools.policy_dataset.collector validate
.venv\Scripts\python.exe -m tools.policy_dataset.host_smoke --output docs/evidence/phase8/host_smoke
.venv\Scripts\python.exe -m tools.policy_dataset.mqtt_smoke --output docs/evidence/phase8/mqtt_smoke
.venv\Scripts\python.exe -m tools.policy_dataset.collector validate --directory docs/evidence/phase8/host_smoke
.venv\Scripts\python.exe -m tools.policy_dataset.collector validate --directory docs/evidence/phase8/mqtt_smoke
git diff --check
```

Use fresh output directories when repeating smoke commands; evidence is protected
against overwriting. MQTT smoke uses dedicated topics and closes its clients
after completion. It leaves the existing broker running.

## Stability and limitations

No firmware source, production inference/server handler, model artifact,
dataset-v1 sample, feature definition, sampling setting or dependency changed.
No ESP32 was flashed; hardware tests were unnecessary because this work uses
host adapters and the existing diagnostic log format.

Device heap, arena usage, frequency, margin, bandwidth and complete pre-decision
network state require future measured inputs. They are null when unavailable;
host process statistics are never substituted for ESP32 statistics. The serial
importer cannot reconstruct pre-decision state from post-inference logs.

The collected host smoke observations are unlabeled, scope-tagged plumbing
evidence, not a comparative benchmark or policy-training campaign. Their reward
and energy values remain null. Energy interfaces accept estimated/simulated
proxy scores only; there is no measured battery-energy claim.

Before future training: supply a genuine ALL_CLOUD executor, finish device-state
instrumentation with isolated hardware validation if needed, run matched-action
experiments under documented network conditions, freeze reward assumptions,
and validate session partitions, labels and coverage. The collector is
single-writer and is not a transactional multi-process database.

All five requested logical commits were created and pushed incrementally after
validation. The preparation infrastructure is ready for this future collection
and learned-policy integration; the current skeleton is deliberately not marked
training-ready.

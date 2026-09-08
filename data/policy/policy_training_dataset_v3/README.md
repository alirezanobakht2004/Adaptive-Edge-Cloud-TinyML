# policy_training_dataset_v3 — M33 simulated state expansion

This dataset contains **72 simulated matched records** derived from 24 accepted
M30 windows. It contains zero new hardware measurements. Source v1/v2 datasets,
models and production inference are unchanged.

Each record has one state, all four action outcomes, reward components,
feasibility, optimal-action label, source hash and per-field measurement
provenance. Infeasible actions have explicit reasons and null measurements,
predictions and rewards; they are excluded from argmax and dominance tests.

New state dimensions are local compute pressure, inference queue pressure,
cloud availability and RTT-derived network quality. Local inference service
availability and feature transport availability explicitly define feasibility.
Pressure, queue, availability and quality assumptions are simulated, not telemetry.
Existing numeric source fields are marked as carried from measured source and
not remeasured. Scenario C's unavailable fresh uncertainty/prediction is null.

| Scenario | Records | LOCAL labels | SPLIT1 | SPLIT2 | CLOUD labels |
|---|---:|---:|---:|---:|---:|
| A: normal device, poor network | 24 | 24 | 0 | 0 | 0 |
| B: local pressure, good network | 24 | 0 | 0 | 0 | 24 |
| C: inference unavailable, cloud reachable | 24 | 0 | 0 | 0 | 24 |

B's split hypothesis is unsupported; expected actions never enter selection.
C means the inference service is unavailable while feature transport still
works. It does not mean a powered-off or disconnected device can offload.

Files: `metadata.json`, `schema_description.json`, versioned JSONL and
`state_coverage_report.json`. The source M30 campaign must remain available for
validation. Rebuild every record and its report through
`tools.policy_dataset.run_state_expansion_campaign.validate_output`.

Generate another run from the repository root into a new directory:

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.run_state_expansion_campaign --samples 24 --scenario all --output <new-directory>
```

**Training remains blocked.** Multiple labels and coverage reporting exist,
but each split remains dominated in all 48 states where it is feasible.
Simulated scenario copies are not independent acquisitions or evidence of real
device-pressure behavior. Energy remains a simulated dimensionless proxy.

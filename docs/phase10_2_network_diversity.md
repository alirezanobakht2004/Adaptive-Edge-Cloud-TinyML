# Phase10.2 M31 — controlled network perturbation framework

M31 is implemented as an explicitly **simulated controlled-profile replay**.
It produces 96 scenario records from the 24 accepted M30 source samples, with
all four actions evaluated for every state. **Zero new hardware measurements
were collected. Training remains blocked.** No measured improvement is claimed.

## Scope and provenance

The architecture's benchmark/cost-supervision objective and current action
mapping are preserved: 0 ALL_LOCAL, 1 SPLIT1, 2 SPLIT2, 3 ALL_CLOUD. Action 3
retains the source M30 full-cloud execution, not Split3. No firmware, production
inference, model, feature extractor, v1 dataset, measured M30 dataset or M29
reward implementation was modified. No OS traffic shaping, network proxy,
CPU load generator, hardware flashing or learned policy was introduced.

`tools/network_conditions/profiles_v1.json` defines the profiles;
`profiles.py` validates and applies their cost/state transformations without
mutating retained measurements. These are scenario assumptions selected before
generating the labels, not calibrated representations of specific hardware.
They are not chosen by an optimizer to balance the output labels.

| Profile | Edge compute multiplier | Network residual multiplier | Cloud compute multiplier | Added RTT |
|---|---:|---:|---:|---:|
| baseline | 1 | 1 | 1 | 0 ms |
| high_latency | 1 | 1 | 1 | 150 ms |
| edge_loaded | 20 | 1 | 1 | 0 ms |
| cloud_favorable | 8 | 0.25 | 0.5 | 0 ms |

`edge_loaded` is a hypothetical edge-compute slowdown, not a claim that the
ESP32 was loaded. `cloud_favorable` combines that hypothetical slowdown with
lower remote costs; it does not demonstrate a physically achieved deployment.
Baseline is also marked simulated, even though its cost transformation is the
identity. All profiles use `application=offline-cost-overlay` and record exact
injected parameters, version, profile name and `simulated=true`.

## Application method

Inputs are the canonical accepted M30 campaign and its raw device/server
evidence. The runner first validates source labels, hashes, action coverage,
physical trace projections and remote responses using the M30 validator.
Each profile then uses the same selected source windows and all four retained
repeats per action. Prediction, correctness, payload size and connection status
are held fixed by assumption; they are not executed or remeasured.

For source durations in milliseconds, define:

```text
P = preprocessing time
S = pre-decision state inference time
H = total source time - selected action time
Q = H - P - S                     # shared probe/scheduling residual
E = selected edge compute         # local action time, or remote prefix time
C = cloud inference time         # zero for ALL_LOCAL
N = selected time - E - C         # remote residual; zero for ALL_LOCAL
e, n, c, d = injected edge, network, cloud multipliers and added RTT

simulated shared overhead = e*(P+S) + n*Q + d
simulated local action    = e*E
simulated remote action   = e*E + c*C + n*N + d
simulated total latency   = shared overhead + selected action
```

The residual includes serialization, transport and scheduling; it is not a
direct measurement of pure network delay. Negative residuals beyond numeric
tolerance are rejected. Adding RTT charges the shared decision probe once for
every action and the selected request/response once more for remote actions.
The original M30 guard probes and serial/between-action overhead remain excluded.

The projected state updates `inference_latency_estimate` by `e` and `mqtt_rtt_ms`
by `n*source_rtt+d`. Other state values retain the source snapshot. Every action
within a profile/sample sees the identical projected state. Memory, CPU clock,
RSSI, uncertainty and model accuracy are not asserted to change with a timing
multiplier; those dependencies require real experiments or a richer simulator.

## Rewards and dataset contract

M29 computes each repeated candidate reward from the projected total latency,
retained correctness/payload and simulated energy proxy. The provisional M30
weights remain unchanged: correctness 1.0; latency, communication and energy
proxy 0.1 each; scales 100 ms, 1,024 bytes and one proxy unit. The proxy is:

```text
(simulated preprocessing + simulated state compute + simulated edge action compute)/10
+ retained selected payload bytes/1024
```

It is dimensionless, **simulated only**, and excludes radio waiting, probe
traffic and cloud energy. It is not joules, measured battery energy or a fitted
power model. Its communication term overlaps the explicit payload penalty.
The energy kind/version/method explicitly identify M31's simulated transformation.
Neither reward weights nor source predictions were changed to obtain a label.

Labels are argmax of mean M29 reward across four source repeats, preserving
the configured `1e-6` tie tolerance. Repeated winners are retained and ambiguous
labels are null. These are scenario-dependent empirical labels, not measured
optimal actions under physically perturbed network conditions.

Output resides separately under:

`data/policy/policy_training_dataset_v2/controlled_network_v1/`

Each profile has `policy_training_dataset_v2.jsonl`, `configuration.json` and
`dataset_report.json`. Every row includes state, four candidate actions with
repeated projected costs/rewards, reward configuration, `optimal_action`, source
identity/hash and:

```json
{
  "type": "controlled_network",
  "profile": "high_latency",
  "simulated": true
}
```

The actual `network_condition` object additionally stores version, application
and injected parameters. Candidate `metric_kind` is
`simulated_from_measured_source`; predictions/payloads are explicitly retained
from M30 under an invariance assumption. Generation manifests record timestamp,
source dataset digest/selection, profile, code commit and implementation hashes.

Because the measured-only M30 v2 contract is strict, controlled records declare
`schema_extension=controlled-network-replay-v1` and use a separate validator.
`validate_output` reconstructs full records from source samples and profile,
including state, all candidate rewards and labels. It rejects modified types,
missing actions, wrong simulation flags, source selection/hash mismatches and
duplicate identities. The original measured validator remains unchanged.

## Results and diversity gate

| Profile | Scenario records | ALL_LOCAL labels | SPLIT1 labels | SPLIT2 labels | ALL_CLOUD labels |
|---|---:|---:|---:|---:|---:|
| baseline | 24 | 24 | 0 | 0 | 0 |
| high_latency | 24 | 24 | 0 | 0 | 0 |
| edge_loaded | 24 | 0 | 0 | 0 | 24 |
| cloud_favorable | 24 | 0 | 0 | 0 | 24 |
| Total | 96 | 48 | 0 | 0 | 48 |

Each profile has 96 candidate projections per action (24 states times four
repeats), covering all four actions. Across profiles this is 1,536 candidate
projections, 384 per action. These are not 1,536 new device executions.
There are still only 24 independent source windows from one M30 session/device;
profile copies and repeated measurements are correlated, not new acquisitions.
All profiles retain the source correctness by assumption. There are no ties.

`diversity_report.json` records these counts and explicitly reports:

- `real_new_measurements=0` and `simulated=true`;
- two optimal-label classes; neither split is optimal in these scenarios;
- no independent physically measured conditions or qualified calibration;
- `diversity_gate.passed=false` and `training_allowed=false`.

The gate is deliberately conservative and has no claim of a statistically
approved sample threshold. The four-label presence field is diagnostic;
synthetic label coverage by itself would not qualify training even if all
four labels appeared. A subsequent milestone must define/approve readiness
criteria and collect independent real condition evidence.

## Reproduction and validation

Generation used the runner API for all four profiles with 24 samples, default
accepted M30 source and separate new output directories. Equivalent CLI:

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.run_controlled_campaign --profile baseline --samples 24 --output <new-baseline-directory>
.venv\Scripts\python.exe -m tools.policy_dataset.run_controlled_campaign --profile high_latency --samples 24 --output <new-high-latency-directory>
.venv\Scripts\python.exe -m tools.policy_dataset.run_controlled_campaign --profile edge_loaded --samples 24 --output <new-edge-loaded-directory>
.venv\Scripts\python.exe -m tools.policy_dataset.run_controlled_campaign --profile cloud_favorable --samples 24 --output <new-cloud-favorable-directory>
```

`--source` can specify another qualified M30 campaign. Samples are distinct
source windows, selected in stored order; invalid counts and existing output
directories are rejected. No automatic duplication or hardware fallback exists.

Validation executed:

```powershell
.venv\Scripts\python.exe -m py_compile tools/network_conditions/__init__.py tools/network_conditions/profiles.py tools/policy_dataset/run_controlled_campaign.py tests/test_network_conditions.py tests/test_controlled_campaign.py
.venv\Scripts\python.exe -m pytest -q
git diff --check
```

**234 tests passed**, with four existing TensorFlow Lite deprecation warnings.
Compilation and whitespace checks passed. Tests cover explicit simulation flags,
profile validation, identity baseline, additional RTT accounting, immutable
sources, deterministic M29 rewards, schema/action coverage, tampering rejection,
persisted campaign/report reconstruction and output overwrite protection.
No ESP32 tests or firmware flash were needed or performed for M31.

## Limitations and next work

This is an offline sensitivity framework, not physical network emulation.
It models no packet loss, retries, deadline failures, changed predictions,
thermal effects, heap changes or CPU contention scheduling. The multipliers
are hypothetical stress settings, not calibrated hardware behavior. Lower
projected latency or a nonlocal label is not evidence of measured improvement.

Keep profile copies of a source window together in future dataset partitions.
Do not merge this simulated extension into the measured M30 canonical snapshot.
Next implement isolated physical perturbation experiments with measured applied
conditions, broaden device/session/uncertainty coverage, calibrate reward/proxy
assumptions and define independent evaluation partitions. **Training remains
blocked; no policy was trained.**

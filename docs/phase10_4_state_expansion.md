# Phase10.4 M33 — adaptive state expansion

M33's versioned state framework and simulated campaign are complete.
`policy_training_dataset_v3` contains **72 simulated matched records**, based
on the same 24 accepted M30 windows. It contains **zero new hardware measurements**.
Labels are 24 ALL_LOCAL and 48 ALL_CLOUD. Neither split becomes optimal, and
**training remains blocked** by the split-dominance readiness criterion.

## Motivation from M32

M32 showed that the splits were dominated on every retained state, including
the M31 simulations. Reweighting nonnegative costs could not uniquely favor
them. Adding labels by fiat or changing split models would not address the
missing evidence. M33 instead makes pressure and availability assumptions
explicit and checks whether they change actual simulated reward rankings.

The action mapping remains 0 ALL_LOCAL, 1 SPLIT1, 2 SPLIT2, 3 ALL_CLOUD.
ALL_CLOUD continues to mean full-cloud execution from 10 normalized features,
not Split3. Gesture models, prefix models, tails, production inference,
feature definitions and v1/v2 datasets are unchanged. No policy model or
training was implemented, and no new dependency was added.

## State contract and provenance

Existing uncertainty, confidence, RTT, memory and application context are
retained where available. New fields are:

| Field | Representation | Current source/meaning |
|---|---|---|
| `state.device.local_compute_pressure` | Number in [0,1] | Simulated extra-pressure index; not measured CPU utilization |
| `state.device.inference_queue_pressure` | Number in [0,1] | Simulated queue-pressure index; not measured queue occupancy |
| `state.network.cloud_availability` | Boolean | Simulated cloud-service availability |
| `state.network.network_quality_score` | Number in [0,1] | Derived from simulated RTT using an explicit reference |

Two additional availability fields avoid ambiguous feasibility:
`state.device.local_inference_available` and
`state.network.transport_available`. These are simulated booleans.

For available inference, compute multiplier `e = 1/(1-pressure)`. Zero means
no injected slowdown relative to M30; it does not assert an idle physical CPU.
Pressure 0.95 corresponds to a hypothetical 20x compute slowdown. Pressure 1
is used with unavailable inference and is not evaluated as an infinite multiplier.

Queue waiting is `queue_pressure * queue_reference_ms`, with the explicit
pilot reference 40 ms. It is charged once to every action requiring the local
inference service: ALL_LOCAL, SPLIT1 and SPLIT2. ALL_CLOUD bypasses this assumed
inference queue while retaining preprocessing/transport. This is a service-queue
model, not a measured RTOS scheduler or a general CPU lockup model.

Network quality is `1/(1 + simulated_RTT_ms/50)`. The 50 ms reference is explicit
in each scenario. A simulated unavailable transport sets quality to zero and
RTT to null. Quality is an RTT proxy only; it does not measure bandwidth, packet
loss, jitter or radio quality independently.

`measurement_provenance.state_fields` contains an entry for every state leaf:

- `carried_from_measured_source_not_remeasured`: retained M30 source values,
  including memory or uncertainty where available; these are not current
  telemetry under the simulated load.
- `simulated_or_derived`: injected/derived state values, including all new fields.
- `unavailable_or_unmeasured`: null values; missing never means zero.

All records have `measurement_provenance.simulated=true` and zero new hardware
measurements. Predictions for feasible actions are retained source predictions
under an invariance assumption. Costs and labels are simulated, not observed
under real pressure. Cloud availability is true in all three generated scenarios;
cloud-down handling is tested but this dataset has no cloud-down coverage.

## Scenarios and feasibility

Definitions are stored in `tools/policy_dataset/state_scenarios_v1.json`.
Expected actions are a separate diagnostic hypothesis and never enter reward
calculation or argmax selection.

| Scenario | Compute pressure | Queue pressure / wait | Network assumption | Feasible actions |
|---|---:|---|---|---|
| A normal device + poor network | 0 | 0 / 0 ms | Source residual costs; +150 ms per RTT | 0,1,2,3 |
| B local pressure + good network | 0.95 | 0.75 / 30 ms | 0.25x source transport/scheduling residual | 0,1,2,3 |
| C inference unavailable + good cloud | 1; inference unavailable | 1; no inference service execution | 0.25x residual; cloud available | 3 |

Cloud-compute multiplier is one in all three scenarios. Queue and compute
pressure affect prefixes as well as complete local inference; no split-specific
exemption is invented. M31's explicit residual-cost simulation is reused.

Scenario C means **local inference service unavailable while feature production
and transport remain available**. A powered-off or fully disconnected device
cannot offload. Tests cover that case: if inference and transport are both
unavailable, no action is feasible and the optimal label is null.

Because C cannot produce fresh local uncertainty, its confidence, entropy,
margin, local latency estimate and local predicted class are null. Historical
values are not silently asserted to be fresh. Its full-cloud projected cost
also removes the unavailable pre-decision state inference and its compute proxy,
while retaining preprocessing and transport. This is a distinct decision
opportunity with missing uncertainty, not a physical continuation measured on
the ESP32. Future policy training needs an explicit missing-state strategy.

All four action outcomes are present in every row. Infeasible actions contain
an explicit reason and null predictions, measurements, rewards and components.
They are excluded from argmax and from dominance comparisons. They are not
reported as executed failures, successful zero-cost inferences or optimal ties.

## Rewards and v3 records

M29 remains the reward calculator. The provisional source weights are unchanged:
accuracy 1.0; latency, communication and proxy 0.1 each; normalization scales
100 ms, 1,024 bytes and one proxy unit. Feasible candidates retain four repeats.

Queue waiting changes latency but not the existing compute/payload energy proxy,
which excludes waiting. The M33 proxy version explicitly identifies available
scaled edge computation plus payload costs, excluding unavailable state compute,
radio waiting and cloud energy. It remains a **simulated dimensionless proxy**,
not measured joules or battery energy. Proxy calibration and its overlap with
the communication penalty remain unresolved from M32.

Each v3 record contains:

- Version, source identity/hash, model/feature/session metadata and scenario.
- One common expanded state and per-field measurement provenance.
- Four action outcomes, feasibility/reasons, repeated results, normalized and
  weighted reward components, and mean reward where feasible.
- `optimal_action`, all tied `optimal_actions`, label status and disabled training.

The label is argmax mean reward over feasible actions, retaining M29's `1e-6`
tie tolerance. All-infeasible opportunities have `label_status=no_feasible_action`.
Internal M29 replay adapters use its unchanged v1 contract; the exported expanded
state is v3, and no source v1/v2 placeholder rewards or metadata are rewritten.

Dataset files are under `data/policy/policy_training_dataset_v3/`:
versioned JSONL, metadata, schema description, state coverage report and README.
The validator reconstructs each record from the hashed source and scenario,
checking schema/types, provenance, missingness, action feasibility, reward,
labels, source selection and duplicates. Source M30 raw/server evidence is
revalidated before campaign generation and dataset validation.

## Campaign results and readiness

| Scenario | Records | LOCAL | SPLIT1 | SPLIT2 | CLOUD | Hypothesis result |
|---|---:|---:|---:|---:|---:|---|
| A | 24 | 24 | 0 | 0 | 0 | LOCAL supported within simulation |
| B | 24 | 0 | 0 | 0 | 24 | Split hypothesis unsupported |
| C | 24 | 0 | 0 | 0 | 24 | CLOUD is the only feasible candidate |
| Total | 72 | 24 | 0 | 0 | 48 | No forced split labels |

Each action appears in 72 outcome records. Actions 0/1/2 are feasible in 48
records each and explicitly infeasible in C's 24 records each. Action 3 is
feasible in all 72. There are 864 exported repeated feasible candidate projections
and 72 infeasible action placeholders, not new device executions.

Both splits are dominated in all **48 feasible states**. C's infeasible splits
are not counted as nondominated to pass the gate. B increases prefix costs and
queue wait while full cloud bypasses the inference queue, so it does not create
an intermediate split advantage. C establishes only a simulated feasibility
decision; it does not demonstrate full cloud outperforming available splits.

`state_coverage_report.json` reports per-field values, missing counts and
provenance kinds, action/label coverage, scenario hypothesis results and:

| Required minimum criterion | Result |
|---|---|
| More than one action label exists | Pass: LOCAL and CLOUD |
| Split actions are not always dominated | Fail: neither split has a nondominated feasible state |
| State coverage report exists | Pass |

The split criterion is explicitly implemented as requiring each split to have
at least one nondominated feasible state. The current result also fails the
weaker criterion requiring only one split to escape dominance.
`minimum_gate_passed=false` and `training_allowed=false`.

These minimum criteria are not by themselves permission to train or proof of
physical representativeness. All 72 rows reuse 24 source windows from one device
and acquisition session; profile copies and repeats must stay grouped in future
partitions. New state dimensions alone do not create measured diversity.

## Validation and reproduction

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.run_state_expansion_campaign --samples 24 --scenario all --output data/policy/policy_training_dataset_v3
.venv\Scripts\python.exe -m py_compile tools/policy_dataset/state_expansion.py tools/policy_dataset/run_state_expansion_campaign.py tests/test_state_expansion.py
.venv\Scripts\python.exe -m pytest -q
git diff --check
```

The generation command was executed once; subsequent runs need a new output
directory. `--samples` counts distinct accepted source windows per scenario.
`--scenario` can select A, B or C by its full versioned name; `--source` can
select another qualified M30 source campaign.

**251 tests passed**, with four existing TensorFlow Lite deprecation warnings.
Compilation and whitespace checks passed. Tests include deterministic source
preservation, schema/provenance checks, invalid pressure/availability rejection,
queue accounting for all inference actions, skipped unavailable state-compute
costs, cloud/transport unavailability, all-infeasible handling, hypothesis/label
separation, reward tampering and persisted v3 coverage reconstruction.
No firmware tests, flash, hardware measurement or production inference changes
were needed or performed for M33.

## Limitations and next work

Pressure-to-slowdown, queue-to-wait and RTT-to-quality relationships are explicit
heuristics, not calibrated telemetry models. Memory behavior, contention effects,
prediction errors, queue scheduling, timeouts and cloud-down behavior have not
been physically measured under these scenarios. The queue bypass and available
feature transport in C are assumptions requiring validation in an isolated
hardware campaign. The missing-uncertainty case also requires a future versioned
policy input/missingness specification; M25's original features are unchanged.

Next collect real pressure/queue/availability telemetry and matched outcomes,
broaden independent sessions and conditions, and validate whether any feasible
split offers a genuine tradeoff. Revisit reward/proxy calibration without forcing
labels. If splits remain dominated, report that honestly and revisit the future
action-space requirement in a separately authorized scope. **Do not train yet.**

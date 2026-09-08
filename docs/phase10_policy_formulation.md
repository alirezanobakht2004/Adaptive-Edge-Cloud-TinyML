# Phase10 / M25 — Policy formulation and dataset analysis

M25 is complete as a formulation and analysis milestone. **Policy training must
not start with the current dataset.** No policy model, learned selector, target
label generator, fitted preprocessing, firmware change or runtime change is
implemented here.

## Scope and architectural alignment

Reviewed the external `Instruction/PROJECT_ARCHITECTURE.md`, especially sections
14-16, plus `phase8_completion.md`, `phase9_completion.md` and
`phase9_5_external_dataset_evaluation.md`. The current user request names this
work Phase10; the architecture document uses different historical phase numbers.
The architectural separation between a local/offload Meta Learner and subsequent
offloading selection remains a design constraint. No neural architecture is
selected or replaced in M25.

The explicit current action contract takes precedence over the architecture's
older benchmark enumeration: **action 3 is ALL_CLOUD, never Split3**. Split3's
validated inference path remains unchanged and outside this four-action space.
The context registry and encoding below are an offline specification, not a
replacement for the architecture's candidate six-/five-input neural interfaces.
Future implementation must define stage-specific views and map remote decisions
to the correct existing executors before deployment.

## Problem formulation

Each decision opportunity has observed context `S_t`, a selected feasible action
`A_t`, and the resulting classification and resource costs. Formulate the first
learning objective as one-decision, cost-sensitive selection:

```text
pi*(S) = argmax over feasible actions a of E[R_t | S_t = S, execute a]
```

This is an objective specification, not a claim that the present observational
logs identify those expectations. The logs have neither qualified matched-state
targets nor randomized action propensities. They also do not provide a validated
transition model or long-horizon rewards, so M25 does not formulate or implement
reinforcement learning.

An execution log row contains the context, executed action, ground-truth gesture
when known, resulting prediction, costs and status. The logged `action` is a
fixed baseline assignment. It is **not** a supervision label saying that action
was optimal. Action balance alone is not sufficient to train a useful selector.

## Pre-decision state

Define the available context registry as:

```text
S_t = (c_t, H_t, m_t, h_t, a_t, f_t, lhat_t, rtt_t, q_t, yhat_local_t)
```

| Symbol | v1 source path | Meaning / units |
|---|---|---|
| c | state.uncertainty.confidence | Maximum pre-decision local mean probability |
| H | state.uncertainty.entropy | Predictive entropy, nats |
| m | state.uncertainty.margin | Largest minus second-largest probability |
| h | state.device.free_heap | Free heap, bytes |
| a | state.device.tensor_arena_usage | B3 arena actually used, bytes; not total reserved memory |
| f | state.device.cpu_frequency | CPU frequency, MHz |
| lhat | state.device.inference_latency_estimate | Measured pre-decision B3/MC duration, ms; not selected-action latency |
| rtt | state.network.mqtt_rtt_ms | Pre-decision application-level MQTT probe RTT, ms |
| q | state.network.connection_state | connected / disconnected / unknown |
| yhat_local | state.application.predicted_class | Pre-decision local class ID |

For local probabilities `p_k`, `c=max(p_k)`, `H=-sum(p_k log p_k)` and
`m=p_(1)-p_(2)`. These are observed pre-action quantities, distinct from
`outcome.confidence` and `outcome.predicted_class`. Confidence is not ground-truth
accuracy. Phase9.5's high-confidence mapped idle errors reinforce that distinction.

The ten policy context fields are not the ten gesture features in features-v1.
Eight context fields are numerical; two are categorical. The proposed encoding
has **16 columns**: eight numeric columns, three connection-state indicators,
and five predicted-class indicators. Class IDs must not be treated as ordinal.
Numeric centering/scaling may be fitted only on future training partitions;
zero training variance uses a denominator of one,
not division by zero. No means, scales or encoded training matrix are fitted here.
Constant schema slots remain documented until a versioned transform is selected.

All selected features are required for offline training rows. Missing selected
values must be reported or rejected, never replaced by outcome-derived values.
The `unknown` connection category is explicit; its absence from current observations
is a coverage gap. Future online missing/out-of-support handling requires a
validated fallback design; M25 adds no such runtime behavior.

Do not use `window_id`, sample/run/session/device IDs, timestamps, action, outcome,
reward or post-action measurements as model inputs. Keep IDs and timestamps for
provenance, grouping and experimental analysis. This prevents direct target leakage
and reduces the risk of learning the collection schedule.

Deferred fields:

- `state.network.packet_size_estimate` and `estimated_bandwidth`: null in every row.
  Post-action payload bytes cannot be substituted into the pre-decision estimate.
- RSSI and PSRAM: available in Phase9 raw sidecars, but absent from the immutable
  v1 state schema. A future correlated, versioned projection must define their use.
- Free-heap ratio: requires a documented denominator; only absolute heap is selected.
- Energy budget: no measured battery state exists. Any future budget must be
  explicitly estimated or simulated and versioned.

## Actions and feasibility

| Action | Meaning | Existing execution contract |
|---:|---|---|
| 0 | ALL_LOCAL | Complete local edge inference |
| 1 | SPLIT1 | 64-D prefix embedding plus Split1 tail |
| 2 | SPLIT2 | 48-D prefix embedding plus Split2 tail |
| 3 | ALL_CLOUD | Ten normalized features; all neural blocks execute on server |

Conceptually, action 0 maps to the local gate outcome; actions 1-3 map to offload
followed by the corresponding remote execution choice. They do not rename the
production MQTT split enumeration. The action registry is fixed in
`ml/policy/policy_config_v1.json` and checked against the existing v1 schema.

Future selection must be constrained by validated runtime availability, model
compatibility, resources and connectivity. A connected MQTT flag alone does not
prove server availability or deadline feasibility. Local fallback requires the
local runtime itself to be available. Resource limits, deadlines and fallback
transitions are not invented or implemented by this milestone.

## Proposed reward

For action `a`, let `C_t(a)` be one for a successful correct prediction and zero
for a wrong prediction or failed/timed-out execution. A successful unlabeled
execution has unknown correctness and cannot receive an accuracy-based reward.
Ground-truth labels are an offline evaluation input, never part of `S_t`.

```text
R_t(a) = w_accuracy * C_t(a)
       - w_latency * L_t(a) / L_ref
       - w_communication * B_t(a) / B_ref
       - w_energy_proxy * Ehat_t(a) / E_ref
```

All weights must be nonnegative and all reference scales positive when a future
experiment configures them. The architecture's example weights are suggestions;
**M25 leaves every weight and scale null**, as requested. No rewards or best-action
labels are computed. Normalized costs are not automatically clipped at one.

| Component | Proposed observable / limitation |
|---|---|
| Accuracy reward | Successful prediction equals known `outcome.true_class`; failed attempts receive zero correctness |
| Latency penalty | `measurements.total_latency_ms`; current instrumented decision interval includes normalization, pre-decision inference, probe and selected action |
| Communication penalty | `request_bytes + response_bytes`; selected-action JSON payload bytes only |
| Energy proxy penalty | Nonnegative, explicitly estimated/simulated proxy with a named model, units and assumptions; currently unavailable |

The latency interval excludes serial feature transfer and original feature
extraction. Current local timing includes a second B3/MC execution after the
pre-decision pass. A later runtime that reuses results needs fresh cost measurements;
these data do not measure that optimization. Communication excludes probe traffic,
MQTT framing, Wi-Fi overhead and retransmissions. Probe bytes remain in sidecars.
Changing these cost definitions requires a new reward version and appropriate data.

Energy is **estimated only**, or explicitly simulated. There is no battery-energy
measurement claim, no invented coefficient and no conversion of unknown values to
zero. If a future experiment omits energy, it must explicitly define a separate
configured objective; it must not silently replace missing energy with zero.
Unlabeled successful outcomes or missing required costs leave the reward unavailable.
Failure attempts must retain measured attempt duration/bytes rather than being
dropped or given fabricated costs.

The `policy_training_dataset_v1` reward schema currently enforces a negative sum of cost placeholders
and has no accuracy component. Therefore `policy-reward-proposal-v1` is a separate
specification. A future derived target artifact must reference source identities
and reward configuration; it must **not overwrite** v1 `reward.final_reward` or
silently relax the existing validator.

## Measured dataset analysis

The generated [analysis report](evidence/phase10_policy_dataset_analysis.json)
validates the canonical observations, named snapshot and campaign manifests. It
reads `observations.jsonl` once; the identical named JSONL and per-run copies are
not additional data. The unmeasured example, zero-sample failed startup and external
Kaggle resource are excluded from observation counts.

| Coverage item | Measured result |
|---|---|
| Observations | 100 physical-device records |
| Action distribution | 25 each for ALL_LOCAL, SPLIT1, SPLIT2, ALL_CLOUD |
| Unique source windows | 25; all four actions executed for every window |
| Source sessions | One: session_02 |
| Device IDs | One: esp32-s3-n8r2 |
| Runs | Four observation-bearing runs; one fixed action in each |
| Selected context completeness | 100/100 rows, all ten fields |
| Full selected state vectors | 100 distinct; none observed under all four actions |
| Uncertainty repeats | All 25 source windows repeat identical uncertainty across action runs |
| Network state | connected in all 100 observations |
| Outcomes | 100 successful, correct labeled executions; no observed classification errors/timeouts |
| Gesture labels | 20 records per class; five distinct source windows per class |
| Configured reward / energy values | 0 / 0 |

Exact equality of continuous state vectors is a descriptive diagnostic, not a
requirement for future experimental matching. However, replaying one window in
four sequential runs does not make the four measured device/network contexts
identical counterfactual observations. Action, run order, host load and network
conditions are confounded. All-correct outcomes also provide no evidence about
when offloading rescues a local classification error.

| Numeric state feature | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| Confidence | 0.766577 | 0.981507 | 0.999999 |
| Entropy, nats | 0.00001459 | 0.067825 | 0.667137 |
| Margin | 0.576300 | 0.965849 | 0.999998 |
| Free heap, bytes | 164368 | 164709.04 | 165344 |
| B3 arena used, bytes | 1216 | 1216 | 1216 |
| CPU, MHz | 240 | 240 | 240 |
| Pre-decision inference estimate, ms | 0.943 | 1.33408 | 2.285 |
| MQTT probe RTT, ms | 9.305 | 12.34024 | 21.696 |

The report includes population standard deviations, medians, per-action state
statistics and categorical counts. Arena usage, CPU frequency and connection state
are constant. Heap varies by only 976 bytes. This is not broad resource-pressure
or network coverage.

Confidence histogram counts are 0 in `[0,0.5)`, 4 in `[0.5,0.8)`, 4 in `[0.8,0.9)`,
0 in `[0.9,0.95)` and 92 in `[0.95,1]`. Those 92 records represent only 23 unique
windows. Entropy and margin histograms are also retained in the report. Bin edges
are descriptive analysis choices, not policy thresholds.

Missing values need different interpretations:

- Bandwidth and packet-size estimates: 100 missing each; deferred state candidates.
- Reward components and final reward: 100 missing each; a training-target blocker.
- Local confidence/entropy/margin under `measurements`: 100 null each; pre-decision
  uncertainty is present under `state`, so do not substitute post-action values.
- UTC send/response timestamps: 100 null; device boot-relative timestamps are
  retained in raw traces rather than falsely converted to UTC.
- Cloud latency/round-trip/server times: null for 25 local actions, as expected.
  Local action inference latency is null for the 75 remote actions.
- `outcome.error`: null for 100 successful executions, not a missing failure label.

## Future supervision and partition specification

Before training, design repeated, randomized or counterbalanced action benchmarks
under documented network/resource conditions. Record condition identity, order,
timestamps and pre-decision contexts; retain failures and low-confidence/misclassified
cases. If later using propensity-based off-policy estimators, logging must provide
known probabilities and adequate action overlap; current fixed-mode logs do not.

Once reward parameters, accounting, feasibility rules and matching criteria are
fixed, derive the set of actions maximizing estimated reward under matched
conditions. Repeated measurements must characterize variability. Ties or differences
within a predeclared uncertainty tolerance require an explicit target rule; no
unique best action may be invented from indistinguishable costs. M25 creates no
such labels and selects no classifier/regressor architecture.

Keep every repeat of `(source session, source window)` together in any partition.
Prefer independently acquired session/participant holdouts and document their
provenance. The current single session cannot supply independent train, validation
and test sessions. Three session groups would only make that partition structurally
possible, not establish statistical sufficiency. No numerical sample quota or
coverage threshold is invented here; those need an approved experiment design.

Fit numeric preprocessing on the training partition only. Freeze selection,
encoding, transforms, reward parameters and target-construction version before
held-out evaluation. Compare against all fixed baselines and an actually executed
rule-based baseline using consistent cost accounting. The external Kaggle data
has no offloading action/cost labels and must not be merged into the policy dataset.

## Readiness decision and deliverables

| Gate | Status |
|---|---|
| Schema, action mapping, required field availability | Pass |
| State/action/reward formulation and versioned placeholders | Defined in M25 |
| Independent partitions and diverse uncertainty/resource/network conditions | Not ready |
| Configured reward, energy assumptions and qualified optimal-action targets | Not ready |
| Learned model design / training / deployment | Not started; disabled |

**Training cannot start.** Complete the collection design and missing target/reward
decisions first, then reassess readiness in a subsequent milestone. Finishing M25
does not by itself authorize or qualify training. Phase8/9 collection closure and
Phase9.5 external evaluation closure are not training-readiness certificates.

Delivered:

- `ml/policy/policy_config_v1.json`: versioned state registry, fixed actions,
  reward placeholders, deferred features, leakage exclusions and partition plan.
- `ml/policy/formulation.py`: configuration/schema and feature-availability checks;
  no selector or model.
- `tools/policy_dataset/analyze_policy_dataset.py`: reproducible descriptive audit.
- `docs/evidence/phase10_policy_dataset_analysis.json`: counts, statistics,
  distributions, missingness, coverage, source/config hashes and readiness gates.
- `tests/test_policy_formulation.py`: mapping, schema compatibility, missing
  selected features, leakage rejection, versioning, placeholder enforcement,
  grouping, duplicate rejection and histogram boundary checks.

Validation commands:

```powershell
python -m tools.policy_dataset.analyze_policy_dataset
python -m py_compile ml/policy/formulation.py tools/policy_dataset/analyze_policy_dataset.py tests/test_policy_formulation.py
python -m pytest -q
git diff --check
```

Full validation: **188 tests passed**, four existing TensorFlow Lite deprecation
warnings; compilation and whitespace checks passed. No ESP32 tests were necessary
or run. Dataset artifacts, learned gesture models, firmware, existing split runtime
and external evaluation artifacts remain unchanged.
The final [validation log](evidence/phase10_m25_validation.log) is retained.

# Phase10.1 M29: reward calibration framework

M29 provides offline candidate evaluation and target generation. It does not
close dataset expansion or enable training. The original 100 observations remain
unchanged; **zero real optimal-action labels were generated in this milestone**.
The existing sequential fixed-mode runs are not matched decision-state experiments.

## Contract and architecture

The action mapping remains 0 ALL_LOCAL, 1 SPLIT1, 2 SPLIT2, 3 ALL_CLOUD.
Action 3 executes the full cloud model; it is never MQTT split 3. No firmware,
model, production inference, feature extractor, or dataset schema was changed.
This implements the architecture's benchmark-derived cost supervision concept,
using the explicit current four-action contract and the M25 state specification.

`tools/policy_dataset/reward_engine/engine.py` exposes:

- `collect_candidates(decision, decision_id=..., matching_evidence=...,
  executors=..., energy_estimator=..., config=...)`: calls four explicitly supplied
  action adapters on independent copies of the same v1 pre-decision record. An
  adapter mutates its copy with actual measurements/outcome. It must retain the
  action and context. Returns the raw candidate group and calibrated result.
- `evaluate_group(group, config)`: validates collected candidate records and
  calculates rewards without performing inference or changing source records.
- `validate_config(config)`: rejects unconfigured or invalid calibration values.

No transport adapters, energy coefficients or automatic physical state reset are
supplied. Existing fixed-action campaign runners cannot simply be combined by
window ID. Hardware adapters must implement the named action honestly; Python
cannot inspect a callback to prove that ALL_CLOUD ran the full cloud graph.

## Matched-state collection requirements

Before collection, freeze the complete pre-decision state, input window and
versions. Supply a nonempty `matching_evidence` reference to a collection protocol
and retained input/trace evidence. Record each actual action's costs and ground
truth. Reuse the same source window, device, session, model/feature versions and
measurement scope. The engine checks equality of the entire logged state and
those metadata fields, and requires all M25 selected state features.

Snapshot equality is necessary, not proof that physical conditions were equal.
Callbacks must not disguise changed physical conditions by copying old metrics.
Use controlled/reset conditions, repetitions and counterbalanced action order;
audit environmental drift and input/artifact identity against retained traces.
The convenience collector invokes 0,1,2,3 in order and does not implement that
experimental design. For counterbalanced campaigns, collect independently with
appropriate adapters and pass the resulting group to `evaluate_group`.

Each group has exactly these fields:

```text
version: candidate-actions-v1
decision_id: globally unique experiment decision identifier
matching_evidence: collection protocol / trace reference
decision: complete policy_training_dataset_v1 reference record
candidates: four objects, each containing:
  record: complete executed v1 observation with a distinct action
  energy:
    value: finite nonnegative estimated/simulated proxy score
    kind: estimated or simulated
    proxy_version: explicit calibrated assumption version
    method: explicit method and coefficient/assumption description
```

Missing actions, duplicate actions, missing ground truth, inconsistent true labels,
unexecuted observations, mismatched states/versions/scopes, missing measured costs
and inconsistent energy assumptions prevent label generation. Timeouts/failures
may be candidates only with known ground truth and actual latency/byte accounting.
The reference record may omit ground truth; when present it must agree.

## Reward and calibration

For action a:

```text
C_a = 1 if execution succeeds and prediction equals ground truth, otherwise 0
L_a = measured total_latency_ms / configured latency_ms scale
B_a = (measured request_bytes + response_bytes) / communication_bytes scale
E_a = estimated_or_simulated_energy_proxy / energy_proxy scale
R_a = w_accuracy*C_a - w_latency*L_a - w_communication*B_a - w_energy_proxy*E_a
```

Weights must be explicit finite nonnegative numbers with at least one positive;
scales must be explicit finite positive numbers. They need not sum to one.
Normalization is not clipped: exceeding a reference cost increases its penalty.
All components are required even if their weight is zero; missing is never zero.
Energy is **estimated/simulated only, never measured battery energy**. Its method,
units, coefficients, version and assumptions must be described in configuration
and match each candidate. The engine accepts externally computed proxy scores;
it does not infer joules or invent an energy model from elapsed time.

`config_template.json` deliberately contains null weights/scales/tie tolerance
and unconfigured energy assumptions. It cannot produce rewards until a researcher
creates and explicitly configures a separate calibration JSON. Choose reference
scales and weights using documented training/calibration data and application
priorities, never by fitting held-out test results. Compare sensitivity across
separately versioned configurations before accepting labels. No final values have
been chosen here; numeric values in tests are synthetic test fixtures only.

Communication costs count selected request/response payload bytes, not Wi-Fi,
TCP/MQTT headers or pre-decision probe traffic. Total latency must have consistent
start/end boundaries across all adapters, including common decision overhead
when present. Failed executions must include timeout time and actual bytes.
Do not mix different accounting boundaries in a candidate group.

## Labels and reproducibility

The maximum observed reward determines `optimal_action`. All candidates within
the explicitly configured absolute `tie_tolerance` of that maximum are retained
in `optimal_actions`. If more than one qualifies, `optimal_action` is null.
Tolerance is a declared indifference band, not an inferred confidence interval.
These are observed candidate argmax labels, not proof of the expected optimal
action under stochastic network/device behavior. Repeat experiments before
qualifying supervision; no statistical optimality or training readiness is claimed.

Outputs use `candidate-rewards-v1`, with per-action correctness, prediction,
status, raw costs, normalized components and reward, plus context/input/record
SHA-256 hashes, complete calibration and its hash, matching evidence and disabled
training status. Retain raw groups and referenced physical traces with outputs.
Canonical JSON hashes identify content independently of whitespace.
The new accuracy-inclusive objective never overwrites the incompatible v1
placeholder `reward` object or M25's intentionally unconfigured policy config.

Run from the repository root after collecting qualified groups and configuring
an explicitly reviewed calibration file:

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.reward_engine --input matched_candidates.jsonl --config calibration.json --output candidate_rewards.jsonl
```

Input is one candidate group per JSONL line. The CLI validates the full batch
before writing, rejects duplicate decision IDs and duplicate JSON keys, and uses
exclusive output creation to avoid overwriting existing evidence. Parent output
directory must already exist. Empty input is rejected. Keep different calibration
runs in separate output files. The tool does not merge outputs into the dataset.

## Validation and remaining work

Validation commands:

```powershell
.venv\Scripts\python.exe -m py_compile tools/policy_dataset/reward_engine/__init__.py tools/policy_dataset/reward_engine/engine.py tools/policy_dataset/reward_engine/__main__.py tests/test_reward_engine.py
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m tools.policy_dataset.reward_engine --help
git diff --check
```

The 15 new tests cover deterministic calculations, source-schema preservation,
action mapping, incomplete/duplicate candidates, unmatched states, missing state
features, incorrect energy claims, invalid numbers/scales, inconsistent labels,
ties, failed-action costs, adapter isolation and CLI validation/output protection.
Validation result: **203 tests passed**, including all 15 new tests, with four
existing TensorFlow Lite deprecation warnings. Compilation, CLI help and
whitespace checks passed. No hardware tests or flash
are required: this milestone adds only a host-side framework.

Remaining prerequisites are real matched-condition adapters/campaigns, calibrated
weights/scales and energy assumptions, repeatability analysis, greater session and
network/resource coverage, qualified labels and independent evaluation partitions.
Training remains intentionally blocked. M29 framework completion is not completion
of Phase10.1 dataset expansion.

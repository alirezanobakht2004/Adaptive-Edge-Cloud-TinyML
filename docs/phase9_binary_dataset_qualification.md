> Historical initial binary-v1 qualification. Its blocked result below is preserved.
> Current R1-reuse v2 qualification and M9 outcome are in [the canonical Phase 9 report](phase9_learned_binary_policy.md) and `docs/evidence/phase9_r1_reuse_dataset_qualification.json`.

# Phase 9 binary dataset qualification

**Training blocked: candidate execution does not match the R1 LOCAL contract.**
The new dataset is a reproducible historical binary comparison. Its labels are
not approved production optimal-action targets. Structural validation and a
balanced-enough label distribution cannot resolve an action-cost mismatch.

Machine-readable results:
[phase9_binary_dataset_qualification.json](evidence/phase9_binary_dataset_qualification.json).
Transformation and sources:
[manifest.json](../data/policy/policy_training_dataset_binary_v1/manifest.json).

## Sources and counts

| Source | Imported | LOCAL | CLOUD | Provenance |
|---|---:|---:|---:|---|
| M30 accepted connected_lab_run02 | 24 | 24 | 0 | Measured timing/outcomes; estimated energy proxy |
| M31 baseline | 24 | 24 | 0 | Simulated replay, including identity overlay |
| M31 high_latency | 24 | 24 | 0 | Simulated replay |
| M31 edge_loaded | 24 | 0 | 24 | Simulated replay |
| M31 cloud_favorable | 24 | 0 | 24 | Simulated replay |
| M33 A_normal_poor_network | 24 | 24 | 0 | Simulated replay |
| M33 B_pressure_good_network | 24 | 0 | 24 | Simulated replay |
| Total | **168** | **96** | **72** | **24 measured / 144 simulated** |

All rows trace to **24 original windows**, `dataset-v1`, `features-v1`,
`session_02`, acquisition run `48d1e9e3cb0747fd9e571ba760f19910`.
There are no newly measured samples in this migration. All CLOUD labels originate
from simulation. Retained predictions are measured source predictions, not new
observations under the simulated condition.

Excluded from derivation: 24 M33 C rows have no feasible LOCAL candidate
measurements and no fresh local uncertainty. Each exclusion identifies its row
and reason. The 100 unmatched v1 observations lack same-state alternatives; they
are not silently paired. M30's canonical mirror is not counted twice. Its first
diagnostic campaign is not imported as an independent qualified run.

## Derivation and provenance

`ml/policy/binary_policy_dataset.py` first runs existing source validators against
M30 raw device traces/server events/model and feature hashes, then reconstructs
M31 and M33 projections. It retains the complete original four-action outcomes,
source metadata, IDs/content hashes, underlying window/session/run, original
reward configuration and controlled profile/scenario.

Only historical candidates **0 (ALL_LOCAL)** and **3 (ALL_CLOUD)** enter the new
comparison. Their repeat-level correctness, latency, bytes and energy proxy are
rescored with the frozen M29 formula, averaged across four repeats, then compared.
New labels are **0 LOCAL / 1 CLOUD**. The old four-action argmax is retained only
as provenance. Ties retain all winners and no fabricated unique label.

No source dataset, model, historical label or report is changed. No measured
energy claim is made. JSON content hashes use canonical JSON, independent of
pretty-print layout; the manifest also references model/feature artifact hashes.

## Feature contract

`ml/policy/policy_config_v2.json`, policy specification `binary-policy-contract-v1`,
freezes this order under `binary-state-normalization-v1`:

| Index | State input | Runtime source before decision | Scaling |
|---:|---|---|---|
| 0 | confidence | Existing local five-pass MC result | unchanged probability |
| 1 | entropy | Existing predictive entropy | divide nats by ln(5) |
| 2 | margin | Largest minus second-largest mean MC probability | unchanged |
| 3 | free_heap | `ESP.getFreeHeap()` after local state calculation | bytes / 1024 |
| 4 | inference_latency_estimate | `micros()` around B3 + five-pass MC | ms / 100 |
| 5 | mqtt_rtt_ms | Correlated pre-decision MQTT application probe | ms / 100 |

The time divisor is the existing M30 100 ms normalization reference, not a fitted
threshold or new reward weight. There is no clipping, imputation or fitting on
holdout. Runtime evidence is the existing isolated M30 harness. The probe still
needs production wiring after deployment gates; defining a source does not claim
that integration has already occurred.

Confidence, entropy, margin and heap are carried from measured source states in
replay; latency and RTT may be transformed. Per-record/per-feature provenance is
stored. All six selected inputs are present in all 168 included rows. Unselected
packet-size estimate and bandwidth are missing in all 168 rows. Synthetic local
compute pressure, queue pressure, cloud availability and quality score are not
learned inputs. No unavailable field is replaced by an action outcome.

## Leakage and holdout

Group identity consists of original dataset version, feature version, session and
window ID; **run/profile/repeat are not independent group identifiers**. Stable
SHA-256 ordering with seed `20260908` assigns six of 24 groups to holdout:

| Partition | Source windows | Records | LOCAL | CLOUD |
|---|---:|---:|---:|---:|
| Reserved train | 18 | 126 | 72 | 54 |
| Reserved holdout | 6 | 42 | 24 | 18 |

Source-group overlap is zero. All seven variants of each window stay together.
There are 120 distinct selected feature vectors; 48 vectors are repeated, with
no conflicting labels. These are retained as dependent study variants, not extra
independent evidence. A later training procedure must account for this repeated
weighting. The split is a **within-session window holdout**, not a new-session or
new-user evaluation. No model or threshold was fitted to either partition.

## Frozen reward

`ml/policy/reward_binary_experiment_v1.json` wraps an exact copy of M30's
`reward-calibration-v1`, with provenance to `connected_lab_v1.json` and commit
`bc24ed1`. Its original four-action names remain inside the historical copy;
the binary action mapping is separate.

```text
reward = 1.0 * correctness
       - 0.1 * latency_ms / 100
       - 0.1 * communication_bytes / 1024
       - 0.1 * energy_proxy / 1
```

Tie tolerance remains `1e-6`. Weights and scales remain provisional, frozen rather
than recalibrated. M30 proxy version is `m30-compute-payload-proxy-v1`; M31/M33
simulated proxy method/version wrappers are preserved with each source. All are
dimensionless compute/payload estimates. None measures joules or battery use.

## Hard gate: production action semantics

The M30 harness `firmware/test/test_phase10_matched_campaign/test_main.cpp`:

1. At line 100, runs `local(normalized, state)` to obtain uncertainty and a result.
2. At lines 131–134, action 0 runs `local(normalized, result)` **again**.
3. At line 155, records `sharedUs + actionUs`, including that second pass.

Across 96 physical action-0 repeats, that extra execution interval was
**0.954–2.322 ms, mean 1.7986875 ms**. These are existing diagnostic-harness timings,
not new measurements of R1. The M30 energy formula also charges selected edge
compute. M31 multiplies that redundant compute under device pressure, and M33
inherits it. The full-cloud candidate does not incur a second edge inference.

R1 explicitly requires LOCAL to **reuse the already computed local result**.
Consequently none of these action-0 measurements represents the required R1
branch. All pre-decision local predictions in the accepted source entries are
already correct; these data do not demonstrate a cloud accuracy benefit either.
With common state-computation cost already incurred, cloud introduces additional
cost. This is an algebraic observation, not a newly measured reward or relabeling.

The structural binary schema, runtime source definitions, frozen normalization,
reproducible reward, provenance, both labels and group holdout checks pass.
**Semantic compatibility with the production action contract fails.** Therefore
the derived comparison contains **0 production-qualified records** and training
remains disabled. Treating the 72 historical CLOUD winners as production-optimal
labels would assert an unmeasured cost comparison, despite correct arithmetic.

Do not silently subtract timings, reset weights, force label balance or run local
inference twice in production to fit the dataset. The next Phase 9 task is an
isolated R1 matched campaign that measures result reuse versus full-cloud using
the production request contract, preserves pre-decision state, and establishes
whether both labels remain supported. If no cloud advantage exists, report it.
Phase 10 failover is not started.

## Reproduce

```powershell
.venv\Scripts\python.exe -m ml.policy.binary_policy_dataset --output data/policy/policy_training_dataset_binary_v1
.venv\Scripts\python.exe -m ml.policy.binary_policy_dataset --validate data/policy/policy_training_dataset_binary_v1
```

The first command requires a new output directory and refuses overwrites. The
second reconstructs records and qualification from immutable source evidence.
Tests cover mapping, source reconstruction, reward determinism, ties, feature
availability, tampering, exclusion, simulation provenance and group isolation.

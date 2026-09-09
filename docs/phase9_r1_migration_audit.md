# Architecture R1 migration audit

Baseline: `4548d56`, clean working tree, origin/main. The entire external canonical
`Instruction/PROJECT_ARCHITECTURE.md` was read before edits. Python baseline:
`.venv\Scripts\python.exe -m pytest -q`: **255 passed**, four TensorFlow Lite
interpreter deprecation warnings, 61.67 seconds. No Python diagnostic failures.
The inventory and canonical before/after hashes are in
[migration_inventory.json](evidence/phase9_r1/migration_inventory.json).

Canonical status: Phase 7 split baselines completed; Phase 8 policy studies
completed; Phase 9 / M9 learned binary policy current; Phase 10 failover not started.
The older Phase 9–10.5 and M25–M34 documents retain their original study numbering.

## A — must migrate for R1 production

| Finding | Disposition |
|---|---|
| README claimed Phase 2 and learned split-point selection | Updated canonical status and binary production scope. |
| Repository canonical copy predates external R1 | Synchronized with the external source after the targeted cleanup below. |
| Canonical sections 12, 14, 18, 23, 25, 29, 31 still contained production Split Controller, embedding MQTT, EDGE/split dashboard or split manifest examples | Reconciled examples only; future examples explicitly do not claim deployment. |
| `config/experiment.yaml` illustrative Phase-0 weights differ from M30 | Marked historical, added binary production mapping; no weights silently changed. R1 reward must freeze M30 separately. |
| `ml/policy/train_split_controller.py`, `firmware/src/policy/split_controller.cpp` reserved implementation placeholders | Marked retired; retained files, no executable controller existed. |
| `ml/policy/train_meta.py`, `build_policy_dataset.py`, firmware `meta_learner.cpp` are placeholders | New versioned binary contract required; no trained policy or policy output tensor currently exists. |
| `policy_config_v1.json`, `formulation.py`, `benchmark.py` and their tests expose four actions | Preserve as historical studies; new v2 binary entry point must not import the old action enum as its production mapping. |
| `server/app/schemas.py` and `mqtt.py` currently require split/embedding | Keep as fixed-split service. R1 production needs a separately identified 10-feature CLOUD contract after deployment gates pass. |
| `server/app/cloud_full.py` says ALL_CLOUD in benchmark diagnostics | Valid full-cloud model implementation available for reuse. Historical action 3 must never be interpreted as production action 1 without explicit translation. No runtime change yet. |
| `docs/mqtt_protocol.md` was only a reserved stub | Documents production target versus existing fixed-split/historical campaign protocols. |
| Production `firmware/src/main.cpp` contains no policy/network execution | Validated local/MC runtime retained pending policy gates; not falsely reported as migrated adaptive firmware. |

No obsolete trained four-output policy, deployed split-controller binary, or
production OTA manifest was found. OTA modules are reserved placeholders.
Server health still reports the existing Phase 6 service milestone; that is not
evidence of a deployed R1 service. No database/dashboard/OTA implementation changed.

## B — retain as experimental fixed-split baselines

All split Keras/TFLite models, prefix headers/data, `prefix_runner.cpp`, cloud tails,
Phase 7 parity/E2E suites, legacy server split routing and their tests remain.
M29–M34 four-action analysis utilities, benchmark enums and campaign firmware remain
executable historical studies. Their old action 3 means ALL_CLOUD (10 features),
not Split3. `cloud_full.py` composes B3 with the existing cloud head continuation;
it is a genuine full-cloud path. No gesture model parameters changed.

## C — historical evidence; do not rewrite

| Study | Exact artifact families |
|---|---|
| M25 | `ml/policy/policy_config_v1.json`, `formulation.py`, `tools/policy_dataset/analyze_policy_dataset.py`, `docs/evidence/phase10_policy_dataset_analysis.json` |
| M29 | `tools/policy_dataset/reward_engine/`, `docs/phase10_1_reward_calibration.md` |
| M30 | `tools/policy_dataset/conditions/connected_lab_v1.json`, `run_matched_campaign.py`, `matched_dataset.py`, `validate_matched_dataset.py`, `data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02/` |
| M31 | `tools/network_conditions/`, `run_controlled_campaign.py`, `data/policy/policy_training_dataset_v2/controlled_network_v1/` |
| M32 | `analyze_split_sensitivity.py`, `docs/evidence/phase10_3_split_sensitivity.json` |
| M33 | `state_expansion.py`, `state_scenarios_v1.json`, `run_state_expansion_campaign.py`, `data/policy/policy_training_dataset_v3/` |
| M34 | `ml/policy/split_architecture_analysis.py`, `docs/phase10_5_split_revision_study.md` and its evidence |

The machine-readable inventory lists individual tracked policy artifacts. All old
datasets, optimal labels, reports, manifests and validation expectations stay
unchanged. The rejected/diagnostic M30 first run is not a new independent campaign.
The actual Split1 tail is 64→48→32→5; the old generic five-block split table is
historical architectural discussion, not permission to redesign that validated tail.

## D — unrelated / no change

Gesture classes, sampling, features-v1, scaler, dataset-v1, external dataset,
local B3/MC inference and models remain unchanged. DB, dashboard, EWC, OTA and
failover remain outside this migration. Dependency changes are unnecessary.

## Diagnostic and qualification concerns

Existing hardware diagnostic failures are recorded in
`docs/evidence/phase7/phase7_validation_summary.json`: internal tensor diagnostic
uses an incompatible TFLM introspection API; the INT8 logits diagnostic has a
known desktop/device class mismatch. These are separate from the six passing
Phase 7 split parity/E2E suites. No new hardware run or INT8 repair is claimed.

M30's isolated harness obtains local uncertainty first, then executes local
inference a second time for action 0. The required R1 LOCAL branch instead reuses
the first result. Qualification must test this action-cost semantic mismatch;
merely obtaining both labels from replay is not proof of production supervision.
The original outcomes and reward parameters must not be silently corrected.

## Canonical cleanup scope

The external source and repository copy now match. Only stale production examples
and annotations were reconciled: edge deployment list, v2 config reference,
10-feature CLOUD request/response, serialization, production action display,
experimental split DB annotation, future OTA manifest, policy tree/version names.
The initial R1 wording now distinguishes validated split execution from a learned
Split Controller that never existed. No historical measured report was rewritten.


## R1 reuse-LOCAL migration outcome

The original audit above is retained as baseline evidence. The cached M30 local
result was salvageable; new binary_r1_v2 records compare incremental costs under
binary-reward-experiment-v2-r1-reuse. Historical data and reward versions were not
rewritten. policy_config_v3.json and the explicit meta-policy-v1.0.0 deployment
manifest supersede the initial v2 deployment target. R1 production schema/server,
cached decision worker, learned artifact and versioned logging are now integrated.
All six fixed-split hardware suites were rerun and passed. Canonical results and
limits are in phase9_learned_binary_policy.md; M9 is closed as a controlled MVP.
The external canonical architecture itself required no additional revision.
Phase10 Failover, dashboard and OTA remain unstarted.

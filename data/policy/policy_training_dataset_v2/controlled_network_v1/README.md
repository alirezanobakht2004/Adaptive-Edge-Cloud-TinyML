# Controlled network replay v1 — simulated data

**All 96 records are simulated transformations of 24 accepted M30 samples.**
There are zero new physical measurements. Do not merge these into the canonical
measured M30 JSONL or count profile copies as independent source windows.

Each profile directory contains a v2 JSONL, configuration and validation report.
The explicit schema extension is `controlled-network-replay-v1`; use
`run_controlled_campaign.validate_output`, not the measured-only M30 validator.
The original v2 contract and measured records remain unchanged.

Required record fields are `dataset_version`, `schema_extension`, `sample_id`,
`source_sample_id`, `source_sha256`, `source_session`, `state`,
`network_condition`, `candidate_actions`, `reward`, `optimal_action`,
`optimal_actions`, `repeat_optimal_actions`, `repeats`, `training_enabled`.
The validator reconstructs the complete record from the hashed source sample
and explicit profile, rejecting extra/missing fields and altered types/values.

`network_condition` always has `type=controlled_network`, profile name,
`simulated=true`, application/version and injected parameters. Each candidate
has `metric_kind=simulated_from_measured_source`. Predictions and payload sizes
are retained from M30; invariance is assumed, not remeasured. All projected
latencies/rewards/energy proxies are scenario results, not hardware results.

| Profile | Simulated records | ALL_LOCAL labels | ALL_CLOUD labels |
|---|---:|---:|---:|
| baseline | 24 | 24 | 0 |
| high_latency | 24 | 24 | 0 |
| edge_loaded | 24 | 0 | 24 |
| cloud_favorable | 24 | 0 | 24 |

Each profile includes four repeated evaluations per action per source state:
96 candidate projections per action per profile, 1,536 across all profiles.
Neither split receives an optimal label. `diversity_report.json` marks the
training gate failed; synthetic variation does not establish measured diversity.

Source evidence remains in `../campaigns/connected_lab_run02/` and must be kept
available for validation. Each manifest hashes the source dataset and records
selected source identities, profile parameters, generation time and code version.
The source SHA is canonical JSON, independent of file whitespace.

Generate a new output directory from the repository root:

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.run_controlled_campaign --profile high_latency --samples 24 --output <new-directory>
```

See `docs/phase10_2_network_diversity.md` for formulas, assumptions and limits.

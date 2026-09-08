# policy_training_dataset_v2: M30 connected-lab pilot

The canonical JSONL contains **24 real matched decision samples**, with four
repetitions of all four actions: **384 executions**, 96 per action. Each window
uses one physical ESP32 pre-decision snapshot and sequential action execution
with live condition guards. This is bounded-condition matching, not simultaneous
identical physical states. The original v1 dataset is unchanged.

Action mapping: 0 ALL_LOCAL, 1 SPLIT1, 2 SPLIT2, 3 ALL_CLOUD. Full cloud takes
10 normalized features and executes the full cloud network; it is not Split3.

`metadata.json` identifies the canonical campaign, calibration, versions and
excluded diagnostic run. `schema_description.json` describes the v2 contract.
Embedded v1 records preserve their source version and placeholder rewards;
top-level v2 candidate measurements and labels are separate derived fields.

Canonical source: `campaigns/connected_lab_run02/`. Of 25 requested windows,
one (124) was rejected for RTT drift of 20.258 ms, exceeding the configured
20 ms limit. Its 16 executions remain in raw/rejected evidence. No limit was
relaxed to accept it. The accepted set has 24 distinct validation windows from
one acquisition session, with class counts 5/4/5/5/5 (class IDs 0 through 4).

`campaigns/connected_lab_run01/` contains 25 diagnostic samples excluded from
the canonical dataset because the measurement run overlapped host pytest.
Do not concatenate it with the canonical samples or double-count the canonical
JSONL and its identical campaign copy.

Reward weights are provisional: correctness 1.0; latency, payload and estimated
energy proxy 0.1 each. Scales are 100 ms, 1,024 bytes and one proxy unit.
The proxy is `(preprocessing_ms + state_ms + selected_edge_compute_ms)/10 +
selected_payload_bytes/1024`. It is dimensionless, estimated only, and excludes
radio waiting, cloud energy and probe traffic. It is not measured energy.

All 24 optimal-action labels are ALL_LOCAL. All four repeats agree per sample.
`reward_sensitivity_report.json` records offline sensitivity checks, not new
measurements or fitted weights. `dataset_report.json` reports actual costs and
revalidates source labels, device traces, server responses and M29 calculations.

Revalidate from the repository root:

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.validate_matched_dataset --directory data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02 --output data/policy/policy_training_dataset_v2/campaigns/connected_lab_run02/dataset_report.json
```

The root JSONL and report are exact copies of the canonical campaign files.
The regression test checks this equality. Retain all raw/server evidence and
configuration files alongside the dataset for reproducibility.

**Policy training remains blocked.** This pilot validates the collection/labeling
path, but has one optimal-action class, one device/session/network condition,
few independent windows and provisional reward/energy assumptions. No learned
policy or training was implemented.

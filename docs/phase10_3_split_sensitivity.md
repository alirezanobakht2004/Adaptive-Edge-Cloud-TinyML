# Phase10.3 M32 — split sensitivity analysis

**Split1 and Split2 currently lose because another action dominates their
observed accuracy/cost tradeoff, not merely because of one unlucky reward
weight setting.** In every accepted M30 state and every M31 scenario, each
split has at least one alternative that is equally accurate, no more costly
on every reward dimension, and strictly cheaper on at least one dimension.
No nonnegative linear reweighting can make either split a unique optimum on
these fixed observations. Zeroing the distinguishing costs can produce ties.

M32 analysis is complete. No policy was trained, weights changed, model modified,
new inference executed or production runtime changed. Training remains blocked.

## Evidence and method

The reproducible utility is
`tools/policy_dataset/analyze_split_sensitivity.py`; the full machine-readable
report is `docs/evidence/phase10_3_split_sensitivity.json`.

The utility validates and analyzes:

- M30: 24 accepted real matched windows, four repeats per action, 384 device
  executions. Timing/payloads are measured; energy is an estimated proxy.
- M31: four separately reported simulated profiles, 24 records each. They
  reuse those same M30 windows and predictions; they are not new observations.

The rejected M30 window, diagnostic run and old unmatched v1 runs are excluded
from four-action comparisons. There are **24 independent source windows**, not
120 independent states after adding the real and simulated cohorts. Even those
24 windows come from one acquisition session/device and connected condition.
All retained actions are correct on the accepted subset. M31 preserves that
correctness by assumption, so it supplies no new evidence about accuracy.

For each state, normalized costs and rewards are averaged across its repeats.
Cohort means then weight each source window equally. Analysis includes:

1. Latency accounting, payload sizes, proxy values and weighted reward terms.
2. A 125-point grid: accuracy weight 1, each cost weight in
   `{0, 0.01, 0.1, 1, 10}`, with original normalization scales unchanged.
3. Per-state Pareto dominance checks.
4. A small linear optimization, solved by enumerating feasible vertices:
   maximize a split's minimum reward margin over all other actions, with
   nonnegative weights summing to one. This checks beyond the finite grid.
   A margin above `1e-8` is classified as a possible unique optimum; margins
   near zero are numerical ties. Feasibility tolerance is `1e-9`.
5. Pairwise break-even changes to one target metric while holding others fixed.

Weight witnesses and thresholds are diagnostics, not recommended settings,
fitted policy parameters, statistical confidence intervals or physical forecasts.
Pareto dominance provides an independent explanation for the zero-margin
optimization results. The optimizer also has tests with a synthetic genuine
split tradeoff and a nondominated but unsupported linear-reward point.

## 1. Current reward configuration impact

The unchanged provisional reward is:

```text
R = 1.0 * correctness
    - 0.1 * latency_ms / 100
    - 0.1 * payload_bytes / 1024
    - 0.1 * energy_proxy

energy_proxy = accounted_edge_compute_ms / 10 + payload_bytes / 1024
```

The proxy includes common preprocessing/state computation and selected edge
computation. It excludes radio waiting, probe traffic and cloud energy. It is
estimated for M30 and simulated for M31, never measured battery energy or joules.

Real M30 mean reward components (cost columns are positive penalties):

| Action | Accuracy contribution | Latency penalty | Payload penalty | Proxy penalty | Reward |
|---|---:|---:|---:|---:|---:|
| ALL_LOCAL | 1.000000 | 0.016063 | 0.000000 | 0.034105 | 0.949831 |
| SPLIT1 | 1.000000 | 0.036320 | 0.074791 | 0.101271 | 0.787617 |
| SPLIT2 | 1.000000 | 0.038642 | 0.064819 | 0.098936 | 0.797603 |
| ALL_CLOUD | 1.000000 | 0.038745 | 0.038278 | 0.054397 | 0.868580 |

Accuracy contributes the same constant to every action; increasing its weight
does not change rankings. Accuracy-only weighting ties all actions. In these
records, an already-perfect split cannot improve its own accuracy further.

Expanding the proxy reveals the effective penalty:

```text
0.001 * latency_ms + 0.2 * payload_bytes / 1024
+ 0.01 * accounted_edge_compute_ms
```

Communication is penalized both directly and through the proxy. A millisecond
of accounted edge compute incurs a proxy penalty ten times the explicit
penalty for a millisecond of end-to-end latency. This is an uncalibrated design
choice, not a measured energy relationship. It helps explain why the simulated
edge slowdown favors ALL_CLOUD, whose selected model compute is zero on the edge.
Common state-compute overhead affects absolute rewards but cancels between
actions within a matched state.

## 2. Communication penalty analysis

| Action | Tensor sent | Raw float32 tensor bytes | Mean selected JSON request + response bytes |
|---|---|---:|---:|
| ALL_LOCAL | None | 0 | 0 |
| SPLIT1 | 64-value embedding | 256 | 765.8646 |
| SPLIT2 | 48-value embedding | 192 | 663.7500 |
| ALL_CLOUD | 10 normalized features | 40 | 391.9688 |

ALL_CLOUD sends extracted features, not a raw IMU window. The split boundary
expands that feature representation. Split1 and Split2 therefore have about
1.95 and 1.69 times the selected payload of ALL_CLOUD in this JSON protocol.
Byte counts include response metadata, not just tensors; probe traffic and
MQTT/TCP/Wi-Fi headers are excluded consistently with M30.

There is no current communication advantage for either split. Increasing a
positive byte penalty cannot create one. Setting the direct communication
weight to zero still leaves the proxy's payload penalty. Even removing both
payload-related penalties cannot make a dominated split uniquely best on the
retained states. Slower transport does not inherently help an embedding that
is larger than the full-cloud input.

No protocol, encoding, split model or full-cloud definition was changed. Any
future compression/boundary proposal needs a separate authorized experiment;
this analysis does not assume those hypothetical savings already exist.

## 3. Latency penalty analysis

Real M30 mean latency accounting, milliseconds:

| Action | Shared decision overhead | Selected edge compute | Server compute | Selected transport/scheduling residual | Total |
|---|---:|---:|---:|---:|---:|
| ALL_LOCAL | 14.2648 | 1.7987 | 0 | 0 | 16.0634 |
| SPLIT1 | 14.2648 | 1.0361 | 5.6491 | 15.3701 | 36.3201 |
| SPLIT2 | 14.2648 | 1.7998 | 7.1437 | 15.4333 | 38.6415 |
| ALL_CLOUD | 14.2648 | 0 | 10.5634 | 13.9169 | 38.7451 |

Shared overhead comprises 0.0207 ms preprocessing, 1.5911 ms state inference
and 12.6529 ms probe/scheduling residual. Residuals are accounting differences,
not pure wire latency or causal attribution to a particular component.
Totals exclude experimental guard probes and between-action/serial overhead.

Split1 reduces mean server compute relative to ALL_CLOUD, but incurs edge-prefix
compute and higher residual overhead. Its total advantage over ALL_CLOUD is
only **2.4250 ms**. Split2's total advantage is only **0.1035 ms**. Both are
substantially slower than ALL_LOCAL. These are small-sample means, not claims
of statistically significant speedups or deployment latency.

The isolated prefix harness includes interpreter/tensor allocation in prefix
execution. Timing also reflects different existing local/cloud heads, host
scheduling and JSON transport. It must not be treated as a pure neural FLOP
comparison or a benchmark of a redesigned production runtime.

Holding all other mean costs fixed, the current reward requires Split1 and
Split2 to be respectively **83.388 ms** and **71.081 ms** faster than ALL_CLOUD
to offset their extra byte/proxy penalties. Their actual advantages are far
smaller. The corresponding target total latencies would be negative:

| Target | Total latency needed to tie ALL_LOCAL | Total latency needed to tie ALL_CLOUD |
|---|---:|---:|
| SPLIT1 | -125.894 ms | -44.643 ms |
| SPLIT2 | -113.587 ms | -32.335 ms |

Thus reducing latency alone, while keeping current payload and proxy values,
cannot close the real-cohort mean reward gap. A genuine compute change could
also change the proxy; these single-variable thresholds deliberately do not
model that coupled change.

## Reward sensitivity across profiles

| Cohort | Mean reward LOCAL | SPLIT1 | SPLIT2 | CLOUD | Dominator present for both splits in every state |
|---|---:|---:|---:|---:|---|
| Real M30 | 0.949831 | 0.787617 | 0.797603 | 0.868580 | ALL_LOCAL |
| Simulated baseline | 0.949831 | 0.787617 | 0.797603 | 0.868580 | ALL_LOCAL |
| Simulated high_latency | 0.799831 | 0.487617 | 0.497603 | 0.568580 | ALL_LOCAL |
| Simulated edge_loaded | 0.237032 | 0.234194 | 0.084567 | 0.531707 | ALL_CLOUD |
| Simulated cloud_favorable | 0.696711 | 0.607566 | 0.559542 | 0.769678 | ALL_CLOUD |

Full per-action simulated latency decompositions, payload/proxy contributions
and per-state results are in the JSON report. M31 high_latency adds a shared
probe delay and another remote delay, hurting offload relative to local.
Edge-loaded profiles increase prefix work as well as local work; full cloud
avoids both selected edge prefixes. M31 also holds all predictions correct.
These transformations do not create a split-specific advantage.

For **both splits in each cohort**, all 24 states are Pareto dominated; zero
states have a positive optimized weight margin. Neither split uniquely wins
any state in any of the 125 grid configurations. This is stronger than a
failure under the default weights. Zero cost weights may tie dominated actions,
but a tie does not establish a split advantage or a justified unique label.

## 4. Conditions needed for split actions to compete

For split s to beat every alternative j, actual condition-dependent values
must satisfy all the following pairwise inequalities:

```text
w_accuracy*(accuracy_s - accuracy_j)
> w_latency*(latency_s - latency_j)/L_ref
  + w_comm*(bytes_s - bytes_j)/B_ref
  + w_energy*(proxy_s - proxy_j)/E_ref
```

Plausible research questions to test, not established results:

- Can harder windows expose an accuracy advantage of a split's existing tail
  over the local head while preserving acceptable offload costs? Comparisons
  must also include full cloud, not just the local baseline.
- Can an edge/cloud compute imbalance make the early edge blocks inexpensive
  while full-cloud early-block processing is costly enough to offset the
  larger embedding? Measure each action's server compute; merely adding the
  same queue delay to every offload is not split-specific evidence.
- Are there real resource or deadline conditions where local inference is
  infeasible and a split remains feasible while full cloud is less attractive?
  Feasibility restrictions need measured evidence and explicit versioned rules,
  not removal of competitors solely to force a desired label.

As a scale illustration, keeping current mean costs and a split's correctness
at 100%, Split1 would need other conditions where local correctness is lower
by more than **16.22 percentage points**, and full-cloud correctness lower by
more than **8.10 points**, to overcome each corresponding cost gap. Split2's
thresholds are **15.22** and **7.10 points**. These are algebraic examples, not
predicted error rates. All current actions are correct, and all competitors
must be beaten simultaneously. Nothing here shows such states exist for the
current models.

## 5. Calibration versus dataset expansion

**Both deserve work, but calibration alone cannot repair current dominance.**
Calibrate reward scales/priorities and audit proxy overlap for a defensible
application objective. Do not tune weights merely to manufacture split labels,
use negative cost penalties, or call an accuracy-only tie a split win.
The framework does not require claiming that every implemented action must
be optimal somewhere; an honestly dominated action may remain unused.

Expand physically measured matched conditions first: independent sessions and
more challenging windows, real edge/cloud compute variation, network variation,
and explicitly accounted failure/resource conditions. Verify the actual
accuracy/cost frontier before producing more scenario labels. Group all copies
and repeats of a source window together in future policy partitions.
Existing M31 replicas do not supply independent error or device evidence.

No model, payload protocol, split boundary or reward config was changed in
M32. Architecture changes, if eventually justified by evidence, require a
separate scope. **Policy training remains blocked.**

## Reproduction and validation

```powershell
.venv\Scripts\python.exe -m tools.policy_dataset.analyze_split_sensitivity
.venv\Scripts\python.exe -m py_compile tools/policy_dataset/analyze_split_sensitivity.py tests/test_split_sensitivity.py
.venv\Scripts\python.exe -m pytest -q
git diff --check
```

**239 tests passed**, with four existing TensorFlow Lite deprecation warnings.
Compilation and whitespace checks passed. Tests cover source preservation,
action validation, deterministic report reconstruction, component accounting,
break-even equality, zero-weight cases, ties, dominance and positive/negative
examples for the weight-feasibility optimization. Existing NumPy is reused;
no dependency was added. No ESP32 tests/flash or new hardware measurements
were needed or performed.

# Phase10.5 M34 — candidate split architecture study

**None of the requested dimensions establishes a competitive split by itself.**
The existing 32-dimensional B3 boundary is the lowest-risk option for a future
matched measurement study because its artifacts already exist. A 16-dimensional
bottleneck is the smallest requested payload, but needs new parameters and an
unknown accuracy tradeoff. No production architecture revision is justified by
the present estimates alone. Policy training remains blocked.

This milestone is analysis only: no model was created, trained or retrained;
no inference runtime was modified; no firmware was deployed.

## M32/M33 motivation

M32 found both current splits dominated in every accepted measured state and
simulated profile. Nonnegative reweighting could not make them unique winners.
The communication term is also present in the energy proxy, strengthening the
cost of sending expanded embeddings. M33 added explicit pressure/availability
state but still produced no split optimum: each split was dominated in all
48 states where it was feasible. Scenario expectations were not forced labels.

The current workload begins with **10 already-extracted features**, not a raw
IMU stream or large image tensor. ALL_CLOUD transmits those 10 features. Split1
and Split2 expand them to 64 and 48 values; thus they do not currently save
communication relative to full cloud. Server-compute savings have not offset
edge-prefix, transport and proxy costs on the retained observations.

## Artifact-backed architecture inspection

`ml/policy/split_architecture_analysis.py` reads the saved `.keras` graph
metadata directly from its archive; it does not instantiate or fit a model.
It records artifact SHA-256 hashes/sizes, verifies the prefix export-report
hashes, and checks the expected Dense input/output dimensions. The complete
output is `docs/evidence/phase10_5_candidate_comparison.json`.

The actual graphs are:

| Artifact/path | Dense sequence |
|---|---|
| Source gesture model | 10 -> 64 -> 48 -> 32 -> local 5-class head |
| Split1 cloud tail | 64 -> 48 -> 32 -> 5 |
| Split2 cloud tail | 48 -> 32 -> 64 -> 32 -> 5 |
| Split3 cloud tail | 32 -> 64 -> 32 -> 5 |
| Full-cloud composition | Source B1/B2/B3 followed by Split3 tail |

Normalization/dropout are present in the source graph but are not Dense MACs.
The Split1 tail is a shorter continuation than the B1–B5 full-cloud graph.
This matters: different tails must not be assumed to have identical accuracy
merely because their output is five classes.

The 48-dimensional boundary is existing B2/Split2. The 32-dimensional boundary
is existing B3/Split3. Neither requires inventing a new width. There is no saved
16- or 24-dimensional boundary. Policy action 3 remains **ALL_CLOUD**; inspecting
the neural Split3 boundary does not rename or replace that policy action.

## Candidate placement and compute estimates

For 16/24 dimensions, the explicit hypothetical construction is a **linear
encoder after B1**, `64 -> d`, followed by a cloud decoder, `d -> 64`, and the
existing Split1 tail. This defines a concrete compute estimate without pretending
that an arbitrary truncation is a valid model. Encoder/decoder weights do not
exist; reduced-rank reconstruction cannot be assumed lossless.

For 32/48, use the already-existing B3/B2 boundaries respectively.

| Candidate width | Placement | Edge Dense MACs | Cloud Dense MACs | Total Dense MACs | New parameters needed? |
|---|---|---:|---:|---:|---|
| 16 | B1 + 64->16 encoder; cloud 16->64 decoder + Split1 tail | 1,664 | 5,792 | 7,456 | Yes |
| 24 | B1 + 64->24 encoder; cloud 24->64 decoder + Split1 tail | 2,176 | 6,304 | 8,480 | Yes |
| 32 | Existing B1–B3 / Split3 tail | 5,248 | 4,256 | 9,504 | No |
| 48 | Existing B1–B2 / Split2 tail; control | 3,712 | 5,792 | 9,504 | No |

Dense MACs are input width times output width per layer. B1/B2/B3 respectively
require 640, 3,072 and 1,536 MACs. Current Split1 uses 640 edge MACs and 4,768
tail MACs. The proposed 16/24 bottlenecks therefore **add** 2,048/3,072 MACs
across the pair; narrower transmission is not free compute reduction. Full
cloud's B1–B5/head composition has 9,504 Dense MACs on the server.

These counts exclude biases, activation functions, normalization, stochastic
uncertainty passes, allocation, serialization, scheduling and transport. They
are **not latency or energy measurements**. No MAC-to-millisecond conversion
is used. The existing 32/48 topologies move computation between devices; they
do not imply identical weights to every alternative tail or known placement
latencies under matched conditions.

## Embedding bytes and communication estimates

With float32, raw embedding size is exactly four bytes per value:

| Width | Raw tensor bytes | Raw bytes / ALL_CLOUD's 40-byte feature tensor | Estimated JSON request + response bytes |
|---|---:|---:|---:|
| 16 | 64 | 1.6x | 459.521 |
| 24 | 96 | 2.4x | 510.578 |
| 32 | 128 | 3.2x | 561.635 |
| 48 | 192 | 4.8x | 663.750 |

Reference: current Split1 is 256 raw tensor bytes and 765.865 measured mean
selected JSON bytes; current Split2 is 192 and 663.750. ALL_CLOUD is 40 and
391.969. Measured means come from the 24 accepted M30 windows, with four repeats
per action; rejected/diagnostic windows and simulated copies are not pooled in.

The JSON column is a **descriptive two-point extrapolation**, not serialization
of candidate embeddings:

```text
estimated_selected_payload(d) = 357.40625 + 6.3821614583 * d bytes
```

It fits the observed 64- and 48-dimensional request-plus-response means. The
intercept absorbs metadata/response overhead and differences between those two
sources; it is not an independently measured constant header size. Numeric
formatting, zeros/sparsity, changed decoder outputs and request/response metadata
can change real JSON sizes. Extrapolation below 48 dimensions is unvalidated.

All requested candidates exceed the 10-feature input in exact raw float32
bytes, even though they are smaller than current Split1. With equal precision
and framing, strictly fewer raw bytes than the 10-feature tensor would require
fewer than 10 values, outside the requested candidate set. Actual JSON size
also depends on value formatting, so raw byte ratios are not guarantees about
every possible JSON message. No quantization or protocol change was assumed.

## Conditional reward impact

The study keeps M30's provisional weights/scales unchanged. Because payload
appears both directly and in the proxy, changing bytes by `delta_B` changes
reward by:

```text
delta_R = -delta_B * (w_comm/B_ref + w_energy/(E_ref*proxy_byte_reference))
        = -delta_B * 0.2/1024
```

The following overlay changes **only modeled payload**, holding the existing
Split1 accuracy, latency and accounted edge compute fixed for every row. It
isolates communication sensitivity; it is not a forecast of the placement's
actual reward and omits its added/moved compute and possible accuracy changes.

| Width | Estimated direct payload penalty | Payload-only reward gain vs Split1 | Conditional reward |
|---|---:|---:|---:|
| 16 | 0.044875 | +0.059833 | 0.847450 |
| 24 | 0.049861 | +0.049861 | 0.837478 |
| 32 | 0.054847 | +0.039889 | 0.827506 |
| 48 | 0.064819 | +0.019944 | 0.807562 |

Current measured mean rewards are LOCAL 0.949831, Split1 0.787617, Split2
0.797603 and CLOUD 0.868580. Even the smallest requested dimension does not
bridge the mean cloud or local reward gap under this isolated payload estimate.
For 48 dimensions, the conditional reward differs from actual Split2 because
the overlay deliberately retains Split1 compute/latency; do not confuse them.

Candidate measured accuracy, measured latency and actual candidate reward are
all **null** in the report. The estimated energy effect here is only the payload
term of a dimensionless proxy; no physical energy or battery claim is made.
M32's proxy overlap remains a calibration concern rather than a reason to
change weights to force candidate success.

## Recommendation

1. **Do not revise production models yet.** Dimensions alone do not establish
   accuracy, feasible edge cost, a latency advantage or a non-dominated action.
2. Prioritize a future **matched 32-dimensional B3 study** using existing
   validated artifacts. It is the lowest model-development risk, not a proven
   winner. Measure its edge/transport/tail costs alongside true full cloud and
   local inference. Keep policy action mapping explicit; do not alias ALL_CLOUD
   to the neural Split3 route. No such campaign or firmware deployment ran here.
3. Treat 48 dimensions as the existing control, not a novel revision. Moving
   back to the same B2 boundary does not address M32/M33 dominance.
4. If a separately authorized bottleneck study is justified, 16 dimensions is
   the smallest requested candidate; 24 is an intermediate quality/size option.
   Evaluate reconstruction/task accuracy and total cost with independent data
   before considering adoption. New encoder/decoder parameters would need to
   be derived/trained and validated in isolation; no final model was retrained.
5. Continue calibration and physical state expansion without forcing splits.
   If no candidate offers a genuine tradeoff, document that result rather than
   assume every split action must become optimal. Do not train the policy yet.

## Reproduction and validation

```powershell
.venv\Scripts\python.exe -m ml.policy.split_architecture_analysis
.venv\Scripts\python.exe -m py_compile ml/policy/split_architecture_analysis.py tests/test_split_architecture_analysis.py
.venv\Scripts\python.exe -m pytest -q
git diff --check
```

The utility accepts `--model-root` for the existing artifact directory and
`--output` for the comparison JSON. It fails if the saved Dense architecture
or prefix/report hashes no longer match its explicit assumptions.

**255 tests passed**, with four existing TensorFlow Lite deprecation warnings.
Compilation and whitespace checks passed. Tests cover saved graph inspection,
native versus hypothetical boundaries, MAC accounting, exact float32 bytes,
payload extrapolation, direct-plus-proxy reward arithmetic, unsupported input
rejection and deterministic report reconstruction with unknown candidate
measurements left null. No firmware test, flash, model export, deployment,
retraining or policy training was performed.

# Development LOCAL/CLOUD oracle audit

Existing LOCAL: gesture-model-v1.1.0 with the validated five-pass MC masking
algorithm. Existing CLOUD: gesture-full-cloud-v1.0.0 (frozen B3 plus cloud tail).
Only TRAIN/session_01 and VALIDATION/session_02 were loaded. Final TEST was not
used for discovery, fitting, selection or thresholds.

| Pool | Windows | Both correct (A) | LOCAL only correct (B) | CLOUD only correct (C) | Both wrong (D) |
|---|---:|---:|---:|---:|---:|
| Train | 600 | 592 | 0 | 1 | 7 |
| Validation | 200 | 197 | 0 | 0 | 3 |

Local/cloud training accuracy is 98.6667%/98.8333%; validation is 98.5%/98.5%.
The single training disagreement is SWIPE_RIGHT. There are no validation
disagreements: both models miss one SWIPE_LEFT, one SWIPE_RIGHT and one ROTATE_CW.
IDLE and SHAKE validation predictions are all correct. Class counts and every
window's predictions, confidence, entropy and margin are in
`docs/evidence/phase9_local_cloud_oracle_audit.json`.

This is desktop inference, not a new hardware or final test result. Explicit masks
use the firmware xorshift32 sequence with seed 42000 + window index. On the 24 M30
source windows, action predictions match retained hardware state, and uncertainty
differences pass the existing 2e-5 prefix-scale convention. Production PRNG seeds
vary, so this audit does not characterize every possible dropout realization.

## Bounded decision

There is no useful validation CLOUD-only correctness set in this audit. Follow
Case B: one server-only experiment, with no new data collection or architecture
search. The existing cloud training froze B1–B3, which were optimized for the
local exit. Fine-tuning a private full-cloud copy end to end removes that limitation
without changing architecture, features, edge weights or split artifacts.

Before fitting, freeze: seed 42, Adam 1e-4, batch 32, at most 200 epochs, early
stopping on validation loss with patience 20 and best-weight restoration. Fit
TRAIN only. Keep the old model unless the candidate produces at least one
LOCAL-wrong/CLOUD-correct validation case and does not reduce validation accuracy.
This acceptance rule is declared before the experiment; it is not a policy
threshold and does not use a policy holdout or final TEST partition.

Server resources motivate assistance; they do not establish measured superiority.
The canonical Phase 9 report records the eventual bounded-experiment outcome.

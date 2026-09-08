# Derived binary comparison dataset v1

Contains historical LOCAL/CLOUD candidate comparisons, not approved R1 training targets.
See manifest.json and qualification.json for immutable sources, exclusions and the
LOCAL re-execution versus result-reuse blocker. Historical datasets are unchanged.
All replays of an original window share a partition. Energy is a proxy only.

Validate: `python -m ml.policy.binary_policy_dataset --validate data/policy/policy_training_dataset_binary_v1`

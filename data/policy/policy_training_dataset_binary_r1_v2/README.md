# R1 reuse-LOCAL policy candidate dataset v2

2424 records from 800 development windows. Includes 24 salvaged physical M30
records and 2400 explicitly simulated network/resource assignments with actual
frozen-model predictions. LOCAL has no second inference. Historical data remain
unchanged. Both CLOUD labels derive from ONE source window, not two independent
observations. The 200-group holdout contains no CLOUD-optimal support; positive
recall/generalization cannot be estimated there.

Validate: `python -m ml.policy.r1_dataset --validate`.
See manifest.json for frozen reward, inputs, source hashes and controlled profiles.
No final TEST partition is used. Energy is an estimated/simulated proxy only.

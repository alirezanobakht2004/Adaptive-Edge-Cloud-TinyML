# Experiment Naming Convention

Do not write experiment outputs into ad-hoc folders.

Use:

```text
experiments/<strategy>/<YYYYMMDD>_<short-name>/
```

Examples:

```text
experiments/all_local/20260915_baseline-v1/
experiments/fixed_split/20260918_split2-v1/
experiments/adaptive/20261001_policy-v1/
```

Every final experiment should record:
- dataset version
- feature version
- model version
- policy version
- firmware version
- hardware settings
- measured metrics only

# Isolated external motion dataset

Source: [Dilhara Jayawardhane's Kaggle dataset](https://www.kaggle.com/datasets/dilharajayawardhane/6-axis-motion-gesture-dataset-hand-waves-and-flicks).
Keep the supplied `gesture_dataset.csv` here. Its SHA-256 is recorded in
`source_context.json`. Neither this raw CSV nor generated numerical NPZ files is
committed; scoped ignore rules keep them local. The original CSV is never edited.

`converted/recordings.npz` holds lossless numeric arrays, original labels/IDs and
recording offsets; `external_dataset_metadata.json` describes them. The separate
diagnostic feature archive uses an explicit interpolation/endpoint-hold adapter.
`evaluation_probabilities.npz` contains frozen-model predictions for both cohorts.
None of these files belongs to dataset-v1 or the policy training dataset.

`external_label_mapping.json` maps only broad idle semantics. Waves and flicks
remain unmapped. Mounting/orientation is unknown, and production contracts are
not met natively. Read the [decision and reproduction guide](../../../docs/phase9_5_external_dataset_evaluation.md)
before interpreting the results. Applicable source-version terms and capture
provenance remain unresolved; `source_context.json` records those limitations.

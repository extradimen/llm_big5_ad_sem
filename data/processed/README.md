# Processed Data

Store the final, analysis-ready CSV files used by the SEM analyses here.

Recommended primary filename:

- `merged_dataset.csv` — the complete dataset used to reproduce the reported
  SEM results.

If analyses use separate datasets by model, advertisement type, or region, use
descriptive filenames and document them in this file. For every released CSV,
record its source files, generating script or notebook, inclusion criteria, and
the analysis that consumes it.

The existing pipeline currently produces intermediate merged CSV files in
`csv_output_merged/`. Copy only the final analysis dataset(s) into this directory
when preparing a reproducible release.

# Data

This directory is the canonical location for datasets prepared for analysis and
public release.

- `raw/` is reserved for source data that have not been transformed.
- `processed/` contains the analysis-ready CSV files used by the current
  observed-variable path analysis.

The existing `response_output/`, `json_output/`, `csv_output/`, and
`csv_output_merged/` directories are retained for compatibility with the current
data-generation pipeline. The documented release datasets in `processed/` are
rebuilt by `python scripts/run_all.py`.

Do not add confidential information, API credentials, or direct personal
identifiers. Each released dataset should be accompanied by enough information
to identify its generating script, processing steps, and role in the analysis.

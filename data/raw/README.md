# Raw Data

This directory is reserved for immutable source datasets used to build the
analysis-ready files.

The current LLM response pipeline stores its source artifacts in
`response_output/` and `json_output/`. Those paths remain unchanged so the
existing scripts and notebooks continue to work. Only place files here when a
release needs a curated raw-data snapshot.

Do not edit raw files after adding them. Record any cleaning or transformation
in a script or notebook and write the result to `../processed/`.

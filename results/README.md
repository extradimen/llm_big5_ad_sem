# Results

This directory is the canonical location for publication-ready analysis
outputs. Generated files are reproducible from the historical CSV inputs by
running `python scripts/run_all.py` from the repository root.

- `sem/` retains the historical directory name and contains machine-readable
  estimates from the observed-variable path analyses.
- `figures/` contains publication-ready figures.
- `tables/` contains publication-ready tables in reusable formats.
- `quality/` contains missingness, range, provenance, and JSON-key audit
  outputs used in the revised manuscript.

The existing `sem_outputs/` and `sem_group_outputs/` directories are retained
as historical artifacts. They are not the source of the primary
results.

The R3 release adds item-level advertising-attitude diagnostics, indirect
effects for the additional two-item measurement experiment, run-specific
stability estimates, and a post-hoc recovery check for purchase-intention
arrays emitted under noncanonical JSON keys.

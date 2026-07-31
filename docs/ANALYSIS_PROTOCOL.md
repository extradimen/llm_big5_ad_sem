# Revised Analysis Protocol

## Purpose

This protocol defines the reviewer-facing, reproducible analysis. It does not
overwrite the historical notebooks or their outputs.

## Primary analysis population

The primary analysis reconstructs the Chinese, promotion-framed DeepSeek-V2
236B data directly from the ten per-run files in `csv_output/`. The preparation
script preserves the run number, source filename, original sample identifier,
and a new globally unique `response_id`.

Each row represents one simulated response instance, not one human participant.
Repeated `sample_id` values across runs are expected and are not presented as
unique individuals.

## Construct operationalization

The fifteen BFI values were randomized before prompting the LLM and were already
reverse-scored by `2_json_to_csv.py`. They are experimental persona-conditioning
inputs, not measurements elicited from a respondent. The revised model therefore
uses the mean of the three inputs for each Big Five dimension as an observed
condition score. Cronbach's alpha for these inputs is reported only as a
diagnostic and is not used to claim psychometric reliability.

The original four ad-attitude items and three purchase-intention items show weak
overall internal consistency in the historical primary dataset. To avoid
data-dependent item selection, the primary analysis retains all pre-specified
items. A measurement-screened sensitivity analysis uses the coherent item pairs:

- ad attitude: `ad_att_1` (like) and `ad_att_2` (interesting);
- purchase intention: `intent_2` (try to buy) and `intent_3` (consider buying).

The sensitivity analysis is explicitly post-hoc and must not replace the primary
result without external validation. The low reliability of the original full
scales must be disclosed in any manuscript that uses the revised outputs.

## Statistical models

Two standardized ordinary least-squares path equations are fitted:

1. all-item ad attitude on the five persona-condition scores and run fixed effects;
2. all-item purchase intention on all-item ad attitude, the five persona-condition
   scores, and run fixed effects.

HC3 heteroskedasticity-consistent standard errors and 95% confidence intervals
are reported. Indirect effects are calculated as the product of standardized
paths and evaluated with 2,000 run-stratified bootstrap resamples using a fixed
seed. Rows missing variables required by a particular model are excluded from
that model; raw and complete-case counts are reported.

## Exploratory configuration analysis

The historical four-group files confound cultural condition with LLM backbone.
They are therefore analyzed only as culture–model configurations. The outputs
must not be interpreted as isolated cultural effects or human-population
estimates. The old multi-group invariance results remain available as historical
artifacts but are not part of the revised primary analysis.

## Reproduction

From the repository root:

```bash
python scripts/run_all.py
```

The command rebuilds processed datasets, validation reports, statistical tables,
bootstrap results, figures, and the generated analysis summary.

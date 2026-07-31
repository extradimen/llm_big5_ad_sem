# Data Dictionary for Revised Analysis

## Identifiers and provenance

| Variable | Meaning |
|---|---|
| `response_id` | Globally unique identifier created by the preparation script |
| `sample_id` | Historical within-run identifier; not a unique person |
| `run` | Generation run parsed from the per-run filename |
| `source_file` | Historical CSV from which the row originated |
| `model` | LLM backbone used for the primary response |
| `region` | Cultural profile label supplied in the prompt |
| `ad_type` | Advertisement framing condition |

## Persona-conditioning inputs

`BFI_11` through `BFI_53` are 1–5 values. The third item in each trait
(`BFI_13`, `BFI_23`, `BFI_33`, `BFI_43`, `BFI_53`) was reverse-scored during
JSON-to-CSV conversion. The derived variables `extraversion`, `agreeableness`,
`conscientiousness`, `neuroticism`, and `openness` are arithmetic means of their
three associated inputs.

These variables describe randomized persona conditions. They must not be
described as observed human personality measurements.

## LLM-generated responses

| Variables | Scale | Construct |
|---|---:|---|
| `ad_att_1`–`ad_att_4` | 1–7 | Ad evaluation items |
| `intent_1`–`intent_3` | 1–7 | Purchase-intention items |
| `ad_attitude_core` | 1–7 | Mean of `ad_att_1`, `ad_att_2` |
| `purchase_intention_core` | 1–7 | Mean of `intent_2`, `intent_3` |
| `ad_attitude_all_items` | 1–7 | Mean of all four ad items; prespecified primary analysis |
| `purchase_intention_all_items` | 1–7 | Mean of all three intention items; prespecified primary analysis |

The two `*_core` composites are used only in the post-hoc measurement-screened
sensitivity analysis. They do not replace the all-item primary specification.

Original missing values are retained. Numeric values outside the documented scale
range are reported in validation tables and converted to missing in processed
analysis data. Models use complete-case selection for their required variables.

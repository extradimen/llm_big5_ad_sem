# Revised analysis summary

This report is generated automatically by `scripts/run_revised_analysis.py`.

## Analysis population

- Source: ten per-run DeepSeek-V2 236B CSV files.
- Cultural condition: Chinese.
- Advertisement condition: promotion-framed.
- Raw simulation responses: 4,987.
- Complete cases for the primary path analysis: 4,695.

## Measurement decision

The Big Five inputs are randomized persona-conditioning values rather than responses
to a reflective psychometric instrument. They are therefore summarized as observed
condition scores, not fitted as latent variables. To avoid post-hoc item selection,
the primary analysis uses all four pre-specified ad-attitude items (alpha =
0.251) and all three pre-specified purchase-intention
items (alpha = 0.630). A measurement-screened
two-item analysis is reported separately as a sensitivity analysis. The low alpha for
the full ad-attitude scale is a material limitation.

## Primary result

The standardized association from the pre-specified all-item ad-attitude composite
to the pre-specified all-item purchase-intention composite was
beta = 0.549, 95% HC3 CI
[0.521, 0.578], p < .0001, controlling
for the five persona-condition scores and run
fixed effects.

Significant persona-condition predictors of ad attitude at alpha = .05:
extraversion, agreeableness, conscientiousness, neuroticism, openness.

Traits with a 2,000-resample run-stratified percentile interval excluding zero:
extraversion, agreeableness, conscientiousness, neuroticism, openness.

## Interpretation boundary

The four-configuration analysis is exploratory. Culture and LLM backbone vary
together in the historical data, so differences cannot be attributed uniquely to
culture. No human-population inference is made.

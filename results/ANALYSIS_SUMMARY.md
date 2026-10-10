# Analysis summary

This report is generated automatically by `scripts/run_analysis.py`.

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

The item-level correlation matrix, corrected item-total correlations, and alpha-if-
deleted statistics are reported in `results/tables/`. They diagnose which generated
item responses account for the weak coherence rather than treating low alpha as a
minor caveat.

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

## Measurement-screened sensitivity result

With the internally more coherent two-item advertising-attitude and purchase-
intention composites, the adjusted advertising-attitude coefficient was beta =
-0.053, 95% HC3 CI
[-0.092, -0.014].
This sign reversal is a central measurement-instability finding, not evidence that
the prespecified result is robust. Sensitivity indirect-effect intervals excluded
zero for: extraversion, agreeableness, conscientiousness, neuroticism, openness.

## Missingness diagnosis

The purchase-intention equation excludes 292
records. The original-JSON audit found these purchase/intention-like key variants:
purchase_interest_scores (n=120); purchase_intentionscores (n=64); purchase_intentions_scores (n=49); purchase_intent_scores (n=48); purchase_interestion_scores (n=6); purchase_intentions (n=3); purchase_interest scores (n=1); purchase_interestionscores (n=1).
The converter did not recognize every emitted key variant. Run- and persona-condition-
specific missingness tables are provided in `results/quality/` so the exclusion can
be assessed rather than described only as generic incomplete output. In a post-hoc
data-repair sensitivity analysis that recovered all 292 preserved three-value arrays,
the focal coefficient was beta = 0.551,
95% HC3 CI [0.524,
0.578], n =
4,987. This check does not replace the archived
primary complete-case specification.

## Interpretation boundary

The four-configuration analysis is exploratory. Culture and LLM backbone vary
together in the historical data, so differences cannot be attributed uniquely to
culture. No human-population inference is made.

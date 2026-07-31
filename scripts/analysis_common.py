"""Shared utilities for the reproducible analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


TRAIT_ITEMS = {
    "extraversion": ["BFI_11", "BFI_12", "BFI_13"],
    "agreeableness": ["BFI_21", "BFI_22", "BFI_23"],
    "conscientiousness": ["BFI_31", "BFI_32", "BFI_33"],
    "neuroticism": ["BFI_41", "BFI_42", "BFI_43"],
    "openness": ["BFI_51", "BFI_52", "BFI_53"],
}
TRAITS = list(TRAIT_ITEMS)
AD_ITEMS = [f"ad_att_{i}" for i in range(1, 5)]
INTENT_ITEMS = [f"intent_{i}" for i in range(1, 4)]
SEM_ITEMS = [item for items in TRAIT_ITEMS.values() for item in items] + AD_ITEMS + INTENT_ITEMS

# The conversion script has already reverse-scored BFI_13, BFI_23, BFI_33,
# BFI_43, and BFI_53. Do not reverse them a second time here.
CORE_AD_ITEMS = ["ad_att_1", "ad_att_2"]
CORE_INTENT_ITEMS = ["intent_2", "intent_3"]


@dataclass(frozen=True)
class FitResult:
    outcome: str
    model: object
    coefficients: pd.DataFrame


def cronbach_alpha(frame: pd.DataFrame) -> float:
    complete = frame.dropna()
    if complete.empty or complete.shape[1] < 2:
        return np.nan
    item_variances = complete.var(axis=0, ddof=1).sum()
    total_variance = complete.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    k = complete.shape[1]
    return float(k / (k - 1) * (1 - item_variances / total_variance))


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"Cannot standardize {series.name}: standard deviation is zero or missing.")
    return (series - series.mean()) / sd


def add_analysis_variables(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in SEM_ITEMS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for trait, items in TRAIT_ITEMS.items():
        data[trait] = data[items].mean(axis=1)
    data["ad_attitude_core"] = data[CORE_AD_ITEMS].mean(axis=1)
    data["purchase_intention_core"] = data[CORE_INTENT_ITEMS].mean(axis=1)
    data["ad_attitude_all_items"] = data[AD_ITEMS].mean(axis=1)
    data["purchase_intention_all_items"] = data[INTENT_ITEMS].mean(axis=1)
    return data


def complete_cases(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return frame.dropna(subset=list(columns)).copy()


def fit_standardized_path(
    frame: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    include_run_effects: bool = True,
) -> FitResult:
    columns = [outcome, *predictors]
    if include_run_effects:
        columns.append("run")
    data = complete_cases(frame, columns)
    for column in [outcome, *predictors]:
        data[f"z_{column}"] = zscore(data[column])
    terms = [f"z_{name}" for name in predictors]
    if include_run_effects:
        terms.append("C(run)")
    formula = f"z_{outcome} ~ " + " + ".join(terms)
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    rows = []
    for predictor in predictors:
        term = f"z_{predictor}"
        low, high = model.conf_int().loc[term]
        rows.append(
            {
                "outcome": outcome,
                "predictor": predictor,
                "beta_standardized": model.params[term],
                "std_error_hc3": model.bse[term],
                "ci_95_low": low,
                "ci_95_high": high,
                "p_value": model.pvalues[term],
                "n": int(model.nobs),
                "r_squared": model.rsquared,
                "adjusted_r_squared": model.rsquared_adj,
                "run_fixed_effects": include_run_effects,
            }
        )
    return FitResult(outcome=outcome, model=model, coefficients=pd.DataFrame(rows))


def stratified_bootstrap_indirect_effects(
    frame: pd.DataFrame,
    mediator: str,
    outcome: str,
    traits: list[str],
    n_boot: int = 2000,
    seed: int = 20251010,
) -> pd.DataFrame:
    required = [mediator, outcome, *traits, "run"]
    data = complete_cases(frame, required)
    rng = np.random.default_rng(seed)
    draws = {trait: [] for trait in traits}

    grouped = [group.reset_index(drop=True) for _, group in data.groupby("run", sort=True)]
    for _ in range(n_boot):
        sample = pd.concat(
            [group.iloc[rng.integers(0, len(group), len(group))] for group in grouped],
            ignore_index=True,
        )
        mediator_fit = fit_standardized_path(sample, mediator, traits, include_run_effects=True)
        outcome_fit = fit_standardized_path(
            sample, outcome, [mediator, *traits], include_run_effects=True
        )
        b_path = float(
            outcome_fit.coefficients.loc[
                outcome_fit.coefficients["predictor"] == mediator, "beta_standardized"
            ].iloc[0]
        )
        for trait in traits:
            a_path = float(
                mediator_fit.coefficients.loc[
                    mediator_fit.coefficients["predictor"] == trait, "beta_standardized"
                ].iloc[0]
            )
            draws[trait].append(a_path * b_path)

    rows = []
    for trait, values in draws.items():
        array = np.asarray(values)
        rows.append(
            {
                "trait": trait,
                "indirect_effect_mean": array.mean(),
                "bootstrap_se": array.std(ddof=1),
                "ci_95_low": np.quantile(array, 0.025),
                "ci_95_high": np.quantile(array, 0.975),
                "significant_95": bool(
                    np.quantile(array, 0.025) > 0 or np.quantile(array, 0.975) < 0
                ),
                "bootstrap_method": "run-stratified percentile",
                "n_boot": n_boot,
                "seed": seed,
                "n_complete": len(data),
            }
        )
    return pd.DataFrame(rows)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

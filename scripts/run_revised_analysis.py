#!/usr/bin/env python3
"""Run the revised, reviewer-facing analyses and generate tables and figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm_big5_ad_sem_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analysis_common import (
    AD_ITEMS,
    CORE_AD_ITEMS,
    CORE_INTENT_ITEMS,
    INTENT_ITEMS,
    TRAIT_ITEMS,
    TRAITS,
    cronbach_alpha,
    ensure_directory,
    fit_standardized_path,
    stratified_bootstrap_indirect_effects,
)


def reliability_table(data: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "ad_attitude_all_4_items": AD_ITEMS,
        "ad_attitude_core_2_items": CORE_AD_ITEMS,
        "purchase_intention_all_3_items": INTENT_ITEMS,
        "purchase_intention_core_2_items": CORE_INTENT_ITEMS,
        **{f"persona_condition_{trait}": items for trait, items in TRAIT_ITEMS.items()},
    }
    rows = []
    for construct, items in definitions.items():
        rows.append(
            {
                "construct": construct,
                "items": ";".join(items),
                "n_items": len(items),
                "complete_n": len(data[items].dropna()),
                "cronbach_alpha": cronbach_alpha(data[items]),
                "interpretation": (
                    "diagnostic only: randomized persona inputs, not a reflective scale"
                    if construct.startswith("persona_condition_")
                    else "measurement diagnostic"
                ),
            }
        )
    return pd.DataFrame(rows)


def descriptive_table(data: pd.DataFrame) -> pd.DataFrame:
    variables = [
        *TRAITS,
        "ad_attitude_core",
        "purchase_intention_core",
        "ad_attitude_all_items",
        "purchase_intention_all_items",
    ]
    rows = []
    for variable in variables:
        series = data[variable]
        rows.append(
            {
                "variable": variable,
                "n": int(series.notna().sum()),
                "missing_n": int(series.isna().sum()),
                "mean": series.mean(),
                "std_dev": series.std(ddof=1),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def primary_models(data: pd.DataFrame) -> pd.DataFrame:
    mediator = fit_standardized_path(data, "ad_attitude_all_items", TRAITS)
    outcome = fit_standardized_path(
        data, "purchase_intention_all_items", ["ad_attitude_all_items", *TRAITS]
    )
    return pd.concat([mediator.coefficients, outcome.coefficients], ignore_index=True)


def sensitivity_models(data: pd.DataFrame) -> pd.DataFrame:
    mediator = fit_standardized_path(data, "ad_attitude_core", TRAITS)
    outcome = fit_standardized_path(
        data,
        "purchase_intention_core",
        ["ad_attitude_core", *TRAITS],
    )
    return pd.concat([mediator.coefficients, outcome.coefficients], ignore_index=True)


def configuration_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for configuration, group in data.groupby("configuration", sort=True):
        # Run identifiers were discarded in the historical merged files, so these
        # exploratory models use HC3 errors without run fixed effects.
        mediator = fit_standardized_path(
            group, "ad_attitude_all_items", TRAITS, include_run_effects=False
        )
        outcome = fit_standardized_path(
            group,
            "purchase_intention_all_items",
            ["ad_attitude_all_items", *TRAITS],
            include_run_effects=False,
        )
        result = pd.concat([mediator.coefficients, outcome.coefficients], ignore_index=True)
        result.insert(0, "configuration", configuration)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def plot_primary_paths(paths: pd.DataFrame, destination: Path) -> None:
    selected = paths[
        ((paths["outcome"] == "ad_attitude_all_items") & paths["predictor"].isin(TRAITS))
        | (
            (paths["outcome"] == "purchase_intention_all_items")
            & (paths["predictor"] == "ad_attitude_all_items")
        )
    ].copy()
    labels = {
        "extraversion": "Extraversion",
        "agreeableness": "Agreeableness",
        "conscientiousness": "Conscientiousness",
        "neuroticism": "Neuroticism",
        "openness": "Openness",
        "ad_attitude_all_items": "Ad attitude",
    }
    selected["label"] = selected["predictor"].map(labels)
    selected = selected.iloc[::-1]
    colors = np.where(selected["p_value"] < 0.05, "#1f6f8b", "#9aa0a6")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(selected))
    ax.errorbar(
        selected["beta_standardized"],
        y,
        xerr=[
            selected["beta_standardized"] - selected["ci_95_low"],
            selected["ci_95_high"] - selected["beta_standardized"],
        ],
        fmt="none",
        ecolor="#6b7280",
        capsize=4,
        linewidth=1.6,
    )
    ax.scatter(selected["beta_standardized"], y, c=colors, s=58, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, selected["label"])
    ax.set_xlabel("Standardized coefficient (95% HC3 CI)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_configurations(paths: pd.DataFrame, destination: Path) -> None:
    subset = paths[
        (paths["outcome"] == "purchase_intention_all_items")
        & (paths["predictor"] == "ad_attitude_all_items")
    ].copy()
    subset = subset.sort_values("beta_standardized")
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    y = np.arange(len(subset))
    ax.errorbar(
        subset["beta_standardized"],
        y,
        xerr=[
            subset["beta_standardized"] - subset["ci_95_low"],
            subset["ci_95_high"] - subset["beta_standardized"],
        ],
        fmt="o",
        color="#6b4c9a",
        capsize=4,
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    display_labels = (
        subset["configuration"]
        .str.replace("__", " / ", regex=False)
        .str.replace("_", " ", regex=False)
    )
    ax.set_yticks(y, display_labels)
    ax.set_xlabel("Standardized ad-attitude coefficient (95% HC3 CI)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_configuration_heatmap(paths: pd.DataFrame, destination: Path) -> None:
    labels = {
        "extraversion": "Extraversion → Ad attitude",
        "agreeableness": "Agreeableness → Ad attitude",
        "conscientiousness": "Conscientiousness → Ad attitude",
        "neuroticism": "Neuroticism → Ad attitude",
        "openness": "Openness → Ad attitude",
        "ad_attitude_all_items": "Ad attitude → Purchase intention",
    }
    subset = paths[
        ((paths["outcome"] == "ad_attitude_all_items") & paths["predictor"].isin(TRAITS))
        | (
            (paths["outcome"] == "purchase_intention_all_items")
            & (paths["predictor"] == "ad_attitude_all_items")
        )
    ].copy()
    subset["path"] = subset["predictor"].map(labels)
    subset["display_configuration"] = (
        subset["configuration"]
        .str.replace("__", " / ", regex=False)
        .str.replace("_", " ", regex=False)
    )
    order = [labels[name] for name in TRAITS] + [labels["ad_attitude_all_items"]]
    matrix = subset.pivot(
        index="path", columns="display_configuration", values="beta_standardized"
    ).reindex(order)
    limit = max(abs(matrix.min().min()), abs(matrix.max().max()))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            text_color = "white" if abs(value) > limit * 0.55 else "#222222"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_color)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=24, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label("Standardized coefficient")
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_revised_path_diagram(paths: pd.DataFrame, destination: Path) -> None:
    labels = {
        "extraversion": "Extraversion",
        "agreeableness": "Agreeableness",
        "conscientiousness": "Conscientiousness",
        "neuroticism": "Neuroticism",
        "openness": "Openness",
    }
    trait_paths = paths[
        (paths["outcome"] == "ad_attitude_all_items") & paths["predictor"].isin(TRAITS)
    ].set_index("predictor")
    attitude_path = paths[
        (paths["outcome"] == "purchase_intention_all_items")
        & (paths["predictor"] == "ad_attitude_all_items")
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    trait_y = dict(zip(TRAITS, [0.84, 0.67, 0.50, 0.33, 0.16]))
    positions = {trait: (0.14, y) for trait, y in trait_y.items()}
    positions.update({"ad": (0.57, 0.50), "intent": (0.88, 0.50)})
    trait_width, trait_height = 0.22, 0.075
    outcome_width, outcome_height = 0.19, 0.105

    def node(x, y, width, height, text, color):
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color,
            edgecolor="#284b63",
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=11, zorder=4)

    for trait, (x, y) in positions.items():
        if trait in TRAITS:
            node(x, y, trait_width, trait_height, labels[trait], "#e8f1f5")
    node(
        *positions["ad"],
        outcome_width,
        outcome_height,
        "Ad attitude\n(4-item composite)",
        "#d6eaf2",
    )
    node(
        *positions["intent"],
        outcome_width,
        outcome_height,
        "Purchase intention\n(3-item composite)",
        "#eadff2",
    )

    target_y = dict(zip(TRAITS, [0.565, 0.5325, 0.50, 0.4675, 0.435]))
    label_x = 0.385
    start_x = positions[TRAITS[0]][0] + trait_width / 2
    end_x = positions["ad"][0] - outcome_width / 2
    for trait in TRAITS:
        start = positions[trait]
        end = (positions["ad"][0], target_y[trait])
        row = trait_paths.loc[trait]
        arrow = FancyArrowPatch(
            (start_x, start[1]),
            (end_x, end[1]),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color="#1f6f8b" if row["p_value"] < 0.05 else "#8a8f98",
            connectionstyle="arc3,rad=0",
            zorder=1,
        )
        ax.add_patch(arrow)
        fraction = (label_x - start_x) / (end_x - start_x)
        label_y = start[1] + fraction * (end[1] - start[1])
        ax.text(
            label_x,
            label_y,
            f"β={row['beta_standardized']:.2f}",
            ha="center",
            va="center",
            fontsize=9.2,
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": "none",
                "linewidth": 0,
            },
        )

    main_start_x = positions["ad"][0] + outcome_width / 2
    main_end_x = positions["intent"][0] - outcome_width / 2
    ax.add_patch(
        FancyArrowPatch(
            (main_start_x, positions["ad"][1]),
            (main_end_x - 0.008, positions["intent"][1]),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=2.2,
            color="#6b4c9a",
            zorder=2,
        )
    )
    ax.text(
        (main_start_x + main_end_x) / 2,
        0.575,
        f"β={attitude_path['beta_standardized']:.2f}***",
        ha="center",
        va="center",
        fontsize=11,
        color="#4e3677",
        zorder=5,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "linewidth": 0,
        },
    )
    ax.text(
        0.5,
        0.045,
        "Standardized coefficients; HC3 robust inference with generation-run fixed effects.  *** p < .001",
        ha="center",
        fontsize=9.5,
        color="#4f555b",
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_summary(
    destination: Path,
    primary: pd.DataFrame,
    paths: pd.DataFrame,
    indirect: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    ad_path = paths[
        (paths["outcome"] == "purchase_intention_all_items")
        & (paths["predictor"] == "ad_attitude_all_items")
    ].iloc[0]
    significant_traits = paths[
        (paths["outcome"] == "ad_attitude_all_items") & (paths["p_value"] < 0.05)
    ]["predictor"].tolist()
    significant_indirect = indirect[indirect["significant_95"]]["trait"].tolist()
    alpha = reliability.set_index("construct")["cronbach_alpha"]
    p_text = "< .0001" if ad_path["p_value"] < 0.0001 else f"= {ad_path['p_value']:.4f}"
    text = f"""# Revised analysis summary

This report is generated automatically by `scripts/run_revised_analysis.py`.

## Analysis population

- Source: ten per-run DeepSeek-V2 236B CSV files.
- Cultural condition: Chinese.
- Advertisement condition: promotion-framed.
- Raw simulation responses: {len(primary):,}.
- Complete cases for the primary path analysis: {len(primary.dropna(subset=['ad_attitude_all_items', 'purchase_intention_all_items'])):,}.

## Measurement decision

The Big Five inputs are randomized persona-conditioning values rather than responses
to a reflective psychometric instrument. They are therefore summarized as observed
condition scores, not fitted as latent variables. To avoid post-hoc item selection,
the primary analysis uses all four pre-specified ad-attitude items (alpha =
{alpha['ad_attitude_all_4_items']:.3f}) and all three pre-specified purchase-intention
items (alpha = {alpha['purchase_intention_all_3_items']:.3f}). A measurement-screened
two-item analysis is reported separately as a sensitivity analysis. The low alpha for
the full ad-attitude scale is a material limitation.

## Primary result

The standardized association from the pre-specified all-item ad-attitude composite
to the pre-specified all-item purchase-intention composite was
beta = {ad_path['beta_standardized']:.3f}, 95% HC3 CI
[{ad_path['ci_95_low']:.3f}, {ad_path['ci_95_high']:.3f}], p {p_text}, controlling
for the five persona-condition scores and run
fixed effects.

Significant persona-condition predictors of ad attitude at alpha = .05:
{', '.join(significant_traits) if significant_traits else 'none'}.

Traits with a 2,000-resample run-stratified percentile interval excluding zero:
{', '.join(significant_indirect) if significant_indirect else 'none'}.

## Interpretation boundary

The four-configuration analysis is exploratory. Culture and LLM backbone vary
together in the historical data, so differences cannot be attributed uniquely to
culture. No human-population inference is made.
"""
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20251010)
    args = parser.parse_args()
    root = args.root.resolve()
    processed = root / "data" / "processed"
    tables = ensure_directory(root / "results" / "tables")
    sem = ensure_directory(root / "results" / "sem")
    figures = ensure_directory(root / "results" / "figures")

    primary = pd.read_csv(processed / "primary_chinese_promotion.csv")
    configurations = pd.read_csv(processed / "exploratory_configurations_promotion.csv")

    reliability = reliability_table(primary)
    descriptives = descriptive_table(primary)
    paths = primary_models(primary)
    sensitivity = sensitivity_models(primary)
    indirect = stratified_bootstrap_indirect_effects(
        primary,
        mediator="ad_attitude_all_items",
        outcome="purchase_intention_all_items",
        traits=TRAITS,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    configuration_paths = configuration_models(configurations)

    reliability.to_csv(tables / "measurement_diagnostics.csv", index=False)
    descriptives.to_csv(tables / "descriptive_statistics.csv", index=False)
    paths.to_csv(sem / "primary_path_coefficients.csv", index=False)
    indirect.to_csv(sem / "primary_indirect_effects_bootstrap.csv", index=False)
    sensitivity.to_csv(sem / "sensitivity_core_items_path_coefficients.csv", index=False)
    configuration_paths.to_csv(
        sem / "exploratory_configuration_path_coefficients.csv", index=False
    )
    plot_primary_paths(paths, figures / "revised_primary_paths.png")
    plot_configurations(configuration_paths, figures / "exploratory_configurations.png")
    plot_configuration_heatmap(configuration_paths, figures / "figure5_configuration_heatmap.png")
    plot_configurations(configuration_paths, figures / "figure6_configuration_forest.png")
    plot_revised_path_diagram(paths, figures / "figure7_revised_path_model.png")
    write_summary(
        root / "results" / "ANALYSIS_SUMMARY.md",
        primary,
        paths,
        indirect,
        reliability,
    )
    print(paths.to_string(index=False))
    print("\nBootstrap indirect effects\n", indirect.to_string(index=False))


if __name__ == "__main__":
    main()

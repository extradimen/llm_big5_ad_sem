#!/usr/bin/env python3
"""Build provenance-preserving analysis datasets from the per-run CSV files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from analysis_common import SEM_ITEMS, add_analysis_variables, ensure_directory


PRIMARY_PATTERN = "deepseek-v2_236b_*.csv"
CONFIGURATIONS = {
    "Chinese__deepseek-r1_14b": "deepseek-r1_14b_promotion.csv",
    "Southeast_Asian__sailor2_20b": "sailor2_20b_promotion.csv",
    "Arab__jais-adaptive_7b": "jwnder_jais-adaptive_7b_promotion.csv",
    "American__gemma2_9b": "gemma2_9b_promotion.csv",
}


def parse_run(path: Path) -> int:
    match = re.search(r"_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse run number from {path.name}")
    return int(match.group(1))


def validate_sem_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in SEM_ITEMS:
        expected_low, expected_high = (1, 5) if column.startswith("BFI_") else (1, 7)
        numeric = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "variable": column,
                "expected_min": expected_low,
                "expected_max": expected_high,
                "observed_min": numeric.min(),
                "observed_max": numeric.max(),
                "missing_n": int(numeric.isna().sum()),
                "out_of_range_n": int(((numeric < expected_low) | (numeric > expected_high)).sum()),
            }
        )
    return pd.DataFrame(rows)


def coerce_invalid_sem_values(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in SEM_ITEMS:
        expected_low, expected_high = (1, 5) if column.startswith("BFI_") else (1, 7)
        numeric = pd.to_numeric(data[column], errors="coerce")
        data[column] = numeric.where(numeric.between(expected_low, expected_high))
    return data


def build_primary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((root / "csv_output").glob(PRIMARY_PATTERN), key=parse_run)
    if not files:
        raise FileNotFoundError(f"No files matched csv_output/{PRIMARY_PATTERN}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        run = parse_run(path)
        frame["run"] = run
        frame["source_file"] = path.name
        frame["model"] = "deepseek-v2_236b"
        frame["response_id"] = [
            f"deepseek-v2_236b__run{run:02d}__sample{sample}__{ad_type}"
            for sample, ad_type in zip(frame["sample_id"], frame["ad_type"])
        ]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["response_id"].duplicated().any():
        raise ValueError("Primary response_id values are not unique.")
    validation = validate_sem_ranges(combined)
    combined = add_analysis_variables(coerce_invalid_sem_values(combined))
    return combined, validation


def build_configurations(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    validation = []
    for configuration, filename in CONFIGURATIONS.items():
        path = root / "csv_output_merged" / filename
        frame = pd.read_csv(path)
        frame["configuration"] = configuration
        frame["source_file"] = filename
        frame["row_in_source"] = range(1, len(frame) + 1)
        frame["response_id"] = [f"{configuration}__row{row:05d}" for row in frame["row_in_source"]]
        check = validate_sem_ranges(frame)
        check.insert(0, "configuration", configuration)
        validation.append(check)
        frame = add_analysis_variables(coerce_invalid_sem_values(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), pd.concat(validation, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    processed = ensure_directory(root / "data" / "processed")
    quality = ensure_directory(root / "results" / "quality")

    primary, primary_validation = build_primary(root)
    configurations, configuration_validation = build_configurations(root)
    primary.to_csv(processed / "primary_chinese_promotion.csv", index=False)
    configurations.to_csv(processed / "exploratory_configurations_promotion.csv", index=False)
    primary_validation.to_csv(quality / "primary_range_and_missing_checks.csv", index=False)
    configuration_validation.to_csv(
        quality / "configuration_range_and_missing_checks.csv", index=False
    )

    summary = pd.DataFrame(
        [
            {
                "dataset": "primary_chinese_promotion",
                "rows_raw": len(primary),
                "rows_complete_primary_analysis": len(
                    primary.dropna(
                        subset=[
                            "ad_attitude_all_items",
                            "purchase_intention_all_items",
                            "extraversion",
                            "agreeableness",
                            "conscientiousness",
                            "neuroticism",
                            "openness",
                        ]
                    )
                ),
                "unique_response_id": primary["response_id"].nunique(),
                "runs": primary["run"].nunique(),
            },
            {
                "dataset": "exploratory_configurations_promotion",
                "rows_raw": len(configurations),
                "rows_complete_primary_analysis": len(
                    configurations.dropna(
                        subset=["ad_attitude_all_items", "purchase_intention_all_items"]
                    )
                ),
                "unique_response_id": configurations["response_id"].nunique(),
                "runs": pd.NA,
            },
        ]
    )
    summary.to_csv(quality / "dataset_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

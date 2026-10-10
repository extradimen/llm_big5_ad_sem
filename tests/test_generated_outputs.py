"""Integrity checks for outputs created by scripts/run_all.py."""

from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class GeneratedOutputTests(unittest.TestCase):
    def test_primary_ids_are_unique(self):
        data = pd.read_csv(ROOT / "data/processed/primary_chinese_promotion.csv")
        self.assertEqual(len(data), 4987)
        self.assertFalse(data["response_id"].duplicated().any())
        self.assertEqual(data["run"].nunique(), 10)

    def test_processed_scales_are_in_range_after_invalid_values_are_coerced(self):
        for filename in [
            "primary_chinese_promotion.csv",
            "exploratory_configurations_promotion.csv",
        ]:
            data = pd.read_csv(ROOT / "data/processed" / filename)
            bfi = [column for column in data if column.startswith("BFI_")]
            responses = [
                column
                for column in data
                if column.startswith("ad_att_") or column.startswith("intent_")
            ]
            self.assertTrue(data[bfi].min().ge(1).all())
            self.assertTrue(data[bfi].max().le(5).all())
            self.assertTrue(data[responses].min().ge(1).all())
            self.assertTrue(data[responses].max().le(7).all())

    def test_historical_out_of_range_values_are_reported(self):
        report = pd.read_csv(
            ROOT / "results/quality/configuration_range_and_missing_checks.csv"
        )
        self.assertGreater(int(report["out_of_range_n"].sum()), 0)

    def test_bootstrap_is_final(self):
        for filename in [
            "primary_indirect_effects_bootstrap.csv",
            "sensitivity_core_items_indirect_effects_bootstrap.csv",
        ]:
            result = pd.read_csv(ROOT / "results/sem" / filename)
            self.assertEqual(set(result["n_boot"]), {2000})
            self.assertEqual(set(result["seed"]), {20251010})
            self.assertEqual(set(result["n_complete"]), {4695})
            self.assertFalse(result.isna().any().any())

    def test_item_and_missingness_diagnostics_are_coherent(self):
        matrix = pd.read_csv(
            ROOT / "results/tables/ad_attitude_inter_item_correlation_matrix.csv",
            index_col=0,
        )
        self.assertEqual(list(matrix.index), [f"ad_att_{i}" for i in range(1, 5)])
        self.assertTrue((matrix.values.diagonal() == 1).all())
        missing = pd.read_csv(ROOT / "results/quality/primary_missingness_patterns.csv")
        incomplete = missing[missing["missing_variables"] != "complete"]
        self.assertEqual(int(incomplete["n"].sum()), 292)
        key_summary = pd.read_csv(
            ROOT / "results/quality/primary_missingness_json_key_summary.csv"
        )
        self.assertEqual(int(key_summary["records"].sum()), 292)

    def test_run_specific_outputs_cover_every_run_and_specification(self):
        result = pd.read_csv(ROOT / "results/sem/run_specific_path_coefficients.csv")
        self.assertEqual(result["run"].nunique(), 10)
        self.assertEqual(
            set(result["specification"]),
            {"primary_all_items", "sensitivity_core_items"},
        )

    def test_post_hoc_json_key_recovery_restores_all_records(self):
        result = pd.read_csv(
            ROOT / "results/sem/post_hoc_recovered_missingness_focal_comparison.csv"
        )
        self.assertEqual(set(result["recovered_records"]), {292})
        self.assertEqual(set(result["n"]), {4987})

    def test_expected_outputs_exist(self):
        expected = [
            "results/sem/primary_path_coefficients.csv",
            "results/sem/sensitivity_core_items_path_coefficients.csv",
            "results/sem/sensitivity_core_items_indirect_effects_bootstrap.csv",
            "results/sem/run_specific_path_coefficients.csv",
            "results/sem/run_specific_path_variability_summary.csv",
            "results/sem/post_hoc_recovered_missingness_path_coefficients.csv",
            "results/sem/post_hoc_recovered_missingness_focal_comparison.csv",
            "results/tables/measurement_diagnostics.csv",
            "results/tables/ad_attitude_inter_item_correlation_matrix.csv",
            "results/tables/ad_attitude_item_diagnostics.csv",
            "results/quality/primary_missingness_by_run.csv",
            "results/quality/primary_missingness_by_persona_condition.csv",
            "results/quality/primary_missingness_json_audit.csv",
            "results/quality/primary_missingness_json_key_summary.csv",
            "results/figures/figure1_experiment_flow.png",
            "results/figures/figure2_bfi_questions.png",
            "results/figures/figure3_advertising_scenarios.png",
            "results/figures/figure4_prompt_template.png",
            "results/figures/figure5_configuration_heatmap.png",
            "results/figures/figure6_configuration_forest.png",
            "results/figures/figure7_path_model.png",
            "results/ANALYSIS_SUMMARY.md",
        ]
        for relative in expected:
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()

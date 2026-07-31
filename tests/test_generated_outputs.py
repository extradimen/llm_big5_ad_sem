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
        result = pd.read_csv(ROOT / "results/sem/primary_indirect_effects_bootstrap.csv")
        self.assertEqual(set(result["n_boot"]), {2000})
        self.assertEqual(set(result["seed"]), {20251010})
        self.assertEqual(set(result["n_complete"]), {4695})
        self.assertFalse(result.isna().any().any())

    def test_expected_outputs_exist(self):
        expected = [
            "results/sem/primary_path_coefficients.csv",
            "results/sem/sensitivity_core_items_path_coefficients.csv",
            "results/tables/measurement_diagnostics.csv",
            "results/figures/primary_path_coefficients.png",
            "results/figures/exploratory_configurations.png",
            "results/figures/figure5_configuration_heatmap.png",
            "results/figures/figure6_configuration_forest.png",
            "results/figures/figure7_path_model.png",
            "results/ANALYSIS_SUMMARY.md",
        ]
        for relative in expected:
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()

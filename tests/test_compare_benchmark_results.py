from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import compare_benchmark_results


class CompareBenchmarkResultsTests(unittest.TestCase):
    def test_build_cross_dataset_comparison(self):
        rows = pd.DataFrame(
            [
                {
                    "dataset": "nordland",
                    "attack_name": "gaussian_noise",
                    "attack_severity": 1,
                    "attack_eps": None,
                    "attack_steps": None,
                    "attack_step_size": None,
                    "attack_keep_ratio": None,
                    "attack_mask_mode": "none",
                    "attacked_r_at_1_mean": 70.0,
                    "attacked_r_at_5_mean": 80.0,
                    "delta_r_at_1_mean": -10.0,
                    "delta_r_at_5_mean": -5.0,
                    "retention_r_at_1_mean": 0.875,
                    "retention_r_at_5_mean": 0.9412,
                    "attacked_r_at_1_worst": 70.0,
                    "delta_r_at_1_worst": -10.0,
                    "retention_r_at_1_worst": 0.875,
                },
                {
                    "dataset": "msls",
                    "attack_name": "gaussian_noise",
                    "attack_severity": 1,
                    "attack_eps": None,
                    "attack_steps": None,
                    "attack_step_size": None,
                    "attack_keep_ratio": None,
                    "attack_mask_mode": "none",
                    "attacked_r_at_1_mean": 50.0,
                    "attacked_r_at_5_mean": 65.0,
                    "delta_r_at_1_mean": -20.0,
                    "delta_r_at_5_mean": -12.0,
                    "retention_r_at_1_mean": 0.7143,
                    "retention_r_at_5_mean": 0.8442,
                    "attacked_r_at_1_worst": 50.0,
                    "delta_r_at_1_worst": -20.0,
                    "retention_r_at_1_worst": 0.7143,
                },
            ]
        )
        rows["attack_key"] = rows.apply(compare_benchmark_results.attack_key_from_row, axis=1)
        comparison = compare_benchmark_results.build_cross_dataset_comparison(rows)

        self.assertEqual(len(comparison), 1)
        self.assertIn("attacked_r_at_1_mean_nordland", comparison.columns)
        self.assertIn("attacked_r_at_1_mean_msls", comparison.columns)
        self.assertAlmostEqual(comparison.iloc[0]["delta_r_at_1_mean_msls"], -20.0)

    def test_load_and_overview_from_run_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "results" / "nordland" / "2026-04-12_10-00-00"
            (run_root / "01_clean_baseline").mkdir(parents=True)

            attack_summary = pd.DataFrame(
                [
                    {
                        "attack_name": "gaussian_noise",
                        "attack_severity": 1,
                        "attack_eps": None,
                        "attack_steps": None,
                        "attack_step_size": None,
                        "attack_keep_ratio": None,
                        "attack_mask_mode": "none",
                        "attacked_r_at_1_mean": 70.0,
                        "attacked_r_at_5_mean": 80.0,
                        "delta_r_at_1_mean": -10.0,
                        "delta_r_at_5_mean": -5.0,
                        "retention_r_at_1_mean": 0.875,
                        "retention_r_at_5_mean": 0.9412,
                        "attacked_r_at_1_worst": 70.0,
                        "delta_r_at_1_worst": -10.0,
                        "retention_r_at_1_worst": 0.875,
                    }
                ]
            )
            attack_summary.to_csv(run_root / "attack_summary.csv", index=False)
            (run_root / "robustness_summary.json").write_text(
                json.dumps(
                    {
                        "attack_count": 1,
                        "mean_retention_r_at_1": 0.875,
                        "mean_retention_r_at_5": 0.9412,
                        "worst_case_r_at_1": 70.0,
                        "worst_case_retention_r_at_1": 0.875,
                        "worst_case_delta_r_at_1": -10.0,
                    }
                )
            )
            (run_root / "01_clean_baseline" / "metrics.json").write_text(
                json.dumps({"recalls": [80.0, 85.0, 90.0, 98.0]})
            )

            loaded_attack_summary, robustness_summary, baseline_metrics = compare_benchmark_results.load_dataset_outputs(run_root)
            overview = compare_benchmark_results.build_dataset_overview(
                dataset="nordland",
                run_root=run_root,
                robustness_summary=robustness_summary,
                baseline_metrics=baseline_metrics,
            )

            self.assertEqual(len(loaded_attack_summary), 1)
            self.assertEqual(overview["baseline_r_at_1"], 80.0)
            self.assertEqual(overview["worst_case_delta_r_at_1"], -10.0)


if __name__ == "__main__":
    unittest.main()

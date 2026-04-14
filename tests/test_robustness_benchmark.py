from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import robustness_benchmark


class RobustnessBenchmarkTests(unittest.TestCase):
    def test_in_selected_range(self):
        self.assertTrue(robustness_benchmark._in_selected_range(1, 1, 22))
        self.assertTrue(robustness_benchmark._in_selected_range(22, 1, 22))
        self.assertFalse(robustness_benchmark._in_selected_range(23, 1, 22))
        self.assertFalse(robustness_benchmark._in_selected_range(1, 23, None))

    def test_compute_paired_outputs_and_aggregate_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_dir = root / "baseline"
            attack_dir = root / "attack"
            baseline_dir.mkdir()
            attack_dir.mkdir()

            baseline_metrics = {
                "recalls": [90.0, 95.0, 97.0, 99.0],
                "attack": {"attack_name": "none", "attack_seed": 0},
            }
            attack_metrics = {
                "recalls": [60.0, 80.0, 90.0, 98.0],
                "attack": {
                    "attack_name": "gaussian_noise",
                    "attack_severity": 2,
                    "attack_seed": 0,
                    "attack_eps": None,
                    "attack_steps": None,
                    "attack_step_size": None,
                    "attack_mask_mode": "none",
                    "attack_keep_ratio": None,
                },
            }
            (baseline_dir / "metrics.json").write_text(json.dumps(baseline_metrics))
            (attack_dir / "metrics.json").write_text(json.dumps(attack_metrics))

            baseline_df = pd.DataFrame(
                [
                    {"query_index": 0, "margin": 1.5, "best_positive_rank": 1, "top1_correct": True},
                    {"query_index": 1, "margin": 0.4, "best_positive_rank": 2, "top1_correct": False},
                ]
            )
            attack_df = pd.DataFrame(
                [
                    {"query_index": 0, "margin": 0.2, "best_positive_rank": 5, "top1_correct": False},
                    {"query_index": 1, "margin": 0.9, "best_positive_rank": 1, "top1_correct": True},
                ]
            )
            baseline_df.to_csv(baseline_dir / "per_query.csv", index=False)
            attack_df.to_csv(attack_dir / "per_query.csv", index=False)

            paired_df, paired_summary = robustness_benchmark.compute_paired_outputs(baseline_dir, attack_dir)

            self.assertEqual(len(paired_df), 2)
            self.assertAlmostEqual(paired_summary["delta_recalls"][0], -30.0)
            self.assertAlmostEqual(paired_summary["retention_recalls"][0], 60.0 / 90.0)
            self.assertAlmostEqual(paired_summary["top1_flip_rate"], 1.0)
            self.assertAlmostEqual(paired_summary["recovery_rate"], 1.0)

            summary_rows = [robustness_benchmark.summary_row(paired_summary)]
            aggregated = robustness_benchmark.aggregate_attack_summary(summary_rows, root)
            self.assertEqual(len(aggregated), 1)
            self.assertEqual(aggregated.iloc[0]["seed_count"], 1)
            self.assertAlmostEqual(aggregated.iloc[0]["attacked_r_at_1_mean"], 60.0)


if __name__ == "__main__":
    unittest.main()

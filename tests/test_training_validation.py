from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

import util


class TrainingValidationTests(unittest.TestCase):
    def test_resolve_validation_state_uses_clean_recalls_without_adv_training(self):
        state = util.resolve_validation_state(
            clean_recalls=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            adv_recalls=None,
            adv_train_enabled=False,
        )

        self.assertEqual(state["selected_metric_name"], "val_clean_r1_r5")
        self.assertAlmostEqual(state["selected_score"], 30.0)
        self.assertTrue(np.array_equal(state["selected_recalls"], np.array([10.0, 20.0, 30.0], dtype=np.float32)))

    def test_resolve_validation_state_uses_adv_recalls_with_adv_training(self):
        state = util.resolve_validation_state(
            clean_recalls=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            adv_recalls=np.array([7.0, 11.0, 13.0], dtype=np.float32),
            adv_train_enabled=True,
        )

        self.assertEqual(state["selected_metric_name"], "val_adv_r1_r5")
        self.assertAlmostEqual(state["selected_score"], 18.0)
        self.assertTrue(np.array_equal(state["selected_recalls"], np.array([7.0, 11.0, 13.0], dtype=np.float32)))

    def test_resume_train_prefers_best_metric_metadata_when_available(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "last_model.pth"
            torch.save(
                {
                    "epoch_num": 3,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_r5": 12.0,
                    "best_metric_score": 18.0,
                    "best_metric_name": "val_adv_r1_r5",
                    "not_improved_num": 2,
                },
                checkpoint_path,
            )
            torch.save({"model_state_dict": model.state_dict()}, Path(tmp_dir) / "best_model.pth")

            args = types.SimpleNamespace(resume=str(checkpoint_path), save_dir=tmp_dir)
            _, _, best_score, start_epoch, not_improved, metric_name = util.resume_train(args, model, optimizer)

        self.assertEqual(start_epoch, 3)
        self.assertEqual(not_improved, 2)
        self.assertAlmostEqual(best_score, 18.0)
        self.assertEqual(metric_name, "val_adv_r1_r5")


if __name__ == "__main__":
    unittest.main()

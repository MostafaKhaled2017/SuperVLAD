from __future__ import annotations

import types
import unittest

import torch

import adversarial


class DummyDescriptorModel(torch.nn.Module):
    def forward(self, x, queryflag=0):
        flat = x.view(x.shape[0], -1)
        return torch.nn.functional.normalize(flat, dim=1)


class AdversarialTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = {
            "attack_name": "none",
            "attack_severity": None,
            "attack_seed": 0,
            "attack_eps": None,
            "attack_steps": None,
            "attack_step_size": None,
            "attack_mask_mode": "none",
            "attack_keep_ratio": None,
            "token_keep_ratio": 1.0,
            "masking_mode": "none",
            "test_method": "hard_resize",
            "pca_dim": None,
        }
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_attack_config_promotes_legacy_token_mask_args(self):
        args = self._args(token_keep_ratio=0.5, masking_mode="random")
        config = adversarial.attack_config_from_args(args)
        self.assertEqual(config.attack_name, "token_mask")
        self.assertEqual(config.attack_keep_ratio, 0.5)
        self.assertEqual(config.attack_mask_mode, "random")

    def test_validate_attack_arguments_rejects_combined_token_mask(self):
        args = self._args(attack_name="fgsm_linf", attack_eps=8.0 / 255.0, token_keep_ratio=0.5)
        with self.assertRaises(ValueError):
            adversarial.validate_attack_arguments(args)

    def test_gaussian_noise_is_deterministic_for_seed(self):
        config = adversarial.AttackConfig(
            attack_name="gaussian_noise",
            attack_severity=2,
            attack_seed=7,
        )
        inputs = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        sample_ids = [10, 11]
        output_a = adversarial.apply_query_attack(inputs, sample_ids, config)
        output_b = adversarial.apply_query_attack(inputs, sample_ids, config)
        self.assertTrue(torch.allclose(output_a, output_b))

    def test_pgd_attack_respects_bounds_and_increases_attack_loss(self):
        model = DummyDescriptorModel()
        inputs = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        database_descriptors = torch.stack(
            [
                torch.nn.functional.normalize(torch.nn.functional.one_hot(torch.tensor(0), num_classes=48).float(), dim=0),
                torch.nn.functional.normalize(torch.ones(48), dim=0),
            ],
            dim=0,
        )
        reference_pairs = {
            "positive_indices": [0],
            "negative_indices": [1],
        }
        config = adversarial.AttackConfig(
            attack_name="pgd_linf",
            attack_seed=3,
            attack_eps=8.0 / 255.0,
            attack_steps=4,
            attack_step_size=2.0 / 255.0,
        )
        original_descriptors = model(inputs)
        original_loss = adversarial._whitebox_loss(
            original_descriptors,
            database_descriptors,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        )

        attacked = adversarial.apply_whitebox_attack(
            inputs=inputs,
            sample_ids=[0],
            config=config,
            model=model,
            database_descriptors=database_descriptors,
            reference_pairs=reference_pairs,
        )
        attacked_descriptors = model(attacked)
        attacked_loss = adversarial._whitebox_loss(
            attacked_descriptors,
            database_descriptors,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        )

        attacked_pixels = adversarial.denormalize_tensor(attacked)
        original_pixels = adversarial.denormalize_tensor(inputs)
        max_delta = torch.max(torch.abs(attacked_pixels - original_pixels)).item()

        self.assertGreaterEqual(attacked_loss.item(), original_loss.item() - 1e-6)
        self.assertLessEqual(max_delta, (8.0 / 255.0) + 1e-6)
        self.assertGreaterEqual(attacked_pixels.min().item(), -1e-6)
        self.assertLessEqual(attacked_pixels.max().item(), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()

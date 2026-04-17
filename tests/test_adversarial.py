from __future__ import annotations

import numpy as np
import types
import unittest

import torch

import adversarial
import parser as parser_module


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

    def test_eval_attack_config_from_adv_train_matches_training_attack(self):
        args = self._args(attack_seed=13)
        config = adversarial.AdvTrainConfig(
            enabled=True,
            attack_name="pgd_linf",
            eps=8.0 / 255.0,
            steps=6,
            step_size=2.0 / 255.0,
        )

        eval_config = adversarial.eval_attack_config_from_adv_train(args, config)

        self.assertEqual(eval_config.attack_name, "pgd_linf")
        self.assertEqual(eval_config.attack_seed, 13)
        self.assertAlmostEqual(eval_config.attack_eps, 8.0 / 255.0)
        self.assertEqual(eval_config.attack_steps, 6)
        self.assertAlmostEqual(eval_config.attack_step_size, 2.0 / 255.0)

    def test_eval_attack_config_from_adv_train_defaults_seed_and_fgsm_steps(self):
        args = types.SimpleNamespace(seed=9)
        config = adversarial.AdvTrainConfig(
            enabled=True,
            attack_name="fgsm_linf",
            eps=4.0 / 255.0,
            steps=10,
        )

        eval_config = adversarial.eval_attack_config_from_adv_train(args, config)

        self.assertEqual(eval_config.attack_name, "fgsm_linf")
        self.assertEqual(eval_config.attack_seed, 9)
        self.assertEqual(eval_config.attack_steps, 1)

    def test_copy_args_with_attack_config_sets_eval_attack_fields(self):
        args = self._args(attack_name="none", attack_seed=0)
        config = adversarial.AttackConfig(
            attack_name="pgd_linf",
            attack_seed=5,
            attack_eps=8.0 / 255.0,
            attack_steps=4,
            attack_step_size=2.0 / 255.0,
        )

        updated_args = adversarial.copy_args_with_attack_config(args, config)

        self.assertEqual(updated_args.attack_name, "pgd_linf")
        self.assertEqual(updated_args.attack_seed, 5)
        self.assertAlmostEqual(updated_args.attack_eps, 8.0 / 255.0)
        self.assertEqual(updated_args.attack_steps, 4)
        self.assertAlmostEqual(updated_args.attack_step_size, 2.0 / 255.0)
        self.assertEqual(args.attack_name, "none")

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

    def test_compute_reference_pairs_marks_queries_without_positives_invalid(self):
        reference_pairs = adversarial.compute_reference_pairs(
            clean_query_features=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            database_features=np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32),
            positives_per_query=[
                np.array([0], dtype=np.int64),
                np.array([], dtype=np.int64),
            ],
            query_indices=[0, 1],
        )

        self.assertTrue(reference_pairs["valid_mask"][0])
        self.assertFalse(reference_pairs["valid_mask"][1])
        self.assertEqual(reference_pairs["positive_indices"][0], 0)
        self.assertEqual(reference_pairs["skipped_query_indices"].tolist(), [1])

    def test_apply_whitebox_attack_leaves_invalid_queries_unchanged(self):
        model = DummyDescriptorModel()
        inputs = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
        database_descriptors = torch.stack(
            [
                torch.nn.functional.normalize(torch.nn.functional.one_hot(torch.tensor(0), num_classes=48).float(), dim=0),
                torch.nn.functional.normalize(torch.ones(48), dim=0),
            ],
            dim=0,
        )
        reference_pairs = {
            "positive_indices": [0, 0],
            "negative_indices": [1, 1],
            "valid_mask": [True, False],
        }
        config = adversarial.AttackConfig(
            attack_name="fgsm_linf",
            attack_seed=3,
            attack_eps=8.0 / 255.0,
            attack_steps=1,
        )

        attacked = adversarial.apply_whitebox_attack(
            inputs=inputs,
            sample_ids=[0, 1],
            config=config,
            model=model,
            database_descriptors=database_descriptors,
            reference_pairs=reference_pairs,
        )

        self.assertTrue(torch.allclose(attacked[1], inputs[1]))

    def test_validate_training_arguments_rejects_missing_eps(self):
        args = types.SimpleNamespace(
            adv_train=True,
            adv_train_attack="fgsm_linf",
            adv_train_eps=None,
            adv_train_steps=10,
            adv_train_query_index=0,
            adv_train_weight=1.0,
            adv_train_clean_weight=1.0,
            adv_train_log_interval=10,
        )
        with self.assertRaises(ValueError):
            parser_module.validate_training_arguments(args)

    def test_validate_training_arguments_rejects_invalid_pgd_steps(self):
        args = types.SimpleNamespace(
            adv_train=True,
            adv_train_attack="pgd_linf",
            adv_train_eps=8.0 / 255.0,
            adv_train_steps=0,
            adv_train_query_index=0,
            adv_train_weight=1.0,
            adv_train_clean_weight=1.0,
            adv_train_log_interval=10,
        )
        with self.assertRaises(ValueError):
            parser_module.validate_training_arguments(args)

    def test_select_training_query_indices(self):
        indices = adversarial.select_training_query_indices(batch_size=3, images_per_place=4, query_index=1)
        self.assertTrue(torch.equal(indices, torch.tensor([1, 5, 9], dtype=torch.long)))

    def test_build_training_descriptor_targets(self):
        descriptors = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.5, 0.0],
                [10.0, 0.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        query_indices = torch.tensor([0, 2], dtype=torch.long)

        valid_indices, positive_targets, negative_targets = adversarial.build_training_descriptor_targets(
            descriptors,
            labels,
            query_indices,
        )

        self.assertTrue(torch.equal(valid_indices.cpu(), query_indices))
        self.assertTrue(torch.equal(positive_targets[0], descriptors[1]))
        self.assertTrue(torch.equal(positive_targets[1], descriptors[3]))
        self.assertTrue(torch.equal(negative_targets[0], descriptors[2]))
        self.assertTrue(torch.equal(negative_targets[1], descriptors[0]))

    def test_fgsm_training_attack_respects_bounds(self):
        model = DummyDescriptorModel()
        inputs = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        positive_descriptors = torch.nn.functional.normalize(torch.zeros((1, 48)), dim=1)
        negative_descriptors = torch.nn.functional.normalize(torch.ones((1, 48)), dim=1)
        config = adversarial.AdvTrainConfig(
            enabled=True,
            attack_name="fgsm_linf",
            eps=8.0 / 255.0,
            steps=1,
        )

        attacked = adversarial.generate_training_adversarial_queries(
            inputs=inputs,
            sample_ids=[0],
            config=config,
            model=model,
            positive_descriptors=positive_descriptors,
            negative_descriptors=negative_descriptors,
        )

        attacked_pixels = adversarial.denormalize_tensor(attacked)
        original_pixels = adversarial.denormalize_tensor(inputs)
        max_delta = torch.max(torch.abs(attacked_pixels - original_pixels)).item()

        self.assertLessEqual(max_delta, (8.0 / 255.0) + 1e-6)
        self.assertGreaterEqual(attacked_pixels.min().item(), -1e-6)
        self.assertLessEqual(attacked_pixels.max().item(), 1.0 + 1e-6)

    def test_pgd_training_attack_increases_descriptor_loss(self):
        model = DummyDescriptorModel()
        inputs = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        positive_descriptors = torch.stack(
            [torch.nn.functional.normalize(torch.nn.functional.one_hot(torch.tensor(0), num_classes=48).float(), dim=0)],
            dim=0,
        )
        negative_descriptors = torch.stack(
            [torch.nn.functional.normalize(torch.ones(48), dim=0)],
            dim=0,
        )
        config = adversarial.AdvTrainConfig(
            enabled=True,
            attack_name="pgd_linf",
            eps=8.0 / 255.0,
            steps=4,
            step_size=2.0 / 255.0,
            random_start=True,
        )

        original_loss = adversarial.descriptor_attack_loss(
            model(inputs),
            positive_descriptors,
            negative_descriptors,
        )
        attacked = adversarial.generate_training_adversarial_queries(
            inputs=inputs,
            sample_ids=[0],
            config=config,
            model=model,
            positive_descriptors=positive_descriptors,
            negative_descriptors=negative_descriptors,
        )
        attacked_loss = adversarial.descriptor_attack_loss(
            model(attacked),
            positive_descriptors,
            negative_descriptors,
        )

        self.assertGreaterEqual(attacked_loss.item(), original_loss.item() - 1e-6)

    def test_total_loss_composition_matches_weights(self):
        clean_loss = torch.tensor(2.0)
        adv_loss = torch.tensor(3.0)
        config = adversarial.AdvTrainConfig(enabled=True, clean_weight=0.5, weight=2.0)
        total_loss = (config.clean_weight * clean_loss) + (config.weight * adv_loss)
        self.assertAlmostEqual(total_loss.item(), 7.0)

    def test_build_training_descriptor_targets_skips_when_targets_missing(self):
        descriptors = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        labels = torch.tensor([0, 0], dtype=torch.long)
        query_indices = torch.tensor([0], dtype=torch.long)

        valid_indices, positive_targets, negative_targets = adversarial.build_training_descriptor_targets(
            descriptors,
            labels,
            query_indices,
        )

        self.assertEqual(valid_indices.numel(), 0)
        self.assertEqual(positive_targets.shape[0], 0)
        self.assertEqual(negative_targets.shape[0], 0)


if __name__ == "__main__":
    unittest.main()

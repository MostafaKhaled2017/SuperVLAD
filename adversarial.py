from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torchvision


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

SUPPORTED_ATTACKS = {
    "none",
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
    "brightness_contrast_shift",
    "patch_occlusion",
    "fgsm_linf",
    "pgd_linf",
    "token_mask",
}
SUPPORTED_MASK_MODES = {"none", "random", "center", "block"}
WHITEBOX_ATTACKS = {"fgsm_linf", "pgd_linf"}
TRAINING_ATTACKS = {"fgsm_linf", "pgd_linf"}
CORRUPTION_ATTACKS = {
    "gaussian_noise",
    "gaussian_blur",
    "jpeg_compression",
    "brightness_contrast_shift",
    "patch_occlusion",
}

CORRUPTION_SEVERITY_DEFAULTS = {
    "gaussian_noise": {1: 0.02, 2: 0.05, 3: 0.08},
    "gaussian_blur": {1: {"kernel_size": 3, "sigma": 0.8}, 2: {"kernel_size": 5, "sigma": 1.2}, 3: {"kernel_size": 7, "sigma": 1.8}},
    "jpeg_compression": {1: 75, 2: 50, 3: 25},
    "brightness_contrast_shift": {
        1: {"brightness_shift": -0.08, "contrast": 0.85},
        2: {"brightness_shift": -0.15, "contrast": 0.70},
        3: {"brightness_shift": -0.22, "contrast": 0.55},
    },
    "patch_occlusion": {1: 0.15, 2: 0.25, 3: 0.35},
}


@dataclass(frozen=True)
class AttackConfig:
    attack_name: str = "none"
    attack_severity: int | None = None
    attack_seed: int = 0
    attack_eps: float | None = None
    attack_steps: int | None = None
    attack_step_size: float | None = None
    attack_mask_mode: str = "none"
    attack_keep_ratio: float | None = None
    token_keep_ratio: float = 1.0
    masking_mode: str = "none"

    @property
    def enabled(self) -> bool:
        return self.attack_name != "none"


@dataclass(frozen=True)
class AdvTrainConfig:
    enabled: bool = False
    attack_name: str = "fgsm_linf"
    eps: float | None = None
    steps: int = 10
    step_size: float | None = None
    weight: float = 1.0
    clean_weight: float = 1.0
    query_index: int = 0
    random_start: bool = False
    log_interval: int = 50


def _to_device_tensor(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return value.to(device=ref.device, dtype=ref.dtype)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = _to_device_tensor(IMAGENET_MEAN, tensor)
    std = _to_device_tensor(IMAGENET_STD, tensor)
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = _to_device_tensor(IMAGENET_MEAN, tensor)
    std = _to_device_tensor(IMAGENET_STD, tensor)
    return tensor * std + mean


def normalized_linf_radius(pixel_radius: float, ref: torch.Tensor) -> torch.Tensor:
    std = _to_device_tensor(IMAGENET_STD, ref)
    return torch.full_like(ref[:1], pixel_radius) / std


def normalized_bounds(ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    min_tensor = normalize_tensor(torch.full_like(ref[:1], PIXEL_MIN))
    max_tensor = normalize_tensor(torch.full_like(ref[:1], PIXEL_MAX))
    return min_tensor, max_tensor


def _clamp_normalized(tensor: torch.Tensor) -> torch.Tensor:
    min_tensor, max_tensor = normalized_bounds(tensor)
    return torch.max(torch.min(tensor, max_tensor), min_tensor)


def _clamp_linf_normalized(candidate: torch.Tensor, original: torch.Tensor, pixel_radius: float) -> torch.Tensor:
    eps = normalized_linf_radius(pixel_radius, original)
    bounded = torch.max(torch.min(candidate, original + eps), original - eps)
    return _clamp_normalized(bounded)


def attack_config_from_args(args: Any) -> AttackConfig:
    attack_name = getattr(args, "attack_name", "none")
    attack_keep_ratio = getattr(args, "attack_keep_ratio", None)
    attack_mask_mode = getattr(args, "attack_mask_mode", "none")

    if attack_name == "none" and (
        attack_keep_ratio is not None
        or attack_mask_mode != "none"
        or getattr(args, "token_keep_ratio", 1.0) < 1.0
        or getattr(args, "masking_mode", "none") != "none"
    ):
        attack_name = "token_mask"

    if attack_name == "token_mask":
        resolved_keep_ratio = attack_keep_ratio
        if resolved_keep_ratio is None:
            resolved_keep_ratio = getattr(args, "token_keep_ratio", 1.0)
        resolved_mask_mode = attack_mask_mode if attack_mask_mode != "none" else getattr(args, "masking_mode", "none")
        return AttackConfig(
            attack_name=attack_name,
            attack_seed=getattr(args, "attack_seed", getattr(args, "token_dropout_seed", 0)),
            attack_mask_mode=resolved_mask_mode,
            attack_keep_ratio=resolved_keep_ratio,
            token_keep_ratio=resolved_keep_ratio,
            masking_mode=resolved_mask_mode,
        )

    return AttackConfig(
        attack_name=attack_name,
        attack_severity=getattr(args, "attack_severity", None),
        attack_seed=getattr(args, "attack_seed", 0),
        attack_eps=getattr(args, "attack_eps", None),
        attack_steps=getattr(args, "attack_steps", None),
        attack_step_size=getattr(args, "attack_step_size", None),
        attack_mask_mode=attack_mask_mode,
        attack_keep_ratio=attack_keep_ratio,
    )


def adv_train_config_from_args(args: Any) -> AdvTrainConfig:
    return AdvTrainConfig(
        enabled=getattr(args, "adv_train", False),
        attack_name=getattr(args, "adv_train_attack", "fgsm_linf"),
        eps=getattr(args, "adv_train_eps", None),
        steps=getattr(args, "adv_train_steps", 10),
        step_size=getattr(args, "adv_train_step_size", None),
        weight=getattr(args, "adv_train_weight", 1.0),
        clean_weight=getattr(args, "adv_train_clean_weight", 1.0),
        query_index=getattr(args, "adv_train_query_index", 0),
        random_start=getattr(args, "adv_train_random_start", False),
        log_interval=getattr(args, "adv_train_log_interval", 50),
    )


def eval_attack_config_from_adv_train(args: Any, config: AdvTrainConfig) -> AttackConfig:
    if not config.enabled:
        return AttackConfig()
    if config.attack_name not in WHITEBOX_ATTACKS:
        raise ValueError(
            f"Adversarial training attack {config.attack_name} does not support matched eval conversion"
        )

    attack_steps = 1 if config.attack_name == "fgsm_linf" else int(config.steps)
    attack_seed = getattr(args, "attack_seed", None)
    if attack_seed is None:
        attack_seed = getattr(args, "seed", 0)

    return AttackConfig(
        attack_name=config.attack_name,
        attack_seed=int(attack_seed),
        attack_eps=float(config.eps) if config.eps is not None else None,
        attack_steps=attack_steps,
        attack_step_size=float(config.step_size) if config.step_size is not None else None,
    )


def copy_args_with_attack_config(args: Any, config: AttackConfig) -> Any:
    updated_args = copy.copy(args)
    updated_args.attack_name = config.attack_name
    updated_args.attack_severity = config.attack_severity
    updated_args.attack_seed = config.attack_seed
    updated_args.attack_eps = config.attack_eps
    updated_args.attack_steps = config.attack_steps
    updated_args.attack_step_size = config.attack_step_size
    updated_args.attack_mask_mode = config.attack_mask_mode
    updated_args.attack_keep_ratio = config.attack_keep_ratio
    updated_args.token_keep_ratio = config.token_keep_ratio
    updated_args.masking_mode = config.masking_mode
    return updated_args


def validate_attack_arguments(args: Any) -> None:
    config = attack_config_from_args(args)
    if config.attack_name not in SUPPORTED_ATTACKS:
        raise ValueError(f"Unsupported attack_name: {config.attack_name}")
    if config.enabled and args.test_method != "hard_resize":
        raise ValueError("Adversarial evaluation currently supports only --test_method=hard_resize")
    if config.attack_name not in {"none", "token_mask"} and (
        getattr(args, "token_keep_ratio", 1.0) < 1.0 or getattr(args, "masking_mode", "none") != "none"
    ):
        raise ValueError("Token masking cannot be combined with another attack in a single eval run")
    if config.attack_mask_mode not in SUPPORTED_MASK_MODES:
        raise ValueError(f"Unsupported attack_mask_mode: {config.attack_mask_mode}")
    if config.attack_name == "token_mask":
        if config.attack_keep_ratio is None:
            raise ValueError("token_mask attack requires --attack_keep_ratio or --token_keep_ratio")
        if not 0.0 < config.attack_keep_ratio <= 1.0:
            raise ValueError("attack_keep_ratio must be in (0, 1]")
        if config.attack_mask_mode == "none":
            raise ValueError("token_mask attack requires a masking mode other than 'none'")
    if config.attack_name in CORRUPTION_ATTACKS:
        if config.attack_severity is None:
            setattr(args, "attack_severity", 1)
        elif config.attack_severity not in (1, 2, 3):
            raise ValueError("attack_severity must be 1, 2, or 3 for corruption attacks")
    if config.attack_name in WHITEBOX_ATTACKS:
        if config.attack_eps is None or config.attack_eps <= 0:
            raise ValueError(f"{config.attack_name} requires --attack_eps > 0")
        if getattr(args, "pca_dim", None) is not None:
            raise ValueError("White-box attacks are not supported when PCA evaluation is enabled")
        if config.attack_name == "pgd_linf":
            steps = config.attack_steps if config.attack_steps is not None else 10
            if steps <= 0:
                raise ValueError("attack_steps must be > 0 for pgd_linf")


def attack_config_to_dict(config: AttackConfig) -> dict[str, Any]:
    return asdict(config)


def attack_short_tag(config: AttackConfig) -> str:
    if config.attack_name == "none":
        return "clean"
    parts = [config.attack_name]
    if config.attack_severity is not None:
        parts.append(f"sev{config.attack_severity}")
    if config.attack_eps is not None:
        parts.append(f"eps{config.attack_eps:.6f}")
    if config.attack_steps is not None:
        parts.append(f"steps{config.attack_steps}")
    if config.attack_keep_ratio is not None:
        parts.append(f"keep{config.attack_keep_ratio:.2f}")
    if config.attack_mask_mode != "none":
        parts.append(config.attack_mask_mode)
    parts.append(f"seed{config.attack_seed}")
    return "_".join(parts)


def _sample_seed(base_seed: int, sample_id: int) -> int:
    return int(base_seed) + int(sample_id)


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _random_uniform_like(reference: torch.Tensor, low: float, high: float, seed: int) -> torch.Tensor:
    generator = _cpu_generator(seed)
    tensor = torch.empty(reference.shape, dtype=reference.dtype, device="cpu")
    tensor.uniform_(low, high, generator=generator)
    return tensor.to(reference.device)


def _random_normal_like(reference: torch.Tensor, std: float, seed: int) -> torch.Tensor:
    generator = _cpu_generator(seed)
    tensor = torch.randn(reference.shape, dtype=reference.dtype, device="cpu", generator=generator)
    return (tensor * std).to(reference.device)


def _apply_gaussian_noise(inputs: torch.Tensor, sample_ids: list[int], severity: int, base_seed: int) -> torch.Tensor:
    sigma = CORRUPTION_SEVERITY_DEFAULTS["gaussian_noise"][severity]
    outputs = inputs.clone()
    for batch_pos, sample_id in enumerate(sample_ids):
        noise = _random_normal_like(outputs[batch_pos : batch_pos + 1], sigma, _sample_seed(base_seed, sample_id))
        pixels = denormalize_tensor(outputs[batch_pos : batch_pos + 1])
        pixels = torch.clamp(pixels + noise, PIXEL_MIN, PIXEL_MAX)
        outputs[batch_pos : batch_pos + 1] = normalize_tensor(pixels)
    return outputs


def _apply_gaussian_blur(inputs: torch.Tensor, severity: int) -> torch.Tensor:
    params = CORRUPTION_SEVERITY_DEFAULTS["gaussian_blur"][severity]
    pixels = denormalize_tensor(inputs)
    blurred = torchvision.transforms.functional.gaussian_blur(
        pixels,
        kernel_size=[params["kernel_size"], params["kernel_size"]],
        sigma=[params["sigma"], params["sigma"]],
    )
    return normalize_tensor(torch.clamp(blurred, PIXEL_MIN, PIXEL_MAX))


def _jpeg_roundtrip(sample: torch.Tensor, quality: int) -> torch.Tensor:
    pixels = denormalize_tensor(sample.unsqueeze(0)).squeeze(0).detach().cpu()
    encoded = torchvision.io.encode_jpeg((pixels * 255.0).round().to(torch.uint8), quality=quality)
    decoded = torchvision.io.decode_jpeg(encoded, mode=torchvision.io.ImageReadMode.RGB).float() / 255.0
    return normalize_tensor(decoded.unsqueeze(0)).squeeze(0).to(sample.device, dtype=sample.dtype)


def _apply_jpeg_compression(inputs: torch.Tensor, severity: int) -> torch.Tensor:
    quality = CORRUPTION_SEVERITY_DEFAULTS["jpeg_compression"][severity]
    outputs = inputs.clone()
    for batch_pos in range(outputs.shape[0]):
        outputs[batch_pos] = _jpeg_roundtrip(outputs[batch_pos], quality)
    return outputs


def _apply_brightness_contrast_shift(inputs: torch.Tensor, severity: int) -> torch.Tensor:
    params = CORRUPTION_SEVERITY_DEFAULTS["brightness_contrast_shift"][severity]
    pixels = denormalize_tensor(inputs)
    shifted = ((pixels - 0.5) * params["contrast"]) + 0.5 + params["brightness_shift"]
    return normalize_tensor(torch.clamp(shifted, PIXEL_MIN, PIXEL_MAX))


def _apply_patch_occlusion(inputs: torch.Tensor, sample_ids: list[int], severity: int, base_seed: int) -> torch.Tensor:
    ratio = CORRUPTION_SEVERITY_DEFAULTS["patch_occlusion"][severity]
    outputs = inputs.clone()
    _, _, height, width = outputs.shape
    patch_h = max(1, int(math.ceil(height * ratio)))
    patch_w = max(1, int(math.ceil(width * ratio)))
    for batch_pos, sample_id in enumerate(sample_ids):
        seed = _sample_seed(base_seed, sample_id)
        max_top = max(0, height - patch_h)
        max_left = max(0, width - patch_w)
        top = int(_random_uniform_like(outputs[batch_pos : batch_pos + 1], 0.0, float(max_top + 1), seed)[0, 0, 0, 0].item()) if max_top > 0 else 0
        left = int(_random_uniform_like(outputs[batch_pos : batch_pos + 1], 0.0, float(max_left + 1), seed + 1)[0, 0, 0, 0].item()) if max_left > 0 else 0
        outputs[batch_pos, :, top : top + patch_h, left : left + patch_w] = 0.0
    return outputs


def apply_corruption_attack(inputs: torch.Tensor, sample_ids: list[int], config: AttackConfig) -> torch.Tensor:
    if config.attack_name == "gaussian_noise":
        return _apply_gaussian_noise(inputs, sample_ids, int(config.attack_severity), config.attack_seed)
    if config.attack_name == "gaussian_blur":
        return _apply_gaussian_blur(inputs, int(config.attack_severity))
    if config.attack_name == "jpeg_compression":
        return _apply_jpeg_compression(inputs, int(config.attack_severity))
    if config.attack_name == "brightness_contrast_shift":
        return _apply_brightness_contrast_shift(inputs, int(config.attack_severity))
    if config.attack_name == "patch_occlusion":
        return _apply_patch_occlusion(inputs, sample_ids, int(config.attack_severity), config.attack_seed)
    raise ValueError(f"Unsupported corruption attack: {config.attack_name}")


def compute_reference_pairs(
    clean_query_features: np.ndarray,
    database_features: np.ndarray,
    positives_per_query: list[np.ndarray],
    query_indices: list[int],
) -> dict[str, np.ndarray]:
    positive_indices = np.zeros(len(query_indices), dtype=np.int64)
    negative_indices = np.zeros(len(query_indices), dtype=np.int64)
    valid_mask = np.ones(len(query_indices), dtype=bool)
    skipped_query_indices: list[int] = []

    for row_idx, query_index in enumerate(query_indices):
        query_feature = clean_query_features[row_idx]
        distances = np.sum((database_features - query_feature[None, :]) ** 2, axis=1)
        positives = np.asarray(positives_per_query[query_index], dtype=np.int64)
        if positives.size == 0:
            valid_mask[row_idx] = False
            skipped_query_indices.append(int(query_index))
            continue
        best_positive_offset = int(np.argmin(distances[positives]))
        positive_indices[row_idx] = int(positives[best_positive_offset])

        negative_mask = np.ones(database_features.shape[0], dtype=bool)
        negative_mask[positives] = False
        if not np.any(negative_mask):
            raise ValueError(f"Query {query_index} has no negatives and cannot be used for white-box attacks")
        negative_candidates = np.nonzero(negative_mask)[0]
        best_negative_offset = int(np.argmin(distances[negative_candidates]))
        negative_indices[row_idx] = int(negative_candidates[best_negative_offset])

    return {
        "positive_indices": positive_indices,
        "negative_indices": negative_indices,
        "valid_mask": valid_mask,
        "skipped_query_indices": np.asarray(skipped_query_indices, dtype=np.int64),
    }


def _whitebox_loss(
    descriptors: torch.Tensor,
    database_descriptors: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
) -> torch.Tensor:
    positive_descriptors = database_descriptors.index_select(0, positive_indices)
    negative_descriptors = database_descriptors.index_select(0, negative_indices)
    positive_distance = torch.sum((descriptors - positive_descriptors) ** 2, dim=1)
    negative_distance = torch.sum((descriptors - negative_descriptors) ** 2, dim=1)
    return (positive_distance - negative_distance).mean()


def descriptor_attack_loss(
    query_descriptors: torch.Tensor,
    positive_descriptors: torch.Tensor,
    negative_descriptors: torch.Tensor,
) -> torch.Tensor:
    positive_distance = torch.sum((query_descriptors - positive_descriptors) ** 2, dim=1)
    negative_distance = torch.sum((query_descriptors - negative_descriptors) ** 2, dim=1)
    return (positive_distance - negative_distance).mean()


def select_training_query_indices(batch_size: int, images_per_place: int, query_index: int) -> torch.Tensor:
    if images_per_place <= 0:
        raise ValueError("images_per_place must be > 0")
    if query_index < 0 or query_index >= images_per_place:
        raise ValueError(f"query_index must be in [0, {images_per_place - 1}]")
    return torch.arange(query_index, batch_size * images_per_place, images_per_place, dtype=torch.long)


def build_training_descriptor_targets(
    clean_descriptors: torch.Tensor,
    labels: torch.Tensor,
    query_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_query_indices = []
    positive_targets = []
    negative_targets = []

    for query_index in query_indices.tolist():
        query_descriptor = clean_descriptors[query_index]
        same_label = torch.nonzero(labels == labels[query_index], as_tuple=False).flatten()
        same_label = same_label[same_label != query_index]
        if same_label.numel() == 0:
            continue

        different_label = torch.nonzero(labels != labels[query_index], as_tuple=False).flatten()
        if different_label.numel() == 0:
            continue

        positive_distances = torch.sum((clean_descriptors.index_select(0, same_label) - query_descriptor) ** 2, dim=1)
        negative_distances = torch.sum((clean_descriptors.index_select(0, different_label) - query_descriptor) ** 2, dim=1)

        hardest_positive = same_label[int(torch.argmax(positive_distances).item())]
        hardest_negative = different_label[int(torch.argmin(negative_distances).item())]

        valid_query_indices.append(query_index)
        positive_targets.append(clean_descriptors[hardest_positive].detach())
        negative_targets.append(clean_descriptors[hardest_negative].detach())

    if not valid_query_indices:
        empty_indices = torch.empty((0,), dtype=torch.long, device=labels.device)
        empty_descriptors = clean_descriptors.new_empty((0, clean_descriptors.shape[1]))
        return empty_indices, empty_descriptors, empty_descriptors

    return (
        torch.tensor(valid_query_indices, dtype=torch.long, device=labels.device),
        torch.stack(positive_targets, dim=0),
        torch.stack(negative_targets, dim=0),
    )


def _make_random_start(original: torch.Tensor, pixel_radius: float, sample_ids: list[int], base_seed: int) -> torch.Tensor:
    eps = normalized_linf_radius(pixel_radius, original)
    perturbed = original.clone()
    for batch_pos, sample_id in enumerate(sample_ids):
        noise = _random_uniform_like(
            perturbed[batch_pos : batch_pos + 1],
            low=-1.0,
            high=1.0,
            seed=_sample_seed(base_seed, sample_id),
        )
        perturbed[batch_pos : batch_pos + 1] = perturbed[batch_pos : batch_pos + 1] + noise * eps
    return _clamp_linf_normalized(perturbed, original, pixel_radius)


def _train_attack_step_size(config: AdvTrainConfig) -> float:
    if config.step_size is not None:
        return float(config.step_size)
    if config.attack_name == "pgd_linf":
        return float(config.eps) / 4.0
    return float(config.eps)


def generate_training_adversarial_queries(
    inputs: torch.Tensor,
    sample_ids: list[int],
    config: AdvTrainConfig,
    model: torch.nn.Module,
    positive_descriptors: torch.Tensor,
    negative_descriptors: torch.Tensor,
) -> torch.Tensor:
    if not config.enabled:
        return inputs.detach()
    if config.attack_name not in TRAINING_ATTACKS:
        raise ValueError(f"Unsupported adversarial training attack: {config.attack_name}")
    if inputs.shape[0] == 0:
        return inputs.detach()

    pixel_radius = float(config.eps)
    attack_steps = 1 if config.attack_name == "fgsm_linf" else int(config.steps)
    step_size = _train_attack_step_size(config)
    step_size_normalized = normalized_linf_radius(step_size, inputs)

    original = inputs.detach()
    adv = original.clone()
    if config.attack_name == "pgd_linf" and config.random_start:
        adv = _make_random_start(original, pixel_radius, sample_ids, 0)

    for _ in range(attack_steps):
        adv = adv.detach().requires_grad_(True)
        descriptors = model(adv, queryflag=0)
        loss = descriptor_attack_loss(descriptors, positive_descriptors, negative_descriptors)
        gradient = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
        adv = adv + step_size_normalized * gradient.sign()
        adv = _clamp_linf_normalized(adv.detach(), original, pixel_radius)

    return adv.detach()


def apply_whitebox_attack(
    inputs: torch.Tensor,
    sample_ids: list[int],
    config: AttackConfig,
    model: torch.nn.Module,
    database_descriptors: torch.Tensor,
    reference_pairs: dict[str, np.ndarray],
) -> torch.Tensor:
    pixel_radius = float(config.attack_eps)
    attack_steps = 1 if config.attack_name == "fgsm_linf" else int(config.attack_steps or 10)
    step_size = float(config.attack_step_size) if config.attack_step_size is not None else pixel_radius / 4.0

    valid_mask_value = reference_pairs.get("valid_mask")
    if valid_mask_value is None:
        valid_mask = torch.ones(inputs.shape[0], device=inputs.device, dtype=torch.bool)
    else:
        valid_mask = torch.as_tensor(valid_mask_value, device=inputs.device, dtype=torch.bool)
    if valid_mask.numel() != inputs.shape[0]:
        raise ValueError("reference_pairs valid_mask must match the batch size for white-box attacks")
    if not torch.any(valid_mask):
        return inputs.detach()

    positive_indices = torch.as_tensor(reference_pairs["positive_indices"], device=inputs.device, dtype=torch.long)
    negative_indices = torch.as_tensor(reference_pairs["negative_indices"], device=inputs.device, dtype=torch.long)

    original = inputs.detach()
    adv = original.clone()
    if config.attack_name == "pgd_linf":
        adv = _make_random_start(original, pixel_radius, sample_ids, config.attack_seed)

    step_size_normalized = normalized_linf_radius(step_size, adv)

    for _ in range(attack_steps):
        adv = adv.detach().requires_grad_(True)
        descriptors = model(adv, queryflag=0)
        loss = _whitebox_loss(
            descriptors[valid_mask],
            database_descriptors,
            positive_indices[valid_mask],
            negative_indices[valid_mask],
        )
        loss.backward()
        gradient_sign = adv.grad.sign()
        gradient_sign = gradient_sign * valid_mask.view(-1, 1, 1, 1)
        adv = adv + step_size_normalized * gradient_sign
        adv = _clamp_linf_normalized(adv.detach(), original, pixel_radius)

    return adv.detach()


def apply_query_attack(
    inputs: torch.Tensor,
    sample_ids: list[int],
    config: AttackConfig,
    model: torch.nn.Module | None = None,
    database_descriptors: torch.Tensor | None = None,
    reference_pairs: dict[str, np.ndarray] | None = None,
) -> torch.Tensor:
    if not config.enabled or config.attack_name == "token_mask":
        return inputs
    if config.attack_name in CORRUPTION_ATTACKS:
        return apply_corruption_attack(inputs, sample_ids, config)
    if config.attack_name in WHITEBOX_ATTACKS:
        if model is None or database_descriptors is None or reference_pairs is None:
            raise ValueError("White-box attacks require model, database descriptors, and reference pairs")
        return apply_whitebox_attack(inputs, sample_ids, config, model, database_descriptors, reference_pairs)
    raise ValueError(f"Unsupported query attack: {config.attack_name}")

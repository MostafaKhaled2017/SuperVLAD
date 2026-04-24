import math
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import Tensor, nn

from .config import amp_autocast, unwrap_model
from .losses import compute_attack_score
from .targets import RetrievalAttackBatch


class RetrievalAttackProxy(nn.Module):
    def __init__(self, model: nn.Module, margin: float, mixed_precision: bool, device: str):
        super().__init__()
        self.model = model
        self.margin = margin
        self.mixed_precision = mixed_precision
        self.device = device
        self.current_targets: RetrievalAttackBatch | None = None

    def set_targets(self, targets: RetrievalAttackBatch) -> None:
        self.current_targets = targets

    def clear_targets(self) -> None:
        self.current_targets = None

    def forward(self, inputs: Tensor) -> Tensor:
        if self.current_targets is None:
            raise RuntimeError("RetrievalAttackProxy.forward() called without targets.")

        with amp_autocast(self.mixed_precision, self.device):
            query_descriptors = self.model(inputs, queryflag=0)
        query_descriptors = query_descriptors.float()
        attack_scores = compute_attack_score(
            query_descriptors,
            self.current_targets.positive_descriptors,
            self.current_targets.negative_descriptors,
            self.margin,
        )
        zeros = torch.zeros_like(attack_scores)
        return torch.stack([zeros, attack_scores], dim=1)


class UnsupportedAttack(nn.Module):
    def __init__(self, *args, attack_name: str, **kwargs):
        super().__init__()
        self.attack_name = attack_name

    def forward(self, inputs: Tensor, targets) -> Tensor:
        raise NotImplementedError(
            f"{self.attack_name} is not supported in perceptual_adv_training.py because it depends on "
            "classification-specific AutoAttack behavior."
        )


class RetrievalAttackWrapper(nn.Module):
    backend_cls = None
    default_kwargs: Dict[str, object] = {}

    def __init__(self, model: nn.Module, margin: float = 0.1, mixed_precision: bool = False, device: str = "cuda", **kwargs):
        super().__init__()
        if self.backend_cls is None:
            raise RuntimeError("backend_cls must be defined by subclasses.")
        self.model = model
        self.margin = margin
        self.proxy_model = RetrievalAttackProxy(
            unwrap_model(model),
            margin,
            mixed_precision=mixed_precision,
            device=device,
        )
        merged_kwargs = dict(self.default_kwargs)
        merged_kwargs.update(kwargs)
        self.backend = self.backend_cls(self.proxy_model, **merged_kwargs)

    def forward(self, inputs: Tensor, targets: RetrievalAttackBatch) -> Tensor:
        self.proxy_model.set_targets(targets)
        fake_labels = torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device)
        try:
            return self.backend(inputs, fake_labels)
        finally:
            self.proxy_model.clear_targets()


class NoAttack(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor, targets) -> Tensor:
        return inputs


def build_attack_namespace(model: nn.Module, args) -> Dict[str, object]:
    submodule_root = Path(__file__).resolve().parent.parent / "submodules" / "perceptual-advex"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))

    from perceptual_advex.attacks import (
        FogAttack as BackendFogAttack,
        JPEGLinfAttack as BackendJPEGLinfAttack,
        L1Attack as BackendL1Attack,
        L2Attack as BackendL2Attack,
        LinfAttack as BackendLinfAttack,
        ReColorAdvAttack as BackendReColorAdvAttack,
        StAdvAttack as BackendStAdvAttack,
    )
    from perceptual_advex.perceptual_attacks import (
        FastLagrangePerceptualAttack as BackendFastLagrangePerceptualAttack,
        LagrangePerceptualAttack as BackendLagrangePerceptualAttack,
        PerceptualPGDAttack as BackendPerceptualPGDAttack,
    )

    class LinfAttack(RetrievalAttackWrapper):
        backend_cls = BackendLinfAttack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class L2Attack(RetrievalAttackWrapper):
        backend_cls = BackendL2Attack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class L1Attack(RetrievalAttackWrapper):
        backend_cls = BackendL1Attack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class JPEGLinfAttack(RetrievalAttackWrapper):
        backend_cls = BackendJPEGLinfAttack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class FogAttack(RetrievalAttackWrapper):
        backend_cls = BackendFogAttack

        def __init__(self, model, dataset_name: str = "imagenet", margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                dataset_name=dataset_name,
                **kwargs,
            )

    class StAdvAttack(RetrievalAttackWrapper):
        backend_cls = BackendStAdvAttack

        def __init__(self, model, margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                **kwargs,
            )

    class ReColorAdvAttack(RetrievalAttackWrapper):
        backend_cls = BackendReColorAdvAttack

        def __init__(self, model, margin: float = args.adv_margin, **kwargs):
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                **kwargs,
            )

    class FastLagrangePerceptualAttack(RetrievalAttackWrapper):
        backend_cls = BackendFastLagrangePerceptualAttack

        def __init__(
            self,
            model,
            lpips_model: str = args.lpips_model or "alexnet",
            margin: float = args.adv_margin,
            **kwargs,
        ):
            if lpips_model == "self":
                raise ValueError("perceptual_adv_training.py does not support lpips_model='self' for SuperVLAD.")
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                lpips_model=lpips_model,
                **kwargs,
            )

    class PerceptualPGDAttack(RetrievalAttackWrapper):
        backend_cls = BackendPerceptualPGDAttack

        def __init__(
            self,
            model,
            lpips_model: str = args.lpips_model or "alexnet",
            margin: float = args.adv_margin,
            **kwargs,
        ):
            if lpips_model == "self":
                raise ValueError("perceptual_adv_training.py does not support lpips_model='self' for SuperVLAD.")
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                lpips_model=lpips_model,
                **kwargs,
            )

    class LagrangePerceptualAttack(RetrievalAttackWrapper):
        backend_cls = BackendLagrangePerceptualAttack

        def __init__(
            self,
            model,
            lpips_model: str = args.lpips_model or "alexnet",
            margin: float = args.adv_margin,
            **kwargs,
        ):
            if lpips_model == "self":
                raise ValueError("perceptual_adv_training.py does not support lpips_model='self' for SuperVLAD.")
            super().__init__(
                model,
                margin=margin,
                mixed_precision=args.mixed_precision,
                device=args.device,
                lpips_model=lpips_model,
                **kwargs,
            )

    return {
        "model": unwrap_model(model),
        "NoAttack": NoAttack,
        "LinfAttack": LinfAttack,
        "L2Attack": L2Attack,
        "L1Attack": L1Attack,
        "JPEGLinfAttack": JPEGLinfAttack,
        "FogAttack": FogAttack,
        "StAdvAttack": StAdvAttack,
        "ReColorAdvAttack": ReColorAdvAttack,
        "FastLagrangePerceptualAttack": FastLagrangePerceptualAttack,
        "PerceptualPGDAttack": PerceptualPGDAttack,
        "LagrangePerceptualAttack": LagrangePerceptualAttack,
        "AutoAttack": lambda *a, **k: UnsupportedAttack(*a, attack_name="AutoAttack", **k),
        "AutoLinfAttack": lambda *a, **k: UnsupportedAttack(*a, attack_name="AutoLinfAttack", **k),
        "AutoL2Attack": lambda *a, **k: UnsupportedAttack(*a, attack_name="AutoL2Attack", **k),
        "math": math,
        "torch": torch,
    }


def instantiate_attacks(model: nn.Module, attack_strings: Sequence[str], args):
    namespace = build_attack_namespace(model, args)
    attacks = []
    for attack_string in attack_strings:
        attack = eval(attack_string, {"__builtins__": {}}, namespace)
        attacks.append(attack)
    return attacks

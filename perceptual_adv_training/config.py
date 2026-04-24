import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")

IMAGENET_MEAN_STD = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
IMAGENET_MEAN = torch.tensor(IMAGENET_MEAN_STD["mean"], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(IMAGENET_MEAN_STD["std"], dtype=torch.float32).view(1, 3, 1, 1)

TRAIN_CITIES = [
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS",
]

SUPPORTED_ATTACK_NAMES = {
    "NoAttack",
    "LinfAttack",
    "L2Attack",
    "L1Attack",
    "JPEGLinfAttack",
    "FogAttack",
    "StAdvAttack",
    "ReColorAdvAttack",
    "FastLagrangePerceptualAttack",
    "PerceptualPGDAttack",
    "LagrangePerceptualAttack",
}
UNSUPPORTED_ATTACK_NAMES = {"AutoAttack", "AutoLinfAttack", "AutoL2Attack"}


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def amp_enabled(mixed_precision: bool, device: str) -> bool:
    return bool(mixed_precision and device == "cuda")


def amp_autocast(mixed_precision: bool, device: str) -> ContextManager:
    if amp_enabled(mixed_precision, device):
        return autocast(enabled=True)
    return nullcontext()


def _parse_sm_arch(arch: str) -> Optional[Tuple[int, int]]:
    if not arch.startswith("sm_"):
        return None
    suffix = arch.split("_", 1)[1]
    if not suffix.isdigit():
        return None
    if len(suffix) == 2:
        return int(suffix[0]), int(suffix[1])
    return int(suffix[:-1]), int(suffix[-1])


def validate_cuda_runtime(args) -> None:
    if args.device != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested with --device cuda, but PyTorch cannot access a CUDA device. "
            "Install a CUDA-enabled PyTorch build or rerun with --device cpu."
        )

    arch_list = torch.cuda.get_arch_list()
    supported_arches = [arch for arch in arch_list if arch.startswith("sm_")]
    parsed_arches = [arch for arch in (_parse_sm_arch(arch) for arch in supported_arches) if arch is not None]
    if not parsed_arches:
        return

    device_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    device_capability = torch.cuda.get_device_capability(device_index)
    max_supported_arch = max(parsed_arches)

    if device_capability > max_supported_arch:
        device_sm = f"sm_{device_capability[0]}{device_capability[1]}"
        supported_sm = f"sm_{max_supported_arch[0]}{max_supported_arch[1]}"
        raise RuntimeError(
            "The installed PyTorch CUDA build does not support this GPU architecture. "
            f"Detected GPU '{gpu_name}' with compute capability {device_sm}, but "
            f"PyTorch {torch.__version__} built against CUDA {torch.version.cuda} only includes kernels "
            f"through {supported_sm}. Install a newer torch/torchvision build for this GPU, or rerun "
            "with --device cpu."
        )


def get_normalized_bounds(device: str) -> Tuple[Tensor, Tensor]:
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    min_value = (torch.zeros_like(mean) - mean) / std
    max_value = (torch.ones_like(mean) - mean) / std
    return min_value, max_value


def create_summary_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard logging requires the 'tensorboard' package. "
            "Install the updated requirements.txt and retry."
        ) from exc
    return SummaryWriter(log_dir=log_dir)

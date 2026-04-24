import logging
import os
from pathlib import Path

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import datasets_ws
from .config import IMAGENET_MEAN_STD, TRAIN_CITIES


def resolve_gsv_cities_base_path(args) -> Path:
    candidate_paths = []
    if args.gsv_cities_base_path:
        candidate_paths.append(Path(args.gsv_cities_base_path).expanduser())

    env_base_path = os.environ.get("GSV_CITIES_BASE_PATH")
    if env_base_path:
        env_candidate = Path(env_base_path).expanduser()
        if env_candidate not in candidate_paths:
            candidate_paths.append(env_candidate)

    default_candidate = Path(args.eval_datasets_folder).expanduser() / "gsv_cities"
    if default_candidate not in candidate_paths:
        candidate_paths.append(default_candidate)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    checked_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        "Could not find the GSV-Cities training dataset. Checked: "
        f"{checked_paths}. Pass --gsv_cities_base_path or set GSV_CITIES_BASE_PATH."
    )


def build_training_dataloader(args) -> DataLoader:
    from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
    from torchvision import transforms as T

    image_size = tuple(args.resize)
    train_transform = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
        ]
    )

    gsv_cities_base_path = resolve_gsv_cities_base_path(args)
    logging.info("Using GSV-Cities training data from %s", gsv_cities_base_path)

    train_dataset = GSVCitiesDataset(
        cities=TRAIN_CITIES,
        img_per_place=4,
        min_img_per_place=4,
        random_sample_from_each_place=True,
        transform=train_transform,
        base_path=gsv_cities_base_path,
    )

    return DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
    )


def setup_datasets(args):
    logging.debug("Loading dataset %s from folder %s", args.eval_dataset_name, args.eval_datasets_folder)
    val_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "val")
    test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    logging.info("Val set: %s", val_ds)
    logging.info("Test set: %s", test_ds)
    return val_ds, test_ds


def extract_database_features(args, eval_ds, model: nn.Module) -> np.ndarray:
    eval_ds.test_method = "hard_resize"
    database_subset = Subset(eval_ds, list(range(eval_ds.database_num)))
    dataloader = DataLoader(
        dataset=database_subset,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        pin_memory=(args.device == "cuda"),
    )

    features = np.empty((eval_ds.database_num, args.features_dim), dtype="float32")
    with np.errstate(all="ignore"):
        for inputs, indices in tqdm(dataloader, ncols=100, desc="Database"):
            outputs = model(inputs.to(args.device), queryflag=0).cpu().numpy()
            features[indices.numpy(), :] = outputs
    return features


def extract_clean_query_features(args, eval_ds, model: nn.Module) -> np.ndarray:
    eval_ds.test_method = args.test_method
    query_indices = list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num))
    query_subset = Subset(eval_ds, query_indices)
    dataloader = DataLoader(
        dataset=query_subset,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        pin_memory=(args.device == "cuda"),
    )

    features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")
    with np.errstate(all="ignore"):
        for inputs, indices in tqdm(dataloader, ncols=100, desc="Queries"):
            outputs = model(inputs.to(args.device), queryflag=0).cpu().numpy()
            local_indices = indices.numpy() - eval_ds.database_num
            features[local_indices, :] = outputs
    return features

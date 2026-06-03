import argparse
import copy
import csv
import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import commons
import parser as parser_module
from perceptual_adv_training.cli import parse_attack_names
from perceptual_adv_training.config import validate_cuda_runtime


SUPPORTED_TEST_METHODS = {"hard_resize", "central_crop", "single_query"}
REQUIRED_RECALL_VALUES = (1, 5, 10, 100)
DEFAULT_ATTACKS = (
    "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)",
    "PerceptualPGDAttack(model, bound=0.1, num_iterations=3)",
)


def build_parser() -> argparse.ArgumentParser:
    parser = parser_module.build_parser()
    parser.description = (
        "Evaluate base and perceptually trained SuperVLAD checkpoints on shared clean and perceptual attack data."
    )
    parser.set_defaults(foundation_model_path="checkpoints/dinov2_vitb14_pretrain.pth")
    parser.add_argument(
        "--base_resume",
        type=str,
        default="checkpoints/SuperVLAD.pth",
        help="Path to the base checkpoint.",
    )
    parser.add_argument(
        "--trained_resume",
        type=str,
        default="checkpoints/perceptual_adv_checkpoint.pth",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names under --eval_datasets_folder to evaluate.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        action="append",
        default=None,
        help="Attack expression. Can be passed multiple times. Defaults to FastLagrange and PerceptualPGD.",
    )
    parser.add_argument(
        "--attack_reference",
        type=str,
        default="base",
        choices=["base"],
        help="Model used to generate the shared adversarial query tensors.",
    )
    parser.add_argument("--adv_margin", type=float, default=0.1, help="Retrieval attack objective margin.")
    parser.add_argument("--adv_negatives", type=int, default=5, help="Hard negatives per retrieval attack target.")
    parser.add_argument(
        "--val_batches",
        type=int,
        default=None,
        help="Optional number of query batches to attack. Clean evaluation always uses all queries.",
    )
    parser.add_argument("--lpips_model", type=str, default=None, help="Optional LPIPS model override.")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="JSON report path. Defaults to test/perceptual_eval/<timestamp>/perceptual_eval_results.json.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="CSV report path. Defaults to the JSON path with a .csv suffix.",
    )
    return parser


def parse_arguments():
    args = build_parser().parse_args()
    args = parser_module.validate_arguments(args)
    args.attack = list(DEFAULT_ATTACKS) if args.attack is None else args.attack
    args.recall_values = list(dict.fromkeys([*args.recall_values, *REQUIRED_RECALL_VALUES]))
    validate_arguments(args)
    return args


def validate_arguments(args) -> None:
    if args.resume is not None:
        raise ValueError("Use --base_resume and --trained_resume instead of --resume.")
    if args.pca_dim is not None:
        raise NotImplementedError("perceptual_eval.py does not support PCA.")
    if args.test_method not in SUPPORTED_TEST_METHODS:
        raise ValueError(
            f"perceptual_eval.py supports only {sorted(SUPPORTED_TEST_METHODS)} for --test_method, "
            f"but received {args.test_method!r}."
        )
    if args.adv_negatives < 1:
        raise ValueError("--adv_negatives must be at least 1.")
    if args.val_batches is not None and args.val_batches < 1:
        raise ValueError("--val_batches must be at least 1 when provided.")

    parse_attack_names(args.attack)
    validate_cuda_runtime(args)
    require_file(args.base_resume, "--base_resume")
    require_file(args.trained_resume, "--trained_resume")
    if args.foundation_model_path is not None:
        require_file(args.foundation_model_path, "--foundation_model_path")
    validate_dataset_layouts(args.eval_datasets_folder, args.datasets)


def require_file(path: str, argument_name: str) -> None:
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{argument_name} does not exist: {resolved_path}")


def validate_dataset_layouts(datasets_root: str, dataset_names: Sequence[str]) -> None:
    root = Path(datasets_root).expanduser()
    for dataset_name in dataset_names:
        test_root = root / dataset_name / "images" / "test"
        database_root = test_root / "database"
        queries_root = test_root / "queries"
        if not database_root.is_dir() or not queries_root.is_dir():
            raise FileNotFoundError(
                f"Dataset {dataset_name!r} must contain images/test/database and images/test/queries under {root}."
            )


def build_output_paths(args) -> Tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_json is None:
        run_dir = Path("test") / "perceptual_eval" / timestamp
        output_json = run_dir / "perceptual_eval_results.json"
    else:
        output_json = Path(args.output_json).expanduser()
        run_dir = output_json.parent / f"perceptual_eval_{timestamp}"

    output_csv = Path(args.output_csv).expanduser() if args.output_csv else output_json.with_suffix(".csv")
    if output_json.parent != run_dir:
        output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.parent != run_dir:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    return output_json, output_csv, run_dir


def serialize_args(args) -> Dict[str, object]:
    serialized = {}
    for key, value in vars(args).items():
        serialized[key] = list(value) if isinstance(value, tuple) else value
    return serialized


def clone_model_args(args, checkpoint_path: str):
    model_args = copy.deepcopy(args)
    model_args.resume = checkpoint_path
    return model_args


def load_model(args, checkpoint_path: str) -> Tuple[nn.Module, object]:
    import util
    from model import network

    model_args = clone_model_args(args, checkpoint_path)
    model = network.SuperVLADModel(
        model_args,
        pretrained_foundation=bool(model_args.foundation_model_path),
        foundation_model_path=model_args.foundation_model_path,
    )
    model = model.to(model_args.device)
    model_args.features_dim *= model_args.supervlad_clusters
    util.resume_model(model_args, model)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model, model_args


def query_batch_size(args) -> int:
    return 1 if args.test_method == "single_query" else args.infer_batch_size


def extract_database_features(args, eval_ds, model: nn.Module) -> np.ndarray:
    eval_ds.test_method = "hard_resize"
    database_subset = Subset(eval_ds, range(eval_ds.database_num))
    dataloader = DataLoader(
        database_subset,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    features = np.empty((eval_ds.database_num, args.features_dim), dtype=np.float32)
    with torch.inference_mode():
        for inputs, indices in tqdm(dataloader, ncols=100, desc="Database"):
            descriptors = model(inputs.to(args.device), queryflag=0).cpu().numpy()
            features[indices.numpy(), :] = descriptors
    return features


def extract_query_features(args, eval_ds, model: nn.Module) -> np.ndarray:
    eval_ds.test_method = args.test_method
    query_indices = range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)
    query_subset = Subset(eval_ds, query_indices)
    dataloader = DataLoader(
        query_subset,
        batch_size=query_batch_size(args),
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    features = np.empty((eval_ds.queries_num, args.features_dim), dtype=np.float32)
    with torch.inference_mode():
        for inputs, indices in tqdm(dataloader, ncols=100, desc="Queries"):
            descriptors = model(inputs.to(args.device), queryflag=0).cpu().numpy()
            local_indices = indices.numpy() - eval_ds.database_num
            features[local_indices, :] = descriptors
    return features


def extract_clean_features(args, eval_ds, model: nn.Module) -> Dict[str, np.ndarray]:
    return {
        "database": extract_database_features(args, eval_ds, model),
        "queries": extract_query_features(args, eval_ds, model),
    }


def attack_display_name(attack: nn.Module, name_counts: MutableMapping[str, int]) -> str:
    base_name = attack.__class__.__name__
    name_counts[base_name] = name_counts.get(base_name, 0) + 1
    if name_counts[base_name] == 1:
        return base_name
    return f"{base_name}_{name_counts[base_name]}"


def make_attack_batch(
    args,
    eval_ds,
    targets: Sequence[Mapping[str, object]],
    database_features: torch.Tensor,
    query_features: np.ndarray,
):
    from perceptual_adv_training.targets import RetrievalAttackBatch

    query_tensors = []
    positive_descriptors = []
    negative_descriptors = []
    query_indices = []

    for target in targets:
        query_index = int(target["query_index"])
        query_tensor, _ = eval_ds[eval_ds.database_num + query_index]
        query_tensors.append(query_tensor)
        query_indices.append(query_index)

        positive_descriptors.append(database_features[int(target["positive_index"])])
        negative_indexes = torch.as_tensor(target["negative_indexes"], dtype=torch.long, device=args.device)
        negative_descriptors.append(database_features[negative_indexes])

    attack_targets = RetrievalAttackBatch(
        query_indices=torch.arange(len(targets), dtype=torch.long, device=args.device),
        clean_query_descriptors=torch.from_numpy(query_features[query_indices]).to(args.device),
        positive_descriptors=torch.stack(positive_descriptors, dim=0),
        negative_descriptors=torch.stack(negative_descriptors, dim=0),
    )
    return torch.stack(query_tensors, dim=0).to(args.device), attack_targets


def evaluate_shared_attack(
    args,
    eval_ds,
    attack: nn.Module,
    attack_name: str,
    models: Mapping[str, Tuple[nn.Module, object]],
    targets: Sequence[Mapping[str, object]],
    base_features: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    eval_ds.test_method = args.test_method
    batch_size = query_batch_size(args)
    database_features = torch.from_numpy(base_features["database"]).to(args.device)
    attacked_features = {
        model_label: np.empty((len(targets), model_args.features_dim), dtype=np.float32)
        for model_label, (_, model_args) in models.items()
    }

    for offset in tqdm(range(0, len(targets), batch_size), ncols=100, desc=f"Attack {attack_name}"):
        batch_targets = targets[offset : offset + batch_size]
        query_inputs, attack_targets = make_attack_batch(
            args,
            eval_ds,
            batch_targets,
            database_features,
            base_features["queries"],
        )
        attacked_queries = attack(query_inputs, attack_targets)

        with torch.inference_mode():
            for model_label, (model, _) in models.items():
                descriptors = model(attacked_queries, queryflag=0).cpu().numpy()
                attacked_features[model_label][offset : offset + len(batch_targets), :] = descriptors

    return attacked_features


def clean_results(
    model_args: Mapping[str, object],
    features_by_model: Mapping[str, Mapping[str, np.ndarray]],
    positives_per_query,
) -> Dict[str, Dict[str, object]]:
    from perceptual_adv_training.eval import compute_recalls_from_features

    results = {}
    for model_label, features in features_by_model.items():
        results[model_label] = {
            "clean": compute_recalls_from_features(
                model_args[model_label],
                features["database"],
                features["queries"],
                positives_per_query,
            )
        }
    return results


def add_attack_results(
    results: MutableMapping[str, Dict[str, object]],
    model_args: Mapping[str, object],
    features_by_model: Mapping[str, Mapping[str, np.ndarray]],
    attack_name: str,
    attacked_features: Mapping[str, np.ndarray],
    positives_per_query,
    query_counts: Mapping[str, int],
) -> None:
    from perceptual_adv_training.eval import compute_recalls_from_features

    for model_label, query_features in attacked_features.items():
        metrics = compute_recalls_from_features(
            model_args[model_label],
            features_by_model[model_label]["database"],
            query_features,
            positives_per_query,
        )
        metrics["attacked_queries"] = int(query_counts["attacked_queries"])
        metrics["skipped_queries_without_positives"] = int(query_counts["skipped_queries_without_positives"])
        results[model_label][attack_name] = metrics


def evaluate_dataset(args, dataset_name: str, models: Mapping[str, Tuple[nn.Module, object]], attacks: Sequence[nn.Module]):
    import datasets_ws
    from perceptual_adv_training.targets import build_attack_targets

    eval_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, dataset_name, "test")
    logging.info("Test set: %s", eval_ds)

    feature_times = {}
    features_by_model = {}
    model_args = {}
    for model_label, (model, current_args) in models.items():
        logging.info("Extracting clean descriptors for %s on %s.", model_label, dataset_name)
        start_time = perf_counter()
        features_by_model[model_label] = extract_clean_features(current_args, eval_ds, model)
        feature_times[model_label] = perf_counter() - start_time
        model_args[model_label] = current_args

    positives = eval_ds.get_positives()
    results = clean_results(model_args, features_by_model, positives)

    limit_queries = None
    if args.val_batches is not None:
        limit_queries = args.val_batches * query_batch_size(args)

    start_time = perf_counter()
    targets, valid_query_indices = build_attack_targets(
        args,
        eval_ds,
        features_by_model["base"]["database"],
        features_by_model["base"]["queries"],
        limit_queries=limit_queries,
    )
    target_time = perf_counter() - start_time
    attacked_positives = [positives[index] for index in valid_query_indices]
    query_counts = {
        "total_queries": int(eval_ds.queries_num),
        "attacked_queries": int(len(valid_query_indices)),
        "skipped_queries_without_positives": int(eval_ds.queries_num - len(valid_query_indices)),
    }
    logging.info(
        "Attacking %d/%d queries on %s.",
        query_counts["attacked_queries"],
        query_counts["total_queries"],
        dataset_name,
    )

    attack_times = {}
    attack_names = {}
    for attack in attacks:
        attack_name = attack_display_name(attack, attack_names)
        logging.info("Generating shared %s examples on %s from the base checkpoint.", attack_name, dataset_name)
        start_time = perf_counter()
        attacked_features = evaluate_shared_attack(
            args,
            eval_ds,
            attack,
            attack_name,
            models,
            targets,
            features_by_model["base"],
        )
        attack_times[attack_name] = perf_counter() - start_time
        add_attack_results(results, model_args, features_by_model, attack_name, attacked_features, attacked_positives, query_counts)

    runtimes = {
        "feature_seconds": feature_times,
        "target_seconds": target_time,
        "attack_seconds": attack_times,
    }
    return results, runtimes, query_counts


def flatten_rows(results: Mapping[str, Mapping[str, Mapping[str, object]]], recall_values: Iterable[int]) -> List[Dict[str, object]]:
    rows = []
    for dataset_name, models in results.items():
        for model_label, conditions in models.items():
            for condition_name, metrics in conditions.items():
                row = {
                    "dataset": dataset_name,
                    "model": model_label,
                    "condition": condition_name,
                    "recalls_str": metrics["recalls_str"],
                }
                for recall_value in recall_values:
                    row[f"R@{recall_value}"] = float(metrics["recalls"][f"R@{recall_value}"])
                rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], recall_values: Sequence[int]) -> None:
    fieldnames = ["dataset", "model", "condition", *[f"R@{value}" for value in recall_values], "recalls_str"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    output_json, output_csv, run_dir = build_output_paths(args)
    args.save_dir = str(run_dir)

    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info("Arguments: %s", args)
    logging.info("The outputs are being saved in %s", run_dir)

    started_at = datetime.now()
    base_model, base_args = load_model(args, args.base_resume)
    trained_model, trained_args = load_model(args, args.trained_resume)
    models = {
        "base": (base_model, base_args),
        "trained": (trained_model, trained_args),
    }
    from perceptual_adv_training.attacks import instantiate_attacks

    attacks = instantiate_attacks(base_model, args.attack, base_args)

    results = {}
    runtimes = {}
    query_counts = {}
    for dataset_name in args.datasets:
        logging.info("Evaluating %s.", dataset_name)
        dataset_results, dataset_runtimes, dataset_query_counts = evaluate_dataset(args, dataset_name, models, attacks)
        results[dataset_name] = dataset_results
        runtimes[dataset_name] = dataset_runtimes
        query_counts[dataset_name] = dataset_query_counts

    rows = flatten_rows(results, args.recall_values)
    report = {
        "timestamp": started_at.isoformat(),
        "command": " ".join(shlex.quote(argument) for argument in sys.argv),
        "argv": sys.argv,
        "checkpoints": {
            "base": args.base_resume,
            "trained": args.trained_resume,
            "attack_reference": args.attack_reference,
        },
        "datasets": list(args.datasets),
        "attack_expressions": list(args.attack),
        "arguments": serialize_args(args),
        "query_counts": query_counts,
        "results": results,
        "runtime_seconds": runtimes,
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "duration_seconds": (datetime.now() - started_at).total_seconds(),
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_csv(output_csv, rows, args.recall_values)

    logging.info("Saved JSON comparison report to %s", output_json)
    logging.info("Saved CSV comparison summary to %s", output_csv)
    logging.info("Finished in %s", str(datetime.now() - started_at)[:-7])


if __name__ == "__main__":
    main()

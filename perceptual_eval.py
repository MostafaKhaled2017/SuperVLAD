import argparse
import copy
import csv
import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

import commons
import parser as parser_module
from perceptual_adv_training.cli import parse_attack_names
from perceptual_adv_training.config import validate_cuda_runtime


SUPPORTED_ATTACK_TEST_METHODS = {"hard_resize", "single_query", "central_crop"}


def build_parser():
    parser = parser_module.build_parser()
    parser.description = "Compare baseline and perceptually adversarially trained checkpoints across datasets."
    parser.add_argument(
        "--base_resume",
        type=str,
        required=True,
        help="Path to the baseline checkpoint.",
    )
    parser.add_argument(
        "--trained_resume",
        type=str,
        required=True,
        help="Path to the perceptually adversarially trained checkpoint.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Benchmark dataset names under --eval_datasets_folder to evaluate.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        action="append",
        default=[],
        help="Attack expression(s) to evaluate, following perceptual_adv_training.py syntax.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to the combined JSON comparison report.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV summary path. Defaults to the JSON path with a .csv suffix.",
    )
    return parser


def parse_arguments():
    args = build_parser().parse_args()
    args = parser_module.validate_arguments(args)
    validate_evaluation_arguments(args)
    return args


def validate_evaluation_arguments(args):
    if args.pca_dim is not None:
        raise NotImplementedError("perceptual_eval.py does not support PCA because attacked descriptors must remain differentiable.")
    if args.resume is not None:
        raise ValueError("Use --base_resume and --trained_resume instead of --resume with perceptual_eval.py.")
    if args.test_method not in SUPPORTED_ATTACK_TEST_METHODS:
        raise ValueError(
            f"perceptual_eval.py supports only {sorted(SUPPORTED_ATTACK_TEST_METHODS)} for --test_method, "
            f"but received {args.test_method!r}"
        )
    if len(args.attack) == 0:
        raise ValueError("At least one --attack expression is required.")

    parse_attack_names(args.attack)
    validate_cuda_runtime(args)

    checkpoint_paths = {
        "base_resume": Path(args.base_resume).expanduser(),
        "trained_resume": Path(args.trained_resume).expanduser(),
    }
    for argument_name, checkpoint_path in checkpoint_paths.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"{argument_name} checkpoint does not exist: {checkpoint_path}")

    datasets_root = Path(args.eval_datasets_folder).expanduser()
    for dataset_name in args.datasets:
        test_root = datasets_root / dataset_name / "images" / "test"
        database_dir = test_root / "database"
        queries_dir = test_root / "queries"
        if not database_dir.is_dir() or not queries_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset {dataset_name!r} is missing the expected test layout under {test_root}. "
                "Expected both 'database' and 'queries' directories."
            )


def setup_run_logging(save_dir: Path):
    commons.setup_logging(str(save_dir))
    logging.info("The outputs are being saved in %s", save_dir)


def build_output_paths(args):
    output_json = Path(args.output_json).expanduser()
    output_csv = Path(args.output_csv).expanduser() if args.output_csv is not None else output_json.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = output_json.parent / f"perceptual_eval_{timestamp}"
    return output_json, output_csv, save_dir


def serialize_args(args):
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


def build_model(eval_args):
    from model import network
    import util

    model = network.SuperVLADModel(eval_args)
    model = model.to(eval_args.device)
    eval_args.features_dim *= eval_args.supervlad_clusters
    model = util.resume_model(eval_args, model)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model


def normalize_condition_name(condition_name: str) -> str:
    if condition_name == "NoAttack":
        return "clean"
    return condition_name


def evaluate_checkpoint_on_dataset(base_args, checkpoint_path: str, dataset_name: str) -> Dict[str, object]:
    import datasets_ws
    from perceptual_adv_training.attacks import instantiate_attacks
    from perceptual_adv_training.eval import evaluate_against_attacks_retrieval

    eval_args = copy.deepcopy(base_args)
    eval_args.resume = checkpoint_path
    eval_args.eval_dataset_name = dataset_name

    model = build_model(eval_args)
    attacks = instantiate_attacks(model, eval_args.attack, eval_args)
    eval_ds = datasets_ws.BaseDataset(eval_args, eval_args.eval_datasets_folder, dataset_name, "test")
    metrics = evaluate_against_attacks_retrieval(eval_args, model, eval_ds, attacks)

    normalized_metrics = {}
    for condition_name, condition_metrics in metrics.items():
        normalized_metrics[normalize_condition_name(condition_name)] = condition_metrics
    return normalized_metrics


def flatten_csv_rows(dataset_results: Dict[str, Dict[str, Dict[str, object]]], recall_values: List[int]):
    rows = []
    for dataset_name, models in dataset_results.items():
        for model_label, conditions in models.items():
            for condition_name, condition_metrics in conditions.items():
                row = {
                    "dataset": dataset_name,
                    "model": model_label,
                    "condition": condition_name,
                    "recalls_str": condition_metrics["recalls_str"],
                }
                for recall_value in recall_values:
                    row[f"R@{recall_value}"] = float(condition_metrics["recalls"][f"R@{recall_value}"])
                rows.append(row)
    return rows


def write_csv(output_csv: Path, rows, recall_values: List[int]):
    fieldnames = ["dataset", "model", "condition", *[f"R@{recall_value}" for recall_value in recall_values], "recalls_str"]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    output_json, output_csv, save_dir = build_output_paths(args)
    args.save_dir = str(save_dir)

    setup_run_logging(save_dir)
    commons.make_deterministic(args.seed)
    logging.info("Arguments: %s", args)

    model_specs = {
        "base": args.base_resume,
        "trained": args.trained_resume,
    }

    started_at = datetime.now()
    datasets_report = {}
    runtime_seconds = {}

    for dataset_name in args.datasets:
        logging.info("Evaluating dataset %s", dataset_name)
        datasets_report[dataset_name] = {}
        runtime_seconds[dataset_name] = {}

        for model_label, checkpoint_path in model_specs.items():
            logging.info("Evaluating %s checkpoint on %s: %s", model_label, dataset_name, checkpoint_path)
            model_started_at = datetime.now()
            datasets_report[dataset_name][model_label] = evaluate_checkpoint_on_dataset(args, checkpoint_path, dataset_name)
            runtime_seconds[dataset_name][model_label] = (datetime.now() - model_started_at).total_seconds()

    csv_rows = flatten_csv_rows(datasets_report, list(args.recall_values))
    report = {
        "timestamp": started_at.isoformat(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "argv": sys.argv,
        "checkpoints": {
            "base": args.base_resume,
            "trained": args.trained_resume,
        },
        "datasets": list(args.datasets),
        "attack_expressions": list(args.attack),
        "arguments": serialize_args(args),
        "results": datasets_report,
        "runtime_seconds": runtime_seconds,
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "duration_seconds": (datetime.now() - started_at).total_seconds(),
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_csv(output_csv, csv_rows, list(args.recall_values))

    logging.info("Saved JSON comparison report to %s", output_json)
    logging.info("Saved CSV comparison summary to %s", output_csv)
    logging.info("Finished in %s", str(datetime.now() - started_at)[:-7])


if __name__ == "__main__":
    main()

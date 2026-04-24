import json
import logging
import re
import shlex
import sys
from datetime import datetime
from os.path import join
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import parser as parser_module


SUPPORTED_ATTACK_TEST_METHODS = {"hard_resize", "single_query", "central_crop"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def build_parser():
    parser = parser_module.build_parser()
    parser.description = "FGSM robustness evaluation for visual geolocalization checkpoints"
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        required=True,
        help="FGSM epsilon values to evaluate. One clean run is always added automatically.",
    )
    parser.add_argument(
        "--fgsm_loss",
        type=str,
        default="positive_distance",
        choices=["positive_distance", "wrong_match", "training_style"],
        help="FGSM objective used to generate adversarial query images.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to the JSON report. If omitted, a timestamped run directory is used.",
    )
    parser.add_argument(
        "--fgsm_negatives",
        type=int,
        default=5,
        help="Number of nearest non-positive database descriptors used for the triplet-style FGSM objective.",
    )
    parser.add_argument(
        "--fgsm_margin",
        type=float,
        default=0.1,
        help="Margin used by the triplet-style FGSM objective.",
    )
    parser.add_argument(
        "--resume_eval_dir",
        type=str,
        default=None,
        help="Existing FGSM evaluation directory whose info.log is used to skip completed clean/epsilon runs.",
    )
    return parser


def parse_arguments():
    args = build_parser().parse_args()
    args = parser_module.validate_arguments(args)

    if args.resume is None:
        raise ValueError("--resume is required for fgsm_eval.py")
    if args.pca_dim is not None:
        raise NotImplementedError("fgsm_eval.py does not support PCA because FGSM needs differentiable descriptors.")
    if args.test_method not in SUPPORTED_ATTACK_TEST_METHODS:
        raise ValueError(
            f"fgsm_eval.py supports only {sorted(SUPPORTED_ATTACK_TEST_METHODS)} for --test_method, "
            f"but received {args.test_method!r}"
        )
    if args.fgsm_negatives < 1:
        raise ValueError("--fgsm_negatives must be at least 1")
    if any(eps < 0 for eps in args.epsilons):
        raise ValueError("--epsilons must be non-negative")

    return args


def get_normalized_bounds(device):
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    min_value = (torch.zeros_like(mean) - mean) / std
    max_value = (torch.ones_like(mean) - mean) / std
    return min_value, max_value


def format_recalls(recalls, recall_values):
    recalls = [float(rec) for rec in recalls]
    return {
        "recalls": {f"R@{value}": recall for value, recall in zip(recall_values, recalls)},
        "recalls_list": recalls,
        "recalls_str": ", ".join([f"R@{value}: {recall:.1f}" for value, recall in zip(recall_values, recalls)]),
    }


def parse_recalls_string(recalls_str, recall_values):
    parsed_recalls = {}
    for recall_name, recall_value in re.findall(r"R@(\d+):\s*([0-9]+(?:\.[0-9]+)?)", recalls_str):
        parsed_recalls[int(recall_name)] = float(recall_value)

    if len(parsed_recalls) != len(recall_values) or any(recall_value not in parsed_recalls for recall_value in recall_values):
        raise ValueError(f"Could not parse recall string {recalls_str!r}")

    return format_recalls([parsed_recalls[recall_value] for recall_value in recall_values], recall_values)


def load_logged_results(run_dir, recall_values):
    info_log_path = Path(run_dir) / "info.log"
    if not info_log_path.exists():
        raise FileNotFoundError(f"Could not find info.log in {run_dir}")

    clean_pattern = re.compile(r"Clean recalls on .*: (?P<recalls>R@\d+:.*)$")
    epsilon_pattern = re.compile(r"FGSM eps=(?P<epsilon>[^ ]+) recalls on .*: (?P<recalls>R@\d+:.*)$")
    query_count_pattern = re.compile(
        r"FGSM attack will evaluate (?P<applied>\d+)/(?P<total>\d+) queries with at least one positive\."
    )

    recovered_results = {}
    query_counts = None
    with info_log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            clean_match = clean_pattern.search(line)
            if clean_match is not None:
                recovered_results["clean"] = parse_recalls_string(clean_match.group("recalls"), recall_values)
                continue

            epsilon_match = epsilon_pattern.search(line)
            if epsilon_match is not None:
                epsilon = float(epsilon_match.group("epsilon"))
                epsilon_result = parse_recalls_string(epsilon_match.group("recalls"), recall_values)
                epsilon_result["epsilon"] = float(epsilon)
                recovered_results[f"eps_{epsilon:g}"] = epsilon_result
                continue

            query_count_match = query_count_pattern.search(line)
            if query_count_match is not None:
                applied_queries = int(query_count_match.group("applied"))
                total_queries = int(query_count_match.group("total"))
                query_counts = {
                    "total_queries": total_queries,
                    "applied_queries": applied_queries,
                    "skipped_queries": total_queries - applied_queries,
                }

    return recovered_results, query_counts


def fill_logged_query_counts(results, query_counts):
    if query_counts is None:
        return

    for result_key, result_value in results.items():
        if result_key == "clean":
            continue
        result_value.setdefault("attacked_queries", query_counts["applied_queries"])
        result_value.setdefault("skipped_queries_without_positives", query_counts["skipped_queries"])


def infer_query_counts(results, eval_ds):
    for result_key, result_value in results.items():
        if result_key == "clean":
            continue
        if "attacked_queries" in result_value and "skipped_queries_without_positives" in result_value:
            return {
                "total_queries": int(result_value["attacked_queries"] + result_value["skipped_queries_without_positives"]),
                "applied_queries": int(result_value["attacked_queries"]),
                "skipped_queries": int(result_value["skipped_queries_without_positives"]),
            }

    return {
        "total_queries": int(eval_ds.queries_num),
        "applied_queries": 0,
        "skipped_queries": int(eval_ds.queries_num),
    }


def compute_recalls(args, database_features, query_features, positives_per_query):
    import faiss

    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    _, predictions = faiss_index.search(query_features, max(args.recall_values))

    recalls = np.zeros(len(args.recall_values), dtype=np.float32)
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.isin(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / len(positives_per_query) * 100
    return format_recalls(recalls, args.recall_values)


def get_query_batch_size(args):
    return 1 if args.test_method == "single_query" else args.infer_batch_size


def extract_database_features(args, eval_ds, model):
    eval_ds.test_method = "hard_resize"
    database_subset = Subset(eval_ds, list(range(eval_ds.database_num)))
    dataloader = DataLoader(
        dataset=database_subset,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        pin_memory=(args.device == "cuda"),
    )

    features = np.empty((eval_ds.database_num, args.features_dim), dtype="float32")
    with torch.no_grad():
        for inputs, indices in tqdm(dataloader, ncols=100, desc="Database"):
            outputs = model(inputs.to(args.device), queryflag=0).cpu().numpy()
            features[indices.numpy(), :] = outputs
    return features


def extract_clean_query_features(args, eval_ds, model):
    eval_ds.test_method = args.test_method
    query_indices = list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num))
    query_subset = Subset(eval_ds, query_indices)
    dataloader = DataLoader(
        dataset=query_subset,
        num_workers=args.num_workers,
        batch_size=get_query_batch_size(args),
        pin_memory=(args.device == "cuda"),
    )

    features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")
    with torch.no_grad():
        for inputs, indices in tqdm(dataloader, ncols=100, desc="Clean queries"):
            outputs = model(inputs.to(args.device), queryflag=0).cpu().numpy()
            local_indices = indices.numpy() - eval_ds.database_num
            features[local_indices, :] = outputs
    return features


def build_attack_targets(eval_ds, database_features, clean_query_features, fgsm_loss, fgsm_negatives):
    import faiss

    targets = []
    positives_per_query = eval_ds.get_positives()
    valid_query_indices = np.flatnonzero(
        np.fromiter((len(positive_candidates) > 0 for positive_candidates in positives_per_query), dtype=bool)
    )

    if len(valid_query_indices) == 0:
        raise RuntimeError("No queries with positives were found, cannot run FGSM evaluation.")

    negative_count = fgsm_negatives if fgsm_loss == "training_style" else 1
    database_features = np.ascontiguousarray(database_features.astype(np.float32, copy=False))
    valid_query_features = np.ascontiguousarray(clean_query_features[valid_query_indices].astype(np.float32, copy=False))

    search_k = min(
        eval_ds.database_num,
        max(128, negative_count + 32),
    )
    faiss_index = faiss.IndexFlatL2(database_features.shape[1])
    faiss_index.add(database_features)
    _, ranked_neighbors = faiss_index.search(valid_query_features, search_k)

    fallback_queries = 0
    progress_desc = "Attack targets"

    for target_offset, query_index in enumerate(tqdm(valid_query_indices, ncols=100, desc=progress_desc)):
        positive_candidates = np.asarray(positives_per_query[query_index], dtype=np.int64)
        query_feature = valid_query_features[target_offset]

        positive_distances = np.sum((database_features[positive_candidates] - query_feature[None, :]) ** 2, axis=1)
        positive_idx = int(positive_candidates[np.argmin(positive_distances)])

        neighbor_candidates = ranked_neighbors[target_offset]
        negative_indexes = neighbor_candidates[~np.isin(neighbor_candidates, positive_candidates)]

        if len(negative_indexes) < negative_count:
            fallback_queries += 1
            distances = np.sum((database_features - query_feature[None, :]) ** 2, axis=1)
            positive_mask = np.zeros(eval_ds.database_num, dtype=bool)
            positive_mask[positive_candidates] = True
            negative_candidates = np.flatnonzero(~positive_mask)
            negative_order = np.argsort(distances[negative_candidates])
            negative_indexes = negative_candidates[negative_order]

        if len(negative_indexes) == 0:
            raise RuntimeError(f"Query {query_index} has no non-positive database candidates.")

        negative_indexes = negative_indexes[:negative_count].astype(np.int64)

        targets.append(
            {
                "query_index": query_index,
                "positive_index": positive_idx,
                "negative_indexes": negative_indexes,
            }
        )

    if fallback_queries != 0:
        logging.info(
            "Fell back to exact full-database negative search for %d/%d queries.",
            fallback_queries,
            len(valid_query_indices),
        )

    return targets, valid_query_indices


def compute_attack_loss(query_descriptor, positive_descriptor, negative_descriptors, args):
    if args.fgsm_loss == "positive_distance":
        return F.pairwise_distance(query_descriptor, positive_descriptor).mean()

    positive_similarity = F.cosine_similarity(query_descriptor, positive_descriptor).mean()
    if args.fgsm_loss == "wrong_match":
        negative_similarity = F.cosine_similarity(query_descriptor, negative_descriptors[:1]).mean()
        return negative_similarity - positive_similarity

    positive_distance = F.pairwise_distance(query_descriptor, positive_descriptor)
    repeated_query = query_descriptor.repeat(negative_descriptors.shape[0], 1)
    negative_distances = F.pairwise_distance(repeated_query, negative_descriptors)
    return F.relu(args.fgsm_margin + positive_distance - negative_distances).mean()


def generate_attacked_query_features(args, eval_ds, model, database_features_tensor, targets, epsilon):
    eval_ds.test_method = args.test_method
    min_value, max_value = get_normalized_bounds(args.device)
    attacked_features = np.empty((len(targets), args.features_dim), dtype="float32")

    for target_offset, target in enumerate(tqdm(targets, ncols=100, desc=f"FGSM eps={epsilon:g}")):
        query_index = target["query_index"]
        global_index = eval_ds.database_num + query_index
        clean_query, _ = eval_ds[global_index]
        clean_query = clean_query.unsqueeze(0).to(args.device)
        clean_query.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        query_descriptor = model(clean_query, queryflag=0)

        positive_descriptor = database_features_tensor[target["positive_index"]].unsqueeze(0)
        negative_indexes = target["negative_indexes"]
        negative_descriptors = database_features_tensor[negative_indexes]

        attack_loss = compute_attack_loss(query_descriptor, positive_descriptor, negative_descriptors, args)
        gradients = torch.autograd.grad(attack_loss, clean_query)[0]
        adversarial_query = torch.clamp(clean_query + epsilon * gradients.sign(), min=min_value, max=max_value)

        with torch.no_grad():
            attacked_features[target_offset, :] = model(adversarial_query, queryflag=0).squeeze(0).cpu().numpy()

    return attacked_features


def serialize_args(args):
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


def build_output_path(args):
    if args.output_json is not None:
        output_path = Path(args.output_json)
    else:
        output_path = Path(args.save_dir) / "fgsm_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def main():
    args = parse_arguments()

    global commons, datasets_ws, util, network

    import commons
    import datasets_ws
    import util
    from model import network
    start_time = datetime.now()
    recovered_results = {}
    recovered_query_counts = None
    if args.resume_eval_dir is not None:
        recovered_results, recovered_query_counts = load_logged_results(args.resume_eval_dir, args.recall_values)
        args.save_dir = str(Path(args.resume_eval_dir))
        commons.setup_logging(args.save_dir, allow_existing=True)
    else:
        args.save_dir = join("test", args.save_dir, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
        commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    if args.resume_eval_dir is not None:
        recovered_keys = sorted(recovered_results.keys())
        logging.info("Recovered completed FGSM evaluation entries from %s: %s", args.resume_eval_dir, recovered_keys)

    model = network.SuperVLADModel(args)
    model = model.to(args.device)
    args.features_dim *= args.supervlad_clusters

    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
    model = torch.nn.DataParallel(model)
    model.eval()

    eval_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    logging.info(f"Test set: {eval_ds}")

    requested_result_keys = ["clean", *[f"eps_{epsilon:g}" for epsilon in args.epsilons]]
    results = {result_key: recovered_results[result_key] for result_key in requested_result_keys if result_key in recovered_results}
    fill_logged_query_counts(results, recovered_query_counts)
    pending_epsilons = [epsilon for epsilon in args.epsilons if f"eps_{epsilon:g}" not in results]
    query_counts = recovered_query_counts

    if "clean" in results and len(pending_epsilons) == 0:
        logging.info("Recovered clean run and all requested epsilons from %s. Skipping feature extraction.", args.resume_eval_dir)
    else:
        database_features = extract_database_features(args, eval_ds, model)
        clean_query_features = extract_clean_query_features(args, eval_ds, model)
        positives_per_query = eval_ds.get_positives()

        if "clean" in results:
            logging.info("Skipping clean recall evaluation because it is already completed in %s.", args.resume_eval_dir)
        else:
            clean_results = compute_recalls(args, database_features, clean_query_features, positives_per_query)
            results["clean"] = clean_results
            logging.info(f"Clean recalls on {eval_ds}: {clean_results['recalls_str']}")

        if len(pending_epsilons) != 0:
            target_build_start = perf_counter()
            logging.info("Building FGSM attack targets.")
            targets, valid_query_indices = build_attack_targets(
                eval_ds,
                database_features,
                clean_query_features,
                args.fgsm_loss,
                args.fgsm_negatives,
            )
            logging.info("Built FGSM attack targets in %.1f seconds.", perf_counter() - target_build_start)
            attack_positives_per_query = [positives_per_query[index] for index in valid_query_indices]
            query_counts = {
                "total_queries": int(eval_ds.queries_num),
                "applied_queries": int(len(valid_query_indices)),
                "skipped_queries": int(eval_ds.queries_num - len(valid_query_indices)),
            }
            logging.info(
                "FGSM attack will evaluate %d/%d queries with at least one positive.",
                query_counts["applied_queries"],
                query_counts["total_queries"],
            )
            fill_logged_query_counts(results, query_counts)
            database_features_tensor = torch.from_numpy(database_features).to(args.device)

            for epsilon in pending_epsilons:
                attacked_query_features = generate_attacked_query_features(
                    args,
                    eval_ds,
                    model,
                    database_features_tensor,
                    targets,
                    epsilon,
                )
                epsilon_results = compute_recalls(
                    args,
                    database_features,
                    attacked_query_features,
                    attack_positives_per_query,
                )
                epsilon_results["epsilon"] = float(epsilon)
                epsilon_results["attacked_queries"] = query_counts["applied_queries"]
                epsilon_results["skipped_queries_without_positives"] = query_counts["skipped_queries"]
                results[f"eps_{epsilon:g}"] = epsilon_results
                logging.info(f"FGSM eps={epsilon:g} recalls on {eval_ds}: {epsilon_results['recalls_str']}")
        else:
            fill_logged_query_counts(results, recovered_query_counts)

    ordered_results = {}
    for result_key in requested_result_keys:
        if result_key in results:
            ordered_results[result_key] = results[result_key]
    query_counts = infer_query_counts(ordered_results, eval_ds) if query_counts is None else query_counts

    output_path = build_output_path(args)
    report = {
        "timestamp": start_time.isoformat(),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "argv": sys.argv,
        "checkpoint": args.resume,
        "dataset": args.eval_dataset_name,
        "arguments": serialize_args(args),
        "attack": {
            "mode": args.fgsm_loss,
            "epsilons": [float(epsilon) for epsilon in args.epsilons],
            "scope": "queries_only",
            "database_features_reused": True,
            "supported_test_methods": sorted(SUPPORTED_ATTACK_TEST_METHODS),
            "query_counts": query_counts,
        },
        "results": ordered_results,
        "output_json": str(output_path),
        "duration_seconds": (datetime.now() - start_time).total_seconds(),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logging.info(f"Saved FGSM evaluation report to {output_path}")
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")


if __name__ == "__main__":
    main()

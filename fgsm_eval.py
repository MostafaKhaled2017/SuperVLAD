import json
import logging
import shlex
import sys
from datetime import datetime
from os.path import join
from pathlib import Path

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
    targets = []
    valid_query_indices = []
    positives_per_query = eval_ds.get_positives()

    for query_index, positive_candidates in enumerate(positives_per_query):
        if len(positive_candidates) == 0:
            continue

        query_feature = clean_query_features[query_index]
        distances = np.sum((database_features - query_feature[None, :]) ** 2, axis=1)

        positive_candidates = np.asarray(positive_candidates, dtype=np.int64)
        positive_idx = int(positive_candidates[np.argmin(distances[positive_candidates])])

        positive_mask = np.zeros(eval_ds.database_num, dtype=bool)
        positive_mask[positive_candidates] = True
        negative_candidates = np.flatnonzero(~positive_mask)
        negative_order = np.argsort(distances[negative_candidates])
        negative_candidates = negative_candidates[negative_order]

        if len(negative_candidates) == 0:
            raise RuntimeError(f"Query {query_index} has no non-positive database candidates.")

        if fgsm_loss == "training_style":
            negative_indexes = negative_candidates[:fgsm_negatives].astype(np.int64)
        else:
            negative_indexes = negative_candidates[:1].astype(np.int64)

        targets.append(
            {
                "query_index": query_index,
                "positive_index": positive_idx,
                "negative_indexes": negative_indexes,
            }
        )
        valid_query_indices.append(query_index)

    if not targets:
        raise RuntimeError("No queries with positives were found, cannot run FGSM evaluation.")

    return targets, np.asarray(valid_query_indices, dtype=np.int64)


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
    args.save_dir = join("test", args.save_dir, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")

    model = network.SuperVLADModel(args)
    model = model.to(args.device)
    args.features_dim *= args.supervlad_clusters

    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
    model = torch.nn.DataParallel(model)
    model.eval()

    eval_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    logging.info(f"Test set: {eval_ds}")

    database_features = extract_database_features(args, eval_ds, model)
    clean_query_features = extract_clean_query_features(args, eval_ds, model)
    positives_per_query = eval_ds.get_positives()
    clean_results = compute_recalls(args, database_features, clean_query_features, positives_per_query)
    logging.info(f"Clean recalls on {eval_ds}: {clean_results['recalls_str']}")

    targets, valid_query_indices = build_attack_targets(
        eval_ds,
        database_features,
        clean_query_features,
        args.fgsm_loss,
        args.fgsm_negatives,
    )
    attack_positives_per_query = [positives_per_query[index] for index in valid_query_indices]
    logging.info(
        "FGSM attack will evaluate %d/%d queries with at least one positive.",
        len(valid_query_indices),
        eval_ds.queries_num,
    )
    database_features_tensor = torch.from_numpy(database_features).to(args.device)

    results = {"clean": clean_results}
    for epsilon in args.epsilons:
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
        epsilon_results["attacked_queries"] = int(len(valid_query_indices))
        epsilon_results["skipped_queries_without_positives"] = int(eval_ds.queries_num - len(valid_query_indices))
        results[f"eps_{epsilon:g}"] = epsilon_results
        logging.info(f"FGSM eps={epsilon:g} recalls on {eval_ds}: {epsilon_results['recalls_str']}")

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
            "query_counts": {
                "total_queries": int(eval_ds.queries_num),
                "applied_queries": int(len(valid_query_indices)),
                "skipped_queries": int(eval_ds.queries_num - len(valid_query_indices)),
            },
        },
        "results": results,
        "output_json": str(output_path),
        "duration_seconds": (datetime.now() - start_time).total_seconds(),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logging.info(f"Saved FGSM evaluation report to {output_path}")
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")


if __name__ == "__main__":
    main()

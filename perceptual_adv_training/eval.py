from typing import Dict, Optional, Sequence

import faiss
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .config import unwrap_model
from .data import extract_clean_query_features, extract_database_features
from .targets import RetrievalAttackBatch, build_attack_targets


def format_recalls(recalls: Sequence[float], recall_values: Sequence[int]) -> Dict[str, object]:
    recall_list = [float(rec) for rec in recalls]
    recall_map = {f"R@{value}": recall for value, recall in zip(recall_values, recall_list)}
    recall_str = ", ".join([f"R@{value}: {recall:.1f}" for value, recall in zip(recall_values, recall_list)])
    return {
        "recalls": recall_map,
        "recalls_list": recall_list,
        "recalls_str": recall_str,
    }


def compute_recalls_from_features(
    args,
    database_features: np.ndarray,
    query_features: np.ndarray,
    positives_per_query,
) -> Dict[str, object]:
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(np.ascontiguousarray(database_features.astype(np.float32, copy=False)))
    _, predictions = faiss_index.search(
        np.ascontiguousarray(query_features.astype(np.float32, copy=False)),
        max(args.recall_values),
    )

    recalls = np.zeros(len(args.recall_values), dtype=np.float32)
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.isin(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / len(positives_per_query) * 100
    return format_recalls(recalls, args.recall_values)


def make_attack_name(attack: nn.Module) -> str:
    return attack.__class__.__name__


def evaluate_against_attacks_retrieval(
    args,
    model: nn.Module,
    eval_ds,
    attacks: Sequence[nn.Module],
    writer=None,
    iteration: Optional[int] = None,
):
    descriptor_model = unwrap_model(model)
    descriptor_model.eval()

    database_features = extract_database_features(args, eval_ds, descriptor_model)
    clean_query_features = extract_clean_query_features(args, eval_ds, descriptor_model)

    limit_queries = None
    if args.val_batches is not None:
        limit_queries = args.val_batches * max(1, args.infer_batch_size)

    targets, valid_query_indices = build_attack_targets(
        args,
        eval_ds,
        database_features,
        clean_query_features,
        limit_queries=limit_queries,
    )
    positives = eval_ds.get_positives()
    valid_positives = [positives[index] for index in valid_query_indices]
    clean_results = compute_recalls_from_features(
        args,
        database_features,
        clean_query_features[valid_query_indices],
        valid_positives,
    )

    metrics = {"NoAttack": clean_results}
    print("ATTACK NoAttack", clean_results["recalls_str"], sep="\t")
    if writer is not None and iteration is not None:
        for recall_value, recall_metric in zip(args.recall_values, clean_results["recalls_list"]):
            writer.add_scalar(f"val/NoAttack/R@{recall_value}", float(recall_metric), iteration)

    if len(attacks) == 0:
        return metrics

    eval_ds.test_method = args.test_method
    database_features_tensor = torch.from_numpy(database_features).to(args.device)

    for attack in attacks:
        attack_name = make_attack_name(attack)
        attacked_query_features = np.empty((len(targets), args.features_dim), dtype="float32")

        for offset in tqdm(range(0, len(targets), args.infer_batch_size), ncols=100, desc=f"Attack {attack_name}"):
            batch_targets = targets[offset : offset + args.infer_batch_size]
            query_tensors = []
            positive_tensors = []
            negative_tensors = []
            for target in batch_targets:
                global_index = eval_ds.database_num + int(target["query_index"])
                query_tensor, _ = eval_ds[global_index]
                query_tensors.append(query_tensor)
                positive_tensors.append(database_features_tensor[int(target["positive_index"])])
                negative_indexes = torch.as_tensor(target["negative_indexes"], dtype=torch.long, device=args.device)
                negative_tensors.append(database_features_tensor[negative_indexes])

            query_inputs = torch.stack(query_tensors, dim=0).to(args.device)
            attack_batch = RetrievalAttackBatch(
                query_indices=torch.arange(len(batch_targets), device=args.device),
                clean_query_descriptors=torch.from_numpy(
                    clean_query_features[[int(target["query_index"]) for target in batch_targets]]
                ).to(args.device),
                positive_descriptors=torch.stack(positive_tensors, dim=0),
                negative_descriptors=torch.stack(negative_tensors, dim=0),
            )
            adv_queries = attack(query_inputs, attack_batch)
            with torch.no_grad():
                batch_features = descriptor_model(adv_queries, queryflag=0).cpu().numpy()
            attacked_query_features[offset : offset + len(batch_targets), :] = batch_features

        attack_results = compute_recalls_from_features(
            args,
            database_features,
            attacked_query_features,
            valid_positives,
        )
        metrics[attack_name] = attack_results
        print(f"ATTACK {attack_name}", attack_results["recalls_str"], sep="\t")
        if writer is not None and iteration is not None:
            for recall_value, recall_metric in zip(args.recall_values, attack_results["recalls_list"]):
                writer.add_scalar(f"val/{attack_name}/R@{recall_value}", float(recall_metric), iteration)

    return metrics

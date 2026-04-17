from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import adversarial


SUPPORTED_TEST_METHODS = [
    "hard_resize",
    "single_query",
    "central_crop",
    "five_crops",
    "nearest_crop",
    "maj_voting",
]

RETRIEVAL_DIAGNOSTICS_MAX_BUFFER_BYTES = 64 * 1024 * 1024


def top_n_voting(topn: str, predictions: np.ndarray, distances: np.ndarray, majority_weight: float) -> None:
    if topn == "top1":
        top_n = 1
    elif topn == "top5":
        top_n = 5
    elif topn == "top10":
        top_n = 10
    else:
        raise ValueError(f"Unsupported top_n_voting mode: {topn}")

    top_predictions = predictions[:, :top_n]
    unique_predictions, counts = np.unique(top_predictions, return_counts=True)
    for pred, count in zip(unique_predictions[counts > 1], counts[counts > 1]):
        distances[predictions == pred] -= majority_weight * count


def _positive_count_bin_label(positive_count: int) -> str:
    if positive_count <= 0:
        return "positives_0"
    if positive_count == 1:
        return "positives_1"
    if positive_count <= 4:
        return "positives_2_4"
    if positive_count <= 9:
        return "positives_5_9"
    return "positives_10_plus"


def _compute_retrieval_per_bin_metrics(per_query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[str, list[dict[str, Any]]] = {}
    for row in per_query_rows:
        bins.setdefault(row["bin_label"], []).append(row)

    bin_order = ["positives_0", "positives_1", "positives_2_4", "positives_5_9", "positives_10_plus"]
    per_bin_rows = []
    for bin_label in bin_order:
        rows = bins.get(bin_label, [])
        if len(rows) == 0:
            continue
        margins = np.array([row["margin"] for row in rows if row["margin"] is not None], dtype=np.float32)
        per_bin_rows.append({
            "bin_label": bin_label,
            "query_count": len(rows),
            "recall_at_1": float(np.mean([row["top1_correct"] for row in rows]) * 100.0),
            "recall_at_5": float(np.mean([row["success_at_5"] for row in rows]) * 100.0),
            "recall_at_10": float(np.mean([row["success_at_10"] for row in rows]) * 100.0),
            "mean_margin": float(np.mean(margins)) if len(margins) > 0 else None,
            "median_margin": float(np.median(margins)) if len(margins) > 0 else None,
            "negative_margin_rate": float(np.mean(margins < 0.0)) if len(margins) > 0 else None,
        })
    return per_bin_rows


def _expand_token_dropout_ids(indices: torch.Tensor, test_method: str) -> np.ndarray:
    sample_ids = indices.numpy()
    if test_method in ["five_crops", "nearest_crop", "maj_voting"]:
        return np.repeat(sample_ids, 5)
    return sample_ids


def _clear_retrieval_outputs(eval_ds: Any) -> None:
    eval_ds.retrieval_query_diagnostics = None
    eval_ds.retrieval_diagnostics_meta = None
    eval_ds.retrieval_query_debug = None
    eval_ds.retrieval_cluster_mass_stats = None
    eval_ds.retrieval_per_bin_metrics = None


def _database_cache_payload(args: Any, eval_ds: Any) -> dict[str, Any]:
    return {
        "dataset_name": eval_ds.dataset_name,
        "database_num": eval_ds.database_num,
        "resume": args.resume,
        "backbone": args.backbone,
        "supervlad_clusters": args.supervlad_clusters,
        "ghost_clusters": args.ghost_clusters,
        "crossimage_encoder": args.crossimage_encoder,
        "resize": list(args.resize),
        "features_dim": args.features_dim,
        "pca_dim": args.pca_dim,
        "split": "test_database",
    }


def _database_cache_paths(args: Any, eval_ds: Any) -> tuple[Path, Path] | tuple[None, None]:
    if args.feature_cache_dir is None:
        return None, None
    payload = _database_cache_payload(args, eval_ds)
    payload_json = json.dumps(payload, sort_keys=True)
    key = hashlib.md5(payload_json.encode("utf-8")).hexdigest()
    cache_dir = Path(args.feature_cache_dir)
    return cache_dir / f"database_features_{key}.npy", cache_dir / f"database_features_{key}.json"


def _compute_batch_descriptors(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    args: Any,
    token_dropout_ids: np.ndarray,
    return_debug: bool = False,
    token_keep_ratio: float = 1.0,
    token_dropout_seed: int | None = None,
    masking_mode: str = "none",
) -> Any:
    return model(
        inputs.to(args.device),
        queryflag=0,
        return_debug=return_debug,
        low_mass_threshold=getattr(args, "low_mass_threshold", 1e-3),
        token_keep_ratio=token_keep_ratio,
        token_dropout_seed=token_dropout_seed,
        token_dropout_ids=token_dropout_ids,
        masking_mode=masking_mode,
    )


def _extract_database_features(args: Any, eval_ds: Any, model: torch.nn.Module, pca: Any = None) -> np.ndarray:
    cache_features_path, cache_meta_path = _database_cache_paths(args, eval_ds)
    if cache_features_path is not None and cache_features_path.exists():
        logging.info("Loading cached database descriptors from %s", cache_features_path)
        return np.load(cache_features_path)

    eval_ds.test_method = "hard_resize"
    database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
    database_dataloader = DataLoader(
        dataset=database_subset_ds,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        pin_memory=(args.device == "cuda"),
    )

    database_features = np.empty((eval_ds.database_num, args.features_dim), dtype="float32")
    model = model.eval()
    with torch.no_grad():
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            descriptors = _compute_batch_descriptors(
                model,
                inputs,
                args,
                token_dropout_ids=indices.numpy(),
                token_keep_ratio=1.0,
                token_dropout_seed=None,
                masking_mode="none",
            )
            descriptors = descriptors.cpu().numpy()
            if pca is not None:
                descriptors = pca.transform(descriptors)
            database_features[indices.numpy(), :] = descriptors

    if cache_features_path is not None and cache_meta_path is not None:
        cache_features_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_features_path, database_features)
        cache_meta_path.write_text(json.dumps(_database_cache_payload(args, eval_ds), indent=2))

    return database_features


def _query_mask_settings(args: Any, attack_config: adversarial.AttackConfig) -> tuple[float, int | None, str]:
    if attack_config.attack_name == "token_mask":
        return attack_config.token_keep_ratio, attack_config.attack_seed, attack_config.attack_mask_mode
    return 1.0, None, "none"


def _prepare_query_inputs(inputs: torch.Tensor, test_method: str) -> torch.Tensor:
    if test_method in ["five_crops", "nearest_crop", "maj_voting"]:
        return torch.cat(tuple(inputs))
    return inputs


def _extract_query_debug_rows(
    pool_debug: dict[str, torch.Tensor],
    query_indexes: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_debug_rows = []
    cluster_mass_rows = []
    per_cluster_mass = pool_debug["per_cluster_mass"].cpu().numpy()
    for batch_pos, query_index in enumerate(query_indexes):
        token_count_value = pool_debug["token_count"]
        if torch.is_tensor(token_count_value):
            if token_count_value.numel() == 1:
                token_count_value = int(token_count_value.item())
            else:
                token_count_value = int(token_count_value[batch_pos].item())
        else:
            token_count_value = int(token_count_value)

        query_debug_rows.append({
            "query_index": int(query_index),
            "token_count": token_count_value,
            "min_mass": float(pool_debug["min_mass"][batch_pos].item()),
            "max_mass": float(pool_debug["max_mass"][batch_pos].item()),
            "mean_mass": float(pool_debug["mean_mass"][batch_pos].item()),
            "p10_mass": float(pool_debug["p10_mass"][batch_pos].item()),
            "num_low_mass_clusters": int(pool_debug["num_low_mass_clusters"][batch_pos].item()),
            "assignment_entropy": float(pool_debug["assignment_entropy"][batch_pos].item()),
        })
        cluster_row = {"query_index": int(query_index)}
        for cluster_idx, cluster_mass in enumerate(per_cluster_mass[batch_pos]):
            cluster_row[f"cluster_mass_{cluster_idx}"] = float(cluster_mass)
        cluster_mass_rows.append(cluster_row)
    return query_debug_rows, cluster_mass_rows


def _extract_query_features(
    args: Any,
    eval_ds: Any,
    model: torch.nn.Module,
    test_method: str,
    pca: Any,
    database_features: np.ndarray,
    attack_config: adversarial.AttackConfig,
) -> tuple[np.ndarray, list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
    eval_ds.test_method = test_method
    queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
    queries_dataloader = DataLoader(
        dataset=queries_subset_ds,
        num_workers=args.num_workers,
        batch_size=queries_infer_batch_size,
        pin_memory=(args.device == "cuda"),
    )

    if test_method in ["nearest_crop", "maj_voting"]:
        query_features = np.empty((5 * eval_ds.queries_num, args.features_dim), dtype="float32")
    else:
        query_features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")

    retrieval_debug = getattr(args, "enable_retrieval_diagnostics", False) and getattr(args, "return_debug_metrics", False)
    query_debug_rows = [None] * eval_ds.queries_num if retrieval_debug else None
    cluster_mass_rows = [None] * eval_ds.queries_num if retrieval_debug else None

    database_descriptors_device = None
    if attack_config.attack_name in adversarial.WHITEBOX_ATTACKS:
        database_descriptors_device = torch.as_tensor(database_features, device=args.device, dtype=torch.float32)

    model = model.eval()
    for inputs, indices in tqdm(queries_dataloader, ncols=100):
        prepared_inputs = _prepare_query_inputs(inputs, test_method)
        token_dropout_ids = _expand_token_dropout_ids(indices, test_method)
        query_indices = (indices.numpy() - eval_ds.database_num).tolist()
        attack_inputs = prepared_inputs.to(args.device)

        if attack_config.attack_name in adversarial.CORRUPTION_ATTACKS:
            attack_inputs = adversarial.apply_query_attack(
                attack_inputs,
                query_indices,
                attack_config,
            )
        elif attack_config.attack_name in adversarial.WHITEBOX_ATTACKS:
            with torch.no_grad():
                clean_descriptors = _compute_batch_descriptors(
                    model,
                    prepared_inputs,
                    args,
                    token_dropout_ids=token_dropout_ids,
                    token_keep_ratio=1.0,
                    token_dropout_seed=None,
                    masking_mode="none",
                ).cpu().numpy()
            reference_pairs = adversarial.compute_reference_pairs(
                clean_query_features=clean_descriptors,
                database_features=database_features,
                positives_per_query=eval_ds.get_positives(),
                query_indices=query_indices,
            )
            skipped_query_indices = reference_pairs.get("skipped_query_indices")
            if skipped_query_indices is not None and len(skipped_query_indices) > 0:
                logging.warning(
                    "Skipping white-box attack for %d queries in this batch because they have no positives: %s",
                    len(skipped_query_indices),
                    skipped_query_indices.tolist(),
                )
            attack_inputs = adversarial.apply_query_attack(
                attack_inputs,
                query_indices,
                attack_config,
                model=model,
                database_descriptors=database_descriptors_device,
                reference_pairs=reference_pairs,
            )

        token_keep_ratio, token_dropout_seed, masking_mode = _query_mask_settings(args, attack_config)
        if retrieval_debug and test_method not in ["five_crops", "nearest_crop", "maj_voting"]:
            with torch.no_grad():
                outputs = _compute_batch_descriptors(
                    model,
                    attack_inputs,
                    args,
                    token_dropout_ids=token_dropout_ids,
                    return_debug=True,
                    token_keep_ratio=token_keep_ratio,
                    token_dropout_seed=token_dropout_seed,
                    masking_mode=masking_mode,
                )
            descriptors = outputs["descriptor"]
            debug_rows, cluster_rows = _extract_query_debug_rows(
                outputs["pool_debug"],
                indices.numpy() - eval_ds.database_num,
            )
            for row in debug_rows:
                query_debug_rows[row["query_index"]] = row
            for row in cluster_rows:
                cluster_mass_rows[row["query_index"]] = row
        else:
            with torch.no_grad():
                descriptors = _compute_batch_descriptors(
                    model,
                    attack_inputs,
                    args,
                    token_dropout_ids=token_dropout_ids,
                    token_keep_ratio=token_keep_ratio,
                    token_dropout_seed=token_dropout_seed,
                    masking_mode=masking_mode,
                )

        if test_method == "five_crops":
            descriptors = torch.stack(torch.split(descriptors, 5)).mean(1)

        descriptors_np = descriptors.cpu().numpy()
        if pca is not None:
            descriptors_np = pca.transform(descriptors_np)

        if test_method in ["nearest_crop", "maj_voting"]:
            start_idx = (indices[0] - eval_ds.database_num) * 5
            end_idx = start_idx + indices.shape[0] * 5
            query_features[start_idx:end_idx, :] = descriptors_np
        else:
            query_features[indices.numpy() - eval_ds.database_num, :] = descriptors_np

    if retrieval_debug:
        query_debug_rows = [row for row in query_debug_rows if row is not None]
        cluster_mass_rows = [row for row in cluster_mass_rows if row is not None]
    return query_features, query_debug_rows, cluster_mass_rows


def _search_predictions(
    args: Any,
    eval_ds: Any,
    database_features: np.ndarray,
    query_features: np.ndarray,
    test_method: str,
) -> tuple[faiss.IndexFlatL2, np.ndarray, np.ndarray]:
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    distances, predictions = faiss_index.search(query_features, max(args.recall_values))

    if test_method == "nearest_crop":
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for query_index in range(eval_ds.queries_num):
            sort_idx = np.argsort(distances[query_index])
            predictions[query_index] = predictions[query_index, sort_idx]
            _, unique_idx = np.unique(predictions[query_index], return_index=True)
            predictions[query_index, :20] = predictions[query_index, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]
    elif test_method == "maj_voting":
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for query_index in range(eval_ds.queries_num):
            top_n_voting("top1", predictions[query_index], distances[query_index], args.majority_weight)
            top_n_voting("top5", predictions[query_index], distances[query_index], args.majority_weight)
            top_n_voting("top10", predictions[query_index], distances[query_index], args.majority_weight)
            flattened_distances = distances[query_index].flatten()
            flattened_predictions = predictions[query_index].flatten()
            sort_idx = np.argsort(flattened_distances)
            flattened_predictions = flattened_predictions[sort_idx]
            _, unique_idx = np.unique(flattened_predictions, return_index=True)
            predictions[query_index, 0, :20] = flattened_predictions[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]

    return faiss_index, distances, predictions


def _compute_recalls(args: Any, eval_ds: Any, predictions: np.ndarray) -> tuple[np.ndarray, str]:
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for recall_index, recall_at in enumerate(args.recall_values):
            if np.any(np.isin(pred[:recall_at], positives_per_query[query_index])):
                recalls[recall_index:] += 1
                break
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str


def _populate_retrieval_diagnostics(
    args: Any,
    eval_ds: Any,
    faiss_index: faiss.IndexFlatL2,
    query_features: np.ndarray,
    test_method: str,
    query_debug_rows: list[dict[str, Any]] | None,
    cluster_mass_rows: list[dict[str, Any]] | None,
) -> None:
    retrieval_debug = getattr(args, "return_debug_metrics", False)
    attack_config = adversarial.attack_config_from_args(args)
    attack_metadata = adversarial.attack_config_to_dict(attack_config)

    if test_method in ["nearest_crop", "maj_voting"]:
        logging.warning(
            "Retrieval per-query diagnostics are not computed for test_method=%s because query refinement changes ranking after FAISS search.",
            test_method,
        )
        eval_ds.retrieval_query_diagnostics = []
        eval_ds.retrieval_per_bin_metrics = []
        eval_ds.retrieval_query_debug = query_debug_rows or []
        eval_ds.retrieval_cluster_mass_stats = cluster_mass_rows or []
        eval_ds.retrieval_diagnostics_meta = {
            "enabled": True,
            "computed": False,
            "metric_type": "l2_distance",
            "margin_definition": "best_negative_distance - best_positive_distance",
            "binning_strategy": "positive_set_size",
            "reason": f"unsupported_test_method:{test_method}",
            "attack": attack_metadata,
        }
        return

    diagnostics = []
    positives_per_query = eval_ds.get_positives()
    bytes_per_query = max(eval_ds.database_num * np.dtype(np.float32).itemsize * 2, 1)
    diag_batch_size = max(1, RETRIEVAL_DIAGNOSTICS_MAX_BUFFER_BYTES // bytes_per_query)
    logging.debug(
        "Calculating retrieval per-query diagnostics in batches of %d queries for database size %d",
        diag_batch_size,
        eval_ds.database_num,
    )

    for batch_start in range(0, len(query_features), diag_batch_size):
        batch_end = min(batch_start + diag_batch_size, len(query_features))
        batch_distances, batch_predictions = faiss_index.search(
            query_features[batch_start:batch_end],
            eval_ds.database_num,
        )

        for batch_offset, (query_distances, query_predictions) in enumerate(zip(batch_distances, batch_predictions)):
            query_index = batch_start + batch_offset
            positives = positives_per_query[query_index]
            positive_count = int(len(positives))
            positive_mask = np.isin(query_predictions, positives)
            negative_mask = np.logical_not(positive_mask)

            best_positive_rank = int(np.argmax(positive_mask)) + 1 if np.any(positive_mask) else None
            best_positive_distance = float(query_distances[best_positive_rank - 1]) if best_positive_rank is not None else None
            best_negative_rank = int(np.argmax(negative_mask)) + 1 if np.any(negative_mask) else None
            best_negative_distance = float(query_distances[best_negative_rank - 1]) if best_negative_rank is not None else None
            margin = None
            if best_positive_distance is not None and best_negative_distance is not None:
                margin = best_negative_distance - best_positive_distance

            diagnostics.append({
                "query_index": int(query_index),
                "positive_count": positive_count,
                "bin_label": _positive_count_bin_label(positive_count),
                "best_positive_distance": best_positive_distance,
                "best_negative_distance": best_negative_distance,
                "margin": float(margin) if margin is not None else None,
                "best_positive_rank": best_positive_rank,
                "top1_prediction": int(query_predictions[0]),
                "top1_correct": bool(positive_mask[0]),
                "success_at_5": bool(np.any(positive_mask[:5])),
                "success_at_10": bool(np.any(positive_mask[:10])),
            })

    eval_ds.retrieval_query_diagnostics = diagnostics
    eval_ds.retrieval_per_bin_metrics = _compute_retrieval_per_bin_metrics(diagnostics)
    if retrieval_debug:
        eval_ds.retrieval_query_debug = query_debug_rows or []
        eval_ds.retrieval_cluster_mass_stats = cluster_mass_rows or []
    eval_ds.retrieval_diagnostics_meta = {
        "enabled": True,
        "computed": True,
        "metric_type": "l2_distance",
        "margin_definition": "best_negative_distance - best_positive_distance",
        "binning_strategy": "positive_set_size",
        "stored_on": "eval_ds.retrieval_query_diagnostics",
        "per_bin_stored_on": "eval_ds.retrieval_per_bin_metrics",
        "debug_metrics": retrieval_debug,
        "attack": attack_metadata,
    }


def test(args: Any, eval_ds: Any, model: torch.nn.Module, test_method: str = "hard_resize", pca: Any = None) -> tuple[np.ndarray, str]:
    assert test_method in SUPPORTED_TEST_METHODS, f"test_method can't be {test_method}"
    if args.efficient_ram_testing:
        raise NotImplementedError("efficient_ram_testing is not implemented for the adversarial evaluation path")

    attack_config = adversarial.attack_config_from_args(args)
    if attack_config.attack_name in adversarial.WHITEBOX_ATTACKS and pca is not None:
        raise ValueError("White-box attacks are not supported together with PCA evaluation")

    _clear_retrieval_outputs(eval_ds)
    eval_ds.retrieval_attack_metadata = adversarial.attack_config_to_dict(attack_config)

    logging.debug("Extracting database features for evaluation/testing")
    database_features = _extract_database_features(args, eval_ds, model, pca)

    logging.debug("Extracting queries features for evaluation/testing")
    query_features, query_debug_rows, cluster_mass_rows = _extract_query_features(
        args=args,
        eval_ds=eval_ds,
        model=model,
        test_method=test_method,
        pca=pca,
        database_features=database_features,
        attack_config=attack_config,
    )

    logging.debug("Calculating recalls")
    faiss_index, _, predictions = _search_predictions(
        args=args,
        eval_ds=eval_ds,
        database_features=database_features,
        query_features=query_features,
        test_method=test_method,
    )
    recalls, recalls_str = _compute_recalls(args, eval_ds, predictions)

    if getattr(args, "enable_retrieval_diagnostics", False):
        _populate_retrieval_diagnostics(
            args=args,
            eval_ds=eval_ds,
            faiss_index=faiss_index,
            query_features=query_features,
            test_method=test_method,
            query_debug_rows=query_debug_rows,
            cluster_mass_rows=cluster_mass_rows,
        )

    return recalls, recalls_str

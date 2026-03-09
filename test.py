
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


def _positive_count_bin_label(positive_count):
    if positive_count <= 0:
        return "positives_0"
    if positive_count == 1:
        return "positives_1"
    if positive_count <= 4:
        return "positives_2_4"
    if positive_count <= 9:
        return "positives_5_9"
    return "positives_10_plus"


def _compute_retrieval_per_bin_metrics(per_query_rows):
    bins = {}
    for row in per_query_rows:
        bin_label = row["bin_label"]
        bins.setdefault(bin_label, []).append(row)

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


def _expand_token_dropout_ids(indices, test_method):
    sample_ids = indices.numpy()
    if test_method in ["five_crops", "nearest_crop", "maj_voting"]:
        return np.repeat(sample_ids, 5)
    return sample_ids


def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""
    
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)
    
    retrieval_diagnostics_enabled = getattr(args, "enable_retrieval_diagnostics", False)
    retrieval_debug = retrieval_diagnostics_enabled and getattr(args, "return_debug_metrics", False)
    eval_ds.retrieval_query_diagnostics = None
    eval_ds.retrieval_diagnostics_meta = None
    eval_ds.retrieval_query_debug = None
    eval_ds.retrieval_cluster_mass_stats = None
    eval_ds.retrieval_per_bin_metrics = None

    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        if test_method == "nearest_crop" or test_method == 'maj_voting':
            all_features = np.empty((5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim), dtype="float32")
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(
                inputs.to(args.device),
                queryflag=0,
                token_keep_ratio=getattr(args, "token_keep_ratio", 1.0),
                token_dropout_seed=getattr(args, "token_dropout_seed", None),
                token_dropout_ids=indices.numpy(),
                masking_mode=getattr(args, "masking_mode", "none"),
            )
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        query_debug_rows = [None] * eval_ds.queries_num if retrieval_debug else None
        cluster_mass_rows = [None] * eval_ds.queries_num if retrieval_debug else None
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            token_dropout_ids = _expand_token_dropout_ids(indices, test_method)
            if retrieval_debug and test_method not in ["five_crops", "nearest_crop", "maj_voting"]:
                outputs = model(
                    inputs.to(args.device),
                    queryflag=0,
                    return_debug=True,
                    low_mass_threshold=getattr(args, "low_mass_threshold", 1e-3),
                    token_keep_ratio=getattr(args, "token_keep_ratio", 1.0),
                    token_dropout_seed=getattr(args, "token_dropout_seed", None),
                    token_dropout_ids=token_dropout_ids,
                    masking_mode=getattr(args, "masking_mode", "none"),
                )
                features = outputs["descriptor"]
                pool_debug = outputs["pool_debug"]
                query_indexes = indices.numpy() - eval_ds.database_num
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
                    query_debug_rows[query_index] = {
                        "query_index": int(query_index),
                        "token_count": token_count_value,
                        "min_mass": float(pool_debug["min_mass"][batch_pos].item()),
                        "max_mass": float(pool_debug["max_mass"][batch_pos].item()),
                        "mean_mass": float(pool_debug["mean_mass"][batch_pos].item()),
                        "p10_mass": float(pool_debug["p10_mass"][batch_pos].item()),
                        "num_low_mass_clusters": int(pool_debug["num_low_mass_clusters"][batch_pos].item()),
                        "assignment_entropy": float(pool_debug["assignment_entropy"][batch_pos].item()),
                    }
                    cluster_mass_row = {"query_index": int(query_index)}
                    for cluster_idx, cluster_mass in enumerate(per_cluster_mass[batch_pos]):
                        cluster_mass_row[f"cluster_mass_{cluster_idx}"] = float(cluster_mass)
                    cluster_mass_rows[query_index] = cluster_mass_row
            else:
                features = model(
                    inputs.to(args.device),
                    queryflag=0,
                    token_keep_ratio=getattr(args, "token_keep_ratio", 1.0),
                    token_dropout_seed=getattr(args, "token_dropout_seed", None),
                    token_dropout_ids=token_dropout_ids,
                    masking_mode=getattr(args, "masking_mode", "none"),
                )
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            
            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = eval_ds.database_num + (indices[0] - eval_ds.database_num) * 5
                end_idx   = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))

    if test_method == 'nearest_crop':
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
    elif test_method == 'maj_voting':
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    if retrieval_diagnostics_enabled:
        if test_method in ["nearest_crop", "maj_voting"]:
            logging.warning(
                "Retrieval per-query diagnostics are not computed for test_method=%s because "
                "query refinement changes ranking after FAISS search.",
                test_method,
            )
            eval_ds.retrieval_query_diagnostics = []
            eval_ds.retrieval_per_bin_metrics = []
            eval_ds.retrieval_diagnostics_meta = {
                "enabled": True,
                "computed": False,
                "metric_type": "l2_distance",
                "margin_definition": "best_negative_distance - best_positive_distance",
                "binning_strategy": "positive_set_size",
                "reason": f"unsupported_test_method:{test_method}",
            }
        else:
            logging.debug("Calculating retrieval per-query diagnostics")
            diag_distances, diag_predictions = faiss_index.search(queries_features, eval_ds.database_num)
            diagnostics = []
            for query_index, (query_distances, query_predictions) in enumerate(zip(diag_distances, diag_predictions)):
                positives = eval_ds.get_positives()[query_index]
                positive_count = int(len(positives))
                positive_mask = np.in1d(query_predictions, positives)
                negative_mask = np.logical_not(positive_mask)

                best_positive_rank = int(np.argmax(positive_mask)) + 1 if np.any(positive_mask) else None
                best_positive_distance = float(query_distances[best_positive_rank - 1]) if best_positive_rank is not None else None
                best_negative_rank = int(np.argmax(negative_mask)) + 1 if np.any(negative_mask) else None
                best_negative_distance = float(query_distances[best_negative_rank - 1]) if best_negative_rank is not None else None
                margin = None
                if best_positive_distance is not None and best_negative_distance is not None:
                    # L2 retrieval uses smaller distances as better matches, so larger margins are better.
                    margin = best_negative_distance - best_positive_distance

                top1_prediction = int(query_predictions[0])
                top1_correct = bool(positive_mask[0])
                success_at_5 = bool(np.any(positive_mask[:5]))
                success_at_10 = bool(np.any(positive_mask[:10]))

                diagnostics.append({
                    "query_index": int(query_index),
                    "positive_count": positive_count,
                    "bin_label": _positive_count_bin_label(positive_count),
                    "best_positive_distance": best_positive_distance,
                    "best_negative_distance": best_negative_distance,
                    "margin": float(margin) if margin is not None else None,
                    "best_positive_rank": best_positive_rank,
                    "top1_prediction": top1_prediction,
                    "top1_correct": top1_correct,
                    "success_at_5": success_at_5,
                    "success_at_10": success_at_10,
                })

            eval_ds.retrieval_query_diagnostics = diagnostics
            eval_ds.retrieval_per_bin_metrics = _compute_retrieval_per_bin_metrics(diagnostics)
            if retrieval_debug:
                eval_ds.retrieval_query_debug = query_debug_rows
                eval_ds.retrieval_cluster_mass_stats = cluster_mass_rows
            eval_ds.retrieval_diagnostics_meta = {
                "enabled": True,
                "computed": True,
                "metric_type": "l2_distance",
                "margin_definition": "best_negative_distance - best_positive_distance",
                "binning_strategy": "positive_set_size",
                "stored_on": "eval_ds.retrieval_query_diagnostics",
                "per_bin_stored_on": "eval_ds.retrieval_per_bin_metrics",
                "debug_metrics": retrieval_debug,
            }

    del database_features, all_features
    return recalls, recalls_str

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from torch import Tensor


@dataclass
class RetrievalAttackBatch:
    query_indices: Tensor
    clean_query_descriptors: Tensor
    positive_descriptors: Tensor
    negative_descriptors: Tensor

    def subset(self, selector: Tensor) -> "RetrievalAttackBatch":
        return RetrievalAttackBatch(
            query_indices=self.query_indices[selector],
            clean_query_descriptors=self.clean_query_descriptors[selector],
            positive_descriptors=self.positive_descriptors[selector],
            negative_descriptors=self.negative_descriptors[selector],
        )

    def __len__(self) -> int:
        return int(self.query_indices.shape[0])


def select_rank_targets(clean_descriptors: Tensor, place_ids: Tensor, adv_negatives: int) -> Optional[RetrievalAttackBatch]:
    batch_size, images_per_place, descriptor_dim = clean_descriptors.shape
    if images_per_place < 2:
        return None

    flat_descriptors = clean_descriptors.reshape(batch_size * images_per_place, descriptor_dim)
    flat_place_ids = place_ids.reshape(-1)
    query_descriptors = clean_descriptors[:, 0, :]

    query_indices: List[int] = []
    clean_queries: List[Tensor] = []
    positives: List[Tensor] = []
    negatives: List[Tensor] = []
    min_negatives = None

    for place_offset in range(batch_size):
        place_label = place_ids[place_offset, 0]
        positive_candidates = clean_descriptors[place_offset, 1:, :]
        if positive_candidates.shape[0] == 0:
            continue

        positive_distances = torch.norm(
            positive_candidates - query_descriptors[place_offset].unsqueeze(0),
            p=2,
            dim=1,
        )
        hardest_positive_index = int(torch.argmax(positive_distances).item())
        negative_candidates = flat_descriptors[flat_place_ids != place_label]
        if negative_candidates.shape[0] == 0:
            continue

        current_k = min(adv_negatives, negative_candidates.shape[0])
        negative_distances = torch.norm(
            negative_candidates - query_descriptors[place_offset].unsqueeze(0),
            p=2,
            dim=1,
        )
        hard_negative_indices = torch.topk(
            negative_distances,
            k=current_k,
            largest=False,
        ).indices

        query_indices.append(place_offset)
        clean_queries.append(query_descriptors[place_offset])
        positives.append(positive_candidates[hardest_positive_index])
        negatives.append(negative_candidates[hard_negative_indices])
        min_negatives = current_k if min_negatives is None else min(min_negatives, current_k)

    if len(query_indices) == 0 or min_negatives is None or min_negatives == 0:
        return None

    trimmed_negatives = [negative[:min_negatives] for negative in negatives]
    device = clean_descriptors.device
    return RetrievalAttackBatch(
        query_indices=torch.tensor(query_indices, dtype=torch.long, device=device),
        clean_query_descriptors=torch.stack(clean_queries, dim=0).detach(),
        positive_descriptors=torch.stack(positives, dim=0).detach(),
        negative_descriptors=torch.stack(trimmed_negatives, dim=0).detach(),
    )


def build_attack_targets(
    args,
    eval_ds,
    database_features: np.ndarray,
    clean_query_features: np.ndarray,
    limit_queries: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    positives_per_query = eval_ds.get_positives()
    valid_query_indices = np.flatnonzero(
        np.fromiter((len(positive_candidates) > 0 for positive_candidates in positives_per_query), dtype=bool)
    )
    if len(valid_query_indices) == 0:
        raise RuntimeError("No queries with positives were found, cannot run attack evaluation.")

    if limit_queries is not None:
        valid_query_indices = valid_query_indices[:limit_queries]

    database_features = np.ascontiguousarray(database_features.astype(np.float32, copy=False))
    query_features = np.ascontiguousarray(clean_query_features[valid_query_indices].astype(np.float32, copy=False))

    search_k = min(eval_ds.database_num, max(128, args.adv_negatives + 32))
    faiss_index = faiss.IndexFlatL2(database_features.shape[1])
    faiss_index.add(database_features)
    _, ranked_neighbors = faiss_index.search(query_features, search_k)

    targets = []
    for target_offset, query_index in enumerate(valid_query_indices):
        positive_candidates = np.asarray(positives_per_query[query_index], dtype=np.int64)
        query_feature = query_features[target_offset]

        positive_distances = np.sum((database_features[positive_candidates] - query_feature[None, :]) ** 2, axis=1)
        positive_index = int(positive_candidates[np.argmin(positive_distances)])

        neighbor_candidates = ranked_neighbors[target_offset]
        negative_indexes = neighbor_candidates[~np.isin(neighbor_candidates, positive_candidates)]
        if len(negative_indexes) < args.adv_negatives:
            distances = np.sum((database_features - query_feature[None, :]) ** 2, axis=1)
            positive_mask = np.zeros(eval_ds.database_num, dtype=bool)
            positive_mask[positive_candidates] = True
            negative_candidates = np.flatnonzero(~positive_mask)
            negative_order = np.argsort(distances[negative_candidates])
            negative_indexes = negative_candidates[negative_order]

        negative_indexes = negative_indexes[: args.adv_negatives].astype(np.int64)
        targets.append(
            {
                "query_index": int(query_index),
                "positive_index": positive_index,
                "negative_indexes": negative_indexes,
            }
        )

    return targets, valid_query_indices

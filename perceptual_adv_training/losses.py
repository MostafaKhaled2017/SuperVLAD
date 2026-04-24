from typing import Sequence

import numpy as np
import torch
from torch import Tensor

LOSS_FN = None
MINER = None


def configure_metric_learning() -> None:
    global LOSS_FN, MINER

    from pytorch_metric_learning import losses, miners
    from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

    LOSS_FN = losses.MultiSimilarityLoss(
        alpha=1.0,
        beta=50,
        base=0.0,
        distance=DotProductSimilarity(),
    )
    MINER = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())


def loss_function(descriptors: Tensor, labels: Tensor) -> Tensor:
    if MINER is not None:
        miner_outputs = MINER(descriptors, labels)
        return LOSS_FN(descriptors, labels, miner_outputs)
    return LOSS_FN(descriptors, labels)


def compute_attack_score(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
    margin: float,
) -> Tensor:
    positive_distance = torch.norm(query_descriptor - positive_descriptor, p=2, dim=1)
    negative_distance = torch.norm(query_descriptor.unsqueeze(1) - negative_descriptors, p=2, dim=2)
    hardest_negative_distance = negative_distance.min(dim=1).values
    return margin + positive_distance - hardest_negative_distance


def compute_rank_loss(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
    margin: float,
) -> Tensor:
    return torch.relu(compute_attack_score(query_descriptor, positive_descriptor, negative_descriptors, margin)).mean()


def compute_align_loss(clean_query_descriptor: Tensor, adv_query_descriptor: Tensor) -> Tensor:
    return (adv_query_descriptor - clean_query_descriptor).pow(2).sum(dim=1).mean()


def query_is_correct(
    query_descriptor: Tensor,
    positive_descriptor: Tensor,
    negative_descriptors: Tensor,
) -> Tensor:
    positive_distance = torch.norm(query_descriptor - positive_descriptor, p=2, dim=1)
    negative_distance = torch.norm(query_descriptor.unsqueeze(1) - negative_descriptors, p=2, dim=2)
    return positive_distance < negative_distance.min(dim=1).values

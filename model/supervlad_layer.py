
import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler

# import model.functional as LF
# import model.normalization as normalization

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class SuperVLAD(nn.Module):
    """SuperVLAD layer implementation"""

    def __init__(self, clusters_num=64, ghost_clusters_num=1, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        clusters_num += ghost_clusters_num
        self.clusters_num = clusters_num
        self.ghost_clusters_num = ghost_clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        # self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        # self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x, return_debug=False, low_mass_threshold=1e-3, token_mask=None):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        active_token_count = x_flatten.shape[-1]
        if token_mask is not None:
            token_mask = token_mask.to(device=x_flatten.device, dtype=x_flatten.dtype)
            token_mask = token_mask.view(N, 1, -1)
            x_flatten = x_flatten * token_mask
            soft_assign = soft_assign * token_mask
            active_token_count = token_mask.sum(dim=2).squeeze(1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = vlad[:,:-self.ghost_clusters_num,:]
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        if not return_debug:
            return vlad

        cluster_mass = soft_assign.sum(dim=2)
        active_cluster_mass = cluster_mass[:, :-self.ghost_clusters_num] if self.ghost_clusters_num > 0 else cluster_mass
        sorted_mass, _ = torch.sort(active_cluster_mass, dim=1)
        p10_index = min(active_cluster_mass.shape[1] - 1, max(0, int(0.1 * active_cluster_mass.shape[1])))
        token_entropy = -(soft_assign * torch.log(soft_assign.clamp_min(1e-12))).sum(dim=1)
        if token_mask is not None:
            token_mask_flat = token_mask.squeeze(1)
            assign_entropy = token_entropy.sum(dim=1) / token_mask_flat.sum(dim=1).clamp_min(1.0)
        else:
            assign_entropy = token_entropy.mean(dim=1)

        debug = {
            "soft_assignments": soft_assign,
            "per_cluster_mass": active_cluster_mass,
            "all_cluster_mass": cluster_mass,
            "min_mass": active_cluster_mass.min(dim=1).values,
            "max_mass": active_cluster_mass.max(dim=1).values,
            "mean_mass": active_cluster_mass.mean(dim=1),
            "p10_mass": sorted_mass[:, p10_index],
            "num_low_mass_clusters": (active_cluster_mass < low_mass_threshold).sum(dim=1),
            "low_mass_threshold": low_mass_threshold,
            "token_count": active_token_count,
            "assignment_entropy": assign_entropy,
            "token_mask": token_mask.squeeze(1) if token_mask is not None else None,
        }
        return vlad, debug

    def initialize_supervlad_layer(self, args, cluster_ds, model):
        backbone = model.backbone
        descriptors_num = 500000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, sampler=random_sampler)
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize SuperVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, args.features_dim), dtype=np.float32)
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)

                ######### for the DINOv2 backbone ###########
                B,P,D = outputs["x_prenorm"].shape
                W = H = int(math.sqrt(P-1))
                outputs = outputs["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2)

                ######### for the CCT backbone ###########
                # outputs = outputs.view(-1,24,24,384).permute(0, 3, 1, 2)

                ######### for the ViT backbone ###########                   
                # B,P,D = outputs.last_hidden_state.shape
                # W = H = int(math.sqrt(P-1))
                # outputs = outputs.last_hidden_state[:, 1:, :].view(B,W,H,D).permute(0, 3, 1, 2)                 

                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], args.features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(args.features_dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"All clusters shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)

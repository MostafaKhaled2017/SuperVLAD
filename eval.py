
"""
With this script you can evaluate checkpoints or test models from two popular
landmark retrieval github repos.
The first is https://github.com/naver/deep-image-retrieval from Naver labs,
provides ResNet-50 and ResNet-101 trained with AP on Google Landmarks 18 clean.
$ python eval.py --off_the_shelf=naver --l2=none --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

The second is https://github.com/filipradenovic/cnnimageretrieval-pytorch from
Radenovic, provides ResNet-50 and ResNet-101 trained with a triplet loss
on Google Landmarks 18 and sfm120k.
$ python eval.py --off_the_shelf=radenovic_gldv1 --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048
$ python eval.py --off_the_shelf=radenovic_sfm --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

Note that although the architectures are almost the same, Naver's
implementation does not use a l2 normalization before/after the GeM aggregation,
while Radenovic's uses it after (and we use it before, which shows better
results in VG)
"""

import os
import sys
import csv
import json
import random
import torch
import parser
import logging
import sklearn
import numpy as np
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url

import test
import util
import commons
import datasets_ws
from model import network
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OFF_THE_SHELF_RADENOVIC = {
    'resnet50conv5_sfm'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'resnet101conv5_sfm'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'resnet50conv5_gldv1'  : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'resnet101conv5_gldv1' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
}

OFF_THE_SHELF_NAVER = {
    "resnet50conv5"  : "1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5",
    'resnet101conv5' : "1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy"
}


def write_csv(path, rows):
    if len(rows) == 0:
        with open(path, "w", newline="") as file:
            file.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_retrieval_diagnostics_outputs(args, eval_ds, recalls, recalls_str):
    output_dir = args.save_dir if args.retrieval_diagnostics_output_dir is None else join(
        args.save_dir, args.retrieval_diagnostics_output_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    per_query_rows = list(eval_ds.retrieval_query_diagnostics or [])
    query_debug_rows = eval_ds.retrieval_query_debug or []
    debug_by_query = {row["query_index"]: row for row in query_debug_rows if row is not None}
    merged_per_query_rows = []
    for row in per_query_rows:
        merged_row = dict(row)
        debug_row = debug_by_query.get(row["query_index"])
        if debug_row is not None:
            for key, value in debug_row.items():
                if key != "query_index":
                    merged_row[key] = value
        merged_per_query_rows.append(merged_row)

    per_bin_rows = list(eval_ds.retrieval_per_bin_metrics or [])
    cluster_mass_rows = [row for row in (eval_ds.retrieval_cluster_mass_stats or []) if row is not None]

    summary = {
        "dataset": eval_ds.dataset_name,
        "test_method": args.test_method,
        "recall_values": list(args.recall_values),
        "recalls": [float(x) for x in recalls],
        "recalls_str": recalls_str,
        "retrieval_diagnostics_enabled": True,
        "return_debug_metrics": args.return_debug_metrics,
        "token_keep_ratio": args.token_keep_ratio,
        "token_keep_ratios": args.token_keep_ratios,
        "masking_mode": args.masking_mode,
        "low_mass_threshold": args.low_mass_threshold,
        "token_dropout_seed": args.token_dropout_seed,
        "query_count": len(merged_per_query_rows),
        "per_bin_count": len(per_bin_rows),
        "cluster_mass_row_count": len(cluster_mass_rows),
        "diagnostics_meta": eval_ds.retrieval_diagnostics_meta,
    }

    with open(join(output_dir, "summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    write_csv(join(output_dir, "per_query.csv"), merged_per_query_rows)
    write_csv(join(output_dir, "per_bin.csv"), per_bin_rows)
    write_csv(join(output_dir, "cluster_mass_stats.csv"), cluster_mass_rows)
    logging.info(f"Retrieval diagnostics exported to {output_dir}")

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
if args.enable_retrieval_diagnostics:
    random.seed(args.token_dropout_seed)
    np.random.seed(args.token_dropout_seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.SuperVLADModel(args)
model = model.to(args.device)

args.features_dim *= args.supervlad_clusters

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)

if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
logging.info(f"Recalls on {test_ds}: {recalls_str}")
if args.enable_retrieval_diagnostics:
    export_retrieval_diagnostics_outputs(args, test_ds, recalls, recalls_str)

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

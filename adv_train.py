import logging
import multiprocessing
import os
import shutil
import warnings
from datetime import datetime
from os.path import exists, join
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import commons
import datasets_ws
import parser as parser_module
import test
import util
from fgsm_eval import get_normalized_bounds
from torchvision import transforms as T

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")

IMAGENET_MEAN_STD = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

TRAIN_CITIES = [
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS",
]

LOSS_FN = None
MINER = None


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _parse_sm_arch(arch):
    if not arch.startswith("sm_"):
        return None
    suffix = arch.split("_", 1)[1]
    if not suffix.isdigit():
        return None
    if len(suffix) == 2:
        return int(suffix[0]), int(suffix[1])
    return int(suffix[:-1]), int(suffix[-1])


def validate_cuda_runtime(args):
    if args.device != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested with --device cuda, but PyTorch cannot access a CUDA device. "
            "Install a CUDA-enabled PyTorch build or rerun with --device cpu."
        )

    arch_list = torch.cuda.get_arch_list()
    supported_arches = [arch for arch in arch_list if arch.startswith("sm_")]
    parsed_arches = [arch for arch in (_parse_sm_arch(arch) for arch in supported_arches) if arch is not None]
    if not parsed_arches:
        return

    device_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    device_capability = torch.cuda.get_device_capability(device_index)
    max_supported_arch = max(parsed_arches)

    if device_capability > max_supported_arch:
        device_sm = f"sm_{device_capability[0]}{device_capability[1]}"
        supported_sm = f"sm_{max_supported_arch[0]}{max_supported_arch[1]}"
        raise RuntimeError(
            "The installed PyTorch CUDA build does not support this GPU architecture. "
            f"Detected GPU '{gpu_name}' with compute capability {device_sm}, but "
            f"PyTorch {torch.__version__} built against CUDA {torch.version.cuda} only includes kernels "
            f"through {supported_sm}. Install a newer torch/torchvision build for this GPU, or rerun "
            "with --device cpu. The current requirements pin torch==2.5.1 from the cu121 index, "
            "which is too old for RTX 50-series / Blackwell GPUs."
        )


def build_parser():
    parser = parser_module.build_parser()
    parser.description = "Rank-aware adversarial training for SuperVLAD"
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help=(
            "Load only the model weights from --resume and reset optimizer, epoch, and "
            "early-stopping state. Recommended when adversarially fine-tuning a converged checkpoint."
        ),
    )
    parser.add_argument(
        "--adv_epsilon",
        type=float,
        default=1e-3,
        help="Maximum L_inf perturbation for adversarial query training.",
    )
    parser.add_argument(
        "--adv_alpha",
        type=float,
        default=None,
        help="PGD step size in normalized image space. Defaults to adv_epsilon / adv_steps.",
    )
    parser.add_argument(
        "--adv_steps",
        type=int,
        default=3,
        help="Number of PGD ascent steps for adversarial query generation.",
    )
    parser.add_argument(
        "--adv_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the adversarial rank-aware loss.",
    )
    parser.add_argument(
        "--adv_align_weight",
        type=float,
        default=0.05,
        help="Weight for clean/adversarial descriptor alignment loss.",
    )
    parser.add_argument(
        "--adv_negatives",
        type=int,
        default=5,
        help="Number of nearest cross-place descriptors used as hard negatives.",
    )
    parser.add_argument(
        "--adv_warmup_epochs",
        type=int,
        default=1,
        help="Number of initial epochs that train only on the clean loss.",
    )
    parser.add_argument(
        "--adv_margin",
        type=float,
        default=0.1,
        help="Margin used in the rank-aware adversarial loss.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum validation score improvement required to reset patience.",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="TensorBoard log directory. Defaults to <save_dir>/tensorboard.",
    )
    parser.add_argument(
        "--save-every",
        dest="save_every",
        type=int,
        default=1,
        help=(
            "Save an additional intermediate checkpoint every N epochs. "
            "The latest checkpoint is always saved after each epoch."
        ),
    )
    parser.add_argument(
        "--gsv_cities_base_path",
        type=str,
        default=None,
        help=(
            "Path to the GSV-Cities training dataset. Defaults to "
            "$GSV_CITIES_BASE_PATH or <eval_datasets_folder>/gsv_cities."
        ),
    )
    return parser


def parse_arguments():
    args = build_parser().parse_args()
    args = parser_module.validate_arguments(args)

    if args.adv_steps < 1:
        raise ValueError("--adv_steps must be at least 1")
    if args.adv_epsilon < 0:
        raise ValueError("--adv_epsilon must be non-negative")
    if args.adv_alpha is not None and args.adv_alpha <= 0:
        raise ValueError("--adv_alpha must be positive when provided")
    if args.adv_loss_weight < 0:
        raise ValueError("--adv_loss_weight must be non-negative")
    if args.adv_align_weight < 0:
        raise ValueError("--adv_align_weight must be non-negative")
    if args.adv_negatives < 1:
        raise ValueError("--adv_negatives must be at least 1")
    if args.adv_warmup_epochs < 0:
        raise ValueError("--adv_warmup_epochs must be non-negative")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta must be non-negative")
    if args.save_every < 1:
        raise ValueError("--save-every must be at least 1")

    if args.adv_alpha is None:
        args.adv_alpha = args.adv_epsilon / args.adv_steps if args.adv_steps > 0 else 0.0

    return args


def create_summary_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard logging requires the 'tensorboard' package. "
            "Install the updated requirements.txt and retry."
        ) from exc
    return SummaryWriter(log_dir=log_dir)


def configure_metric_learning():
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


def loss_function(descriptors, labels):
    if MINER is not None:
        miner_outputs = MINER(descriptors, labels)
        return LOSS_FN(descriptors, labels, miner_outputs)
    return LOSS_FN(descriptors, labels)


def select_rank_targets(clean_desc, place_ids, adv_negatives):
    batch_size, images_per_place, desc_dim = clean_desc.shape
    if images_per_place < 2:
        return None

    flat_desc = clean_desc.reshape(batch_size * images_per_place, desc_dim)
    flat_place_ids = place_ids.reshape(-1)
    query_desc = clean_desc[:, 0, :]

    valid_query_indices = []
    positive_descs = []
    negative_descs = []
    min_negatives = None

    for place_offset in range(batch_size):
        place_label = place_ids[place_offset, 0]
        positive_candidates = clean_desc[place_offset, 1:, :]
        if positive_candidates.shape[0] == 0:
            continue

        positive_distances = torch.norm(
            positive_candidates - query_desc[place_offset].unsqueeze(0),
            p=2,
            dim=1,
        )
        hardest_positive_idx = int(torch.argmax(positive_distances).item())

        negative_candidates = flat_desc[flat_place_ids != place_label]
        if negative_candidates.shape[0] == 0:
            continue

        current_k = min(adv_negatives, negative_candidates.shape[0])
        negative_distances = torch.norm(
            negative_candidates - query_desc[place_offset].unsqueeze(0),
            p=2,
            dim=1,
        )
        hard_negative_indices = torch.topk(
            negative_distances,
            k=current_k,
            largest=False,
        ).indices

        valid_query_indices.append(place_offset)
        positive_descs.append(positive_candidates[hardest_positive_idx])
        negative_descs.append(negative_candidates[hard_negative_indices])
        min_negatives = current_k if min_negatives is None else min(min_negatives, current_k)

    if len(valid_query_indices) == 0 or min_negatives is None or min_negatives == 0:
        return None

    trimmed_negatives = [neg[:min_negatives] for neg in negative_descs]
    query_indices = torch.tensor(valid_query_indices, dtype=torch.long, device=clean_desc.device)

    return {
        "query_indices": query_indices,
        "clean_query_desc": query_desc[query_indices].detach(),
        "positive_desc": torch.stack(positive_descs, dim=0).detach(),
        "negative_desc": torch.stack(trimmed_negatives, dim=0).detach(),
    }


def compute_rank_loss(query_desc, pos_desc, neg_descs, margin):
    positive_distances = torch.norm(query_desc - pos_desc, p=2, dim=1)
    negative_distances = torch.norm(query_desc.unsqueeze(1) - neg_descs, p=2, dim=2)
    hardest_negative_distances = negative_distances.min(dim=1).values
    return torch.relu(margin + positive_distances - hardest_negative_distances).mean()


def compute_align_loss(clean_query_desc, adv_query_desc):
    return (adv_query_desc - clean_query_desc).pow(2).sum(dim=1).mean()


def pgd_attack_queries(model, clean_query_images, pos_desc, neg_descs, args):
    min_value, max_value = get_normalized_bounds(args.device)
    original_queries = clean_query_images.detach()
    adversarial_queries = original_queries.clone()

    was_training = model.training
    model.eval()
    try:
        for _ in range(args.adv_steps):
            adversarial_queries.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            query_desc = model(adversarial_queries, queryflag=0).float()
            attack_loss = compute_rank_loss(query_desc, pos_desc, neg_descs, args.adv_margin)
            gradients = torch.autograd.grad(attack_loss, adversarial_queries)[0]

            adversarial_queries = adversarial_queries.detach() + args.adv_alpha * gradients.sign()
            perturbation = torch.clamp(
                adversarial_queries - original_queries,
                min=-args.adv_epsilon,
                max=args.adv_epsilon,
            )
            adversarial_queries = torch.clamp(original_queries + perturbation, min=min_value, max=max_value).detach()
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

    return adversarial_queries


def get_current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def resolve_gsv_cities_base_path(args):
    candidate_paths = []
    if args.gsv_cities_base_path:
        candidate_paths.append(Path(args.gsv_cities_base_path).expanduser())

    env_base_path = os.environ.get("GSV_CITIES_BASE_PATH")
    if env_base_path:
        env_candidate = Path(env_base_path).expanduser()
        if env_candidate not in candidate_paths:
            candidate_paths.append(env_candidate)

    default_candidate = Path(args.eval_datasets_folder).expanduser() / "gsv_cities"
    if default_candidate not in candidate_paths:
        candidate_paths.append(default_candidate)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    checked_paths = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        "Could not find the GSV-Cities training dataset. Checked: "
        f"{checked_paths}. Pass --gsv_cities_base_path or set GSV_CITIES_BASE_PATH."
    )


def setup_datasets(args):
    logging.debug("Loading dataset %s from folder %s", args.eval_dataset_name, args.eval_datasets_folder)

    triplets_ds = None
    if not args.resume:
        triplets_ds = datasets_ws.TripletsDataset(
            args,
            args.eval_datasets_folder,
            args.eval_dataset_name,
            "train",
            args.negs_num_per_query,
        )
        logging.info("Train query set: %s", triplets_ds)
    else:
        logging.info(
            "Skipping %s train split loading because --resume is set; "
            "the checkpoint already provides initialized model weights.",
            args.eval_dataset_name,
        )

    val_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "val")
    logging.info("Val set: %s", val_ds)

    test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
    logging.info("Test set: %s", test_ds)
    return triplets_ds, val_ds, test_ds


def build_training_dataloader(args):
    from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

    batch_size = args.train_batch_size
    img_per_place = 4
    min_img_per_place = 4
    shuffle_all = False
    image_size = (224, 224)
    num_workers = 4

    mean_dataset = IMAGENET_MEAN_STD["mean"]
    std_dataset = IMAGENET_MEAN_STD["std"]
    train_transform = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_dataset, std=std_dataset),
        ]
    )

    gsv_cities_base_path = resolve_gsv_cities_base_path(args)
    logging.info("Using GSV-Cities training data from %s", gsv_cities_base_path)

    train_dataset = GSVCitiesDataset(
        cities=TRAIN_CITIES,
        img_per_place=img_per_place,
        min_img_per_place=min_img_per_place,
        random_sample_from_each_place=True,
        transform=train_transform,
        base_path=gsv_cities_base_path,
    )

    train_loader_config = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": True,
        "shuffle": shuffle_all,
    }
    return DataLoader(dataset=train_dataset, **train_loader_config)


def maybe_copy_resume_checkpoint(args):
    if args.resume is None:
        return

    initial_checkpoint_path = join(args.save_dir, "initial_model.pth")
    if not exists(initial_checkpoint_path):
        shutil.copyfile(args.resume, initial_checkpoint_path)

    best_checkpoint_path = join(args.save_dir, "best_model.pth")
    if not args.resume_model_only and not exists(best_checkpoint_path):
        shutil.copyfile(args.resume, best_checkpoint_path)


def main():
    args = parse_arguments()
    from model import network
    from model.sync_batchnorm import convert_model

    start_time = datetime.now()
    args.save_dir = join("logs", args.save_dir, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
    args.tensorboard_dir = args.tensorboard_dir or join(args.save_dir, "tensorboard")

    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)

    logging.info("Arguments: %s", args)
    logging.info("The outputs are being saved in %s", args.save_dir)
    logging.info("TensorBoard logs will be written to %s", args.tensorboard_dir)
    logging.info("Using %d GPUs and %d CPUs", torch.cuda.device_count(), multiprocessing.cpu_count())
    validate_cuda_runtime(args)

    writer = create_summary_writer(args.tensorboard_dir)
    try:
        configure_metric_learning()
        triplets_ds, val_ds, test_ds = setup_datasets(args)

        model = network.SuperVLADModel(
            args,
            pretrained_foundation=True,
            foundation_model_path=args.foundation_model_path,
        )
        model = model.to(args.device)

        if triplets_ds is not None:
            triplets_ds.is_inference = True
            model.aggregation.initialize_supervlad_layer(args, triplets_ds, model)
        args.features_dim *= args.supervlad_clusters

        model = torch.nn.DataParallel(model)

        if args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            optim_encoder = None
            if args.crossimage_encoder:
                optim_encoder = torch.optim.Adam(unwrap_model(model).encoder.parameters(), lr=args.lr_encoder)
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
            optim_encoder = None
        elif args.optim == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=9.5e-9)
            optim_encoder = None
        else:
            raise ValueError(f"Unsupported optimizer {args.optim!r}")

        if args.resume:
            if args.resume_model_only:
                util.resume_model(args, unwrap_model(model))
                best_r5 = -1.0
                start_epoch_num = 0
                not_improved_num = 0
                logging.info(
                    "Loaded model weights from %s and reset optimizer, epoch, and early-stopping state.",
                    args.resume,
                )
            else:
                model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
                logging.info("Resuming from epoch %d with best validation score %.1f", start_epoch_num, best_r5)
                logging.info(
                    "Reused optimizer and early-stopping state from %s. "
                    "For adversarial fine-tuning from a converged checkpoint, consider --resume_model_only.",
                    args.resume,
                )
        else:
            best_r5 = -1.0
            start_epoch_num = 0
            not_improved_num = 0

        logging.info("Output dimension of the model is %d", args.features_dim)

        if torch.cuda.device_count() >= 2:
            model = convert_model(model)
            model = model.cuda()

        maybe_copy_resume_checkpoint(args)

        scaler = GradScaler(enabled=args.mixed_precision)
        ds = build_training_dataloader(args)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=len(ds) * 3,
            gamma=0.5,
            last_epoch=-1,
        )

        global_step = start_epoch_num * len(ds)

        for epoch_num in range(start_epoch_num, args.epochs_num):
            logging.info("Start training epoch: %02d", epoch_num)
            epoch_start_time = datetime.now()
            model = model.train()

            epoch_stats = {
                "clean_loss": [],
                "adv_rank_loss": [],
                "align_loss": [],
                "total_loss": [],
            }
            adv_batches = 0
            skipped_adv_batches = 0

            for images, place_id in tqdm(ds):
                images = images.to(args.device, non_blocking=True)
                place_id = place_id.to(args.device, non_blocking=True)

                batch_size, images_per_place, channels, height, width = images.shape
                flat_images = images.reshape(batch_size * images_per_place, channels, height, width)
                labels = place_id.reshape(-1)

                with autocast(enabled=args.mixed_precision):
                    clean_descriptors = model(flat_images, queryflag=0)
                    clean_loss = loss_function(clean_descriptors, labels)

                adv_rank_loss = clean_loss.new_zeros(())
                align_loss = clean_loss.new_zeros(())
                total_loss = clean_loss

                use_adversarial_branch = epoch_num >= args.adv_warmup_epochs
                if use_adversarial_branch:
                    clean_desc_view = clean_descriptors.detach().float().reshape(batch_size, images_per_place, -1)
                    rank_targets = select_rank_targets(clean_desc_view, place_id, args.adv_negatives)

                    if rank_targets is None:
                        skipped_adv_batches += 1
                    else:
                        clean_query_images = images[rank_targets["query_indices"], 0, :, :, :].float()
                        adversarial_queries = pgd_attack_queries(
                            model,
                            clean_query_images,
                            rank_targets["positive_desc"],
                            rank_targets["negative_desc"],
                            args,
                        )

                        with autocast(enabled=args.mixed_precision):
                            adv_query_desc = model(adversarial_queries, queryflag=0)

                        adv_query_desc = adv_query_desc.float()
                        adv_rank_loss = compute_rank_loss(
                            adv_query_desc,
                            rank_targets["positive_desc"],
                            rank_targets["negative_desc"],
                            args.adv_margin,
                        )
                        align_loss = compute_align_loss(rank_targets["clean_query_desc"], adv_query_desc)
                        total_loss = (
                            clean_loss
                            + args.adv_loss_weight * adv_rank_loss
                            + args.adv_align_weight * align_loss
                        )
                        adv_batches += 1
                else:
                    skipped_adv_batches += 1

                optimizer.zero_grad(set_to_none=True)
                if optim_encoder is not None:
                    optim_encoder.zero_grad(set_to_none=True)

                if args.mixed_precision:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if args.crossimage_encoder and optim_encoder is not None:
                        for parameter in unwrap_model(model).encoder.parameters():
                            parameter.requires_grad = True
                        optim_encoder.step()
                        for parameter in unwrap_model(model).encoder.parameters():
                            parameter.requires_grad = False
                    optimizer.step()

                scheduler.step()

                batch_clean_loss = float(clean_loss.item())
                batch_adv_rank_loss = float(adv_rank_loss.item())
                batch_align_loss = float(align_loss.item())
                batch_total_loss = float(total_loss.item())
                current_lr = get_current_lr(optimizer)

                epoch_stats["clean_loss"].append(batch_clean_loss)
                epoch_stats["adv_rank_loss"].append(batch_adv_rank_loss)
                epoch_stats["align_loss"].append(batch_align_loss)
                epoch_stats["total_loss"].append(batch_total_loss)

                writer.add_scalar("train/clean_loss", batch_clean_loss, global_step)
                writer.add_scalar("train/adv_rank_loss", batch_adv_rank_loss, global_step)
                writer.add_scalar("train/align_loss", batch_align_loss, global_step)
                writer.add_scalar("train/total_loss", batch_total_loss, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                global_step += 1

            clean_loss_mean = float(np.mean(epoch_stats["clean_loss"]))
            adv_rank_loss_mean = float(np.mean(epoch_stats["adv_rank_loss"]))
            align_loss_mean = float(np.mean(epoch_stats["align_loss"]))
            total_loss_mean = float(np.mean(epoch_stats["total_loss"]))

            logging.info(
                "Finished epoch %02d in %s, average clean loss = %.4f, adv rank loss = %.4f, "
                "align loss = %.4f, total loss = %.4f, adv batches = %d, skipped adv batches = %d",
                epoch_num,
                str(datetime.now() - epoch_start_time)[:-7],
                clean_loss_mean,
                adv_rank_loss_mean,
                align_loss_mean,
                total_loss_mean,
                adv_batches,
                skipped_adv_batches,
            )

            writer.add_scalar("epoch/clean_loss_mean", clean_loss_mean, epoch_num)
            writer.add_scalar("epoch/adv_rank_loss_mean", adv_rank_loss_mean, epoch_num)
            writer.add_scalar("epoch/align_loss_mean", align_loss_mean, epoch_num)
            writer.add_scalar("epoch/total_loss_mean", total_loss_mean, epoch_num)
            writer.add_scalar("epoch/adv_batches", adv_batches, epoch_num)
            writer.add_scalar("epoch/skipped_adv_batches", skipped_adv_batches, epoch_num)

            recalls, recalls_str = test.test(args, val_ds, model)
            logging.info("Recalls on val set %s: %s", val_ds, recalls_str)

            val_score = float(recalls[0] + recalls[1])
            writer.add_scalar("val/score_r1_plus_r5", val_score, epoch_num)
            for recall_value, recall_metric in zip(args.recall_values, recalls):
                writer.add_scalar(f"val/R@{recall_value}", float(recall_metric), epoch_num)

            is_best = val_score > (best_r5 + args.early_stop_min_delta)
            next_best_r5 = val_score if is_best else best_r5
            next_not_improved_num = 0 if is_best else not_improved_num + 1
            completed_epochs = epoch_num + 1
            checkpoint_state = {
                "epoch_num": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "recalls": recalls,
                "best_r5": next_best_r5,
                "not_improved_num": next_not_improved_num,
                "early_stop_min_delta": args.early_stop_min_delta,
                "tensorboard_dir": args.tensorboard_dir,
                "save_every": args.save_every,
            }

            util.save_checkpoint(
                args,
                checkpoint_state,
                is_best,
                filename="last_model.pth",
            )
            logging.info(
                "Saved latest checkpoint after epoch %02d%s.",
                completed_epochs,
                " (best model updated)" if is_best else "",
            )

            if completed_epochs % args.save_every == 0:
                intermediate_checkpoint_name = f"checkpoint_epoch_{completed_epochs:04d}.pth"
                util.save_checkpoint(
                    args,
                    checkpoint_state,
                    False,
                    filename=intermediate_checkpoint_name,
                )
                logging.info(
                    "Saved intermediate checkpoint %s after epoch %02d because --save-every=%d.",
                    intermediate_checkpoint_name,
                    completed_epochs,
                    args.save_every,
                )

            if is_best:
                improvement = val_score - best_r5
                logging.info(
                    "Improved: previous best score = %.1f, current score = %.1f, improvement = %.4f",
                    best_r5,
                    val_score,
                    improvement,
                )
            else:
                improvement = val_score - best_r5
                logging.info(
                    "Not improved: %d / %d, best score = %.1f, current score = %.1f, delta = %.4f, "
                    "required delta = %.4f",
                    next_not_improved_num,
                    args.patience,
                    best_r5,
                    val_score,
                    improvement,
                    args.early_stop_min_delta,
                )

            best_r5 = next_best_r5
            not_improved_num = next_not_improved_num

            writer.add_scalar("early_stop/best_score", best_r5, epoch_num)
            writer.add_scalar("early_stop/not_improved_epochs", not_improved_num, epoch_num)

            if not_improved_num >= args.patience:
                logging.info(
                    "Early stopping triggered after %d non-improving epochs. Best score = %.1f.",
                    not_improved_num,
                    best_r5,
                )
                break

        logging.info("Best validation score (R@1 + R@5): %.1f", best_r5)
        logging.info("Trained for %02d epochs, in total in %s", epoch_num + 1, str(datetime.now() - start_time)[:-7])

        best_model_state_dict = util.load_trusted_checkpoint(
            join(args.save_dir, "best_model.pth"),
            map_location=args.device,
        )["model_state_dict"]
        model.load_state_dict(best_model_state_dict)

        recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
        logging.info("Recalls on %s: %s", test_ds, recalls_str)

        for recall_value, recall_metric in zip(args.recall_values, recalls):
            writer.add_scalar(f"test/R@{recall_value}", float(recall_metric), 0)
    finally:
        writer.close()


if __name__ == "__main__":
    main()


import os
import torch
import argparse

import adversarial


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=60,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=1000,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--lr_encoder", type=float, default=0.0001, help="Learning rate for the cross-image encoder")
    parser.add_argument("--lr_crn_net", type=float, default=5e-4, help="Learning rate to finetune pretrained network when using CRN")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd", "adamw"])
    parser.add_argument("--mixed_precision", action='store_true', help="Training with Automatic Mixed Precision")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet18conv4",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5", 
                                 "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                 "cct384", "vit", "dino","swin"], help="_")
    parser.add_argument("--crossimage_encoder", action='store_true', help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument('--supervlad_clusters', type=int, default=4, help="Number of clusters for SuperVLAD layer.")
    parser.add_argument('--ghost_clusters', type=int, default=1, help="Number of ghost clusters for SuperVLAD layer.")
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--num_non_local', type=int, default=1, help="Num of non local blocks")
    parser.add_argument("--non_local", action='store_true', help="_")
    parser.add_argument('--channel_bottleneck', type=int, default=128, help="Channel bottleneck for Non-Local blocks")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")
    parser.add_argument("--off_the_shelf", type=str, default="imagenet", choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 11)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(0, 11)))
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None,
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[322, 322], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument("--enable_retrieval_diagnostics", action='store_true',
                        help="Enable retrieval evaluation diagnostics and file export.")
    parser.add_argument("--return_debug_metrics", action='store_true',
                        help="Request pooling debug metrics during retrieval diagnostics evaluation.")
    parser.add_argument("--token_keep_ratio", type=float, default=1.0,
                        help="Optional token keep ratio for eval-time masking hooks.")
    parser.add_argument("--token_keep_ratios", type=float, default=None, nargs="+",
                        help="Optional list of token keep ratios for future eval sweeps.")
    parser.add_argument("--masking_mode", type=str, default="none",
                        choices=["none", "random", "center", "block"],
                        help="Optional masking mode for eval-time token filtering.")
    parser.add_argument("--low_mass_threshold", type=float, default=1e-3,
                        help="Threshold used for low-mass cluster statistics.")
    parser.add_argument("--token_dropout_seed", type=int, default=0,
                        help="Seed used by eval-time token masking logic.")
    parser.add_argument("--retrieval_diagnostics_output_dir", type=str, default=None,
                        help="Optional subdirectory inside the eval run directory for retrieval diagnostics exports.")
    parser.add_argument("--attack_name", type=str, default="none",
                        choices=sorted(adversarial.SUPPORTED_ATTACKS),
                        help="Optional adversarial or corruption attack applied to query images during evaluation.")
    parser.add_argument("--attack_severity", type=int, default=None,
                        help="Severity level for corruption attacks. Supported values are 1, 2, and 3.")
    parser.add_argument("--attack_seed", type=int, default=0,
                        help="Seed used for deterministic attack variants.")
    parser.add_argument("--attack_eps", type=float, default=None,
                        help="L_inf radius in pixel space for white-box attacks.")
    parser.add_argument("--attack_steps", type=int, default=None,
                        help="Number of optimization steps for iterative white-box attacks.")
    parser.add_argument("--attack_step_size", type=float, default=None,
                        help="Per-step L_inf update size in pixel space for iterative white-box attacks.")
    parser.add_argument("--attack_mask_mode", type=str, default="none",
                        choices=sorted(adversarial.SUPPORTED_MASK_MODES),
                        help="Masking mode used when attack_name=token_mask.")
    parser.add_argument("--attack_keep_ratio", type=float, default=None,
                        help="Token keep ratio used when attack_name=token_mask.")
    parser.add_argument("--feature_cache_dir", type=str, default=None,
                        help="Optional directory for cached clean database descriptors during evaluation.")
    parser.add_argument("--adv_train", action="store_true",
                        help="Enable descriptor-space adversarial training during optimization.")
    parser.add_argument("--adv_train_attack", type=str, default="fgsm_linf",
                        choices=["fgsm_linf", "pgd_linf"],
                        help="White-box attack used to generate adversarial training queries.")
    parser.add_argument("--adv_train_eps", type=float, default=None,
                        help="L_inf radius in pixel space for adversarial training.")
    parser.add_argument("--adv_train_steps", type=int, default=10,
                        help="Number of optimization steps for PGD adversarial training.")
    parser.add_argument("--adv_train_step_size", type=float, default=None,
                        help="Per-step L_inf update size in pixel space for PGD adversarial training.")
    parser.add_argument("--adv_train_weight", type=float, default=1.0,
                        help="Weight applied to the adversarial descriptor loss.")
    parser.add_argument("--adv_train_clean_weight", type=float, default=1.0,
                        help="Weight applied to the clean metric-learning loss.")
    parser.add_argument("--adv_train_query_index", type=int, default=0,
                        help="Per-place image index used as the attacked query during training.")
    parser.add_argument("--adv_train_random_start", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a random start inside the L_inf ball for PGD adversarial training.")
    parser.add_argument("--adv_train_log_interval", type=int, default=50,
                        help="How often to log batch-level adversarial training statistics.")
    parser.add_argument("--checkpoint_save_interval", type=int, default=1,
                        help="Save a retained epoch checkpoint every N epochs during training.")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--eval_datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--train_dataset_folder", type=str, default=None,
                        help="Path to the GSV-Cities training dataset root")
    parser.add_argument("--eval_dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()
    
    if args.eval_datasets_folder == None:
        try:
            args.eval_datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")
    
    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError("msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}")
    
    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5", "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    if args.train_dataset_folder is None:
        raise ValueError("Please specify --train_dataset_folder pointing to the gsv_cities dataset root")

    validate_training_arguments(args)
    adversarial.validate_attack_arguments(args)
    
    return args


def validate_training_arguments(args):
    if not getattr(args, "adv_train", False):
        return

    if getattr(args, "adv_train_attack", None) not in {"fgsm_linf", "pgd_linf"}:
        raise ValueError("adv_train_attack must be one of {'fgsm_linf', 'pgd_linf'}")
    if getattr(args, "adv_train_eps", None) is None or args.adv_train_eps <= 0:
        raise ValueError("adv_train requires --adv_train_eps > 0")
    if getattr(args, "adv_train_query_index", 0) < 0:
        raise ValueError("adv_train_query_index must be >= 0")
    if getattr(args, "adv_train_weight", 1.0) < 0:
        raise ValueError("adv_train_weight must be >= 0")
    if getattr(args, "adv_train_clean_weight", 1.0) < 0:
        raise ValueError("adv_train_clean_weight must be >= 0")
    if getattr(args, "adv_train_log_interval", 1) <= 0:
        raise ValueError("adv_train_log_interval must be > 0")
    if getattr(args, "checkpoint_save_interval", 1) <= 0:
        raise ValueError("checkpoint_save_interval must be > 0")
    if getattr(args, "adv_train_attack", "fgsm_linf") == "pgd_linf" and getattr(args, "adv_train_steps", 0) <= 0:
        raise ValueError("PGD adversarial training requires --adv_train_steps > 0")
    if getattr(args, "adv_train_attack", "fgsm_linf") == "fgsm_linf" and getattr(args, "adv_train_steps", 1) <= 0:
        raise ValueError("adv_train_steps must be > 0")

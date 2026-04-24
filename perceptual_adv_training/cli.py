import parser as parser_module

from .config import SUPPORTED_ATTACK_NAMES, UNSUPPORTED_ATTACK_NAMES


def parse_attack_names(attack_strings):
    attack_names = []
    for attack_string in attack_strings:
        attack_name = attack_string.split("(", 1)[0].strip()
        attack_names.append(attack_name)
        if attack_name in UNSUPPORTED_ATTACK_NAMES:
            raise NotImplementedError(
                f"{attack_name} is not supported in perceptual_adv_training.py because it relies on "
                "classification-specific AutoAttack behavior."
            )
        if attack_name not in SUPPORTED_ATTACK_NAMES:
            raise ValueError(
                f"Unsupported attack {attack_name!r}. Supported attacks are: "
                f"{sorted(SUPPORTED_ATTACK_NAMES)}"
            )
    return attack_names


def build_parser():
    parser = parser_module.build_parser()
    parser.description = "Perceptual adversarial training for SuperVLAD"
    parser.add_argument(
        "--attack",
        type=str,
        action="append",
        default=[],
        help="Attack expression(s) to harden against, following the perceptual-advex style.",
    )
    parser.add_argument("--num_epochs", dest="epochs_num", type=int, default=None, help="Number of training epochs.")
    parser.add_argument(
        "--batch_size",
        dest="train_batch_size",
        type=int,
        default=None,
        help="Training batch size alias kept for perceptual-advex style.",
    )
    parser.add_argument("--val_batches", type=int, default=10, help="Number of validation query batches to attack.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Base folder for perceptual adversarial training runs.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of GPUs to use when CUDA is available.")
    parser.add_argument(
        "--only_attack_correct",
        action="store_true",
        default=False,
        help="Attack only queries that are cleanly retrieved before the attack.",
    )
    parser.add_argument(
        "--randomize_attack",
        action="store_true",
        default=False,
        help="Use one random configured attack per training step.",
    )
    parser.add_argument(
        "--maximize_attack",
        action="store_true",
        default=False,
        help="Use the strongest configured attack loss for each training step.",
    )
    parser.add_argument("--continue", dest="continue_training", action="store_true", default=False)
    parser.add_argument("--keep_every", type=int, default=1, help="Keep one intermediate checkpoint every N epochs.")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Clip gradients to this absolute value.")
    parser.add_argument("--lpips_model", type=str, default=None, help="Optional LPIPS model override.")
    parser.add_argument("--lr_schedule", type=str, default=None, help="Epochs when the learning rate drops by 10x.")
    parser.add_argument(
        "--resume_model_only",
        action="store_true",
        help="Load only model weights from --resume and reset optimizer and epoch state.",
    )
    parser.add_argument(
        "--adv_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the adversarial rank loss.",
    )
    parser.add_argument(
        "--adv_align_weight",
        type=float,
        default=0.05,
        help="Weight for the descriptor alignment loss.",
    )
    parser.add_argument(
        "--adv_negatives",
        type=int,
        default=5,
        help="Number of hard negatives to use for retrieval attacks.",
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
        help="Margin used for the retrieval adversarial objective.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum score increase needed to refresh patience.",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="TensorBoard log directory. Defaults to <save_dir>/tensorboard.",
    )
    parser.add_argument(
        "--gsv_cities_base_path",
        type=str,
        default=None,
        help="Path to the GSV-Cities training dataset.",
    )
    return parser


def parse_arguments():
    args = build_parser().parse_args()
    args = parser_module.validate_arguments(args)

    if args.train_batch_size is None:
        args.train_batch_size = 60
    if args.epochs_num is None:
        args.epochs_num = 15
    if args.lr_schedule is None:
        if args.optim == "sgd":
            args.lr_schedule = "30,60,80"
        else:
            args.lr_schedule = "120"
    if args.keep_every < 1:
        raise ValueError("--keep_every must be at least 1")
    if args.adv_negatives < 1:
        raise ValueError("--adv_negatives must be at least 1")
    if args.adv_warmup_epochs < 0:
        raise ValueError("--adv_warmup_epochs must be non-negative")
    if args.clip_grad <= 0:
        raise ValueError("--clip_grad must be positive")

    parse_attack_names(args.attack)
    return args

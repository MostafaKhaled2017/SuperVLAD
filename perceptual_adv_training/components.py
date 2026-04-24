import logging

import torch
from torch.cuda.amp import GradScaler

import util
from .config import unwrap_model
from .data import build_training_dataloader, setup_datasets


def build_training_components(args):
    from model import network
    from model.sync_batchnorm import convert_model

    val_ds, test_ds = setup_datasets(args)

    model = network.SuperVLADModel(
        args,
        pretrained_foundation=True,
        foundation_model_path=args.foundation_model_path,
    )
    model = model.to(args.device)
    args.features_dim *= args.supervlad_clusters

    if args.parallel > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=list(range(args.parallel)))
        if torch.cuda.device_count() >= 2:
            model = convert_model(model)
            model = model.cuda()
    else:
        model = torch.nn.DataParallel(model)

    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=9.5e-9)
    else:
        raise ValueError(f"Unsupported optimizer {args.optim!r}")

    best_score = -1.0
    not_improved = 0
    start_epoch = 0
    if args.resume:
        if args.resume_model_only or not args.continue_training:
            util.resume_model(args, unwrap_model(model))
            logging.info(
                "Loaded model weights from %s and reset optimizer, epoch, and early-stopping state.",
                args.resume,
            )
        else:
            model, optimizer, best_score, start_epoch, not_improved = util.resume_train(
                args,
                model,
                optimizer,
            )
            logging.info("Resuming from epoch %d with best validation score %.1f", start_epoch, best_score)

    scaler = GradScaler(enabled=args.mixed_precision)
    train_loader = build_training_dataloader(args)
    return model, optimizer, scaler, train_loader, val_ds, test_ds, best_score, start_epoch, not_improved

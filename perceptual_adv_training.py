import logging
import multiprocessing
from datetime import datetime
from os.path import join

import commons
import torch

from perceptual_adv_training.attacks import instantiate_attacks
from perceptual_adv_training.checkpoints import maybe_copy_resume_checkpoint
from perceptual_adv_training.cli import parse_arguments
from perceptual_adv_training.components import build_training_components
from perceptual_adv_training.config import create_summary_writer, validate_cuda_runtime
from perceptual_adv_training.losses import configure_metric_learning
from perceptual_adv_training.train_loop import run_training


def main():
    args = parse_arguments()
    validate_cuda_runtime(args)

    start_time = datetime.now()
    args.save_dir = join(args.log_dir, args.save_dir, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
    args.tensorboard_dir = args.tensorboard_dir or join(args.save_dir, "tensorboard")

    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    configure_metric_learning()

    logging.info("Arguments: %s", args)
    logging.info("The outputs are being saved in %s", args.save_dir)
    logging.info("TensorBoard logs will be written to %s", args.tensorboard_dir)
    logging.info("Using %d GPUs and %d CPUs", torch.cuda.device_count(), multiprocessing.cpu_count())

    writer = create_summary_writer(args.tensorboard_dir)
    try:
        model, optimizer, scaler, train_loader, val_ds, test_ds, best_score, start_epoch, not_improved = (
            build_training_components(args)
        )
        maybe_copy_resume_checkpoint(args)

        train_attacks = instantiate_attacks(model, args.attack, args)
        validation_attacks = [instantiate_attacks(model, [attack_string], args)[0] for attack_string in args.attack]
        run_training(
            args,
            model,
            optimizer,
            scaler,
            train_loader,
            val_ds,
            test_ds,
            best_score,
            start_epoch,
            not_improved,
            writer,
            train_attacks,
            validation_attacks,
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()

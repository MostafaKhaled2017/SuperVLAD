import csv
import json
import logging
from datetime import datetime
from os.path import exists, join
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

import test
import util
from .checkpoints import apply_lr_schedule, maybe_remove_old_checkpoint
from .config import amp_autocast, unwrap_model
from .eval import evaluate_against_attacks_retrieval, make_attack_name
from .losses import compute_align_loss, compute_rank_loss, loss_function, query_is_correct
from .targets import RetrievalAttackBatch, select_rank_targets


REPORTED_RECALL_VALUES = (1, 5, 10, 100)


def compute_attack_losses(
    model: nn.Module,
    query_inputs: Tensor,
    attack_targets: RetrievalAttackBatch,
    attacks: Sequence[nn.Module],
    args,
) -> Dict[str, object]:
    if len(attack_targets) == 0 or len(attacks) == 0:
        zero = query_inputs.new_zeros(())
        return {
            "attack_logs": [],
            "adv_rank_loss": zero,
            "align_loss": zero,
            "combined_adv_loss": zero,
        }

    attack_logs = []
    combined_losses = []
    rank_losses = []
    align_losses = []
    for attack in attacks:
        adv_queries = attack(query_inputs, attack_targets)
        with amp_autocast(False, args.device):
            adv_query_descriptors = model(adv_queries, queryflag=0)
        adv_query_descriptors = adv_query_descriptors.float()
        rank_loss = compute_rank_loss(
            adv_query_descriptors,
            attack_targets.positive_descriptors,
            attack_targets.negative_descriptors,
            args.adv_margin,
        )
        align_loss = compute_align_loss(
            attack_targets.clean_query_descriptors,
            adv_query_descriptors,
        )
        combined_loss = args.adv_loss_weight * rank_loss + args.adv_align_weight * align_loss
        attack_logs.append(
            {
                "name": make_attack_name(attack),
                "rank_loss": rank_loss,
                "align_loss": align_loss,
                "combined_loss": combined_loss,
            }
        )
        combined_losses.append(combined_loss)
        rank_losses.append(rank_loss)
        align_losses.append(align_loss)

    stacked_combined = torch.stack(combined_losses)
    stacked_rank = torch.stack(rank_losses)
    stacked_align = torch.stack(align_losses)

    if args.maximize_attack:
        max_index = int(torch.argmax(stacked_combined).item())
        combined_adv_loss = stacked_combined[max_index]
        adv_rank_loss = stacked_rank[max_index]
        align_loss = stacked_align[max_index]
    else:
        combined_adv_loss = stacked_combined.mean()
        adv_rank_loss = stacked_rank.mean()
        align_loss = stacked_align.mean()

    return {
        "attack_logs": attack_logs,
        "adv_rank_loss": adv_rank_loss,
        "align_loss": align_loss,
        "combined_adv_loss": combined_adv_loss,
    }


def compute_validation_selection_scores(metrics: Dict[str, object]) -> Dict[str, float]:
    clean_score = compute_recall_score(metrics["NoAttack"])

    attacked_scores: List[float] = []
    for attack_name, attack_metrics in metrics.items():
        if attack_name == "NoAttack":
            continue
        attacked_scores.append(compute_recall_score(attack_metrics))

    if len(attacked_scores) == 0:
        robust_score = clean_score
    else:
        robust_score = float(np.mean(attacked_scores))

    selection_score = 0.25 * clean_score + 0.75 * robust_score
    return {
        "clean_score": clean_score,
        "robust_score": robust_score,
        "selection_score": selection_score,
    }


def recall_value(metric: Dict[str, object], recall_at: int) -> float:
    key = f"R@{recall_at}"
    recalls = metric["recalls"]
    if key not in recalls:
        raise KeyError(f"Validation metric {key} is missing. Available metrics: {sorted(recalls)}")
    return float(recalls[key])


def compute_recall_score(metric: Dict[str, object]) -> float:
    return recall_value(metric, 1) + recall_value(metric, 5)


def format_reported_recalls(metric: Dict[str, object]) -> str:
    return ", ".join(f"R@{recall_at}={recall_value(metric, recall_at):.2f}" for recall_at in REPORTED_RECALL_VALUES)


def compute_attacked_mean_metrics(metrics: Dict[str, object]) -> Dict[str, object] | None:
    attacked_metrics = [metric for attack_name, metric in metrics.items() if attack_name != "NoAttack"]
    if len(attacked_metrics) == 0:
        return None

    recalls = {}
    for recall_at in REPORTED_RECALL_VALUES:
        recalls[f"R@{recall_at}"] = float(np.mean([recall_value(metric, recall_at) for metric in attacked_metrics]))

    return {
        "recalls": recalls,
        "recalls_list": [recalls[f"R@{recall_at}"] for recall_at in REPORTED_RECALL_VALUES],
        "recalls_str": ", ".join(f"R@{recall_at}: {recalls[f'R@{recall_at}']:.1f}" for recall_at in REPORTED_RECALL_VALUES),
    }


def log_validation_recalls(epoch_label: str, metrics: Dict[str, object]) -> None:
    logging.info("Validation recalls %s clean: %s", epoch_label, format_reported_recalls(metrics["NoAttack"]))
    for attack_name, attack_metrics in metrics.items():
        if attack_name == "NoAttack":
            continue
        logging.info(
            "Validation recalls %s attacked/%s: %s",
            epoch_label,
            attack_name,
            format_reported_recalls(attack_metrics),
        )

    attacked_mean = compute_attacked_mean_metrics(metrics)
    if attacked_mean is not None:
        logging.info("Validation recalls %s attacked/mean: %s", epoch_label, format_reported_recalls(attacked_mean))


def build_validation_metrics_record(
    epoch_num: int,
    metrics: Dict[str, object],
    validation_scores: Dict[str, float],
) -> Dict[str, object]:
    attacks = {
        attack_name: {f"R@{recall_at}": recall_value(attack_metrics, recall_at) for recall_at in REPORTED_RECALL_VALUES}
        for attack_name, attack_metrics in metrics.items()
        if attack_name != "NoAttack"
    }
    attacked_mean = compute_attacked_mean_metrics(metrics)
    return {
        "epoch": epoch_num,
        "clean": {f"R@{recall_at}": recall_value(metrics["NoAttack"], recall_at) for recall_at in REPORTED_RECALL_VALUES},
        "attacks": attacks,
        "attacked_mean": (
            {f"R@{recall_at}": recall_value(attacked_mean, recall_at) for recall_at in REPORTED_RECALL_VALUES}
            if attacked_mean is not None
            else None
        ),
        "clean_score": validation_scores["clean_score"],
        "robust_score": validation_scores["robust_score"],
        "selection_score": validation_scores["selection_score"],
    }


def append_validation_metrics(args, record: Dict[str, object]) -> None:
    csv_path = join(args.save_dir, "validation_recalls.csv")
    jsonl_path = join(args.save_dir, "validation_recalls.jsonl")
    fieldnames = [
        "epoch",
        "split",
        "attack",
        "R@1",
        "R@5",
        "R@10",
        "R@100",
        "clean_score",
        "robust_score",
        "selection_score",
    ]

    rows = [
        {
            "epoch": record["epoch"],
            "split": "clean",
            "attack": "NoAttack",
            **record["clean"],
            "clean_score": record["clean_score"],
            "robust_score": record["robust_score"],
            "selection_score": record["selection_score"],
        }
    ]
    for attack_name, attack_recalls in record["attacks"].items():
        rows.append(
            {
                "epoch": record["epoch"],
                "split": "attacked",
                "attack": attack_name,
                **attack_recalls,
                "clean_score": record["clean_score"],
                "robust_score": record["robust_score"],
                "selection_score": record["selection_score"],
            }
        )
    if record["attacked_mean"] is not None:
        rows.append(
            {
                "epoch": record["epoch"],
                "split": "attacked_mean",
                "attack": "mean",
                **record["attacked_mean"],
                "clean_score": record["clean_score"],
                "robust_score": record["robust_score"],
                "selection_score": record["selection_score"],
            }
        )

    write_header = not exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")


def build_checkpoint_state(
    args,
    model,
    optimizer,
    epoch_num: int,
    metrics: Dict[str, object],
    validation_scores: Dict[str, float],
    next_best_score: float,
    next_not_improved: int,
) -> Dict[str, object]:
    return {
        "epoch_num": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "recalls": metrics["NoAttack"]["recalls_list"],
        "clean_score": validation_scores["clean_score"],
        "robust_score": validation_scores["robust_score"],
        "selection_score": validation_scores["selection_score"],
        "best_r5": next_best_score,
        "not_improved_num": next_not_improved,
        "tensorboard_dir": args.tensorboard_dir,
        "validation_metrics": build_validation_metrics_record(epoch_num, metrics, validation_scores),
    }


def run_training(
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
):
    start_time = datetime.now()
    lr_drop_epochs = [int(epoch_str) for epoch_str in args.lr_schedule.split(",") if epoch_str.strip()]
    iteration = start_epoch * len(train_loader)

    if start_epoch == 0 and not args.skip_initial_validation:
        logging.info("Begin initial validation before training")
        initial_metrics = evaluate_against_attacks_retrieval(
            args,
            model,
            val_ds,
            validation_attacks,
            writer=writer,
            iteration=iteration,
        )
        initial_scores = compute_validation_selection_scores(initial_metrics)
        initial_record = build_validation_metrics_record(-1, initial_metrics, initial_scores)
        append_validation_metrics(args, initial_record)
        log_validation_recalls("before training", initial_metrics)

        best_score = initial_scores["selection_score"]
        not_improved = 0
        initial_checkpoint_state = build_checkpoint_state(
            args,
            model,
            optimizer,
            -1,
            initial_metrics,
            initial_scores,
            best_score,
            not_improved,
        )
        util.save_checkpoint(args, initial_checkpoint_state, True, filename="initial_validation_model.pth")
        writer.add_scalar("val_initial/clean_score", initial_scores["clean_score"], 0)
        writer.add_scalar("val_initial/robust_score", initial_scores["robust_score"], 0)
        writer.add_scalar("val_initial/selection_score", initial_scores["selection_score"], 0)
        logging.info(
            "Validation scores before training: clean = %.2f, robust = %.2f, selection = %.2f",
            initial_scores["clean_score"],
            initial_scores["robust_score"],
            initial_scores["selection_score"],
        )

    for epoch_num in range(start_epoch, args.epochs_num):
        epoch_start = datetime.now()
        model = model.train()

        lr = args.lr
        for lr_drop_epoch in lr_drop_epochs:
            if epoch_num >= lr_drop_epoch:
                lr *= 0.1
        apply_lr_schedule(optimizer, lr)
        logging.info("Start epoch %02d with lr %.2e", epoch_num, lr)

        epoch_clean_losses: List[float] = []
        epoch_adv_rank_losses: List[float] = []
        epoch_align_losses: List[float] = []
        epoch_total_losses: List[float] = []
        skipped_nonfinite_batches = 0

        for images, place_id in tqdm(train_loader, ncols=100, desc=f"Epoch {epoch_num:02d}"):
            images = images.to(args.device, non_blocking=True)
            place_id = place_id.to(args.device, non_blocking=True)

            batch_size, images_per_place, channels, height, width = images.shape
            flat_images = images.reshape(batch_size * images_per_place, channels, height, width)
            labels = place_id.reshape(-1)

            model.eval()
            with torch.no_grad():
                with amp_autocast(args.mixed_precision, args.device):
                    clean_descriptors_eval = model(flat_images, queryflag=0)
                clean_descriptors_eval = clean_descriptors_eval.float()
            clean_descriptor_view = clean_descriptors_eval.reshape(batch_size, images_per_place, -1)
            rank_targets = select_rank_targets(clean_descriptor_view, place_id, args.adv_negatives)

            use_adversarial_branch = epoch_num >= args.adv_warmup_epochs
            step_attacks = train_attacks if use_adversarial_branch else []
            if args.randomize_attack and len(step_attacks) > 0:
                step_attacks = [train_attacks[np.random.randint(0, len(train_attacks))]]

            selected_targets = rank_targets
            query_inputs = None
            if selected_targets is not None and len(selected_targets) > 0:
                query_inputs = images[selected_targets.query_indices, 0, :, :, :]
                if args.only_attack_correct:
                    clean_query_descriptors = clean_descriptor_view[selected_targets.query_indices, 0, :]
                    correct_mask = query_is_correct(
                        clean_query_descriptors,
                        selected_targets.positive_descriptors,
                        selected_targets.negative_descriptors,
                    )
                    if correct_mask.any():
                        selected_targets = selected_targets.subset(correct_mask)
                        query_inputs = query_inputs[correct_mask]
                    else:
                        selected_targets = None
                        query_inputs = None

            optimizer.zero_grad(set_to_none=True)
            model.train()
            with amp_autocast(args.mixed_precision, args.device):
                clean_descriptors = model(flat_images, queryflag=0)
            clean_descriptors = clean_descriptors.float()
            with amp_autocast(args.mixed_precision, args.device):
                clean_loss = loss_function(clean_descriptors, labels)

                attack_outputs = compute_attack_losses(
                    model,
                    query_inputs,
                    selected_targets,
                    step_attacks,
                    args,
                ) if query_inputs is not None and selected_targets is not None else {
                    "attack_logs": [],
                    "adv_rank_loss": clean_loss.new_zeros(()),
                    "align_loss": clean_loss.new_zeros(()),
                    "combined_adv_loss": clean_loss.new_zeros(()),
                }

                total_loss = clean_loss + attack_outputs["combined_adv_loss"]

            if not torch.isfinite(total_loss):
                skipped_nonfinite_batches += 1
                logging.warning(
                    "Skipping non-finite batch at iter %06d during epoch %02d: clean_loss=%s adv_rank_loss=%s "
                    "align_loss=%s combined_adv_loss=%s total_loss=%s",
                    iteration,
                    epoch_num,
                    float(clean_loss.detach().item()),
                    float(attack_outputs["adv_rank_loss"].detach().item()),
                    float(attack_outputs["align_loss"].detach().item()),
                    float(attack_outputs["combined_adv_loss"].detach().item()),
                    float(total_loss.detach().item()),
                )
                optimizer.zero_grad(set_to_none=True)
                iteration += 1
                continue

            if args.mixed_precision:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if not torch.isfinite(grad_norm):
                    skipped_nonfinite_batches += 1
                    logging.warning(
                        "Skipping optimizer step with non-finite gradient norm at iter %06d during epoch %02d.",
                        iteration,
                        epoch_num,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    iteration += 1
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if not torch.isfinite(grad_norm):
                    skipped_nonfinite_batches += 1
                    logging.warning(
                        "Skipping optimizer step with non-finite gradient norm at iter %06d during epoch %02d.",
                        iteration,
                        epoch_num,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    iteration += 1
                    continue
                optimizer.step()

            clean_loss_value = float(clean_loss.item())
            adv_rank_loss_value = float(attack_outputs["adv_rank_loss"].item())
            align_loss_value = float(attack_outputs["align_loss"].item())
            combined_adv_loss_value = float(attack_outputs["combined_adv_loss"].item())
            total_loss_value = float(total_loss.item())
            epoch_clean_losses.append(clean_loss_value)
            epoch_adv_rank_losses.append(adv_rank_loss_value)
            epoch_align_losses.append(align_loss_value)
            epoch_total_losses.append(total_loss_value)

            writer.add_scalar("train/clean_loss", clean_loss_value, iteration)
            writer.add_scalar("train/adv_rank_loss", adv_rank_loss_value, iteration)
            writer.add_scalar("train/align_loss", align_loss_value, iteration)
            writer.add_scalar("train/combined_adv_loss", combined_adv_loss_value, iteration)
            writer.add_scalar("train/total_loss", total_loss_value, iteration)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), iteration)

            for attack_log in attack_outputs["attack_logs"]:
                attack_name = attack_log["name"]
                writer.add_scalar(f"train/{attack_name}/rank_loss", float(attack_log["rank_loss"].item()), iteration)
                writer.add_scalar(f"train/{attack_name}/align_loss", float(attack_log["align_loss"].item()), iteration)
                writer.add_scalar(
                    f"train/{attack_name}/combined_loss",
                    float(attack_log["combined_loss"].item()),
                    iteration,
                )

            if iteration % 10 == 0:
                logging.info(
                    "ITER %06d\tclean_loss=%.4f\tadv_rank_loss=%.4f\talign_loss=%.4f\tcombined_adv_loss=%.4f\ttotal_loss=%.4f",
                    iteration,
                    clean_loss_value,
                    adv_rank_loss_value,
                    align_loss_value,
                    combined_adv_loss_value,
                    total_loss_value,
                )
            iteration += 1

        mean_clean_loss = float(np.mean(epoch_clean_losses)) if epoch_clean_losses else 0.0
        mean_adv_rank_loss = float(np.mean(epoch_adv_rank_losses)) if epoch_adv_rank_losses else 0.0
        mean_align_loss = float(np.mean(epoch_align_losses)) if epoch_align_losses else 0.0
        mean_total_loss = float(np.mean(epoch_total_losses)) if epoch_total_losses else 0.0
        writer.add_scalar("epoch/clean_loss_mean", mean_clean_loss, epoch_num)
        writer.add_scalar("epoch/adv_rank_loss_mean", mean_adv_rank_loss, epoch_num)
        writer.add_scalar("epoch/align_loss_mean", mean_align_loss, epoch_num)
        writer.add_scalar("epoch/total_loss_mean", mean_total_loss, epoch_num)

        logging.info(
            "Finished epoch %02d in %s, clean loss = %.4f, adv rank loss = %.4f, align loss = %.4f, total loss = %.4f, skipped non-finite batches = %d",
            epoch_num,
            str(datetime.now() - epoch_start)[:-7],
            mean_clean_loss,
            mean_adv_rank_loss,
            mean_align_loss,
            mean_total_loss,
            skipped_nonfinite_batches,
        )

        logging.info("Begin validation")
        metrics = evaluate_against_attacks_retrieval(
            args,
            model,
            val_ds,
            validation_attacks,
            writer=writer,
            iteration=iteration,
        )
        validation_scores = compute_validation_selection_scores(metrics)
        clean_score = validation_scores["clean_score"]
        robust_score = validation_scores["robust_score"]
        selection_score = validation_scores["selection_score"]
        validation_record = build_validation_metrics_record(epoch_num + 1, metrics, validation_scores)
        append_validation_metrics(args, validation_record)
        log_validation_recalls(f"after epoch {epoch_num + 1:02d}", metrics)

        writer.add_scalar("val/clean_score", clean_score, epoch_num)
        writer.add_scalar("val/robust_score", robust_score, epoch_num)
        writer.add_scalar("val/selection_score", selection_score, epoch_num)

        is_best = selection_score > (best_score + args.early_stop_min_delta)
        next_best_score = selection_score if is_best else best_score
        next_not_improved = 0 if is_best else not_improved + 1

        checkpoint_state = build_checkpoint_state(
            args,
            model,
            optimizer,
            epoch_num,
            metrics,
            validation_scores,
            next_best_score,
            next_not_improved,
        )
        util.save_checkpoint(args, checkpoint_state, is_best, filename="last_model.pth")
        intermediate_name = f"checkpoint_epoch_{epoch_num + 1:04d}.pth"
        util.save_checkpoint(args, checkpoint_state, False, filename=intermediate_name)
        maybe_remove_old_checkpoint(args, epoch_num + 1)
        logging.info(
            "Saved latest checkpoint after epoch %02d%s.",
            epoch_num + 1,
            " (best model updated)" if is_best else "",
        )
        logging.info(
            "Validation scores after epoch %02d: clean = %.2f, robust = %.2f, selection = %.2f",
            epoch_num + 1,
            clean_score,
            robust_score,
            selection_score,
        )

        best_score = next_best_score
        not_improved = next_not_improved
        writer.add_scalar("early_stop/best_score", best_score, epoch_num)
        writer.add_scalar("early_stop/not_improved_epochs", not_improved, epoch_num)

        if not_improved >= args.patience:
            logging.info(
                "Early stopping triggered after %d non-improving epochs. Best score = %.1f.",
                not_improved,
                best_score,
            )
            break

    logging.info("Best validation selection score: %.2f", best_score)
    best_model_state_dict = util.load_trusted_checkpoint(
        join(args.save_dir, "best_model.pth"),
        map_location=args.device,
    )["model_state_dict"]
    model.load_state_dict(best_model_state_dict)

    recalls, recalls_str = test.test(args, test_ds, unwrap_model(model), test_method=args.test_method)
    logging.info("Final test recalls on %s: %s", test_ds, recalls_str)
    for recall_value, recall_metric in zip(args.recall_values, recalls):
        writer.add_scalar(f"test/R@{recall_value}", float(recall_metric), 0)
    logging.info("Finished in %s", str(datetime.now() - start_time)[:-7])

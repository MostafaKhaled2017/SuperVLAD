
import math
import shlex
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import datasets_ws
import adversarial
from model import network
from model.sync_batchnorm import convert_model

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")


def write_command_file(save_dir):
    command = shlex.join([sys.executable, *sys.argv])
    with open(join(save_dir, "command.txt"), "w", encoding="utf-8") as command_file:
        command_file.write(command + "\n")
    return command


run_command = write_command_file(args.save_dir)
tb_writer = SummaryWriter(log_dir=join(args.save_dir, "tensorboard"))
logging.info(f"Run command: {run_command}")
adv_train_config = adversarial.adv_train_config_from_args(args)
if adv_train_config.enabled:
    logging.info(
        "Adversarial training enabled: attack=%s eps=%s steps=%s step_size=%s clean_weight=%.3f adv_weight=%.3f query_index=%d random_start=%s",
        adv_train_config.attack_name,
        adv_train_config.eps,
        adv_train_config.steps,
        adv_train_config.step_size,
        adv_train_config.clean_weight,
        adv_train_config.weight,
        adv_train_config.query_index,
        adv_train_config.random_start,
    )

#### Creation of Datasets
logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")

triplets_ds = None
if not args.resume:
    triplets_ds = datasets_ws.TripletsDataset(
        args,
        args.eval_datasets_folder,
        args.eval_dataset_name,
        "train",
        args.negs_num_per_query,
    )
    logging.info(f"Train query set: {triplets_ds}")
else:
    logging.info(
        "Skipping %s train split loading because --resume was provided and SuperVLAD initialization is not needed",
        args.eval_dataset_name,
    )

val_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
use_foundation_init = args.resume is None
model = network.SuperVLADModel(
    args,
    pretrained_foundation=use_foundation_init,
    foundation_model_path=args.foundation_model_path,
)

model = model.to(args.device)

resume_training_state = False
if args.resume:
    checkpoint = torch.load(args.resume, map_location=args.device)
    resume_training_state = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint and "epoch_num" in checkpoint
    del checkpoint

# Initialize (the conv layer in) SuperVLAD layer
if not args.resume:
    triplets_ds.is_inference = True
    model.aggregation.initialize_supervlad_layer(args, triplets_ds, model)
args.features_dim *= args.supervlad_clusters

if args.resume and not resume_training_state:
    model = util.resume_model(args, model)
    logging.info(f"Loaded model weights from {args.resume} for fine-tuning")

model = torch.nn.DataParallel(model)

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.crossimage_encoder:
        optim_encoder = torch.optim.Adam(model.module.encoder.parameters(), lr=args.lr_encoder)
    # optim_conv = torch.optim.Adam(model.module.conv.parameters(), lr=args.lr_encoder)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=9.5e-9)

#### Resume model, optimizer, and other training parameters
if args.resume and resume_training_state:
    model, optimizer, best_r5, start_epoch_num, not_improved_num, best_metric_name = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best {best_metric_name} score {best_r5:.1f}")
else:
    best_r5 = start_epoch_num = not_improved_num = 0
    best_metric_name = "val_adv_r1_r5" if adv_train_config.enabled else "val_clean_r1_r5"

logging.info(f"Output dimension of the model is {args.features_dim}")

if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.cuda()

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

TRAIN_CITIES = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL', # refers to Oslo
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT', # refers to Toronto
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG', # refers to Prague
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS', # refers to Paris
]

batch_size=args.train_batch_size #32
img_per_place=4
min_img_per_place=4
shuffle_all=False
image_size=(224, 224)#(256,256)#(320,320)#
num_workers=4
cities=TRAIN_CITIES
mean_std=IMAGENET_MEAN_STD
random_sample_from_each_place=True

mean_dataset = mean_std['mean']
std_dataset = mean_std['std']
train_transform = T.Compose([
    T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
    T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=mean_dataset, std=std_dataset),
])

train_loader_config = {
    'batch_size': batch_size,
    'num_workers': num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': shuffle_all}

train_dataset = GSVCitiesDataset(
            cities=cities,
            img_per_place=img_per_place,
            min_img_per_place=min_img_per_place,
            random_sample_from_each_place=random_sample_from_each_place,
            transform=train_transform,
            base_path=args.train_dataset_folder)

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
#  The loss function call (this method will be called at each training iteration)
def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        loss = loss_fn(descriptors, labels, miner_outputs)

        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss


def compute_adversarial_training_stats(model, images, labels, batch_size, images_per_place):
    query_indices = adversarial.select_training_query_indices(
        batch_size=batch_size,
        images_per_place=images_per_place,
        query_index=adv_train_config.query_index,
    ).to(labels.device)

    with torch.no_grad():
        clean_descriptors = model(images)

    valid_query_indices, positive_targets, negative_targets = adversarial.build_training_descriptor_targets(
        clean_descriptors.detach(),
        labels,
        query_indices,
    )

    if valid_query_indices.numel() == 0:
        return None

    query_images = images.index_select(0, valid_query_indices)
    adv_inputs = adversarial.generate_training_adversarial_queries(
        inputs=query_images,
        sample_ids=valid_query_indices.detach().cpu().tolist(),
        config=adv_train_config,
        model=model,
        positive_descriptors=positive_targets,
        negative_descriptors=negative_targets,
    )

    perturbation = torch.abs(
        adversarial.denormalize_tensor(adv_inputs) - adversarial.denormalize_tensor(query_images)
    ).reshape(adv_inputs.shape[0], -1).max(dim=1)[0].mean()

    return {
        "adv_inputs": adv_inputs,
        "positive_targets": positive_targets,
        "negative_targets": negative_targets,
        "perturbation": perturbation,
        "valid_count": int(valid_query_indices.numel()),
    }


def run_validation_pass(args, eval_ds, model, tb_writer, epoch_num, namespace, log_label):
    recalls, recalls_str = test.test(args, eval_ds, model)
    logging.info(f"Recalls on {log_label} {eval_ds}: {recalls_str}")
    for recall_value, recall_score in zip(args.recall_values, recalls):
        tb_writer.add_scalar(f"{namespace}/R@{recall_value}", float(recall_score), epoch_num)
    return recalls, recalls_str

#### Training loop
if args.mixed_precision:   # For training with Automatic Mixed Precision
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.15, total_iters=4000)
global_step = start_epoch_num * len(ds)
last_epoch_num = None
for epoch_num in range(start_epoch_num, args.epochs_num):
    last_epoch_num = epoch_num
    logging.info(f"Start training epoch: {epoch_num:02d}")
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    model = model.train()
    epoch_losses=[]
    epoch_clean_losses = []
    epoch_adv_losses = []
    epoch_adv_perturbations = []
    skipped_adv_batches = 0
    for batch_index, (images, place_id) in enumerate(tqdm(ds), start=1):
        BS, N, ch, h, w = images.shape
        if adv_train_config.enabled and adv_train_config.query_index >= N:
            raise ValueError(f"adv_train_query_index={adv_train_config.query_index} must be < images_per_place={N}")
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)
        images = images.to(args.device)
        labels = labels.to(args.device)

        if not args.mixed_precision:
            clean_loss = loss_function(model(images), labels)
            adv_loss = clean_loss.new_zeros(())
            adv_perturbation = clean_loss.new_zeros(())

            if adv_train_config.enabled:
                was_training = model.training
                model.eval()
                adv_stats = compute_adversarial_training_stats(model, images, labels, BS, N)
                model.train(was_training)
                if adv_stats is None:
                    skipped_adv_batches += 1
                else:
                    adv_perturbation = adv_stats["perturbation"]
                    adv_descriptors = model(adv_stats["adv_inputs"])
                    adv_loss = adversarial.descriptor_attack_loss(
                        adv_descriptors,
                        adv_stats["positive_targets"],
                        adv_stats["negative_targets"],
                    )

            total_loss = (adv_train_config.clean_weight * clean_loss) + (adv_train_config.weight * adv_loss)

            optimizer.zero_grad()
            if args.crossimage_encoder:
                optim_encoder.zero_grad()
            total_loss.backward()
            if args.crossimage_encoder:
                for p in model.module.encoder.parameters():
                    p.requires_grad = True
                optim_encoder.step() 
                for p in model.module.encoder.parameters():
                    p.requires_grad = False 
            optimizer.step()
            scheduler.step()     

        # Training with Automatic Mixed Precision for faster training speed and less GPU memory usage. 
        # In this case, the cross-image encoder is not optimized separately and may not perform well.
        else:
            adv_loss = torch.zeros((), device=args.device)
            adv_perturbation = torch.zeros((), device=args.device)
            if adv_train_config.enabled:
                was_training = model.training
                model.eval()
                adv_stats = compute_adversarial_training_stats(model, images, labels, BS, N)
                model.train(was_training)
                if adv_stats is None:
                    skipped_adv_batches += 1
                else:
                    adv_perturbation = adv_stats["perturbation"]
            with autocast():
                clean_loss = loss_function(model(images), labels)
                if adv_train_config.enabled and adv_stats is not None:
                    adv_descriptors = model(adv_stats["adv_inputs"])
                    adv_loss = adversarial.descriptor_attack_loss(
                        adv_descriptors,
                        adv_stats["positive_targets"],
                        adv_stats["negative_targets"],
                    )
                total_loss = (adv_train_config.clean_weight * clean_loss) + (adv_train_config.weight * adv_loss)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()       
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = total_loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        epoch_clean_losses.append(float(clean_loss.item()))
        epoch_adv_losses.append(float(adv_loss.item()))
        if adv_train_config.enabled:
            epoch_adv_perturbations.append(float(adv_perturbation.item()))
        tb_writer.add_scalar("train/total_loss_batch", batch_loss, global_step)
        tb_writer.add_scalar("train/clean_loss_batch", float(clean_loss.item()), global_step)
        tb_writer.add_scalar("train/adv_loss_batch", float(adv_loss.item()), global_step)
        if adv_train_config.enabled:
            tb_writer.add_scalar("train/adv_linf_batch", float(adv_perturbation.item()), global_step)
        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
        global_step += 1

        if adv_train_config.enabled and batch_index % adv_train_config.log_interval == 0:
            logging.info(
                "Epoch %02d batch %04d: clean_loss=%.4f adv_loss=%.4f total_loss=%.4f mean_linf=%.6f",
                epoch_num,
                batch_index,
                clean_loss.item(),
                adv_loss.item(),
                total_loss.item(),
                adv_perturbation.item(),
            )
        del total_loss

    
    log_message = (
        f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"average epoch total loss = {epoch_losses.mean():.4f}, "
        f"clean loss = {np.mean(epoch_clean_losses):.4f}"
    )
    if adv_train_config.enabled:
        mean_adv_loss = float(np.mean(epoch_adv_losses)) if epoch_adv_losses else 0.0
        mean_adv_perturbation = float(np.mean(epoch_adv_perturbations)) if epoch_adv_perturbations else 0.0
        log_message += (
            f", adv loss = {mean_adv_loss:.4f}, "
            f"mean attack linf = {mean_adv_perturbation:.6f}, "
            f"skipped adv batches = {skipped_adv_batches}"
        )
    logging.info(log_message)
    tb_writer.add_scalar("train/total_loss_epoch", float(epoch_losses.mean()), epoch_num)
    tb_writer.add_scalar("train/clean_loss_epoch", float(np.mean(epoch_clean_losses)), epoch_num)
    tb_writer.add_scalar("train/adv_loss_epoch", float(np.mean(epoch_adv_losses)) if epoch_adv_losses else 0.0, epoch_num)
    if adv_train_config.enabled:
        tb_writer.add_scalar("train/adv_linf_epoch", mean_adv_perturbation, epoch_num)
        tb_writer.add_scalar("train/skipped_adv_batches_epoch", skipped_adv_batches, epoch_num)
    
    # Compute recalls on validation set
    clean_recalls, _ = run_validation_pass(
        args=args,
        eval_ds=val_ds,
        model=model,
        tb_writer=tb_writer,
        epoch_num=epoch_num,
        namespace="val_clean",
        log_label="clean val set",
    )

    adv_recalls = None
    if adv_train_config.enabled:
        adv_eval_config = adversarial.eval_attack_config_from_adv_train(args, adv_train_config)
        adv_eval_args = adversarial.copy_args_with_attack_config(args, adv_eval_config)
        adv_recalls, _ = run_validation_pass(
            args=adv_eval_args,
            eval_ds=val_ds,
            model=model,
            tb_writer=tb_writer,
            epoch_num=epoch_num,
            namespace="val_adv",
            log_label=f"matched adversarial val set ({adv_eval_config.attack_name})",
        )

    validation_state = util.resolve_validation_state(
        clean_recalls=clean_recalls,
        adv_recalls=adv_recalls,
        adv_train_enabled=adv_train_config.enabled,
    )
    selection_score = validation_state["selected_score"]
    selected_recalls = validation_state["selected_recalls"]
    selected_metric_name = validation_state["selected_metric_name"]
    is_best = selection_score > best_r5

    # If the selected validation metric did not improve for "many" epochs, stop training
    if is_best:
        logging.info(
            "Improved on %s: previous best = %.1f, current = %.1f",
            selected_metric_name,
            best_r5,
            selection_score,
        )
        best_r5 = selection_score
        best_metric_name = selected_metric_name
        not_improved_num = 0
        logging.info("Epochs without improvement: %d", not_improved_num)
    else:
        not_improved_num += 1
        logging.info(
            "Not improved on %s: %d / %d, best = %.1f, current = %.1f",
            selected_metric_name,
            not_improved_num,
            args.patience,
            best_r5,
            selection_score,
        )
        logging.info("Epochs without improvement: %d", not_improved_num)

    checkpoint_state = {
        "epoch_num": epoch_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "recalls": selected_recalls,
        "best_r5": best_r5,
        "best_metric_score": best_r5,
        "best_metric_name": best_metric_name,
        "not_improved_num": not_improved_num,
        "adv_train_config": vars(adv_train_config),
        "val_clean_recalls": validation_state["clean_recalls"],
        "val_adv_recalls": validation_state["adv_recalls"],
    }

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, checkpoint_state, is_best, filename="last_model.pth")
    if (epoch_num + 1) % args.checkpoint_save_interval == 0:
        util.save_checkpoint(args, checkpoint_state, is_best=False, filename=f"checkpoint_epoch_{epoch_num + 1:03d}.pth")

    if not is_best and not_improved_num >= args.patience:
        logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
        break


logging.info(f"Best {best_metric_name}: {best_r5:.1f}")
trained_epochs = 0 if last_epoch_num is None else last_epoch_num + 1
logging.info(f"Trained for {trained_epochs:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_path = join(args.save_dir, "best_model.pth")
test_epoch = last_epoch_num if last_epoch_num is not None else start_epoch_num
if os.path.exists(best_model_path):
    best_model_state_dict = torch.load(best_model_path)["model_state_dict"]
    model.load_state_dict(best_model_state_dict)
else:
    logging.warning("best_model.pth was not created; evaluating the current model weights instead.")

recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")
for recall_value, recall_score in zip(args.recall_values, recalls):
    tb_writer.add_scalar(f"test/R@{recall_value}", float(recall_score), test_epoch)
tb_writer.close()

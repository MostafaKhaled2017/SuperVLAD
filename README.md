# Adversarial Robustness for Visual Place Recognition

This repository contains our work on adversarial robustness for visual place recognition. The main goal of the project is to study how VPR systems behave under adversarial attacks and how their robustness can be improved with retrieval-aware defense methods.

In our experiments, we use SuperVLAD as the main model because it is a strong and recent method for the VPR task. The project started from the original SuperVLAD repository, but the focus here is the adversarial robustness study rather than SuperVLAD itself.

Our work has two main parts:

1. Retrieval-aware adversarial training evaluated with FGSM attacks.
2. Perceptual adversarial training adapted from classification to the VPR setting.

## Project Contents

- `adv_train.py`: retrieval-aware adversarial training for FGSM-style robustness.
- `fgsm_eval.py`: FGSM robustness evaluation on clean and attacked queries.
- `perceptual_adv_training/`: adapted perceptual adversarial training package for the VPR setting. This directory contains the code that replaces classification-oriented assumptions with retrieval-aware components. It defines the retrieval targets, rank-style losses, attack wrappers, and evaluation logic used by the perceptual adversarial training workflow.
- `perceptual_adv_training.py`: entry point that launches the perceptual adversarial training package.
- `scripts/adv_training_different_epsilons.sh`: runs FGSM adversarial training with different epsilon values.
- `scripts/run_fgsm_checkpoint_list_eval.sh`: evaluates a list of checkpoints with FGSM.
- `scripts/run_perceptual_adv_training.sh`: runs perceptual adversarial training.

## Models and Checkpoints

The experiments in this repository use SuperVLAD as the main VPR model. The repo uses a base SuperVLAD checkpoint and saves experiment checkpoints locally during training.

- Base SuperVLAD checkpoint: [download link](https://drive.google.com/file/d/1yomnWGTJko6nf3F2Ju6RWsLhP2EG82tL/view?usp=drive_link)
- Trained models checkpoints: [download link](https://drive.google.com/drive/folders/1dP61euhUI2I5e9e-FE1A_Vvf09b-fLQ1?usp=sharing)


## Datasets and Setup

This repo follows the framework of [GSV-Cities](https://github.com/amaralibey/gsv-cities) for training and the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) for evaluation.

You can download:

- GSV-Cities from [Kaggle](https://www.kaggle.com/datasets/amaralibey/gsv-cities)
- evaluation datasets with [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader)
- the DINOv2 ViT-B/14 checkpoint from [here](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)

An example evaluation dataset layout is:

```
├── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```

Before training, you should download the pre-trained foundation model DINOv2(ViT-B/14) [HERE](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth).

## Train
```
python3 train.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=msls --foundation_model_path=/path/to/pre-trained/dinov2_vitb14_pretrain.pth --backbone=dino --supervlad_clusters=4 --crossimage_encoder --patience=3 --lr=0.00005 --epochs_num=20 --train_batch_size=120 --freeze_te=8
```

## Adversarial Train
```
python3 adv_train.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=msls --foundation_model_path=/path/to/pre-trained/dinov2_vitb14_pretrain.pth --backbone=dino --supervlad_clusters=4 --crossimage_encoder --patience=3 --lr=0.00005 --epochs_num=20 --train_batch_size=120 --freeze_te=8 --adv_epsilon=0.001 --adv_steps=3 --adv_loss_weight=1.0 --adv_align_weight=0.05 --adv_negatives=5
```

To adversarially fine-tune the current SuperVLAD checkpoint instead of training from scratch, resume from `checkpoints/SuperVLAD.pth` and point the script to the GSV-Cities training set used for optimization. When fine-tuning a converged checkpoint, prefer `--resume_model_only` so the adversarial run starts with fresh optimizer and early-stopping state instead of inheriting the clean-training state:

```
python3 adv_train.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --gsv_cities_base_path=/path/to/your/gsv_cities --eval_dataset_name=msls --resume=checkpoints/SuperVLAD.pth --resume_model_only --foundation_model_path=/path/to/pre-trained/dinov2_vitb14_pretrain.pth --backbone=dino --supervlad_clusters=4 --crossimage_encoder --patience=3 --lr=0.00005 --epochs_num=20 --train_batch_size=120 --freeze_te=8 --adv_epsilon=0.001 --adv_steps=3 --adv_loss_weight=1.0 --adv_align_weight=0.05 --adv_negatives=5
```

If `--gsv_cities_base_path` is omitted, the script will try `$GSV_CITIES_BASE_PATH` and then `<eval_datasets_folder>/gsv_cities`.

With the current DINO-based training script, `--foundation_model_path` is still required when resuming because the backbone is constructed before the checkpoint weights are loaded.

To monitor training with TensorBoard:

```
tensorboard --logdir logs
```

## Test
```
python3 eval.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=msls --resume=/path/to/trained/model/SuperVLAD.pth --backbone=dino --supervlad_clusters=4 --crossimage_encoder --infer_batch_size=8
```

## FGSM Robustness Evaluation
Use `fgsm_eval.py` to evaluate one clean run and one attacked run for each epsilon value that you provide. The script keeps the database descriptors clean, perturbs only query images, and saves a JSON report with the command, arguments, attack configuration, and recall metrics.

If an FGSM evaluation was interrupted, pass `--resume_eval_dir` with a previous run directory such as `test/default/2026-04-18_04-41-30`. The script will read completed `Clean recalls` and `FGSM eps=...` entries from that directory's `info.log`, skip those finished runs, append the remaining logs to the same directory, and then write `fgsm_eval_results.json` there.

Supported FGSM objectives:

- `positive_distance`: move each query descriptor away from a true positive database match.
- `wrong_match`: make each query descriptor more similar to a nearby wrong database match while reducing similarity to a true positive.
- `training_style`: use a lightweight triplet-style ranking loss with one positive and the nearest non-positive descriptors.

The FGSM script currently supports `--test_method hard_resize`, `single_query`, and `central_crop`.

Example with one clean run and multiple epsilon values:

```
python3 fgsm_eval.py --eval_datasets_folder=datasets --eval_dataset_name=msls --resume=checkpoints/SuperVLAD.pth --backbone=dino --supervlad_clusters=4 --crossimage_encoder --infer_batch_size=32 --epsilons 0.00001 0.0001 0.001 --fgsm_loss positive_distance
```

Example resume command:

```
python3 fgsm_eval.py --eval_datasets_folder=datasets --eval_dataset_name=msls --resume=checkpoints/SuperVLAD.pth --backbone=dino --supervlad_clusters=4 --crossimage_encoder --infer_batch_size=32 --epsilons 0.00001 0.0001 0.001 --fgsm_loss positive_distance --resume_eval_dir test/default/2026-04-18_04-41-30
```

## SuperVLAD without cross-image encoder

Remove parameter `--crossimage_encoder` to run the SuperVLAD without cross-image encoder.

## 1-cluster VLAD

Set `--supervlad_clusters=1` and `--ghost_clusters=2` to run the 1-cluster VLAD. For example,

```
python3 eval.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=msls --resume=/path/to/trained/model/1-clusterVLAD.pth --backbone=dino --supervlad_clusters=1 --ghost_clusters=2
```

### 4. Training With Different FGSM Epsilon Values

Use:

```bash
bash scripts/adv_training_different_epsilons.sh
```

Then evaluate saved checkpoints with:

```bash
bash scripts/run_fgsm_checkpoint_list_eval.sh
```

### 5. Perceptual Adversarial Training

Use:

```bash
bash scripts/run_perceptual_adv_training.sh
```

This part of the project adapts a classification-based perceptual adversarial training pipeline so it can work in the VPR setting, with SuperVLAD used as the experimental model together with retrieval targets, rank-style losses, and recall-based validation.

## Evaluation and Monitoring

To evaluate a checkpoint on the retrieval benchmark:

```bash
python3 eval.py \
  --eval_datasets_folder=/path/to/datasets_vg/datasets \
  --eval_dataset_name=msls \
  --resume=/path/to/model.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --infer_batch_size=8
```

To monitor training:

```bash
tensorboard --logdir logs
```

## Notes

- Remove `--crossimage_encoder` to run SuperVLAD without the cross-image encoder.
- Set `--supervlad_clusters=1 --ghost_clusters=2` to run the 1-cluster VLAD setup.
- Add `--mixed_precision` if you want mixed precision training.

## Acknowledgements

This project builds on ideas and code from:

- [SuperVLAD](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b135d408253205ba501d55c6539bfc7-Abstract-Conference.html)
- [GSV-Cities](https://github.com/amaralibey/gsv-cities)
- [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex)

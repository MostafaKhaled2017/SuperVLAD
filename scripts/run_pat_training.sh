#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

exec python3 pat_training.py \
  --eval_datasets_folder=datasets \
  --gsv_cities_base_path=datasets/gsv_cities \
  --eval_dataset_name=msls \
  --resume=checkpoints/SuperVLAD.pth \
  --resume_model_only \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --freeze_te=8 \
  --lr=0.00005 \
  --num_epochs=50 \
  --batch_size=4 \
  --attack "FastLagrangePerceptualAttack(model, bound=0.5, num_iterations=5)" \
  --attack "PerceptualPGDAttack(model, bound=0.5, num_iterations=3)" \
  --adv_loss_weight=1.0 \
  --adv_align_weight=0.05 \
  --adv_negatives=5 \
  --mixed_precision

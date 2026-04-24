#!/usr/bin/env bash

EPSILONS=(
  0.01
  0.1
  0.2
)

for epsilon in "${EPSILONS[@]}"; do
  python adv_train.py \
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
    --lr=1e-5 \
    --patience=15 \
    --epochs_num=15 \
    --train_batch_size=32 \
    --adv_epsilon="${epsilon}" \
    --adv_steps=3 \
    --adv_loss_weight=0.25 \
    --adv_align_weight=0.05 \
    --adv_warmup_epochs=2 \
    --adv_negatives=5
done
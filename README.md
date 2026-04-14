# SuperVLAD
This is the official repository for the NeurIPS 2024 paper "[SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b135d408253205ba501d55c6539bfc7-Abstract-Conference.html)".

<img src="image/architecture.png" width="800px">

## Getting Started

This repo follows the framework of [GSV-Cities](https://github.com/amaralibey/gsv-cities) for training, and the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) for evaluation. You can download the GSV-Cities datasets [HERE](https://www.kaggle.com/datasets/amaralibey/gsv-cities), and refer to [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) to prepare test datasets.

The test dataset should be organized in a directory tree as such:

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

## Test
```
python3 eval.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=msls --resume=/path/to/trained/model/SuperVLAD.pth --backbone=dino --supervlad_clusters=4 --crossimage_encoder --infer_batch_size=8
```

## Adversarial Evaluation

The evaluation pipeline supports query-side robustness testing with:

- Image corruptions: `gaussian_noise`, `gaussian_blur`, `jpeg_compression`, `brightness_contrast_shift`, `patch_occlusion`
- White-box attacks: `fgsm_linf`, `pgd_linf`
- Token masking attacks: `token_mask` with `random`, `center`, or `block`

Attack runs currently support `--test_method=hard_resize`.

Example single attack run:

```
python3 eval.py \
  --eval_datasets_folder=/path/to/your/datasets_vg/datasets \
  --eval_dataset_name=msls \
  --resume=/path/to/trained/model/SuperVLAD.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --infer_batch_size=8 \
  --enable_retrieval_diagnostics \
  --return_debug_metrics \
  --attack_name=pgd_linf \
  --attack_eps=0.03137254901960784 \
  --attack_steps=10 \
  --attack_seed=0
```

Token masking example:

```
python3 eval.py \
  --eval_datasets_folder=/path/to/your/datasets_vg/datasets \
  --eval_dataset_name=msls \
  --resume=/path/to/trained/model/SuperVLAD.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --infer_batch_size=8 \
  --enable_retrieval_diagnostics \
  --return_debug_metrics \
  --attack_name=token_mask \
  --attack_mask_mode=block \
  --attack_keep_ratio=0.5 \
  --attack_seed=0
```

## Robustness Benchmark

Use `robustness_benchmark.py` to run a full clean-vs-attacked benchmark from a JSON manifest. The default manifest is provided at `configs/adversarial_benchmark.json`.

```
python3 robustness_benchmark.py --manifest configs/adversarial_benchmark.json --output-root results
```

The benchmark writes:

- per-run `metrics.json`, `per_query.csv`, `per_bin.csv`, and `cluster_mass_stats.csv`
- per-attack `paired_metrics.json` and `query_pair_deltas.csv`
- benchmark-level `attack_summary.csv`, `attack_summary.json`, `query_pair_deltas.csv`, and `robustness_summary.json`

When `--feature_cache_dir` is provided, clean database descriptors are cached and reused across attack runs.

## Multi-Dataset Attack Runner

Use `run_adverserial_attacks.sh` to launch the first split of the adversarial benchmark. By default it runs both `nordland` and `sped` and then aggregates the outputs into a single comparison directory.

```
bash run_adverserial_attacks.sh
```

To run only one dataset, pass it as an argument:

```
bash run_adverserial_attacks.sh sped
```

This first script runs experiment indices `01` through `20`.

Use `run_adverserial_attacks_part2.sh` to continue the same benchmark run with experiment indices `21+`. It follows the same dataset selection behavior.

```
bash run_adverserial_attacks_part2.sh
```

For example:

```
bash run_adverserial_attacks_part2.sh sped
```

If you launch the second script later and want to target a specific split run, pass the same `RUN_STAMP` to both scripts:

```
RUN_STAMP=2026-04-13_12-00-00 bash run_adverserial_attacks.sh
RUN_STAMP=2026-04-13_12-00-00 bash run_adverserial_attacks_part2.sh
```

If `RUN_STAMP` is omitted for the second script, it appends to the latest available run under each dataset.

Useful environment variables:

- `EVAL_DATASETS_FOLDER`
- `NORDLAND_CHECKPOINT`
- `SPED_CHECKPOINT`
- `RESULTS_ROOT`
- `DEVICE`
- `USE_CROSSIMAGE_ENCODER`

The script writes:

- one benchmark run under `results/nordland/<timestamp>`
- one benchmark run under `results/sped/<timestamp>`
- one comparison bundle under `results/comparisons/<timestamp>`

The comparison bundle includes:

- `dataset_overview.csv` for clean baseline and worst-case robustness per dataset
- `dataset_attack_summary.csv` for all attack rows across both datasets
- `cross_dataset_comparison.csv` for side-by-side attack metrics across datasets
- `comparison_summary.json` with the source run roots and output metadata

You can also compare benchmark runs later without rerunning attacks:

```
python3 compare_benchmark_results.py \
  --results-root results \
  --datasets nordland sped
```

## SuperVLAD without cross-image encoder

Remove parameter `--crossimage_encoder` to run the SuperVLAD without cross-image encoder.

## 1-cluster VLAD

Set `--supervlad_clusters=1` and `--ghost_clusters=2` to run the 1-cluster VLAD. For example,

```
python3 eval.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=msls --resume=/path/to/trained/model/1-clusterVLAD.pth --backbone=dino --supervlad_clusters=1 --ghost_clusters=2
```

## Training with Automatic Mixed Precision

If you want to train models with Automatic Mixed Precision for faster training speed and less GPU memory usage. Just add parameter `--mixed_precision`. In this case, the cross-image encoder is not optimized separately and may not perform well.

## Trained Model

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th>cross-image<br />encoder</th>
      <th>download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SuperVLAD</td>
      <td align="center">:white_check_mark:</td>
      <td><a href="https://drive.google.com/file/d/1yomnWGTJko6nf3F2Ju6RWsLhP2EG82tL/view?usp=drive_link">LINK</a></td>
    </tr>
    <tr>
      <td>SuperVLAD</td>
      <td align="center">:x:</td>
      <td><a href="https://drive.google.com/file/d/1wRkUO4E8s5hNRNNIWcuA8RUvlGob3Tbf/view?usp=drive_link">LINK</a></td>
    </tr>
    <tr>
      <td>1-ClusterVLAD</td>
      <td align="center">:x:</td>
      <td><a href="https://drive.google.com/file/d/1pQcJx9n2-keAh9TttssZkz6D0vjpFWU6/view?usp=drive_link">LINK</a></td>
    </tr>
  </tbody>
</table>

## Acknowledgements

Parts of this repo are inspired by the following repositories:

[GSV-Cities](https://github.com/amaralibey/gsv-cities)

[Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)

[DINOv2](https://github.com/facebookresearch/dinov2)

## Citation

If you find this repo useful for your research, please cite the paper

```
@inproceedings{lu2024supervlad,
  title={SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition},
  author={Lu, Feng and Zhang, Xinyao and Ye, Canming and Dong, Shuting and Zhang, Lijun and Lan, Xiangyuan and Yuan, Chun},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  pages={5789--5816},
  year={2024}
}
```

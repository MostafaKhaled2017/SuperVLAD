from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RETRIEVAL_OUTPUT_DIR = "retrieval_diagnostics"


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SuperVLAD adversarial robustness benchmark.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/adversarial_benchmark.json"),
        help="Path to the benchmark manifest JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Root directory where benchmark results are written.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python3",
        help="Python executable used for nested eval.py runs.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path("eval.py"),
        help="Evaluation script to execute for each benchmark run.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip experiments whose output files already exist.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional existing or target run root. Use this to append later experiment ranges to the same benchmark run.",
    )
    parser.add_argument(
        "--exp-start-index",
        type=int,
        default=1,
        help="Inclusive experiment index to start from. 01 is the clean baseline.",
    )
    parser.add_argument(
        "--exp-end-index",
        type=int,
        default=None,
        help="Inclusive experiment index to stop at. Defaults to the last available experiment.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")
    return json.loads(path.read_text())


def _stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("Boolean flags must be handled separately")
    return str(value)


def build_cli_args(options: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in options.items():
        if value is None:
            continue
        cli_key = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(cli_key)
            continue
        if isinstance(value, list):
            args.append(cli_key)
            args.extend(_stringify_value(item) for item in value)
            continue
        args.extend([cli_key, _stringify_value(value)])
    return args


def experiment_dir_name(prefix: str, attack: dict[str, Any]) -> str:
    name = attack["attack_name"]
    parts = [prefix, name]
    if attack.get("attack_severity") is not None:
        parts.append(f"sev_{attack['attack_severity']}")
    if attack.get("attack_eps") is not None:
        eps_tag = str(attack["attack_eps"]).replace(".", "p")
        parts.append(f"eps_{eps_tag}")
    if attack.get("attack_steps") is not None:
        parts.append(f"steps_{attack['attack_steps']}")
    if attack.get("attack_keep_ratio") is not None:
        keep_tag = str(attack["attack_keep_ratio"]).replace(".", "p")
        parts.append(f"keep_{keep_tag}")
    if attack.get("attack_mask_mode") not in (None, "none"):
        parts.append(str(attack["attack_mask_mode"]))
    parts.append(f"seed_{attack.get('attack_seed', 0)}")
    return "_".join(parts)


def flatten_attacks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    attacks: list[dict[str, Any]] = []
    attack_groups = manifest.get("attacks", {})

    for attack_name, spec in attack_groups.get("corruptions", {}).items():
        for severity in spec.get("attack_severities", [1, 2, 3]):
            for seed in spec.get("attack_seeds", [0]):
                attacks.append({
                    "attack_name": attack_name,
                    "attack_severity": severity,
                    "attack_seed": seed,
                })

    for attack_name, spec in attack_groups.get("white_box", {}).items():
        for eps in spec.get("attack_eps", []):
            for seed in spec.get("attack_seeds", [0]):
                attack = {
                    "attack_name": attack_name,
                    "attack_eps": eps,
                    "attack_seed": seed,
                }
                if attack_name == "pgd_linf":
                    attack["attack_steps"] = spec.get("attack_steps", 10)
                    attack["attack_step_size"] = spec.get("attack_step_size")
                attacks.append(attack)

    token_masks = attack_groups.get("token_mask", {})
    for mask_mode, spec in token_masks.items():
        for keep_ratio in spec.get("attack_keep_ratios", []):
            for seed in spec.get("attack_seeds", [0]):
                attacks.append({
                    "attack_name": "token_mask",
                    "attack_mask_mode": mask_mode,
                    "attack_keep_ratio": keep_ratio,
                    "attack_seed": seed,
                })

    return attacks


def write_command(exp_dir: Path, command: list[str]) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "command.txt").write_text(subprocess.list2cmdline(command))


def _print_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _format_attack_summary(attack_options: dict[str, Any]) -> str:
    parts = [f"name={attack_options['attack_name']}"]
    if attack_options.get("attack_severity") is not None:
        parts.append(f"severity={attack_options['attack_severity']}")
    if attack_options.get("attack_eps") is not None:
        parts.append(f"eps={attack_options['attack_eps']}")
    if attack_options.get("attack_steps") is not None:
        parts.append(f"steps={attack_options['attack_steps']}")
    if attack_options.get("attack_keep_ratio") is not None:
        parts.append(f"keep_ratio={attack_options['attack_keep_ratio']}")
    if attack_options.get("attack_mask_mode") not in (None, "none"):
        parts.append(f"mask_mode={attack_options['attack_mask_mode']}")
    if attack_options.get("attack_seed") is not None:
        parts.append(f"seed={attack_options['attack_seed']}")
    return ", ".join(parts)


def run_command(exp_dir: Path, command: list[str]) -> None:
    write_command(exp_dir, command)
    log_path = exp_dir / "run.log"
    with log_path.open("w") as log_file:
        log_file.write(f"[command] {subprocess.list2cmdline(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            sys.stderr.write(line)
            sys.stderr.flush()
        process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed for {exp_dir.name}: {subprocess.list2cmdline(command)}")


def find_eval_output_dir(save_dir_key: str) -> Path:
    base = Path("test") / save_dir_key
    if not base.exists():
        raise FileNotFoundError(f"Eval output root not found: {base}")
    candidates = sorted([path for path in base.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No timestamped eval output directories found under {base}")
    return candidates[-1]


def collect_eval_outputs(exp_dir: Path, save_dir_key: str, retrieval_output_dir: str) -> None:
    eval_output_dir = find_eval_output_dir(save_dir_key)
    diagnostics_dir = eval_output_dir / retrieval_output_dir
    if not diagnostics_dir.exists():
        raise FileNotFoundError(f"Missing retrieval diagnostics output directory: {diagnostics_dir}")

    copies = {
        "summary.json": "metrics.json",
        "per_query.csv": "per_query.csv",
        "per_bin.csv": "per_bin.csv",
        "cluster_mass_stats.csv": "cluster_mass_stats.csv",
    }
    for source_name, target_name in copies.items():
        source = diagnostics_dir / source_name
        if not source.exists():
            raise FileNotFoundError(f"Missing expected eval output: {source}")
        shutil.copy2(source, exp_dir / target_name)


def metrics_complete(exp_dir: Path) -> bool:
    required = [
        exp_dir / "metrics.json",
        exp_dir / "per_query.csv",
        exp_dir / "per_bin.csv",
        exp_dir / "cluster_mass_stats.csv",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def _in_selected_range(index: int, start_index: int, end_index: int | None) -> bool:
    if index < start_index:
        return False
    if end_index is not None and index > end_index:
        return False
    return True


def run_eval_experiment(
    experiment_index: int,
    total_experiments: int,
    exp_dir: Path,
    save_dir_key: str,
    eval_options: dict[str, Any],
    attack_options: dict[str, Any],
    python_bin: str,
    eval_script: Path,
    retrieval_output_dir: str,
    skip_completed: bool,
) -> None:
    if skip_completed and metrics_complete(exp_dir):
        _print_status(
            f"[{experiment_index}/{total_experiments}] Skipping {exp_dir.name} "
            f"({_format_attack_summary(attack_options)})"
        )
        return

    _print_status(
        f"[{experiment_index}/{total_experiments}] Starting {exp_dir.name} "
        f"({_format_attack_summary(attack_options)})"
    )
    command = [
        python_bin,
        str(eval_script),
        *build_cli_args(eval_options),
        *build_cli_args(attack_options),
        "--save_dir",
        save_dir_key,
        "--retrieval_diagnostics_output_dir",
        retrieval_output_dir,
        "--enable_retrieval_diagnostics",
        "--return_debug_metrics",
    ]
    run_command(exp_dir, command)
    collect_eval_outputs(exp_dir, save_dir_key, retrieval_output_dir)
    _print_status(f"[{experiment_index}/{total_experiments}] Finished {exp_dir.name}")


def read_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compute_paired_outputs(baseline_dir: Path, attack_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_metrics = read_metrics(baseline_dir / "metrics.json")
    attack_metrics = read_metrics(attack_dir / "metrics.json")
    baseline_df = pd.read_csv(baseline_dir / "per_query.csv")
    attack_df = pd.read_csv(attack_dir / "per_query.csv")

    merged = baseline_df.merge(attack_df, on="query_index", suffixes=("_clean", "_attack"), how="inner")
    merged["top1_correct_clean"] = as_bool_series(merged["top1_correct_clean"])
    merged["top1_correct_attack"] = as_bool_series(merged["top1_correct_attack"])
    merged["margin_drop"] = merged["margin_clean"] - merged["margin_attack"]
    merged["best_positive_rank_delta"] = merged["best_positive_rank_attack"] - merged["best_positive_rank_clean"]
    merged["top1_flip"] = merged["top1_correct_clean"] & (~merged["top1_correct_attack"])
    merged["recovered"] = (~merged["top1_correct_clean"]) & merged["top1_correct_attack"]

    clean_recalls = baseline_metrics["recalls"]
    attacked_recalls = attack_metrics["recalls"]
    delta_recalls = [float(attacked - clean) for clean, attacked in zip(clean_recalls, attacked_recalls)]
    retention_recalls = [
        float(attacked / clean) if float(clean) != 0.0 else None
        for clean, attacked in zip(clean_recalls, attacked_recalls)
    ]

    clean_successes = int(merged["top1_correct_clean"].sum())
    clean_failures = int((~merged["top1_correct_clean"]).sum())
    top1_flips = int(merged["top1_flip"].sum())
    recoveries = int(merged["recovered"].sum())

    summary = {
        "attack": attack_metrics.get("attack", {}),
        "clean_recalls": [float(value) for value in clean_recalls],
        "attacked_recalls": [float(value) for value in attacked_recalls],
        "delta_recalls": delta_recalls,
        "retention_recalls": retention_recalls,
        "mean_margin_drop": float(merged["margin_drop"].dropna().mean()) if "margin_drop" in merged else None,
        "median_margin_drop": float(merged["margin_drop"].dropna().median()) if "margin_drop" in merged else None,
        "mean_best_positive_rank_delta": float(merged["best_positive_rank_delta"].dropna().mean()) if "best_positive_rank_delta" in merged else None,
        "top1_flip_rate": float(top1_flips / clean_successes) if clean_successes > 0 else None,
        "recovery_rate": float(recoveries / clean_failures) if clean_failures > 0 else None,
        "query_count": int(len(merged)),
        "clean_success_count": clean_successes,
        "clean_failure_count": clean_failures,
        "top1_flip_count": top1_flips,
        "recovery_count": recoveries,
    }

    merged.to_csv(attack_dir / "query_pair_deltas.csv", index=False)
    (attack_dir / "paired_metrics.json").write_text(json.dumps(summary, indent=2))
    return merged, summary


def summary_row(paired_summary: dict[str, Any]) -> dict[str, Any]:
    attack = paired_summary["attack"]
    row = {
        "attack_name": attack.get("attack_name"),
        "attack_severity": attack.get("attack_severity"),
        "attack_seed": attack.get("attack_seed"),
        "attack_eps": attack.get("attack_eps"),
        "attack_steps": attack.get("attack_steps"),
        "attack_step_size": attack.get("attack_step_size"),
        "attack_mask_mode": attack.get("attack_mask_mode"),
        "attack_keep_ratio": attack.get("attack_keep_ratio"),
        "query_count": paired_summary["query_count"],
        "mean_margin_drop": paired_summary["mean_margin_drop"],
        "median_margin_drop": paired_summary["median_margin_drop"],
        "mean_best_positive_rank_delta": paired_summary["mean_best_positive_rank_delta"],
        "top1_flip_rate": paired_summary["top1_flip_rate"],
        "recovery_rate": paired_summary["recovery_rate"],
    }
    recall_names = ["r_at_1", "r_at_5", "r_at_10", "r_at_100"]
    for index, recall_name in enumerate(recall_names):
        row[f"clean_{recall_name}"] = paired_summary["clean_recalls"][index]
        row[f"attacked_{recall_name}"] = paired_summary["attacked_recalls"][index]
        row[f"delta_{recall_name}"] = paired_summary["delta_recalls"][index]
        row[f"retention_{recall_name}"] = paired_summary["retention_recalls"][index]
    return row


def recompute_available_outputs(run_root: Path, baseline_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    all_query_pair_rows: list[pd.DataFrame] = []
    paired_summary_rows: list[dict[str, Any]] = []

    if not metrics_complete(baseline_dir):
        raise FileNotFoundError(f"Baseline metrics are required before aggregating outputs: {baseline_dir}")

    experiment_dirs = sorted(
        [
            path for path in run_root.iterdir()
            if path.is_dir() and path.name != "01_clean_baseline" and metrics_complete(path)
        ]
    )

    for exp_dir in experiment_dirs:
        paired_df, paired_summary = compute_paired_outputs(baseline_dir, exp_dir)
        paired_df["attack_name"] = paired_summary["attack"].get("attack_name")
        paired_df["attack_seed"] = paired_summary["attack"].get("attack_seed")
        paired_df["attack_severity"] = paired_summary["attack"].get("attack_severity")
        paired_df["attack_eps"] = paired_summary["attack"].get("attack_eps")
        paired_df["attack_steps"] = paired_summary["attack"].get("attack_steps")
        paired_df["attack_step_size"] = paired_summary["attack"].get("attack_step_size")
        paired_df["attack_mask_mode"] = paired_summary["attack"].get("attack_mask_mode")
        paired_df["attack_keep_ratio"] = paired_summary["attack"].get("attack_keep_ratio")
        paired_df["experiment_dir"] = exp_dir.name
        all_query_pair_rows.append(paired_df)
        paired_summary_rows.append(summary_row(paired_summary))

    if all_query_pair_rows:
        combined_query_pairs = pd.concat(all_query_pair_rows, ignore_index=True)
    else:
        combined_query_pairs = pd.DataFrame()
    return combined_query_pairs, paired_summary_rows


def aggregate_attack_summary(summary_rows: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows_df = pd.DataFrame(summary_rows)
    if rows_df.empty:
        rows_df.to_csv(output_dir / "attack_summary.csv", index=False)
        (output_dir / "attack_summary.json").write_text("[]")
        return rows_df

    group_cols = [
        "attack_name",
        "attack_severity",
        "attack_eps",
        "attack_steps",
        "attack_step_size",
        "attack_mask_mode",
        "attack_keep_ratio",
    ]
    aggregated = (
        rows_df.groupby(group_cols, dropna=False)
        .agg(
            seed_count=("attack_seed", "nunique"),
            seeds=("attack_seed", lambda values: sorted(int(value) for value in pd.Series(values).dropna().unique())),
            clean_r_at_1=("clean_r_at_1", "mean"),
            clean_r_at_5=("clean_r_at_5", "mean"),
            clean_r_at_10=("clean_r_at_10", "mean"),
            clean_r_at_100=("clean_r_at_100", "mean"),
            attacked_r_at_1_mean=("attacked_r_at_1", "mean"),
            attacked_r_at_1_worst=("attacked_r_at_1", "min"),
            attacked_r_at_5_mean=("attacked_r_at_5", "mean"),
            attacked_r_at_5_worst=("attacked_r_at_5", "min"),
            attacked_r_at_10_mean=("attacked_r_at_10", "mean"),
            attacked_r_at_10_worst=("attacked_r_at_10", "min"),
            attacked_r_at_100_mean=("attacked_r_at_100", "mean"),
            attacked_r_at_100_worst=("attacked_r_at_100", "min"),
            delta_r_at_1_mean=("delta_r_at_1", "mean"),
            delta_r_at_1_worst=("delta_r_at_1", "min"),
            delta_r_at_5_mean=("delta_r_at_5", "mean"),
            delta_r_at_5_worst=("delta_r_at_5", "min"),
            delta_r_at_10_mean=("delta_r_at_10", "mean"),
            delta_r_at_10_worst=("delta_r_at_10", "min"),
            delta_r_at_100_mean=("delta_r_at_100", "mean"),
            delta_r_at_100_worst=("delta_r_at_100", "min"),
            retention_r_at_1_mean=("retention_r_at_1", "mean"),
            retention_r_at_1_worst=("retention_r_at_1", "min"),
            retention_r_at_5_mean=("retention_r_at_5", "mean"),
            retention_r_at_5_worst=("retention_r_at_5", "min"),
            retention_r_at_10_mean=("retention_r_at_10", "mean"),
            retention_r_at_10_worst=("retention_r_at_10", "min"),
            retention_r_at_100_mean=("retention_r_at_100", "mean"),
            retention_r_at_100_worst=("retention_r_at_100", "min"),
            mean_margin_drop=("mean_margin_drop", "mean"),
            median_margin_drop=("median_margin_drop", "median"),
            mean_best_positive_rank_delta=("mean_best_positive_rank_delta", "mean"),
            top1_flip_rate=("top1_flip_rate", "mean"),
            recovery_rate=("recovery_rate", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    aggregated.to_csv(output_dir / "attack_summary.csv", index=False)
    (output_dir / "attack_summary.json").write_text(json.dumps(aggregated.to_dict(orient="records"), indent=2))
    return aggregated


def write_robustness_summary(attack_summary: pd.DataFrame, output_dir: Path) -> None:
    if attack_summary.empty:
        summary = {
            "attack_count": 0,
            "mean_retention_r_at_1": None,
            "mean_retention_r_at_5": None,
            "worst_case_r_at_1": None,
        }
    else:
        summary = {
            "attack_count": int(len(attack_summary)),
            "mean_retention_r_at_1": float(attack_summary["retention_r_at_1_mean"].dropna().mean()),
            "mean_retention_r_at_5": float(attack_summary["retention_r_at_5_mean"].dropna().mean()),
            "worst_case_r_at_1": float(attack_summary["attacked_r_at_1_worst"].min()),
            "worst_case_retention_r_at_1": float(attack_summary["retention_r_at_1_worst"].min()),
            "worst_case_delta_r_at_1": float(attack_summary["delta_r_at_1_worst"].min()),
        }
    (output_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    eval_options = dict(manifest.get("eval", {}))
    retrieval_output_dir = eval_options.pop("retrieval_diagnostics_output_dir", DEFAULT_RETRIEVAL_OUTPUT_DIR)
    eval_options.setdefault("test_method", "hard_resize")

    dataset_name = eval_options.get("eval_dataset_name", "dataset")
    if args.run_root is not None:
        run_root = args.run_root.resolve()
    else:
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_root = args.output_root / dataset_name / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)

    feature_cache_dir = run_root / "_feature_cache"
    manifest_copy_path = run_root / "manifest.json"
    manifest_copy_path.write_text(json.dumps(manifest, indent=2))

    temp_prefix = Path("tmp_eval_runs") / dataset_name / run_root.name
    eval_options["feature_cache_dir"] = str(feature_cache_dir)

    attacks = flatten_attacks(manifest)
    selected_indices = [
        index for index in range(1, len(attacks) + 2)
        if _in_selected_range(index, args.exp_start_index, args.exp_end_index)
    ]
    total_experiments = len(selected_indices)

    baseline_attack = {"attack_name": "none", "attack_seed": 0}
    baseline_dir = run_root / "01_clean_baseline"
    baseline_save_dir = str(temp_prefix / "01_clean_baseline")
    if _in_selected_range(1, args.exp_start_index, args.exp_end_index):
        selected_position = selected_indices.index(1) + 1
        run_eval_experiment(
            experiment_index=selected_position,
            total_experiments=total_experiments,
            exp_dir=baseline_dir,
            save_dir_key=baseline_save_dir,
            eval_options=eval_options,
            attack_options=baseline_attack,
            python_bin=args.python_bin,
            eval_script=args.eval_script,
            retrieval_output_dir=retrieval_output_dir,
            skip_completed=args.skip_completed,
        )
    elif not metrics_complete(baseline_dir):
        raise FileNotFoundError(
            f"Baseline experiment 01 is missing under {baseline_dir}. Run a range that includes 01 first."
        )

    for attack_index, attack in enumerate(attacks, start=2):
        if not _in_selected_range(attack_index, args.exp_start_index, args.exp_end_index):
            continue
        selected_position = selected_indices.index(attack_index) + 1
        exp_name = experiment_dir_name(f"{attack_index:02d}", attack)
        exp_dir = run_root / exp_name
        save_dir_key = str(temp_prefix / exp_name)
        run_eval_experiment(
            experiment_index=selected_position,
            total_experiments=total_experiments,
            exp_dir=exp_dir,
            save_dir_key=save_dir_key,
            eval_options=eval_options,
            attack_options=attack,
            python_bin=args.python_bin,
            eval_script=args.eval_script,
            retrieval_output_dir=retrieval_output_dir,
            skip_completed=args.skip_completed,
        )

    combined_query_pairs, paired_summary_rows = recompute_available_outputs(run_root, baseline_dir)
    combined_query_pairs.to_csv(run_root / "query_pair_deltas.csv", index=False)

    attack_summary = aggregate_attack_summary(paired_summary_rows, run_root)
    write_robustness_summary(attack_summary, run_root)
    print(run_root)


if __name__ == "__main__":
    main()

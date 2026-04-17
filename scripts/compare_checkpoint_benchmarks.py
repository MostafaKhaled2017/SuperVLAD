from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two checkpoint benchmark runs on clean and attacked metrics.")
    parser.add_argument("--reference-run-root", type=Path, required=True, help="Benchmark run root for the reference checkpoint.")
    parser.add_argument("--candidate-run-root", type=Path, required=True, help="Benchmark run root for the candidate checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison outputs. Defaults to results/checkpoint_comparisons/<timestamp>.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def attack_key_columns() -> list[str]:
    return [
        "attack_name",
        "attack_severity",
        "attack_eps",
        "attack_steps",
        "attack_step_size",
        "attack_mask_mode",
        "attack_keep_ratio",
    ]


def load_attack_summary(run_root: Path) -> pd.DataFrame:
    summary_path = require_file(run_root / "attack_summary.csv")
    return pd.read_csv(summary_path)


def load_clean_metrics(run_root: Path) -> dict[str, Any]:
    metrics_path = require_file(run_root / "01_clean_baseline" / "metrics.json")
    return load_json(metrics_path)


def build_clean_comparison(reference_run_root: Path, candidate_run_root: Path) -> dict[str, Any]:
    reference_metrics = load_clean_metrics(reference_run_root)
    candidate_metrics = load_clean_metrics(candidate_run_root)

    reference_recalls = [float(value) for value in reference_metrics.get("recalls", [])]
    candidate_recalls = [float(value) for value in candidate_metrics.get("recalls", [])]
    delta_recalls = [candidate - reference for reference, candidate in zip(reference_recalls, candidate_recalls)]

    return {
        "reference_run_root": str(reference_run_root),
        "candidate_run_root": str(candidate_run_root),
        "reference_checkpoint": reference_metrics.get("resume"),
        "candidate_checkpoint": candidate_metrics.get("resume"),
        "recall_values": reference_metrics.get("recall_values", []),
        "reference_recalls": reference_recalls,
        "candidate_recalls": candidate_recalls,
        "delta_recalls": delta_recalls,
    }


def build_attacked_comparison(reference_run_root: Path, candidate_run_root: Path) -> pd.DataFrame:
    reference_df = load_attack_summary(reference_run_root).copy()
    candidate_df = load_attack_summary(candidate_run_root).copy()

    key_cols = attack_key_columns()
    merged = reference_df.merge(
        candidate_df,
        on=key_cols,
        how="outer",
        suffixes=("_reference", "_candidate"),
    )

    compare_cols = [
        "clean_r_at_1",
        "clean_r_at_5",
        "clean_r_at_10",
        "clean_r_at_100",
        "attacked_r_at_1_mean",
        "attacked_r_at_5_mean",
        "attacked_r_at_10_mean",
        "attacked_r_at_100_mean",
        "delta_r_at_1_mean",
        "delta_r_at_5_mean",
        "delta_r_at_10_mean",
        "delta_r_at_100_mean",
        "retention_r_at_1_mean",
        "retention_r_at_5_mean",
        "retention_r_at_10_mean",
        "retention_r_at_100_mean",
        "attacked_r_at_1_worst",
        "delta_r_at_1_worst",
        "retention_r_at_1_worst",
    ]
    for column in compare_cols:
        ref_col = f"{column}_reference"
        cand_col = f"{column}_candidate"
        if ref_col in merged.columns and cand_col in merged.columns:
            merged[f"{column}_candidate_minus_reference"] = merged[cand_col] - merged[ref_col]

    return merged.sort_values(key_cols)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("results") / "checkpoint_comparisons" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_comparison = build_clean_comparison(
        reference_run_root=args.reference_run_root.resolve(),
        candidate_run_root=args.candidate_run_root.resolve(),
    )
    attacked_comparison = build_attacked_comparison(
        reference_run_root=args.reference_run_root.resolve(),
        candidate_run_root=args.candidate_run_root.resolve(),
    )

    (output_dir / "clean_comparison.json").write_text(json.dumps(clean_comparison, indent=2))
    attacked_comparison.to_csv(output_dir / "attacked_comparison.csv", index=False)

    summary = {
        "reference_run_root": str(args.reference_run_root.resolve()),
        "candidate_run_root": str(args.candidate_run_root.resolve()),
        "clean_delta_r_at_1": clean_comparison["delta_recalls"][0] if len(clean_comparison["delta_recalls"]) > 0 else None,
        "clean_delta_r_at_5": clean_comparison["delta_recalls"][1] if len(clean_comparison["delta_recalls"]) > 1 else None,
        "attack_rows": int(len(attacked_comparison)),
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare robustness benchmark outputs across datasets.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory that contains per-dataset benchmark runs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nordland", "msls"],
        help="Dataset names to compare when run roots are not provided explicitly.",
    )
    parser.add_argument(
        "--run-roots",
        nargs="*",
        default=None,
        help="Optional explicit benchmark run roots. When omitted, the latest run for each dataset is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated comparison outputs. Defaults to results/comparisons/<timestamp>.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_latest_run_root(results_root: Path, dataset: str) -> Path:
    dataset_root = results_root / dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset results directory not found: {dataset_root}")
    candidates = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No benchmark runs found for dataset: {dataset}")
    return candidates[-1]


def resolve_run_roots(results_root: Path, datasets: list[str], run_roots: list[str] | None) -> list[Path]:
    if run_roots:
        return [Path(path).resolve() for path in run_roots]
    return [find_latest_run_root(results_root, dataset).resolve() for dataset in datasets]


def infer_dataset_name(run_root: Path) -> str:
    return run_root.parent.name


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark output not found: {path}")
    return path


def attack_key_from_row(row: pd.Series) -> str:
    parts = [str(row["attack_name"])]
    if pd.notna(row.get("attack_severity")):
        parts.append(f"sev={int(row['attack_severity'])}")
    if pd.notna(row.get("attack_eps")):
        parts.append(f"eps={row['attack_eps']}")
    if pd.notna(row.get("attack_steps")):
        parts.append(f"steps={int(row['attack_steps'])}")
    if pd.notna(row.get("attack_step_size")):
        parts.append(f"step={row['attack_step_size']}")
    if pd.notna(row.get("attack_keep_ratio")):
        parts.append(f"keep={row['attack_keep_ratio']}")
    if pd.notna(row.get("attack_mask_mode")) and str(row["attack_mask_mode"]) != "none":
        parts.append(f"mode={row['attack_mask_mode']}")
    return "|".join(parts)


def load_dataset_outputs(run_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    attack_summary_path = require_file(run_root / "attack_summary.csv")
    robustness_summary_path = require_file(run_root / "robustness_summary.json")
    baseline_metrics_path = require_file(run_root / "01_clean_baseline" / "metrics.json")

    attack_summary = pd.read_csv(attack_summary_path)
    robustness_summary = load_json(robustness_summary_path)
    baseline_metrics = load_json(baseline_metrics_path)
    return attack_summary, robustness_summary, baseline_metrics


def build_dataset_overview(
    dataset: str,
    run_root: Path,
    robustness_summary: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    recalls = baseline_metrics.get("recalls", [])
    return {
        "dataset": dataset,
        "run_root": str(run_root),
        "attack_count": robustness_summary.get("attack_count"),
        "baseline_r_at_1": recalls[0] if len(recalls) > 0 else None,
        "baseline_r_at_5": recalls[1] if len(recalls) > 1 else None,
        "baseline_r_at_10": recalls[2] if len(recalls) > 2 else None,
        "baseline_r_at_100": recalls[3] if len(recalls) > 3 else None,
        "mean_retention_r_at_1": robustness_summary.get("mean_retention_r_at_1"),
        "mean_retention_r_at_5": robustness_summary.get("mean_retention_r_at_5"),
        "worst_case_r_at_1": robustness_summary.get("worst_case_r_at_1"),
        "worst_case_retention_r_at_1": robustness_summary.get("worst_case_retention_r_at_1"),
        "worst_case_delta_r_at_1": robustness_summary.get("worst_case_delta_r_at_1"),
    }


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flattened = []
    for column in df.columns:
        if isinstance(column, tuple):
            flattened.append("_".join(str(part) for part in column if part not in ("", None)))
        else:
            flattened.append(str(column))
    df.columns = flattened
    return df


def build_cross_dataset_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return combined

    combined = combined.copy()
    combined["attack_severity"] = combined["attack_severity"].where(combined["attack_severity"].notna(), -1)
    combined["attack_eps"] = combined["attack_eps"].where(combined["attack_eps"].notna(), "none")
    combined["attack_steps"] = combined["attack_steps"].where(combined["attack_steps"].notna(), -1)
    combined["attack_step_size"] = combined["attack_step_size"].where(combined["attack_step_size"].notna(), "none")
    combined["attack_keep_ratio"] = combined["attack_keep_ratio"].where(combined["attack_keep_ratio"].notna(), "none")
    combined["attack_mask_mode"] = combined["attack_mask_mode"].where(combined["attack_mask_mode"].notna(), "none")

    value_columns = [
        "attacked_r_at_1_mean",
        "attacked_r_at_5_mean",
        "delta_r_at_1_mean",
        "delta_r_at_5_mean",
        "retention_r_at_1_mean",
        "retention_r_at_5_mean",
        "attacked_r_at_1_worst",
        "delta_r_at_1_worst",
        "retention_r_at_1_worst",
    ]
    comparison = combined.pivot_table(
        index=[
            "attack_key",
            "attack_name",
            "attack_severity",
            "attack_eps",
            "attack_steps",
            "attack_step_size",
            "attack_keep_ratio",
            "attack_mask_mode",
        ],
        columns="dataset",
        values=value_columns,
        aggfunc="first",
    ).reset_index()
    comparison = flatten_columns(comparison)
    return comparison.sort_values("attack_key")


def main() -> None:
    args = parse_args()
    run_roots = resolve_run_roots(args.results_root, args.datasets, args.run_roots)
    output_dir = args.output_dir or (args.results_root / "comparisons" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    overview_rows = []

    for run_root in run_roots:
        dataset = infer_dataset_name(run_root)
        attack_summary, robustness_summary, baseline_metrics = load_dataset_outputs(run_root)
        attack_summary = attack_summary.copy()
        if not attack_summary.empty:
            attack_summary["dataset"] = dataset
            attack_summary["run_root"] = str(run_root)
            attack_summary["attack_key"] = attack_summary.apply(attack_key_from_row, axis=1)
            combined_rows.append(attack_summary)
        overview_rows.append(build_dataset_overview(dataset, run_root, robustness_summary, baseline_metrics))

    combined_df = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    overview_df = pd.DataFrame(overview_rows).sort_values("dataset")
    comparison_df = build_cross_dataset_comparison(combined_df)

    combined_df.to_csv(output_dir / "dataset_attack_summary.csv", index=False)
    overview_df.to_csv(output_dir / "dataset_overview.csv", index=False)
    comparison_df.to_csv(output_dir / "cross_dataset_comparison.csv", index=False)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset_overview": overview_df.to_dict(orient="records"),
        "dataset_attack_summary_rows": int(len(combined_df)),
        "cross_dataset_comparison_rows": int(len(comparison_df)),
        "source_run_roots": [str(path) for path in run_roots],
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(payload, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()

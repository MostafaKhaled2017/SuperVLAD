#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def get_token_dropout_seed(metrics: dict) -> int | None:
    if "token_dropout_seed" in metrics:
        return metrics["token_dropout_seed"]
    return metrics.get("step_a_seed")


def series_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "count": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def summarize_group(df: pd.DataFrame, columns: list[str]) -> dict:
    out = {}
    for col in columns:
        if col in df.columns:
            out[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std(ddof=1)),
            }
    return out


def validate_run_root(run_root: Path) -> None:
    required_paths = [
        run_root / "01_baseline" / "metrics.json",
        run_root / "01_baseline" / "per_query.csv",
        run_root / "01_baseline" / "per_bin.csv",
        run_root / "01_baseline" / "cluster_mass_stats.csv",
        run_root / "06_failure_transition" / "failure_transition_matrix.csv",
        run_root / "07_correlation_analysis" / "correlations.csv",
        run_root / "08_case_studies" / "case_studies.csv",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        missing_lines = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Run root is missing required outputs:\n{missing_lines}")


def baseline_summary(run_root: Path) -> dict:
    metrics = load_json(run_root / "01_baseline" / "metrics.json")
    per_query = pd.read_csv(run_root / "01_baseline" / "per_query.csv")
    per_bin = pd.read_csv(run_root / "01_baseline" / "per_bin.csv")
    cluster = pd.read_csv(run_root / "01_baseline" / "cluster_mass_stats.csv")

    per_query["top1_correct"] = per_query["top1_correct"].astype(bool)
    per_query["success_at_5"] = per_query["success_at_5"].astype(bool)
    per_query["success_at_10"] = per_query["success_at_10"].astype(bool)

    success = per_query[per_query["top1_correct"]].copy()
    failure = per_query[~per_query["top1_correct"]].copy()

    cluster_cols = [col for col in cluster.columns if col.startswith("cluster_mass_")]
    cluster_only = cluster[cluster_cols].copy()
    total_mass = cluster_only.sum(axis=1)
    cluster_share = cluster_only.div(total_mass, axis=0)

    enriched = per_query.merge(cluster, on="query_index", how="left")
    enriched["total_mass"] = total_mass
    enriched["dominant_share"] = cluster_share.max(axis=1)
    enriched["mass_range"] = cluster_only.max(axis=1) - cluster_only.min(axis=1)
    enriched["mass_cv"] = cluster_only.std(axis=1, ddof=1) / cluster_only.mean(axis=1)

    success_enriched = enriched[enriched["top1_correct"]].copy()
    failure_enriched = enriched[~enriched["top1_correct"]].copy()

    feature_cols = [
        "best_positive_distance",
        "best_negative_distance",
        "margin",
        "best_positive_rank",
        "min_mass",
        "max_mass",
        "mean_mass",
        "p10_mass",
        "num_low_mass_clusters",
        "assignment_entropy",
        "total_mass",
        "dominant_share",
        "mass_range",
        "mass_cv",
    ]

    easiest = (
        per_query.sort_values(["margin", "best_positive_rank"], ascending=[False, True])
        .head(10)[
            [
                "query_index",
                "positive_count",
                "margin",
                "best_positive_distance",
                "best_negative_distance",
                "best_positive_rank",
                "top1_prediction",
            ]
        ]
        .to_dict(orient="records")
    )
    hardest_success = (
        success.sort_values(["margin", "best_positive_rank"], ascending=[True, False])
        .head(10)[
            [
                "query_index",
                "positive_count",
                "margin",
                "best_positive_distance",
                "best_negative_distance",
                "best_positive_rank",
                "top1_prediction",
            ]
        ]
        .to_dict(orient="records")
    )
    hardest_failure = (
        failure.sort_values(["margin", "best_positive_rank"], ascending=[True, False])
        .head(10)[
            [
                "query_index",
                "positive_count",
                "margin",
                "best_positive_distance",
                "best_negative_distance",
                "best_positive_rank",
                "top1_prediction",
            ]
        ]
        .to_dict(orient="records")
    )
    borderline = (
        per_query.assign(abs_margin=per_query["margin"].abs())
        .sort_values(["abs_margin", "best_positive_rank"], ascending=[True, False])
        .head(10)[
            [
                "query_index",
                "top1_correct",
                "positive_count",
                "margin",
                "best_positive_distance",
                "best_negative_distance",
                "best_positive_rank",
                "top1_prediction",
            ]
        ]
        .to_dict(orient="records")
    )

    return {
        "metrics": metrics,
        "per_bin": per_bin.to_dict(orient="records"),
        "counts": {
            "query_count": int(len(per_query)),
            "success_count": int(success.shape[0]),
            "failure_count": int(failure.shape[0]),
            "top1_rate": float(success.shape[0] / len(per_query)),
            "top5_rate": float(per_query["success_at_5"].mean()),
            "top10_rate": float(per_query["success_at_10"].mean()),
        },
        "margin_overall": series_stats(per_query["margin"]),
        "margin_success": series_stats(success["margin"]),
        "margin_failure": series_stats(failure["margin"]),
        "rank_stats": series_stats(per_query["best_positive_rank"]),
        "feature_means_success": summarize_group(success_enriched, feature_cols),
        "feature_means_failure": summarize_group(failure_enriched, feature_cols),
        "feature_mean_deltas_failure_minus_success": {
            col: float(failure_enriched[col].mean() - success_enriched[col].mean())
            for col in feature_cols
            if col in success_enriched.columns and col in failure_enriched.columns
        },
        "cluster_columns": {
            col: series_stats(cluster[col]) for col in cluster_cols
        },
        "cluster_share_stats": {
            col: series_stats(cluster_share[col]) for col in cluster_share.columns
        },
        "derived_cluster_stats": {
            "total_mass": series_stats(total_mass),
            "dominant_share": series_stats(cluster_share.max(axis=1)),
            "mass_range": series_stats(cluster_only.max(axis=1) - cluster_only.min(axis=1)),
            "mass_cv": series_stats(cluster_only.std(axis=1, ddof=1) / cluster_only.mean(axis=1)),
        },
        "easiest_queries": easiest,
        "hardest_successes": hardest_success,
        "hardest_failures": hardest_failure,
        "borderline_queries": borderline,
        "token_count_unique": sorted(pd.to_numeric(per_query["token_count"], errors="coerce").dropna().unique().tolist()),
        "positive_count_stats": series_stats(per_query["positive_count"]),
    }


def dropout_summary(run_root: Path) -> dict:
    rows = []
    per_bin_rows = []
    for metrics_path in sorted(run_root.glob("04_dropout_keep_*_seed_*/metrics.json")):
        exp_dir = metrics_path.parent
        metrics = load_json(metrics_path)
        per_bin = pd.read_csv(exp_dir / "per_bin.csv")
        rows.append(
            {
                "exp": exp_dir.name,
                "keep_ratio": float(metrics["token_keep_ratio"]),
                "seed": int(get_token_dropout_seed(metrics)),
                "recall_at_1": float(metrics["recalls"][0]),
                "recall_at_5": float(metrics["recalls"][1]),
                "recall_at_10": float(metrics["recalls"][2]),
                "recall_at_100": float(metrics["recalls"][3]),
            }
        )
        if not per_bin.empty:
            row = per_bin.iloc[0].to_dict()
            row["keep_ratio"] = float(metrics["token_keep_ratio"])
            row["seed"] = int(get_token_dropout_seed(metrics))
            row["exp"] = exp_dir.name
            per_bin_rows.append(row)

    metrics_df = pd.DataFrame(rows).sort_values(["keep_ratio", "seed"])
    per_bin_df = pd.DataFrame(per_bin_rows).sort_values(["keep_ratio", "seed"])
    merged = metrics_df.merge(
        per_bin_df[["keep_ratio", "seed", "mean_margin", "median_margin", "negative_margin_rate"]],
        on=["keep_ratio", "seed"],
    )

    grouped = (
        merged.groupby("keep_ratio")
        .agg(
            recall_at_1_mean=("recall_at_1", "mean"),
            recall_at_1_std=("recall_at_1", "std"),
            recall_at_5_mean=("recall_at_5", "mean"),
            recall_at_5_std=("recall_at_5", "std"),
            recall_at_10_mean=("recall_at_10", "mean"),
            recall_at_10_std=("recall_at_10", "std"),
            recall_at_100_mean=("recall_at_100", "mean"),
            recall_at_100_std=("recall_at_100", "std"),
            mean_margin_mean=("mean_margin", "mean"),
            mean_margin_std=("mean_margin", "std"),
            median_margin_mean=("median_margin", "mean"),
            negative_margin_rate_mean=("negative_margin_rate", "mean"),
            negative_margin_rate_std=("negative_margin_rate", "std"),
            runs=("seed", "count"),
        )
        .reset_index()
        .sort_values("keep_ratio")
    )

    baseline = grouped[grouped["keep_ratio"] == 1.0].iloc[0]
    pairwise = []
    ordered = sorted(grouped["keep_ratio"].tolist(), reverse=True)
    for keep_a, keep_b in zip(ordered, ordered[1:]):
        row_a = grouped[grouped["keep_ratio"] == keep_a].iloc[0]
        row_b = grouped[grouped["keep_ratio"] == keep_b].iloc[0]
        pairwise.append(
            {
                "from_keep_ratio": float(keep_a),
                "to_keep_ratio": float(keep_b),
                "delta_recall_at_1": float(row_b["recall_at_1_mean"] - row_a["recall_at_1_mean"]),
                "delta_recall_at_5": float(row_b["recall_at_5_mean"] - row_a["recall_at_5_mean"]),
                "delta_recall_at_10": float(row_b["recall_at_10_mean"] - row_a["recall_at_10_mean"]),
                "delta_mean_margin": float(row_b["mean_margin_mean"] - row_a["mean_margin_mean"]),
                "delta_negative_margin_rate": float(row_b["negative_margin_rate_mean"] - row_a["negative_margin_rate_mean"]),
            }
        )

    vs_baseline = []
    for _, row in grouped.iterrows():
        vs_baseline.append(
            {
                "keep_ratio": float(row["keep_ratio"]),
                "delta_recall_at_1": float(row["recall_at_1_mean"] - baseline["recall_at_1_mean"]),
                "delta_recall_at_5": float(row["recall_at_5_mean"] - baseline["recall_at_5_mean"]),
                "delta_recall_at_10": float(row["recall_at_10_mean"] - baseline["recall_at_10_mean"]),
                "delta_mean_margin": float(row["mean_margin_mean"] - baseline["mean_margin_mean"]),
                "delta_negative_margin_rate": float(row["negative_margin_rate_mean"] - baseline["negative_margin_rate_mean"]),
            }
        )

    return {
        "runs": merged.to_dict(orient="records"),
        "grouped": grouped.to_dict(orient="records"),
        "pairwise": pairwise,
        "vs_baseline": vs_baseline,
    }


def transition_summary(run_root: Path) -> dict:
    pivot = pd.read_csv(run_root / "06_failure_transition" / "failure_transition_matrix.csv")
    keep_cols = [col for col in pivot.columns if col not in {"seed", "query_index"}]
    ordered = sorted(keep_cols, key=float)
    for col in ordered:
        pivot[col] = pivot[col].astype(bool)

    pivot["pattern"] = pivot[ordered].apply(lambda row: "".join("S" if bool(value) else "F" for value in row), axis=1)
    pattern_counts = pivot["pattern"].value_counts().rename_axis("pattern").reset_index(name="count")
    pattern_counts["fraction"] = pattern_counts["count"] / len(pivot)

    monotonic_patterns = 0
    for pattern, count in pattern_counts[["pattern", "count"]].itertuples(index=False):
        seen_success = False
        non_monotonic = False
        for char in pattern:
            if char == "S":
                seen_success = True
            elif seen_success:
                non_monotonic = True
                break
        if not non_monotonic:
            monotonic_patterns += count

    per_query = pivot.groupby("query_index")[ordered].agg(["sum", "mean"])
    per_query.columns = [f"{col}_{agg}" for col, agg in per_query.columns]

    robust_all_seeds = int(
        (
            (per_query["0.25_sum"] == 3)
            & (per_query["0.5_sum"] == 3)
            & (per_query["0.75_sum"] == 3)
            & (per_query["1.0_sum"] == 3)
        ).sum()
    )
    fragile_all_seeds = int(
        (
            (per_query["0.25_sum"] == 0)
            & (per_query["0.5_sum"] == 0)
            & (per_query["0.75_sum"] == 0)
            & (per_query["1.0_sum"] == 0)
        ).sum()
    )
    fail_only_025_all_seeds = int(
        (
            (per_query["0.25_sum"] == 0)
            & (per_query["0.5_sum"] == 3)
            & (per_query["0.75_sum"] == 3)
            & (per_query["1.0_sum"] == 3)
        ).sum()
    )
    fail_up_to_05_all_seeds = int(
        (
            (per_query["0.25_sum"] == 0)
            & (per_query["0.5_sum"] == 0)
            & (per_query["0.75_sum"] == 3)
            & (per_query["1.0_sum"] == 3)
        ).sum()
    )

    return {
        "ordered_keep_cols": ordered,
        "pattern_counts": pattern_counts.to_dict(orient="records"),
        "monotonic_count": int(monotonic_patterns),
        "monotonic_fraction": float(monotonic_patterns / len(pivot)),
        "all_pairs_count": int(len(pivot)),
        "robust_queries_all_seeds": robust_all_seeds,
        "fragile_queries_all_seeds": fragile_all_seeds,
        "fail_only_at_025_all_seeds": fail_only_025_all_seeds,
        "fail_at_025_and_05_all_seeds": fail_up_to_05_all_seeds,
    }


def correlation_summary(run_root: Path) -> dict:
    corr = pd.read_csv(run_root / "07_correlation_analysis" / "correlations.csv")
    corr = corr.sort_values("abs_pearson", ascending=False)
    strongest_positive = corr.sort_values("pearson", ascending=False).head(10)
    strongest_negative = corr.sort_values("pearson", ascending=True).head(10)
    weakest = corr.assign(abs_pearson=corr["pearson"].abs()).sort_values("abs_pearson").head(10)
    cluster_related = corr[corr["feature"].str.contains("mass|entropy|cluster", case=False, regex=True)]
    return {
        "all": corr.to_dict(orient="records"),
        "strongest_positive": strongest_positive.to_dict(orient="records"),
        "strongest_negative": strongest_negative.to_dict(orient="records"),
        "weakest": weakest.to_dict(orient="records"),
        "cluster_related": cluster_related.sort_values("abs_pearson", ascending=False).to_dict(orient="records"),
    }


def case_summary(run_root: Path) -> dict:
    cases = pd.read_csv(run_root / "08_case_studies" / "case_studies.csv")
    baseline = pd.read_csv(run_root / "01_baseline" / "per_query.csv")
    baseline["top1_correct"] = baseline["top1_correct"].astype(bool)
    cases["top1_correct"] = cases["top1_correct"].astype(bool)

    success = cases[cases["top1_correct"]].copy()
    failure = cases[~cases["top1_correct"]].copy()
    borderline = baseline.assign(abs_margin=baseline["margin"].abs()).sort_values("abs_margin").head(12)

    return {
        "selected_successes": success[
            [
                "query_index",
                "positive_count",
                "margin",
                "best_positive_rank",
                "best_positive_distance",
                "best_negative_distance",
                "assignment_entropy",
                "min_mass",
                "max_mass",
            ]
        ].to_dict(orient="records"),
        "selected_failures": failure[
            [
                "query_index",
                "positive_count",
                "margin",
                "best_positive_rank",
                "best_positive_distance",
                "best_negative_distance",
                "assignment_entropy",
                "min_mass",
                "max_mass",
            ]
        ].to_dict(orient="records"),
        "success_margin_stats": series_stats(success["margin"]),
        "failure_margin_stats": series_stats(failure["margin"]),
        "borderline_examples": borderline[
            [
                "query_index",
                "top1_correct",
                "margin",
                "best_positive_rank",
                "best_positive_distance",
                "best_negative_distance",
                "positive_count",
            ]
        ].to_dict(orient="records"),
    }


def derive_summary(run_root: Path) -> dict:
    return {
        "baseline": baseline_summary(run_root),
        "dropout": dropout_summary(run_root),
        "transitions": transition_summary(run_root),
        "correlations": correlation_summary(run_root),
        "case_studies": case_summary(run_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive a consolidated summary from an experiment run root.")
    parser.add_argument(
        "run_root",
        type=Path,
        help="Path to an experiment run root, for example results/nordland/2026-03-08_21-23-31",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <run_root>/_analysis_tmp/derived_summary.json",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    validate_run_root(run_root)

    output_json = args.output_json or (run_root / "_analysis_tmp" / "derived_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = derive_summary(run_root)
    output_json.write_text(json.dumps(payload, indent=2))
    print(output_json)


if __name__ == "__main__":
    main()

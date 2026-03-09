#!/usr/bin/env python3
"""Post-processing utilities for SuperVLAD retrieval diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    return pd.read_csv(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def success_failure(args: argparse.Namespace) -> None:
    df = _read_csv(Path(args.per_query_csv))
    if "top1_correct" not in df.columns:
        raise ValueError("per_query.csv must contain 'top1_correct'")

    df["top1_correct"] = df["top1_correct"].astype(bool)
    success = df[df["top1_correct"]]
    failure = df[~df["top1_correct"]]

    summary = {
        "query_count": int(len(df)),
        "success_count": int(len(success)),
        "failure_count": int(len(failure)),
        "success_rate_top1": float(len(success) / len(df)) if len(df) > 0 else None,
        "failure_rate_top1": float(len(failure) / len(df)) if len(df) > 0 else None,
    }

    if "margin" in df.columns:
        summary["margin"] = {
            "mean_success": float(success["margin"].dropna().mean()) if len(success) else None,
            "mean_failure": float(failure["margin"].dropna().mean()) if len(failure) else None,
            "median_success": float(success["margin"].dropna().median()) if len(success) else None,
            "median_failure": float(failure["margin"].dropna().median()) if len(failure) else None,
        }

    _write_json(Path(args.summary_json), summary)


def aggregate_per_bin_dropout(args: argparse.Namespace) -> None:
    base = Path(args.dropout_root)
    rows = []
    for per_bin_path in sorted(base.glob("04_dropout_keep_*_seed_*/per_bin.csv")):
        exp_dir = per_bin_path.parent.name
        parts = exp_dir.split("_")
        keep_tag = parts[3]
        seed = int(parts[-1])
        keep_ratio = float(f"{keep_tag[0]}.{keep_tag[1:]}") if len(keep_tag) == 3 else float(keep_tag)

        df = pd.read_csv(per_bin_path)
        if df.empty:
            continue
        df = df.copy()
        df["keep_ratio"] = keep_ratio
        df["seed"] = seed
        df["source_exp"] = exp_dir
        rows.append(df)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        merged = pd.concat(rows, ignore_index=True)
        merged.to_csv(out_path, index=False)
        grouped = (
            merged.groupby(["keep_ratio", "bin_label"], as_index=False)[["recall_at_1", "recall_at_5", "recall_at_10", "mean_margin"]]
            .mean(numeric_only=True)
            .sort_values(["keep_ratio", "bin_label"])
        )
        grouped.to_csv(Path(args.summary_csv), index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)
        pd.DataFrame().to_csv(Path(args.summary_csv), index=False)


def failure_transition(args: argparse.Namespace) -> None:
    base = Path(args.dropout_root)
    rows = []
    for per_query_path in sorted(base.glob("04_dropout_keep_*_seed_*/per_query.csv")):
        exp_dir = per_query_path.parent.name
        parts = exp_dir.split("_")
        keep_tag = parts[3]
        seed = int(parts[-1])
        keep_ratio = float(f"{keep_tag[0]}.{keep_tag[1:]}") if len(keep_tag) == 3 else float(keep_tag)

        df = pd.read_csv(per_query_path)
        if df.empty:
            continue
        df = df[["query_index", "top1_correct"]].copy()
        df["top1_correct"] = df["top1_correct"].astype(bool)
        df["keep_ratio"] = keep_ratio
        df["seed"] = seed
        rows.append(df)

    out_detail = Path(args.detail_csv)
    out_summary = Path(args.summary_json)
    out_detail.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        pd.DataFrame().to_csv(out_detail, index=False)
        _write_json(out_summary, {"error": "No dropout per_query.csv files found"})
        return

    all_df = pd.concat(rows, ignore_index=True)
    pivot = (
        all_df.pivot_table(
            index=["seed", "query_index"],
            columns="keep_ratio",
            values="top1_correct",
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    keep_levels = list(pivot.columns)

    transitions = []
    for i in range(len(keep_levels) - 1):
        hi = keep_levels[i]
        lo = keep_levels[i + 1]
        sub = pivot[[hi, lo]].dropna()
        if sub.empty:
            continue
        improved = int(((sub[hi] == False) & (sub[lo] == True)).sum())
        degraded = int(((sub[hi] == True) & (sub[lo] == False)).sum())
        stayed_success = int(((sub[hi] == True) & (sub[lo] == True)).sum())
        stayed_failure = int(((sub[hi] == False) & (sub[lo] == False)).sum())
        transitions.append(
            {
                "from_keep_ratio": float(hi),
                "to_keep_ratio": float(lo),
                "pair_count": int(len(sub)),
                "improved_count": improved,
                "degraded_count": degraded,
                "stayed_success_count": stayed_success,
                "stayed_failure_count": stayed_failure,
            }
        )

    pivot.to_csv(out_detail)
    _write_json(
        out_summary,
        {
            "keep_ratios": [float(k) for k in keep_levels],
            "transition_pairs": transitions,
        },
    )


def correlation(args: argparse.Namespace) -> None:
    per_query = _read_csv(Path(args.per_query_csv))
    cluster = _read_csv(Path(args.cluster_csv))
    if "query_index" not in per_query.columns or "query_index" not in cluster.columns:
        raise ValueError("Both CSVs must have query_index")
    if "margin" not in per_query.columns:
        raise ValueError("per_query.csv must contain margin")

    merged = per_query.merge(cluster, on="query_index", how="inner")
    numeric_cols = [c for c in merged.columns if c != "query_index" and pd.api.types.is_numeric_dtype(merged[c])]
    if "margin" not in numeric_cols:
        raise ValueError("margin must be numeric")

    rows = []
    for col in numeric_cols:
        if col == "margin":
            continue
        sub = merged[["margin", col]].dropna()
        if len(sub) < 3:
            continue
        pearson = sub["margin"].corr(sub[col], method="pearson")
        spearman = sub["margin"].corr(sub[col], method="spearman")
        rows.append(
            {
                "feature": col,
                "n": int(len(sub)),
                "pearson": float(pearson) if pearson is not None else np.nan,
                "spearman": float(spearman) if spearman is not None else np.nan,
                "abs_pearson": float(abs(pearson)) if pearson is not None else np.nan,
            }
        )

    out_csv = Path(args.output_csv)
    out_json = Path(args.summary_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr_df = pd.DataFrame(rows).sort_values("abs_pearson", ascending=False) if rows else pd.DataFrame()
    corr_df.to_csv(out_csv, index=False)
    top = corr_df.head(10).to_dict(orient="records") if not corr_df.empty else []
    _write_json(out_json, {"pair_count": int(len(merged)), "top_correlations": top})


def case_studies(args: argparse.Namespace) -> None:
    df = _read_csv(Path(args.per_query_csv))
    if "top1_correct" not in df.columns:
        raise ValueError("per_query.csv must contain top1_correct")

    df["top1_correct"] = df["top1_correct"].astype(bool)
    has_margin = "margin" in df.columns
    n_each = int(args.n_each)

    success = df[df["top1_correct"]].copy()
    failure = df[~df["top1_correct"]].copy()

    if has_margin:
        success = success.sort_values("margin", ascending=False)
        failure = failure.sort_values("margin", ascending=True)

    selected = pd.concat([success.head(n_each), failure.head(n_each)], ignore_index=True)
    selected.to_csv(args.output_csv, index=False)

    summary = {
        "total_queries": int(len(df)),
        "selected_rows": int(len(selected)),
        "success_selected": int((selected["top1_correct"] == True).sum()),
        "failure_selected": int((selected["top1_correct"] == False).sum()),
        "n_each_requested": n_each,
    }
    _write_json(Path(args.summary_json), summary)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-processing for SuperVLAD retrieval diagnostics")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sf = sub.add_parser("success_failure")
    p_sf.add_argument("--per-query-csv", required=True)
    p_sf.add_argument("--summary-json", required=True)
    p_sf.set_defaults(func=success_failure)

    p_pb = sub.add_parser("aggregate_per_bin_dropout")
    p_pb.add_argument("--dropout-root", required=True)
    p_pb.add_argument("--output-csv", required=True)
    p_pb.add_argument("--summary-csv", required=True)
    p_pb.set_defaults(func=aggregate_per_bin_dropout)

    p_ft = sub.add_parser("failure_transition")
    p_ft.add_argument("--dropout-root", required=True)
    p_ft.add_argument("--detail-csv", required=True)
    p_ft.add_argument("--summary-json", required=True)
    p_ft.set_defaults(func=failure_transition)

    p_corr = sub.add_parser("correlation")
    p_corr.add_argument("--per-query-csv", required=True)
    p_corr.add_argument("--cluster-csv", required=True)
    p_corr.add_argument("--output-csv", required=True)
    p_corr.add_argument("--summary-json", required=True)
    p_corr.set_defaults(func=correlation)

    p_case = sub.add_parser("case_studies")
    p_case.add_argument("--per-query-csv", required=True)
    p_case.add_argument("--output-csv", required=True)
    p_case.add_argument("--summary-json", required=True)
    p_case.add_argument("--n-each", type=int, default=10)
    p_case.set_defaults(func=case_studies)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

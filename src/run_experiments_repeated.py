#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_dataset import load_dataset
from src.data.split_data import make_split_indices
from src.evaluation.evaluate_models import compute_metrics
from src.features.build_feature_views import build_feature_views
from src.models.train_models import train_model
from src.utils.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed Step 3 experiments")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--reclass-model",
        default="random_forest",
        help="Model key for reclassification analysis (default: random_forest).",
    )
    parser.add_argument(
        "--reclass-feature-a",
        default="financial_enriched",
        help="Baseline feature set key for reclassification analysis.",
    )
    parser.add_argument(
        "--reclass-feature-b",
        default="esg_financial_enriched",
        help="Comparison feature set key for reclassification analysis.",
    )
    parser.add_argument(
        "--reclass-splits",
        nargs="*",
        default=None,
        help="Optional split strategies to include in reclassification analysis (defaults to all).",
    )
    parser.add_argument(
        "--hy-pattern",
        default=r"high yield",
        help="Case-insensitive regex identifying High Yield labels.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_agg_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat = []
    for left, right in df.columns.to_flat_index():
        if right:
            flat.append(f"{left}_{right}")
        else:
            flat.append(str(left))
    df.columns = flat
    return df


def _to_hy_mask(labels: pd.Series, hy_regex: re.Pattern[str]) -> pd.Series:
    return labels.astype(str).str.contains(hy_regex, na=False)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value)


def _build_reclassification_frames(
    predictions_by_seed: dict[tuple[int, str, str, str], pd.Series],
    seeds: list[int],
    split_types: list[str],
    model_name: str,
    feature_set_a: str,
    feature_set_b: str,
    hy_regex: re.Pattern[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows: list[dict[str, Any]] = []

    for split_type in split_types:
        for seed in seeds:
            key_a = (seed, split_type, model_name, feature_set_a)
            key_b = (seed, split_type, model_name, feature_set_b)
            pred_a = predictions_by_seed.get(key_a)
            pred_b = predictions_by_seed.get(key_b)

            if pred_a is None or pred_b is None:
                continue

            if pred_a.index.equals(pred_b.index):
                aligned_a = pred_a
                aligned_b = pred_b
                dropped_count = 0
            else:
                common_idx = pred_a.index.intersection(pred_b.index)
                aligned_a = pred_a.loc[common_idx]
                aligned_b = pred_b.loc[common_idx]
                dropped_count = int((len(pred_a) - len(aligned_a)) + (len(pred_b) - len(aligned_b)))

            diffs = aligned_a != aligned_b
            a_hy = _to_hy_mask(aligned_a, hy_regex)
            b_hy = _to_hy_mask(aligned_b, hy_regex)

            total_reclass = int(diffs.sum())
            hy_to_ig = int((a_hy & ~b_hy).sum())
            ig_to_hy = int((~a_hy & b_hy).sum())

            per_seed_rows.append(
                {
                    "split_type": split_type,
                    "seed": seed,
                    "model_name": model_name,
                    "feature_set_a": feature_set_a,
                    "feature_set_b": feature_set_b,
                    "n_compared": int(len(aligned_a)),
                    "dropped_due_to_misalignment": dropped_count,
                    "total_reclassified": total_reclass,
                    "hy_to_ig": hy_to_ig,
                    "ig_to_hy": ig_to_hy,
                    "reclass_rate": (float(total_reclass) / len(aligned_a)) if len(aligned_a) else 0.0,
                }
            )

    per_seed = pd.DataFrame(per_seed_rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame()

    summary = (
        per_seed.groupby(["split_type", "model_name", "feature_set_a", "feature_set_b"], as_index=False)
        .agg(
            seeds_compared=("seed", "count"),
            avg_n_compared=("n_compared", "mean"),
            avg_total_reclassified=("total_reclassified", "mean"),
            avg_hy_to_ig=("hy_to_ig", "mean"),
            avg_ig_to_hy=("ig_to_hy", "mean"),
            avg_reclass_rate=("reclass_rate", "mean"),
        )
    )
    return per_seed, summary


def _build_reclassification_markdown(
    reclass_summary: pd.DataFrame,
    reclass_per_seed: pd.DataFrame,
    seeds: list[int],
    model_name: str,
    feature_set_a: str,
    feature_set_b: str,
    hy_pattern: str,
) -> str:
    lines = [
        "# Reclassification Summary (Repeated Seeds)",
        "",
        f"Model: `{model_name}`",
        f"Baseline feature set (A): `{feature_set_a}`",
        f"Comparison feature set (B): `{feature_set_b}`",
        f"HY detection regex: `{hy_pattern}`",
        f"Requested seeds: `{seeds[0]}..{seeds[-1]}` (n={len(seeds)})",
        "",
        "A reclassification is counted when a test-row prediction differs between A and B.",
        "HY->IG means A predicts High Yield and B predicts Investment Grade.",
        "IG->HY means A predicts Investment Grade and B predicts High Yield.",
        "",
    ]

    if reclass_summary.empty:
        lines.append("No matching prediction sets were available for this comparison.")
        return "\n".join(lines)

    lines.extend(
        [
            "## Average counts per seed",
            "",
            "| split_type | seeds_compared | avg_n_compared | avg_reclassified | avg_HY_to_IG | avg_IG_to_HY | avg_reclass_rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in reclass_summary.iterrows():
        lines.append(
            f"| {row['split_type']} | {int(row['seeds_compared'])} | {row['avg_n_compared']:.1f} | "
            f"{row['avg_total_reclassified']:.1f} | {row['avg_hy_to_ig']:.1f} | "
            f"{row['avg_ig_to_hy']:.1f} | {row['avg_reclass_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Seed-level details",
            "",
            "| split_type | seed | n_compared | dropped_due_to_misalignment | total_reclassified | HY_to_IG | IG_to_HY | reclass_rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in reclass_per_seed.sort_values(["split_type", "seed"]).iterrows():
        lines.append(
            f"| {row['split_type']} | {int(row['seed'])} | {int(row['n_compared'])} | "
            f"{int(row['dropped_due_to_misalignment'])} | {int(row['total_reclassified'])} | "
            f"{int(row['hy_to_ig'])} | {int(row['ig_to_hy'])} | {row['reclass_rate']:.3f} |"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.n_seeds <= 0:
        raise ValueError("--n-seeds must be >= 1")

    cfg = load_config(args.config)
    outputs = cfg["outputs"]

    ensure_dir(outputs["metrics"])
    ensure_dir(outputs["reports"])

    df = load_dataset(cfg["dataset"]["input_csv"])
    target_col = cfg["dataset"]["target_column"]
    group_col = cfg["dataset"]["group_column"]
    views = build_feature_views(df, cfg["feature_sets"])

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    rows: list[dict[str, Any]] = []
    predictions_by_seed: dict[tuple[int, str, str, str], pd.Series] = {}

    split_types = cfg["split"]["strategies"]
    if args.reclass_splits is None:
        reclass_splits = split_types
    else:
        unknown_splits = sorted(set(args.reclass_splits) - set(split_types))
        if unknown_splits:
            raise ValueError(f"--reclass-splits contains unknown split(s): {unknown_splits}")
        reclass_splits = args.reclass_splits

    if args.reclass_feature_a not in views:
        raise ValueError(f"--reclass-feature-a '{args.reclass_feature_a}' is not in config feature_sets")
    if args.reclass_feature_b not in views:
        raise ValueError(f"--reclass-feature-b '{args.reclass_feature_b}' is not in config feature_sets")
    if args.reclass_model not in cfg["models"]:
        raise ValueError(f"--reclass-model '{args.reclass_model}' is not in config models")

    for seed in seeds:
        print(f"Running repeated-seed batch for seed={seed}")
        for split_type in split_types:
            split = make_split_indices(
                df=df,
                target_col=target_col,
                group_col=group_col,
                strategy=split_type,
                test_size=cfg["split"]["test_size"],
                random_state=seed,
            )

            for feature_set_name, view in views.items():
                X = view["frame"][view["features"]]
                y = view["frame"][target_col]

                X_train = X.loc[split["train_idx"]]
                X_test = X.loc[split["test_idx"]]
                y_train = y.loc[split["train_idx"]]
                y_test = y.loc[split["test_idx"]]

                for model_name, model_cfg in cfg["models"].items():
                    model = train_model(
                        model_name=model_name,
                        model_params=model_cfg.get("params", {}),
                        class_weight_mode=cfg["training"]["class_weight_mode"],
                        X_train=X_train,
                        y_train=y_train,
                        resampling=cfg["training"].get("resampling", "none"),
                        random_state=seed,
                    )

                    y_prob = model.predict_proba(X_test)
                    if hasattr(model, "_label_encoder"):
                        label_encoder = getattr(model, "_label_encoder")
                        y_pred_encoded = model.predict(X_test)
                        y_pred = label_encoder.inverse_transform(y_pred_encoded.astype(int))
                        classes = [str(c) for c in label_encoder.classes_]
                    else:
                        y_pred = model.predict(X_test)
                        classes = [str(c) for c in model.named_steps["model"].classes_]

                    metrics = compute_metrics(y_test, y_pred, y_prob, classes)
                    metrics.update(
                        {
                            "seed": seed,
                            "feature_set": feature_set_name,
                            "model_name": model_name,
                            "split_type": split_type,
                            "train_rows": int(X_train.shape[0]),
                            "test_rows": int(X_test.shape[0]),
                        }
                    )
                    rows.append(metrics)

                    predictions_by_seed[(seed, split_type, model_name, feature_set_name)] = pd.Series(
                        data=y_pred,
                        index=X_test.index,
                    )

    summary = pd.DataFrame(rows)
    summary_path = Path(outputs["metrics"]) / "repeated_seed_run_summary.csv"
    summary.to_csv(summary_path, index=False)

    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
    ]
    aggregate = (
        summary.groupby(["split_type", "feature_set", "model_name"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate = _flatten_agg_columns(aggregate)
    aggregate["n_seeds"] = len(seeds)
    std_cols = [c for c in aggregate.columns if c.endswith("_std")]
    aggregate[std_cols] = aggregate[std_cols].fillna(0.0)
    aggregate_path = Path(outputs["metrics"]) / "repeated_seed_aggregate_mean_std.csv"
    aggregate.to_csv(aggregate_path, index=False)

    hy_regex = re.compile(args.hy_pattern, re.IGNORECASE)
    reclass_per_seed, reclass_summary = _build_reclassification_frames(
        predictions_by_seed=predictions_by_seed,
        seeds=seeds,
        split_types=reclass_splits,
        model_name=args.reclass_model,
        feature_set_a=args.reclass_feature_a,
        feature_set_b=args.reclass_feature_b,
        hy_regex=hy_regex,
    )

    reclass_stem = (
        f"reclass_{_safe_name(args.reclass_model)}_"
        f"{_safe_name(args.reclass_feature_a)}_vs_{_safe_name(args.reclass_feature_b)}"
    )
    reclass_per_seed_path = Path(outputs["metrics"]) / f"{reclass_stem}_by_seed.csv"
    reclass_summary_path = Path(outputs["metrics"]) / f"{reclass_stem}_summary.csv"
    reclass_report_path = Path(outputs["reports"]) / f"{reclass_stem}.md"
    if not reclass_per_seed.empty:
        reclass_per_seed.to_csv(reclass_per_seed_path, index=False)
        reclass_summary.to_csv(reclass_summary_path, index=False)
    reclass_report_text = _build_reclassification_markdown(
        reclass_summary=reclass_summary,
        reclass_per_seed=reclass_per_seed,
        seeds=seeds,
        model_name=args.reclass_model,
        feature_set_a=args.reclass_feature_a,
        feature_set_b=args.reclass_feature_b,
        hy_pattern=args.hy_pattern,
    )
    reclass_report_path.write_text(reclass_report_text, encoding="utf-8")

    top = aggregate.sort_values("f1_macro_mean", ascending=False).head(10)
    report_path = Path(outputs["reports"]) / "repeated_seed_top_by_f1_mean.md"
    lines = [
        "# Top Runs by Macro F1 (Mean Across Seeds)",
        "",
        f"Seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)})",
        "",
        "| split_type | feature_set | model_name | f1_macro_mean | f1_macro_std | roc_auc_ovr_macro_mean | roc_auc_ovr_macro_std |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['split_type']} | {row['feature_set']} | {row['model_name']} | "
            f"{row['f1_macro_mean']:.4f} | {row['f1_macro_std']:.4f} | "
            f"{row['roc_auc_ovr_macro_mean']:.4f} | {row['roc_auc_ovr_macro_std']:.4f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nStep 3 repeated-seed evaluation complete.")
    print(f"Per-run summary: {summary_path}")
    print(f"Aggregate mean/std: {aggregate_path}")
    print(f"Top report: {report_path}")
    if not reclass_per_seed.empty:
        print(f"Reclassification by seed: {reclass_per_seed_path}")
        print(f"Reclassification summary: {reclass_summary_path}")
        print(f"Reclassification report: {reclass_report_path}")
        for _, row in reclass_summary.iterrows():
            print(
                "  "
                f"[{row['split_type']}] Avg reclassified per seed: {row['avg_total_reclassified']:.1f}, "
                f"Avg HY->IG: {row['avg_hy_to_ig']:.1f}, "
                f"Avg IG->HY: {row['avg_ig_to_hy']:.1f}"
            )
    else:
        print("No reclassification comparison produced (missing matching prediction sets).")
        print(f"Reclassification report: {reclass_report_path}")


if __name__ == "__main__":
    main()

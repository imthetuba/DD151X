#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

    for seed in seeds:
        print(f"Running repeated-seed batch for seed={seed}")
        for split_type in cfg["split"]["strategies"]:
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


if __name__ == "__main__":
    main()

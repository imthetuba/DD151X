#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_dataset import load_dataset
from src.data.split_data import make_split_indices
from src.evaluation.evaluate_models import compute_metrics, extract_feature_importance, save_confusion_plot
from src.features.build_feature_views import build_feature_views
from src.models.train_models import train_model
from src.utils.io_utils import ensure_dir, run_prefix, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 2 model experiments")
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_split_indices(out_path: Path, train_idx: list[int], test_idx: list[int]) -> None:
    payload = {
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    save_json(out_path, payload)


def build_predictions_frame(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: Any,
    y_prob: Any,
    classes: list[str],
) -> pd.DataFrame:
    pred = pd.DataFrame(index=X_test.index)
    pred["true_label"] = y_test.values
    pred["predicted_label"] = y_pred

    for col_idx, class_name in enumerate(classes):
        pred[f"prob_{class_name}"] = y_prob[:, col_idx]

    pred = pred.reset_index(drop=False).rename(columns={"index": "row_index"})
    return pred


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    df = load_dataset(cfg["dataset"]["input_csv"])
    target_col = cfg["dataset"]["target_column"]
    group_col = cfg["dataset"]["group_column"]

    feature_sets = cfg["feature_sets"]
    views = build_feature_views(df, feature_sets)

    outputs = cfg["outputs"]
    ensure_dir(outputs["splits"])
    ensure_dir(outputs["trained_models"])
    ensure_dir(outputs["metrics"])
    ensure_dir(outputs["plots"])
    ensure_dir(outputs["feature_importance"])
    ensure_dir(outputs["predictions"])
    ensure_dir(outputs["reports"])

    summary_rows: list[dict[str, Any]] = []

    for split_type in cfg["split"]["strategies"]:
        split = make_split_indices(
            df=df,
            target_col=target_col,
            group_col=group_col,
            strategy=split_type,
            test_size=cfg["split"]["test_size"],
            random_state=cfg["split"]["random_state"],
        )

        split_name = f"split_{split_type}.json"
        save_split_indices(Path(outputs["splits"]) / split_name, split["train_idx"], split["test_idx"])

        for feature_set_name, view in views.items():
            X = view["frame"][view["features"]]
            y = view["frame"][target_col]

            X_train = X.loc[split["train_idx"]]
            X_test = X.loc[split["test_idx"]]
            y_train = y.loc[split["train_idx"]]
            y_test = y.loc[split["test_idx"]]

            for model_name, model_cfg in cfg["models"].items():
                prefix = run_prefix(feature_set_name, model_name, split_type)
                print(f"Running {prefix} ...")

                model = train_model(
                    model_name=model_name,
                    model_params=model_cfg.get("params", {}),
                    class_weight_mode=cfg["training"]["class_weight_mode"],
                    X_train=X_train,
                    y_train=y_train,
                    resampling=cfg["training"].get("resampling", "none"),
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
                        "run_id": prefix,
                        "feature_set": feature_set_name,
                        "model_name": model_name,
                        "split_type": split_type,
                        "train_rows": int(X_train.shape[0]),
                        "test_rows": int(X_test.shape[0]),
                    }
                )

                metrics_path = Path(outputs["metrics"]) / f"{prefix}.json"
                save_json(metrics_path, metrics)

                model_path = Path(outputs["trained_models"]) / f"{prefix}.joblib"
                joblib.dump(model, model_path)

                pred_frame = build_predictions_frame(X_test, y_test, y_pred, y_prob, classes)
                pred_path = Path(outputs["predictions"]) / f"{prefix}.csv"
                pred_frame.to_csv(pred_path, index=False)

                cm_path = Path(outputs["plots"]) / f"{prefix}_confusion_matrix.png"
                save_confusion_plot(y_test, y_pred, classes, cm_path)

                importance = extract_feature_importance(model.named_steps["model"], view["features"])
                if not importance.empty:
                    imp_path = Path(outputs["feature_importance"]) / f"{prefix}.csv"
                    importance.to_csv(imp_path, index=False)

                summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path(outputs["metrics"]) / "run_summary_all.csv"
    summary_df.to_csv(summary_csv, index=False)

    by_f1 = summary_df.sort_values("f1_macro", ascending=False).head(10)
    report_path = Path(outputs["reports"]) / "top_runs_by_f1.md"
    lines = [
        "# Top Runs by Macro F1",
        "",
        "| run_id | feature_set | model_name | split_type | f1_macro | roc_auc_ovr_macro |",
        "|---|---|---|---|---:|---:|",
    ]
    for _, row in by_f1.iterrows():
        lines.append(
            f"| {row['run_id']} | {row['feature_set']} | {row['model_name']} | {row['split_type']} | "
            f"{row['f1_macro']:.4f} | {row['roc_auc_ovr_macro']:.4f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nAll experiments finished.")
    print(f"Summary: {summary_csv}")
    print(f"Top-runs report: {report_path}")


if __name__ == "__main__":
    main()

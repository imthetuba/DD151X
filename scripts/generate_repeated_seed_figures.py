#!/usr/bin/env python3
"""Generate figures and LaTeX tables from repeated-seed experiment results.

Produces:
  1. Bar charts comparing F1 macro (mean +/- std) across models and feature sets
  2. Bar charts comparing ROC AUC (mean +/- std) across models and feature sets
  3. ROC curves (one-vs-rest) from single-run predictions
  4. Feature importance plots from single-run trained models
  5. LaTeX tables: top-N configs, full results per split, per-model summary
  6. Confusion matrices for top-5 configurations by mean F1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
METRICS = OUTPUTS / "03_metrics"
PREDICTIONS = OUTPUTS / "06_predictions"
FEATURE_IMPORTANCE = OUTPUTS / "05_feature_importance"
FIGURES = OUTPUTS / "08_figures"
TABLES = OUTPUTS / "09_tables"

MODEL_ORDER = ["logistic_regression", "naive_bayes", "random_forest", "xgboost"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
FEATURE_SET_ORDER = [
    "financial_only",
    "financial_enriched",
    "esg_only",
    "esg_financial",
    "esg_financial_enriched",
]
FEATURE_SET_LABELS = {
    "financial_only": "Financial Only",
    "financial_enriched": "Financial Enriched",
    "esg_only": "ESG Only",
    "esg_financial": "ESG + Financial",
    "esg_financial_enriched": "ESG + Financial Enriched",
}
SPLIT_LABELS = {"stratified": "Stratified Split", "grouped": "Grouped Split"}

MODEL_COLORS = {
    "logistic_regression": "#08306B",  # dark navy
    "naive_bayes": "#2171B5",          # medium blue
    "random_forest": "#6BAED6",        # steel blue
    "xgboost": "#BDD7E7",             # light blue
}

CLASS_COLORS = ["#08306B", "#4292C6", "#BDD7E7"]

# Shared style: clean white background, subtle grid
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate repeated-seed figures")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1 & 2  –  Grouped bar charts (F1 macro, ROC AUC)
# ---------------------------------------------------------------------------

def plot_metric_bars(
    agg: pd.DataFrame,
    metric: str,
    ylabel: str,
    title_suffix: str,
    out_path: Path,
    dpi: int,
) -> None:
    splits = [s for s in ["stratified", "grouped"] if s in agg["split_type"].values]

    fig, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 5), squeeze=False)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    for ax, split in zip(axes[0], splits):
        sub = agg[agg["split_type"] == split].copy()

        fs_order = [f for f in FEATURE_SET_ORDER if f in sub["feature_set"].values]
        mod_order = [m for m in MODEL_ORDER if m in sub["model_name"].values]

        n_fs = len(fs_order)
        n_mod = len(mod_order)
        bar_width = 0.8 / n_mod
        x = np.arange(n_fs)

        for i, model in enumerate(mod_order):
            means = []
            stds = []
            for fs in fs_order:
                row = sub[(sub["feature_set"] == fs) & (sub["model_name"] == model)]
                means.append(row[mean_col].values[0] if len(row) else 0)
                stds.append(row[std_col].values[0] if len(row) else 0)

            offset = (i - (n_mod - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                means,
                bar_width,
                yerr=stds,
                capsize=3,
                ecolor="#555555",
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_SET_LABELS[f] for f in fs_order], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(SPLIT_LABELS[split], fontsize=11)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")
        ax.set_ylim(0, 1.0)

    fig.suptitle(title_suffix, fontsize=13, fontweight="bold", color="#08306B")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 3  –  ROC curves (one-vs-rest from single-run predictions)
# ---------------------------------------------------------------------------

_DATE_PREFIX_RE = re.compile(r"^run_\d{4}-\d{2}-\d{2}_")


def _parse_run_stem(stem: str, split_type: str) -> tuple[str, str] | None:
    """Extract (feature_set, model_name) from a run filename stem.

    Expected format: run_YYYY-MM-DD_{feature_set}_{model}_{split_type}
    """
    if not stem.endswith(f"_{split_type}"):
        return None
    # Strip date prefix
    body = _DATE_PREFIX_RE.sub("", stem)
    if body == stem:
        return None
    # Strip split suffix
    body = body[: -(len(split_type) + 1)]  # remove _splittype

    # Try each known feature set (longest first) then check model
    for fs in sorted(FEATURE_SET_ORDER, key=len, reverse=True):
        if body.startswith(fs + "_"):
            model = body[len(fs) + 1 :]
            if model in MODEL_ORDER:
                return fs, model
    return None


def _find_prediction_files(split_type: str) -> dict[tuple[str, str], Path]:
    """Return {(feature_set, model_name): path} for a given split type."""
    files: dict[tuple[str, str], Path] = {}
    for p in sorted(PREDICTIONS.glob("*.csv")):
        result = _parse_run_stem(p.stem, split_type)
        if result is not None:
            files[result] = p
    return files


def plot_roc_curves(split_type: str, out_dir: Path, dpi: int) -> None:
    pred_files = _find_prediction_files(split_type)
    if not pred_files:
        print(f"  No prediction files for split={split_type}, skipping ROC curves")
        return

    feature_sets_avail = sorted({fs for fs, _ in pred_files})

    for fs in feature_sets_avail:
        models_avail = [m for m in MODEL_ORDER if (fs, m) in pred_files]
        if not models_avail:
            continue

        # Read one file to discover class names
        sample = pd.read_csv(pred_files[(fs, models_avail[0])])
        prob_cols = [c for c in sample.columns if c.startswith("prob_")]
        classes = [c.replace("prob_", "") for c in prob_cols]
        n_classes = len(classes)

        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5), squeeze=False)

        for model in models_avail:
            df = pd.read_csv(pred_files[(fs, model)])
            y_true = df["true_label"].values
            y_prob = df[prob_cols].values
            y_bin = label_binarize(y_true, classes=classes)

            for ci, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, ci], y_prob[:, ci])
                roc_auc = auc(fpr, tpr)
                axes[0][ci].plot(
                    fpr, tpr,
                    color=MODEL_COLORS[model],
                    label=f"{MODEL_LABELS[model]} (AUC={roc_auc:.3f})",
                    linewidth=2,
                )

        for ci, cls in enumerate(classes):
            ax = axes[0][ci]
            ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", linewidth=1)
            ax.set_xlabel("False Positive Rate", fontsize=10)
            ax.set_ylabel("True Positive Rate", fontsize=10)
            ax.set_title(f"OvR: {cls}", fontsize=10)
            ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

        fig.suptitle(
            f"ROC Curves — {FEATURE_SET_LABELS.get(fs, fs)}, {SPLIT_LABELS[split_type]}",
            fontsize=12,
            fontweight="bold",
            color="#08306B",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"roc_{fs}_{split_type}.png"
        fig.savefig(out_dir / fname, dpi=dpi)
        plt.close(fig)
        print(f"  Saved {(out_dir / fname).relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 4  –  Feature importance plots
# ---------------------------------------------------------------------------

def plot_feature_importance(split_type: str, out_dir: Path, dpi: int, top_n: int = 15) -> None:
    imp_files: dict[tuple[str, str], Path] = {}
    for p in sorted(FEATURE_IMPORTANCE.glob("*.csv")):
        result = _parse_run_stem(p.stem, split_type)
        if result is not None:
            imp_files[result] = p

    if not imp_files:
        print(f"  No feature importance files for split={split_type}, skipping")
        return

    feature_sets_avail = sorted({fs for fs, _ in imp_files})

    for fs in feature_sets_avail:
        models_avail = [m for m in MODEL_ORDER if (fs, m) in imp_files]
        if not models_avail:
            continue

        n_models = len(models_avail)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), squeeze=False)

        for i, model in enumerate(models_avail):
            df = pd.read_csv(imp_files[(fs, model)])
            df = df.sort_values("importance", ascending=False).head(top_n)
            df = df.sort_values("importance", ascending=True)  # flip for horizontal bar

            ax = axes[0][i]
            # Gradient from light to dark blue based on importance rank
            n = len(df)
            bar_colors = [plt.cm.Blues(0.3 + 0.6 * j / max(n - 1, 1)) for j in range(n)]
            ax.barh(df["feature"], df["importance"], color=bar_colors, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Importance", fontsize=10)
            ax.set_title(MODEL_LABELS[model], fontsize=10)
            ax.tick_params(axis="y", labelsize=8)

        fig.suptitle(
            f"Feature Importance (Top {top_n}) — {FEATURE_SET_LABELS.get(fs, fs)}, {SPLIT_LABELS[split_type]}",
            fontsize=12,
            fontweight="bold",
            color="#08306B",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fname = f"feature_importance_{fs}_{split_type}.png"
        fig.savefig(out_dir / fname, dpi=dpi)
        plt.close(fig)
        print(f"  Saved {(out_dir / fname).relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 5  –  LaTeX tables
# ---------------------------------------------------------------------------

def _fmt_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def generate_latex_tables(agg: pd.DataFrame, out_dir: Path, n_seeds: int) -> None:
    """Generate LaTeX .tex files ready for \\input{} in a report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 1: Top 5 configurations by mean macro F1 ---
    top = agg.sort_values("f1_macro_mean", ascending=False).head(5)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{Top 5 configurations by mean macro F1 across {n_seeds} seeds.}}",
        r"  \label{tab:top5_f1}",
        r"  \begin{tabular}{llllll}",
        r"    \toprule",
        r"    Split & Feature set & Model & F1\textsubscript{macro} mean $\pm$ std & AUC mean $\pm$ std \\",
        r"    \midrule",
    ]
    for _, row in top.iterrows():
        split = row["split_type"]
        fs = FEATURE_SET_LABELS.get(row["feature_set"], row["feature_set"])
        model = MODEL_LABELS.get(row["model_name"], row["model_name"])
        f1 = _fmt_mean_std(row["f1_macro_mean"], row["f1_macro_std"])
        roc = _fmt_mean_std(row["roc_auc_ovr_macro_mean"], row["roc_auc_ovr_macro_std"])
        lines.append(f"    {split} & {fs} & {model} & {f1} & {roc} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    path = out_dir / "top5_f1.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {path.relative_to(ROOT)}")

    # --- Table 2: Full results per split strategy ---
    for split in ["stratified", "grouped"]:
        sub = agg[agg["split_type"] == split].sort_values("f1_macro_mean", ascending=False)
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            f"  \\caption{{All configurations — {SPLIT_LABELS[split]} ({n_seeds} seeds).}}",
            f"  \\label{{tab:full_{split}}}",
            r"  \small",
            r"  \begin{tabular}{llrrrr}",
            r"    \toprule",
            r"    Feature set & Model & Accuracy & Precision\textsubscript{macro} & F1\textsubscript{macro} & AUC \\",
            r"    \midrule",
        ]
        for _, row in sub.iterrows():
            fs = FEATURE_SET_LABELS.get(row["feature_set"], row["feature_set"])
            model = MODEL_LABELS.get(row["model_name"], row["model_name"])
            acc = _fmt_mean_std(row["accuracy_mean"], row["accuracy_std"])
            prec = _fmt_mean_std(row["precision_macro_mean"], row["precision_macro_std"])
            f1 = _fmt_mean_std(row["f1_macro_mean"], row["f1_macro_std"])
            roc = _fmt_mean_std(row["roc_auc_ovr_macro_mean"], row["roc_auc_ovr_macro_std"])
            lines.append(f"    {fs} & {model} & {acc} & {prec} & {f1} & {roc} \\\\")
        lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
        path = out_dir / f"full_results_{split}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Saved {path.relative_to(ROOT)}")

    # --- Table 3: Per-model summary (best feature set per model) ---
    best_per_model = agg.loc[agg.groupby("model_name")["f1_macro_mean"].idxmax()]
    best_per_model = best_per_model.sort_values("f1_macro_mean", ascending=False)
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{Best configuration per model ({n_seeds} seeds).}}",
        r"  \label{tab:best_per_model}",
        r"  \begin{tabular}{lllll}",
        r"    \toprule",
        r"    Model & Feature set & Split & F1\textsubscript{macro} mean $\pm$ std & AUC mean $\pm$ std \\",
        r"    \midrule",
    ]
    for _, row in best_per_model.iterrows():
        model = MODEL_LABELS.get(row["model_name"], row["model_name"])
        fs = FEATURE_SET_LABELS.get(row["feature_set"], row["feature_set"])
        split = row["split_type"]
        f1 = _fmt_mean_std(row["f1_macro_mean"], row["f1_macro_std"])
        roc = _fmt_mean_std(row["roc_auc_ovr_macro_mean"], row["roc_auc_ovr_macro_std"])
        lines.append(f"    {model} & {fs} & {split} & {f1} & {roc} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    path = out_dir / "best_per_model.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 6  –  Confusion matrices for top-5 configurations
# ---------------------------------------------------------------------------

def plot_top_confusion_matrices(agg: pd.DataFrame, out_dir: Path, dpi: int, top_n: int = 5) -> None:
    top = agg.sort_values("f1_macro_mean", ascending=False).head(top_n)

    # Collect the prediction files we can find
    pred_all: dict[str, dict[tuple[str, str], Path]] = {}
    for split in ["stratified", "grouped"]:
        pred_all[split] = _find_prediction_files(split)

    # Discover class labels from any prediction file
    any_file = next(iter(next(iter(pred_all.values())).values()))
    sample = pd.read_csv(any_file)
    classes = [c.replace("prob_", "") for c in sample.columns if c.startswith("prob_")]

    fig, axes = plt.subplots(1, top_n, figsize=(4.5 * top_n, 4), squeeze=False)

    blue_cmap = plt.cm.Blues

    for i, (_, row) in enumerate(top.iterrows()):
        fs = row["feature_set"]
        model = row["model_name"]
        split = row["split_type"]
        ax = axes[0][i]

        key = (fs, model)
        pred_file = pred_all.get(split, {}).get(key)
        if pred_file is None:
            ax.set_visible(False)
            continue

        df = pd.read_csv(pred_file)
        y_true = df["true_label"].values
        y_pred = df["predicted_label"].values

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, cmap=blue_cmap, colorbar=False, xticks_rotation=30)
        ax.set_title(
            f"{MODEL_LABELS[model]}\n{FEATURE_SET_LABELS.get(fs, fs)}\n({split})",
            fontsize=8,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(
        f"Confusion Matrices — Top {top_n} Configurations by Mean F1",
        fontsize=13,
        fontweight="bold",
        color="#08306B",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = out_dir / "confusion_matrix_top5.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    agg_path = METRICS / "repeated_seed_aggregate_mean_std.csv"
    if not agg_path.exists():
        print(f"ERROR: {agg_path} not found. Run run_experiments_repeated.py first.")
        sys.exit(1)

    agg = pd.read_csv(agg_path)
    n_seeds = int(agg["n_seeds"].iloc[0])

    # --- 1. F1 macro bar charts ---
    print("Generating F1 macro bar charts …")
    plot_metric_bars(
        agg, metric="f1_macro",
        ylabel="F1 Macro",
        title_suffix="F1 Macro (Mean ± Std, 10 Seeds)",
        out_path=FIGURES / "bar_f1_macro.png",
        dpi=args.dpi,
    )

    # --- 2. ROC AUC bar charts ---
    print("Generating ROC AUC bar charts …")
    plot_metric_bars(
        agg, metric="roc_auc_ovr_macro",
        ylabel="ROC AUC (OvR Macro)",
        title_suffix="ROC AUC Macro (Mean ± Std, 10 Seeds)",
        out_path=FIGURES / "bar_roc_auc_macro.png",
        dpi=args.dpi,
    )

    # --- 3. ROC curves (from single-run predictions) ---
    print("Generating ROC curves …")
    for split in ["stratified", "grouped"]:
        plot_roc_curves(split, FIGURES, args.dpi)

    # --- 4. Feature importance ---
    print("Generating feature importance plots …")
    for split in ["stratified", "grouped"]:
        plot_feature_importance(split, FIGURES, args.dpi)

    # --- 5. LaTeX tables ---
    print("Generating LaTeX tables …")
    generate_latex_tables(agg, TABLES, n_seeds)

    # --- 6. Confusion matrices for top 5 ---
    print("Generating top-5 confusion matrices …")
    plot_top_confusion_matrices(agg, FIGURES, args.dpi)

    print(f"\nAll figures saved to {FIGURES.relative_to(ROOT)}/")
    print(f"All tables saved to {TABLES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

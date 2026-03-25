from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    y_true_arr = y_true.to_numpy()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision_macro": float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0)),
    }

    y_true_bin = label_binarize(y_true_arr, classes=class_labels)
    if y_true_bin.shape[1] == y_prob.shape[1]:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
        )
    else:
        metrics["roc_auc_ovr_macro"] = None

    return metrics


def save_confusion_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: str | Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            scores = np.mean(np.abs(coef), axis=0)
        else:
            scores = np.abs(coef)
    else:
        # Naive Bayes and unsupported models: return empty table.
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame({"feature": feature_names, "importance": scores})
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    return frame

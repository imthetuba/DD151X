from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.models.model_factory import build_model


def _sample_weight_for_xgb(y: pd.Series) -> np.ndarray:
    counts = Counter(y)
    n = len(y)
    k = len(counts)
    class_weight = {c: n / (k * cnt) for c, cnt in counts.items()}
    return np.asarray([class_weight[val] for val in y], dtype=float)


def train_model(
    model_name: str,
    model_params: dict[str, Any],
    class_weight_mode: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    resampling: str,
) -> Any:
    estimator = build_model(model_name, model_params, class_weight_mode)

    needs_scaling = model_name in {"logistic_regression", "naive_bayes", "xgboost"}

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)

    X_fit = X_train
    y_fit = y_train

    if resampling == "random_oversample":
        try:
            from imblearn.over_sampling import RandomOverSampler
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is not installed. Install requirements.txt for resampling support"
            ) from exc

        ros = RandomOverSampler(random_state=42)
        X_fit, y_fit = ros.fit_resample(X_train, y_train)

    fit_kwargs = {}
    if model_name == "xgboost":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(pd.Series(y_fit))
        if class_weight_mode == "balanced":
            fit_kwargs["model__sample_weight"] = _sample_weight_for_xgb(pd.Series(y_encoded))
        pipeline.fit(X_fit, y_encoded, **fit_kwargs)
        setattr(pipeline, "_label_encoder", label_encoder)
        return pipeline

    pipeline.fit(X_fit, y_fit, **fit_kwargs)
    return pipeline

from __future__ import annotations

import inspect
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def _filter_kwargs(estimator_cls: Any, params: dict[str, Any]) -> dict[str, Any]:
    valid = set(inspect.signature(estimator_cls.__init__).parameters.keys())
    return {k: v for k, v in params.items() if k in valid}


def _apply_random_state_if_supported(params: dict[str, Any], estimator_cls: Any, random_state: int | None) -> dict[str, Any]:
    if random_state is None:
        return params

    valid = set(inspect.signature(estimator_cls.__init__).parameters.keys())
    if "random_state" in valid:
        params["random_state"] = random_state
    return params


def build_model(
    model_name: str,
    params: dict[str, Any],
    class_weight_mode: str,
    random_state: int | None = None,
):
    if model_name == "logistic_regression":
        cfg = _filter_kwargs(LogisticRegression, dict(params))
        cfg = _apply_random_state_if_supported(cfg, LogisticRegression, random_state)
        if class_weight_mode == "balanced":
            cfg["class_weight"] = "balanced"
        return LogisticRegression(**cfg)

    if model_name == "naive_bayes":
        cfg = _filter_kwargs(GaussianNB, dict(params))
        return GaussianNB(**cfg)

    if model_name == "random_forest":
        cfg = _filter_kwargs(RandomForestClassifier, dict(params))
        cfg = _apply_random_state_if_supported(cfg, RandomForestClassifier, random_state)
        if class_weight_mode == "balanced":
            cfg["class_weight"] = "balanced"
        return RandomForestClassifier(**cfg)

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install dependencies from requirements.txt"
            ) from exc

        cfg = dict(params)
        cfg = _apply_random_state_if_supported(cfg, XGBClassifier, random_state)
        # Use one-vs-rest class weighting via sample weights in training loop if needed.
        return XGBClassifier(**cfg)

    raise ValueError(f"Unsupported model: {model_name}")

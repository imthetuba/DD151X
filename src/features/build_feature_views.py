from __future__ import annotations

from typing import Any

import pandas as pd


TARGET_COLUMN = "risk_class_3"


def build_feature_views(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}

    for set_name, features in feature_sets.items():
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Feature set '{set_name}' has missing columns: {missing}")

        subset = df[features + [TARGET_COLUMN, "organization_number", "period_year"]].copy()
        views[set_name] = {
            "features": features,
            "frame": subset,
        }

    return views

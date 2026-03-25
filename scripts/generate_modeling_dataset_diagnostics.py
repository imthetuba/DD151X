#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "data/modeling_dataset.csv"
    config_path = repo_root / "configs/experiment_config.yaml"
    feature_dict_path = repo_root / "data/feature_dictionary_step1.json"
    output_path = repo_root / "data/modeling_dataset_diagnostics.json"

    df = pd.read_csv(dataset_path)

    required_columns = ["organization_number", "period_year", "risk_class_3"]
    missing_required = [c for c in required_columns if c not in df.columns]

    duplicate_rows = (
        int(df.duplicated(subset=["organization_number", "period_year"]).sum())
        if {"organization_number", "period_year"}.issubset(df.columns)
        else None
    )

    yf_cols = [c for c in df.columns if c.startswith("yf_")]
    yahoo_coverage: dict[str, object] = {}
    if yf_cols:
        has_any_yf = df[yf_cols].notna().any(axis=1)
        yahoo_coverage = {
            "ratio_columns": yf_cols,
            "rows_with_any_yf": int(has_any_yf.sum()),
            "rows_without_yf": int((~has_any_yf).sum()),
            "non_null_rate_per_column": {c: float(df[c].notna().mean()) for c in yf_cols},
        }

    class_counts = (
        {str(k): int(v) for k, v in df["risk_class_3"].value_counts(dropna=False).items()}
        if "risk_class_3" in df.columns
        else {}
    )
    class_proportions = (
        {str(k): float(v) for k, v in df["risk_class_3"].value_counts(normalize=True, dropna=False).items()}
        if "risk_class_3" in df.columns
        else {}
    )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    feature_sets: dict[str, list[str]] = cfg.get("feature_sets", {})

    feature_set_compatibility: dict[str, dict[str, object]] = {}
    missingness_by_feature_set: dict[str, dict[str, float]] = {}

    for set_name, cols in feature_sets.items():
        missing = [c for c in cols if c not in df.columns]
        feature_set_compatibility[set_name] = {
            "feature_count": len(cols),
            "missing_columns": missing,
            "compatible": len(missing) == 0,
        }

        present = [c for c in cols if c in df.columns]
        if present:
            non_null_rates = df[present].notna().mean()
            missingness_by_feature_set[set_name] = {
                "avg_non_null_rate": float(non_null_rates.mean()),
                "min_non_null_rate": float(non_null_rates.min()),
                "max_non_null_rate": float(non_null_rates.max()),
            }
        else:
            missingness_by_feature_set[set_name] = {
                "avg_non_null_rate": 0.0,
                "min_non_null_rate": 0.0,
                "max_non_null_rate": 0.0,
            }

    payload = {
        "dataset_file": "data/modeling_dataset.csv",
        "feature_dictionary_file": "data/feature_dictionary_step1.json",
        "experiment_config_file": "configs/experiment_config.yaml",
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "required_column_check": {
            "required_columns": required_columns,
            "missing_columns": missing_required,
            "passed": len(missing_required) == 0,
        },
        "duplicate_key_check": {
            "key_columns": ["organization_number", "period_year"],
            "duplicate_rows": duplicate_rows,
            "passed": duplicate_rows == 0 if duplicate_rows is not None else False,
        },
        "period_year_range": {
            "min": int(df["period_year"].min()) if "period_year" in df.columns else None,
            "max": int(df["period_year"].max()) if "period_year" in df.columns else None,
            "unique_years": sorted([int(v) for v in df["period_year"].dropna().unique().tolist()])
            if "period_year" in df.columns
            else [],
        },
        "class_balance": {
            "counts": class_counts,
            "proportions": class_proportions,
        },
        "yahoo_ratio_coverage": yahoo_coverage,
        "missingness_by_feature_set": missingness_by_feature_set,
        "step2_input_verification": {
            "dataset_path_in_config": cfg.get("dataset", {}).get("input_csv"),
            "target_column_in_config": cfg.get("dataset", {}).get("target_column"),
            "group_column_in_config": cfg.get("dataset", {}).get("group_column"),
            "feature_set_compatibility": feature_set_compatibility,
            "all_feature_sets_compatible": all(v["compatible"] for v in feature_set_compatibility.values())
            if feature_set_compatibility
            else False,
        },
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

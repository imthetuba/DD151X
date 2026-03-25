#!/usr/bin/env python3
"""Step 1 data pipeline for the DD151X credit-risk project.

Implements:
1) Data quality checks (target coverage, missing critical fields, duplicates)
2) Deterministic 3-class target engineering from siI_RatingNormalized
3) Finalized factor set (financial + ESG) with a feature dictionary
4) Clean modelling dataset export
5) Initial class balance diagnostics and weighting recommendation
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RISK_CLASS_LABELS = {
    "ig_low_risk": "Investment Grade Low Risk",
    "ig_high_risk": "Investment Grade High Risk",
    "high_yield": "High Yield",
}


@dataclass
class Step1Stats:
    rows_input: int
    rows_with_non_null_target: int
    rows_after_required_fields: int
    duplicate_issuer_year_keys: int
    rows_after_dedup: int
    rows_after_target_mapping: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 3-class modelling dataset for DD151X")
    parser.add_argument("--input", default="data/curated_rated_esg_dataset_max_rows.csv")
    parser.add_argument("--output", default="data/modeling_dataset_step1.csv")
    parser.add_argument("--diagnostics", default="data/modeling_dataset_step1_diagnostics.json")
    parser.add_argument("--feature-dictionary", default="data/feature_dictionary_step1.json")
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int_rating(value: Any) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    rounded = int(number)
    if rounded != number:
        return None
    return rounded


def has_text(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def parse_year(row: dict[str, Any]) -> int | None:
    for key in ("To", "to", "From", "from"):
        raw = row.get(key)
        if not has_text(raw):
            continue
        text = str(raw).strip()
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
    return None


def map_rating_to_three_class(normalized_rating: int | None) -> str | None:
    if normalized_rating is None:
        return None
    if normalized_rating in (1, 2):
        return RISK_CLASS_LABELS["ig_low_risk"]
    if normalized_rating == 3:
        return RISK_CLASS_LABELS["ig_high_risk"]
    if normalized_rating in (4, 5, 6):
        return RISK_CLASS_LABELS["high_yield"]
    return None


def missing_required_fields(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    checks = {
        "OrganizationNumber": has_text(row.get("OrganizationNumber")),
        "siI_RatingNormalized": has_text(row.get("siI_RatingNormalized")),
        "period_year": parse_year(row) is not None,
        "BookValueDebt": parse_float(row.get("BookValueDebt")) is not None,
        "BookValueEquity": parse_float(row.get("BookValueEquity")) not in (None, 0.0),
        "TotalGHGEmission": parse_float(row.get("TotalGHGEmission")) is not None,
        "Revenue": parse_float(row.get("Revenue")) not in (None, 0.0),
    }

    for key, ok in checks.items():
        if not ok:
            missing.append(key)

    return missing


def optional_float(row: dict[str, Any], key: str) -> float | None:
    return parse_float(row.get(key))


def optional_int_flag(row: dict[str, Any], key: str) -> int | None:
    value = parse_float(row.get(key))
    if value is None:
        return None
    if value in (0.0, 1.0):
        return int(value)
    return int(value)


def feature_completeness_score(features: dict[str, Any]) -> int:
    score = 0
    for value in features.values():
        if value is not None and str(value).strip() != "":
            score += 1
    return score


def build_feature_row(source: dict[str, Any]) -> dict[str, Any]:
    book_value_debt = optional_float(source, "BookValueDebt")
    book_value_equity = optional_float(source, "BookValueEquity")
    total_ghg = optional_float(source, "TotalGHGEmission")
    revenue = optional_float(source, "Revenue")
    female_board = optional_float(source, "FemaleBoard")
    male_board = optional_float(source, "MaleBoard")

    debt_to_equity = None
    if book_value_debt is not None and book_value_equity not in (None, 0.0):
        debt_to_equity = book_value_debt / book_value_equity

    carbon_intensity = None
    if total_ghg is not None and revenue not in (None, 0.0):
        carbon_intensity = total_ghg / revenue

    female_board_share = None
    if female_board is not None and male_board is not None and (female_board + male_board) > 0:
        female_board_share = female_board / (female_board + male_board)

    return {
        "enterprise_value": optional_float(source, "EnterpriseValue"),
        "revenue": revenue,
        "book_value_equity": book_value_equity,
        "book_value_debt": book_value_debt,
        "debt_to_equity_ratio": debt_to_equity,
        "total_ghg_emission": total_ghg,
        "scope1": optional_float(source, "Scope1"),
        "scope2_location": optional_float(source, "Scope2Location"),
        "carbon_intensity": carbon_intensity,
        "carbon_target": optional_int_flag(source, "CarbonTarget"),
        "high_impact_climate_sector": optional_int_flag(source, "HighImpactClimateSector"),
        "fossil_fuel_sector": optional_int_flag(source, "FossilFuelSector"),
        "report_biodiversity": optional_int_flag(source, "ReportBiodiversity"),
        "negative_affect_biodiversity": optional_int_flag(source, "NegativeAffectBiodiversity"),
        "female_board": female_board,
        "male_board": male_board,
        "female_board_share": female_board_share,
        "exp_controversial_weapons": optional_int_flag(source, "ExpControversialWeapons"),
        "exp_controversial_products": optional_int_flag(source, "ExpControversialProducts"),
        "exp_debt_collection_or_loans": optional_int_flag(source, "ExpDebtCollectionOrLoans"),
        "is_listed": optional_int_flag(source, "IsListed"),
        "is_consolidated_corp_account": optional_int_flag(source, "IsConsolidatedCorpAccount"),
    }


def finalize_output_row(source: dict[str, Any]) -> dict[str, Any] | None:
    rating_normalized = parse_int_rating(source.get("siI_RatingNormalized"))
    risk_class_3 = map_rating_to_three_class(rating_normalized)
    if risk_class_3 is None:
        return None

    period_year = parse_year(source)
    if period_year is None:
        return None

    features = build_feature_row(source)

    base = {
        "organization_number": str(source.get("OrganizationNumber") or "").strip(),
        "issuer_name": source.get("Name"),
        "period_from": source.get("From"),
        "period_to": source.get("To"),
        "period_year": period_year,
        "rating_agency": source.get("siI_RatingCompany"),
        "rating_symbol": source.get("siI_Rating"),
        "rating_normalized": rating_normalized,
        "risk_class_3": risk_class_3,
    }

    base.update(features)
    return base


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_dictionary() -> dict[str, dict[str, str]]:
    return {
        "organization_number": {
            "type": "identifier",
            "description": "Issuer organization number",
            "source": "OrganizationNumber",
        },
        "period_year": {
            "type": "time",
            "description": "Observation year extracted from period end/start",
            "source": "To/From",
        },
        "rating_normalized": {
            "type": "target_input",
            "description": "Normalized credit rating (numeric scale)",
            "source": "siI_RatingNormalized",
        },
        "risk_class_3": {
            "type": "target",
            "description": "Three-class target: IG low risk, IG high risk, high yield",
            "source": "Engineered from rating_normalized",
        },
        "debt_to_equity_ratio": {
            "type": "financial_derived",
            "description": "BookValueDebt / BookValueEquity",
            "source": "BookValueDebt, BookValueEquity",
        },
        "carbon_intensity": {
            "type": "esg_derived",
            "description": "TotalGHGEmission / Revenue",
            "source": "TotalGHGEmission, Revenue",
        },
        "enterprise_value": {
            "type": "financial",
            "description": "Enterprise value",
            "source": "EnterpriseValue",
        },
        "revenue": {
            "type": "financial",
            "description": "Revenue",
            "source": "Revenue",
        },
        "book_value_equity": {
            "type": "financial",
            "description": "Book value equity",
            "source": "BookValueEquity",
        },
        "book_value_debt": {
            "type": "financial",
            "description": "Book value debt",
            "source": "BookValueDebt",
        },
        "total_ghg_emission": {
            "type": "esg",
            "description": "Total GHG emissions",
            "source": "TotalGHGEmission",
        },
        "scope1": {
            "type": "esg",
            "description": "Scope 1 emissions",
            "source": "Scope1",
        },
        "scope2_location": {
            "type": "esg",
            "description": "Scope 2 emissions (location-based)",
            "source": "Scope2Location",
        },
        "carbon_target": {
            "type": "esg_flag",
            "description": "Company has carbon target",
            "source": "CarbonTarget",
        },
        "high_impact_climate_sector": {
            "type": "esg_flag",
            "description": "Issuer in high impact climate sector",
            "source": "HighImpactClimateSector",
        },
        "fossil_fuel_sector": {
            "type": "esg_flag",
            "description": "Issuer in fossil fuel sector",
            "source": "FossilFuelSector",
        },
        "report_biodiversity": {
            "type": "esg_flag",
            "description": "Company reports biodiversity information",
            "source": "ReportBiodiversity",
        },
        "negative_affect_biodiversity": {
            "type": "esg_flag",
            "description": "Company negatively affects biodiversity",
            "source": "NegativeAffectBiodiversity",
        },
        "female_board_share": {
            "type": "governance_derived",
            "description": "Female board members / (female + male board members)",
            "source": "FemaleBoard, MaleBoard",
        },
        "female_board": {
            "type": "governance",
            "description": "Count of female board members",
            "source": "FemaleBoard",
        },
        "male_board": {
            "type": "governance",
            "description": "Count of male board members",
            "source": "MaleBoard",
        },
        "exp_controversial_weapons": {
            "type": "esg_flag",
            "description": "Exposure to controversial weapons",
            "source": "ExpControversialWeapons",
        },
        "exp_controversial_products": {
            "type": "esg_flag",
            "description": "Exposure to controversial products",
            "source": "ExpControversialProducts",
        },
        "exp_debt_collection_or_loans": {
            "type": "esg_flag",
            "description": "Exposure to debt collection or loans",
            "source": "ExpDebtCollectionOrLoans",
        },
        "is_listed": {
            "type": "metadata_flag",
            "description": "Issuer is listed",
            "source": "IsListed",
        },
        "is_consolidated_corp_account": {
            "type": "metadata_flag",
            "description": "Consolidated corporate account flag",
            "source": "IsConsolidatedCorpAccount",
        },
    }


def choose_best_duplicate(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    existing_features = {k: v for k, v in existing.items() if k not in ("organization_number", "period_year")}
    candidate_features = {k: v for k, v in candidate.items() if k not in ("organization_number", "period_year")}

    existing_score = feature_completeness_score(existing_features)
    candidate_score = feature_completeness_score(candidate_features)

    if candidate_score > existing_score:
        return candidate

    # Tie-breaker: keep the row with later period end if available.
    if str(candidate.get("period_to") or "") >= str(existing.get("period_to") or ""):
        return candidate
    return existing


def compute_class_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["risk_class_3"] for r in rows)
    total = sum(counts.values())

    proportions = {k: (v / total if total else 0.0) for k, v in counts.items()}
    weights = {
        k: (total / (len(counts) * v) if v > 0 else None)
        for k, v in counts.items()
    }

    recommendation = "No weighting required"
    if counts:
        max_prop = max(proportions.values())
        min_prop = min(proportions.values())
        if max_prop > 0.60 or min_prop < 0.15:
            recommendation = "Class imbalance detected: use class_weight='balanced' and test SMOTE/oversampling"

    return {
        "counts": dict(counts),
        "proportions": proportions,
        "recommended_class_weights": weights,
        "recommendation": recommendation,
    }


def run_pipeline(input_path: Path, output_path: Path, diagnostics_path: Path, feature_dict_path: Path) -> None:
    raw_rows = read_csv_rows(input_path)

    rows_with_target = [r for r in raw_rows if has_text(r.get("siI_RatingNormalized"))]

    required_field_filtered: list[dict[str, Any]] = []
    missing_required_examples: list[dict[str, Any]] = []
    for row in rows_with_target:
        missing = missing_required_fields(row)
        if missing:
            if len(missing_required_examples) < 20:
                missing_required_examples.append(
                    {
                        "OrganizationNumber": row.get("OrganizationNumber"),
                        "To": row.get("To"),
                        "missing": missing,
                    }
                )
            continue
        required_field_filtered.append(row)

    deduped: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_counter = 0
    duplicate_keys: Counter[str] = Counter()

    target_mapped_rows: list[dict[str, Any]] = []
    for row in required_field_filtered:
        finalized = finalize_output_row(row)
        if finalized is None:
            continue

        key = (finalized["organization_number"], finalized["period_year"])
        key_label = f"{key[0]}-{key[1]}"

        if key in deduped:
            duplicate_counter += 1
            duplicate_keys[key_label] += 1
            deduped[key] = choose_best_duplicate(deduped[key], finalized)
        else:
            deduped[key] = finalized

    target_mapped_rows = list(deduped.values())
    target_mapped_rows.sort(key=lambda r: (r["organization_number"], r["period_year"]))

    write_csv(output_path, target_mapped_rows)

    feature_dictionary = build_feature_dictionary()
    feature_dict_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dict_path.write_text(json.dumps(feature_dictionary, indent=2), encoding="utf-8")

    class_diag = compute_class_diagnostics(target_mapped_rows)
    stats = Step1Stats(
        rows_input=len(raw_rows),
        rows_with_non_null_target=len(rows_with_target),
        rows_after_required_fields=len(required_field_filtered),
        duplicate_issuer_year_keys=duplicate_counter,
        rows_after_dedup=len(target_mapped_rows),
        rows_after_target_mapping=len(target_mapped_rows),
    )

    diagnostics = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "feature_dictionary_file": str(feature_dict_path),
        "risk_class_mapping": {
            "1-2": RISK_CLASS_LABELS["ig_low_risk"],
            "3": RISK_CLASS_LABELS["ig_high_risk"],
            "4-6": RISK_CLASS_LABELS["high_yield"],
        },
        "quality_summary": {
            "rows_input": stats.rows_input,
            "rows_with_non_null_target": stats.rows_with_non_null_target,
            "target_non_null_coverage": (
                stats.rows_with_non_null_target / stats.rows_input if stats.rows_input else 0.0
            ),
            "rows_after_required_fields": stats.rows_after_required_fields,
            "duplicate_issuer_year_rows_detected": stats.duplicate_issuer_year_keys,
            "rows_after_dedup_and_mapping": stats.rows_after_target_mapping,
        },
        "missing_required_examples": missing_required_examples,
        "duplicate_issuer_year_examples": duplicate_keys.most_common(20),
        "class_balance": class_diag,
    }

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print("Step 1 dataset build complete")
    print(f"Input rows: {stats.rows_input}")
    print(f"Rows with non-null siI_RatingNormalized: {stats.rows_with_non_null_target}")
    print(f"Rows after required-field quality filter: {stats.rows_after_required_fields}")
    print(f"Duplicate issuer-year rows detected: {stats.duplicate_issuer_year_keys}")
    print(f"Final modelling rows: {stats.rows_after_target_mapping}")
    print(f"Dataset written to: {output_path}")
    print(f"Diagnostics written to: {diagnostics_path}")
    print(f"Feature dictionary written to: {feature_dict_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_path=Path(args.input),
        output_path=Path(args.output),
        diagnostics_path=Path(args.diagnostics),
        feature_dict_path=Path(args.feature_dictionary),
    )


if __name__ == "__main__":
    main()
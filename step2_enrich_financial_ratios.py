#!/usr/bin/env python3
"""Step 2 enrichment: add extra financial ratios from Yahoo Finance.

Workflow:
1) Create/edit a ticker mapping file for listed issuers.
2) Pull yearly statement fields from Yahoo Finance via yfinance.
3) Build extra financial ratios and merge into Step 1 modeling dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich modeling dataset with Yahoo Finance ratios")
    parser.add_argument("--base-dataset", default="data/modeling_dataset_step1.csv")
    parser.add_argument("--ticker-map", default="configs/yahoo_ticker_map.json")
    parser.add_argument("--create-template", action="store_true")
    parser.add_argument("--auto-suggest", action="store_true")
    parser.add_argument("--suggestions-output", default="data/yahoo_ticker_suggestions.json")
    parser.add_argument("--apply-sto-suggestions", action="store_true")
    parser.add_argument("--build-fallback-shortlist", action="store_true")
    parser.add_argument("--fallback-output", default="data/yahoo_ticker_fallback_shortlist.csv")
    parser.add_argument("--fallback-top-k", type=int, default=3)
    parser.add_argument("--apply-fallback-shortlist", action="store_true")
    parser.add_argument("--fallback-input", default="data/yahoo_ticker_fallback_shortlist.csv")
    parser.add_argument("--fallback-exchanges", default="CPH,OSL,HEL,ICE")
    parser.add_argument("--fallback-min-score", type=int, default=0)
    parser.add_argument("--ratios-output", default="data/yahoo_financial_ratios_raw.csv")
    parser.add_argument("--enriched-output", default="data/modeling_dataset.csv")
    parser.add_argument("--request-sleep", type=float, default=0.2)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(as_float):
        return None
    return as_float


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def make_ticker_map_template(base_df: pd.DataFrame, output_path: Path) -> None:
    listed = base_df[base_df["is_listed"] == 1].copy()
    listed = listed[["organization_number", "issuer_name"]].drop_duplicates()
    listed = listed.sort_values(["issuer_name", "organization_number"])

    payload: list[dict[str, Any]] = []
    for _, row in listed.iterrows():
        payload.append(
            {
                "organization_number": str(row["organization_number"]),
                "issuer_name": row["issuer_name"],
                "yahoo_ticker": "",
                "notes": "",
            }
        )

    write_json(output_path, payload)
    print(f"Template written: {output_path} ({len(payload)} listed issuers)")


def suggest_yahoo_tickers(mapping: list[dict[str, Any]], sleep_seconds: float) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    for item in mapping:
        query = str(item.get("issuer_name") or "").strip()
        if not query:
            continue

        if str(item.get("yahoo_ticker") or "").strip():
            continue

        try:
            search_result = yf.Search(query=query, max_results=5)
            quotes = search_result.quotes or []
        except Exception as exc:  # noqa: BLE001
            suggestions.append(
                {
                    "organization_number": str(item.get("organization_number") or ""),
                    "issuer_name": query,
                    "error": str(exc),
                    "candidates": [],
                }
            )
            continue

        candidates: list[dict[str, str]] = []
        for q in quotes:
            symbol = str(q.get("symbol") or "").strip()
            if not symbol:
                continue
            candidates.append(
                {
                    "symbol": symbol,
                    "shortname": str(q.get("shortname") or ""),
                    "exchange": str(q.get("exchange") or ""),
                    "quote_type": str(q.get("quoteType") or ""),
                }
            )

        suggestions.append(
            {
                "organization_number": str(item.get("organization_number") or ""),
                "issuer_name": query,
                "candidates": candidates,
            }
        )
        time.sleep(max(sleep_seconds, 0.0))

    return suggestions


def apply_sto_suggestions_to_map(
    mapping: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_org = {str(s.get("organization_number") or "").strip(): s for s in suggestions}

    stats = {
        "updated_with_sto": 0,
        "kept_existing": 0,
        "without_sto_match": 0,
        "total_rows": len(mapping),
    }

    for row in mapping:
        org = str(row.get("organization_number") or "").strip()
        existing = str(row.get("yahoo_ticker") or "").strip()

        if existing:
            stats["kept_existing"] += 1
            continue

        suggestion = by_org.get(org)
        if not suggestion:
            stats["without_sto_match"] += 1
            continue

        candidates = suggestion.get("candidates") or []
        sto = next(
            (c for c in candidates if str(c.get("exchange") or "").strip().upper() == "STO"),
            None,
        )

        if sto:
            row["yahoo_ticker"] = str(sto.get("symbol") or "").strip()
            note = str(row.get("notes") or "").strip()
            auto_note = "auto-selected from STO suggestion"
            row["notes"] = f"{note}; {auto_note}" if note else auto_note
            stats["updated_with_sto"] += 1
        else:
            stats["without_sto_match"] += 1

    return mapping, stats


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return {tok for tok in normalized.split() if len(tok) >= 2}


def _name_overlap_score(issuer_name: str, shortname: str) -> int:
    a = _tokens(issuer_name)
    b = _tokens(shortname)
    if not a or not b:
        return 0
    shared = len(a.intersection(b))
    return int(100 * shared / max(len(a), 1))


def _exchange_priority(exchange: str) -> int:
    score = {
        # Nordic fallbacks first (after STO)
        "CPH": 100,
        "OSL": 98,
        "HEL": 96,
        "ICE": 94,
        # High-quality European and primary listings
        "EBS": 90,
        "LSE": 86,
        "PAR": 84,
        "AMS": 84,
        "FRA": 80,
        "MUN": 78,
        "DUS": 78,
        "HAM": 76,
        "BER": 76,
        # US primaries for non-European issuers
        "NMS": 82,
        "NYQ": 82,
        # Secondary and OTC venues
        "IOB": 60,
        "CXE": 58,
        "DXE": 58,
        "NEO": 55,
        "BUE": 52,
        "PNK": 35,
    }
    return score.get(exchange.upper(), 40)


def build_fallback_shortlist(
    mapping: list[dict[str, Any]],
    suggestions: list[dict[str, Any]],
    top_k: int,
) -> pd.DataFrame:
    by_org = {str(s.get("organization_number") or "").strip(): s for s in suggestions}
    rows: list[dict[str, Any]] = []

    for row in mapping:
        org = str(row.get("organization_number") or "").strip()
        issuer_name = str(row.get("issuer_name") or "").strip()
        existing = str(row.get("yahoo_ticker") or "").strip()
        if existing:
            continue

        suggestion = by_org.get(org)
        if not suggestion:
            rows.append(
                {
                    "organization_number": org,
                    "issuer_name": issuer_name,
                    "has_candidates": 0,
                    "best_symbol": "",
                    "best_exchange": "",
                    "best_shortname": "",
                    "best_score": 0,
                    "alternatives": "",
                }
            )
            continue

        candidates = suggestion.get("candidates") or []
        scored: list[tuple[int, dict[str, Any]]] = []
        for cand in candidates:
            quote_type = str(cand.get("quote_type") or "").upper().strip()
            if quote_type and quote_type != "EQUITY":
                continue

            exchange = str(cand.get("exchange") or "").strip().upper()
            shortname = str(cand.get("shortname") or "").strip()
            ex_score = _exchange_priority(exchange)
            overlap = _name_overlap_score(issuer_name, shortname)
            total = ex_score + overlap
            scored.append((total, cand))

        if not scored:
            rows.append(
                {
                    "organization_number": org,
                    "issuer_name": issuer_name,
                    "has_candidates": 0,
                    "best_symbol": "",
                    "best_exchange": "",
                    "best_shortname": "",
                    "best_score": 0,
                    "alternatives": "",
                }
            )
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        top = scored[: max(top_k, 1)]
        alternatives = " || ".join(
            [
                f"{str(c.get('symbol') or '').strip()} [{str(c.get('exchange') or '').strip()}]"
                for _, c in top[1:]
            ]
        )

        rows.append(
            {
                "organization_number": org,
                "issuer_name": issuer_name,
                "has_candidates": 1,
                "best_symbol": str(best.get("symbol") or "").strip(),
                "best_exchange": str(best.get("exchange") or "").strip().upper(),
                "best_shortname": str(best.get("shortname") or "").strip(),
                "best_score": int(best_score),
                "alternatives": alternatives,
            }
        )

    fallback_df = pd.DataFrame(rows)
    if fallback_df.empty:
        return fallback_df
    fallback_df = fallback_df.sort_values(["has_candidates", "best_score"], ascending=[False, False])
    return fallback_df


def apply_fallback_shortlist_to_map(
    mapping: list[dict[str, Any]],
    fallback_df: pd.DataFrame,
    allowed_exchanges: set[str],
    min_score: int,
    allow_all_exchanges: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "updated_from_fallback": 0,
        "kept_existing": 0,
        "not_in_allowed_exchange_or_score": 0,
        "missing_fallback_record": 0,
        "total_rows": len(mapping),
    }

    if fallback_df.empty:
        stats["missing_fallback_record"] = len(mapping)
        return mapping, stats

    f = fallback_df.copy()
    f["organization_number"] = f["organization_number"].astype(str).str.strip()
    by_org = f.set_index("organization_number").to_dict(orient="index")

    for row in mapping:
        org = str(row.get("organization_number") or "").strip()
        existing = str(row.get("yahoo_ticker") or "").strip()
        if existing:
            stats["kept_existing"] += 1
            continue

        rec = by_org.get(org)
        if not rec:
            stats["missing_fallback_record"] += 1
            continue

        has_candidates = int(rec.get("has_candidates") or 0)
        best_symbol = str(rec.get("best_symbol") or "").strip()
        best_exchange = str(rec.get("best_exchange") or "").strip().upper()
        best_score = int(rec.get("best_score") or 0)

        if (
            has_candidates == 1
            and best_symbol
            and (allow_all_exchanges or best_exchange in allowed_exchanges)
            and best_score >= min_score
        ):
            row["yahoo_ticker"] = best_symbol
            note = str(row.get("notes") or "").strip()
            auto_note = f"auto-selected from fallback shortlist ({best_exchange}, score={best_score})"
            row["notes"] = f"{note}; {auto_note}" if note else auto_note
            stats["updated_from_fallback"] += 1
        else:
            stats["not_in_allowed_exchange_or_score"] += 1

    return mapping, stats


def latest_col_for_year(frame: pd.DataFrame, year: int) -> Any:
    matching_cols = [c for c in frame.columns if getattr(c, "year", None) == year]
    if not matching_cols:
        return None
    return sorted(matching_cols)[-1]


def get_statement_value(frame: pd.DataFrame, year: int, aliases: list[str]) -> float | None:
    if frame is None or frame.empty:
        return None

    column = latest_col_for_year(frame, year)
    if column is None:
        return None

    for alias in aliases:
        if alias in frame.index:
            return safe_float(frame.at[alias, column])
    return None


def extract_ratios_for_ticker(ticker: str, years: list[int]) -> list[dict[str, Any]]:
    yft = yf.Ticker(ticker)

    try:
        fin = yft.financials
    except Exception:  # noqa: BLE001
        fin = pd.DataFrame()
    try:
        bal = yft.balance_sheet
    except Exception:  # noqa: BLE001
        bal = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for year in years:
        revenue = get_statement_value(fin, year, ["Total Revenue"])
        gross_profit = get_statement_value(fin, year, ["Gross Profit"])
        operating_income = get_statement_value(fin, year, ["Operating Income", "EBIT"])
        net_income = get_statement_value(fin, year, ["Net Income"])
        interest_expense = get_statement_value(fin, year, ["Interest Expense"])

        total_assets = get_statement_value(bal, year, ["Total Assets"])
        total_liabilities = get_statement_value(
            bal,
            year,
            ["Total Liabilities Net Minority Interest", "Total Liabilities"],
        )
        total_debt = get_statement_value(bal, year, ["Total Debt", "Net Debt"])
        stockholders_equity = get_statement_value(
            bal,
            year,
            ["Stockholders Equity", "Total Equity Gross Minority Interest"],
        )
        current_assets = get_statement_value(bal, year, ["Current Assets", "Total Current Assets"])
        current_liabilities = get_statement_value(
            bal,
            year,
            ["Current Liabilities", "Total Current Liabilities"],
        )
        cash = get_statement_value(
            bal,
            year,
            ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
        )
        inventory = get_statement_value(bal, year, ["Inventory"])

        debt_base = total_debt if total_debt is not None else total_liabilities

        rows.append(
            {
                "period_year": int(year),
                "yahoo_ticker": ticker,
                "yf_current_ratio": safe_ratio(current_assets, current_liabilities),
                "yf_quick_ratio": safe_ratio(
                    None if current_assets is None else current_assets - (inventory or 0.0),
                    current_liabilities,
                ),
                "yf_cash_ratio": safe_ratio(cash, current_liabilities),
                "yf_debt_to_assets": safe_ratio(debt_base, total_assets),
                "yf_debt_to_equity": safe_ratio(debt_base, stockholders_equity),
                "yf_equity_ratio": safe_ratio(stockholders_equity, total_assets),
                "yf_interest_coverage": safe_ratio(operating_income, abs(interest_expense) if interest_expense else None),
                "yf_gross_margin": safe_ratio(gross_profit, revenue),
                "yf_operating_margin": safe_ratio(operating_income, revenue),
                "yf_net_margin": safe_ratio(net_income, revenue),
                "yf_roa": safe_ratio(net_income, total_assets),
                "yf_roe": safe_ratio(net_income, stockholders_equity),
                "yf_asset_turnover": safe_ratio(revenue, total_assets),
            }
        )

    return rows


def build_ratio_table(base_df: pd.DataFrame, mapping: list[dict[str, Any]], sleep_seconds: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_org_years = (
        base_df.groupby("organization_number")["period_year"].apply(lambda s: sorted({int(v) for v in s})).to_dict()
    )

    for item in mapping:
        org = str(item.get("organization_number") or "").strip()
        ticker = str(item.get("yahoo_ticker") or "").strip()
        if not org or not ticker:
            continue

        years = by_org_years.get(org, [])
        if not years:
            continue

        try:
            extracted = extract_ratios_for_ticker(ticker=ticker, years=years)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed {org} {ticker}: {exc}")
            extracted = []

        for row in extracted:
            row["organization_number"] = org
            row["issuer_name"] = item.get("issuer_name", "")
            rows.append(row)

        time.sleep(max(sleep_seconds, 0.0))

    if not rows:
        return pd.DataFrame(columns=["organization_number", "period_year"])

    ratios = pd.DataFrame(rows)
    ratios = ratios.drop_duplicates(subset=["organization_number", "period_year"], keep="last")
    return ratios


def main() -> None:
    args = parse_args()

    base_path = Path(args.base_dataset)
    map_path = Path(args.ticker_map)
    suggestions_path = Path(args.suggestions_output)
    ratios_path = Path(args.ratios_output)
    enriched_path = Path(args.enriched_output)

    base_df = pd.read_csv(base_path)
    base_df["organization_number"] = base_df["organization_number"].astype(str).str.strip()
    base_df["period_year"] = base_df["period_year"].astype(int)

    if args.create_template:
        make_ticker_map_template(base_df, map_path)
        return

    if not map_path.exists():
        raise FileNotFoundError(
            f"Ticker map not found: {map_path}. Run with --create-template first and fill yahoo_ticker values."
        )

    mapping = read_json(map_path)
    if not isinstance(mapping, list):
        raise ValueError("Ticker map must be a JSON list of issuer mapping objects")

    if args.auto_suggest:
        suggestions = suggest_yahoo_tickers(mapping, sleep_seconds=args.request_sleep)
        write_json(suggestions_path, suggestions)
        print(f"Suggestions written: {suggestions_path}")
        return

    if args.apply_sto_suggestions:
        if not suggestions_path.exists():
            raise FileNotFoundError(
                f"Suggestions file not found: {suggestions_path}. Run with --auto-suggest first."
            )

        suggestions = read_json(suggestions_path)
        if not isinstance(suggestions, list):
            raise ValueError("Suggestions file must be a JSON list")

        updated_mapping, stats = apply_sto_suggestions_to_map(mapping, suggestions)
        write_json(map_path, updated_mapping)

        print(f"Ticker map updated: {map_path}")
        print(f"updated_with_sto={stats['updated_with_sto']}")
        print(f"kept_existing={stats['kept_existing']}")
        print(f"without_sto_match={stats['without_sto_match']}")
        print(f"total_rows={stats['total_rows']}")
        return

    if args.build_fallback_shortlist:
        if not suggestions_path.exists():
            raise FileNotFoundError(
                f"Suggestions file not found: {suggestions_path}. Run with --auto-suggest first."
            )

        suggestions = read_json(suggestions_path)
        if not isinstance(suggestions, list):
            raise ValueError("Suggestions file must be a JSON list")

        shortlist = build_fallback_shortlist(mapping, suggestions, top_k=args.fallback_top_k)
        fallback_path = Path(args.fallback_output)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        shortlist.to_csv(fallback_path, index=False)

        with_candidates = int(shortlist["has_candidates"].sum()) if not shortlist.empty else 0
        print(f"Fallback shortlist written: {fallback_path}")
        print(f"unresolved_issuers={len(shortlist)}")
        print(f"with_ranked_candidates={with_candidates}")
        return

    if args.apply_fallback_shortlist:
        fallback_input = Path(args.fallback_input)
        if not fallback_input.exists():
            raise FileNotFoundError(
                f"Fallback shortlist not found: {fallback_input}. Run with --build-fallback-shortlist first."
            )

        fallback_df = pd.read_csv(fallback_input)
        raw_fallback_exchanges = str(args.fallback_exchanges).strip()
        allow_all_exchanges = raw_fallback_exchanges.upper() == "ALL"
        allowed_exchanges = (
            set()
            if allow_all_exchanges
            else {ex.strip().upper() for ex in raw_fallback_exchanges.split(",") if ex.strip()}
        )

        updated_mapping, stats = apply_fallback_shortlist_to_map(
            mapping=mapping,
            fallback_df=fallback_df,
            allowed_exchanges=allowed_exchanges,
            min_score=int(args.fallback_min_score),
            allow_all_exchanges=allow_all_exchanges,
        )
        write_json(map_path, updated_mapping)

        print(f"Ticker map updated: {map_path}")
        print(
            "allowed_exchanges=ALL"
            if allow_all_exchanges
            else f"allowed_exchanges={','.join(sorted(allowed_exchanges))}"
        )
        print(f"fallback_min_score={int(args.fallback_min_score)}")
        print(f"updated_from_fallback={stats['updated_from_fallback']}")
        print(f"kept_existing={stats['kept_existing']}")
        print(f"not_in_allowed_exchange_or_score={stats['not_in_allowed_exchange_or_score']}")
        print(f"missing_fallback_record={stats['missing_fallback_record']}")
        print(f"total_rows={stats['total_rows']}")
        return

    ratios = build_ratio_table(base_df=base_df, mapping=mapping, sleep_seconds=args.request_sleep)
    ratios_path.parent.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(ratios_path, index=False)

    enriched = base_df.merge(ratios, on=["organization_number", "period_year"], how="left")
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(enriched_path, index=False)

    ratio_cols = [c for c in enriched.columns if c.startswith("yf_")]
    non_null_any = int(enriched[ratio_cols].notna().any(axis=1).sum()) if ratio_cols else 0

    print(f"Raw ratio table: {ratios_path} ({len(ratios)} org-year rows)")
    print(f"Enriched dataset: {enriched_path} ({len(enriched)} rows)")
    print(f"Rows with at least one Yahoo ratio: {non_null_any}")


if __name__ == "__main__":
    main()

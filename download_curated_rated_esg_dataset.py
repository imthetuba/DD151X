#!/usr/bin/env python3
"""Build a curated ESG + ratings dataset from Stamdata feeds.

Goal:
- Pull fresh data from Stamdata async feeds.
- Keep only issuers that have ratings (target available).
- Select one "best" ESG row per issuer for modeling.
- Export a compact, collaborator-friendly CSV.

Usage:
    python3 download_curated_rated_esg_dataset.py

Environment variables in .env or shell:
    STAMDATA_API_URL
    STAMDATA_API_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests


DEFAULT_API_URL = "https://api.stamdata.com"
DEFAULT_ESG_ENDPOINT = "/api/v1/feed/esg-common-units"
DEFAULT_RATINGS_ENDPOINT = "/api/v1.0/feed/issuer-ratings"

# Keep this list opinionated and short: these are generally useful and available.
DEFAULT_ESG_FEATURE_COLUMNS = [
    "CarbonTarget",
    "EnterpriseValue",
    "Revenue",
    "BookValue",
    "BookValueEquity",
    "BookValueDebt",
    "IsListed",
    "IsConsolidatedCorpAccount",
    "HighImpactClimateSector",
    "FossilFuelSector",
    "ExpControversialWeapons",
    "ExpControversialProducts",
    "ExpDebtCollectionOrLoans",
    "FemaleBoard",
    "MaleBoard",
    "ReportBiodiversity",
    "NegativeAffectBiodiversity",
    "TotalGHGEmission",
    "Scope1",
    "Scope2Location",
]


class StamdataError(RuntimeError):
    """Raised when Stamdata API returns an error response."""


@dataclass
class JobResult:
    job_id: str
    feed_urls: list[str]
    status_payload: dict[str, Any]


@dataclass
class RunStats:
    ratings_rows_raw: int
    esg_rows_raw: int
    rated_orgs: int
    rated_orgs_with_esg: int
    matched_rows_before_target_filter: int
    final_rows: int


def read_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def pick_value(name: str, env_file_values: dict[str, str], default: str | None = None) -> str | None:
    return os.getenv(name) or env_file_values.get(name) or default


def safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}


def raise_for_api_error(response: requests.Response) -> None:
    if response.ok:
        return

    payload = safe_json(response)
    detail = payload.get("detail") if isinstance(payload, dict) else None
    title = payload.get("title") if isinstance(payload, dict) else None
    message = detail or title or str(payload)
    raise StamdataError(f"HTTP {response.status_code}: {message}")


def extract_job_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    for candidate in (payload, payload.get("Data"), payload.get("data")):
        if not isinstance(candidate, dict):
            continue
        for key in ("JobId", "jobId", "ID", "id", "Id"):
            value = candidate.get(key)
            if value:
                return str(value)

    return None


def extract_filename(content_disposition: str) -> str | None:
    if not content_disposition:
        return None

    for part in content_disposition.split(";"):
        token = part.strip()
        if token.lower().startswith("filename*="):
            value = token.split("=", 1)[1].strip().strip('"')
            if "''" in value:
                value = value.split("''", 1)[1]
            return Path(unquote(value)).name or None

    for part in content_disposition.split(";"):
        token = part.strip()
        if token.lower().startswith("filename="):
            value = token.split("=", 1)[1].strip().strip('"')
            return Path(value).name or None

    return None


def parse_date(value: Any) -> datetime:
    if value is None:
        return datetime.min

    text = str(value).strip()
    if not text:
        return datetime.min

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return datetime.min


def extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("Data", "data", "Items", "items", "Results", "results"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [x for x in candidate if isinstance(x, dict)]
        return [payload]

    return []


def pick_org_number(record: dict[str, Any]) -> str | None:
    for key in (
        "OrganizationNumber",
        "organizationNumber",
        "IssuerOrganizationNumber",
        "issuerOrganizationNumber",
    ):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def non_empty_count(row: dict[str, Any], columns: list[str]) -> int:
    count = 0
    for col in columns:
        value = row.get(col)
        if value is not None and str(value).strip() != "":
            count += 1
    return count


class StamdataClient:
    def __init__(self, api_url: str, api_key: str, timeout: int = 60) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "dd151x-curated-esg-ratings/1.0",
            }
        )

    def absolute(self, endpoint_or_url: str) -> str:
        if endpoint_or_url.startswith("http://") or endpoint_or_url.startswith("https://"):
            return endpoint_or_url
        return f"{self.api_url}/{endpoint_or_url.lstrip('/')}"

    def create_feed_job(self, endpoint: str, payload: dict[str, Any] | None = None) -> str:
        max_attempts = 20
        wait_seconds = 10

        for attempt in range(1, max_attempts + 1):
            response = self.session.post(self.absolute(endpoint), json=payload or {}, timeout=self.timeout)
            if response.ok:
                job_id = extract_job_id(response.json())
                if job_id:
                    return job_id
                raise StamdataError(f"Could not parse job id from response: {response.text}")

            payload_json = safe_json(response)
            detail = payload_json.get("detail") if isinstance(payload_json, dict) else ""
            if (
                response.status_code == 409
                and isinstance(detail, str)
                and "running job of same type" in detail.lower()
                and attempt < max_attempts
            ):
                print(
                    f"Feed {endpoint} already has a running job (attempt {attempt}/{max_attempts}); "
                    f"waiting {wait_seconds}s before retry..."
                )
                time.sleep(wait_seconds)
                continue

            raise_for_api_error(response)

        raise StamdataError(f"Could not create feed job for {endpoint} after {max_attempts} attempts")

    def poll_job(self, job_id: str, poll_interval: int = 10, timeout_seconds: int = 1800) -> JobResult:
        status_url = self.absolute(f"/api/v1/feed/{job_id}")
        started = time.time()

        while True:
            response = self.session.get(status_url, timeout=self.timeout)
            raise_for_api_error(response)
            payload = response.json()

            status = str(payload.get("Status", payload.get("status", ""))).lower()
            if status == "processed":
                feed_urls = (
                    payload.get("FeedUrls")
                    or payload.get("feedUrls")
                    or payload.get("FeedURLs")
                    or payload.get("feedURLs")
                    or []
                )
                if not isinstance(feed_urls, list):
                    raise StamdataError(f"Unexpected feed URL structure: {type(feed_urls)}")
                return JobResult(job_id=job_id, feed_urls=[str(u) for u in feed_urls], status_payload=payload)

            if status in {"failed", "error"}:
                raise StamdataError(f"Job {job_id} failed: {payload}")

            if time.time() - started > timeout_seconds:
                raise TimeoutError(f"Polling timed out for job {job_id} after {timeout_seconds}s")

            time.sleep(max(10, poll_interval))

    def download_feed_files(self, result: JobResult, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for idx, feed_url in enumerate(result.feed_urls, start=1):
            response = self.session.get(self.absolute(feed_url), timeout=self.timeout)
            raise_for_api_error(response)

            filename = extract_filename(response.headers.get("Content-Disposition", ""))
            if not filename:
                filename = f"{result.job_id}_{idx}.json"

            file_path = out_dir / filename
            file_path.write_bytes(response.content)
            written.append(file_path)

        return written


def load_records_from_files(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(extract_records(payload))
    return rows


def build_latest_ratings_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[datetime, dict[str, Any]]] = {}

    for row in rows:
        org = pick_org_number(row)
        if not org:
            continue

        date = parse_date(row.get("RatingDate") or row.get("Date") or row.get("To") or row.get("to"))
        current = latest.get(org)
        if current is None or date >= current[0]:
            latest[org] = (date, row)

    mapped: dict[str, dict[str, Any]] = {}
    for org, (_, row) in latest.items():
        mapped[org] = {
            "siI_Rating": row.get("LongTermIDR") or row.get("ShortTermIDR") or row.get("Rating"),
            "siI_RatingNormalized": row.get("LongTerm_CQS_SII")
            or row.get("ShortTerm_CQS_SII")
            or row.get("LongTerm_CQS_CRDIV")
            or row.get("ShortTerm_CQS_CRDIV"),
            "siI_RatingCompany": row.get("RatingCompany") or row.get("ratingCompany"),
            "ratingDate": row.get("RatingDate") or row.get("Date"),
        }

    return mapped


def choose_best_esg_row_per_org(
    esg_rows: list[dict[str, Any]],
    rated_orgs: set[str],
    feature_columns: list[str],
) -> dict[str, dict[str, Any]]:
    best: dict[str, tuple[tuple[int, datetime], dict[str, Any]]] = {}

    for row in esg_rows:
        org = pick_org_number(row)
        if not org or org not in rated_orgs:
            continue

        score = non_empty_count(row, feature_columns)
        date = parse_date(row.get("To") or row.get("to") or row.get("From") or row.get("from"))
        rank = (score, date)

        current = best.get(org)
        if current is None or rank >= current[0]:
            best[org] = (rank, row)

    return {org: row for org, (_, row) in best.items()}


def build_curated_rows(
    ratings_map: dict[str, dict[str, Any]],
    best_esg_by_org: dict[str, dict[str, Any]],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for org, rating in ratings_map.items():
        esg = best_esg_by_org.get(org)
        if not esg:
            continue

        out: dict[str, Any] = {
            "OrganizationNumber": org,
            "Name": esg.get("Name") or esg.get("name"),
            "From": esg.get("From") or esg.get("from"),
            "To": esg.get("To") or esg.get("to"),
            "siI_Rating": rating.get("siI_Rating"),
            "siI_RatingNormalized": rating.get("siI_RatingNormalized"),
            "siI_RatingCompany": rating.get("siI_RatingCompany"),
            "ratingDate": rating.get("ratingDate"),
        }

        for col in feature_columns:
            out[col] = esg.get(col)

        rows.append(out)

    return rows


def build_curated_rows_max_coverage(
    ratings_map: dict[str, dict[str, Any]],
    esg_rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Keep every ESG row that can be matched to a rated issuer.

    This maximizes row-level coverage (for example, multiple years per issuer).
    """
    rows: list[dict[str, Any]] = []

    for esg in esg_rows:
        org = pick_org_number(esg)
        if not org:
            continue

        rating = ratings_map.get(org)
        if not rating:
            continue

        out: dict[str, Any] = {
            "OrganizationNumber": org,
            "Name": esg.get("Name") or esg.get("name"),
            "From": esg.get("From") or esg.get("from"),
            "To": esg.get("To") or esg.get("to"),
            "siI_Rating": rating.get("siI_Rating"),
            "siI_RatingNormalized": rating.get("siI_RatingNormalized"),
            "siI_RatingCompany": rating.get("siI_RatingCompany"),
            "ratingDate": rating.get("ratingDate"),
        }

        for col in feature_columns:
            out[col] = esg.get(col)

        rows.append(out)

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    client: StamdataClient,
    esg_endpoint: str,
    ratings_endpoint: str,
    run_dir: Path,
    output_csv: Path,
    feature_columns: list[str],
    coverage_mode: str,
    require_normalized_target: bool,
    poll_interval: int,
    poll_timeout: int,
) -> RunStats:
    print(f"Creating ESG feed job: {esg_endpoint}")
    esg_job_id = client.create_feed_job(esg_endpoint, payload={})
    print(f"ESG job id: {esg_job_id}")

    print(f"Creating ratings feed job: {ratings_endpoint}")
    ratings_job_id = client.create_feed_job(ratings_endpoint, payload={})
    print(f"Ratings job id: {ratings_job_id}")

    print("Polling ESG job status...")
    esg_job = client.poll_job(esg_job_id, poll_interval=poll_interval, timeout_seconds=poll_timeout)
    print(f"ESG job processed with {len(esg_job.feed_urls)} file(s)")

    print("Polling ratings job status...")
    ratings_job = client.poll_job(ratings_job_id, poll_interval=poll_interval, timeout_seconds=poll_timeout)
    print(f"Ratings job processed with {len(ratings_job.feed_urls)} file(s)")

    esg_files = client.download_feed_files(esg_job, run_dir / "downloads" / "esg")
    ratings_files = client.download_feed_files(ratings_job, run_dir / "downloads" / "ratings")

    esg_rows = load_records_from_files(esg_files)
    ratings_rows = load_records_from_files(ratings_files)

    ratings_map = build_latest_ratings_map(ratings_rows)
    rated_orgs = set(ratings_map.keys())

    best_esg_by_org = choose_best_esg_row_per_org(esg_rows, rated_orgs, feature_columns)
    if coverage_mode == "max_rows":
        curated_rows = build_curated_rows_max_coverage(ratings_map, esg_rows, feature_columns)
    else:
        curated_rows = build_curated_rows(ratings_map, best_esg_by_org, feature_columns)

    matched_rows_before_target_filter = len(curated_rows)

    if require_normalized_target:
        curated_rows = [r for r in curated_rows if str(r.get("siI_RatingNormalized") or "").strip() != ""]
    else:
        curated_rows = [
            r
            for r in curated_rows
            if str(r.get("siI_Rating") or "").strip() != ""
            or str(r.get("siI_RatingNormalized") or "").strip() != ""
            or str(r.get("siI_RatingCompany") or "").strip() != ""
        ]

    # Sort for deterministic output.
    curated_rows.sort(
        key=lambda r: (
            str(r.get("OrganizationNumber") or ""),
            str(r.get("To") or ""),
        )
    )

    write_csv(curated_rows, output_csv)

    return RunStats(
        ratings_rows_raw=len(ratings_rows),
        esg_rows_raw=len(esg_rows),
        rated_orgs=len(rated_orgs),
        rated_orgs_with_esg=len(best_esg_by_org),
        matched_rows_before_target_filter=matched_rows_before_target_filter,
        final_rows=len(curated_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated rated ESG dataset from Stamdata")
    parser.add_argument("--esg-endpoint", default=DEFAULT_ESG_ENDPOINT)
    parser.add_argument("--ratings-endpoint", default=DEFAULT_RATINGS_ENDPOINT)
    parser.add_argument("--output", default="data/curated_rated_esg_dataset.csv")
    parser.add_argument("--run-dir", default="data/run_latest")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--poll-timeout", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--feature-columns",
        default=",".join(DEFAULT_ESG_FEATURE_COLUMNS),
        help="Comma-separated list of ESG feature columns to keep",
    )
    parser.add_argument(
        "--coverage-mode",
        choices=["max_rows", "max_companies"],
        default="max_rows",
        help="max_rows keeps all matched ESG rows; max_companies keeps one best ESG row per issuer",
    )
    parser.add_argument(
        "--require-normalized-target",
        action="store_true",
        help="Keep only rows where siI_RatingNormalized is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_values = read_env_file(Path(".env"))
    api_key = pick_value("STAMDATA_API_KEY", env_values)
    api_url = pick_value("STAMDATA_API_URL", env_values, DEFAULT_API_URL)

    if not api_key:
        raise SystemExit("Missing STAMDATA_API_KEY in environment or .env file")

    feature_columns = [c.strip() for c in args.feature_columns.split(",") if c.strip()]
    if not feature_columns:
        raise SystemExit("No feature columns configured")

    client = StamdataClient(api_url=api_url or DEFAULT_API_URL, api_key=api_key, timeout=args.timeout)

    run_dir = Path(args.run_dir)
    output_csv = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)

    stats = run_pipeline(
        client=client,
        esg_endpoint=args.esg_endpoint,
        ratings_endpoint=args.ratings_endpoint,
        run_dir=run_dir,
        output_csv=output_csv,
        feature_columns=feature_columns,
        coverage_mode=args.coverage_mode,
        require_normalized_target=args.require_normalized_target,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
    )

    print("\nDone.")
    print(f"Raw ratings rows: {stats.ratings_rows_raw}")
    print(f"Raw ESG rows: {stats.esg_rows_raw}")
    print(f"Rated organizations: {stats.rated_orgs}")
    print(f"Rated organizations with ESG row: {stats.rated_orgs_with_esg}")
    print(f"Matched rows before target filter: {stats.matched_rows_before_target_filter}")
    print(f"Final curated rows (target available): {stats.final_rows}")
    print(f"Output file: {output_csv}")


if __name__ == "__main__":
    main()

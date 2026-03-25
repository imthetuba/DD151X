#!/usr/bin/env python3
"""Download asynchronous Stamdata feeds and build an ESG+ratings dataset.

Usage:
    python download_and_merge_stamdata.py

Environment variables (loaded from shell or .env):
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

import requests


DEFAULT_API_URL = "https://api.stamdata.com"
DEFAULT_ESG_ENDPOINT = "/api/v1/feed/esg-common-units"
DEFAULT_RATINGS_ENDPOINT = "/api/v1.0/feed/issuer-ratings"


class StamdataError(RuntimeError):
    """Raised when the Stamdata API returns an error response."""


@dataclass
class JobResult:
    job_id: str
    feed_urls: list[str]
    status_payload: dict[str, Any]


def _read_env_file(env_path: Path) -> dict[str, str]:
    """Parse key=value lines from .env without external dependencies."""
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


def _pick_value(name: str, env_file_values: dict[str, str], default: str | None = None) -> str | None:
    return os.getenv(name) or env_file_values.get(name) or default


def _safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}


def _raise_for_api_error(response: requests.Response) -> None:
    if response.ok:
        return

    payload = _safe_json(response)
    detail = payload.get("detail") if isinstance(payload, dict) else None
    title = payload.get("title") if isinstance(payload, dict) else None
    message = detail or title or str(payload)
    raise StamdataError(f"HTTP {response.status_code}: {message}")


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
                "User-Agent": "dd151x-stamdata-client/1.0",
            }
        )

    def _absolute(self, endpoint_or_url: str) -> str:
        if endpoint_or_url.startswith("http://") or endpoint_or_url.startswith("https://"):
            return endpoint_or_url
        return f"{self.api_url}/{endpoint_or_url.lstrip('/')}"

    def create_feed_job(self, endpoint: str, payload: dict[str, Any] | None = None) -> str:
        response = self.session.post(self._absolute(endpoint), json=payload or {}, timeout=self.timeout)
        _raise_for_api_error(response)

        data = response.json()

        # Some endpoints return the created job id directly, while others wrap it.
        if isinstance(data, dict):
            for candidate in (data, data.get("Data"), data.get("data")):
                if not isinstance(candidate, dict):
                    continue
                for key in ("JobId", "jobId", "ID", "id", "Id"):
                    job_id = candidate.get(key)
                    if job_id:
                        return str(job_id)

        raise StamdataError(f"Could not locate job id in response: {data}")

    def poll_job(self, job_id: str, poll_interval: int = 10, timeout_seconds: int = 1800) -> JobResult:
        status_url = self._absolute(f"/api/v1/feed/{job_id}")
        started = time.time()

        while True:
            response = self.session.get(status_url, timeout=self.timeout)
            _raise_for_api_error(response)
            payload = response.json()

            status = str(payload.get("Status", payload.get("status", ""))).lower()
            if status == "processed":
                feed_urls = payload.get("FeedUrls") or payload.get("feedUrls") or []
                if not isinstance(feed_urls, list):
                    raise StamdataError(f"Unexpected FeedUrls type in job status: {type(feed_urls)}")
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
            response = self.session.get(self._absolute(feed_url), timeout=self.timeout)
            _raise_for_api_error(response)

            disposition = response.headers.get("Content-Disposition", "")
            filename = _extract_filename(disposition) or f"{result.job_id}_{idx}.json"
            file_path = out_dir / filename
            file_path.write_bytes(response.content)
            written.append(file_path)

        return written


def _extract_filename(content_disposition: str) -> str | None:
    marker = "filename="
    if marker not in content_disposition:
        return None
    filename = content_disposition.split(marker, 1)[1].strip().strip('"')
    return Path(filename).name or None


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("Data", "data", "Items", "items", "Results", "results"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [x for x in candidate if isinstance(x, dict)]
        return [payload]

    return []


def _load_records_from_files(files: list[Path]) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        all_rows.extend(_extract_records(payload))
    return all_rows


def _parse_date(value: Any) -> datetime:
    if not value:
        return datetime.min
    text = str(value)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt)
        except ValueError:
            continue
    return datetime.min


def _pick_org_number(record: dict[str, Any]) -> str | None:
    for key in ("OrganizationNumber", "organizationNumber", "IssuerOrganizationNumber", "issuerOrganizationNumber"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _build_latest_ratings_map(rating_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[datetime, dict[str, Any]]] = {}

    for row in rating_rows:
        org = _pick_org_number(row)
        if not org:
            continue

        row_date = _parse_date(row.get("To") or row.get("to") or row.get("Date") or row.get("date"))
        best = latest.get(org)
        if best is None or row_date >= best[0]:
            latest[org] = (row_date, row)

    result: dict[str, dict[str, Any]] = {}
    for org, (_, row) in latest.items():
        result[org] = {
            "siI_Rating": row.get("siI_Rating") or row.get("SiiRating") or row.get("Rating") or row.get("rating"),
            "siI_RatingNormalized": row.get("siI_RatingNormalized")
            or row.get("SiiRatingNormalized")
            or row.get("ratingNormalized"),
            "siI_RatingCompany": row.get("siI_RatingCompany")
            or row.get("SiiRatingCompany")
            or row.get("ratingCompany"),
        }
    return result


def _merge_rows(esg_rows: list[dict[str, Any]], ratings_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in esg_rows:
        row_copy = dict(row)
        org = _pick_org_number(row_copy)
        rating_info = ratings_map.get(org or "", {})
        row_copy["siI_Rating"] = rating_info.get("siI_Rating")
        row_copy["siI_RatingNormalized"] = rating_info.get("siI_RatingNormalized")
        row_copy["siI_RatingCompany"] = rating_info.get("siI_RatingCompany")
        merged.append(row_copy)
    return merged


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    client: StamdataClient,
    esg_endpoint: str,
    ratings_endpoint: str,
    out_dir: Path,
    output_csv: Path,
    poll_interval: int,
    poll_timeout: int,
) -> None:
    print(f"Creating ESG feed job: {esg_endpoint}")
    esg_job_id = client.create_feed_job(esg_endpoint, payload={})
    print(f"ESG job id: {esg_job_id}")

    print(f"Creating ratings feed job: {ratings_endpoint}")
    ratings_job_id = client.create_feed_job(ratings_endpoint, payload={})
    print(f"Ratings job id: {ratings_job_id}")

    print("Polling ESG job status...")
    esg_result = client.poll_job(esg_job_id, poll_interval=poll_interval, timeout_seconds=poll_timeout)
    print(f"ESG job processed with {len(esg_result.feed_urls)} file(s)")

    print("Polling ratings job status...")
    ratings_result = client.poll_job(ratings_job_id, poll_interval=poll_interval, timeout_seconds=poll_timeout)
    print(f"Ratings job processed with {len(ratings_result.feed_urls)} file(s)")

    esg_files = client.download_feed_files(esg_result, out_dir / "esg")
    ratings_files = client.download_feed_files(ratings_result, out_dir / "ratings")

    esg_rows = _load_records_from_files(esg_files)
    rating_rows = _load_records_from_files(ratings_files)

    ratings_map = _build_latest_ratings_map(rating_rows)
    merged_rows = _merge_rows(esg_rows, ratings_map)
    _write_csv(merged_rows, output_csv)

    matched = sum(1 for row in merged_rows if row.get("siI_Rating") is not None)
    print(f"Wrote {len(merged_rows)} rows to {output_csv}")
    print(f"Rows with rating label: {matched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and merge Stamdata ESG + ratings feeds")
    parser.add_argument("--esg-endpoint", default=DEFAULT_ESG_ENDPOINT)
    parser.add_argument("--ratings-endpoint", default=DEFAULT_RATINGS_ENDPOINT)
    parser.add_argument("--download-dir", default="downloads")
    parser.add_argument("--output", default="companies_with_ratings.csv")
    parser.add_argument("--poll-interval", type=int, default=10)
    parser.add_argument("--poll-timeout", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_values = _read_env_file(Path(".env"))
    api_key = _pick_value("STAMDATA_API_KEY", env_values)
    api_url = _pick_value("STAMDATA_API_URL", env_values, DEFAULT_API_URL)

    if not api_key:
        raise SystemExit("Missing STAMDATA_API_KEY in environment or .env file")

    client = StamdataClient(api_url=api_url or DEFAULT_API_URL, api_key=api_key, timeout=args.timeout)
    build_dataset(
        client=client,
        esg_endpoint=args.esg_endpoint,
        ratings_endpoint=args.ratings_endpoint,
        out_dir=Path(args.download_dir),
        output_csv=Path(args.output),
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
    )


if __name__ == "__main__":
    main()

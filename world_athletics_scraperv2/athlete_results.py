"""
Multithreaded athlete results fetcher with retry and backoff.

- Uses ThreadPoolExecutor (8 workers)
- Adds AthleteName, DOB, Gender, CompetitorURL
- Retries on HTTP 503 and network failures
- Supports Brotli, Zstd, Gzip, Deflate decompression
"""

import csv
import gzip
import json
import os
import re
import time
import zlib
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import brotli
import zstandard as zstd
import pandas as pd
import requests
import urllib3
from tqdm import tqdm

# Disable TLS verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
REQUEST_KWARGS = {"verify": False}

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DB = os.path.join(BASE_DIR, "data", "db_cleaned.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "athlete_results")

MASTER_CSV = os.path.join(OUT_DIR,"data", "master_athlete_results.csv")

# API endpoint + headers
GRAPHQL_URL = "https://graphql-prod-4752.prod.aws.worldathletics.org/graphql"
HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://worldathletics.org",
    "Referer": "https://worldathletics.org/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "x-amz-user-agent": "aws-amplify/3.0.2",
    "x-api-key": "da2-qmxd4dl6zfbihixs5ik7uhwor4",
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ID_RE = re.compile(r"(\d{5,})$")


def extract_id(url: str) -> Optional[str]:
    m = ID_RE.search(url)
    return m.group(1) if m else None


def maybe_decompress(raw: bytes) -> bytes:
    for func in (
        lambda b: brotli.decompress(b),
        lambda b: zstd.ZstdDecompressor().decompress(b),
        lambda b: gzip.decompress(b),
        lambda b: zlib.decompress(b, -zlib.MAX_WBITS),
    ):
        try:
            return func(raw)
        except Exception:
            continue
    return raw


def build_payload(aid: str, year: Optional[int] = None) -> Dict:
    return {
        "operationName": "GetSingleCompetitorResultsDate",
        "query": (
            "query GetSingleCompetitorResultsDate($id: Int, "
            "$resultsByYearOrderBy: String, $resultsByYear: Int) { "
            "getSingleCompetitorResultsDate(id: $id, "
            "resultsByYear: $resultsByYear, "
            "resultsByYearOrderBy: $resultsByYearOrderBy) { "
            "parameters {resultsByYear resultsByYearOrderBy __typename} "
            "activeYears "
            "resultsByDate {date competition venue indoor disciplineCode "
            "disciplineNameUrlSlug typeNameUrlSlug discipline country category "
            "race place mark wind notLegal resultScore remark __typename} "
            "__typename } }"
        ),
        "variables": {"id": int(aid), "resultsByYearOrderBy": "date", "resultsByYear": year},
    }


def fetch_results_threaded(aid: str, meta: Dict[str, str], retries: int = 3, delay: float = 1.0) -> List[Dict]:
    body = build_payload(aid)

    for attempt in range(retries):
        try:
            resp = requests.post(
                GRAPHQL_URL, headers=HEADERS, data=json.dumps(body),
                timeout=30, **REQUEST_KWARGS
            )

            if resp.status_code == 503:
                logger.warning("ID %s – HTTP 503, retrying (%d/%d)", aid, attempt + 1, retries)
                time.sleep(delay * (attempt + 1))
                continue
            elif resp.status_code != 200:
                logger.warning("ID %s – HTTP %s", aid, resp.status_code)
                return []

            raw = resp.content.strip()
            if not raw:
                return []

            try:
                data = resp.json()
                result = data["data"].get("getSingleCompetitorResultsDate")
                if result and "resultsByDate" in result:
                    rows = result["resultsByDate"]
                else:
                    return []
            except Exception:
                try:
                    dec = maybe_decompress(raw)
                    data = json.loads(dec.decode("utf-8"))
                    result = data["data"].get("getSingleCompetitorResultsDate")
                    if result and "resultsByDate" in result:
                        rows = result["resultsByDate"]
                    else:
                        return []
                except Exception:
                    return []

            for r in rows:
                r.update(meta)
            return rows

        except requests.RequestException as exc:
            logger.warning("ID %s – network error (%s), retrying (%d/%d)", aid, exc, attempt + 1, retries)
            time.sleep(delay * (attempt + 1))

    logger.error("ID %s – failed after %d retries", aid, retries)
    return []


def main():
    print("Using DB at:", INPUT_DB)
    if not os.path.exists(INPUT_DB):
        raise FileNotFoundError(INPUT_DB)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_DB)
    df.columns = [c.strip() for c in df.columns]
    required = ("CompetitorURL", "Competitor", "DOB", "Gender")
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{col} column missing in db_cleaned.csv")

    meta: Dict[str, Dict[str, str]] = {}
    for _, row in df[list(required)].dropna().iterrows():
        aid = extract_id(row["CompetitorURL"])
        if aid:
            meta[aid] = {
                "AthleteName": row["Competitor"],
                "DOB": row["DOB"],
                "Gender": row["Gender"],
                "CompetitorURL": row["CompetitorURL"],
            }

    athlete_ids = list(meta.keys())
    logger.info("Fetching results for %d athletes using threads...", len(athlete_ids))

    all_rows = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_results_threaded, aid, meta[aid]): aid
            for aid in athlete_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching results"):
            rows = future.result()
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        logger.warning("No results fetched.")
        return

    # Write output
    fieldnames = list(all_rows[0].keys())
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("✅ Done – master CSV written to %s", MASTER_CSV)


if __name__ == "__main__":
    main()

"""
Scrape championship qualification status via GraphQL.

Tracks qualification progress for:
- Asian Games 2026 (Aichi-Nagoya)
- World Championships
- LA 2028 Olympics

Usage:
    python -m scrapers.scrape_qualifications
    python -m scrapers.scrape_qualifications --comp-id 7156 --country KSA
"""

import asyncio
import argparse
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.wa_client import WAClient

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped"

# Known major championship competition IDs
# These need to be updated when new championships are announced
MAJOR_CHAMPIONSHIPS = {
    # "asian_games_2026": {"id": TBD, "name": "Asian Games 2026 Aichi-Nagoya"},
    # "world_champs_2025": {"id": TBD, "name": "World Championships 2025 Tokyo"},
    # "olympics_2028": {"id": TBD, "name": "Olympics 2028 Los Angeles"},
}


async def scrape_qualifications(
    client: WAClient,
    competition_id: int,
    country: Optional[str] = None,
) -> pd.DataFrame:
    """Scrape qualification status for a championship."""
    print(f"\n{'='*60}")
    print(f"Scraping Qualifications for Competition {competition_id}")
    print(f"{'='*60}")

    # Get qualification data
    data = await client.get_championship_qualifications(
        competition_id=competition_id,
        country=country,
    )

    if not data:
        print("   No qualification data found")
        return pd.DataFrame()

    rows = []
    for qual in (data.get("qualifications") or []):
        event = qual.get("event")
        event_id = qual.get("eventId")
        gender = qual.get("gender")
        entry_standard = qual.get("entryStandard")
        qual_type = qual.get("qualificationType")

        for athlete in (qual.get("qualifiedAthletes") or []):
            rows.append({
                "competition_id": competition_id,
                "event": event,
                "event_id": event_id,
                "gender": gender,
                "entry_standard": entry_standard,
                "qualification_type": qual_type,
                "athlete": athlete.get("athlete"),
                "athlete_id": athlete.get("aaAthleteId"),
                "country": athlete.get("country"),
                "qualifying_mark": athlete.get("mark"),
                "qualifying_venue": athlete.get("venue"),
                "qualifying_date": athlete.get("date"),
                "athlete_qual_type": athlete.get("qualificationType"),
                "scraped_at": datetime.now().isoformat(),
            })

    df = pd.DataFrame(rows)
    print(f"   Found {len(df)} qualified athletes")
    return df


async def scrape_latest_qualified(
    client: WAClient,
    competition_id: int,
    max_pages: int = 10,
) -> pd.DataFrame:
    """Scrape recently qualified athletes for a championship."""
    print(f"\n   Fetching latest qualified athletes...")

    rows = []
    offset = 0
    limit = 50

    for page in range(max_pages):
        data = await client.get_latest_qualified(
            competition_id=competition_id, limit=limit, offset=offset
        )
        if not data:
            break

        competitors = data.get("competitors") or []
        if not competitors:
            break

        for c in competitors:
            rows.append({
                "competition_id": competition_id,
                "athlete": c.get("athlete"),
                "athlete_id": c.get("aaAthleteId"),
                "country": c.get("country"),
                "event": c.get("event"),
                "qualifying_mark": c.get("mark"),
                "qualifying_venue": c.get("venue"),
                "qualifying_date": c.get("date"),
                "qualification_type": c.get("qualificationType"),
            })

        offset += limit
        if len(competitors) < limit:
            break

    df = pd.DataFrame(rows)
    print(f"   Found {len(df)} recently qualified athletes")
    return df


async def main():
    parser = argparse.ArgumentParser(description="Scrape championship qualifications")
    parser.add_argument("--comp-id", type=int, help="Championship competition ID")
    parser.add_argument("--country", type=str, default=None, help="Filter by country (e.g. KSA)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.comp_id and not MAJOR_CHAMPIONSHIPS:
        print("No competition IDs configured. Use --comp-id or update MAJOR_CHAMPIONSHIPS dict.")
        print("\nTo find competition IDs, run:")
        print("  python -m scrapers.scrape_competitions")
        print("  Then look for the championship in calendar.parquet")
        return

    async with WAClient(max_per_second=3.0) as client:
        comp_ids = [args.comp_id] if args.comp_id else [c["id"] for c in MAJOR_CHAMPIONSHIPS.values()]

        all_quals = []
        for comp_id in comp_ids:
            df = await scrape_qualifications(client, comp_id, country=args.country)
            if len(df) > 0:
                all_quals.append(df)

            df_latest = await scrape_latest_qualified(client, comp_id)
            if len(df_latest) > 0:
                all_quals.append(df_latest)

        if all_quals:
            df_combined = pd.concat(all_quals, ignore_index=True)
            df_combined.drop_duplicates(
                subset=["competition_id", "athlete_id", "event"], keep="last", inplace=True
            )
            path = OUTPUT_DIR / "qualifications.parquet"
            df_combined.to_parquet(path, index=False, engine="pyarrow")
            print(f"\n   Saved {len(df_combined)} qualification records to {path}")

    print(f"\n{'='*60}")
    print("QUALIFICATION SCRAPE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

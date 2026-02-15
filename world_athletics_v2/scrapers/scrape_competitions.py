"""
Scrape competition calendar and results via GraphQL.

Features:
- Competition calendar with filters (date range, region, ranking category)
- Full competition results (competition viewer data)
- Upcoming competitions
- Recent results feed

Usage:
    python -m scrapers.scrape_competitions                           # Calendar + upcoming
    python -m scrapers.scrape_competitions --results --comp-id 7156  # Results for specific comp
    python -m scrapers.scrape_competitions --recent                  # Recent results feed
"""

import asyncio
import argparse
import platform
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.wa_client import WAClient

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped"


async def scrape_calendar(
    client: WAClient,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Scrape competition calendar."""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if not end_date:
        end_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"Scraping Competition Calendar ({start_date} to {end_date})")
    print(f"{'='*60}")

    all_rows = []
    offset = 0

    while True:
        cal = await client.get_calendar_events(
            start_date=start_date, end_date=end_date,
            limit=limit, offset=offset
        )
        if not cal:
            break

        comps = cal.get("results", [])
        if not comps:
            break

        for c in comps:
            all_rows.append({
                "competition_id": c.get("id"),
                "name": c.get("name"),
                "venue": c.get("venue"),
                "area": c.get("area"),
                "ranking_category": c.get("rankingCategory"),
                "disciplines": c.get("disciplines"),
                "start_date": c.get("startDate"),
                "end_date": c.get("endDate"),
                "has_results": c.get("hasResults"),
                "has_api_results": c.get("hasApiResults"),
                "has_startlist": c.get("hasStartlist"),
            })

        print(f"   Fetched {len(all_rows)} competitions...")

        if len(comps) < limit:
            break
        offset += limit

    df = pd.DataFrame(all_rows)
    print(f"   Total competitions: {len(df)}")
    return df


async def scrape_competition_results(
    client: WAClient,
    competition_id: int,
) -> Dict[str, pd.DataFrame]:
    """Scrape full results for a single competition (competition viewer)."""
    print(f"\n{'='*60}")
    print(f"Scraping Results for Competition {competition_id}")
    print(f"{'='*60}")

    data = await client.get_competition_results(competition_id)
    if not data:
        print("   No results found")
        return {}

    comp_info = data.get("competition", {})
    print(f"   {comp_info.get('name')} | {comp_info.get('venue')} | {comp_info.get('startDate')}")

    # Parse results into flat rows
    result_rows = []
    for event_title_group in (data.get("eventTitles") or []):
        event_meta = event_title_group.get("eventTitle", {})
        event_name = event_meta.get("name", "")
        event_gender = event_meta.get("gender", "")

        for event in (event_title_group.get("events") or []):
            for phase in (event.get("phases") or []):
                phase_name = phase.get("phase", "")

                for result in (phase.get("results") or []):
                    competitor = result.get("competitor", {}) or {}
                    result_rows.append({
                        "competition_id": competition_id,
                        "competition_name": comp_info.get("name"),
                        "venue": comp_info.get("venue"),
                        "comp_date": comp_info.get("startDate"),
                        "event": event_name,
                        "gender": event_gender,
                        "phase": phase_name,
                        "athlete": competitor.get("name"),
                        "athlete_id": competitor.get("aaAthleteId"),
                        "country": competitor.get("country"),
                        "birth_date": competitor.get("birthDate"),
                        "mark": result.get("mark"),
                        "wind": result.get("wind"),
                        "place": result.get("place"),
                        "points": result.get("points"),
                        "qualified": result.get("qualified"),
                        "remark": result.get("remark"),
                    })

    df_results = pd.DataFrame(result_rows)
    print(f"   Parsed {len(df_results)} individual results")
    return {"results": df_results, "competition": comp_info}


async def scrape_recent_results(client: WAClient, limit: int = 50) -> pd.DataFrame:
    """Scrape recent global results feed."""
    print(f"\n{'='*60}")
    print(f"Scraping Recent Results (limit={limit})")
    print(f"{'='*60}")

    data = await client.get_recent_results(limit=limit)
    if not data:
        return pd.DataFrame()

    rows = []
    for r in (data.get("results") or []):
        event = r.get("event", {}) or {}
        rows.append({
            "competition_id": r.get("id"),
            "iaaf_id": r.get("iaafId"),
            "name": r.get("name"),
            "venue": r.get("venue"),
            "start_date": r.get("startDate"),
            "end_date": r.get("endDate"),
            "event_name": event.get("name"),
            "event_venue": event.get("venue"),
            "country_code": event.get("countryCode"),
            "country_name": event.get("countryName"),
            "area_name": event.get("areaName"),
            "category_name": event.get("categoryName"),
        })

    df = pd.DataFrame(rows)
    print(f"   {len(df)} recent results")
    return df


async def scrape_upcoming(client: WAClient) -> pd.DataFrame:
    """Scrape upcoming competitions."""
    print(f"\n{'='*60}")
    print("Scraping Upcoming Competitions")
    print(f"{'='*60}")

    data = await client.get_upcoming_competitions(
        today=datetime.now().strftime("%Y-%m-%d")
    )
    if not data:
        return pd.DataFrame()

    rows = []
    # Response is a list of groups, each with 'label' and 'competitions'
    groups = data if isinstance(data, list) else [data]
    for group in groups:
        label = group.get("label", "") if isinstance(group, dict) else ""
        comps = group.get("competitions", []) if isinstance(group, dict) else []
        for c in comps:
            rows.append({
                "competition_id": c.get("competitionId"),
                "name": c.get("name"),
                "venue": c.get("venue"),
                "start_date": c.get("startDate"),
                "end_date": c.get("endDate"),
                "date_range": c.get("dateRange"),
                "is_next_event": c.get("isNextEvent"),
                "category": label,
            })

    df = pd.DataFrame(rows)
    print(f"   {len(df)} upcoming competitions")
    return df


async def main():
    parser = argparse.ArgumentParser(description="Scrape competition data")
    parser.add_argument("--results", action="store_true", help="Scrape competition results")
    parser.add_argument("--comp-id", type=int, help="Competition ID for results")
    parser.add_argument("--recent", action="store_true", help="Scrape recent results feed")
    parser.add_argument("--start", type=str, help="Calendar start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Calendar end date (YYYY-MM-DD)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with WAClient(max_per_second=3.0) as client:
        # Calendar
        df_cal = await scrape_calendar(client, start_date=args.start, end_date=args.end)
        if len(df_cal) > 0:
            df_cal.to_parquet(OUTPUT_DIR / "calendar.parquet", index=False, engine="pyarrow")

        # Upcoming
        df_upcoming = await scrape_upcoming(client)
        if len(df_upcoming) > 0:
            df_upcoming.to_parquet(OUTPUT_DIR / "upcoming.parquet", index=False, engine="pyarrow")

        # Competition results
        if args.results and args.comp_id:
            result_data = await scrape_competition_results(client, args.comp_id)
            if "results" in result_data and len(result_data["results"]) > 0:
                path = OUTPUT_DIR / f"comp_results_{args.comp_id}.parquet"
                result_data["results"].to_parquet(path, index=False, engine="pyarrow")

        # Recent results
        if args.recent:
            df_recent = await scrape_recent_results(client)
            if len(df_recent) > 0:
                df_recent.to_parquet(OUTPUT_DIR / "recent_results.parquet", index=False, engine="pyarrow")

    print(f"\n{'='*60}")
    print("COMPETITION SCRAPE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

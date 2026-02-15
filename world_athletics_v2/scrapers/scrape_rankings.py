"""
Scrape world rankings and season toplists for all events via GraphQL.

Usage:
    python -m scrapers.scrape_rankings                    # Rankings + toplists
    python -m scrapers.scrape_rankings --rankings-only    # Rankings only
    python -m scrapers.scrape_rankings --toplists-only    # Toplists only
"""

import asyncio
import argparse
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.wa_client import WAClient
from data.event_utils import EVENT_GROUPS, DISCIPLINE_CODES, EVENT_CATEGORIES, MENS_ONLY, WOMENS_ONLY

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped"


# Events to scrape (all major individual events)
EVENTS_TO_SCRAPE = []
for category, events in EVENT_CATEGORIES.items():
    if category != "Relays":  # Skip relays for now
        EVENTS_TO_SCRAPE.extend(events)


async def scrape_world_rankings(client: WAClient, limit_per_event: int = 200) -> pd.DataFrame:
    """Scrape current world rankings for all events.

    NOTE: getWorldRankings (plural) doesn't support gender parameter.
    It returns whichever gender the API maps to for each eventGroup slug.
    Gender is inferred from the response eventGroup string (e.g. "Women's 100m").
    For complete gender-specific rankings, individual athlete profiles
    contain worldRankings.current with gender info.
    """
    print(f"\n{'='*60}")
    print("Scraping World Rankings")
    print(f"{'='*60}")

    all_rows = []
    today = datetime.now().strftime("%Y-%m-%d")
    seen_event_groups = set()

    for event in EVENTS_TO_SCRAPE:
        event_group = EVENT_GROUPS.get(event)
        if not event_group:
            continue

        # API returns one gender per eventGroup slug - skip duplicates
        if event_group in seen_event_groups:
            continue
        seen_event_groups.add(event_group)

        print(f"   {event}...", end=" ")

        try:
            rankings = await client.get_world_rankings(
                event_group=event_group, limit=limit_per_event
            )
            if rankings and rankings.get("rankings"):
                returned_eg = rankings.get("eventGroup") or ""
                # Infer gender from response eventGroup string
                if "Women" in returned_eg or "women" in returned_eg:
                    gender = "F"
                elif "Men" in returned_eg or "men" in returned_eg:
                    gender = "M"
                else:
                    gender = "U"  # Unknown

                for r in rankings["rankings"]:
                    all_rows.append({
                        "event": event,
                        "event_group": returned_eg or event_group,
                        "gender": gender,
                        "rank": r.get("place"),
                        "athlete": r.get("competitorName"),
                        "athlete_id": r.get("id"),
                        "athlete_slug": r.get("competitorUrlSlug"),
                        "country": r.get("countryCode"),
                        "ranking_score": r.get("rankingScore"),
                        "country_place": r.get("countryPlace"),
                        "previous_place": r.get("previousPlace"),
                        "rank_date": rankings.get("rankDate") or today,
                    })
                print(f"{gender} {len(rankings['rankings'])} athletes")
            else:
                print("no data")
        except Exception as e:
            print(f"error: {e}")

    df = pd.DataFrame(all_rows)
    print(f"\n   Total ranking entries: {len(df)}")
    return df


async def scrape_top_rankings(client: WAClient) -> pd.DataFrame:
    """Scrape #1 ranked athletes per event for BOTH genders.

    This is the only authorized query that returns men's ranking data
    from the API. Returns the top-ranked athlete per event for all events.
    Use this to get men's event leaders + event URL slugs.
    """
    print(f"\n{'='*60}")
    print("Scraping Top Rankings (Men + Women #1 per event)")
    print(f"{'='*60}")

    try:
        data = await client.get_top_rankings(limit=50)
    except Exception as e:
        print(f"   Error: {e}")
        return pd.DataFrame()

    if not data or not data.get("rankings"):
        print("   No data returned")
        return pd.DataFrame()

    rows = []
    today = datetime.now().strftime("%Y-%m-%d")
    for r in data["rankings"]:
        sex_code = r.get("sexCode", "")
        gender = "M" if sex_code in ("M", "male") else "F" if sex_code in ("W", "F", "female") else "U"
        rows.append({
            "event": r.get("eventName"),
            "event_slug": r.get("eventUrlSlug"),
            "gender": gender,
            "rank": 1,
            "athlete": r.get("competitorName"),
            "athlete_id": r.get("competitorId"),
            "ranking_calc_id": r.get("rankingCalculationId"),
            "athlete_slug": None,
            "country": r.get("countryCode"),
            "ranking_score": r.get("score"),
            "rank_date": today,
            "source": "getTopRankings",
        })

    df = pd.DataFrame(rows)
    men = df[df["gender"] == "M"]
    women = df[df["gender"] == "F"]
    print(f"   {len(men)} men's events, {len(women)} women's events")
    return df


def extract_rankings_from_profiles(profiles: List[Dict], source_label: str = "profile") -> pd.DataFrame:
    """Extract world ranking data from athlete profiles.

    Each athlete profile has worldRankings.current with their rank and score
    per event. This gives us men's rankings for all KSA athletes + rivals.
    """
    import re

    rows = []
    today = datetime.now().strftime("%Y-%m-%d")

    for deep in profiles:
        p = deep.get("profile", {}) or {}
        basic = p.get("basicData", {}) or {}
        rankings = (p.get("worldRankings", {}) or {}).get("current", []) or []

        # Get name using new API fields
        name = basic.get("friendlyName") or ""
        if not name:
            given = basic.get("givenName") or basic.get("firstName") or ""
            family = basic.get("familyName") or basic.get("lastName") or ""
            name = f"{given} {family}".strip()

        country = basic.get("countryCode")
        is_male = basic.get("male")  # Boolean from profile

        for r in rankings:
            event_group = r.get("eventGroup", "")
            # Infer gender from event name or profile
            if "Men" in event_group:
                gender = "M"
            elif "Women" in event_group:
                gender = "F"
            elif is_male is True:
                gender = "M"
            elif is_male is False:
                gender = "F"
            else:
                gender = "U"

            # Skip "Overall Ranking" entries
            if "overall" in event_group.lower():
                continue

            rows.append({
                "event": event_group,
                "event_slug": re.sub(r"^(Men's|Women's)\s+", "", event_group).lower().replace(" ", "-"),
                "gender": gender,
                "rank": r.get("place"),
                "athlete": name or None,
                "athlete_id": deep.get("id"),
                "ranking_calc_id": None,
                "athlete_slug": basic.get("urlSlug"),
                "country": country,
                "ranking_score": r.get("rankingScore"),
                "rank_date": today,
                "source": source_label,
            })

    return pd.DataFrame(rows)


async def scrape_season_toplists(
    client: WAClient,
    year: int = None,
    limit_per_event: int = 100,
) -> pd.DataFrame:
    """Scrape season toplists for all events, both genders.

    NOTE: getTopList may not be authorized with current API key.
    If auth fails on first call, the function returns an empty DataFrame.

    API response fields:
    - payload[].result, achiever, nationality, venue, date, resultScore
    """
    if year is None:
        year = datetime.now().year

    print(f"\n{'='*60}")
    print(f"Scraping {year} Season Toplists")
    print(f"{'='*60}")

    all_rows = []
    auth_failed = False

    for event in EVENTS_TO_SCRAPE:
        if auth_failed:
            break

        disc_code = DISCIPLINE_CODES.get(event)
        if not disc_code:
            continue

        # Determine genders for this event
        if event in MENS_ONLY:
            genders = ["M"]
        elif event in WOMENS_ONLY:
            genders = ["F"]
        else:
            genders = ["M", "F"]

        for sex in genders:
            if auth_failed:
                break

            print(f"   {event} {sex}...", end=" ")

            try:
                toplist = await client.get_top_list(
                    discipline_code=disc_code, gender=sex, year=year,
                    limit=limit_per_event
                )
                entries = None
                if toplist:
                    entries = toplist.get("payload") or toplist.get("toplists")
                if entries:
                    for t in entries:
                        all_rows.append({
                            "event": event,
                            "discipline_code": disc_code,
                            "gender": sex,
                            "season": year,
                            "place": t.get("place") or t.get("position"),
                            "mark": t.get("result") or t.get("mark"),
                            "competitor": t.get("achiever") or t.get("competitor"),
                            "country": t.get("nationality") or t.get("nat"),
                            "venue": t.get("venue"),
                            "date": t.get("date"),
                            "result_score": t.get("resultScore"),
                        })
                    print(f"{len(entries)} marks")
                else:
                    print("no data")
            except Exception as e:
                err_msg = str(e)
                if "Not Authorized" in err_msg:
                    print(f"NOT AUTHORIZED - skipping all toplists")
                    auth_failed = True
                else:
                    print(f"error: {e}")

    if auth_failed:
        print("\n   WARNING: getTopList is not authorized with current API key.")
        print("   Season toplists will be empty. Try again later or update API key.")

    df = pd.DataFrame(all_rows)
    print(f"\n   Total toplist entries: {len(df)}")
    return df


async def main():
    parser = argparse.ArgumentParser(description="Scrape world rankings and toplists")
    parser.add_argument("--rankings-only", action="store_true")
    parser.add_argument("--toplists-only", action="store_true")
    parser.add_argument("--year", type=int, default=None, help="Season year for toplists")
    parser.add_argument("--limit", type=int, default=200, help="Max athletes per event")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with WAClient(max_per_second=3.0) as client:
        if not args.toplists_only:
            df_rankings = await scrape_world_rankings(client, limit_per_event=args.limit)
            if len(df_rankings) > 0:
                path = OUTPUT_DIR / "world_rankings.parquet"
                df_rankings.to_parquet(path, index=False, engine="pyarrow")
                print(f"   Saved to {path}")

        if not args.rankings_only:
            df_toplists = await scrape_season_toplists(
                client, year=args.year, limit_per_event=args.limit
            )
            if len(df_toplists) > 0:
                path = OUTPUT_DIR / "season_toplists.parquet"
                df_toplists.to_parquet(path, index=False, engine="pyarrow")
                print(f"   Saved to {path}")

    print(f"\n{'='*60}")
    print("RANKINGS SCRAPE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

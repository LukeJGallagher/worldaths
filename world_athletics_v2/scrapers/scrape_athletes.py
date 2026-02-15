"""
Scrape deep KSA athlete profiles + top rivals per event via GraphQL.

This is the initial deep scrape. After first run, daily_sync.yml handles updates.

Usage:
    python -m scrapers.scrape_athletes              # KSA athletes
    python -m scrapers.scrape_athletes --rivals      # + top 20 rivals per event
    python -m scrapers.scrape_athletes --country QAT # Other country
"""

import asyncio
import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from api.wa_client import WAClient
from data.event_utils import ASIAN_COUNTRY_CODES, get_discipline_code, DISCIPLINE_CODES

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped"


def _get_full_name(basic: Dict) -> Optional[str]:
    """Extract best available name from basicData.

    New API schema (AthleteNewData) has familyName/givenName/friendlyName
    instead of firstName/lastName/fullName (which now return null).
    """
    # Try new fields first (friendlyName is "GIVEN FAMILY" format)
    if basic.get("friendlyName"):
        return basic["friendlyName"]
    # Build from givenName + familyName
    given = basic.get("givenName") or ""
    family = basic.get("familyName") or ""
    if given or family:
        return f"{given} {family}".strip()
    # Fall back to old fields
    if basic.get("fullName"):
        return basic["fullName"]
    first = basic.get("firstName") or ""
    last = basic.get("lastName") or ""
    if first or last:
        return f"{first} {last}".strip()
    return None


async def scrape_ksa_athletes(client: WAClient, country_code: str = "KSA") -> List[Dict]:
    """Fetch all athletes for a country with full profiles."""
    print(f"\n{'='*60}")
    print(f"Scraping {country_code} Athletes - Deep Profiles")
    print(f"{'='*60}")

    # Step 1: Get athlete list
    print(f"\n1. Finding all {country_code} athletes...")
    athletes = await client.search_all_athletes_for_country(country_code)
    print(f"   Found {len(athletes)} athletes")

    # Step 2: Fetch deep profiles
    print(f"\n2. Fetching deep profiles...")
    profiles = []
    for i, athlete in enumerate(athletes):
        aid = athlete.get("aaAthleteId")
        if not aid:
            continue

        url_slug = athlete.get("urlSlug") or ""
        try:
            deep = await client.get_deep_athlete_data(int(aid), url_slug=url_slug)
            if deep and deep.get("profile"):
                profiles.append(deep)
                basic = deep["profile"].get("basicData", {}) or {}
                name = _get_full_name(basic) or "Unknown"
                pbs_count = len((deep["profile"].get("personalBests", {}) or {}).get("results", []) or [])
                print(f"   [{i+1}/{len(athletes)}] {name} ({pbs_count} PBs)")
        except Exception as e:
            print(f"   [{i+1}/{len(athletes)}] Error for {aid}: {e}")

    print(f"\n   Successfully scraped {len(profiles)} complete profiles")
    return profiles


async def scrape_rivals(
    client: WAClient,
    ksa_profiles: List[Dict],
    top_n: int = 30,
) -> List[Dict]:
    """Scrape top N rivals for each event KSA athletes compete in."""
    print(f"\n{'='*60}")
    print(f"Scraping Top {top_n} Rivals Per Event")
    print(f"{'='*60}")

    import re

    def _extract_athlete_id(url_slug: str) -> Optional[str]:
        """Extract athlete ID from ranking URL slug like 'name-name-14756997'."""
        if not url_slug:
            return None
        m = re.search(r'(\d+)$', url_slug)
        return m.group(1) if m else None

    # Find all events KSA athletes compete in
    # Profile eventGroup values have gender prefix e.g. "Men's 100m"
    # Strip it for the API which expects just "100m"
    ksa_events = {}  # slug -> gender
    for profile in ksa_profiles:
        p = profile.get("profile", {}) or {}
        rankings = (p.get("worldRankings", {}) or {}).get("current", []) or []

        for r in rankings:
            event_group = r.get("eventGroup")
            if not event_group:
                continue
            # Strip gender prefix: "Men's 100m" -> "100m", "Women's High Jump" -> "High Jump"
            slug = re.sub(r"^(Men's|Women's)\s+", "", event_group)
            # Skip "Overall Ranking" - not a real event
            if "overall" in slug.lower():
                continue
            # Convert display-style to API slug format
            slug_lower = slug.lower().replace(" ", "-")
            gender = "M" if event_group.startswith("Men") else "F"
            ksa_events[slug_lower] = gender

    print(f"\n   KSA competes in {len(ksa_events)} events: {', '.join(sorted(ksa_events))}")

    # For each event, get top N from Asian region + top N global
    rival_profiles = []
    seen_ids = set()

    for event_slug in sorted(ksa_events):
        gender = ksa_events[event_slug]
        print(f"\n   Event: {event_slug} ({gender})")

        try:
            # Asian top list
            rankings = await client.get_world_rankings(
                event_group=event_slug,
                region_type="area", region="asia",
                limit=top_n
            )
            if rankings:
                for r in (rankings.get("rankings") or []):
                    # Extract real athlete ID from URL slug (ranking 'id' is entry ID)
                    aid = _extract_athlete_id(r.get("competitorUrlSlug"))
                    if aid and aid not in seen_ids:
                        seen_ids.add(aid)
                        rival_profiles.append({
                            "id": aid,
                            "url_slug": r.get("competitorUrlSlug"),
                            "name": r.get("competitorName"),
                            "country": r.get("countryCode"),
                            "event": event_slug,
                            "rank": r.get("place"),
                            "score": r.get("rankingScore"),
                            "region": "asia",
                            "gender": gender,
                        })
        except Exception as e:
            print(f"   Asian rankings error: {str(e)[:60]}")

        try:
            # Global top list
            rankings_global = await client.get_world_rankings(
                event_group=event_slug, limit=top_n
            )
            if rankings_global:
                for r in (rankings_global.get("rankings") or []):
                    aid = _extract_athlete_id(r.get("competitorUrlSlug"))
                    if aid and aid not in seen_ids:
                        seen_ids.add(aid)
                        rival_profiles.append({
                            "id": aid,
                            "url_slug": r.get("competitorUrlSlug"),
                            "name": r.get("competitorName"),
                            "country": r.get("countryCode"),
                            "event": event_slug,
                            "rank": r.get("place"),
                            "score": r.get("rankingScore"),
                            "region": "global",
                            "gender": gender,
                        })
        except Exception as e:
            print(f"   Global rankings error: {str(e)[:60]}")

        print(f"   Total unique rivals so far: {len(rival_profiles)}")

    print(f"\n   Total rival profiles to fetch: {len(rival_profiles)}")

    # Fetch basic profiles for rivals (not deep - just profile + PBs)
    print(f"\n3. Fetching rival profiles...")
    enriched_rivals = []
    for i, rival in enumerate(rival_profiles):
        try:
            profile = await client.get_athlete_profile(int(rival["id"]))
            if profile:
                rival["profile"] = profile
                enriched_rivals.append(rival)
                if (i + 1) % 50 == 0:
                    print(f"   [{i+1}/{len(rival_profiles)}] fetched...")
        except Exception as e:
            pass  # Skip failed rivals

    print(f"\n   Successfully enriched {len(enriched_rivals)} rival profiles")
    return enriched_rivals


def profiles_to_dataframe(profiles: List[Dict]) -> pd.DataFrame:
    """Convert deep athlete profiles to a flat DataFrame."""
    rows = []
    for deep in profiles:
        p = deep.get("profile", {}) or {}
        basic = p.get("basicData", {}) or {}
        # primaryMedia is now a LIST in the new schema
        media_raw = p.get("primaryMedia")
        if isinstance(media_raw, list) and media_raw:
            media = media_raw[0]
        elif isinstance(media_raw, dict):
            media = media_raw
        else:
            media = {}
        pbs = (p.get("personalBests", {}) or {}).get("results", []) or []
        rankings_current = (p.get("worldRankings", {}) or {}).get("current", []) or []

        # Find best ranking
        best_rank = None
        best_score = 0
        best_event = None
        for r in rankings_current:
            if r and (r.get("rankingScore") or 0) > best_score:
                best_rank = r.get("place")
                best_score = r.get("rankingScore", 0)
                best_event = r.get("eventGroup")

        # Count honours from profile - honours is now a LIST of categories,
        # each with categoryName + results list
        honours_data = p.get("honours")
        all_honour_results = []
        if isinstance(honours_data, list):
            for cat in honours_data:
                if isinstance(cat, dict) and cat.get("results"):
                    all_honour_results.extend(cat["results"])
                elif isinstance(cat, dict) and cat.get("place"):
                    # Old format: flat list of results
                    all_honour_results.append(cat)
        elif isinstance(honours_data, dict):
            all_honour_results = honours_data.get("results", []) or []
        golds = sum(1 for h in all_honour_results if h and str(h.get("place", "")).strip() == "1")
        silvers = sum(1 for h in all_honour_results if h and str(h.get("place", "")).strip() == "2")
        bronzes = sum(1 for h in all_honour_results if h and str(h.get("place", "")).strip() == "3")

        full_name = _get_full_name(basic)

        rows.append({
            "athlete_id": deep.get("id"),
            "iaaf_id": basic.get("iaafId"),
            "full_name": full_name,
            "first_name": basic.get("givenName") or basic.get("firstName"),
            "last_name": basic.get("familyName") or basic.get("lastName"),
            "gender": (
                basic.get("sexName", "").lower()
                if basic.get("sexName")
                else ("M" if basic.get("male") else "F" if basic.get("male") is False else None)
            ),
            "country_code": basic.get("countryCode"),
            "country_name": basic.get("countryName"),
            "birth_date": basic.get("birthDate"),
            "birth_place": basic.get("birthPlace"),
            "photo_url": media.get("fileNameUrl") if media else None,
            "primary_event": best_event,
            "best_world_rank": best_rank,
            "best_ranking_score": best_score if best_score > 0 else None,
            "pb_count": len(pbs),
            "gold_medals": golds,
            "silver_medals": silvers,
            "bronze_medals": bronzes,
            "total_medals": golds + silvers + bronzes,
            "scraped_at": datetime.now().isoformat(),
        })

    return pd.DataFrame(rows)


def pbs_to_dataframe(profiles: List[Dict]) -> pd.DataFrame:
    """Extract all personal bests into a flat DataFrame."""
    rows = []
    for deep in profiles:
        aid = deep.get("id")
        p = deep.get("profile", {}) or {}
        basic = p.get("basicData", {}) or {}
        pbs = (p.get("personalBests", {}) or {}).get("results", []) or []

        full_name = _get_full_name(basic)

        for pb in pbs:
            rows.append({
                "athlete_id": aid,
                "full_name": full_name,
                "country_code": basic.get("countryCode"),
                "discipline": pb.get("discipline"),
                "mark": pb.get("mark"),
                "result_score": pb.get("resultScore"),
                "wind": pb.get("wind"),
                "venue": pb.get("venue"),
                "date": pb.get("date"),
                "indoor": pb.get("indoor", False),
            })

    return pd.DataFrame(rows)


def results_to_dataframe(profiles: List[Dict]) -> pd.DataFrame:
    """Extract all competition results into a flat DataFrame."""
    rows = []
    for deep in profiles:
        aid = deep.get("id")
        results_data = deep.get("results_by_discipline")
        if not results_data:
            continue

        basic = deep.get("profile", {}).get("basicData", {}) or {}
        full_name = _get_full_name(basic)

        for event_group in (results_data.get("resultsByEvent") or []):
            discipline = event_group.get("discipline")
            for result in (event_group.get("results") or []):
                rows.append({
                    "athlete_id": aid,
                    "full_name": full_name,
                    "country_code": basic.get("countryCode"),
                    "discipline": discipline,
                    "indoor": event_group.get("indoor", False),
                    "date": result.get("date"),
                    "competition": result.get("competition"),
                    "venue": result.get("venue"),
                    "country": result.get("country"),
                    "category": result.get("category"),
                    "round": result.get("race"),
                    "place": result.get("place"),
                    "mark": result.get("mark"),
                    "wind": result.get("wind"),
                    "result_score": result.get("resultScore"),
                    "not_legal": result.get("notLegal", False),
                })

    return pd.DataFrame(rows)


def save_parquet(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame to parquet in the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"   Saved {len(df)} rows to {path}")
    return path


async def main():
    parser = argparse.ArgumentParser(description="Scrape World Athletics athlete data")
    parser.add_argument("--country", default="KSA", help="Country code (default: KSA)")
    parser.add_argument("--rivals", action="store_true", help="Also scrape top rivals per event")
    parser.add_argument("--rivals-top", type=int, default=20, help="Top N rivals per event (default: 20)")
    args = parser.parse_args()

    async with WAClient(max_per_second=3.0) as client:
        # Scrape KSA athletes
        profiles = await scrape_ksa_athletes(client, args.country)

        if not profiles:
            print("No profiles found. Exiting.")
            return

        # Save athlete profiles
        print(f"\n{'='*60}")
        print("Saving to Parquet...")
        print(f"{'='*60}")

        df_athletes = profiles_to_dataframe(profiles)
        save_parquet(df_athletes, f"{args.country.lower()}_athletes.parquet")

        df_pbs = pbs_to_dataframe(profiles)
        save_parquet(df_pbs, f"{args.country.lower()}_personal_bests.parquet")

        df_results = results_to_dataframe(profiles)
        save_parquet(df_results, f"{args.country.lower()}_results.parquet")

        # Save raw JSON for deep data
        json_path = OUTPUT_DIR / f"{args.country.lower()}_deep_profiles.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2, default=str)
        print(f"   Saved raw JSON to {json_path}")

        # Optionally scrape rivals
        if args.rivals:
            rivals = await scrape_rivals(client, profiles, top_n=args.rivals_top)

            if rivals:
                # Flatten rival data
                rival_rows = []
                for r in rivals:
                    rp = r.get("profile", {}) or {}
                    basic = rp.get("basicData", {}) or {}
                    rival_name = _get_full_name(basic)
                    rival_rows.append({
                        "athlete_id": r["id"],
                        "full_name": rival_name or r.get("name"),
                        "country_code": basic.get("countryCode") or r.get("country"),
                        "event": r.get("event"),
                        "world_rank": r.get("rank"),
                        "ranking_score": r.get("score"),
                        "region": r.get("region"),
                        "gender": r.get("gender"),
                        "is_asian": (basic.get("countryCode") or r.get("country", "")) in ASIAN_COUNTRY_CODES,
                    })

                df_rivals = pd.DataFrame(rival_rows)
                save_parquet(df_rivals, "rivals.parquet")

        print(f"\n{'='*60}")
        print("SCRAPE COMPLETE")
        print(f"{'='*60}")
        print(f"Athletes: {len(df_athletes)}")
        print(f"Personal Bests: {len(df_pbs)}")
        print(f"Competition Results: {len(df_results)}")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

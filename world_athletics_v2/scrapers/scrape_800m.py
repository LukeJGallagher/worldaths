"""
Scrape 800m race data: Diamond League results + expanded championship data.

Uses the World Athletics GraphQL API to collect:
1. Diamond League 800m results (2019-2025) — times, positions, athletes (no splits)
2. Merged with existing championship results for the 800m Race Analysis page

Output:
- data/scraped/800m_diamond_league_results.parquet  (DL results)
- data/scraped/800m_all_results.parquet             (DL + Championships merged)

Usage:
    python -m scrapers.scrape_800m                   # Scrape Diamond League results
    python -m scrapers.scrape_800m --dry-run         # Show calendar hits only
    python -m scrapers.scrape_800m --year 2024       # Single season
"""

import asyncio
import argparse
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.wa_client import WAClient

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped"

# Diamond League competition group ID (from WA calendar API)
# If this changes, use --find-group to discover the current ID
DL_COMPETITION_GROUP_ID = 1  # Wanda Diamond League


async def find_diamond_league_competitions(
    client: WAClient,
    start_year: int = 2019,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Find all Diamond League competitions in the calendar."""
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    print(f"\nSearching for Diamond League competitions ({start_date} to {end_date})...")

    all_comps = []
    offset = 0
    limit = 200

    while True:
        cal = await client.get_calendar_events(
            start_date=start_date,
            end_date=end_date,
            query="Diamond League",
            limit=limit,
            offset=offset,
            hide_competitions_with_no_results=True,
        )
        if not cal:
            break

        comps = cal.get("results", [])
        if not comps:
            break

        for c in comps:
            has_results = c.get("hasApiResults") or c.get("hasResults")
            all_comps.append({
                "competition_id": c.get("id"),
                "name": c.get("name"),
                "venue": c.get("venue"),
                "area": c.get("area"),
                "start_date": c.get("startDate"),
                "end_date": c.get("endDate"),
                "ranking_category": c.get("rankingCategory"),
                "has_api_results": c.get("hasApiResults"),
                "has_results": has_results,
            })

        if len(comps) < limit:
            break
        offset += limit

    df = pd.DataFrame(all_comps)
    if not df.empty:
        # Filter to actual Diamond League meetings (ranking_category or name match)
        mask = (
            df["name"].str.contains("Diamond League", case=False, na=False)
            | df["ranking_category"].str.contains("Diamond League", case=False, na=False)
        )
        df = df[mask].copy()

    print(f"   Found {len(df)} Diamond League competitions with results")
    return df


async def scrape_800m_from_competition(
    client: WAClient,
    competition_id: int,
    comp_name: str = "",
) -> List[dict]:
    """Extract 800m results from a single competition."""
    data = await client.get_competition_results(competition_id)
    if not data:
        return []

    comp_info = data.get("competition", {})
    rows = []

    for event_title_group in (data.get("eventTitles") or []):
        event_meta = event_title_group.get("eventTitle", {})
        event_name = (event_meta.get("name") or "").strip()

        # Match 800m events only
        if "800" not in event_name:
            continue

        event_gender = event_meta.get("gender", "")
        # Normalize gender
        sex = "M" if event_gender and event_gender.upper() in ("M", "MALE", "MEN") else "W"

        for event in (event_title_group.get("events") or []):
            for phase in (event.get("phases") or []):
                phase_name = phase.get("phase", "")

                for result in (phase.get("results") or []):
                    competitor = result.get("competitor", {}) or {}
                    mark = result.get("mark", "")
                    place = result.get("place")

                    # Skip DNS/DNF/DQ entries with no mark
                    remark = result.get("remark", "")

                    rows.append({
                        "competition_id": competition_id,
                        "competition_name": comp_info.get("name") or comp_name,
                        "venue": comp_info.get("venue", ""),
                        "date": comp_info.get("startDate", ""),
                        "event": event_name,
                        "sex": sex,
                        "phase": phase_name,
                        "name": competitor.get("name", ""),
                        "athlete_id": competitor.get("aaAthleteId"),
                        "country": competitor.get("country", ""),
                        "mark": mark,
                        "rank": place,
                        "wind": result.get("wind", ""),
                        "remark": remark,
                        "competition_type": "Diamond League",
                    })

    return rows


async def scrape_diamond_league_800m(
    client: WAClient,
    start_year: int = 2019,
    end_year: int = 2025,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Scrape all 800m results from Diamond League meetings."""

    # Find DL competitions
    df_comps = await find_diamond_league_competitions(client, start_year, end_year)

    if df_comps.empty:
        print("   No Diamond League competitions found.")
        return pd.DataFrame()

    # Filter to those with API results
    df_with_results = df_comps[df_comps["has_api_results"] == True].copy()
    print(f"   {len(df_with_results)} competitions have API results available")

    if dry_run:
        print("\n   DRY RUN — competitions found:")
        for _, row in df_with_results.iterrows():
            print(f"   [{row['competition_id']}] {row['name']} | {row['venue']} | {row['start_date']}")
        return pd.DataFrame()

    # Scrape 800m results from each competition
    all_rows = []
    total = len(df_with_results)
    for i, (_, comp) in enumerate(df_with_results.iterrows()):
        comp_id = comp["competition_id"]
        comp_name = comp["name"]
        print(f"   [{i+1}/{total}] {comp_name} ({comp_id})...", end="")

        try:
            rows = await scrape_800m_from_competition(client, comp_id, comp_name)
            all_rows.extend(rows)
            count = len(rows)
            print(f" {count} results" if count > 0 else " no 800m")
        except Exception as e:
            print(f" ERROR: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        # Drop entries without a time
        df = df[df["mark"].str.strip().ne("") & df["mark"].notna()].copy()
        df = df.drop_duplicates(subset=["competition_id", "sex", "phase", "name"], keep="first")

    print(f"\n   Total 800m DL results: {len(df)}")
    return df


def build_merged_results(
    df_dl: pd.DataFrame,
    championship_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Merge Diamond League results with championship results."""
    if championship_path is None:
        championship_path = OUTPUT_DIR / "800m_championship_results.parquet"

    frames = []

    # Load championship data
    if championship_path.exists():
        df_champ = pd.read_parquet(championship_path)
        # Add competition_type if missing
        if "competition_type" not in df_champ.columns:
            df_champ["competition_type"] = df_champ["championship"].apply(
                lambda c: "Olympic Games" if "OG" in str(c)
                else "World Championships"
            )
        frames.append(df_champ)
        print(f"   Championship data: {len(df_champ)} results")

    # Add Diamond League data
    if not df_dl.empty:
        # Align columns with championship schema
        dl_aligned = df_dl.rename(columns={
            "competition_name": "championship",
        })
        # Keep only columns that exist in championship data
        common_cols = ["championship", "phase", "sex", "rank", "name", "country",
                       "mark", "competition_type"]
        for col in common_cols:
            if col not in dl_aligned.columns:
                dl_aligned[col] = None
        dl_aligned = dl_aligned[common_cols].copy()
        frames.append(dl_aligned)
        print(f"   Diamond League data: {len(dl_aligned)} results")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    print(f"   Merged total: {len(merged)} results")
    return merged


async def main():
    parser = argparse.ArgumentParser(description="Scrape 800m Diamond League data")
    parser.add_argument("--dry-run", action="store_true", help="Show calendar hits only")
    parser.add_argument("--year", type=int, help="Single season to scrape")
    parser.add_argument("--start-year", type=int, default=2019, help="Start year (default 2019)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default 2025)")
    parser.add_argument("--no-merge", action="store_true", help="Skip merging with championship data")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_year = args.year or args.start_year
    end_year = args.year or args.end_year

    print(f"\n{'='*60}")
    print(f"800m Diamond League Scraper")
    print(f"Seasons: {start_year}-{end_year}")
    print(f"{'='*60}")

    async with WAClient(max_per_second=2.0) as client:
        df_dl = await scrape_diamond_league_800m(
            client,
            start_year=start_year,
            end_year=end_year,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("\nDry run complete — no files written.")
        return

    # Save Diamond League results
    if not df_dl.empty:
        dl_path = OUTPUT_DIR / "800m_diamond_league_results.parquet"
        df_dl.to_parquet(dl_path, index=False, engine="pyarrow")
        print(f"\nSaved: {dl_path} ({len(df_dl)} rows)")

        # Print summary
        print(f"\n   Men's results: {len(df_dl[df_dl['sex'] == 'M'])}")
        print(f"   Women's results: {len(df_dl[df_dl['sex'] == 'W'])}")
        print(f"   Competitions: {df_dl['competition_name'].nunique()}")
    else:
        print("\nNo Diamond League 800m results found.")

    # Merge with championship data
    if not args.no_merge:
        print(f"\n{'='*60}")
        print("Merging with Championship Data")
        print(f"{'='*60}")
        df_merged = build_merged_results(df_dl)
        if not df_merged.empty:
            merged_path = OUTPUT_DIR / "800m_all_results.parquet"
            df_merged.to_parquet(merged_path, index=False, engine="pyarrow")
            print(f"\nSaved: {merged_path} ({len(df_merged)} rows)")

    print(f"\n{'='*60}")
    print("800m SCRAPER COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

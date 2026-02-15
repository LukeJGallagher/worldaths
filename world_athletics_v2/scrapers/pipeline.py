"""
Data pipeline orchestrator.

Runs scrapers in sequence, validates output, optionally uploads to Azure.

Usage:
    python -m scrapers.pipeline --initial    # First-time deep scrape (everything)
    python -m scrapers.pipeline --daily      # Daily update (rankings + profiles)
    python -m scrapers.pipeline --weekly     # Weekly full refresh
"""

import asyncio
import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env for API keys
try:
    from dotenv import load_dotenv
    _env_v2 = Path(__file__).parent.parent / ".env"
    _env_root = Path(__file__).parent.parent.parent / ".env"
    for _p in (_env_v2, _env_root):
        if _p.exists():
            load_dotenv(_p)
            break
except ImportError:
    pass

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped"
AZURE_DIR = "personal-data/athletics"


async def run_initial_scrape():
    """First-time deep scrape: everything."""
    print(f"\n{'#'*60}")
    print(f"# INITIAL DEEP SCRAPE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}")

    import pandas as pd
    from scrapers.scrape_athletes import scrape_ksa_athletes, scrape_rivals, _get_full_name
    from scrapers.scrape_athletes import profiles_to_dataframe, pbs_to_dataframe, results_to_dataframe, save_parquet
    from scrapers.scrape_rankings import (
        scrape_world_rankings, scrape_season_toplists,
        scrape_top_rankings, extract_rankings_from_profiles,
    )
    from scrapers.scrape_competitions import scrape_calendar, scrape_upcoming, scrape_recent_results
    from api.wa_client import WAClient

    async with WAClient(max_per_second=3.0) as client:
        # 1. KSA Athletes (deep profiles)
        profiles = await scrape_ksa_athletes(client, "KSA")
        if profiles:
            save_parquet(profiles_to_dataframe(profiles), "ksa_athletes.parquet")
            save_parquet(pbs_to_dataframe(profiles), "ksa_personal_bests.parquet")
            save_parquet(results_to_dataframe(profiles), "ksa_results.parquet")

            # Save raw JSON for deep data
            json_path = OUTPUT_DIR / "ksa_deep_profiles.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(profiles, f, ensure_ascii=False, indent=2, default=str)

        # 2. Rivals
        rival_deep_profiles = []
        if profiles:
            rivals = await scrape_rivals(client, profiles, top_n=30)
            if rivals:
                rival_rows = []
                for r in rivals:
                    rp = r.get("profile", {}) or {}
                    basic = rp.get("basicData", {}) or {}
                    rival_rows.append({
                        "athlete_id": r["id"],
                        "full_name": _get_full_name(basic) or r.get("name"),
                        "country_code": basic.get("countryCode") or r.get("country"),
                        "event": r.get("event"),
                        "world_rank": r.get("rank"),
                        "ranking_score": r.get("score"),
                        "region": r.get("region"),
                        "gender": r.get("gender"),
                    })
                save_parquet(pd.DataFrame(rival_rows), "rivals.parquet")
                # Keep rival profiles for ranking extraction
                rival_deep_profiles = [{"id": r["id"], "profile": r.get("profile")} for r in rivals if r.get("profile")]

        # 3. World Rankings (Women's from API)
        df_rankings = await scrape_world_rankings(client)
        if len(df_rankings) > 0:
            df_rankings.to_parquet(OUTPUT_DIR / "world_rankings.parquet", index=False, engine="pyarrow")

        # 4. Men's Rankings (multi-source composite)
        print(f"\n{'='*60}")
        print("Building Men's Rankings (composite)")
        print(f"{'='*60}")

        dfs_men = []

        # 4a. Top rankings (#1 per event, both genders)
        df_top = await scrape_top_rankings(client)
        if len(df_top) > 0:
            df_top.to_parquet(OUTPUT_DIR / "top_rankings.parquet", index=False, engine="pyarrow")
            dfs_men.append(df_top[df_top["gender"] == "M"])

        # 4b. KSA athlete rankings from profiles
        if profiles:
            df_ksa_ranks = extract_rankings_from_profiles(profiles, source_label="ksa_profile")
            if len(df_ksa_ranks) > 0:
                dfs_men.append(df_ksa_ranks[df_ksa_ranks["gender"] == "M"])
                print(f"   KSA profiles: {len(df_ksa_ranks[df_ksa_ranks['gender'] == 'M'])} men's ranking entries")

        # 4c. Rival rankings from profiles
        if rival_deep_profiles:
            df_rival_ranks = extract_rankings_from_profiles(rival_deep_profiles, source_label="rival_profile")
            if len(df_rival_ranks) > 0:
                dfs_men.append(df_rival_ranks[df_rival_ranks["gender"] == "M"])
                print(f"   Rival profiles: {len(df_rival_ranks[df_rival_ranks['gender'] == 'M'])} men's ranking entries")

        # Combine and deduplicate men's rankings
        if dfs_men:
            df_all_men = pd.concat(dfs_men, ignore_index=True)
            # Deduplicate: keep highest-source-priority per athlete per event
            # Sort: ksa_profile > rival_profile > getTopRankings (for data quality)
            source_priority = {"ksa_profile": 0, "rival_profile": 1, "getTopRankings": 2}
            df_all_men["_priority"] = df_all_men["source"].map(source_priority).fillna(9)
            df_all_men = df_all_men.sort_values("_priority").drop_duplicates(
                subset=["athlete_id", "event_slug"], keep="first"
            ).drop(columns=["_priority"]).sort_values(["event_slug", "rank"])
            df_all_men.to_parquet(OUTPUT_DIR / "mens_rankings.parquet", index=False, engine="pyarrow")
            print(f"   Combined men's rankings: {len(df_all_men)} entries across {df_all_men['event_slug'].nunique()} events")

        # 5. Season Toplists (likely NOT AUTHORIZED)
        df_toplists = await scrape_season_toplists(client)
        if len(df_toplists) > 0:
            df_toplists.to_parquet(OUTPUT_DIR / "season_toplists.parquet", index=False, engine="pyarrow")

        # 6. Competition Calendar
        df_cal = await scrape_calendar(client)
        if len(df_cal) > 0:
            df_cal.to_parquet(OUTPUT_DIR / "calendar.parquet", index=False, engine="pyarrow")

        # 7. Upcoming Competitions
        df_upcoming = await scrape_upcoming(client)
        if len(df_upcoming) > 0:
            df_upcoming.to_parquet(OUTPUT_DIR / "upcoming.parquet", index=False, engine="pyarrow")

        # 8. Recent Results
        df_recent = await scrape_recent_results(client, limit=100)
        if len(df_recent) > 0:
            df_recent.to_parquet(OUTPUT_DIR / "recent_results.parquet", index=False, engine="pyarrow")

    print(f"\n{'#'*60}")
    print("INITIAL SCRAPE COMPLETE")
    print(f"{'#'*60}")
    _print_summary()


async def run_daily_update():
    """Daily update: rankings + KSA athlete profiles."""
    print(f"\n{'#'*60}")
    print(f"# DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}")

    import pandas as pd
    from scrapers.scrape_athletes import scrape_ksa_athletes
    from scrapers.scrape_athletes import profiles_to_dataframe, pbs_to_dataframe, results_to_dataframe, save_parquet
    from scrapers.scrape_rankings import scrape_world_rankings, scrape_top_rankings, extract_rankings_from_profiles
    from scrapers.scrape_competitions import scrape_upcoming, scrape_recent_results
    from api.wa_client import WAClient

    async with WAClient(max_per_second=3.0) as client:
        # Rankings (Women's from API)
        df_rankings = await scrape_world_rankings(client, limit_per_event=100)
        if len(df_rankings) > 0:
            df_rankings.to_parquet(OUTPUT_DIR / "world_rankings.parquet", index=False, engine="pyarrow")

        # Top rankings (Men + Women #1 per event)
        df_top = await scrape_top_rankings(client)
        if len(df_top) > 0:
            df_top.to_parquet(OUTPUT_DIR / "top_rankings.parquet", index=False, engine="pyarrow")

        # KSA profiles + results
        profiles = await scrape_ksa_athletes(client, "KSA")
        if profiles:
            save_parquet(profiles_to_dataframe(profiles), "ksa_athletes.parquet")
            save_parquet(pbs_to_dataframe(profiles), "ksa_personal_bests.parquet")
            save_parquet(results_to_dataframe(profiles), "ksa_results.parquet")

            # Update men's rankings from KSA profiles
            df_ksa_ranks = extract_rankings_from_profiles(profiles, source_label="ksa_profile")
            # Merge with existing rival rankings if available
            mens_path = OUTPUT_DIR / "mens_rankings.parquet"
            if mens_path.exists():
                df_existing = pd.read_parquet(mens_path)
                # Remove old KSA entries, add fresh ones
                df_existing = df_existing[df_existing["source"] != "ksa_profile"]
                df_combined = pd.concat([df_ksa_ranks[df_ksa_ranks["gender"] == "M"], df_existing], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=["athlete_id", "event_slug"], keep="first")
                df_combined = df_combined.sort_values(["event_slug", "rank"])
                df_combined.to_parquet(mens_path, index=False, engine="pyarrow")

        # Upcoming
        df_upcoming = await scrape_upcoming(client)
        if len(df_upcoming) > 0:
            df_upcoming.to_parquet(OUTPUT_DIR / "upcoming.parquet", index=False, engine="pyarrow")

        # Recent Results
        df_recent = await scrape_recent_results(client, limit=100)
        if len(df_recent) > 0:
            df_recent.to_parquet(OUTPUT_DIR / "recent_results.parquet", index=False, engine="pyarrow")

    _print_summary()


async def run_weekly_full():
    """Weekly full refresh: everything including toplists and results."""
    await run_initial_scrape()


def _print_summary():
    """Print summary of scraped files."""
    print(f"\n{'='*60}")
    print("Scraped Files Summary:")
    print(f"{'='*60}")

    if OUTPUT_DIR.exists():
        import pandas as pd
        for f in sorted(OUTPUT_DIR.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                size_kb = f.stat().st_size / 1024
                print(f"   {f.name:40s} {len(df):>8,} rows  ({size_kb:.1f} KB)")
            except Exception:
                print(f"   {f.name:40s} (error reading)")


def upload_to_azure():
    """Upload scraped parquet files to Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        print("azure-storage-blob not installed. Skipping upload.")
        return

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        print("AZURE_STORAGE_CONNECTION_STRING not set. Skipping upload.")
        return

    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client("personal-data")

    print(f"\nUploading to Azure Blob Storage...")
    for f in OUTPUT_DIR.glob("*.parquet"):
        blob_name = f"athletics/v2/scraped/{f.name}"
        with open(f, "rb") as data:
            container_client.upload_blob(blob_name, data, overwrite=True)
            print(f"   Uploaded: {blob_name}")

    print("Upload complete.")


async def main():
    parser = argparse.ArgumentParser(description="Run data pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--initial", action="store_true", help="First-time deep scrape")
    group.add_argument("--daily", action="store_true", help="Daily incremental update")
    group.add_argument("--weekly", action="store_true", help="Weekly full refresh")
    parser.add_argument("--upload", action="store_true", help="Upload to Azure after scrape")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.initial:
        await run_initial_scrape()
    elif args.daily:
        await run_daily_update()
    elif args.weekly:
        await run_weekly_full()

    if args.upload:
        upload_to_azure()


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

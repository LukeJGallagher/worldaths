"""
Add missing Project East athletes by ID.

Usage:
    cd world_athletics_v2
    python -m scrapers.add_missing_athletes
"""

import asyncio
import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.wa_client import WAClient
from scrapers.scrape_athletes import (
    _get_full_name,
    profiles_to_dataframe,
    pbs_to_dataframe,
    results_to_dataframe,
)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "scraped"

# Athletes to add: (athlete_id, url_slug)
MISSING_ATHLETES = [
    (14720601, "mohammed-al-dubaisi-14720601"),
    (14850878, "mohammed-duhaim-al-muawi-14850878"),
    (14976748, "nasser-mahmoud-mohammed-14976748"),
]


async def main():
    print("=" * 60)
    print("Adding Missing Project East Athletes")
    print("=" * 60)

    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    async with WAClient(max_per_second=2.0) as client:
        profiles = []
        for aid, slug in MISSING_ATHLETES:
            print(f"\nFetching athlete {aid} ({slug})...")
            try:
                deep = await client.get_deep_athlete_data(aid, url_slug=slug)
                if deep and deep.get("profile"):
                    basic = deep["profile"].get("basicData", {}) or {}
                    name = _get_full_name(basic) or "Unknown"
                    pbs_count = len(
                        (deep["profile"].get("personalBests", {}) or {}).get("results", []) or []
                    )
                    print(f"  OK: {name} ({pbs_count} PBs)")
                    profiles.append(deep)
                else:
                    print(f"  WARN: No profile returned for {aid}")
            except Exception as e:
                print(f"  ERROR: {e}")

        if not profiles:
            print("\nNo profiles fetched. Exiting.")
            return

        # Convert to DataFrames
        new_athletes = profiles_to_dataframe(profiles)
        new_pbs = pbs_to_dataframe(profiles)
        new_results = results_to_dataframe(profiles)

        print(f"\nNew data: {len(new_athletes)} athletes, {len(new_pbs)} PBs, {len(new_results)} results")

        # Load existing parquet files and append
        athletes_path = OUTPUT_DIR / "ksa_athletes.parquet"
        pbs_path = OUTPUT_DIR / "ksa_personal_bests.parquet"
        results_path = OUTPUT_DIR / "ksa_results.parquet"

        # Athletes
        if athletes_path.exists():
            existing = pd.read_parquet(athletes_path)
            # Remove any existing entries for these IDs to avoid duplicates
            new_ids = set(new_athletes["athlete_id"].tolist())
            existing = existing[~existing["athlete_id"].isin(new_ids)]
            combined = pd.concat([existing, new_athletes], ignore_index=True)
            print(f"Athletes: {len(existing)} existing + {len(new_athletes)} new = {len(combined)}")
        else:
            combined = new_athletes
        combined.to_parquet(athletes_path, index=False, engine="pyarrow")

        # PBs
        if pbs_path.exists():
            existing_pbs = pd.read_parquet(pbs_path)
            new_pb_ids = set(new_pbs["athlete_id"].tolist())
            existing_pbs = existing_pbs[~existing_pbs["athlete_id"].isin(new_pb_ids)]
            combined_pbs = pd.concat([existing_pbs, new_pbs], ignore_index=True)
            print(f"PBs: {len(existing_pbs)} existing + {len(new_pbs)} new = {len(combined_pbs)}")
        else:
            combined_pbs = new_pbs
        combined_pbs.to_parquet(pbs_path, index=False, engine="pyarrow")

        # Results
        if results_path.exists():
            existing_results = pd.read_parquet(results_path)
            new_res_ids = set(new_results["athlete_id"].tolist())
            existing_results = existing_results[~existing_results["athlete_id"].isin(new_res_ids)]
            combined_results = pd.concat([existing_results, new_results], ignore_index=True)
            print(f"Results: {len(existing_results)} existing + {len(new_results)} new = {len(combined_results)}")
        else:
            combined_results = new_results
        combined_results.to_parquet(results_path, index=False, engine="pyarrow")

        print("\n" + "=" * 60)
        print("DONE - Athletes added to parquet files")
        print("=" * 60)
        for p in profiles:
            basic = p.get("profile", {}).get("basicData", {}) or {}
            name = _get_full_name(basic) or "Unknown"
            print(f"  + {name} (ID: {p['id']})")


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

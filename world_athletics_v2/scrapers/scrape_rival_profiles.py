"""
Scrape rival athlete profiles from World Athletics GraphQL API.

Enriches rivals.parquet with:
- Personal Best (PB) mark for their primary event
- Season Best (SB) mark for current season
- Best 5 marks and their average
- Latest performance date
- Latest mark

Usage:
    python -m scrapers.scrape_rival_profiles [--limit N]
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure parent is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

from api.wa_client import WAClient


async def scrape_rival_profiles(limit: Optional[int] = None) -> pd.DataFrame:
    """Fetch PBs, SBs, and latest results for all rivals."""

    data_dir = Path(__file__).parent.parent / "data" / "scraped"
    rivals_path = data_dir / "rivals.parquet"

    if not rivals_path.exists():
        print("ERROR: rivals.parquet not found. Run scrape_rankings first.")
        return pd.DataFrame()

    rivals = pd.read_parquet(rivals_path)
    print(f"Loaded {len(rivals)} rivals from {rivals_path.name}")

    if limit:
        # When limiting, work on a copy so we don't lose data
        rivals = rivals.head(limit).copy()
        print(f"Limited to first {limit} rivals")

    # Initialize columns
    rivals["pb_mark"] = None
    rivals["sb_mark"] = None
    rivals["best5_avg"] = None
    rivals["latest_mark"] = None
    rivals["latest_date"] = None
    rivals["latest_competition"] = None
    rivals["performances_count"] = None

    async with WAClient() as client:
        total = len(rivals)
        success = 0
        errors = 0

        for idx, (row_idx, rival) in enumerate(rivals.iterrows()):
            athlete_id = rival.get("athlete_id")
            name = rival.get("full_name", "Unknown")
            event = rival.get("event", "")

            if not athlete_id:
                continue

            # Clean athlete_id (may be string with leading zeros)
            try:
                aid = int(str(athlete_id).lstrip("0"))
            except (ValueError, TypeError):
                continue

            # Encode-safe print for Windows cp1252
            safe_name = name.encode("ascii", "replace").decode("ascii")
            print(f"  [{idx+1}/{total}] {safe_name} ({event})...", end=" ", flush=True)

            try:
                # 1. Get profile (includes PBs and Season Bests)
                profile = await client.get_athlete_profile(aid)
                if profile:
                    # Extract PBs: personalBests -> {results: [...]}
                    pbs_data = profile.get("personalBests") or {}
                    pb_results = pbs_data.get("results", []) if isinstance(pbs_data, dict) else (
                        pbs_data if isinstance(pbs_data, list) else []
                    )
                    for pb in pb_results:
                        if not isinstance(pb, dict):
                            continue
                        discipline = pb.get("discipline", "")
                        if _event_matches(discipline, event):
                            rivals.at[row_idx, "pb_mark"] = pb.get("mark", pb.get("result"))
                            break

                    # Extract SBs: seasonsBests -> {results: [...]}
                    sb_data = profile.get("seasonsBests") or {}
                    sb_results = sb_data.get("results", []) if isinstance(sb_data, dict) else (
                        sb_data if isinstance(sb_data, list) else []
                    )
                    for sb in sb_results:
                        if not isinstance(sb, dict):
                            continue
                        discipline = sb.get("discipline", "")
                        if _event_matches(discipline, event):
                            rivals.at[row_idx, "sb_mark"] = sb.get("mark", sb.get("result"))
                            break

                # Rate limit: 1 request per 0.5s to avoid 503
                await asyncio.sleep(0.5)

                # 2. Get results by discipline (resultsByDate returns null)
                results_data = await client.get_athlete_results_by_discipline(aid)

                found_results = False
                if results_data:
                    # resultsByEvent is a list of {discipline, indoor, results: [...]}
                    events_list = results_data.get("resultsByEvent") or []
                    all_marks = []
                    latest = None
                    for ev in events_list:
                        if not isinstance(ev, dict):
                            continue
                        disc = ev.get("discipline", "")
                        if not _event_matches(disc, event):
                            continue
                        for result in (ev.get("results") or []):
                            if not isinstance(result, dict):
                                continue
                            mark = result.get("mark", result.get("result", ""))
                            if mark:
                                all_marks.append(mark)
                                if latest is None:
                                    latest = {
                                        "mark": mark,
                                        "date": result.get("date", ""),
                                        "competition": result.get("competition", ""),
                                    }

                    if latest:
                        rivals.at[row_idx, "latest_mark"] = latest["mark"]
                        rivals.at[row_idx, "latest_date"] = latest["date"]
                        rivals.at[row_idx, "latest_competition"] = latest["competition"]
                        found_results = True

                    rivals.at[row_idx, "performances_count"] = len(all_marks)

                    # Best 5 average from season marks
                    if all_marks:
                        numeric_marks = _parse_marks(all_marks)
                        if numeric_marks:
                            is_time = _is_time_event(event)
                            sorted_marks = sorted(numeric_marks, reverse=not is_time)[:5]
                            avg5 = sum(sorted_marks) / len(sorted_marks)
                            rivals.at[row_idx, "best5_avg"] = _format_mark(avg5)

                success += 1
                print("OK")

                # Periodic save every 50 athletes to prevent data loss on crash
                if (idx + 1) % 50 == 0:
                    rivals.to_parquet(rivals_path, index=False)
                    print(f"    [checkpoint] Saved progress at {idx+1}/{total}")

            except Exception as e:
                errors += 1
                print(f"ERROR: {e}")

    # Save enriched data — merge back into full file if we only processed a subset
    if limit:
        full_rivals = pd.read_parquet(rivals_path)
        # Update matching rows
        for col in ["pb_mark", "sb_mark", "best5_avg", "latest_mark", "latest_date",
                     "latest_competition", "performances_count"]:
            if col not in full_rivals.columns:
                full_rivals[col] = None
        for _, row in rivals.iterrows():
            mask = full_rivals["athlete_id"] == row["athlete_id"]
            if mask.any():
                for col in ["pb_mark", "sb_mark", "best5_avg", "latest_mark",
                             "latest_date", "latest_competition", "performances_count"]:
                    if row.get(col) is not None:
                        full_rivals.loc[mask, col] = row[col]
        full_rivals.to_parquet(rivals_path, index=False)
    else:
        rivals.to_parquet(rivals_path, index=False)
    print(f"\nDone: {success} scraped, {errors} errors, saved to {rivals_path}")
    print(f"Columns: {list(rivals.columns)}")

    # Summary
    has_pb = rivals["pb_mark"].notna().sum()
    has_sb = rivals["sb_mark"].notna().sum()
    has_latest = rivals["latest_mark"].notna().sum()
    print(f"With PB: {has_pb}, With SB: {has_sb}, With Latest: {has_latest}")

    return rivals


def _parse_marks(marks: list) -> list:
    """Parse a list of mark strings into floats."""
    numeric = []
    for m in marks:
        try:
            numeric.append(float(m))
        except (ValueError, TypeError):
            try:
                parts = str(m).split(":")
                if len(parts) == 2:
                    numeric.append(float(parts[0]) * 60 + float(parts[1]))
            except (ValueError, TypeError):
                pass
    return numeric


def _format_mark(val: float) -> str:
    """Format a numeric mark for display."""
    if val >= 3600:
        return f"{val:.0f}"
    elif val >= 60:
        mins = int(val // 60)
        secs = val - mins * 60
        return f"{mins}:{secs:05.2f}"
    elif val >= 10:
        return f"{val:.2f}"
    else:
        return f"{val:.2f}"


def _event_matches(discipline: str, event: str) -> bool:
    """Check if a discipline name matches the target event."""
    if not discipline or not event:
        return False
    # Normalize both
    d = discipline.lower().replace("-", "").replace(" ", "")
    e = event.lower().replace("-", "").replace(" ", "")

    # Direct match
    if d == e:
        return True

    # Common mappings
    mappings = {
        "100metres": "100m",
        "200metres": "200m",
        "400metres": "400m",
        "800metres": "800m",
        "1500metres": "1500m",
        "5000metres": "5000m",
        "10000metres": "10000m",
        "100metreshurdles": "100mh",
        "110metreshurdles": "110mh",
        "400metreshurdles": "400mh",
        "3000metressteeplechase": "3000msc",
        "highjump": "highjump",
        "polevault": "polevault",
        "longjump": "longjump",
        "triplejump": "triplejump",
        "shotput": "shotput",
        "discusthrow": "discusthrow",
        "hammerthrow": "hammerthrow",
        "javelinthrow": "javelinthrow",
        "decathlon": "decathlon",
        "heptathlon": "heptathlon",
    }

    d_normalized = mappings.get(d, d)
    e_normalized = mappings.get(e, e)

    return d_normalized == e_normalized


def _is_time_event(event: str) -> bool:
    """Check if an event is a time event (lower is better)."""
    e = event.lower().replace("-", "").replace(" ", "")
    field_events = {
        "highjump", "polevault", "longjump", "triplejump",
        "shotput", "discusthrow", "hammerthrow", "javelinthrow",
        "discus", "hammer", "javelin",
        "decathlon", "heptathlon",
    }
    return e not in field_events


if __name__ == "__main__":
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    asyncio.run(scrape_rival_profiles(limit=limit))

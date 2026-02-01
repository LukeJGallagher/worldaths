"""
Generate Markdown Briefings from Parquet Data for NotebookLM Upload.

Creates structured briefings that NotebookLM can use for RAG:
1. KSA Athlete Overview - top athletes by world rank
2. Event Analysis - performance by event category
3. Gap Analysis - gaps to medal standards
4. Competitor Intelligence - top rivals in key events

Usage:
    python generate_briefings.py              # Generate all briefings
    python generate_briefings.py --upload     # Generate and upload to NotebookLM
"""

import os
import duckdb
import pandas as pd
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
PARQUET_DIR = BASE_DIR / "Data" / "parquet"
BRIEFINGS_DIR = BASE_DIR / "briefings"

# Ensure briefings directory exists
BRIEFINGS_DIR.mkdir(exist_ok=True)


def load_parquet(filename: str) -> pd.DataFrame:
    """Load a Parquet file using DuckDB."""
    filepath = PARQUET_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()

    conn = duckdb.connect()
    return conn.execute(f"SELECT * FROM '{filepath}'").df()


def generate_athlete_overview() -> str:
    """Generate KSA Athlete Overview briefing."""
    df = load_parquet("ksa_profiles.parquet")
    if df.empty:
        return "# KSA Athlete Overview\n\nNo data available."

    # Clean and sort by world rank
    df = df.dropna(subset=['best_world_rank'])
    df = df.sort_values('best_world_rank')

    today = datetime.now().strftime('%d %B %Y')

    briefing = f"""# KSA Athletics - Athlete Overview
*Generated: {today}*

## Summary Statistics
- **Total Athletes**: {len(df)}
- **Men**: {len(df[df['gender'] == 'men'])}
- **Women**: {len(df[df['gender'] == 'women'])}

## Top 20 Athletes by World Ranking

| Rank | Athlete | Event | World Rank | Score |
|------|---------|-------|------------|-------|
"""

    for _, row in df.head(20).iterrows():
        briefing += f"| {int(row['best_world_rank'])} | {row['full_name']} | {row['primary_event']} | #{int(row['best_world_rank'])} | {row['best_score']:.0f} |\n"

    # Group by event
    briefing += "\n## Athletes by Event\n\n"

    for event in sorted(df['primary_event'].unique()):
        event_athletes = df[df['primary_event'] == event].head(5)
        briefing += f"### {event}\n"
        for _, row in event_athletes.iterrows():
            briefing += f"- {row['full_name']} (World Rank #{int(row['best_world_rank'])})\n"
        briefing += "\n"

    return briefing


def generate_gap_analysis() -> str:
    """Generate Gap Analysis to medal standards."""
    profiles = load_parquet("ksa_profiles.parquet")
    benchmarks = load_parquet("benchmarks.parquet")

    if profiles.empty or benchmarks.empty:
        return "# Gap Analysis\n\nNo data available."

    today = datetime.now().strftime('%d %B %Y')

    briefing = f"""# KSA Athletics - Gap to Medal Standards
*Generated: {today}*

## Overview
This analysis shows how KSA athletes compare to championship medal standards.

## Medal Standards Reference

| Event | Gender | Gold | Silver | Bronze | Final (8th) |
|-------|--------|------|--------|--------|-------------|
"""

    # Add benchmark rows
    for _, row in benchmarks.head(20).iterrows():
        briefing += f"| {row['Event']} | {row['Gender']} | {row['Gold Standard']} | {row['Silver Standard']} | {row['Bronze Standard']} | {row['Final Standard (8th)']} |\n"

    briefing += "\n## Key Insights\n\n"
    briefing += "- Athletes within 5% of a medal standard are considered 'medal contenders'\n"
    briefing += "- Athletes within 10% are 'potential medallists with development'\n"
    briefing += "- Focus areas should prioritize events where KSA has athletes in top 50 world ranking\n"

    # Top ranked athletes with potential
    top_athletes = profiles[profiles['best_world_rank'] <= 100].sort_values('best_world_rank')
    if not top_athletes.empty:
        briefing += "\n## Medal Potential Athletes (Top 100 World Rank)\n\n"
        for _, row in top_athletes.iterrows():
            briefing += f"- **{row['full_name']}** - {row['primary_event']} - World Rank #{int(row['best_world_rank'])}\n"

    return briefing


def generate_competitor_intelligence() -> str:
    """Generate Competitor Intelligence briefing."""
    master = load_parquet("master.parquet")
    profiles = load_parquet("ksa_profiles.parquet")

    if master.empty or profiles.empty:
        return "# Competitor Intelligence\n\nNo data available."

    today = datetime.now().strftime('%d %B %Y')

    briefing = f"""# KSA Athletics - Competitor Intelligence
*Generated: {today}*

## Overview
Key competitors in events where KSA has ranked athletes.

"""

    # Get KSA's primary events
    ksa_events = profiles['primary_event'].unique()

    for event in sorted(ksa_events)[:10]:  # Top 10 events
        # Get KSA athletes in this event
        ksa_in_event = profiles[profiles['primary_event'] == event].sort_values('best_world_rank')

        if ksa_in_event.empty:
            continue

        briefing += f"## {event}\n\n"
        briefing += f"**KSA Athletes**: "
        briefing += ", ".join([f"{row['full_name']} (#{int(row['best_world_rank'])})"
                              for _, row in ksa_in_event.head(3).iterrows()])
        briefing += "\n\n"

        # Get top competitors from master (non-KSA)
        # Normalize event name for matching
        event_normalized = event.lower().replace(' ', '-').replace('metres', 'metres')

        competitors = master[
            (master['event'].str.lower().str.contains(event_normalized.split('-')[0], na=False)) &
            (master['nat'] != 'KSA')
        ].drop_duplicates(subset=['competitor']).head(10)

        if not competitors.empty:
            briefing += "**Top Competitors**:\n"
            for _, row in competitors.iterrows():
                briefing += f"- {row['competitor']} ({row['nat']}) - {row.get('result', 'N/A')}\n"

        briefing += "\n"

    return briefing


# Project East 2026 Athletes - Priority list for individual profiles
PROJECT_EAST_ATHLETES = [
    {"name": "ATAFI Abdulaziz Abdui", "events": "200m, 100m, 4x100m Relay", "gender": "Men", "dob": "02 DEC 2001"},
    {"name": "AL HIZAM Hussain Asim", "events": "Pole Vault, 60m", "gender": "Men", "dob": "04 JAN 1998"},
    {"name": "TOLU Mohammed Daoud B", "events": "Shot Put, Discus Throw", "gender": "Men", "dob": "30 MAR 2001"},
    {"name": "BAKHEET Sami", "events": "Triple Jump, Long Jump", "gender": "Men", "dob": "06 DEC 2005"},
    {"name": "MOHAMMED Abdullah Abkar", "events": "100m, 200m, 4x100m Relay", "gender": "Men", "dob": "01 JAN 1997"},
    {"name": "AL JUMAH Baqer", "events": "110m Hurdles", "gender": "Men", "dob": "07 MAR 2001"},
    {"name": "NASSER DAROUICHE Hassan", "events": "Triple Jump, Long Jump", "gender": "Men", "dob": "06 JUN 1991"},
    {"name": "MULAHYI Abdullah Rizqallah", "events": "400m Hurdles, 110m Hurdles", "gender": "Men", "dob": "11 JUN 1990"},
    {"name": "MAGHRABI Faisal", "events": "800m, 1500m", "gender": "Men", "dob": "11 APR 1999"},
    {"name": "AL JADANI Abdulaziz Rabie", "events": "100m, 200m", "gender": "Men", "dob": "05 MAY 1996"},
    {"name": "AL-MAJRASHI Ahmed Mabrouk", "events": "4x100m Relay, 100m, 200m", "gender": "Men", "dob": "25 NOV 1997"},
    {"name": "ALSUBAIE Naif Rashid", "events": "400m Hurdles", "gender": "Men", "dob": "28 JUN 2007"},
    {"name": "AL SABYANI Ismail Mohamed", "events": "4x400m Relay, 400m", "gender": "Men", "dob": "25 APR 1987"},
    {"name": "FUTAYNI FUTUANI Ibrahim Mohammed", "events": "4x400m Relay, 400m", "gender": "Men", "dob": "13 JUN 1999"},
    {"name": "ABU BAKR Azzam Ibrahim", "events": "400m Hurdles, 800m", "gender": "Men", "dob": "13 SEP 2004"},
    {"name": "AL JADANI Raed", "events": "1500m, 800m, 3000m Steeplechase", "gender": "Men", "dob": "02 AUG 1996"},
    {"name": "AL YAMI Sami Masoud", "events": "800m", "gender": "Men", "dob": "04 MAY 1999"},
    {"name": "ALYAMI Mubarak Salem", "events": "100m, 200m", "gender": "Men", "dob": "30 APR 2008"},
    {"name": "ALQAHTANI Mubarak Bader", "events": "100m, 200m", "gender": "Men", "dob": "24 SEP 2006"},
    {"name": "HAZAZI Meshal Abdullah", "events": "200m, 400m", "gender": "Men", "dob": "12 FEB 2008"},
    {"name": "HAZAZI Khalid Mohhamed", "events": "3000m Steeplechase, 5000m", "gender": "Men", "dob": "08 JAN 1989"},
    {"name": "HAQAWI Saud Jaber", "events": "100m, 200m", "gender": "Men", "dob": "05 JAN 2006"},
    {"name": "SUFYANI Idris Ayil", "events": "400m Hurdles, 400m", "gender": "Men", "dob": "13 MAR 1995"},
    {"name": "ALZAHRANI Anbar Jamaan", "events": "200m, 400m", "gender": "Men", "dob": "13 MAR 2002"},
    {"name": "HASSAN AL ASMARI Faisal", "events": "100m, Long Jump", "gender": "Men", "dob": "22 MAY 2008"},
    {"name": "AL-SUBAIE Musaad Obaid", "events": "4x400m Relay, 400m", "gender": "Men", "dob": "20 FEB 2008"},
    {"name": "ALDABBOUS Mohsen Hassan", "events": "Decathlon", "gender": "Men", "dob": "25 JUL 2000"},
    {"name": "AHMED ADAM Abdulrahman", "events": "Long Jump, Triple Jump", "gender": "Men", "dob": "14 SEP 2008"},
    {"name": "ALHUMAID Lujain Ibrahim", "events": "100m, 200m", "gender": "Women", "dob": "29 NOV 1999"},
]


def generate_individual_athlete_profiles() -> dict:
    """Generate individual profile pages for Project East athletes."""
    profiles = load_parquet("ksa_profiles.parquet")
    benchmarks = load_parquet("benchmarks.parquet")
    master = load_parquet("master.parquet")

    today = datetime.now().strftime('%d %B %Y')

    # Create athletes subdirectory
    athletes_dir = BRIEFINGS_DIR / "athletes"
    athletes_dir.mkdir(exist_ok=True)

    generated = {}

    for athlete in PROJECT_EAST_ATHLETES:
        name = athlete["name"]
        events = athlete["events"]
        gender = athlete["gender"]
        dob = athlete["dob"]

        # Try to find in profiles data
        name_parts = name.lower().split()
        profile_match = None
        if not profiles.empty:
            for _, row in profiles.iterrows():
                row_name = str(row.get('full_name', '')).lower()
                if any(part in row_name for part in name_parts[:2]):
                    profile_match = row
                    break

        # Calculate age
        try:
            from datetime import datetime as dt
            birth_date = dt.strptime(dob, "%d %b %Y")
            age = (dt.now() - birth_date).days // 365
        except:
            age = "N/A"

        # Build profile content
        profile = f"""# {name}
*KSA Athletics - Project East 2026*
*Profile generated: {today}*

## Athlete Information
- **Name**: {name}
- **Country**: Saudi Arabia (KSA)
- **Gender**: {gender}
- **Date of Birth**: {dob}
- **Age**: {age}
- **Primary Events**: {events}

"""

        # Add ranking info if found
        if profile_match is not None:
            world_rank = profile_match.get('best_world_rank', 'N/A')
            score = profile_match.get('best_score', 'N/A')
            primary_event = profile_match.get('primary_event', events.split(',')[0])

            profile += f"""## Current Rankings
- **World Rank**: #{int(world_rank) if pd.notna(world_rank) else 'N/A'}
- **WA Score**: {int(score) if pd.notna(score) else 'N/A'}
- **Primary Event**: {primary_event}

"""

        # Add event-specific benchmarks
        primary_event = events.split(',')[0].strip()
        profile += f"""## Event Standards
Target standards for {primary_event}:

"""

        if not benchmarks.empty:
            # Find matching benchmark
            event_bench = benchmarks[benchmarks['Event'].str.contains(primary_event.replace('m', ''), case=False, na=False)]
            if not event_bench.empty:
                bench = event_bench.iloc[0]
                profile += f"""| Standard | Time/Distance |
|----------|---------------|
| Gold | {bench.get('Gold Standard', 'N/A')} |
| Silver | {bench.get('Silver Standard', 'N/A')} |
| Bronze | {bench.get('Bronze Standard', 'N/A')} |
| Final (8th) | {bench.get('Final Standard (8th)', 'N/A')} |

"""

        # Add Asian Games 2026 section
        profile += f"""## Asian Games 2026 Target
- **Competition**: Aichi-Nagoya, Japan
- **Dates**: September 19 - October 4, 2026
- **Medal Target**: Based on current form and Asian rankings

## Training Focus
1. Event-specific technical development
2. Competition experience at international level
3. Peak performance timing for September 2026

## Key Competitors (Asian Region)
Athletes from China, Japan, India, Qatar, and Bahrain are primary competition in Asian Games.

---
*Part of Project East 2026 - KSA Athletics Medal Program*
"""

        # Generate safe filename
        safe_name = name.lower().replace(' ', '_').replace("'", "")
        filename = f"{safe_name}.md"
        filepath = athletes_dir / filename

        filepath.write_text(profile, encoding='utf-8')
        generated[filename] = profile
        print(f"  Created athlete profile: {filename}")

    return generated


def generate_asian_games_focus() -> str:
    """Generate Asian Games 2026 focus briefing."""
    profiles = load_parquet("ksa_profiles.parquet")
    benchmarks = load_parquet("benchmarks.parquet")

    today = datetime.now().strftime('%d %B %Y')

    briefing = f"""# Project East 2026 - Asian Games Strategy
*Generated: {today}*

## Mission
Medal targets and qualification tracking for the 2026 Asian Games in Nagoya, Japan.

## Priority Events
Based on current world rankings and historical Asian Games medal standards.

### Tier 1: Medal Contenders (Top 20 Asian Rank)
"""

    # Athletes with good world ranking
    if not profiles.empty:
        top_athletes = profiles[profiles['best_world_rank'] <= 50].sort_values('best_world_rank')
        for _, row in top_athletes.iterrows():
            briefing += f"- **{row['full_name']}** - {row['primary_event']} (World #{int(row['best_world_rank'])})\n"

    briefing += """
### Tier 2: Final Potential (Top 8 Asian)
Athletes who can realistically reach the final with targeted development.

### Tier 3: Experience & Development
Athletes gaining international experience for future Games.

## Key Dates
- **2026 Asian Games**: September 19 - October 4, 2026
- **Qualification Deadline**: TBD
- **Selection Trials**: TBD

## Focus Areas
1. Sprint events (100m, 200m, 400m)
2. Middle distance (800m, 1500m)
3. Field events with current ranked athletes
4. Relay team development

## Historical Context
Asian Games athletics is highly competitive with traditional powerhouses including:
- China
- Japan
- India
- Qatar
- Bahrain

KSA athletes must target personal bests to be competitive at medal level.
"""

    return briefing


def generate_all_briefings():
    """Generate all briefings and save to files."""
    print("Generating briefings...")

    briefings = {
        "01_athlete_overview.md": generate_athlete_overview(),
        "02_gap_analysis.md": generate_gap_analysis(),
        "03_competitor_intelligence.md": generate_competitor_intelligence(),
        "04_asian_games_focus.md": generate_asian_games_focus(),
    }

    for filename, content in briefings.items():
        filepath = BRIEFINGS_DIR / filename
        filepath.write_text(content, encoding='utf-8')
        print(f"  Created: {filepath}")

    # Create combined briefing
    combined = "\n\n---\n\n".join(briefings.values())
    combined_path = BRIEFINGS_DIR / "00_combined_briefing.md"
    combined_path.write_text(combined, encoding='utf-8')
    print(f"  Created: {combined_path}")

    # Generate individual athlete profiles
    print("\nGenerating individual athlete profiles...")
    athlete_profiles = generate_individual_athlete_profiles()
    print(f"  Created {len(athlete_profiles)} athlete profiles")

    print(f"\nAll briefings saved to: {BRIEFINGS_DIR}")
    print(f"Athlete profiles saved to: {BRIEFINGS_DIR / 'athletes'}")
    return briefings


def upload_to_notebooklm():
    """Upload briefings to NotebookLM using notebooklm-py."""
    try:
        from notebooklm import NotebookLM
    except ImportError:
        print("notebooklm-py not installed. Install with: pip install notebooklm-py")
        print("Then run: notebooklm login")
        return False

    nlm = NotebookLM()
    notebook_name = "KSA Athletics Intelligence"

    print(f"Uploading to NotebookLM notebook: {notebook_name}")

    for file in BRIEFINGS_DIR.glob("*.md"):
        print(f"  Uploading: {file.name}")
        try:
            nlm.source_add(str(file), notebook=notebook_name)
        except Exception as e:
            print(f"    Error: {e}")

    print("Upload complete!")
    return True


if __name__ == "__main__":
    import sys

    # Generate briefings
    generate_all_briefings()

    # Upload if --upload flag provided
    if "--upload" in sys.argv:
        upload_to_notebooklm()

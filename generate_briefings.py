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

    print(f"\nAll briefings saved to: {BRIEFINGS_DIR}")
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

"""
World Athletics Dashboard v2 - Main Entry Point

Multi-page Streamlit app for Team Saudi athletics intelligence.
Event-centric design for coaches and performance analysts.

Run: streamlit run world_athletics_v2/app.py
"""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Load .env: try v2 local first, then project root
_env_v2 = Path(__file__).parent / ".env"
_env_root = Path(__file__).parent.parent / ".env"
for _env_path in (_env_v2, _env_root):
    if _env_path.exists():
        load_dotenv(_env_path)
        break

# Apply theme first (must be before any other st. calls)
from components.theme import apply_theme
apply_theme()

# Home page content
from components.theme import render_page_header, render_section_header, render_metric_card
from data.connector import get_connector

# ── Page Header ───────────────────────────────────────────────────────

render_page_header(
    "World Athletics Intelligence",
    "Team Saudi Performance Analysis | Asian Games 2026 | LA 2028"
)

# ── Data Status ───────────────────────────────────────────────────────

dc = get_connector()
status = dc.get_status()
views = status.get("views", [])
counts = status.get("counts", {})

# Determine if we have v2 data or only legacy
has_v2 = "ksa_athletes" in views or "world_rankings" in views
has_legacy = "master" in views or "ksa_profiles" in views or "benchmarks" in views

# Overview metrics - adapt to available data
col1, col2, col3, col4 = st.columns(4)

if has_v2:
    ksa_count = counts.get("ksa_athletes", 0)
    rankings_count = counts.get("world_rankings", 0)
    comps_count = counts.get("calendar", 0)
    rivals_count = counts.get("rivals", 0)
else:
    # Legacy data counts
    ksa_count = counts.get("ksa_profiles", 0)
    rankings_count = counts.get("master", 0)
    comps_count = counts.get("benchmarks", 0)
    rivals_count = 0

with col1:
    render_metric_card("KSA Athletes", str(ksa_count), "excellent" if ksa_count > 0 else "neutral")
with col2:
    label = "Ranking Entries" if has_v2 else "Master Records"
    render_metric_card(label, f"{rankings_count:,}", "excellent" if rankings_count > 0 else "neutral")
with col3:
    label = "Competitions" if has_v2 else "Benchmark Events"
    render_metric_card(label, str(comps_count), "good" if comps_count > 0 else "neutral")
with col4:
    render_metric_card("Rivals Tracked", str(rivals_count), "good" if rivals_count > 0 else "neutral")

# ── Quick Actions ─────────────────────────────────────────────────────

st.markdown("---")

if ksa_count == 0 and rankings_count == 0:
    st.warning(
        "No data loaded yet. Run the initial scrape to populate the database:\n\n"
        "```bash\n"
        "cd world_athletics_v2\n"
        "python -m scrapers.pipeline --initial\n"
        "```"
    )
elif not has_v2 and has_legacy:
    st.info(
        "Using legacy data. Run the scraper for enriched v2 data:\n\n"
        "```bash\n"
        "cd world_athletics_v2\n"
        "python -m scrapers.pipeline --initial\n"
        "```"
    )

# ── KSA Athletes in Form ─────────────────────────────────────────────

if ksa_count > 0:
    render_section_header("KSA Athletes", "Top athletes by ranking score")

    df_athletes = dc.get_ksa_athletes()
    if len(df_athletes) > 0:
        # Build display columns based on what's available (v2 vs legacy)
        # v2 columns: full_name, primary_event, best_world_rank, best_ranking_score, gold_medals, silver_medals, bronze_medals
        # Legacy columns: full_name, primary_event, best_world_rank, best_score, date_of_birth, profile_image_url
        display_cols_v2 = ["full_name", "primary_event", "best_world_rank", "best_ranking_score",
                           "gold_medals", "silver_medals", "bronze_medals"]
        display_cols_legacy = ["full_name", "primary_event", "best_world_rank", "best_score"]

        # Try v2 columns first, fallback to legacy
        if "best_ranking_score" in df_athletes.columns:
            available_cols = [c for c in display_cols_v2 if c in df_athletes.columns]
        else:
            available_cols = [c for c in display_cols_legacy if c in df_athletes.columns]

        column_config = {
            "full_name": st.column_config.TextColumn("Athlete", width="medium"),
            "primary_event": st.column_config.TextColumn("Event"),
            "best_world_rank": st.column_config.NumberColumn("World Rank"),
        }
        # Add score column config based on which exists
        if "best_ranking_score" in df_athletes.columns:
            column_config["best_ranking_score"] = st.column_config.NumberColumn("Score")
        elif "best_score" in df_athletes.columns:
            column_config["best_score"] = st.column_config.NumberColumn("Score")

        if "gold_medals" in df_athletes.columns:
            column_config["gold_medals"] = st.column_config.NumberColumn("G")
        if "silver_medals" in df_athletes.columns:
            column_config["silver_medals"] = st.column_config.NumberColumn("S")
        if "bronze_medals" in df_athletes.columns:
            column_config["bronze_medals"] = st.column_config.NumberColumn("B")

        # Convert numeric columns to int for clean display
        df_display = df_athletes[available_cols].head(10).copy()
        for col in ["best_world_rank", "best_ranking_score", "best_score"]:
            if col in df_display.columns:
                df_display[col] = pd.to_numeric(df_display[col], errors="coerce").fillna(0).astype(int)

        st.dataframe(
            df_display,
            hide_index=True,
            column_config=column_config,
        )

# ── Upcoming Competitions ─────────────────────────────────────────────

if counts.get("upcoming", 0) > 0 or counts.get("calendar", 0) > 0:
    render_section_header("Upcoming Competitions", "Next events on the calendar")

    df_upcoming = dc.get_upcoming_competitions()
    if len(df_upcoming) > 0:
        display_cols = ["name", "venue", "start_date", "ranking_category"]
        available_cols = [c for c in display_cols if c in df_upcoming.columns]
        st.dataframe(
            df_upcoming[available_cols].head(10),
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Competition", width="large"),
                "venue": st.column_config.TextColumn("Venue"),
                "start_date": st.column_config.TextColumn("Date"),
                "ranking_category": st.column_config.TextColumn("Category"),
            },
        )

# ── Recent Results ────────────────────────────────────────────────────

if counts.get("recent_results", 0) > 0:
    render_section_header("Recent Results", "Latest global results")

    df_recent = dc.get_recent_results(limit=10)
    if len(df_recent) > 0:
        st.dataframe(df_recent.head(10), hide_index=True)

# ── Data Sources ──────────────────────────────────────────────────────

with st.expander("Data Status"):
    st.json(status)

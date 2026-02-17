"""
LA 2028 - Road to the Olympics.

Features:
- Olympic qualification tracking
- Entry standards + ranking pathway
- KSA athletes' qualification status
- Timeline to qualification deadline
- Legacy road_to_tokyo data fallback
"""

import streamlit as st
from datetime import datetime

from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar,
)
from data.connector import get_connector

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="LA 2028", page_icon="🏛️", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header("Road to LA 2028", "Olympic Games Los Angeles | July 2028")

dc = get_connector()

# ── Countdown ─────────────────────────────────────────────────────────

olympics_date = datetime(2028, 7, 14)
days_to_go = (olympics_date - datetime.now()).days

col1, col2, col3 = st.columns(3)
with col1:
    render_metric_card("Days to Go", str(max(days_to_go, 0)), "excellent")
with col2:
    render_metric_card("Qualification Window", "Open 2027", "good")
with col3:
    render_metric_card("Pathways", "Entry Standard + Rankings", "neutral")

# ── Qualification Standards ───────────────────────────────────────────

render_section_header("Olympic Entry Standards", "LA 2028 qualification marks (TBC)")

st.markdown("""
Olympic qualification uses two pathways:
1. **Entry Standard** - Achieve the qualifying mark in the qualification window
2. **World Rankings** - Qualify via World Athletics ranking position

Standards for LA 2028 have not been officially published yet.
Based on historical patterns (Tokyo 2020, Paris 2024), expected standards will be similar to:
""")

# Placeholder standards based on Paris 2024 patterns
import pandas as pd
standards_data = {
    "Event": ["100m", "200m", "400m", "800m", "1500m", "5000m", "10,000m",
              "110m H", "400m H", "High Jump", "Long Jump", "Shot Put", "Javelin"],
    "Men Standard": ["10.00", "20.16", "44.90", "1:44.70", "3:33.50", "13:05.00", "27:00.00",
                     "13.27", "48.70", "2.33", "8.27", "21.50", "85.50"],
    "Women Standard": ["11.07", "22.57", "50.40", "1:58.35", "3:55.00", "14:52.00", "30:40.00",
                       "12.77", "54.85", "1.97", "6.86", "18.80", "64.00"],
    "Quota (each)": [48, 48, 48, 48, 45, 42, 42, 40, 40, 32, 32, 32, 32],
}
df_standards = pd.DataFrame(standards_data)
st.dataframe(df_standards, hide_index=True, height=400)

st.caption("*Standards are estimated based on Paris 2024. Official LA 2028 standards will be updated when published.*")

# ── KSA Development Track ────────────────────────────────────────────

render_section_header("KSA Development Track", "Athletes building towards 2028")

ksa = dc.get_ksa_athletes()
if len(ksa) > 0:
    st.markdown("Athletes currently ranked with potential for Olympic development:")

    # Handle both v2 and legacy column names
    display_cols_v2 = ["full_name", "primary_event", "best_world_rank", "best_ranking_score"]
    display_cols_legacy = ["full_name", "primary_event", "best_world_rank", "best_score"]

    if "best_ranking_score" in ksa.columns:
        available_cols = [c for c in display_cols_v2 if c in ksa.columns]
    else:
        available_cols = [c for c in display_cols_legacy if c in ksa.columns]

    # Build column config
    column_config = {
        "full_name": st.column_config.TextColumn("Athlete", width="medium"),
        "primary_event": st.column_config.TextColumn("Event"),
        "best_world_rank": st.column_config.NumberColumn("World Rank", format=".0f"),
    }
    if "best_ranking_score" in ksa.columns:
        column_config["best_ranking_score"] = st.column_config.NumberColumn("Score", format=",.0f")
    elif "best_score" in ksa.columns:
        column_config["best_score"] = st.column_config.NumberColumn("Score", format=",.0f")

    st.dataframe(
        ksa[available_cols].head(20),
        hide_index=True,
        column_config=column_config,
    )
else:
    st.info("No KSA athlete data loaded.")

# ── Qualification Tracking (from legacy road_to_tokyo or v2) ─────────

render_section_header("Qualification Tracker", "KSA qualification status from available data")

quals = dc.get_qualifications(country="KSA")
if len(quals) > 0:
    st.markdown("Qualification data from the most recent tracking period:")

    # Handle both v2 and legacy road_to_tokyo column names
    display_cols_v2 = ["event", "athlete", "country", "status", "qualification_mark"]
    display_cols_legacy = ["Actual_Event_Name", "Athlete", "Federation", "Qualification_Status",
                           "QP", "FP", "Status", "Details", "Event_Type"]

    available_cols = [c for c in display_cols_v2 if c in quals.columns]
    if not available_cols or len(available_cols) < 2:
        available_cols = [c for c in display_cols_legacy if c in quals.columns]

    if len(available_cols) > 0:
        # Summary metrics
        total_athletes = 0
        qualified = 0
        if "Athlete" in quals.columns:
            total_athletes = quals["Athlete"].nunique()
        elif "athlete" in quals.columns:
            total_athletes = quals["athlete"].nunique()

        if "Qualification_Status" in quals.columns:
            qualified = len(quals[quals["Qualification_Status"].str.lower().str.contains("qualified", na=False)])
        elif "status" in quals.columns:
            qualified = len(quals[quals["status"].str.lower().str.contains("qualified", na=False)])

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Athletes Tracked", str(total_athletes), "good")
        with m2:
            render_metric_card("Qualified", str(qualified), "excellent" if qualified > 0 else "neutral")
        with m3:
            events_tracked = 0
            if "Actual_Event_Name" in quals.columns:
                events_tracked = quals["Actual_Event_Name"].nunique()
            elif "event" in quals.columns:
                events_tracked = quals["event"].nunique()
            render_metric_card("Events", str(events_tracked), "neutral")

        # Build column config for legacy columns
        column_config = {}
        if "Athlete" in available_cols:
            column_config["Athlete"] = st.column_config.TextColumn("Athlete", width="medium")
        if "Actual_Event_Name" in available_cols:
            column_config["Actual_Event_Name"] = st.column_config.TextColumn("Event")
        if "Event_Type" in available_cols:
            column_config["Event_Type"] = st.column_config.TextColumn("Category")
        if "Federation" in available_cols:
            column_config["Federation"] = st.column_config.TextColumn("Country")
        if "Qualification_Status" in available_cols:
            column_config["Qualification_Status"] = st.column_config.TextColumn("Status")
        if "QP" in available_cols:
            column_config["QP"] = st.column_config.TextColumn("Qual Points")
        if "FP" in available_cols:
            column_config["FP"] = st.column_config.TextColumn("Final Points")
        if "Details" in available_cols:
            column_config["Details"] = st.column_config.TextColumn("Notes", width="large")

        st.dataframe(
            quals[available_cols],
            hide_index=True,
            column_config=column_config,
            height=500,
        )
    else:
        st.dataframe(quals, hide_index=True)
else:
    st.info(
        "No qualification tracking data available yet.\n\n"
        "LA 2028 qualification tracking will be activated when the qualification window opens in 2027."
    )

# ── Benchmarks Reference ─────────────────────────────────────────────

render_section_header("Historical Standards Reference", "Past championship standards for context")

benchmarks = dc.get_benchmarks()
if len(benchmarks) > 0:
    st.markdown("Use these historical championship standards as reference points for development targets:")

    display_cols = ["Event", "Gender", "Year", "Gold Standard", "Silver Standard", "Bronze Standard",
                    "Final Standard (8th)", "Top 8 Average"]
    available_cols = [c for c in display_cols if c in benchmarks.columns]

    if len(available_cols) > 0:
        st.dataframe(benchmarks[available_cols], hide_index=True, height=400)

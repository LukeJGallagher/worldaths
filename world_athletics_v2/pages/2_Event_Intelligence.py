"""
Event Intelligence - THE main page.

User flow: Pick Event (100m) -> Pick Gender (Men) -> See everything:
- What It Takes to Win (medal/finals/heats standards)
- KSA Athletes in this event
- Rival Intelligence (Asian + Global)
- Qualification Status
- Season Toplist
"""

import streamlit as st
from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar, TEAL_PRIMARY, GOLD_ACCENT,
)
from components.filters import event_gender_picker, age_category_filter, region_filter
from components.charts import (
    progression_chart, standards_waterfall, gap_to_medal_chart,
    season_progression_chart,
)
from data.connector import get_connector
from data.event_utils import (
    get_discipline_code, get_event_group, get_event_type,
    display_to_db, ASIAN_COUNTRY_CODES, format_event_name,
    normalize_event_for_match,
)

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Event Intelligence", page_icon="🎯", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header("Event Intelligence", "Deep dive into any athletics event")

dc = get_connector()

# ── Event & Gender Selection ──────────────────────────────────────────

col_event, col_age = st.columns([3, 1])
with col_event:
    event, gender = event_gender_picker(key_prefix="ei")
with col_age:
    age_cat = age_category_filter(key="ei_age")

event_type = get_event_type(event)
lower_is_better = event_type == "time"
gender_label = "Men" if gender == "M" else "Women"
age_label = f" ({age_cat})" if age_cat else ""

st.markdown(f"### {event} {gender_label}{age_label}")

# ── Section A: What It Takes to Win ───────────────────────────────────

render_section_header(
    "What It Takes to Win",
    f"Medal standards and round benchmarks for {event} {gender_label}"
)

# Get season toplist for standards (falls back to master data)
disc_code = get_discipline_code(event)
if disc_code:
    # Season picker
    import datetime
    current_year = datetime.date.today().year
    season_options = [current_year - i for i in range(10)]
    season_col1, season_col2 = st.columns([1, 5])
    with season_col1:
        selected_season = st.selectbox(
            "Season", season_options, index=0, key="ei_season"
        )

    toplist = dc.get_season_toplist(event, gender, season=selected_season, age_category=age_cat, limit=100)

    if len(toplist) > 0:
        col1, col2, col3, col4 = st.columns(4)

        # Column mapping: v2 uses "mark", legacy master uses "result"
        mark_col = "mark" if "mark" in toplist.columns else "result" if "result" in toplist.columns else None

        if mark_col:
            marks = toplist[mark_col].tolist()
            with col1:
                render_metric_card("World Leader", str(marks[0]) if marks else "N/A", "gold")
            with col2:
                render_metric_card("Top 8 (Finals)", str(marks[7]) if len(marks) > 7 else "N/A", "excellent")
            with col3:
                render_metric_card("Top 16 (Semis)", str(marks[15]) if len(marks) > 15 else "N/A", "good")
            with col4:
                render_metric_card("Top 32 (Heats)", str(marks[31]) if len(marks) > 31 else "N/A", "neutral")

        # Toplist table
        st.markdown(f"#### {selected_season} Season Toplist")

        # Handle both v2 and legacy column names
        display_cols_v2 = ["place", "mark", "competitor", "country", "venue", "date"]
        display_cols_legacy = ["place", "result", "competitor", "nat", "venue", "date"]

        available_cols = [c for c in display_cols_v2 if c in toplist.columns]
        if not available_cols or len(available_cols) < 3:
            available_cols = [c for c in display_cols_legacy if c in toplist.columns]

        # Highlight KSA athletes - check both country columns
        country_col = "country" if "country" in toplist.columns else "nat" if "nat" in toplist.columns else None

        def highlight_ksa(row):
            if country_col and row.get(country_col) == "KSA":
                return ["background-color: rgba(35, 80, 50, 0.15)"] * len(row)
            return [""] * len(row)

        if len(available_cols) > 0:
            st.dataframe(
                toplist[available_cols].head(50).style.apply(highlight_ksa, axis=1),
                hide_index=True,
                height=400,
            )
    else:
        st.info(f"No toplist data available for {event}. Run the rankings scraper first.")

# Get benchmarks from legacy data
benchmarks = dc.get_benchmarks()
if len(benchmarks) > 0:
    event_db = display_to_db(event)
    event_norm = normalize_event_for_match(event_db)

    event_benchmarks = benchmarks[
        benchmarks.apply(
            lambda row: normalize_event_for_match(str(row.get("Event", ""))) == event_norm
            and str(row.get("Gender", "")).lower().startswith(gender.lower()),
            axis=1
        )
    ]

    if len(event_benchmarks) > 0:
        st.markdown("#### Historical Championship Standards")
        st.dataframe(event_benchmarks, hide_index=True)

# ── Section B: KSA Athletes in This Event ─────────────────────────────

render_section_header(
    "KSA Athletes",
    f"Saudi athletes competing in {event}"
)

# Find KSA athletes in this event from rankings (falls back to master)
ksa_rankings = dc.get_ksa_rankings(event)

if len(ksa_rankings) > 0:
    for _, athlete in ksa_rankings.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                # Handle both v2 'athlete' and legacy 'competitor' column names
                name = athlete.get('athlete', athlete.get('competitor', 'Unknown'))
                st.markdown(f"**{name}**")
            with col2:
                rank_val = athlete.get('rank', 'N/A')
                render_metric_card("Rank", f"#{rank_val}", "excellent")
            with col3:
                score = athlete.get('ranking_score', athlete.get('resultscore', 0))
                score_str = f"{score:.0f}" if score and score > 0 else "N/A"
                render_metric_card("Score", score_str, "good")
            with col4:
                comps = athlete.get("competitions_scored", "N/A")
                render_metric_card("Comps", str(comps), "neutral")

    # Get detailed results for KSA athletes (falls back to master)
    ksa_results = dc.get_ksa_results(discipline=event, limit=50)
    if len(ksa_results) > 0:
        st.markdown("#### Recent KSA Results")
        # Handle both v2 and legacy column names
        display_cols_v2 = ["full_name", "date", "competition", "mark", "place", "venue"]
        display_cols_legacy = ["competitor", "date", "event", "result", "pos", "venue"]

        available_cols = [c for c in display_cols_v2 if c in ksa_results.columns]
        if not available_cols or len(available_cols) < 3:
            available_cols = [c for c in display_cols_legacy if c in ksa_results.columns]

        if len(available_cols) > 0:
            # Sort most recent first (parse date strings like "19 MAR 2025")
            date_col = "date" if "date" in ksa_results.columns else None
            if date_col:
                import pandas as _pd
                ksa_results["_parsed_date"] = _pd.to_datetime(ksa_results[date_col], format="mixed", errors="coerce")
                ksa_results = ksa_results.sort_values("_parsed_date", ascending=False).drop(columns=["_parsed_date"])
            st.dataframe(ksa_results[available_cols], hide_index=True)
else:
    # Try PBs
    ksa_pbs = dc.get_ksa_athlete_pbs()
    if len(ksa_pbs) > 0:
        # Handle both v2 'discipline' and legacy 'event' column names
        disc_col = "discipline" if "discipline" in ksa_pbs.columns else "event" if "event" in ksa_pbs.columns else None
        if disc_col:
            event_pbs = ksa_pbs[
                ksa_pbs[disc_col].apply(
                    lambda d: normalize_event_for_match(str(d)) == normalize_event_for_match(event)
                )
            ]
            if len(event_pbs) > 0:
                st.dataframe(event_pbs, hide_index=True)
            else:
                st.info(f"No KSA athletes found competing in {event}.")
        else:
            st.info(f"No KSA athletes found competing in {event}.")
    else:
        st.info(f"No KSA data loaded. Run: `python -m scrapers.scrape_athletes`")

# ── Section C: Rival Intelligence ─────────────────────────────────────

render_section_header(
    "Rival Intelligence",
    f"Top competitors in {event} {gender_label}"
)

tab_asian, tab_global = st.tabs(["Asian Rivals", "Global Rivals"])

with tab_asian:
    asian_rivals = dc.get_rivals(event=event, gender=gender, region="asia", limit=30)
    if len(asian_rivals) > 0:
        display_cols = ["full_name", "country_code", "world_rank", "ranking_score"]
        available_cols = [c for c in display_cols if c in asian_rivals.columns]
        st.dataframe(asian_rivals[available_cols], hide_index=True)
    else:
        # Try to show Asian athletes from world rankings as fallback
        world_ranks = dc.get_world_rankings(event=event, gender=gender, age_category=age_cat, limit=200)
        if len(world_ranks) > 0:
            country_col = "country" if "country" in world_ranks.columns else "nat" if "nat" in world_ranks.columns else None
            if country_col:
                asian_df = world_ranks[world_ranks[country_col].isin(ASIAN_COUNTRY_CODES)]
                if len(asian_df) > 0:
                    st.dataframe(asian_df.head(20), hide_index=True)
                else:
                    st.info("No Asian athletes found in rankings for this event.")
            else:
                st.info("No Asian rival data loaded.")
        else:
            st.info("No Asian rival data loaded. Run: `python -m scrapers.scrape_athletes --rivals`")

with tab_global:
    global_rivals = dc.get_rivals(event=event, gender=gender, region="global", limit=30)
    if len(global_rivals) > 0:
        display_cols = ["full_name", "country_code", "world_rank", "ranking_score"]
        available_cols = [c for c in display_cols if c in global_rivals.columns]
        st.dataframe(global_rivals[available_cols], hide_index=True)
    else:
        # Show world rankings as fallback
        world_ranks = dc.get_world_rankings(event=event, gender=gender, age_category=age_cat, limit=30)
        if len(world_ranks) > 0:
            st.dataframe(world_ranks.head(20), hide_index=True)
        else:
            st.info("No global rival data loaded.")

# ── Section D: World Rankings ─────────────────────────────────────────

render_section_header(
    "World Rankings",
    f"Current WA world rankings for {event} {gender_label} (updated weekly)"
)

world_rankings = dc.get_world_rankings(event=event, gender=gender, age_category=age_cat, limit=50)
if len(world_rankings) > 0:
    # Show ranking date if available
    if "rank_date" in world_rankings.columns:
        rd = str(world_rankings["rank_date"].iloc[0])[:10]
        st.caption(f"Rankings as of: {rd}")

    # Handle both v2 and legacy column names
    display_cols_v2 = ["rank", "athlete", "country", "ranking_score", "competitions_scored"]
    display_cols_legacy = ["rank", "competitor", "nat", "resultscore", "event"]

    available_cols = [c for c in display_cols_v2 if c in world_rankings.columns]
    if not available_cols or len(available_cols) < 3:
        available_cols = [c for c in display_cols_legacy if c in world_rankings.columns]

    # Determine country column for highlighting
    country_col = "country" if "country" in world_rankings.columns else "nat" if "nat" in world_rankings.columns else None

    def highlight_ksa_rank(row):
        if country_col and row.get(country_col) == "KSA":
            return ["background-color: rgba(35, 80, 50, 0.15); font-weight: bold"] * len(row)
        elif country_col and row.get(country_col) in ASIAN_COUNTRY_CODES:
            return ["background-color: rgba(160, 142, 102, 0.1)"] * len(row)
        return [""] * len(row)

    if len(available_cols) > 0:
        st.dataframe(
            world_rankings[available_cols].style.apply(highlight_ksa_rank, axis=1),
            hide_index=True,
            height=500,
        )
else:
    st.info(f"No ranking data for {event}. Run: `python -m scrapers.scrape_rankings`")

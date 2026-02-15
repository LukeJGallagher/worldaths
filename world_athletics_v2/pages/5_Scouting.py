"""
Scouting - Rival analysis, H2H, form rankings.

Features:
- Full competitive landscape by event
- Head-to-head builder
- Age/country/event filters
- "Who's Hot" trending athletes
"""

import streamlit as st
from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_sidebar, TEAL_PRIMARY, GOLD_ACCENT,
)
from components.filters import event_gender_picker, country_filter, age_category_filter
from data.connector import get_connector
from data.event_utils import ASIAN_COUNTRY_CODES

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Scouting", page_icon="🔍", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header("Scouting & Intelligence", "Rival analysis and competitive landscape")

dc = get_connector()

# ── Filters ───────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    event, gender = event_gender_picker(key_prefix="scout")
with col2:
    country = country_filter(key="scout_country")
with col3:
    age_cat = age_category_filter(key="scout_age")

gender_label = "Men" if gender == "M" else "Women"

# ── Tabs ──────────────────────────────────────────────────────────────

tab_rankings, tab_rivals, tab_h2h = st.tabs([
    "World Rankings", "Rival Watch", "Head-to-Head"
])

# ── World Rankings Tab ────────────────────────────────────────────────

with tab_rankings:
    render_section_header(
        f"World Rankings - {event} {gender_label}",
        "Current world ranking positions with KSA highlighted"
    )

    df = dc.get_world_rankings(event=event, gender=gender, country=country, age_category=age_cat, limit=200)

    if len(df) > 0:
        # Determine country column (v2 uses 'country', legacy uses 'nat')
        country_col = "country" if "country" in df.columns else "nat" if "nat" in df.columns else None

        # Summary metrics
        ksa_in_rankings = df[df[country_col] == "KSA"] if country_col else df.head(0)
        asian_in_rankings = df[df[country_col].isin(ASIAN_COUNTRY_CODES)] if country_col else df.head(0)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Ranked", len(df))
        with m2:
            st.metric("KSA Athletes", len(ksa_in_rankings))
        with m3:
            st.metric("Asian Athletes", len(asian_in_rankings))

        # Table with highlighting - handle both v2 and legacy column names
        display_cols_v2 = ["rank", "athlete", "country", "ranking_score", "competitions_scored"]
        display_cols_legacy = ["rank", "competitor", "nat", "resultscore", "result"]

        available_cols = [c for c in display_cols_v2 if c in df.columns]
        if not available_cols or len(available_cols) < 3:
            available_cols = [c for c in display_cols_legacy if c in df.columns]

        def highlight_row(row):
            if country_col and row.get(country_col) == "KSA":
                return [f"background-color: rgba(35, 80, 50, 0.2); font-weight: bold"] * len(row)
            elif country_col and row.get(country_col) in ASIAN_COUNTRY_CODES:
                return ["background-color: rgba(160, 142, 102, 0.1)"] * len(row)
            return [""] * len(row)

        if len(available_cols) > 0:
            st.dataframe(
                df[available_cols].style.apply(highlight_row, axis=1),
                hide_index=True,
                height=600,
            )
    else:
        st.info(f"No ranking data for {event}. Run: `python -m scrapers.scrape_rankings`")

# ── Rival Watch Tab ───────────────────────────────────────────────────

with tab_rivals:
    render_section_header(
        "Rival Watch",
        f"Key competitors to monitor in {event} {gender_label}"
    )

    tab_a, tab_g = st.tabs(["Asian Rivals", "Global Rivals"])

    with tab_a:
        asian = dc.get_rivals(event=event, gender=gender, region="asia", limit=30)
        if len(asian) > 0:
            st.dataframe(asian, hide_index=True, height=400)
        else:
            # Fallback: filter world rankings for Asian countries
            world_ranks = dc.get_world_rankings(event=event, gender=gender, age_category=age_cat, limit=200)
            if len(world_ranks) > 0:
                country_col = "country" if "country" in world_ranks.columns else "nat" if "nat" in world_ranks.columns else None
                if country_col:
                    asian_df = world_ranks[world_ranks[country_col].isin(ASIAN_COUNTRY_CODES)]
                    if len(asian_df) > 0:
                        st.markdown(f"**{len(asian_df)} Asian athletes found in rankings**")
                        st.dataframe(asian_df.head(30), hide_index=True, height=400)
                    else:
                        st.info("No Asian athletes found in rankings for this event.")
                else:
                    st.info("No Asian rival data. Run: `python -m scrapers.scrape_athletes --rivals`")
            else:
                st.info("No Asian rival data. Run: `python -m scrapers.scrape_athletes --rivals`")

    with tab_g:
        glob_rivals = dc.get_rivals(event=event, gender=gender, region="global", limit=30)
        if len(glob_rivals) > 0:
            st.dataframe(glob_rivals, hide_index=True, height=400)
        else:
            # Fallback: show world rankings
            world_ranks = dc.get_world_rankings(event=event, gender=gender, age_category=age_cat, limit=30)
            if len(world_ranks) > 0:
                st.markdown(f"**Top {len(world_ranks)} in world rankings**")
                st.dataframe(world_ranks, hide_index=True, height=400)
            else:
                st.info("No global rival data.")

# ── Head-to-Head Tab ──────────────────────────────────────────────────

with tab_h2h:
    render_section_header(
        "Head-to-Head Builder",
        "Compare two athletes directly"
    )

    st.info(
        "Head-to-head data requires live API queries. "
        "This feature will use the `HeadToHead` GraphQL query to fetch "
        "direct matchup records between any two athletes.\n\n"
        "Coming in Phase 5."
    )

    col1, col2 = st.columns(2)
    with col1:
        athlete1 = st.text_input("Athlete 1", placeholder="e.g. Hussain Al Hizam", key="h2h_a1")
    with col2:
        athlete2 = st.text_input("Athlete 2", placeholder="e.g. Fred Kerley", key="h2h_a2")

    if athlete1 and athlete2:
        st.markdown(f"### {athlete1} vs {athlete2}")

        # Try to find results for both athletes in legacy data
        results_1 = dc.get_ksa_results(athlete_name=athlete1, limit=20)
        results_2 = dc.get_ksa_results(athlete_name=athlete2, limit=20)

        if len(results_1) > 0 or len(results_2) > 0:
            c1, c2 = st.columns(2)
            with c1:
                if len(results_1) > 0:
                    st.markdown(f"**{athlete1} - Recent Results**")
                    mark_col = "mark" if "mark" in results_1.columns else "result" if "result" in results_1.columns else None
                    date_col = "date" if "date" in results_1.columns else None
                    if mark_col and date_col:
                        display = [date_col, mark_col]
                        event_col = "discipline" if "discipline" in results_1.columns else "event" if "event" in results_1.columns else None
                        if event_col:
                            display.insert(1, event_col)
                        st.dataframe(results_1[display], hide_index=True)
                else:
                    st.info(f"No results found for {athlete1}")
            with c2:
                if len(results_2) > 0:
                    st.markdown(f"**{athlete2} - Recent Results**")
                    mark_col = "mark" if "mark" in results_2.columns else "result" if "result" in results_2.columns else None
                    date_col = "date" if "date" in results_2.columns else None
                    if mark_col and date_col:
                        display = [date_col, mark_col]
                        event_col = "discipline" if "discipline" in results_2.columns else "event" if "event" in results_2.columns else None
                        if event_col:
                            display.insert(1, event_col)
                        st.dataframe(results_2[display], hide_index=True)
                else:
                    st.info(f"No results found for {athlete2}")
        else:
            st.markdown("*H2H data will be fetched via API when connected.*")

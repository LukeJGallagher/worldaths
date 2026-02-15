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

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    event, gender = event_gender_picker(key_prefix="scout")
with col2:
    region_options = ["All", "Asian", "KSA Only"]
    region_filter = st.selectbox("Region", region_options, key="scout_region")
with col3:
    country = country_filter(key="scout_country")
with col4:
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

        # Apply region filter
        df_full = df.copy()
        if region_filter == "Asian" and country_col:
            df = df[df[country_col].isin(ASIAN_COUNTRY_CODES)]
        elif region_filter == "KSA Only" and country_col:
            df = df[df[country_col] == "KSA"]

        # Summary metrics
        ksa_in_rankings = df_full[df_full[country_col] == "KSA"] if country_col else df_full.head(0)
        asian_in_rankings = df_full[df_full[country_col].isin(ASIAN_COUNTRY_CODES)] if country_col else df_full.head(0)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Showing", len(df))
        with m2:
            st.metric("Total Ranked", len(df_full))
        with m3:
            st.metric("KSA Athletes", len(ksa_in_rankings))
        with m4:
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

    # Enriched rival display
    RIVAL_DISPLAY_COLS = [
        "full_name", "country_code", "world_rank", "ranking_score",
        "pb_mark", "sb_mark", "latest_mark", "latest_date",
        "best5_avg", "performances_count",
    ]
    RIVAL_COL_LABELS = {
        "full_name": "Athlete", "country_code": "Nat", "world_rank": "World Rank",
        "ranking_score": "Score", "pb_mark": "PB", "sb_mark": "SB",
        "latest_mark": "Latest", "latest_date": "Date",
        "best5_avg": "Best 5 Avg", "performances_count": "# Perfs",
    }

    def _show_rivals_table(rivals_df):
        if rivals_df.empty:
            st.info("No rival data. Run: `python -m scrapers.scrape_rival_profiles`")
            return
        available = [c for c in RIVAL_DISPLAY_COLS if c in rivals_df.columns]
        if not available:
            st.dataframe(rivals_df, hide_index=True, height=400)
            return
        display = rivals_df[available].copy()
        display = display.rename(columns={c: RIVAL_COL_LABELS.get(c, c) for c in available})

        def _hl(row):
            nat = row.get("Nat", "")
            if str(nat).upper() in ("KSA", "SAU"):
                return ["background-color: rgba(35, 80, 50, 0.15); font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display.style.apply(_hl, axis=1), hide_index=True,
            height=min(500, 40 + len(display) * 35),
        )

    def _show_performers_table(df):
        """Display top performers from master data."""
        if df.empty:
            return False
        display = df.rename(columns={
            "full_name": "Athlete", "country_code": "Nat",
            "pb_mark": "PB", "best_mark_numeric": "Best Mark",
            "performances_count": "# Perfs", "latest_date": "Date",
            "latest_venue": "Venue",
        })
        show_cols = [c for c in ["Athlete", "Nat", "PB", "Best Mark", "# Perfs", "Date"] if c in display.columns]

        # Format Best Mark
        if "Best Mark" in display.columns:
            from data.event_utils import get_event_type
            et = get_event_type(event)
            def _fmt(v):
                try:
                    v = float(v)
                    if v >= 60:
                        return f"{int(v // 60)}:{(v % 60):05.2f}"
                    return f"{v:.2f}"
                except (ValueError, TypeError):
                    return str(v)
            display["Best Mark"] = display["Best Mark"].apply(_fmt)

        def _hl(row):
            nat = str(row.get("Nat", ""))
            if nat.upper() in ("KSA", "SAU"):
                return ["background-color: rgba(35, 80, 50, 0.15); font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display[show_cols].style.apply(_hl, axis=1), hide_index=True,
            height=min(500, 40 + len(display) * 35),
        )
        return True

    tab_a, tab_g = st.tabs(["Asian Rivals", "Global Rivals"])

    with tab_a:
        # Use master data for correct gender (rivals.parquet has Women's data)
        asian_performers = dc.get_top_performers(
            event=event, gender=gender,
            country_codes=list(ASIAN_COUNTRY_CODES), limit=30,
        )
        if len(asian_performers) > 0:
            st.caption(f"Top {len(asian_performers)} Asian {gender_label} from recent competition data")
            _show_performers_table(asian_performers)
        else:
            # Fall back to rivals
            asian = dc.get_rivals(event=event, gender=gender, region="asia", limit=30)
            if len(asian) > 0:
                _show_rivals_table(asian)
            else:
                st.info("No Asian rival data available for this event.")

    with tab_g:
        global_performers = dc.get_top_performers(
            event=event, gender=gender, limit=30,
        )
        if len(global_performers) > 0:
            st.caption(f"Top {len(global_performers)} {gender_label} from recent competition data")
            _show_performers_table(global_performers)
        else:
            glob_rivals = dc.get_rivals(event=event, gender=gender, region="global", limit=30)
            if len(glob_rivals) > 0:
                _show_rivals_table(glob_rivals)
            else:
                st.info("No global rival data.")

# ── Head-to-Head Tab ──────────────────────────────────────────────────

with tab_h2h:
    render_section_header(
        "Head-to-Head Builder",
        "Compare two athletes directly"
    )

    # Build name lists from KSA athletes + rivals
    h2h_names_ksa = []
    ksa_athletes = dc.get_ksa_athletes()
    if len(ksa_athletes) > 0:
        h2h_names_ksa = sorted(ksa_athletes["full_name"].dropna().unique().tolist())

    h2h_names_rivals = []
    rivals_all = dc.get_rivals(event=event, gender=gender, limit=50)
    if len(rivals_all) > 0 and "full_name" in rivals_all.columns:
        h2h_names_rivals = sorted(rivals_all["full_name"].dropna().unique().tolist())

    # Event filter for H2H
    h2h_event = st.selectbox("H2H Event", [event], key="h2h_event", disabled=True)
    st.caption(f"Comparing results in **{event}** — change event in the main filter above")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**KSA Athlete**")
        athlete1 = st.selectbox(
            "Athlete 1", h2h_names_ksa,
            key="h2h_a1",
            label_visibility="collapsed",
        ) if h2h_names_ksa else None
    with col2:
        st.markdown(f"**Rival / Competitor**")
        all_rival_names = h2h_names_rivals + [n for n in h2h_names_ksa if n not in h2h_names_rivals]
        athlete2 = st.selectbox(
            "Athlete 2", all_rival_names,
            key="h2h_a2",
            label_visibility="collapsed",
        ) if all_rival_names else None

    if athlete1 and athlete2:
        st.markdown(f"### {athlete1} vs {athlete2}")

        # Get results for both athletes filtered to the selected event
        results_1 = dc.get_ksa_results(athlete_name=athlete1, discipline=event, limit=20)
        results_2 = dc.get_ksa_results(athlete_name=athlete2, discipline=event, limit=20)

        if len(results_1) > 0 or len(results_2) > 0:
            c1, c2 = st.columns(2)
            with c1:
                if len(results_1) > 0:
                    st.markdown(f"**{athlete1} - Recent {event} Results**")
                    mark_col = "mark" if "mark" in results_1.columns else "result" if "result" in results_1.columns else None
                    date_col = "date" if "date" in results_1.columns else None
                    venue_col = "competition" if "competition" in results_1.columns else "venue" if "venue" in results_1.columns else None
                    if mark_col and date_col:
                        display = [date_col, mark_col]
                        if venue_col:
                            display.append(venue_col)
                        st.dataframe(results_1[display], hide_index=True)
                else:
                    st.info(f"No {event} results found for {athlete1}")
            with c2:
                if len(results_2) > 0:
                    st.markdown(f"**{athlete2} - Recent {event} Results**")
                    mark_col = "mark" if "mark" in results_2.columns else "result" if "result" in results_2.columns else None
                    date_col = "date" if "date" in results_2.columns else None
                    venue_col = "competition" if "competition" in results_2.columns else "venue" if "venue" in results_2.columns else None
                    if mark_col and date_col:
                        display = [date_col, mark_col]
                        if venue_col:
                            display.append(venue_col)
                        st.dataframe(results_2[display], hide_index=True)
                else:
                    st.info(f"No {event} results found for {athlete2}")

            # Quick comparison summary
            st.markdown("---")
            from data.event_utils import format_event_name as _fmt_h2h
            import pandas as pd

            def _best_mark(res_df, lower_better):
                mc = "mark" if "mark" in res_df.columns else "result"
                if mc not in res_df.columns:
                    return None
                nums = pd.to_numeric(res_df[mc], errors="coerce").dropna()
                if len(nums) == 0:
                    return None
                return nums.min() if lower_better else nums.max()

            from data.event_utils import get_event_type
            et = get_event_type(event)
            lib = et == "time"

            pb1 = _best_mark(results_1, lib) if len(results_1) > 0 else None
            pb2 = _best_mark(results_2, lib) if len(results_2) > 0 else None

            c1, c2, c3 = st.columns(3)
            with c1:
                from components.theme import render_metric_card
                render_metric_card(athlete1.split()[-1], f"{pb1:.2f}" if pb1 else "-", "excellent")
            with c2:
                if pb1 and pb2:
                    diff = pb1 - pb2
                    if lib:
                        leader = athlete1 if diff < 0 else athlete2
                    else:
                        leader = athlete1 if diff > 0 else athlete2
                    render_metric_card("Advantage", leader.split()[-1], "gold")
                    st.caption(f"Gap: {abs(diff):.2f}")
                else:
                    render_metric_card("Gap", "-", "neutral")
            with c3:
                render_metric_card(athlete2.split()[-1], f"{pb2:.2f}" if pb2 else "-", "excellent")
        else:
            st.info(f"No {event} results available for either athlete in the database.")

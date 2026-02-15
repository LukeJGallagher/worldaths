"""
Event Intelligence - THE main page.

User flow: Pick Event (100m) -> Pick Gender (Men) -> See everything:
- What It Takes to Win (medal/finals/heats standards)
- Championship-specific results
- KSA Athletes in this event
- Rival Intelligence (Asian + Global) with enriched data
- Season Toplist
"""

import datetime
import streamlit as st
import pandas as pd
from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar, TEAL_PRIMARY, GOLD_ACCENT,
)
from components.filters import event_gender_picker, age_category_filter
from data.connector import get_connector
from data.event_utils import (
    get_event_type, display_to_db, ASIAN_COUNTRY_CODES,
    format_event_name, normalize_event_for_match,
)

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Event Intelligence", page_icon="🎯", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header("Event Intelligence", "Deep dive into any athletics event")

dc = get_connector()

# ── Event, Gender & Championship Selection ───────────────────────────

col_event, col_champ, col_season = st.columns([3, 2, 1])

with col_event:
    event, gender = event_gender_picker(key_prefix="ei")

with col_champ:
    CHAMPIONSHIPS = {
        "All Competitions": {"championship_type": None, "label": "All"},
        "World Championships": {"championship_type": "World Championships", "label": "World Champs"},
        "Olympic Games": {"championship_type": "Olympic Games", "label": "Olympics"},
        "Asian Games": {"championship_type": "Asian Games", "label": "Asian Games"},
        "Diamond League": {"championship_type": None, "label": "Diamond League"},
    }
    selected_champ = st.selectbox(
        "Championship Filter", list(CHAMPIONSHIPS.keys()), key="ei_champ"
    )
    champ_config = CHAMPIONSHIPS[selected_champ]

with col_season:
    current_year = datetime.date.today().year
    season_options = ["All"] + [str(current_year - i) for i in range(15)]
    selected_season_str = st.selectbox("Season", season_options, index=0, key="ei_season")
    selected_season = int(selected_season_str) if selected_season_str != "All" else None

event_type = get_event_type(event)
lower_is_better = event_type == "time"
gender_label = "Men" if gender == "M" else "Women"
season_label = selected_season_str
champ_label = champ_config["label"]

st.markdown(
    f"### {event} {gender_label} — {champ_label}"
    + (f" ({season_label})" if selected_season else "")
)

# ── Section A: What It Takes to Win ───────────────────────────────────

render_section_header(
    "What It Takes to Win",
    f"Standards & toplist for {event} {gender_label}"
)

# Get season toplist (from master data)
toplist = dc.get_season_toplist(
    event, gender, season=selected_season, limit=100,
)

if len(toplist) > 0:
    mark_col = "mark" if "mark" in toplist.columns else "result" if "result" in toplist.columns else None

    if mark_col:
        marks = toplist[mark_col].tolist()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("World Leader", str(marks[0]) if marks else "N/A", "gold")
        with col2:
            render_metric_card("Top 8 (Finals)", str(marks[7]) if len(marks) > 7 else "N/A", "excellent")
        with col3:
            render_metric_card("Top 16 (Semis)", str(marks[15]) if len(marks) > 15 else "N/A", "good")
        with col4:
            render_metric_card("Top 32 (Heats)", str(marks[31]) if len(marks) > 31 else "N/A", "neutral")

    # Toplist table
    st.markdown(f"#### {season_label} Season Toplist")

    display_cols_v2 = ["place", "mark", "competitor", "country", "venue", "date"]
    display_cols_legacy = ["rank", "result", "competitor", "nat", "venue", "date"]

    available_cols = [c for c in display_cols_v2 if c in toplist.columns]
    if not available_cols or len(available_cols) < 3:
        available_cols = [c for c in display_cols_legacy if c in toplist.columns]

    country_col = "country" if "country" in toplist.columns else "nat" if "nat" in toplist.columns else None

    def highlight_ksa(row):
        if country_col and row.get(country_col) == "KSA":
            return ["background-color: rgba(35, 80, 50, 0.15)"] * len(row)
        return [""] * len(row)

    if available_cols:
        st.dataframe(
            toplist[available_cols].head(50).style.apply(highlight_ksa, axis=1),
            hide_index=True, height=400,
        )
else:
    st.info(f"No toplist data for {event} {season_label}. Try a different season or run scrapers.")

# ── Championship Standards (benchmarks.parquet) ─────────────────────

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
        st.markdown("#### Championship Standards")
        display_bench_cols = [
            "Event", "Gender", "Year", "Gold Standard", "Silver Standard",
            "Bronze Standard", "Final Standard (8th)", "Top 8 Average",
        ]
        available_bench = [c for c in display_bench_cols if c in event_benchmarks.columns]
        st.dataframe(event_benchmarks[available_bench], hide_index=True)

# ── Championship Results (filtered by competition keywords) ──────────

if champ_config["championship_type"]:
    render_section_header(
        f"{champ_label} Results",
        f"Historical {champ_label} results for {event} {gender_label}"
    )

    champ_results = dc.get_championship_results(
        event=event,
        gender=gender,
        championship_type=champ_config["championship_type"],
        finals_only=True,
        limit=200,
    )

    if len(champ_results) > 0:
        # Show medal standards from championship data
        mark_col = "result_numeric" if "result_numeric" in champ_results.columns else None
        if mark_col:
            cm_sorted = champ_results.sort_values(mark_col, ascending=lower_is_better)
            result_col = "result" if "result" in cm_sorted.columns else mark_col

            # Medal line stats
            golds = cm_sorted[cm_sorted["pos"].astype(str).str.strip() == "1"]
            silvers = cm_sorted[cm_sorted["pos"].astype(str).str.strip() == "2"]
            bronzes = cm_sorted[cm_sorted["pos"].astype(str).str.strip() == "3"]

            col1, col2, col3 = st.columns(3)
            with col1:
                gold_avg = golds[result_col].iloc[0] if len(golds) > 0 else "N/A"
                render_metric_card(f"Latest {champ_label} Gold", str(gold_avg), "gold")
            with col2:
                silver_avg = silvers[result_col].iloc[0] if len(silvers) > 0 else "N/A"
                render_metric_card(f"Latest {champ_label} Silver", str(silver_avg), "neutral")
            with col3:
                bronze_avg = bronzes[result_col].iloc[0] if len(bronzes) > 0 else "N/A"
                render_metric_card(f"Latest {champ_label} Bronze", str(bronze_avg), "neutral")

        # Results table
        display_cols = ["competitor", "nat", "result", "pos", "venue", "date", "year"]
        available_cols = [c for c in display_cols if c in champ_results.columns]

        def highlight_medals(row):
            pos = str(row.get("pos", "")).strip()
            if pos == "1":
                return ["background-color: rgba(160, 142, 102, 0.25)"] * len(row)
            elif pos == "2":
                return ["background-color: rgba(192, 192, 192, 0.2)"] * len(row)
            elif pos == "3":
                return ["background-color: rgba(205, 127, 50, 0.2)"] * len(row)
            return [""] * len(row)

        if available_cols:
            st.dataframe(
                champ_results[available_cols].head(50).style.apply(highlight_medals, axis=1),
                hide_index=True, height=400,
            )
    else:
        st.info(f"No {champ_label} results found for {event}.")

# ── Section B: KSA Athletes in This Event ─────────────────────────────

render_section_header("KSA Athletes", f"Saudi athletes competing in {event}")

ksa_rankings = dc.get_ksa_rankings(event)

if len(ksa_rankings) > 0:
    for _, athlete in ksa_rankings.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                name = athlete.get("athlete", athlete.get("competitor", "Unknown"))
                st.markdown(f"**{name}**")
            with col2:
                rank_val = athlete.get("rank", "N/A")
                render_metric_card("Rank", f"#{rank_val}", "excellent")
            with col3:
                score = athlete.get("ranking_score", athlete.get("resultscore", 0))
                score_str = f"{score:.0f}" if score and score > 0 else "N/A"
                render_metric_card("Score", score_str, "good")
            with col4:
                comps = athlete.get("competitions_scored", "N/A")
                render_metric_card("Comps", str(comps), "neutral")

    # Recent KSA results
    ksa_results = dc.get_ksa_results(discipline=event, limit=50)
    if len(ksa_results) > 0:
        st.markdown("#### Recent KSA Results")
        display_cols_v2 = ["full_name", "date", "competition", "mark", "place", "venue"]
        display_cols_legacy = ["competitor", "date", "event", "result", "pos", "venue"]

        available_cols = [c for c in display_cols_v2 if c in ksa_results.columns]
        if not available_cols or len(available_cols) < 3:
            available_cols = [c for c in display_cols_legacy if c in ksa_results.columns]

        if available_cols:
            date_col = "date" if "date" in ksa_results.columns else None
            if date_col:
                ksa_results["_parsed_date"] = pd.to_datetime(
                    ksa_results[date_col], format="mixed", errors="coerce"
                )
                ksa_results = ksa_results.sort_values(
                    "_parsed_date", ascending=False
                ).drop(columns=["_parsed_date"])
            st.dataframe(ksa_results[available_cols], hide_index=True)
else:
    # Try PBs
    ksa_pbs = dc.get_ksa_athlete_pbs()
    if len(ksa_pbs) > 0:
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
        st.info("No KSA data loaded. Run: `python -m scrapers.scrape_athletes`")

# ── Section C: Rival Intelligence ─────────────────────────────────────

render_section_header(
    "Rival Intelligence",
    f"Top competitors in {event} {gender_label} — with PBs, SBs & recent form"
)

tab_asian, tab_global = st.tabs(["Asian Rivals", "Global Rivals"])

# Enriched display columns
RIVAL_DISPLAY_COLS = [
    "full_name", "country_code", "world_rank", "ranking_score",
    "pb_mark", "sb_mark", "latest_mark", "latest_date",
    "best5_avg", "performances_count",
]
RIVAL_COL_LABELS = {
    "full_name": "Athlete",
    "country_code": "Nat",
    "world_rank": "World Rank",
    "ranking_score": "Score",
    "pb_mark": "PB",
    "sb_mark": "SB",
    "latest_mark": "Latest",
    "latest_date": "Date",
    "best5_avg": "Best 5 Avg",
    "performances_count": "# Perfs",
}


def _show_rivals(rivals_df: pd.DataFrame):
    """Display rivals table with enriched data."""
    if rivals_df.empty:
        st.info("No rival data. Run: `python -m scrapers.scrape_rival_profiles`")
        return

    available = [c for c in RIVAL_DISPLAY_COLS if c in rivals_df.columns]
    if not available:
        st.dataframe(rivals_df, hide_index=True)
        return

    display_df = rivals_df[available].copy()
    display_df = display_df.rename(columns={c: RIVAL_COL_LABELS.get(c, c) for c in available})

    def highlight_ksa_rival(row):
        nat = row.get("Nat", "")
        if str(nat).upper() in ("KSA", "SAU"):
            return ["background-color: rgba(35, 80, 50, 0.15); font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_ksa_rival, axis=1),
        hide_index=True,
        height=min(400, 40 + len(display_df) * 35),
    )


with tab_asian:
    asian_rivals = dc.get_rivals(event=event, gender=gender, region="asia", limit=30)
    if len(asian_rivals) > 0:
        _show_rivals(asian_rivals)
    else:
        # Fallback to world rankings filtered by Asian countries
        world_ranks = dc.get_world_rankings(event=event, gender=gender, limit=200)
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
        _show_rivals(global_rivals)
    else:
        world_ranks = dc.get_world_rankings(event=event, gender=gender, limit=30)
        if len(world_ranks) > 0:
            st.dataframe(world_ranks.head(20), hide_index=True)
        else:
            st.info("No global rival data loaded.")

# ── Section D: World Rankings ─────────────────────────────────────────

render_section_header(
    "World Rankings",
    f"Current WA world rankings for {event} {gender_label}"
)

world_rankings = dc.get_world_rankings(event=event, gender=gender, limit=50)
if len(world_rankings) > 0:
    if "rank_date" in world_rankings.columns:
        rd = str(world_rankings["rank_date"].iloc[0])[:10]
        st.caption(f"Rankings as of: {rd}")

    display_cols_v2 = ["rank", "athlete", "country", "ranking_score", "competitions_scored"]
    display_cols_legacy = ["rank", "competitor", "nat", "resultscore", "event"]

    available_cols = [c for c in display_cols_v2 if c in world_rankings.columns]
    if not available_cols or len(available_cols) < 3:
        available_cols = [c for c in display_cols_legacy if c in world_rankings.columns]

    country_col = "country" if "country" in world_rankings.columns else "nat" if "nat" in world_rankings.columns else None

    def highlight_ksa_rank(row):
        if country_col and row.get(country_col) == "KSA":
            return ["background-color: rgba(35, 80, 50, 0.15); font-weight: bold"] * len(row)
        elif country_col and row.get(country_col) in ASIAN_COUNTRY_CODES:
            return ["background-color: rgba(160, 142, 102, 0.1)"] * len(row)
        return [""] * len(row)

    if available_cols:
        st.dataframe(
            world_rankings[available_cols].style.apply(highlight_ksa_rank, axis=1),
            hide_index=True, height=500,
        )
else:
    st.info(f"No ranking data for {event}. Run: `python -m scrapers.scrape_rankings`")

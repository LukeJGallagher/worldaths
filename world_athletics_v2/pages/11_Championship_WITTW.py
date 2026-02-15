"""
Championship What It Takes to Win - Historical performance standards.

Shows medal/finals/semi standards from championship results data,
plus year-by-year finals trends and KSA championship results.
Standards are computed dynamically from the selected championship.
"""

import streamlit as st
import pandas as pd
import re

from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar, TEAL_PRIMARY, GOLD_ACCENT,
    TEAL_LIGHT, GRAY_BLUE, HEADER_GRADIENT,
)
from components.charts import (
    championship_trends_chart, place_distribution_chart, standards_waterfall,
)
from components.filters import event_gender_picker
from data.connector import get_connector
from data.event_utils import (
    get_event_type, format_event_name, normalize_event_for_match,
    display_to_db,
)

from analytics.standards import get_finals_summary_by_place, get_standards_by_year


# ── Page Setup ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="What It Takes to Win - Team Saudi",
    page_icon="🏅",
    layout="wide",
)
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

dc = get_connector()

render_page_header(
    "What It Takes to Win",
    "Championship standards and historical final performance analysis",
)

# ── Constants ─────────────────────────────────────────────────────────────

PLACE_LABELS = {
    1: "1st", 2: "2nd", 3: "3rd", 4: "4th",
    5: "5th", 6: "6th", 7: "7th", 8: "8th",
}

CHAMPIONSHIP_OPTIONS = [
    "All Major Championships",
    "Asian Games",
    "World Championships",
    "Olympic Games",
]

# Keywords to match competition names in ksa_results.parquet (v2 scraped data)
KSA_CHAMPIONSHIP_KEYWORDS = {
    "All Major Championships": None,
    "Asian Games": ["Asian Games", "Asian Athletics", "Asian Championships"],
    "World Championships": ["World Athletics Championships", "World Championships"],
    "Olympic Games": ["Olympic"],
}

# Championship type keys for master.parquet venue+year lookup
MASTER_CHAMPIONSHIP_TYPE = {
    "All Major Championships": None,
    "Asian Games": "Asian Games",
    "World Championships": "World Championships",
    "Olympic Games": "Olympic Games",
}


# ── Styling Helpers ───────────────────────────────────────────────────────

def format_mark(value, event_type: str) -> str:
    """Format a numeric mark for display based on event type."""
    if pd.isna(value):
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return str(value)
    if event_type == "time":
        if value >= 60:
            minutes = int(value // 60)
            seconds = value - (minutes * 60)
            return f"{minutes}:{seconds:05.2f}"
        return f"{value:.2f}"
    return f"{value:.2f}"


# ── Filters Row ───────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([3, 1, 2])

with col1:
    event, gender = event_gender_picker(key_prefix="cw")

with col3:
    championship_type = st.selectbox(
        "Championship",
        CHAMPIONSHIP_OPTIONS,
        key="cw_championship",
    )

event_type = get_event_type(event)
lower_is_better = event_type == "time"
gender_label = "Men" if gender == "M" else "Women"
gender_legacy = "men" if gender == "M" else "women"

st.markdown(
    f"<p style='color: {GRAY_BLUE}; font-size: 0.95rem; margin: 0.25rem 0 1rem 0;'>"
    f"Showing <strong>{event} {gender_label}</strong> &mdash; {championship_type}</p>",
    unsafe_allow_html=True,
)

# ── Load championship results FIRST (used by both Section 1 and 2) ───────

champ_type_for_master = MASTER_CHAMPIONSHIP_TYPE.get(championship_type)
results = dc.get_championship_results(
    event=event,
    gender=gender,
    finals_only=True,
    championship_type=champ_type_for_master,
    limit=10000,
)

# Fall back to ALL finals if championship-specific filter returned nothing
_fell_back = False
if len(results) == 0 and champ_type_for_master:
    results = dc.get_championship_results(
        event=event,
        gender=gender,
        finals_only=True,
        championship_type=None,
        limit=10000,
    )
    if len(results) > 0:
        _fell_back = True


# ── Section 1: Championship Standards ─────────────────────────────────────
# When a specific championship is selected, compute standards DYNAMICALLY
# from the filtered results. Otherwise use static benchmarks as fallback.

render_section_header(
    "Championship Standards",
    f"Medal and finals benchmarks for {event} {gender_label} — {championship_type}",
)

# Try to compute dynamic standards from filtered results
_dynamic_standards = {}
if len(results) > 0:
    results["_pos_num"] = pd.to_numeric(results["pos"], errors="coerce")
    finals_8 = results[results["_pos_num"].between(1, 8)]

    if len(finals_8) > 0:
        gold_marks = finals_8[finals_8["_pos_num"] == 1]["result_numeric"]
        silver_marks = finals_8[finals_8["_pos_num"] == 2]["result_numeric"]
        bronze_marks = finals_8[finals_8["_pos_num"] == 3]["result_numeric"]
        eighth_marks = finals_8[finals_8["_pos_num"] == 8]["result_numeric"]
        top8_marks = finals_8["result_numeric"]

        best_fn = "min" if lower_is_better else "max"

        if len(gold_marks) > 0:
            _dynamic_standards["gold_avg"] = gold_marks.mean()
            _dynamic_standards["gold_best"] = getattr(gold_marks, best_fn)()
        if len(silver_marks) > 0:
            _dynamic_standards["silver_avg"] = silver_marks.mean()
        if len(bronze_marks) > 0:
            _dynamic_standards["bronze_avg"] = bronze_marks.mean()
            _dynamic_standards["bronze_best"] = getattr(bronze_marks, best_fn)()
        if len(eighth_marks) > 0:
            _dynamic_standards["eighth_avg"] = eighth_marks.mean()
        if len(top8_marks) > 0:
            _dynamic_standards["top8_avg"] = top8_marks.mean()

        _dynamic_standards["n_results"] = len(finals_8)

    results = results.drop(columns=["_pos_num"])

# Display standards: prefer dynamic from filtered results
if _dynamic_standards:
    source = championship_type if champ_type_for_master else "all competitions"
    if _fell_back:
        st.caption(
            f"No results matched '{championship_type}' host venues. "
            f"Showing standards from **all** finals data."
        )
    else:
        st.caption(f"Standards computed from {_dynamic_standards.get('n_results', 0)} "
                   f"finals results at {source}")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        val = format_mark(_dynamic_standards.get("gold_avg"), event_type)
        render_metric_card("Gold Standard", val, "gold")
    with col_m2:
        val = format_mark(_dynamic_standards.get("bronze_avg"), event_type)
        render_metric_card("Bronze Standard", val, "excellent")
    with col_m3:
        val = format_mark(_dynamic_standards.get("eighth_avg"), event_type)
        render_metric_card("Finals (8th)", val, "good")
    with col_m4:
        val = format_mark(_dynamic_standards.get("top8_avg"), event_type)
        render_metric_card("Top 8 Average", val, "neutral")

    st.markdown("")

    # Standards detail table
    detail_data = {
        "Gold (avg)": format_mark(_dynamic_standards.get("gold_avg"), event_type),
        "Silver (avg)": format_mark(_dynamic_standards.get("silver_avg"), event_type),
        "Bronze (avg)": format_mark(_dynamic_standards.get("bronze_avg"), event_type),
        "Finals 8th (avg)": format_mark(_dynamic_standards.get("eighth_avg"), event_type),
        "Top 8 Average": format_mark(_dynamic_standards.get("top8_avg"), event_type),
        "Sample Size": str(_dynamic_standards.get("n_results", 0)),
    }
    detail_df = pd.DataFrame([detail_data])

    def medal_style(row):
        return [f"background-color: rgba(160, 142, 102, 0.15)"] * len(row)

    st.dataframe(detail_df.style.apply(medal_style, axis=1), hide_index=True)

    # Standards waterfall chart
    standards_dict = {}
    if "eighth_avg" in _dynamic_standards:
        standards_dict["Finals (8th place)"] = _dynamic_standards["eighth_avg"]
    if "bronze_avg" in _dynamic_standards:
        standards_dict["Bronze Medal"] = _dynamic_standards["bronze_avg"]
    if "silver_avg" in _dynamic_standards:
        standards_dict["Silver Medal"] = _dynamic_standards["silver_avg"]
    if "gold_avg" in _dynamic_standards:
        standards_dict["Gold Medal"] = _dynamic_standards["gold_avg"]

    if len(standards_dict) >= 2:
        fig_waterfall = standards_waterfall(
            standards_dict,
            title=f"{event} {gender_label} - {source} Standards",
            lower_is_better=lower_is_better,
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

else:
    # Fallback to static benchmarks
    benchmarks = dc.get_benchmarks()
    event_db = display_to_db(event)
    event_norm = normalize_event_for_match(event_db)

    event_benchmarks = pd.DataFrame()
    if len(benchmarks) > 0:
        event_benchmarks = benchmarks[
            benchmarks.apply(
                lambda row: (
                    normalize_event_for_match(str(row.get("Event", ""))) == event_norm
                    and str(row.get("Gender", "")).lower() == gender_legacy
                ),
                axis=1,
            )
        ]

    if len(event_benchmarks) > 0:
        st.caption("Using static benchmark standards (no championship-specific data available)")
        row = event_benchmarks.iloc[0]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            render_metric_card("Gold Standard", str(row.get("Gold Standard", "N/A")), "gold")
        with col_m2:
            render_metric_card("Bronze Standard", str(row.get("Bronze Standard", "N/A")), "excellent")
        with col_m3:
            render_metric_card("Finals (8th)", str(row.get("Final Standard (8th)", "N/A")), "good")
        with col_m4:
            render_metric_card("Top 8 Average", str(row.get("Top 8 Average", "N/A")), "neutral")
    else:
        st.info(
            f"No standards available for {event} {gender_label}. "
            f"No finals data found in the database for this event."
        )


# ── Section 2: Year-by-Year Finals Trends ─────────────────────────────────

render_section_header(
    "Finals Performance by Year",
    f"Year-by-year best finishing marks — {championship_type}",
)

if len(results) > 0:
    source_label = championship_type if champ_type_for_master else "all competitions"
    st.caption(f"{len(results):,} finals results from {source_label}")

    # Year-by-year trend chart
    trend_data = get_standards_by_year(
        results,
        max_place=6,
        lower_is_better=lower_is_better,
    )

    if len(trend_data) > 0:
        fig = championship_trends_chart(
            trend_data,
            title=f"{event} {gender_label} - Finals Performance by Place",
            lower_is_better=lower_is_better,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View trend data"):
            pivot_data = trend_data.copy()
            pivot_data["place_label"] = pivot_data["place"].map(PLACE_LABELS)

            pivot_table = pivot_data.pivot_table(
                index="year",
                columns="place_label",
                values="mark",
                aggfunc="first",
            )
            pivot_table = pivot_table.sort_index(ascending=False)

            for col in pivot_table.columns:
                pivot_table[col] = pivot_table[col].apply(
                    lambda v: format_mark(v, event_type)
                )
            st.dataframe(pivot_table, height=300)
    else:
        st.info("Insufficient data for year-by-year trends.")

    # Finals summary by place
    render_section_header(
        "Finals Summary by Place",
        f"Aggregate finishing position analysis",
    )

    summary = get_finals_summary_by_place(
        results,
        max_place=8,
        lower_is_better=lower_is_better,
    )

    if len(summary) > 0:
        display_summary = summary.copy()
        display_summary["place_label"] = display_summary["place"].map(PLACE_LABELS)

        for col in ["avg_mark", "fastest", "slowest"]:
            if col in display_summary.columns:
                display_summary[col] = display_summary[col].apply(
                    lambda v: format_mark(v, event_type)
                )

        display_cols = ["place_label", "avg_mark", "fastest", "slowest", "n_results"]
        available_cols = [c for c in display_cols if c in display_summary.columns]

        rename_map = {
            "place_label": "Place",
            "avg_mark": "Average Mark",
            "fastest": "Best",
            "slowest": "Slowest",
            "n_results": "# Results",
        }

        styled_df = display_summary[available_cols].rename(columns=rename_map)

        def apply_medal_style(row):
            idx = row.name
            if idx < len(summary):
                place_val = summary.iloc[idx]["place"]
                if place_val == 1:
                    return [f"background-color: rgba(160, 142, 102, 0.3)"] * len(row)
                elif place_val == 2:
                    return [f"background-color: rgba(192, 192, 192, 0.3)"] * len(row)
                elif place_val == 3:
                    return [f"background-color: rgba(205, 127, 50, 0.3)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            styled_df.style.apply(apply_medal_style, axis=1),
            hide_index=True,
            height=350,
        )

    # Place distribution chart
    render_section_header(
        "Place Distribution",
        "Frequency of finishing positions",
    )

    fig_dist = place_distribution_chart(results, country="KSA")
    st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.warning(
        f"No finals data found for {event} {gender_label} in the master database. "
        f"This event may not be in the legacy data."
    )

# ── Section: KSA Championship Results (from v2 ksa_results) ──────────────

render_section_header(
    f"Team Saudi at {championship_type}",
    f"KSA athlete results filtered by championship",
)

# Query v2 ksa_results (has competition names for proper championship filtering)
ksa_champ_results = dc.get_ksa_results(discipline=event, limit=500)

# If v2 returned data but without 'competition' column (legacy fallback), add it from venue
if len(ksa_champ_results) > 0 and "competition" not in ksa_champ_results.columns:
    if "venue" in ksa_champ_results.columns:
        ksa_champ_results = ksa_champ_results.copy()
        ksa_champ_results["competition"] = ksa_champ_results["venue"]
    if "competitor" in ksa_champ_results.columns:
        ksa_champ_results = ksa_champ_results.rename(columns={
            "competitor": "full_name", "nat": "country_code",
            "result": "mark", "pos": "place",
        })

# Also try master data for historical KSA results
master_ksa_champ = dc.get_championship_results(
    event=event, gender=gender, country="KSA",
    championship_type=champ_type_for_master,
    finals_only=False, limit=500,
)
if len(master_ksa_champ) > 0 and "competitor" in master_ksa_champ.columns:
    master_renamed = master_ksa_champ.rename(columns={
        "competitor": "full_name", "nat": "country_code",
        "result": "mark", "pos": "place",
    })
    if "venue" in master_renamed.columns:
        master_renamed["competition"] = master_renamed["venue"]
    if len(ksa_champ_results) > 0:
        ksa_champ_results = pd.concat([ksa_champ_results, master_renamed], ignore_index=True)
        ksa_champ_results = ksa_champ_results.drop_duplicates(
            subset=["full_name", "date", "mark"], keep="first"
        )
    else:
        ksa_champ_results = master_renamed

if len(ksa_champ_results) > 0 and "competition" in ksa_champ_results.columns:
    champ_keywords = KSA_CHAMPIONSHIP_KEYWORDS.get(championship_type)

    if champ_keywords:
        mask = ksa_champ_results["competition"].apply(
            lambda c: any(kw.lower() in str(c).lower() for kw in champ_keywords)
            if pd.notna(c) else False
        )
        ksa_filtered = ksa_champ_results[mask]
    else:
        ksa_filtered = ksa_champ_results

    if len(ksa_filtered) > 0:
        st.success(
            f"Found **{len(ksa_filtered)}** KSA result(s) in {event} "
            f"at {championship_type}."
        )

        display_cols = ["full_name", "date", "competition", "mark", "place", "venue"]
        available_cols = [c for c in display_cols if c in ksa_filtered.columns]

        if available_cols:
            if "date" in ksa_filtered.columns:
                ksa_filtered = ksa_filtered.copy()
                ksa_filtered["_parsed_date"] = pd.to_datetime(
                    ksa_filtered["date"], format="mixed", errors="coerce"
                )
                ksa_filtered = ksa_filtered.sort_values("_parsed_date", ascending=False)
                ksa_filtered = ksa_filtered.drop(columns=["_parsed_date"])

            rename_map = {
                "full_name": "Athlete",
                "date": "Date",
                "competition": "Competition",
                "mark": "Mark",
                "place": "Place",
                "venue": "Venue",
            }

            display_df = ksa_filtered[available_cols].rename(
                columns={c: rename_map.get(c, c) for c in available_cols}
            )
            st.dataframe(display_df, hide_index=True, height=300)

            if "place" in ksa_filtered.columns:
                places = pd.to_numeric(
                    ksa_filtered["place"].astype(str).str.strip(), errors="coerce"
                ).dropna()
                if len(places) > 0:
                    gold_count = int((places == 1).sum())
                    silver_count = int((places == 2).sum())
                    bronze_count = int((places == 3).sum())
                    total_medals = gold_count + silver_count + bronze_count

                    if total_medals > 0:
                        col_g, col_s, col_b = st.columns(3)
                        with col_g:
                            render_metric_card("Gold", str(gold_count), "gold")
                        with col_s:
                            render_metric_card("Silver", str(silver_count), "neutral")
                        with col_b:
                            render_metric_card("Bronze", str(bronze_count), "warning")
    else:
        if champ_keywords:
            st.info(
                f"No KSA results found for {event} at {championship_type}. "
                f"Try selecting 'All Major Championships' to see all results."
            )
        else:
            st.info(f"No KSA results found for {event}.")

elif len(results) > 0:
    ksa_finals = (
        results[results["nat"].str.upper() == "KSA"]
        if "nat" in results.columns
        else pd.DataFrame()
    )

    if len(ksa_finals) > 0:
        st.success(
            f"Found **{len(ksa_finals)}** KSA final appearance(s) in {event} {gender_label}."
        )

        display_cols_priority = [
            "pos", "competitor", "result", "venue", "date", "year",
        ]
        available_cols = [c for c in display_cols_priority if c in ksa_finals.columns]
        rename_map = {
            "pos": "Place", "competitor": "Athlete", "result": "Mark",
            "venue": "Venue", "date": "Date", "year": "Year",
        }
        ksa_display = ksa_finals[available_cols].copy().rename(
            columns={c: rename_map.get(c, c) for c in available_cols}
        )
        if "Year" in ksa_display.columns:
            ksa_display = ksa_display.sort_values("Year", ascending=False)
        st.dataframe(ksa_display, hide_index=True, height=300)
    else:
        st.info(
            f"No KSA athletes found in finals for {event} {gender_label}. "
            f"This is an opportunity for growth!"
        )
else:
    st.info(f"No KSA data available for {event}.")

# ── Footer ────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    f"<p style='color: {GRAY_BLUE}; font-size: 0.8rem; text-align: center;'>"
    f"Standards computed dynamically from championship finals (positions 1-8). "
    f"Championship filtering uses known host venue + year matching. "
    f"KSA results from v2 scraped data with competition name filtering. "
    f"</p>",
    unsafe_allow_html=True,
)

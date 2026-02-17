"""
Competitor Analysis - KSA athlete-centric competitive landscape.

Features:
- Select a KSA athlete and see their competitive position
- Top rivals filtered by championship scope (Asian / Global)
- Recent results with progression chart
- Race intelligence with form score and trend
"""

import datetime
from typing import Optional

import streamlit as st
import pandas as pd


def _safe(val):
    """Safely check a value that might be pd.NA, None, NaN, etc."""
    try:
        return pd.notna(val) and str(val) not in ("None", "nan", "NaT", "")
    except (ValueError, TypeError):
        return False


from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar, TEAL_PRIMARY, GOLD_ACCENT,
    TEAL_DARK, TEAL_LIGHT, GRAY_BLUE, HEADER_GRADIENT,
)
from components.charts import progression_chart, form_gauge
from components.filters import event_gender_picker, championship_selector
from data.connector import get_connector
from data.event_utils import get_event_type, format_event_name, ASIAN_COUNTRY_CODES
from analytics.form_engine import calculate_form_score, detect_trend

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Competitor Analysis - Team Saudi", page_icon="🎯", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

dc = get_connector()

render_page_header("Competitor Analysis", "KSA athlete competitive landscape")

# ── Constants ─────────────────────────────────────────────────────────

COUNTRY_FLAGS = {
    "KSA": "\U0001f1f8\U0001f1e6", "QAT": "\U0001f1f6\U0001f1e6",
    "BRN": "\U0001f1e7\U0001f1ed", "UAE": "\U0001f1e6\U0001f1ea",
    "KUW": "\U0001f1f0\U0001f1fc", "OMA": "\U0001f1f4\U0001f1f2",
    "JOR": "\U0001f1ef\U0001f1f4", "IRQ": "\U0001f1ee\U0001f1f6",
    "IND": "\U0001f1ee\U0001f1f3", "CHN": "\U0001f1e8\U0001f1f3",
    "JPN": "\U0001f1ef\U0001f1f5", "KOR": "\U0001f1f0\U0001f1f7",
    "USA": "\U0001f1fa\U0001f1f8", "GBR": "\U0001f1ec\U0001f1e7",
    "JAM": "\U0001f1ef\U0001f1f2", "KEN": "\U0001f1f0\U0001f1ea",
    "ETH": "\U0001f1ea\U0001f1f9", "RSA": "\U0001f1ff\U0001f1e6",
    "NGR": "\U0001f1f3\U0001f1ec", "THA": "\U0001f1f9\U0001f1ed",
    "MAS": "\U0001f1f2\U0001f1fe", "SGP": "\U0001f1f8\U0001f1ec",
    "PHI": "\U0001f1f5\U0001f1ed", "INA": "\U0001f1ee\U0001f1e9",
    "IRI": "\U0001f1ee\U0001f1f7", "PAK": "\U0001f1f5\U0001f1f0",
    "UZB": "\U0001f1fa\U0001f1ff", "KAZ": "\U0001f1f0\U0001f1ff",
    "TPE": "\U0001f1f9\U0001f1fc", "HKG": "\U0001f1ed\U0001f1f0",
    "GER": "\U0001f1e9\U0001f1ea", "FRA": "\U0001f1eb\U0001f1f7",
    "ITA": "\U0001f1ee\U0001f1f9", "ESP": "\U0001f1ea\U0001f1f8",
    "NED": "\U0001f1f3\U0001f1f1", "AUS": "\U0001f1e6\U0001f1fa",
    "CAN": "\U0001f1e8\U0001f1e6", "BRA": "\U0001f1e7\U0001f1f7",
    "CUB": "\U0001f1e8\U0001f1fa", "TTO": "\U0001f1f9\U0001f1f9",
}

# Mapping: display name (from event_gender_picker) -> rivals.parquet event format
DISPLAY_TO_RIVALS_EVENT = {
    "100m": "100m", "200m": "200m", "400m": "400m", "800m": "800m",
    "1500m": "1500m", "3000m": "3000m", "5000m": "5000m",
    "10,000m": "10000m", "Mile": "mile",
    "3000m SC": "3000msc",
    "100m H": "100mh", "110m H": "110mh", "400m H": "400mh",
    "High Jump": "high-jump", "Pole Vault": "pole-vault",
    "Long Jump": "long-jump", "Triple Jump": "triple-jump",
    "Shot Put": "shot-put", "Discus": "discus-throw",
    "Hammer": "hammer-throw", "Javelin": "javelin-throw",
    "Decathlon": "decathlon", "Heptathlon": "heptathlon",
    "Marathon": "marathon", "Half Marathon": "half-marathon",
    "20km Walk": "20km-walk", "35km Walk": "35km-walk",
    "4x100m": "4x100m", "4x400m": "4x400m",
}


def _get_flag(country_code: str) -> str:
    """Return flag emoji for a country code, or empty string."""
    return COUNTRY_FLAGS.get(country_code.upper().strip(), "")


def _match_rivals_event(display_event: str) -> str:
    """Convert display event name to rivals.parquet event format."""
    return DISPLAY_TO_RIVALS_EVENT.get(display_event, display_event.lower().replace(" ", "-"))


# ── Load Athletes ─────────────────────────────────────────────────────

df_athletes = dc.get_ksa_athletes()

if len(df_athletes) == 0:
    st.warning("No athlete data loaded. Run: `python -m scrapers.scrape_athletes`")
    st.stop()

# ── Filters Row ───────────────────────────────────────────────────────

col_athlete, col_event, col_gender, col_champ = st.columns([2, 2, 1, 2])

with col_athlete:
    athlete_names = df_athletes["full_name"].tolist()
    selected_name = st.selectbox("Select Athlete", athlete_names, key="ca_athlete")

# Get athlete row for primary event default
athlete = df_athletes[df_athletes["full_name"] == selected_name].iloc[0]
primary_event = athlete.get("primary_event", "100m")

with col_gender:
    # Detect gender from athlete data, default to Men
    _raw_gender = athlete.get("gender")
    _default_gender_idx = 0  # Men
    if _safe(_raw_gender) and str(_raw_gender).strip().upper() in ("F", "FEMALE", "WOMEN", "W"):
        _default_gender_idx = 1
    selected_gender = st.selectbox("Gender", ["Men", "Women"], index=_default_gender_idx, key="ca_gender")

with col_event:
    # Build event list from athlete PBs + primary event
    pbs = dc.get_ksa_athlete_pbs(selected_name)
    pb_event_col = "discipline" if "discipline" in pbs.columns else "event" if "event" in pbs.columns else None
    athlete_events = []
    if pb_event_col and len(pbs) > 0:
        athlete_events = sorted(pbs[pb_event_col].unique().tolist())

    if not athlete_events:
        athlete_events = [primary_event] if primary_event else ["100m"]

    # Format events for display
    display_events = []
    for ev in athlete_events:
        formatted = format_event_name(str(ev))
        if formatted not in display_events:
            display_events.append(formatted)

    # Determine default index
    primary_display = format_event_name(str(primary_event)) if primary_event else display_events[0]
    default_idx = display_events.index(primary_display) if primary_display in display_events else 0

    selected_event = st.selectbox("Event", display_events, index=default_idx, key="ca_event")

with col_champ:
    championship = championship_selector(key="ca_champ")

# Determine event type for lower_is_better logic
event_type = get_event_type(selected_event)
lower_is_better = event_type == "time"

# Gender from explicit selector
athlete_gender = "M" if selected_gender == "Men" else "F"
gender_label = selected_gender

# ── Section 1: Hero / Athlete Info ────────────────────────────────────

render_section_header("Athlete Overview", f"{selected_name} - competitive snapshot")

col_photo, col_info = st.columns([1, 3])

with col_photo:
    photo_url = athlete.get("photo_url", athlete.get("profile_image_url"))
    if pd.notna(photo_url) and str(photo_url) not in ("None", "nan", ""):
        st.image(photo_url, width=200)
    else:
        st.markdown(f"""
        <div style="width: 200px; height: 200px; background: {TEAL_PRIMARY};
             border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="color: white; font-size: 4rem; font-weight: bold;">
                {selected_name[0] if selected_name else "?"}
            </span>
        </div>
        """, unsafe_allow_html=True)

with col_info:
    country_code = athlete.get("country_code", "KSA")
    flag = _get_flag(str(country_code))
    st.markdown(f"### {flag} {selected_name}")

    info_parts = []
    country_name = athlete.get("country_name")
    if _safe(country_name):
        info_parts.append(f"**{country_name}** ({country_code})")
    elif _safe(country_code):
        info_parts.append(f"**{country_code}**")

    if _safe(primary_event):
        info_parts.append(f"Primary: **{format_event_name(str(primary_event))}**")

    birth_date = athlete.get("birth_date", athlete.get("date_of_birth"))
    if _safe(birth_date):
        info_parts.append(f"Born: {birth_date}")

    if info_parts:
        st.markdown(" | ".join(info_parts))

    # Key metric cards
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        # Season Best from PBs
        mark_col = "mark" if "mark" in pbs.columns else "result" if "result" in pbs.columns else None
        sb_val = "N/A"
        if mark_col and len(pbs) > 0:
            # Filter PBs for the selected event
            if pb_event_col:
                event_pbs = pbs[pbs[pb_event_col].apply(
                    lambda x: format_event_name(str(x)) == selected_event
                )]
                if len(event_pbs) > 0:
                    sb_val = str(event_pbs.iloc[0][mark_col])
        render_metric_card("PB", sb_val, "gold" if sb_val != "N/A" else "neutral")

    with m2:
        rank = athlete.get("best_world_rank")
        rank_display = f"#{rank}" if _safe(rank) else "N/R"
        render_metric_card(
            "World Rank", rank_display,
            "excellent" if _safe(rank) and int(rank) <= 50 else "good",
        )

    with m3:
        score = athlete.get("best_ranking_score", athlete.get("best_score"))
        if _safe(score):
            try:
                render_metric_card("WA Score", f"{float(score):.0f}", "good")
            except (ValueError, TypeError):
                render_metric_card("WA Score", str(score), "good")
        else:
            render_metric_card("WA Score", "N/A", "neutral")

    with m4:
        n_events = len(display_events)
        render_metric_card("Events", str(n_events), "good" if n_events > 1 else "neutral")

# ── Section 2: Competitors Table ──────────────────────────────────────

render_section_header("Competitors", f"Top rivals in {selected_event} {gender_label}")

# Determine region based on championship selection
region = None
if "Asian" in championship:
    region = "asia"

# Use get_top_performers() for accurate gender-filtered competitors from master data
# (rivals.parquet currently contains Women's data for all events due to API limitation)
country_codes = list(ASIAN_COUNTRY_CODES) if region == "asia" else None
rivals_df = dc.get_top_performers(
    event=selected_event, gender=athlete_gender,
    country_codes=country_codes, limit=30,
)
_using_master_performers = len(rivals_df) > 0

# Fall back to rivals.parquet if master data unavailable
if len(rivals_df) == 0:
    rivals_event = _match_rivals_event(selected_event)
    rivals_df = dc.get_rivals(event=rivals_event, gender=athlete_gender, region=region, limit=30)
    if len(rivals_df) == 0 and rivals_event != selected_event.lower():
        rivals_df = dc.get_rivals(event=selected_event.lower(), gender=athlete_gender, region=region, limit=30)
    _using_master_performers = False

if len(rivals_df) > 0:
    # Sort appropriately based on data source
    if _using_master_performers:
        # Already sorted by best_mark from get_top_performers()
        pass
    elif "world_rank" in rivals_df.columns:
        rivals_df = rivals_df.sort_values("world_rank", na_position="last")

    # Determine event type for formatting
    from data.event_utils import get_event_type as _get_et
    _etype = _get_et(selected_event)

    def _fmt_mark(val):
        """Format a numeric mark for display."""
        try:
            v = float(val)
            if v >= 60:
                mins = int(v // 60)
                secs = v - mins * 60
                return f"{mins}:{secs:05.2f}"
            return f"{v:.2f}"
        except (ValueError, TypeError):
            return str(val) if _safe(val) else "-"

    # Build display dataframe with all available columns
    display_data = []
    for rank_idx, (_, row) in enumerate(rivals_df.iterrows(), 1):
        nat = str(row.get("country_code", row.get("country", "")))
        flag = _get_flag(nat)

        if _using_master_performers:
            best_num = row.get("best_mark_numeric")
            entry = {
                "#": rank_idx,
                "Athlete": str(row.get("full_name", "")),
                "Nat": f"{flag} {nat}",
                "PB": _fmt_mark(row.get("pb_mark")) if _safe(row.get("pb_mark")) else (_fmt_mark(best_num) if _safe(best_num) else "-"),
                "SB": _fmt_mark(best_num) if _safe(best_num) else "-",
                "# Perfs": int(row.get("performances_count", 0)) if _safe(row.get("performances_count")) else None,
                "Latest Date": str(row.get("latest_date", ""))[:10] if _safe(row.get("latest_date")) else "-",
            }
        else:
            world_rank = row.get("world_rank")
            ranking_score = row.get("ranking_score")
            pb_val = row.get("pb_mark")
            sb_val = row.get("sb_mark")
            avg5 = row.get("best5_avg")
            latest = row.get("latest_mark")
            latest_date = row.get("latest_date")
            entry = {
                "#": int(world_rank) if _safe(world_rank) else rank_idx,
                "Athlete": str(row.get("full_name", row.get("athlete", ""))),
                "Nat": f"{flag} {nat}",
                "PB": str(pb_val) if _safe(pb_val) else "-",
                "SB": str(sb_val) if _safe(sb_val) else "-",
                "# Perfs": int(row.get("performances_count", 0)) if _safe(row.get("performances_count")) else None,
                "Latest Date": str(latest_date)[:11] if _safe(latest_date) else "-",
            }

        display_data.append(entry)

    display_df = pd.DataFrame(display_data)

    # Summary metrics
    total_rivals = len(display_df)
    asian_rivals = sum(1 for d in display_data if any(
        code in str(d["Nat"]) for code in ASIAN_COUNTRY_CODES
    ))
    if _using_master_performers:
        best_mark = rivals_df["best_mark_numeric"].iloc[0] if len(rivals_df) > 0 else None
        best_str = _fmt_mark(best_mark) if best_mark else "N/A"
    else:
        best_rank = rivals_df["world_rank"].min() if "world_rank" in rivals_df.columns else None
        best_str = f"#{int(best_rank)}" if best_rank else "N/A"

    sm1, sm2, sm3 = st.columns(3)
    with sm1:
        render_metric_card("Total Competitors", str(total_rivals), "neutral")
    with sm2:
        render_metric_card("Asian Competitors", str(asian_rivals), "good" if asian_rivals > 0 else "neutral")
    with sm3:
        label = "Best Mark" if _using_master_performers else "Best Rank"
        render_metric_card(label, best_str, "excellent")

    # Highlight KSA rows
    def highlight_ksa(row):
        if "KSA" in str(row.get("Nat", "")):
            return [f"background-color: rgba(35, 80, 50, 0.15); font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_ksa, axis=1),
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn("#", width="small"),
            "Athlete": st.column_config.TextColumn("Athlete", width="medium"),
            "Nat": st.column_config.TextColumn("Nat", width="small"),
            "PB": st.column_config.TextColumn("PB"),
            "SB": st.column_config.TextColumn("SB"),
            "# Perfs": st.column_config.NumberColumn("# Perfs", format="d"),
            "Latest Date": st.column_config.TextColumn("Latest Date"),
        },
        height=min(500, 35 * len(display_df) + 38),
    )

    if all(d.get("PB") == "-" for d in display_data):
        st.caption(
            "PB/SB marks not yet scraped. "
            "Run: `python -m scrapers.scrape_rival_profiles`"
        )
else:
    # Fallback: try to pull from world rankings and filter to region
    world_ranks = dc.get_world_rankings(event=selected_event, gender=athlete_gender, limit=50)
    if len(world_ranks) > 0:
        country_col = "country" if "country" in world_ranks.columns else "nat" if "nat" in world_ranks.columns else None
        if country_col and region == "asia":
            world_ranks = world_ranks[world_ranks[country_col].isin(ASIAN_COUNTRY_CODES)]

        if len(world_ranks) > 0:
            st.markdown(f"**{len(world_ranks)} competitors from world rankings**")
            st.dataframe(world_ranks.head(30), hide_index=True, height=400)
        else:
            st.info(f"No competitor data for {selected_event}. Run: `python -m scrapers.scrape_athletes --rivals`")
    else:
        st.info(f"No competitor data for {selected_event}. Run: `python -m scrapers.scrape_athletes --rivals`")

# ── Section 3: Athlete Recent Results ─────────────────────────────────

render_section_header("Recent Results", f"{selected_name}'s competition history")

results = dc.get_ksa_results(athlete_name=selected_name, discipline=selected_event, limit=20)

if len(results) > 0:
    # Handle both v2 and legacy column names
    display_cols_v2 = ["date", "competition", "venue", "mark", "place", "round", "category"]
    display_cols_legacy = ["date", "venue", "result", "pos", "event"]

    available_cols = [c for c in display_cols_v2 if c in results.columns]
    if not available_cols or len(available_cols) < 3:
        available_cols = [c for c in display_cols_legacy if c in results.columns]

    column_config = {}
    if "date" in available_cols:
        column_config["date"] = st.column_config.TextColumn("Date")
    if "competition" in available_cols:
        column_config["competition"] = st.column_config.TextColumn("Competition", width="large")
    if "venue" in available_cols:
        column_config["venue"] = st.column_config.TextColumn("Venue")
    if "mark" in available_cols:
        column_config["mark"] = st.column_config.TextColumn("Mark")
    elif "result" in available_cols:
        column_config["result"] = st.column_config.TextColumn("Mark")
    if "place" in available_cols:
        column_config["place"] = st.column_config.TextColumn("Place")
    elif "pos" in available_cols:
        column_config["pos"] = st.column_config.TextColumn("Place")
    if "round" in available_cols:
        column_config["round"] = st.column_config.TextColumn("Round")
    if "category" in available_cols:
        column_config["category"] = st.column_config.TextColumn("Category")

    st.dataframe(
        results[available_cols],
        hide_index=True,
        column_config=column_config,
        height=min(400, 35 * len(results) + 38),
    )

    # Progression chart from recent results
    mark_col = "mark" if "mark" in results.columns else "result" if "result" in results.columns else None
    if mark_col and "date" in results.columns:
        chart_df = results.copy()
        if "result_numeric" in chart_df.columns:
            chart_df["mark_numeric"] = pd.to_numeric(chart_df["result_numeric"], errors="coerce")
        else:
            chart_df["mark_numeric"] = pd.to_numeric(chart_df[mark_col], errors="coerce")
        chart_df = chart_df.dropna(subset=["mark_numeric"])

        if len(chart_df) > 1:
            # Filter outliers using median-based approach
            median_val = chart_df["mark_numeric"].median()
            if median_val > 0:
                chart_df = chart_df[
                    (chart_df["mark_numeric"] >= median_val * 0.2)
                    & (chart_df["mark_numeric"] <= median_val * 5.0)
                ]

            if len(chart_df) > 1:
                fig = progression_chart(
                    chart_df.sort_values("date"),
                    x_col="date",
                    y_col="mark_numeric",
                    title=f"Performance Progression - {selected_event}",
                    lower_is_better=lower_is_better,
                )
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No recent results found for {selected_name} in {selected_event}.")

# ── Section 4: Race Intelligence ──────────────────────────────────────

with st.expander("Race Intelligence", expanded=False):
    render_section_header("Form & Trend Analysis", "Current form assessment based on recent performances")

    # Gather numeric marks for form scoring
    mark_col = "mark" if "mark" in results.columns else "result" if "result" in results.columns else None if len(results) > 0 else None
    recent_marks = []
    pb_numeric = None

    if mark_col and len(results) > 0:
        numeric_marks = pd.to_numeric(
            results.get("result_numeric", results[mark_col]),
            errors="coerce",
        ).dropna().tolist()

        # Filter outlier marks
        if numeric_marks:
            med = pd.Series(numeric_marks).median()
            if med > 0:
                recent_marks = [m for m in numeric_marks if med * 0.2 <= m <= med * 5.0]

    # Get PB numeric value for the selected event
    if pb_event_col and mark_col and len(pbs) > 0:
        event_pbs = pbs[pbs[pb_event_col].apply(
            lambda x: format_event_name(str(x)) == selected_event
        )]
        if len(event_pbs) > 0:
            pb_mark_col = "mark" if "mark" in event_pbs.columns else "result" if "result" in event_pbs.columns else None
            if pb_mark_col:
                pb_numeric = pd.to_numeric(event_pbs.iloc[0].get(pb_mark_col), errors="coerce")
                if pd.isna(pb_numeric):
                    pb_numeric = None

    if len(recent_marks) >= 2:
        form_score = calculate_form_score(
            recent_marks, pb=pb_numeric, lower_is_better=lower_is_better
        )
        trend = detect_trend(recent_marks, lower_is_better=lower_is_better)

        fi_col1, fi_col2 = st.columns([1, 2])

        with fi_col1:
            fig_gauge = form_gauge(form_score, label="Current Form")
            st.plotly_chart(fig_gauge, use_container_width=True)

        with fi_col2:
            trend_icons = {
                "improving": "Improving - athlete trending toward better marks",
                "stable": "Stable - consistent recent performances",
                "declining": "Declining - recent marks dropping off",
            }
            trend_colors = {"improving": "excellent", "stable": "good", "declining": "warning"}

            render_metric_card("Trend", trend.title(), trend_colors.get(trend, "neutral"))
            st.caption(trend_icons.get(trend, ""))

            if pb_numeric and not pd.isna(pb_numeric):
                best_recent = min(recent_marks) if lower_is_better else max(recent_marks)
                gap_to_pb = abs(best_recent - pb_numeric)
                pct_of_pb = (pb_numeric / best_recent * 100) if lower_is_better and best_recent > 0 else (
                    best_recent / pb_numeric * 100 if pb_numeric > 0 else 0
                )
                st.markdown(
                    f"**Best recent mark:** {best_recent:.2f} | "
                    f"**PB:** {pb_numeric:.2f} | "
                    f"**Gap:** {gap_to_pb:.2f} ({pct_of_pb:.1f}% of PB)"
                )

            st.markdown(f"*Based on {len(recent_marks)} recent performances*")
    else:
        st.info(
            "Not enough recent numeric results to calculate form score. "
            "At least 2 results with valid marks are required."
        )

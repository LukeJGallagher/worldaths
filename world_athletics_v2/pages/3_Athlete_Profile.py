"""
Athlete Profile - Deep dive into individual athlete.

User flow: Pick Athlete -> See everything:
- Hero card with photo, PBs, ranking
- Multi-event summary
- Season progression
- Major championship history
- Competition results
"""

import pandas as pd
import streamlit as st
from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar, TEAL_PRIMARY, GOLD_ACCENT, TEAL_LIGHT,
)
from components.charts import progression_chart, form_gauge
from data.connector import get_connector
from data.event_utils import get_event_type, normalize_event_for_match


def _safe(val):
    """Safely check a value that might be pd.NA, None, NaN, etc."""
    try:
        return pd.notna(val) and str(val) not in ("None", "nan", "NaT", "")
    except (ValueError, TypeError):
        return False


# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Athlete Profile", page_icon="🏃", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header("Athlete Profile", "Deep dive into individual athlete performance")

dc = get_connector()

# ── Athlete Selector ──────────────────────────────────────────────────

df_athletes = dc.get_ksa_athletes()

if len(df_athletes) == 0:
    st.warning("No athlete data loaded. Run: `python -m scrapers.scrape_athletes`")
    st.stop()

athlete_names = df_athletes["full_name"].tolist()
selected_name = st.selectbox("Select Athlete", athlete_names, key="ap_athlete")

# Get athlete row
athlete = df_athletes[df_athletes["full_name"] == selected_name].iloc[0]

# ── Hero Card ─────────────────────────────────────────────────────────

col_photo, col_info = st.columns([1, 3])

with col_photo:
    # Check both v2 and legacy photo column names
    photo_url = athlete.get("photo_url", athlete.get("profile_image_url"))
    photo_shown = False

    # Try URL first
    if pd.notna(photo_url) and str(photo_url) not in ("None", "nan", ""):
        st.image(photo_url, width=200)
        photo_shown = True

    # Try local photo file as fallback
    if not photo_shown:
        from pathlib import Path
        aid = str(athlete.get("athlete_id", ""))
        photos_dir = Path(__file__).parent.parent / "data" / "scraped" / "photos"
        for ext in (".jpg", ".png", ".webp"):
            local_photo = photos_dir / f"{aid}{ext}"
            if local_photo.exists():
                st.image(str(local_photo), width=200)
                photo_shown = True
                break

    if not photo_shown:
        # Avatar placeholder
        st.markdown(f"""
        <div style="width: 200px; height: 200px; background: {TEAL_PRIMARY};
             border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="color: white; font-size: 4rem; font-weight: bold;">
                {selected_name[0] if selected_name else "?"}
            </span>
        </div>
        """, unsafe_allow_html=True)

with col_info:
    st.markdown(f"# {selected_name}")

    info_parts = []
    # Handle both v2 'country_name' and legacy 'country_code'
    country_name = athlete.get("country_name")
    country_code = athlete.get("country_code", "KSA")
    if _safe(country_name):
        info_parts.append(f"**{country_name}** ({country_code})")
    elif _safe(country_code):
        info_parts.append(f"**{country_code}**")

    # Handle both v2 'birth_date' and legacy 'date_of_birth'
    birth_date = athlete.get("birth_date", athlete.get("date_of_birth"))
    if _safe(birth_date):
        info_parts.append(f"Born: {birth_date}")

    primary_event = athlete.get("primary_event")
    if _safe(primary_event):
        info_parts.append(f"Primary: **{primary_event}**")

    st.markdown(" | ".join(info_parts))

    # Key metrics - handle both v2 and legacy column names
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        rank = athlete.get("best_world_rank")
        render_metric_card("World Rank", f"#{rank}" if _safe(rank) else "N/R", "excellent" if _safe(rank) and int(rank) <= 50 else "good")
    with m2:
        # v2 uses 'best_ranking_score', legacy uses 'best_score'
        score = athlete.get("best_ranking_score", athlete.get("best_score"))
        if _safe(score):
            try:
                render_metric_card("Score", f"{float(score):.0f}", "good")
            except (ValueError, TypeError):
                render_metric_card("Score", str(score), "good")
        else:
            render_metric_card("Score", "N/A", "neutral")
    with m3:
        # v2 has 'total_medals', legacy doesn't
        medals = athlete.get("total_medals", 0)
        render_metric_card("Medals", str(medals) if _safe(medals) else "0", "gold" if _safe(medals) and int(medals) > 0 else "neutral")
    with m4:
        # Count events from PBs for accurate event count
        pbs_for_count = dc.get_ksa_athlete_pbs(selected_name)
        event_col_for_count = "discipline" if "discipline" in pbs_for_count.columns else "event" if "event" in pbs_for_count.columns else None
        actual_event_count = pbs_for_count[event_col_for_count].nunique() if event_col_for_count and len(pbs_for_count) > 0 else athlete.get("pb_count", 0)
        render_metric_card("Events", str(actual_event_count) if _safe(actual_event_count) else "N/A", "good" if _safe(actual_event_count) and int(actual_event_count) > 1 else "neutral")

# ── Multi-Event Summary ──────────────────────────────────────────────

pbs = dc.get_ksa_athlete_pbs(selected_name)

if len(pbs) > 0 and len(pbs) > 1:
    from data.event_utils import format_event_name
    event_col = "discipline" if "discipline" in pbs.columns else "event" if "event" in pbs.columns else None
    mark_col = "mark" if "mark" in pbs.columns else "result" if "result" in pbs.columns else None
    score_col = "result_score" if "result_score" in pbs.columns else "resultscore" if "resultscore" in pbs.columns else None

    if event_col and mark_col:
        # Filter outlier events: remove zero-score and events scoring <65% of max
        display_pbs_summary = pbs.copy()
        if score_col:
            scores_numeric = pd.to_numeric(display_pbs_summary[score_col], errors="coerce").fillna(0)
            max_score = scores_numeric.max()
            if max_score > 0:
                threshold = max_score * 0.65
                display_pbs_summary = display_pbs_summary[scores_numeric >= threshold]
            else:
                display_pbs_summary = display_pbs_summary[scores_numeric > 0]

        n_core_events = len(display_pbs_summary)
        render_section_header("Multi-Event Summary", f"{selected_name} — {n_core_events} core events")

        # Show event cards in columns (max 4 per row)
        cols_per_row = min(n_core_events, 4) if n_core_events > 0 else 1
        cols = st.columns(cols_per_row)

        for idx, (_, pb_row) in enumerate(display_pbs_summary.iterrows()):
            with cols[idx % cols_per_row]:
                event_name = format_event_name(str(pb_row[event_col]))
                mark_val = str(pb_row[mark_col])
                score_val = pb_row.get(score_col, 0) if score_col else 0

                # Color code: best scoring event = gold, others = teal
                status = "gold" if idx == 0 else "excellent" if idx < 3 else "good"
                render_metric_card(event_name, mark_val, status)

                if _safe(score_val) and float(score_val) > 0:
                    st.caption(f"WA Points: {float(score_val):.0f}")

# ── Personal Bests & WA Points Analysis ──────────────────────────────

render_section_header("Personal Bests & Scoring Analysis", "All-time bests + where to earn ranking points")

if len(pbs) > 0:
    from data.event_utils import format_event_name as _fmt_ev
    from components.charts import event_scoring_bars, wa_points_heatmap

    _ev_col = "discipline" if "discipline" in pbs.columns else "event" if "event" in pbs.columns else None
    _mk_col = "mark" if "mark" in pbs.columns else "result" if "result" in pbs.columns else None
    _sc_col = "result_score" if "result_score" in pbs.columns else "resultscore" if "resultscore" in pbs.columns else None

    # ─ Event filter ─
    if _ev_col:
        pb_event_list = ["All Events"] + sorted(pbs[_ev_col].unique().tolist())
        pb_filter = st.selectbox("Filter by event", pb_event_list, key="ap_pb_filter")
        display_pbs = pbs if pb_filter == "All Events" else pbs[pbs[_ev_col] == pb_filter]
    else:
        display_pbs = pbs

    # ─ PB Table ─
    display_cols_v2 = ["discipline", "mark", "result_score", "venue", "date", "indoor"]
    display_cols_legacy = ["event", "result", "resultscore", "venue", "date", "wind"]
    available_cols = [c for c in display_cols_v2 if c in display_pbs.columns]
    if not available_cols or len(available_cols) < 3:
        available_cols = [c for c in display_cols_legacy if c in display_pbs.columns]

    column_config = {}
    if "discipline" in available_cols:
        column_config["discipline"] = st.column_config.TextColumn("Event")
    elif "event" in available_cols:
        column_config["event"] = st.column_config.TextColumn("Event")
    if "mark" in available_cols:
        column_config["mark"] = st.column_config.TextColumn("PB")
    elif "result" in available_cols:
        column_config["result"] = st.column_config.TextColumn("PB")
    if "result_score" in available_cols:
        column_config["result_score"] = st.column_config.NumberColumn("WA Points", format="%.0f")
    elif "resultscore" in available_cols:
        column_config["resultscore"] = st.column_config.NumberColumn("WA Points", format="%.0f")
    if "venue" in available_cols:
        column_config["venue"] = st.column_config.TextColumn("Venue")
    if "date" in available_cols:
        column_config["date"] = st.column_config.TextColumn("Date")
    if "indoor" in available_cols:
        column_config["indoor"] = st.column_config.CheckboxColumn("Indoor")
    if "wind" in available_cols:
        column_config["wind"] = st.column_config.TextColumn("Wind")

    st.dataframe(display_pbs[available_cols], hide_index=True, column_config=column_config)

    # ─ WA Points Scoring Bar Chart ("Easy Points" analysis) ─
    if _sc_col and _ev_col and len(pbs) > 1:
        st.markdown("---")
        st.markdown(f"#### WA Points by Event")
        st.caption("Highest-scoring events highlighted in gold — focus training here for maximum ranking points")

        # Build clean chart dataframe with only needed columns
        chart_pbs = pbs[[_ev_col, _sc_col]].copy()
        chart_pbs["discipline"] = chart_pbs[_ev_col].apply(lambda x: _fmt_ev(str(x)))
        chart_pbs["result_score"] = pd.to_numeric(chart_pbs[_sc_col], errors="coerce")
        chart_pbs = chart_pbs[["discipline", "result_score"]].drop_duplicates(subset=["discipline"], keep="first")
        chart_pbs = chart_pbs[chart_pbs["result_score"] > 0]

        fig_bars = event_scoring_bars(
            chart_pbs,
            title=f"{selected_name} — WA Points by Event (PB)",
        )
        if fig_bars.data:
            st.plotly_chart(fig_bars, use_container_width=True)

    # ─ WA Points Scoring Heat Map (results over time) ─
    results_for_heatmap = dc.get_ksa_results(athlete_name=selected_name, limit=200)
    if len(results_for_heatmap) > 0:
        _res_ev_col = "discipline" if "discipline" in results_for_heatmap.columns else "event"
        _res_sc_col = "result_score" if "result_score" in results_for_heatmap.columns else "resultscore"

        if _res_sc_col in results_for_heatmap.columns and "date" in results_for_heatmap.columns:
            # Format event names for display
            hm_data = results_for_heatmap.copy()
            hm_data["discipline"] = hm_data[_res_ev_col].apply(lambda x: _fmt_ev(str(x)))
            if _res_sc_col != "result_score":
                hm_data["result_score"] = hm_data[_res_sc_col]

            # Filter out zero scores
            hm_data = hm_data[pd.to_numeric(hm_data["result_score"], errors="coerce") > 0]

            if len(hm_data) > 2:
                st.markdown("---")
                st.markdown("#### Scoring Heat Map")
                st.caption("WA Points earned per event per month — darker = higher scoring performances")

                fig_hm = wa_points_heatmap(hm_data, title=f"{selected_name} — Competition Scoring Map")
                if fig_hm.data:
                    st.plotly_chart(fig_hm, use_container_width=True)

    # ─ Easy Points Summary ─
    if _sc_col and _ev_col and len(pbs) > 1:
        scores = pd.to_numeric(pbs[_sc_col], errors="coerce").dropna()
        if len(scores) > 0:
            max_score = scores.max()
            best_event_idx = scores.idxmax()
            best_event = _fmt_ev(str(pbs.loc[best_event_idx, _ev_col]))
            avg_score = scores.mean()

            st.markdown("---")
            st.markdown("#### Quick Insights")
            ins_c1, ins_c2, ins_c3 = st.columns(3)
            with ins_c1:
                render_metric_card("Best Scoring Event", best_event, "gold")
                st.caption(f"{max_score:.0f} WA pts")
            with ins_c2:
                render_metric_card("Avg PB Score", f"{avg_score:.0f}", "excellent")
                st.caption("Across all events")
            with ins_c3:
                above_1000 = sum(1 for s in scores if s >= 1000)
                render_metric_card("Events 1000+ pts", str(above_1000), "good" if above_1000 > 0 else "neutral")
                st.caption(f"Out of {len(scores)} events")
else:
    st.info("No PB data available for this athlete.")

# ── Season Bests ─────────────────────────────────────────────────────

import datetime
current_year = datetime.datetime.now().year

render_section_header("Season Bests", f"Best marks by event for {current_year}")

sbs = dc.get_ksa_athlete_season_bests(selected_name, season=current_year)
if len(sbs) > 0:
    sb_display = ["event", "result", "resultscore", "venue", "date"]
    sb_avail = [c for c in sb_display if c in sbs.columns]
    st.dataframe(
        sbs[sb_avail],
        hide_index=True,
        column_config={
            "event": st.column_config.TextColumn("Event"),
            "result": st.column_config.TextColumn("SB"),
            "resultscore": st.column_config.NumberColumn("WA Points", format="%.0f"),
            "venue": st.column_config.TextColumn("Venue"),
            "date": st.column_config.TextColumn("Date"),
        },
    )
else:
    # Try previous year
    sbs_prev = dc.get_ksa_athlete_season_bests(selected_name, season=current_year - 1)
    if len(sbs_prev) > 0:
        st.info(f"No {current_year} results yet. Showing {current_year - 1} season bests.")
        sb_display = ["event", "result", "resultscore", "venue", "date"]
        sb_avail = [c for c in sb_display if c in sbs_prev.columns]
        st.dataframe(sbs_prev[sb_avail], hide_index=True)
    else:
        st.info(f"No season best data for {current_year} or {current_year - 1}.")

# ── Top 5 Average ────────────────────────────────────────────────────

render_section_header("Top 5 Average", "Average of 5 best performances per event (all-time)")

top5 = dc.get_ksa_athlete_top5_avg(selected_name)
if len(top5) > 0:
    from data.event_utils import format_event_name as fmt_event

    # Show as metric cards
    n_events_top5 = len(top5)
    t5_cols = st.columns(min(n_events_top5, 4))
    for idx, (_, row) in enumerate(top5.iterrows()):
        with t5_cols[idx % min(n_events_top5, 4)]:
            ev_name = fmt_event(str(row["event"]))
            avg_val = row.get("top5_avg", 0)
            best_val = row.get("top5_best", 0)
            n_perfs = row.get("n_performances", 0)
            pts = row.get("avg_wa_points", 0)

            render_metric_card(ev_name, f"{avg_val:.2f}", "excellent")
            st.caption(f"Best: {best_val} | {n_perfs} perfs | Avg pts: {pts:.0f}")
else:
    st.info("Not enough performances for top 5 average.")

# ── Year-by-Year Progression ────────────────────────────────────────

render_section_header("Year-by-Year Progression", "Annual best marks across career")

progression = dc.get_ksa_athlete_year_progression(selected_name)
if len(progression) > 0:
    from data.event_utils import format_event_name as fmt_event_prog

    # Get unique events
    prog_events = progression["event"].unique().tolist()

    # Default to athlete's primary event (not "All Events")
    primary_ev = athlete.get("primary_event", "")
    default_prog_idx = 0
    sorted_prog = sorted(prog_events)
    if _safe(primary_ev) and str(primary_ev) in sorted_prog:
        default_prog_idx = sorted_prog.index(str(primary_ev))

    selected_prog_event = st.selectbox(
        "Event for progression",
        sorted_prog,
        index=default_prog_idx,
        key="ap_prog_event",
    )
    progression = progression[progression["event"] == selected_prog_event]

    # Display table
    prog_display = ["year", "event", "best_mark", "best_score", "venue", "n_comps"]
    prog_avail = [c for c in prog_display if c in progression.columns]
    st.dataframe(
        progression[prog_avail],
        hide_index=True,
        column_config={
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "event": st.column_config.TextColumn("Event"),
            "best_mark": st.column_config.TextColumn("Best Mark"),
            "best_score": st.column_config.NumberColumn("WA Points", format="%.0f"),
            "venue": st.column_config.TextColumn("Venue"),
            "n_comps": st.column_config.NumberColumn("Comps"),
        },
    )

    # Progression chart
    import pandas as pd
    chart_prog = progression.copy()
    chart_prog["best_numeric"] = pd.to_numeric(chart_prog.get("best_numeric", chart_prog.get("best_mark")), errors="coerce")
    chart_prog = chart_prog.dropna(subset=["best_numeric"])

    if len(chart_prog) > 1:
        single_event = chart_prog["event"].iloc[0]
        evt_type = get_event_type(single_event)

        # Get PB for reference line
        _prog_pb = None
        _prog_sb = None
        if len(pbs) > 0:
            _ev_match_col = "discipline" if "discipline" in pbs.columns else "event"
            _mk_match_col = "mark" if "mark" in pbs.columns else "result"
            ev_pbs = pbs[pbs[_ev_match_col] == single_event]
            if len(ev_pbs) > 0:
                _prog_pb = pd.to_numeric(ev_pbs[_mk_match_col].iloc[0], errors="coerce")
                if pd.isna(_prog_pb):
                    _prog_pb = None

        fig = progression_chart(
            chart_prog,
            x_col="year",
            y_col="best_numeric",
            title=f"Year-by-Year: {fmt_event_prog(single_event)}",
            lower_is_better=(evt_type == "time"),
            pb_value=_prog_pb,
            hover_cols=["venue", "best_score", "n_comps"],
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No year-by-year data available.")

# ── Competition History ───────────────────────────────────────────────

render_section_header("Competition History", "Recent competition results")

results = dc.get_ksa_results(athlete_name=selected_name, limit=100)
if len(results) > 0:
    disc_col = "discipline" if "discipline" in results.columns else "event" if "event" in results.columns else None

    events_in_results = results[disc_col].unique().tolist() if disc_col else []
    selected_discipline = None
    if events_in_results:
        sorted_events = sorted(events_in_results)
        # Default to primary event
        primary_ev_hist = athlete.get("primary_event", "")
        default_hist_idx = 0
        if _safe(primary_ev_hist) and str(primary_ev_hist) in sorted_events:
            default_hist_idx = sorted_events.index(str(primary_ev_hist))

        selected_discipline = st.selectbox(
            "Filter by event",
            sorted_events,
            index=default_hist_idx,
            key="ap_discipline",
        )
        results = results[results[disc_col] == selected_discipline]

    display_cols_v2 = ["date", "competition", "discipline", "mark", "place", "venue", "result_score"]
    display_cols_legacy = ["date", "event", "result", "pos", "venue", "resultscore", "rank"]

    available_cols = [c for c in display_cols_v2 if c in results.columns]
    if not available_cols or len(available_cols) < 3:
        available_cols = [c for c in display_cols_legacy if c in results.columns]

    column_config = {}
    if "date" in available_cols:
        column_config["date"] = st.column_config.TextColumn("Date")
    if "competition" in available_cols:
        column_config["competition"] = st.column_config.TextColumn("Competition", width="large")
    if "discipline" in available_cols:
        column_config["discipline"] = st.column_config.TextColumn("Event")
    elif "event" in available_cols:
        column_config["event"] = st.column_config.TextColumn("Event")
    if "mark" in available_cols:
        column_config["mark"] = st.column_config.TextColumn("Mark")
    elif "result" in available_cols:
        column_config["result"] = st.column_config.TextColumn("Mark")
    if "place" in available_cols:
        column_config["place"] = st.column_config.TextColumn("Place")
    elif "pos" in available_cols:
        column_config["pos"] = st.column_config.TextColumn("Place")
    if "venue" in available_cols:
        column_config["venue"] = st.column_config.TextColumn("Venue")
    if "result_score" in available_cols:
        column_config["result_score"] = st.column_config.NumberColumn("Points", format="%.0f")
    elif "resultscore" in available_cols:
        column_config["resultscore"] = st.column_config.NumberColumn("Points", format="%.0f")

    st.dataframe(
        results[available_cols],
        hide_index=True,
        column_config=column_config,
        height=400,
    )

    # Progression chart with PB line and hover details
    mark_col = "mark" if "mark" in results.columns else "result" if "result" in results.columns else None
    if mark_col and "date" in results.columns:
        import pandas as pd
        chart_df = results.copy()
        if "result_numeric" in chart_df.columns:
            chart_df["mark_numeric"] = pd.to_numeric(chart_df["result_numeric"], errors="coerce")
        else:
            chart_df["mark_numeric"] = pd.to_numeric(chart_df[mark_col], errors="coerce")
        chart_df = chart_df.dropna(subset=["mark_numeric"])

        if len(chart_df) > 1:
            evt_type = get_event_type(selected_discipline or "100m")

            # Get PB for reference line
            _comp_pb = None
            if len(pbs) > 0:
                _ev_match = "discipline" if "discipline" in pbs.columns else "event"
                _mk_match = "mark" if "mark" in pbs.columns else "result"
                ev_pbs_hist = pbs[pbs[_ev_match] == selected_discipline] if selected_discipline else pbs
                if len(ev_pbs_hist) > 0:
                    _comp_pb = pd.to_numeric(ev_pbs_hist[_mk_match].iloc[0], errors="coerce")
                    if pd.isna(_comp_pb):
                        _comp_pb = None

            # Determine hover columns
            _hover = [c for c in ["competition", "venue", "place", "result_score"] if c in chart_df.columns]
            if not _hover:
                _hover = [c for c in ["venue", "pos", "resultscore"] if c in chart_df.columns]

            fig = progression_chart(
                chart_df,
                x_col="date",
                y_col="mark_numeric",
                title=f"Performance Progression - {selected_discipline or 'All'}",
                lower_is_better=(evt_type == "time"),
                pb_value=_comp_pb,
                hover_cols=_hover,
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No competition results available. Run: `python -m scrapers.scrape_athletes`")

# ── Rankings Position ─────────────────────────────────────────────────

render_section_header("World Rankings", "Current ranking positions across all events")

ksa_rankings = dc.get_ksa_rankings()
if len(ksa_rankings) > 0:
    name_col = "athlete" if "athlete" in ksa_rankings.columns else "competitor" if "competitor" in ksa_rankings.columns else None
    if name_col:
        last_name = selected_name.split()[-1] if selected_name else ""
        athlete_rankings = ksa_rankings[
            ksa_rankings[name_col].str.contains(last_name, case=False, na=False)
        ]
    else:
        athlete_rankings = ksa_rankings.head(0)

    if len(athlete_rankings) > 0:
        display_cols_v2 = ["event", "rank", "ranking_score", "competitions_scored"]
        display_cols_legacy = ["event", "rank", "resultscore"]

        available_cols = [c for c in display_cols_v2 if c in athlete_rankings.columns]
        if not available_cols or len(available_cols) < 2:
            available_cols = [c for c in display_cols_legacy if c in athlete_rankings.columns]

        if len(available_cols) > 0:
            st.dataframe(athlete_rankings[available_cols], hide_index=True)
    else:
        st.info("No world ranking entries found for this athlete.")

# ── Pre-Competition Report ───────────────────────────────────────────

render_section_header("Pre-Competition Report", "Generate PDF briefing for coaches")

# Build event list from athlete's PBs
report_events = []
if len(pbs) > 0:
    event_col_report = "discipline" if "discipline" in pbs.columns else "event" if "event" in pbs.columns else None
    if event_col_report:
        report_events = pbs[event_col_report].dropna().unique().tolist()

if report_events:
    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        report_event = st.selectbox("Report Event", report_events, key="ap_report_event")
    with col_rep2:
        report_format = st.radio("Format", ["PDF", "HTML Preview"], horizontal=True, key="ap_report_fmt")

    if st.button("Generate Report", type="primary", key="ap_gen_report"):
        with st.spinner("Generating report..."):
            try:
                from analytics.report_generator import PreCompReportGenerator
                gen = PreCompReportGenerator(dc)
                if report_format == "PDF":
                    pdf_bytes = gen.generate(selected_name, report_event, format="pdf")
                    safe_filename = selected_name.replace(" ", "_").replace("'", "")
                    st.download_button(
                        "Download PDF",
                        pdf_bytes,
                        file_name=f"{safe_filename}_precomp_report.pdf",
                        mime="application/pdf",
                        key="ap_download_pdf",
                    )
                else:
                    html = gen.generate(selected_name, report_event, format="html")
                    st.html(html)
            except Exception as e:
                st.error(f"Report generation failed: {e}")
else:
    st.info("No event data available for report generation. Run the scraper to populate PB data.")

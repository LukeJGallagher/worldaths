"""
Competition Strategy — Qualification Pathway Planner.

Athlete-specific competition planning:
- Current qualification situation
- Competition recommendations (ranked by point potential)
- Scenario calculator ("what if I run X at category Y?")
- Competition timeline with deadlines
"""

import datetime

import numpy as np
import pandas as pd
import streamlit as st

from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar,
    TEAL_PRIMARY, TEAL_DARK, GOLD_ACCENT, TEAL_LIGHT, GRAY_BLUE,
)
from components.charts import points_scenario_chart, competition_timeline_chart
from data.connector import get_connector
from data.event_utils import format_event_name, get_event_type
from analytics.competition_strategy import (
    PLACE_POINTS, CATEGORY_ORDER, QUALIFICATION_DEADLINES,
    CATEGORY_COLORS, CATEGORY_NAMES,
    estimate_total_points, calculate_points_gain,
    rank_competitions, interpolate_result_score,
    build_qualification_status,
)


def _safe(val):
    try:
        return pd.notna(val) and str(val) not in ("None", "nan", "NaT", "")
    except (ValueError, TypeError):
        return False


# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Competition Strategy", page_icon="🎯", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()
render_page_header(
    "Competition Strategy",
    "Qualification pathway planner — where to compete for maximum ranking impact",
)

dc = get_connector()

# ── Athlete + Event Selector ─────────────────────────────────────────

df_athletes = dc.get_ksa_athletes()
if len(df_athletes) == 0:
    st.warning("No athlete data loaded.")
    st.stop()

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    athlete_names = df_athletes["full_name"].tolist()
    selected_name = st.selectbox("Select Athlete", athlete_names, key="cs_athlete")
with col_sel2:
    pbs = dc.get_ksa_athlete_pbs(selected_name)
    ev_col = "discipline" if "discipline" in pbs.columns else "event" if "event" in pbs.columns else None
    mk_col = "mark" if "mark" in pbs.columns else "result" if "result" in pbs.columns else None
    sc_col = "result_score" if "result_score" in pbs.columns else "resultscore" if "resultscore" in pbs.columns else None

    if ev_col and len(pbs) > 0:
        events = sorted(pbs[ev_col].dropna().unique().tolist())
        # Default to primary event
        athlete_row = df_athletes[df_athletes["full_name"] == selected_name].iloc[0]
        primary = athlete_row.get("primary_event", "")
        default_idx = 0
        if _safe(primary) and str(primary) in events:
            default_idx = events.index(str(primary))
        selected_event = st.selectbox("Event", events, index=default_idx, key="cs_event")
    else:
        st.info("No PB data for this athlete.")
        st.stop()

# ── Gather athlete data ──────────────────────────────────────────────

# PB for selected event
ev_pbs = pbs[pbs[ev_col] == selected_event]
pb_mark = None
pb_score = 0
if len(ev_pbs) > 0:
    pb_mark_raw = ev_pbs[mk_col].iloc[0] if mk_col else None
    pb_mark = pd.to_numeric(pb_mark_raw, errors="coerce") if pb_mark_raw else None
    if sc_col and sc_col in ev_pbs.columns:
        pb_score = pd.to_numeric(ev_pbs[sc_col].iloc[0], errors="coerce")
        if pd.isna(pb_score):
            pb_score = 0
        pb_score = float(pb_score)

# All results for the athlete + event
all_results = dc.get_ksa_results(athlete_name=selected_name, limit=200)
disc_col = "discipline" if "discipline" in all_results.columns else "event" if "event" in all_results.columns else None
if disc_col and len(all_results) > 0:
    event_results = all_results[all_results[disc_col] == selected_event].copy()
else:
    event_results = pd.DataFrame()

# Build top-5 scoring list
res_sc_col = "result_score" if "result_score" in event_results.columns else "resultscore" if "resultscore" in event_results.columns else None
top5_scores = []
if res_sc_col and len(event_results) > 0:
    scores_series = pd.to_numeric(event_results[res_sc_col], errors="coerce").dropna()
    # Estimate total including place bonus (approximate with avg category)
    top5_scores = sorted(scores_series.tolist(), reverse=True)[:5]

event_type = get_event_type(selected_event)
event_display = format_event_name(selected_event)

# ── Tabs ──────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Current Situation",
    "Competition Recommendations",
    "Scenario Calculator",
    "Competition Timeline",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: Current Situation
# ══════════════════════════════════════════════════════════════════════

with tabs[0]:
    render_section_header(
        f"{selected_name} — {event_display}",
        "Current ranking position and qualification status",
    )

    # Metric cards
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)

    with mc1:
        rank = athlete_row.get("best_world_rank")
        render_metric_card(
            "World Rank",
            f"#{int(rank)}" if _safe(rank) else "N/R",
            "excellent" if _safe(rank) and int(rank) <= 50 else "good",
        )
    with mc2:
        render_metric_card(
            f"PB ({event_display})",
            str(pb_mark_raw) if pb_mark_raw else "N/A",
            "gold",
        )
    with mc3:
        render_metric_card(
            "PB WA Points",
            f"{pb_score:.0f}" if pb_score else "N/A",
            "excellent" if pb_score >= 1100 else "good" if pb_score >= 900 else "neutral",
        )
    with mc4:
        top5_avg = round(np.mean(top5_scores), 1) if top5_scores else 0
        render_metric_card(
            "Top 5 Avg",
            f"{top5_avg:.0f}" if top5_avg else "N/A",
            "excellent" if top5_avg >= 1100 else "good" if top5_avg >= 900 else "neutral",
        )
    with mc5:
        weakest = round(min(top5_scores), 1) if top5_scores else 0
        render_metric_card(
            "Weakest Top 5",
            f"{weakest:.0f}" if weakest else "N/A",
            "warning" if top5_scores and len(top5_scores) >= 5 else "neutral",
        )

    # ── Qualification Status ──
    st.markdown("---")
    st.markdown("#### Qualification Status")

    qual_rows = build_qualification_status(
        athlete_pb_mark=float(pb_mark) if pb_mark is not None and not pd.isna(pb_mark) else None,
        athlete_sb_mark=None,
        event=selected_event,
        event_type=event_type,
    )

    if qual_rows:
        qual_df = pd.DataFrame(qual_rows)
        # Format marks for display
        if event_type == "time":
            for col in ["Standard", "Athlete PB", "Gap"]:
                if col in qual_df.columns:
                    qual_df[col] = qual_df[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "—"
                    )
        else:
            for col in ["Standard", "Athlete PB", "Gap"]:
                if col in qual_df.columns:
                    qual_df[col] = qual_df[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "—"
                    )

        # Status emoji
        status_map = {"qualified": "✅", "close": "🟡", "gap": "🔴", "unknown": "—"}
        qual_df["Status"] = qual_df["Status"].map(status_map).fillna("—")

        st.dataframe(
            qual_df[["Championship", "Level", "Standard", "Athlete PB", "Gap", "Status"]],
            hide_index=True,
            column_config={
                "Championship": st.column_config.TextColumn("Championship", width="large"),
                "Level": st.column_config.TextColumn("Level"),
                "Standard": st.column_config.TextColumn("Standard"),
                "Athlete PB": st.column_config.TextColumn("PB"),
                "Gap": st.column_config.TextColumn("Gap"),
                "Status": st.column_config.TextColumn("Status"),
            },
        )
    else:
        st.info(f"No championship standards found for {event_display}.")

    # ── Top 5 Scoring Performances ──
    if len(event_results) > 0 and res_sc_col:
        st.markdown("---")
        st.markdown("#### Top 5 Scoring Performances")
        st.caption("Only the **top 5** scoring results count toward your world ranking average")

        res_mark_col = "mark" if "mark" in event_results.columns else "result" if "result" in event_results.columns else None
        event_results["_score_num"] = pd.to_numeric(event_results[res_sc_col], errors="coerce")
        top5_df = event_results.nlargest(5, "_score_num")

        display_cols = []
        col_cfg = {}
        if res_mark_col:
            display_cols.append(res_mark_col)
            col_cfg[res_mark_col] = st.column_config.TextColumn("Mark")
        display_cols.append(res_sc_col)
        col_cfg[res_sc_col] = st.column_config.NumberColumn("WA Points", format=",.0f")
        for c in ["competition", "venue", "date"]:
            if c in top5_df.columns:
                display_cols.append(c)
                col_cfg[c] = st.column_config.TextColumn(c.title())

        avail = [c for c in display_cols if c in top5_df.columns]
        st.dataframe(top5_df[avail], hide_index=True, column_config=col_cfg)

        # Chart
        fig_top5 = points_scenario_chart(top5_scores, title=f"Current Top 5 — {event_display}")
        st.plotly_chart(fig_top5, use_container_width=True)

    # ── Rivals Context ──
    rivals = dc.get_rivals(event=selected_event, gender=str(athlete_row.get("gender", "male")), limit=5)
    if len(rivals) > 0:
        st.markdown("---")
        st.markdown("#### Key Rivals")
        rival_cols = [c for c in ["athlete", "country_code", "world_rank", "pb_mark"] if c in rivals.columns]
        if rival_cols:
            st.dataframe(rivals[rival_cols].head(5), hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: Competition Recommendations
# ══════════════════════════════════════════════════════════════════════

with tabs[1]:
    render_section_header(
        "Competition Recommendations",
        f"Best upcoming competitions for {selected_name} — {event_display}",
    )

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        champ_options = list(QUALIFICATION_DEADLINES.keys())
        selected_champ = st.selectbox(
            "Target Championship",
            champ_options,
            index=1,  # Default: Asian Games 2026
            key="cs_target_champ",
        )
        deadline = QUALIFICATION_DEADLINES[selected_champ]["deadline"]
    with fc2:
        area_options = ["All Regions", "Asia", "Europe", "Africa",
                        "North and Central America", "South America", "Oceania"]
        selected_area = st.selectbox("Region Preference", area_options, key="cs_area")
        area_filter = None if selected_area == "All Regions" else selected_area
    with fc3:
        cat_options = ["F (All)", "E+", "D+", "C+", "B+", "A+", "GL+", "GW+"]
        cat_map = {"F (All)": "F", "E+": "E", "D+": "D", "C+": "C", "B+": "B",
                    "A+": "A", "GL+": "GL", "GW+": "GW"}
        selected_min_cat = st.selectbox("Min Category", cat_options, index=4, key="cs_min_cat")
        min_cat = cat_map.get(selected_min_cat, "F")

    # Get and rank competitions
    today = datetime.date.today().isoformat()
    cal = dc.get_calendar_for_event(
        start_date=today,
        end_date=deadline,
        min_category=min_cat,
        area=area_filter,
    )

    if len(cal) > 0 and pb_score > 0:
        ranked = rank_competitions(
            cal,
            athlete_result_score=pb_score,
            deadline=deadline,
            area_preference=area_filter,
            min_category=min_cat,
            current_top5=top5_scores if top5_scores else None,
        )

        if len(ranked) > 0:
            # Top 3 recommendation cards
            st.markdown("#### Top Recommendations")
            top3 = ranked.head(3)
            rec_cols = st.columns(min(3, len(top3)))
            for idx, (_, row) in enumerate(top3.iterrows()):
                with rec_cols[idx]:
                    cat = row.get("ranking_category", "F")
                    cat_name = CATEGORY_NAMES.get(cat, cat)
                    days = int(row.get("days_until", 0))
                    st.markdown(f"""
<div style="background: white; border-radius: 8px; padding: 1rem;
     border-left: 4px solid {CATEGORY_COLORS.get(cat, GRAY_BLUE)};
     box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 0.5rem;">
    <div style="font-weight: 600; color: {TEAL_PRIMARY};">{str(row.get('name', ''))[:45]}</div>
    <div style="font-size: 0.85rem; color: #666;">
        {row.get('venue', '')} | {row.get('start_date', '')}<br>
        <strong>{cat_name}</strong> | +{int(row.get('place_pts', 0))} place pts<br>
        Est. total: <strong>{int(row.get('est_total', 0))}</strong> pts |
        Gain: <strong>+{int(row.get('est_gain', 0))}</strong> |
        {days} days away
    </div>
</div>
""", unsafe_allow_html=True)

            # Full table
            st.markdown("---")
            st.markdown(f"#### All Competitions ({len(ranked)} matches)")

            table_cols = ["name", "venue", "start_date", "ranking_category",
                          "place_pts", "est_total", "est_gain", "days_until"]
            if "area" in ranked.columns:
                table_cols.insert(3, "area")
            avail = [c for c in table_cols if c in ranked.columns]

            st.dataframe(
                ranked[avail].head(50),
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Competition", width="large"),
                    "venue": st.column_config.TextColumn("Venue"),
                    "area": st.column_config.TextColumn("Region"),
                    "start_date": st.column_config.TextColumn("Date"),
                    "ranking_category": st.column_config.TextColumn("Cat"),
                    "place_pts": st.column_config.NumberColumn("Place Pts", format=".0f"),
                    "est_total": st.column_config.NumberColumn("Est. Total", format=",.0f"),
                    "est_gain": st.column_config.NumberColumn("Est. Gain", format=".0f"),
                    "days_until": st.column_config.NumberColumn("Days", format=".0f"),
                },
                height=600,
            )

            st.caption(
                "**Note:** Event availability at each competition should be confirmed — "
                "calendar shows all Track & Field meets. Est. Total = PB result score + "
                "place points (assuming 1st place at that category level)."
            )
        else:
            st.info("No competitions match the current filters.")
    elif pb_score == 0:
        st.warning("No WA points data for this event. Cannot estimate competition value.")
    else:
        st.info(f"No Track & Field competitions found before {deadline}.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3: Scenario Calculator
# ══════════════════════════════════════════════════════════════════════

with tabs[2]:
    render_section_header(
        "Scenario Calculator",
        "Model 'what if' scenarios — estimate ranking impact of hypothetical performances",
    )

    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        default_mark = float(pb_mark) if pb_mark is not None and not pd.isna(pb_mark) else 10.00
        hyp_mark = st.number_input(
            f"Hypothetical Mark ({'seconds' if event_type == 'time' else 'metres'})",
            value=default_mark,
            step=0.01 if event_type == "time" else 0.01,
            format="%.2f",
            key="cs_hyp_mark",
        )
    with sc2:
        cat_labels = [f"{code} — {CATEGORY_NAMES.get(code, code)}" for code in CATEGORY_ORDER]
        cat_idx = st.selectbox(
            "Competition Category",
            range(len(cat_labels)),
            format_func=lambda i: cat_labels[i],
            index=4,  # Default: A
            key="cs_hyp_cat",
        )
        hyp_cat = CATEGORY_ORDER[cat_idx]
    with sc3:
        hyp_place = st.selectbox(
            "Expected Finish",
            [1, 2, 3, 4, 5, 6, 7, 8],
            index=0,
            key="cs_hyp_place",
        )

    # Estimate result score for hypothetical mark
    est_result_score = pb_score  # Default: same as PB
    if len(event_results) >= 2 and res_sc_col:
        res_mark_col2 = "mark" if "mark" in event_results.columns else "result"
        marks_list = pd.to_numeric(event_results[res_mark_col2], errors="coerce").dropna().tolist()
        scores_list = pd.to_numeric(event_results[res_sc_col], errors="coerce").dropna().tolist()
        if len(marks_list) >= 2:
            interp = interpolate_result_score(
                marks_list, scores_list, hyp_mark,
                lower_is_better=(event_type == "time"),
            )
            if interp is not None:
                est_result_score = interp

    # Calculate total and gain
    est_total = estimate_total_points(est_result_score, hyp_cat, hyp_place)
    gain_info = calculate_points_gain(top5_scores, est_total)

    # Display results
    st.markdown("---")
    st.markdown("#### Scenario Result")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        render_metric_card("Est. Result Score", f"{est_result_score:.0f}", "good")
    with r2:
        place_pts = PLACE_POINTS.get(hyp_cat, 0)
        render_metric_card("Place Points", f"+{place_pts}", "good")
    with r3:
        render_metric_card("Est. Total", f"{est_total:.0f}", "gold")
    with r4:
        if gain_info["replaces"]:
            render_metric_card("Ranking Impact", f"+{gain_info['improvement']:.0f}", "excellent")
        else:
            render_metric_card("Ranking Impact", "No change", "neutral")

    # Detail explanation
    if gain_info["replaces"]:
        displaced = gain_info.get("displaced_score")
        if displaced:
            st.success(
                f"This performance would **displace** your current #5 score ({displaced:.0f}) — "
                f"new Top 5 average: **{gain_info['new_avg']:.0f}** "
                f"(up from {gain_info['current_avg']:.0f}, +{gain_info['improvement']:.0f} improvement)"
            )
        else:
            st.success(
                f"This would be added to your Top 5 — "
                f"new average: **{gain_info['new_avg']:.0f}**"
            )
    else:
        st.info(
            f"This score ({est_total:.0f}) would NOT displace your current weakest "
            f"Top 5 entry ({min(top5_scores):.0f}). "
            f"Aim for a higher-category competition or a faster mark."
        )

    # Scenario chart
    if top5_scores:
        fig_scenario = points_scenario_chart(
            top5_scores,
            hypothetical_new=est_total if gain_info["replaces"] else None,
            title=f"Top 5 Impact — {hyp_mark:.2f} at {CATEGORY_NAMES.get(hyp_cat, hyp_cat)}",
        )
        st.plotly_chart(fig_scenario, use_container_width=True)

    # Quick comparisons
    st.markdown("---")
    st.markdown("#### Quick Category Comparison")
    st.caption("Same mark at different competition levels:")

    comp_rows = []
    for cat in ["OW", "DF", "GW", "A", "B", "C"]:
        total = estimate_total_points(est_result_score, cat, hyp_place)
        gain = calculate_points_gain(top5_scores, total)
        comp_rows.append({
            "Category": f"{cat} — {CATEGORY_NAMES.get(cat, cat)}",
            "Place Pts": PLACE_POINTS.get(cat, 0),
            "Est. Total": round(total),
            "Would Displace": "Yes" if gain["replaces"] else "No",
            "New Avg": round(gain["new_avg"]),
            "Improvement": f"+{gain['improvement']:.0f}" if gain["improvement"] > 0 else "—",
        })
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, hide_index=True, column_config={
        "Category": st.column_config.TextColumn("Category", width="large"),
        "Place Pts": st.column_config.NumberColumn("Place Pts", format=".0f"),
        "Est. Total": st.column_config.NumberColumn("Est. Total", format=",.0f"),
        "Would Displace": st.column_config.TextColumn("Displaces #5?"),
        "New Avg": st.column_config.NumberColumn("New Avg", format=",.0f"),
        "Improvement": st.column_config.TextColumn("Improvement"),
    })


# ══════════════════════════════════════════════════════════════════════
# TAB 4: Competition Timeline
# ══════════════════════════════════════════════════════════════════════

with tabs[3]:
    render_section_header(
        "Competition Timeline",
        f"Upcoming competition calendar toward {selected_champ}",
    )

    # Key dates
    st.markdown("#### Key Dates")
    deadline_cols = st.columns(len(QUALIFICATION_DEADLINES))
    for col, (champ, info) in zip(deadline_cols, QUALIFICATION_DEADLINES.items()):
        with col:
            dl_date = datetime.date.fromisoformat(info["deadline"])
            days_remaining = (dl_date - datetime.date.today()).days
            status = "excellent" if days_remaining > 365 else "good" if days_remaining > 180 else "warning" if days_remaining > 60 else "danger"
            short_name = champ.split("(")[0].strip()
            render_metric_card(short_name, f"{days_remaining} days", status)
            st.caption(f"Deadline: {info['deadline']}")

    # Timeline chart
    st.markdown("---")
    today = datetime.date.today().isoformat()
    timeline_cal = dc.get_calendar_for_event(
        start_date=today,
        end_date=deadline,
        min_category="B",
    )

    if len(timeline_cal) > 0:
        # Show as sorted table with category coloring
        st.markdown(f"#### Upcoming B+ Competitions (before {deadline})")

        tl_df = timeline_cal.sort_values("start_date").head(30).copy()
        tl_df["Category"] = tl_df["ranking_category"].map(CATEGORY_NAMES).fillna(tl_df["ranking_category"])
        tl_df["Place Pts"] = tl_df["ranking_category"].map(PLACE_POINTS).fillna(0).astype(int)
        tl_df["Est. Total"] = (pb_score + tl_df["Place Pts"]).astype(int)

        display = ["name", "venue", "start_date", "Category", "Place Pts", "Est. Total"]
        if "area" in tl_df.columns:
            display.insert(3, "area")
        avail = [c for c in display if c in tl_df.columns]

        st.dataframe(
            tl_df[avail],
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("Competition", width="large"),
                "venue": st.column_config.TextColumn("Venue"),
                "area": st.column_config.TextColumn("Region"),
                "start_date": st.column_config.TextColumn("Date"),
                "Category": st.column_config.TextColumn("Category"),
                "Place Pts": st.column_config.NumberColumn("Place Pts", format=".0f"),
                "Est. Total": st.column_config.NumberColumn("Est. Total", format=",.0f"),
            },
            height=600,
        )

        # Category distribution
        st.markdown("---")
        st.markdown("#### Category Distribution")
        cat_counts = timeline_cal["ranking_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_counts["Name"] = cat_counts["Category"].map(CATEGORY_NAMES)

        import plotly.graph_objects as go
        fig_cats = go.Figure()
        fig_cats.add_trace(go.Bar(
            x=cat_counts["Category"],
            y=cat_counts["Count"],
            marker_color=[CATEGORY_COLORS.get(c, GRAY_BLUE) for c in cat_counts["Category"]],
            text=cat_counts["Count"],
            textposition="outside",
            hovertemplate="%{x}: %{y} competitions<extra></extra>",
        ))
        fig_cats.update_layout(
            title="Available Competitions by Category",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif", color="#333"),
            margin=dict(l=10, r=10, t=40, b=30),
            showlegend=False,
            yaxis_title="Count",
        )
        st.plotly_chart(fig_cats, use_container_width=True)
    else:
        st.info(f"No Track & Field competitions found before {deadline}.")

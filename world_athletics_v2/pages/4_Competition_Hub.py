"""
Competition Hub - Calendar, results viewer, upcoming events.

Features:
- Competition calendar with filters (date, region, ranking category)
- Click into competition -> see full results
- Upcoming events
- Which competitions offer best ranking points
- Qualification data fallback from road_to_tokyo
"""

import streamlit as st
from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar, TEAL_PRIMARY,
)
from components.filters import date_range_filter
from data.connector import get_connector

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Competition Hub", page_icon="🏟️", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header("Competition Hub", "Calendar, results, and competition intelligence")

dc = get_connector()
views = dc.available_views

# Determine what data is available
has_calendar = "calendar" in views
has_upcoming = "upcoming" in views
has_road_to_tokyo = "road_to_tokyo" in views

# ── Tabs ──────────────────────────────────────────────────────────────

# Build tab list based on available data
tab_names = ["Calendar", "Upcoming", "Competition Viewer", "Best Points Competitions"]
if has_road_to_tokyo and not has_calendar:
    tab_names.append("Qualification Tracking")

tabs = st.tabs(tab_names)

# ── Calendar Tab ──────────────────────────────────────────────────────

with tabs[0]:
    render_section_header("Competition Calendar", "Filter by date range and category")

    if has_calendar:
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date, end_date = date_range_filter(key="cal")
        # Map display labels → actual codes in calendar.parquet
        _CAL_CATEGORY_MAP = {
            "All": None,
            "Olympic/World Champs (OW)": "OW",
            "Diamond League Final (DF)": "DF",
            "Gold World (GW)": "GW",
            "Gold Label (GL)": "GL",
            "Category A": "A",
            "Category B": "B",
            "Category C": "C",
            "Category D": "D",
            "Category E": "E",
            "Category F": "F",
        }
        with col2:
            ranking_cat_label = st.selectbox(
                "Ranking Category",
                list(_CAL_CATEGORY_MAP.keys()),
                key="cal_cat",
            )
            ranking_cat = _CAL_CATEGORY_MAP.get(ranking_cat_label)
        with col3:
            area_filter = st.selectbox(
                "Area",
                ["All", "Asia", "Europe", "Africa", "North America", "South America", "Oceania"],
                key="cal_area",
            )

        df_cal = dc.get_calendar(
            start_date=start_date,
            end_date=end_date,
            ranking_category=ranking_cat,
        )

        if len(df_cal) > 0:
            st.markdown(f"**{len(df_cal)} competitions found**")

            display_cols = ["name", "venue", "area", "start_date", "end_date",
                           "ranking_category", "has_results"]
            available_cols = [c for c in display_cols if c in df_cal.columns]

            st.dataframe(
                df_cal[available_cols],
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Competition", width="large"),
                    "venue": st.column_config.TextColumn("Venue"),
                    "area": st.column_config.TextColumn("Area"),
                    "start_date": st.column_config.TextColumn("Start"),
                    "end_date": st.column_config.TextColumn("End"),
                    "ranking_category": st.column_config.TextColumn("Category"),
                    "has_results": st.column_config.CheckboxColumn("Results"),
                },
                height=500,
            )
        else:
            st.info("No competitions found for the selected filters.")
    else:
        st.info(
            "No calendar data available. "
            "Run: `python -m scrapers.scrape_competitions` to populate the competition calendar."
        )
        # If we have road_to_tokyo, show a summary of events from there
        if has_road_to_tokyo:
            st.markdown("---")
            st.markdown("**Qualification events from legacy data:**")
            quals = dc.get_qualifications()
            if len(quals) > 0:
                # Show unique events being tracked
                event_col = "Actual_Event_Name" if "Actual_Event_Name" in quals.columns else None
                if event_col:
                    events = sorted(quals[event_col].dropna().unique().tolist())
                    st.markdown(f"Tracking {len(events)} events: {', '.join(events[:20])}")

# ── Upcoming Tab ──────────────────────────────────────────────────────

with tabs[1]:
    render_section_header("Upcoming Competitions", "Next events on the athletics calendar")

    df_upcoming = dc.get_upcoming_competitions()
    if len(df_upcoming) > 0:
        for _, comp in df_upcoming.head(20).iterrows():
            with st.container():
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.markdown(f"**{comp.get('name', 'Unknown')}**")
                    st.caption(f"{comp.get('venue', '')} | {comp.get('area', '')}")
                with col2:
                    st.markdown(f"📅 {comp.get('start_date', 'TBD')}")
                with col3:
                    cat = comp.get("ranking_category", "")
                    if cat:
                        st.markdown(f"🏷️ {cat}")
                st.divider()
    else:
        st.info("No upcoming competition data.")

# ── Competition Viewer Tab ────────────────────────────────────────────

with tabs[2]:
    render_section_header("Competition Viewer", "View results for a specific competition")

    st.info(
        "To view competition results, you need to scrape them first:\n\n"
        "```bash\n"
        "python -m scrapers.scrape_competitions --results --comp-id <ID>\n"
        "```\n\n"
        "Find competition IDs from the Calendar tab above."
    )

    # Check for any scraped competition results
    import glob
    from pathlib import Path
    scraped_dir = Path(__file__).parent.parent / "data" / "scraped"
    comp_files = list(scraped_dir.glob("comp_results_*.parquet"))

    if comp_files:
        selected_file = st.selectbox(
            "Available Competition Results",
            [f.stem.replace("comp_results_", "Competition ") for f in comp_files],
            key="cv_file",
        )

        import pandas as pd
        idx = [f.stem.replace("comp_results_", "Competition ") for f in comp_files].index(selected_file)
        df_results = pd.read_parquet(comp_files[idx])

        if len(df_results) > 0:
            comp_name = df_results["competition_name"].iloc[0] if "competition_name" in df_results.columns else "Unknown"
            st.markdown(f"### {comp_name}")

            # Filter by event
            if "event" in df_results.columns:
                events = sorted(df_results["event"].unique())
                selected_event = st.selectbox("Filter by event", ["All"] + events, key="cv_event")
                if selected_event != "All":
                    df_results = df_results[df_results["event"] == selected_event]

            display_cols = ["event", "gender", "phase", "place", "athlete", "country", "mark", "wind"]
            available_cols = [c for c in display_cols if c in df_results.columns]

            st.dataframe(
                df_results[available_cols],
                hide_index=True,
                height=500,
            )

# ── Best Points Competitions Tab ──────────────────────────────────────

with tabs[3]:
    render_section_header(
        "Best Points Competitions",
        "Which competitions offer the most ranking points for KSA athletes?"
    )

    import pandas as pd
    from components.charts import competition_points_chart

    # WA ranking points by competition category (official WA scoring)
    CATEGORY_POINTS = {
        # WA category codes (as stored in calendar.parquet)
        "OW": {"place_pts": "80-100", "avg": 100, "tier": "Major"},
        "DF": {"place_pts": "80-100", "avg": 90, "tier": "Platinum"},
        "GW": {"place_pts": "60-80", "avg": 70, "tier": "Gold"},
        "GL": {"place_pts": "60-80", "avg": 65, "tier": "Gold"},
        "A": {"place_pts": "40-60", "avg": 50, "tier": "Silver"},
        "B": {"place_pts": "20-40", "avg": 30, "tier": "Bronze"},
        "C": {"place_pts": "10-20", "avg": 15, "tier": "Bronze"},
        "D": {"place_pts": "5-10", "avg": 8, "tier": "Basic"},
        "E": {"place_pts": "2-5", "avg": 4, "tier": "Basic"},
        "F": {"place_pts": "0-2", "avg": 1, "tier": "Basic"},
        # Legacy label-based keys (fallback)
        "Diamond League": {"place_pts": "80-100", "avg": 90, "tier": "Platinum"},
        "Gold": {"place_pts": "60-80", "avg": 70, "tier": "Gold"},
        "Silver": {"place_pts": "40-60", "avg": 50, "tier": "Silver"},
        "Bronze": {"place_pts": "20-40", "avg": 30, "tier": "Bronze"},
        "World Championships": {"place_pts": "80-100", "avg": 100, "tier": "Major"},
        "Olympic Games": {"place_pts": "80-100", "avg": 100, "tier": "Major"},
        "Asian Games": {"place_pts": "60-80", "avg": 75, "tier": "Continental"},
    }

    # WA Competition Category definitions (A-F system)
    COMPETITION_CATEGORIES = {
        "OW": {
            "name": "Olympic/World Championships",
            "color": "#FFD700",
            "bg": "rgba(255, 215, 0, 0.15)",
            "place_pts": 170,
            "desc": "Olympic Games & World Athletics Championships. Highest points available.",
            "examples": "Olympics, World Championships",
        },
        "DF": {
            "name": "Diamond League Final",
            "color": "#C0C0C0",
            "bg": "rgba(192, 192, 192, 0.12)",
            "place_pts": 140,
            "desc": "Diamond League Final. Elite invitational with top world athletes.",
            "examples": "Wanda Diamond League Final",
        },
        "GW": {
            "name": "Gold (World)",
            "color": "#a08e66",
            "bg": "rgba(160, 142, 102, 0.12)",
            "place_pts": 100,
            "desc": "Continental Tour Gold & Diamond League regular season. Top-tier circuit meets.",
            "examples": "Diamond League meets, Continental Tour Gold events",
        },
        "GL": {
            "name": "Gold Label (Road)",
            "color": "#a08e66",
            "bg": "rgba(160, 142, 102, 0.10)",
            "place_pts": 80,
            "desc": "World Athletics Gold Label road races. Premium road running events.",
            "examples": "Major marathons, elite road races",
        },
        "A": {
            "name": "Category A",
            "color": "#007167",
            "bg": "rgba(0, 113, 103, 0.12)",
            "place_pts": 60,
            "desc": "Continental Tour Silver, Area Championships, & multi-sport games. Good points for regional athletes.",
            "examples": "Asian Games, Asian Championships, Continental Tour Silver",
        },
        "B": {
            "name": "Category B",
            "color": "#009688",
            "bg": "rgba(0, 150, 136, 0.10)",
            "place_pts": 40,
            "desc": "Continental Tour Bronze & National Championships. Accessible points for development athletes.",
            "examples": "National Championships, Continental Tour Bronze, Area permits",
        },
        "C": {
            "name": "Category C",
            "color": "#78909C",
            "bg": "rgba(120, 144, 156, 0.10)",
            "place_pts": 20,
            "desc": "National permit meetings. Lower-tier but still ranking-eligible.",
            "examples": "National meets, permit meetings",
        },
        "D": {
            "name": "Category D",
            "color": "#90A4AE",
            "bg": "rgba(144, 164, 174, 0.08)",
            "place_pts": 10,
            "desc": "Minor national events. Minimal place points but still WA-sanctioned.",
            "examples": "Regional permits, club meetings",
        },
        "E": {
            "name": "Category E",
            "color": "#B0BEC5",
            "bg": "rgba(176, 190, 197, 0.08)",
            "place_pts": 5,
            "desc": "Lower-tier national events. Minimal ranking impact.",
            "examples": "Lower-tier national events",
        },
        "F": {
            "name": "Category F",
            "color": "#CFD8DC",
            "bg": "rgba(207, 216, 220, 0.06)",
            "place_pts": 0,
            "desc": "Basic-level events. Result score only, no place points bonus.",
            "examples": "Local meets, uncategorised events",
        },
    }

    # ── Sub-tabs inside Best Points ──
    bp_tabs = st.tabs(["Competition Categories", "Points Strategy", "KSA Athlete Scoring", "Competition Targets"])

    # ── Sub-tab 0: Competition Categories ──
    with bp_tabs[0]:
        st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, #005a51 100%);
     padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #a08e66;">
    <h4 style="color: white; margin: 0;">WA Competition Categories (A-F)</h4>
    <p style="color: rgba(255,255,255,0.85); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
        World Athletics categorises competitions from OW (highest) to F (lowest) — higher categories = more ranking points
    </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
**How Categories Work:**
- Every WA-sanctioned competition has a **category** (OW, DF, GW, GL, A, B, C, D, E, F)
- Category determines the **place points bonus** added to your result score
- Higher-category meets give more bonus points for the same performance
- **KSA athletes** should target **Category A or higher** for maximum ranking gains
""")

        # Render category cards
        for code, cat in COMPETITION_CATEGORIES.items():
            st.markdown(f"""
<div style="background: {cat['bg']}; border-left: 4px solid {cat['color']};
     padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem; display: flex; align-items: center;">
    <div style="min-width: 60px; text-align: center;">
        <span style="font-size: 1.4rem; font-weight: 800; color: {cat['color']};">{code}</span>
    </div>
    <div style="flex: 1; margin-left: 0.75rem;">
        <p style="margin: 0; font-weight: 600; font-size: 0.95rem; color: #333;">
            {cat['name']}
            <span style="float: right; background: {cat['color']}; color: white; padding: 1px 10px;
                 border-radius: 12px; font-size: 0.8rem; font-weight: 500;">
                +{cat['place_pts']} pts
            </span>
        </p>
        <p style="margin: 2px 0 0 0; font-size: 0.85rem; color: #555;">{cat['desc']}</p>
        <p style="margin: 2px 0 0 0; font-size: 0.8rem; color: #888;"><em>e.g. {cat['examples']}</em></p>
    </div>
</div>
""", unsafe_allow_html=True)

        # Show calendar breakdown by category
        df_cal_cats = dc.get_calendar()
        if len(df_cal_cats) > 0 and "ranking_category" in df_cal_cats.columns:
            st.markdown("---")
            st.markdown("##### Competition Calendar by Category")

            cat_counts = df_cal_cats["ranking_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]

            # Add category metadata
            cat_counts["Full Name"] = cat_counts["Category"].map(
                lambda c: COMPETITION_CATEGORIES.get(c, {}).get("name", c)
            )
            cat_counts["Place Points"] = cat_counts["Category"].map(
                lambda c: COMPETITION_CATEGORIES.get(c, {}).get("place_pts", 0)
            )
            cat_counts = cat_counts.sort_values("Place Points", ascending=False)

            # Convert numeric columns to int for clean display
            cat_display = cat_counts[["Category", "Full Name", "Place Points", "Count"]].copy()
            for col in ["Place Points", "Count"]:
                if col in cat_display.columns:
                    cat_display[col] = pd.to_numeric(cat_display[col], errors="coerce").fillna(0).astype(int)

            st.dataframe(
                cat_display,
                hide_index=True,
                column_config={
                    "Category": st.column_config.TextColumn("Code"),
                    "Full Name": st.column_config.TextColumn("Category Name", width="large"),
                    "Place Points": st.column_config.NumberColumn("Place Pts"),
                    "Count": st.column_config.NumberColumn("Competitions"),
                },
            )

            # Visual distribution
            import plotly.graph_objects as go
            cat_order = ["OW", "DF", "GW", "GL", "A", "B", "C", "D", "E", "F"]
            cat_counts_ordered = cat_counts.set_index("Category").reindex(cat_order).dropna()
            colors = [COMPETITION_CATEGORIES.get(c, {}).get("color", "#ccc") for c in cat_counts_ordered.index]

            fig = go.Figure(go.Bar(
                x=cat_counts_ordered.index,
                y=cat_counts_ordered["Count"],
                marker_color=colors,
                text=cat_counts_ordered["Count"].astype(int),
                textposition="outside",
            ))
            fig.update_layout(
                title="Competition Distribution by Category",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter, sans-serif", color="#333"),
                margin=dict(l=10, r=10, t=40, b=30),
                xaxis_title="Category",
                yaxis_title="Number of Competitions",
                height=350,
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
            st.plotly_chart(fig, use_container_width=True)

    # ── Sub-tab 1: Points Strategy ──
    with bp_tabs[1]:
        st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, #005a51 100%);
     padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #a08e66;">
    <h4 style="color: white; margin: 0;">WA Ranking Points Guide</h4>
    <p style="color: rgba(255,255,255,0.85); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
        How World Athletics ranking points work and where to maximise them
    </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
**How WA Ranking Points Work:**
- Points = **Result Score** (performance quality) + **Place Score** (competition level)
- Result Score is based on IAAF Scoring Tables — faster/further = more points
- Place Score depends on **competition category** — higher-tier meets = more bonus points
- Only your **top 5 scoring performances** count toward world ranking

**Priority for KSA athletes:** Target the highest-category competition available for your events.
A 10.15s 100m at Diamond League scores MORE than the same 10.15s at a Bronze meet.
""")

        # Points reference table
        ref_data = []
        for cat, info in CATEGORY_POINTS.items():
            ref_data.append({
                "Category": cat,
                "Tier": info["tier"],
                "Place Points Range": info["place_pts"],
                "Avg Points": info["avg"],
            })
        ref_df = pd.DataFrame(ref_data)

        st.dataframe(
            ref_df.sort_values("Avg Points", ascending=False),
            hide_index=True,
            column_config={
                "Category": st.column_config.TextColumn("Competition Category", width="large"),
                "Tier": st.column_config.TextColumn("Tier"),
                "Place Points Range": st.column_config.TextColumn("Place Points"),
                "Avg Points": st.column_config.ProgressColumn("Avg Points", min_value=0, max_value=100),
            },
        )

        # Calendar category chart
        df_cal = dc.get_calendar()
        if len(df_cal) > 0 and "ranking_category" in df_cal.columns:
            cat_counts = df_cal["ranking_category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            cat_counts["avg_points"] = cat_counts["category"].map(
                lambda c: CATEGORY_POINTS.get(c, {}).get("avg", 25)
            )
            cat_counts = cat_counts.sort_values("avg_points", ascending=False)

            fig = competition_points_chart(cat_counts.head(10))
            st.plotly_chart(fig, use_container_width=True)

    # ── Sub-tab 2: KSA Athlete Scoring ──
    with bp_tabs[2]:
        st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, #005a51 100%);
     padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #a08e66;">
    <h4 style="color: white; margin: 0;">KSA Athlete Best Scoring Performances</h4>
    <p style="color: rgba(255,255,255,0.85); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
        Where each athlete earns their highest WA ranking points
    </p>
</div>
""", unsafe_allow_html=True)

        athletes = dc.get_ksa_athletes()
        if athletes is not None and not athletes.empty:
            name_col = "full_name" if "full_name" in athletes.columns else "competitor"
            athlete_names = sorted(athletes[name_col].dropna().unique().tolist())

            selected_athlete = st.selectbox(
                "Select KSA Athlete", athlete_names, key="bp_athlete",
            )

            if selected_athlete:
                # Get all results for this athlete (with scoring data)
                all_results = dc.get_ksa_results(athlete_name=selected_athlete, limit=200)

                if all_results is not None and not all_results.empty:
                    # Find the score column
                    score_col = None
                    for col in ["result_score", "resultscore", "score"]:
                        if col in all_results.columns:
                            score_col = col
                            break

                    mark_col = "mark" if "mark" in all_results.columns else "result"
                    date_col = "date" if "date" in all_results.columns else None
                    comp_col = "competition" if "competition" in all_results.columns else "venue"
                    disc_col = "discipline" if "discipline" in all_results.columns else "event"
                    place_col = "place" if "place" in all_results.columns else "pos" if "pos" in all_results.columns else None

                    if score_col:
                        all_results[score_col] = pd.to_numeric(all_results[score_col], errors="coerce")
                        scored = all_results.dropna(subset=[score_col])
                        scored = scored[scored[score_col] > 0].sort_values(score_col, ascending=False)

                        if not scored.empty:
                            # Summary metrics
                            m1, m2, m3, m4 = st.columns(4)
                            with m1:
                                render_metric_card(
                                    "Best WA Points", f"{scored[score_col].max():.0f}", "excellent",
                                )
                            with m2:
                                render_metric_card(
                                    "Avg Top 5 Points",
                                    f"{scored[score_col].head(5).mean():.0f}",
                                    "good",
                                )
                            with m3:
                                render_metric_card(
                                    "Total Performances", str(len(scored)), "neutral",
                                )
                            with m4:
                                events_scored = scored[disc_col].nunique() if disc_col in scored.columns else 0
                                render_metric_card(
                                    "Events Scored", str(events_scored), "neutral",
                                )

                            # Top 10 scoring performances
                            st.markdown("##### Top Scoring Performances")
                            display_data = []
                            for _, row in scored.head(10).iterrows():
                                display_data.append({
                                    "Event": str(row.get(disc_col, "-")),
                                    "Mark": str(row.get(mark_col, "-")),
                                    "WA Points": int(row[score_col]) if pd.notna(row[score_col]) else 0,
                                    "Competition": str(row.get(comp_col, "-")),
                                    "Date": str(row.get(date_col, "-")) if date_col else "-",
                                    "Place": str(row.get(place_col, "-")) if place_col else "-",
                                })

                            scored_display_df = pd.DataFrame(display_data)
                            # WA Points already cast to int in display_data construction
                            st.dataframe(
                                scored_display_df,
                                hide_index=True,
                                column_config={
                                    "WA Points": st.column_config.NumberColumn(
                                        "WA Points",
                                    ),
                                    "Competition": st.column_config.TextColumn("Competition", width="large"),
                                },
                            )

                            # Points by event (best per event)
                            if disc_col in scored.columns:
                                event_best = (
                                    scored.groupby(disc_col)[score_col]
                                    .agg(["max", "mean", "count"])
                                    .reset_index()
                                )
                                event_best.columns = ["Event", "Best Points", "Avg Points", "# Results"]
                                event_best = event_best.sort_values("Best Points", ascending=False)

                                st.markdown("##### Best Points by Event")
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=event_best["Event"],
                                    y=event_best["Best Points"],
                                    name="Best",
                                    marker_color=TEAL_PRIMARY,
                                    text=event_best["Best Points"].apply(lambda x: f"{x:.0f}"),
                                    textposition="outside",
                                ))
                                fig.add_trace(go.Bar(
                                    x=event_best["Event"],
                                    y=event_best["Avg Points"],
                                    name="Average",
                                    marker_color="#a08e66",
                                    text=event_best["Avg Points"].apply(lambda x: f"{x:.0f}"),
                                    textposition="outside",
                                ))
                                fig.update_layout(
                                    barmode="group",
                                    plot_bgcolor="white",
                                    paper_bgcolor="white",
                                    font=dict(family="Inter, sans-serif", color="#333"),
                                    margin=dict(l=10, r=10, t=40, b=30),
                                    title="WA Points by Event (Best vs Average)",
                                    yaxis_title="WA Points",
                                    showlegend=True,
                                    height=400,
                                )
                                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
                                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No scored performances found for this athlete.")
                    else:
                        # No score column - show results without scoring
                        st.info("WA scoring data not available in this dataset. Showing results by mark.")
                        st.dataframe(all_results.head(20), hide_index=True)
                else:
                    st.info(f"No results found for {selected_athlete}.")
        else:
            st.info("No KSA athlete data available.")

    # ── Sub-tab 3: Competition Targets ──
    with bp_tabs[3]:
        st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, #005a51 100%);
     padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #a08e66;">
    <h4 style="color: white; margin: 0;">Recommended Competition Targets</h4>
    <p style="color: rgba(255,255,255,0.85); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
        Upcoming high-value competitions sorted by ranking point potential
    </p>
</div>
""", unsafe_allow_html=True)

        df_cal = dc.get_calendar()
        if len(df_cal) > 0 and "ranking_category" in df_cal.columns:
            # Add estimated points column
            df_cal["est_points"] = df_cal["ranking_category"].map(
                lambda c: CATEGORY_POINTS.get(c, {}).get("avg", 25)
            )

            # Filter: future competitions only
            import datetime
            today = datetime.date.today().isoformat()
            if "start_date" in df_cal.columns:
                future = df_cal[df_cal["start_date"] >= today].copy()
            else:
                future = df_cal.copy()

            # Sort by est_points descending, then by date
            future = future.sort_values(
                ["est_points", "start_date"], ascending=[False, True],
            )

            # Region filter
            area_col = "area" if "area" in future.columns else None
            if area_col:
                areas = ["All Regions"] + sorted(future[area_col].dropna().unique().tolist())
                sel_area = st.selectbox("Filter by Region", areas, key="bp_area")
                if sel_area != "All Regions":
                    future = future[future[area_col] == sel_area]

            # Category filter
            cat_options = ["All Categories"] + sorted(future["ranking_category"].dropna().unique().tolist())
            sel_cat = st.selectbox("Filter by Category", cat_options, key="bp_cat2")
            if sel_cat != "All Categories":
                future = future[future["ranking_category"] == sel_cat]

            if len(future) > 0:
                st.markdown(f"**{len(future)} upcoming competitions** (sorted by point potential)")

                display_cols = ["name", "venue", "start_date", "ranking_category", "est_points"]
                if area_col:
                    display_cols.insert(2, area_col)
                available_cols = [c for c in display_cols if c in future.columns]

                # Convert numeric columns to int for clean display
                future_display = future[available_cols].head(50).copy()
                if "est_points" in future_display.columns:
                    future_display["est_points"] = pd.to_numeric(future_display["est_points"], errors="coerce").fillna(0).astype(int)

                st.dataframe(
                    future_display,
                    hide_index=True,
                    column_config={
                        "name": st.column_config.TextColumn("Competition", width="large"),
                        "venue": st.column_config.TextColumn("Venue"),
                        "area": st.column_config.TextColumn("Region"),
                        "start_date": st.column_config.TextColumn("Date"),
                        "ranking_category": st.column_config.TextColumn("Category"),
                        "est_points": st.column_config.NumberColumn(
                            "Est. Points",
                        ),
                    },
                    height=600,
                )
            else:
                st.info("No upcoming competitions match the selected filters.")
        else:
            st.info(
                "No calendar data available. "
                "Run: `python -m scrapers.scrape_competitions` to populate."
            )

# ── Qualification Tracking Tab (legacy road_to_tokyo data) ────────────

if has_road_to_tokyo and not has_calendar and len(tab_names) > 4:
    with tabs[4]:
        render_section_header(
            "Qualification Tracking",
            "Championship qualification status from legacy data"
        )

        quals = dc.get_qualifications()
        if len(quals) > 0:
            # Show summary metrics
            total_athletes = 0
            qualified_count = 0
            if "Athlete" in quals.columns:
                total_athletes = quals["Athlete"].nunique()
            if "Qualification_Status" in quals.columns:
                qualified_count = len(quals[quals["Qualification_Status"].str.lower().str.contains("qualified", na=False)])

            m1, m2, m3 = st.columns(3)
            with m1:
                render_metric_card("Total Athletes", str(total_athletes), "good")
            with m2:
                render_metric_card("Qualified", str(qualified_count), "excellent" if qualified_count > 0 else "neutral")
            with m3:
                events_tracked = quals["Actual_Event_Name"].nunique() if "Actual_Event_Name" in quals.columns else 0
                render_metric_card("Events Tracked", str(events_tracked), "neutral")

            # Event filter
            if "Actual_Event_Name" in quals.columns:
                event_options = ["All Events"] + sorted(quals["Actual_Event_Name"].dropna().unique().tolist())
                selected_event = st.selectbox("Filter by Event", event_options, key="qt_event")
                if selected_event != "All Events":
                    quals = quals[quals["Actual_Event_Name"] == selected_event]

            # Display columns from road_to_tokyo schema
            display_cols = ["Athlete", "Actual_Event_Name", "Federation", "Qualification_Status",
                           "QP", "FP", "Status", "Details"]
            available_cols = [c for c in display_cols if c in quals.columns]

            st.dataframe(
                quals[available_cols],
                hide_index=True,
                column_config={
                    "Athlete": st.column_config.TextColumn("Athlete", width="medium"),
                    "Actual_Event_Name": st.column_config.TextColumn("Event"),
                    "Federation": st.column_config.TextColumn("Country"),
                    "Qualification_Status": st.column_config.TextColumn("Status"),
                    "QP": st.column_config.TextColumn("Qual Points"),
                    "FP": st.column_config.TextColumn("Final Points"),
                    "Status": st.column_config.TextColumn("Detail"),
                    "Details": st.column_config.TextColumn("Notes", width="large"),
                },
                height=500,
            )
        else:
            st.info("No qualification data available.")

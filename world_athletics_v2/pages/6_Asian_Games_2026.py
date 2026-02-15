"""
Asian Games 2026 - Project East strategy page.

Features:
- Medal target tracker (3-5 medals)
- Per-athlete readiness matrix with gap analysis
- Asian Games championship targets per athlete
- Progress tracking with form trends
- Asian rivals per event
- Countdown
"""

import pandas as pd
import streamlit as st
from datetime import datetime

from components.theme import (
    get_theme_css, render_page_header, render_section_header,
    render_metric_card, render_sidebar, TEAL_PRIMARY, GOLD_ACCENT,
    TEAL_LIGHT,
)
from components.report_components import (
    ASIAN_GAMES_2026_TARGETS,
    _normalize_event_for_lookup,
    format_mark_display,
)
from data.connector import get_connector
from data.event_utils import ASIAN_COUNTRY_CODES, get_event_type, format_event_name

# ── Page Setup ────────────────────────────────────────────────────────

st.set_page_config(page_title="Asian Games 2026", page_icon="🏅", layout="wide")
st.markdown(get_theme_css(), unsafe_allow_html=True)
render_sidebar()

render_page_header(
    "Project East 2026",
    "Asian Games Aichi-Nagoya | 19 September - 4 October 2026"
)

dc = get_connector()

# ── Countdown + Overview Metrics ──────────────────────────────────────

asian_games_date = datetime(2026, 9, 19)
days_to_go = (asian_games_date - datetime.now()).days

ksa = dc.get_ksa_athletes()
ranked = 0
event_count = 0
if len(ksa) > 0:
    if "best_world_rank" in ksa.columns:
        ranked = len(ksa[ksa["best_world_rank"].notna()])
    if "primary_event" in ksa.columns:
        event_count = len(ksa["primary_event"].dropna().unique())

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    color = "excellent" if days_to_go > 200 else ("warning" if days_to_go > 100 else "danger")
    render_metric_card("Days to Go", str(max(days_to_go, 0)), color)
with col2:
    render_metric_card("Medal Target", "3-5", "gold")
with col3:
    render_metric_card("Ranked Athletes", str(ranked), "good")
with col4:
    render_metric_card("Events Covered", str(event_count), "neutral")
with col5:
    render_metric_card("Host City", "Nagoya", "neutral")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Athlete Readiness",
    "Individual Progress",
    "Asian Rivals",
    "Medal Pathways",
])


# ════════════════════════════════════════════════════════════════════════
# TAB 1: Athlete Readiness Matrix
# ════════════════════════════════════════════════════════════════════════

with tabs[0]:
    render_section_header(
        "Athlete Readiness Matrix",
        "KSA athletes' current marks vs Asian Games 2026 medal targets"
    )

    if len(ksa) == 0:
        st.warning("No KSA athlete data loaded.")
    else:
        # Build readiness data for each athlete
        readiness_rows = []
        name_col = "full_name" if "full_name" in ksa.columns else "competitor"

        for _, athlete in ksa.iterrows():
            name = athlete.get(name_col, "Unknown")
            primary_event = athlete.get("primary_event", "")
            world_rank = athlete.get("best_world_rank")
            score = athlete.get("best_ranking_score", athlete.get("best_score"))

            if not primary_event or pd.isna(primary_event):
                continue

            # Look up Asian Games targets for this event
            event_display = format_event_name(str(primary_event))
            event_lookup = _normalize_event_for_lookup(event_display)
            targets = ASIAN_GAMES_2026_TARGETS.get(event_lookup)

            # Get athlete PB
            pbs = dc.get_ksa_athlete_pbs(name)
            pb_mark = None
            if pbs is not None and not pbs.empty:
                disc_col = "discipline" if "discipline" in pbs.columns else "event"
                mark_col_pb = "mark" if "mark" in pbs.columns else "result"
                if disc_col in pbs.columns and mark_col_pb in pbs.columns:
                    event_norm = str(primary_event).lower().replace(" ", "")
                    for _, pb_row in pbs.iterrows():
                        disc = str(pb_row.get(disc_col, "")).lower().replace(" ", "")
                        if event_norm in disc:
                            val = pd.to_numeric(pb_row.get(mark_col_pb), errors="coerce")
                            if not pd.isna(val):
                                pb_mark = float(val)
                                break

            if targets and pb_mark is not None:
                event_type = get_event_type(str(primary_event))
                lower_is_better = event_type == "time"

                gold_mark = targets["gold"]
                medal_mark = targets["medal"]
                final_mark = targets["final"]

                # Calculate gaps
                if lower_is_better:
                    gold_gap = pb_mark - gold_mark
                    medal_gap = pb_mark - medal_mark
                    final_gap = pb_mark - final_mark
                else:
                    gold_gap = gold_mark - pb_mark
                    medal_gap = medal_mark - pb_mark
                    final_gap = final_mark - pb_mark

                # Determine readiness status
                if gold_gap <= 0 if lower_is_better else gold_gap <= 0:
                    status = "Gold Contender"
                    status_color = "excellent"
                elif medal_gap <= 0 if lower_is_better else medal_gap <= 0:
                    status = "Medal Range"
                    status_color = "good"
                elif final_gap <= 0 if lower_is_better else final_gap <= 0:
                    status = "Finalist"
                    status_color = "warning"
                elif abs(final_gap) < abs(final_mark * 0.03):
                    status = "Near Final"
                    status_color = "warning"
                else:
                    status = "Development"
                    status_color = "neutral"

                readiness_rows.append({
                    "Athlete": name,
                    "Event": event_display,
                    "PB": format_mark_display(pb_mark, event_type),
                    "PB_raw": pb_mark,
                    "Gold Target": format_mark_display(gold_mark, event_type),
                    "Gap to Gold": f"{abs(gold_gap):.2f}" if gold_gap > 0 else "Achieved",
                    "Medal Target": format_mark_display(medal_mark, event_type),
                    "Gap to Medal": f"{abs(medal_gap):.2f}" if medal_gap > 0 else "Achieved",
                    "Final Target": format_mark_display(final_mark, event_type),
                    "Gap to Final": f"{abs(final_gap):.2f}" if final_gap > 0 else "Achieved",
                    "Status": status,
                    "Status_color": status_color,
                    "World Rank": int(world_rank) if pd.notna(world_rank) else 9999,
                })

        if readiness_rows:
            df_ready = pd.DataFrame(readiness_rows)
            # Sort: Gold Contenders first, then Medal Range, etc.
            status_order = {"Gold Contender": 0, "Medal Range": 1, "Finalist": 2, "Near Final": 3, "Development": 4}
            df_ready["_sort"] = df_ready["Status"].map(status_order)
            df_ready = df_ready.sort_values(["_sort", "World Rank"])

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                gold_count = len(df_ready[df_ready["Status"] == "Gold Contender"])
                render_metric_card("Gold Contenders", str(gold_count), "excellent" if gold_count > 0 else "neutral")
            with m2:
                medal_count = len(df_ready[df_ready["Status"].isin(["Gold Contender", "Medal Range"])])
                render_metric_card("Medal Range", str(medal_count), "good" if medal_count > 0 else "neutral")
            with m3:
                finalist_count = len(df_ready[df_ready["Status"].isin(["Gold Contender", "Medal Range", "Finalist"])])
                render_metric_card("Potential Finalists", str(finalist_count), "good" if finalist_count > 0 else "neutral")
            with m4:
                render_metric_card("Athletes Tracked", str(len(df_ready)), "neutral")

            # Display table
            display_cols = ["Athlete", "Event", "PB", "Gold Target", "Gap to Gold",
                           "Medal Target", "Gap to Medal", "Status"]
            st.dataframe(
                df_ready[display_cols],
                hide_index=True,
                column_config={
                    "Athlete": st.column_config.TextColumn("Athlete", width="medium"),
                    "Event": st.column_config.TextColumn("Event"),
                    "PB": st.column_config.TextColumn("PB"),
                    "Gold Target": st.column_config.TextColumn("Gold"),
                    "Gap to Gold": st.column_config.TextColumn("Gap to Gold"),
                    "Medal Target": st.column_config.TextColumn("Medal"),
                    "Gap to Medal": st.column_config.TextColumn("Gap to Medal"),
                    "Status": st.column_config.TextColumn("Readiness"),
                },
                height=600,
            )

            # Status breakdown
            st.markdown("##### Readiness Breakdown")
            for status_name, color_hex, emoji in [
                ("Gold Contender", TEAL_PRIMARY, "🥇"),
                ("Medal Range", TEAL_LIGHT, "🥈"),
                ("Finalist", GOLD_ACCENT, "📍"),
                ("Near Final", "#FFB800", "🔜"),
                ("Development", "#78909C", "📈"),
            ]:
                count = len(df_ready[df_ready["Status"] == status_name])
                if count > 0:
                    athletes = ", ".join(df_ready[df_ready["Status"] == status_name]["Athlete"].tolist())
                    st.markdown(
                        f"<div style='padding: 6px 12px; margin: 4px 0; border-left: 4px solid {color_hex}; "
                        f"background: #f8f9fa; border-radius: 4px;'>"
                        f"<strong>{emoji} {status_name}</strong> ({count}): {athletes}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No athletes matched to Asian Games events. Check that athletes have primary events set.")

# ════════════════════════════════════════════════════════════════════════
# TAB 2: Individual Progress
# ════════════════════════════════════════════════════════════════════════

with tabs[1]:
    render_section_header(
        "Individual Athlete Progress",
        "Detailed progress tracker for each KSA athlete toward Asian Games targets"
    )

    if len(ksa) == 0:
        st.warning("No KSA athlete data loaded.")
    else:
        name_col = "full_name" if "full_name" in ksa.columns else "competitor"
        athlete_names = sorted(ksa[name_col].dropna().unique().tolist())

        selected_athlete = st.selectbox("Select Athlete", athlete_names, key="ag_ind_athlete")

        if selected_athlete:
            athlete_row = ksa[ksa[name_col] == selected_athlete].iloc[0]

            # Hero section
            st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, #005a51 100%);
     padding: 1.25rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {GOLD_ACCENT};">
    <h3 style="color: white; margin: 0;">{selected_athlete}</h3>
    <p style="color: rgba(255,255,255,0.85); margin: 0.25rem 0 0 0;">
        Primary Event: <strong>{athlete_row.get('primary_event', 'N/A')}</strong> |
        World Rank: <strong>#{athlete_row.get('best_world_rank', 'N/R')}</strong> |
        {days_to_go} days to Asian Games
    </p>
</div>
""", unsafe_allow_html=True)

            # Get all PBs for this athlete
            all_pbs = dc.get_ksa_athlete_pbs(selected_athlete)
            all_results = dc.get_ksa_results(athlete_name=selected_athlete, limit=100)

            if all_pbs is not None and not all_pbs.empty:
                disc_col = "discipline" if "discipline" in all_pbs.columns else "event"
                mark_col_pb = "mark" if "mark" in all_pbs.columns else "result"

                # Build target cards for each event that has Asian Games targets
                st.markdown("##### Asian Games Target Cards")

                event_cards = []
                for _, pb_row in all_pbs.iterrows():
                    event_raw = str(pb_row.get(disc_col, ""))
                    event_display = format_event_name(event_raw)
                    event_lookup = _normalize_event_for_lookup(event_display)
                    targets = ASIAN_GAMES_2026_TARGETS.get(event_lookup)

                    if not targets:
                        continue

                    pb_val = pd.to_numeric(pb_row.get(mark_col_pb), errors="coerce")
                    if pd.isna(pb_val):
                        continue

                    event_type = get_event_type(event_raw)
                    lower_is_better = event_type == "time"

                    event_cards.append({
                        "event_raw": event_raw,
                        "event_display": event_display,
                        "event_type": event_type,
                        "lower_is_better": lower_is_better,
                        "pb": float(pb_val),
                        "targets": targets,
                    })

                if event_cards:
                    for card in event_cards:
                        et = card["event_type"]
                        lib = card["lower_is_better"]
                        pb = card["pb"]
                        targets = card["targets"]

                        gold_mark = targets["gold"]
                        medal_mark = targets["medal"]
                        final_mark = targets["final"]

                        # Calculate gaps
                        if lib:
                            gold_gap = pb - gold_mark
                            medal_gap = pb - medal_mark
                            final_gap = pb - final_mark
                        else:
                            gold_gap = gold_mark - pb
                            medal_gap = medal_mark - pb
                            final_gap = final_mark - pb

                        with st.expander(f"**{card['event_display']}** — PB: {format_mark_display(pb, et)}", expanded=True):
                            # Target metric cards
                            tc1, tc2, tc3 = st.columns(3)
                            with tc1:
                                if gold_gap <= 0:
                                    render_metric_card("Gold", format_mark_display(gold_mark, et), "excellent")
                                    st.caption("Achieved!")
                                else:
                                    render_metric_card("Gold", format_mark_display(gold_mark, et), "warning")
                                    st.caption(f"{abs(gold_gap):.2f} to go")
                            with tc2:
                                if medal_gap <= 0:
                                    render_metric_card("Medal", format_mark_display(medal_mark, et), "excellent")
                                    st.caption("Achieved!")
                                else:
                                    render_metric_card("Medal", format_mark_display(medal_mark, et), "warning")
                                    st.caption(f"{abs(medal_gap):.2f} to go")
                            with tc3:
                                if final_gap <= 0:
                                    render_metric_card("Final", format_mark_display(final_mark, et), "excellent")
                                    st.caption("Achieved!")
                                else:
                                    render_metric_card("Final", format_mark_display(final_mark, et), "warning")
                                    st.caption(f"{abs(final_gap):.2f} to go")

                            # Progress bar visualization
                            if lib:
                                # Time event: progress = how close PB is to gold (lower = better)
                                worst = final_mark * 1.10  # 10% beyond final as baseline
                                if pb <= gold_mark:
                                    pct = 100
                                elif pb >= worst:
                                    pct = 0
                                else:
                                    pct = max(0, min(100, (worst - pb) / (worst - gold_mark) * 100))
                            else:
                                # Field event: progress = how close PB is to gold (higher = better)
                                worst = final_mark * 0.90
                                if pb >= gold_mark:
                                    pct = 100
                                elif pb <= worst:
                                    pct = 0
                                else:
                                    pct = max(0, min(100, (pb - worst) / (gold_mark - worst) * 100))

                            bar_color = TEAL_PRIMARY if pct >= 75 else (GOLD_ACCENT if pct >= 50 else "#dc3545")
                            st.markdown(f"""
<div style="margin: 0.5rem 0;">
  <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
    <span style="font-size: 0.8rem; font-weight: 600;">Progress to Gold</span>
    <span style="font-size: 0.8rem; font-weight: 700; color: {bar_color};">{pct:.0f}%</span>
  </div>
  <div style="background: #e9ecef; border-radius: 4px; height: 14px; overflow: hidden; position: relative;">
    <div style="width: {pct:.0f}%; height: 100%; background: {bar_color}; border-radius: 4px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

                            # Recent results for this event
                            if all_results is not None and not all_results.empty:
                                res_disc_col = "discipline" if "discipline" in all_results.columns else "event"
                                event_results = all_results[
                                    all_results[res_disc_col].str.lower().str.replace(" ", "") ==
                                    card["event_raw"].lower().replace(" ", "")
                                ]
                                if len(event_results) > 0:
                                    st.markdown("**Recent Results:**")
                                    res_mark_col = "mark" if "mark" in event_results.columns else "result"
                                    res_date_col = "date" if "date" in event_results.columns else None
                                    res_comp_col = "competition" if "competition" in event_results.columns else "venue"

                                    display_data = []
                                    for _, rr in event_results.head(5).iterrows():
                                        display_data.append({
                                            "Date": str(rr.get(res_date_col, "-")) if res_date_col else "-",
                                            "Competition": str(rr.get(res_comp_col, "-")),
                                            "Mark": str(rr.get(res_mark_col, "-")),
                                        })
                                    if display_data:
                                        st.dataframe(pd.DataFrame(display_data), hide_index=True)
                else:
                    st.info("No events matched to Asian Games targets for this athlete.")
            else:
                st.info("No PB data available for this athlete.")

# ════════════════════════════════════════════════════════════════════════
# TAB 3: Asian Rivals
# ════════════════════════════════════════════════════════════════════════

with tabs[2]:
    render_section_header("Asian Rivals", "Top competitors from Asian nations by event")

    rivals = dc.get_rivals(region="asia", limit=50)
    if len(rivals) > 0:
        # Group by event
        if "event" in rivals.columns:
            events = sorted(rivals["event"].unique())
            selected_event = st.selectbox("Filter by event", ["All Events"] + events, key="ag_event")
            if selected_event != "All Events":
                rivals = rivals[rivals["event"] == selected_event]

        display_cols = ["full_name", "country_code", "event", "world_rank", "ranking_score"]
        available_cols = [c for c in display_cols if c in rivals.columns]

        st.dataframe(
            rivals[available_cols],
            hide_index=True,
            column_config={
                "full_name": st.column_config.TextColumn("Athlete", width="medium"),
                "country_code": st.column_config.TextColumn("Country"),
                "event": st.column_config.TextColumn("Event"),
                "world_rank": st.column_config.NumberColumn("World Rank", format="%d"),
                "ranking_score": st.column_config.NumberColumn("Score", format="%.0f"),
            },
            height=500,
        )
    else:
        # Fallback: get Asian athletes from world rankings
        st.info("No dedicated rival data. Showing Asian athletes from world rankings.")

        popular_events = ["100m", "200m", "400m", "800m", "1500m", "110m H", "400m H",
                          "High Jump", "Long Jump", "Shot Put", "Javelin"]

        all_asian = []
        for evt in popular_events:
            ranks = dc.get_world_rankings(event=evt, gender="M", limit=100)
            if len(ranks) > 0:
                country_col = "country" if "country" in ranks.columns else "nat" if "nat" in ranks.columns else None
                if country_col:
                    asian_only = ranks[ranks[country_col].isin(ASIAN_COUNTRY_CODES)].head(5)
                    if len(asian_only) > 0:
                        asian_only = asian_only.copy()
                        if "event" not in asian_only.columns:
                            asian_only["event_name"] = evt
                        all_asian.append(asian_only)

        if all_asian:
            combined = pd.concat(all_asian, ignore_index=True)
            st.dataframe(combined.head(50), hide_index=True, height=500)
        else:
            st.info("No Asian rival data. Run: `python -m scrapers.scrape_athletes --rivals`")

# ════════════════════════════════════════════════════════════════════════
# TAB 4: Medal Pathways
# ════════════════════════════════════════════════════════════════════════

with tabs[3]:
    render_section_header(
        "Medal Pathways",
        "Events where KSA has the best chance of Asian Games medals"
    )

    st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, #005a51 100%);
     padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {GOLD_ACCENT};">
    <h4 style="color: white; margin: 0;">Medal Pathway Strategy</h4>
    <p style="color: rgba(255,255,255,0.85); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
        Events ranked by KSA's proximity to medal standards (Hangzhou 2023 medal lines)
    </p>
</div>
""", unsafe_allow_html=True)

    if len(ksa) > 0:
        name_col = "full_name" if "full_name" in ksa.columns else "competitor"

        pathway_rows = []
        for _, athlete in ksa.iterrows():
            name = athlete.get(name_col, "Unknown")
            primary_event = athlete.get("primary_event", "")
            if not primary_event or pd.isna(primary_event):
                continue

            event_display = format_event_name(str(primary_event))
            event_lookup = _normalize_event_for_lookup(event_display)
            targets = ASIAN_GAMES_2026_TARGETS.get(event_lookup)
            if not targets:
                continue

            # Get PB
            pbs = dc.get_ksa_athlete_pbs(name)
            pb_mark = None
            if pbs is not None and not pbs.empty:
                disc_col = "discipline" if "discipline" in pbs.columns else "event"
                mark_col_pb = "mark" if "mark" in pbs.columns else "result"
                event_norm = str(primary_event).lower().replace(" ", "")
                for _, pb_row in pbs.iterrows():
                    disc = str(pb_row.get(disc_col, "")).lower().replace(" ", "")
                    if event_norm in disc:
                        val = pd.to_numeric(pb_row.get(mark_col_pb), errors="coerce")
                        if not pd.isna(val):
                            pb_mark = float(val)
                            break

            if pb_mark is None:
                continue

            event_type = get_event_type(str(primary_event))
            lower_is_better = event_type == "time"

            medal_mark = targets["medal"]
            if lower_is_better:
                medal_gap = pb_mark - medal_mark
                gap_pct = medal_gap / medal_mark * 100
            else:
                medal_gap = medal_mark - pb_mark
                gap_pct = medal_gap / medal_mark * 100

            pathway_rows.append({
                "Athlete": name,
                "Event": event_display,
                "PB": format_mark_display(pb_mark, event_type),
                "Medal Standard": format_mark_display(medal_mark, event_type),
                "Gap": f"{abs(medal_gap):.2f}" if medal_gap > 0 else "ACHIEVED",
                "Gap %": round(gap_pct, 2) if medal_gap > 0 else 0.0,
                "Medal Chance": "HIGH" if medal_gap <= 0 else ("MEDIUM" if gap_pct < 2 else ("LOW" if gap_pct < 5 else "LONG SHOT")),
                "_sort": gap_pct,
            })

        if pathway_rows:
            df_path = pd.DataFrame(pathway_rows).sort_values("_sort")

            for _, row in df_path.iterrows():
                chance = row["Medal Chance"]
                if chance == "HIGH":
                    icon, border_color = "🥇", TEAL_PRIMARY
                elif chance == "MEDIUM":
                    icon, border_color = "🎯", GOLD_ACCENT
                elif chance == "LOW":
                    icon, border_color = "📈", "#FFB800"
                else:
                    icon, border_color = "🏋️", "#78909C"

                gap_display = f"Gap: {row['Gap']}" if row["Gap"] != "ACHIEVED" else "MEDAL STANDARD ACHIEVED"

                st.markdown(f"""
<div style="padding: 10px 16px; margin: 6px 0; border-left: 4px solid {border_color};
     background: #f8f9fa; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <strong>{icon} {row['Athlete']}</strong> — {row['Event']}
        <br><span style="color: #666; font-size: 0.85rem;">PB: {row['PB']} | Medal: {row['Medal Standard']} | {gap_display}</span>
    </div>
    <div style="background: {border_color}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">
        {chance}
    </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No athletes matched to Asian Games events with PB data.")
    else:
        st.warning("No KSA athlete data loaded.")

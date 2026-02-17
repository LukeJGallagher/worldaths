"""
Asian Games 2026 - Project East strategy page.

Features:
- Medal target tracker (3-5 medals)
- Per-athlete readiness matrix with gap analysis (uses RECENT marks, not just PBs)
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
    TEAL_LIGHT, GRAY_BLUE,
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


# ── Helper: Get best recent mark per event from ksa_results ──────────

def _get_recent_marks(athlete_name: str) -> dict:
    """Get best recent mark per event from ksa_results (competition data).

    Returns dict: {event_display: {"sb": float, "last3_avg": float, "last_date": str, "n_results": int}}
    """
    results = dc.get_ksa_results(athlete_name=athlete_name, limit=200)
    if results is None or results.empty:
        return {}

    disc_col = "discipline" if "discipline" in results.columns else "event"
    mark_col = "mark" if "mark" in results.columns else "result"
    date_col = "date" if "date" in results.columns else None

    if disc_col not in results.columns or mark_col not in results.columns:
        return {}

    recent = {}
    for event_raw, grp in results.groupby(disc_col):
        event_display = format_event_name(str(event_raw))
        event_type = get_event_type(str(event_raw))
        lower_is_better = event_type == "time"

        marks = pd.to_numeric(grp[mark_col], errors="coerce").dropna()
        if len(marks) == 0:
            continue

        # Season best = best mark from all results
        sb = float(marks.min()) if lower_is_better else float(marks.max())

        # Last 3 average (by date order if available)
        if date_col and date_col in grp.columns:
            grp_sorted = grp.copy()
            grp_sorted["_parsed"] = pd.to_datetime(grp_sorted[date_col], format="mixed", errors="coerce")
            grp_sorted = grp_sorted.sort_values("_parsed", ascending=False)
            last3_marks = pd.to_numeric(grp_sorted[mark_col].head(3), errors="coerce").dropna()
            last_date = str(grp_sorted[date_col].iloc[0]) if len(grp_sorted) > 0 else None
        else:
            last3_marks = marks.tail(3)
            last_date = None

        last3_avg = float(last3_marks.mean()) if len(last3_marks) > 0 else None

        recent[event_display] = {
            "sb": sb,
            "last3_avg": last3_avg,
            "last_date": last_date,
            "n_results": len(marks),
        }

    return recent


def _get_best_current_mark(pb_mark, recent_data, event_display, event_type):
    """Get the best current mark: use SB if available and better than PB, else PB.

    For time events: lower is better. For field: higher is better.
    """
    lower_is_better = event_type == "time"
    best = pb_mark
    source = "PB"

    if event_display in recent_data:
        sb = recent_data[event_display].get("sb")
        if sb is not None:
            if lower_is_better:
                if sb < best:
                    best = sb
                    source = "SB"
            else:
                if sb > best:
                    best = sb
                    source = "SB"

    return best, source


# ── Tabs ──────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Project East Squad",
    "Athlete Readiness",
    "Individual Progress",
    "Asian Rivals",
    "Medal Pathways",
])


# ════════════════════════════════════════════════════════════════════════
# TAB 0: Project East Squad
# ════════════════════════════════════════════════════════════════════════

# Core Project East athletes from PDF strategy document
# db_name links to scraped data; updated dynamically where possible
PROJECT_EAST_SQUAD = [
    {"name": "Abdulaziz Abdul ATAFI", "db_name": "Abdulaziz Abdui ATAFI", "event": "200m", "medal_std": "20.65", "status": "Medal Zone", "project": "Project Sprints"},
    {"name": "Hussain Asim AL HIZAM", "db_name": "Hussain Asim AL HIZAM", "event": "Pole Vault", "medal_std": "5.58m", "status": "Medal Zone", "project": "Project Hussain"},
    {"name": "Mohamed Daouda TOLO", "db_name": "Mohammed Daoud B TOLU", "event": "Shot Put", "medal_std": "19.69m", "status": "Medal Zone", "project": "Project Tolu"},
    {"name": "Sami BAKHEET", "db_name": "Sami BAKHEET", "event": "Triple Jump", "medal_std": "16.62m", "status": "Medal Zone", "project": "Project Sami"},
    {"name": "Yusuf BIZIMANA", "db_name": None, "event": "800m", "medal_std": "1:47.5", "status": "Medal Zone", "project": "Project 800m"},
    {"name": "Mohammed AL DUBAISI", "db_name": "Mohammed AL DUBAISI", "event": "Hammer Throw", "medal_std": "72.43m", "status": "Rising", "project": "Project Al Dubaisi"},
    {"name": "Mohammed AL MUAWI", "db_name": "Mohammed Duhaim AL MUAWI", "event": "400m Hurdles", "medal_std": "49.14", "status": "Rising", "project": "Project Al Muawi"},
    {"name": "Nasser MOHAMMED", "db_name": "Nasser Mahmoud MOHAMMED", "event": "100m", "medal_std": "10.06", "status": "Rising", "project": "Project Sprints"},
    {"name": "4x100m Relay Squad", "db_name": None, "event": "4x100m", "medal_std": "38.65", "status": "Rising", "project": "Project 4x100m"},
]

with tabs[0]:
    render_section_header(
        "Project East Squad",
        "9 athletes across sprint, middle-distance, and technical events targeting 3-5 medals"
    )

    # Medal Zone vs Rising
    mz_count = sum(1 for a in PROJECT_EAST_SQUAD if a["status"] == "Medal Zone")
    rising_count = sum(1 for a in PROJECT_EAST_SQUAD if a["status"] == "Rising")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        render_metric_card("Total Squad", str(len(PROJECT_EAST_SQUAD)), "neutral")
    with mc2:
        render_metric_card("Medal Zone", str(mz_count), "excellent")
    with mc3:
        render_metric_card("Rising", str(rising_count), "warning")
    with mc4:
        render_metric_card("Medal Target", "3-5", "gold")

    # Medal Zone athletes
    st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, #005a51 100%);
     padding: 0.75rem 1rem; border-radius: 6px; margin: 1rem 0 0.5rem 0;">
    <h4 style="color: white; margin: 0;">Already in Medal Zone</h4>
</div>
""", unsafe_allow_html=True)

    for athlete in PROJECT_EAST_SQUAD:
        if athlete["status"] != "Medal Zone":
            continue

        in_db = athlete["db_name"] is not None
        db_tag = "" if in_db else f" <span style='color: {GRAY_BLUE}; font-size: 0.75rem;'>NOT IN DATABASE</span>"

        # Look up live PB + SB from DB
        live_pb = "-"
        live_sb = ""
        if in_db:
            pbs = dc.get_ksa_athlete_pbs(athlete["db_name"])
            if pbs is not None and not pbs.empty:
                disc_col = "discipline" if "discipline" in pbs.columns else "event"
                mark_col = "mark" if "mark" in pbs.columns else "result"
                for _, pb_row in pbs.iterrows():
                    if format_event_name(str(pb_row.get(disc_col, ""))) == format_event_name(athlete["event"]):
                        live_pb = str(pb_row.get(mark_col, "-"))
                        break
            # Recent SB
            recent = _get_recent_marks(athlete["db_name"])
            evt_disp = format_event_name(athlete["event"])
            if evt_disp in recent:
                sb_val = recent[evt_disp].get("sb")
                if sb_val is not None:
                    et = get_event_type(athlete["event"])
                    live_sb = f" | SB: {format_mark_display(sb_val, et)}"

        st.markdown(f"""
<div style="padding: 10px 16px; margin: 4px 0; border-left: 4px solid {TEAL_PRIMARY};
     background: #f8f9fa; border-radius: 4px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong>{athlete['name']}</strong>{db_tag}
            <br><span style="color: #666; font-size: 0.85rem;">
                {athlete['event']} | PB: {live_pb}{live_sb} | Medal Std: {athlete['medal_std']} | {athlete['project']}
            </span>
        </div>
        <div style="background: {TEAL_PRIMARY}; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: bold;">
            MEDAL ZONE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Rising athletes
    st.markdown(f"""
<div style="background: linear-gradient(135deg, {GOLD_ACCENT} 0%, #8a7850 100%);
     padding: 0.75rem 1rem; border-radius: 6px; margin: 1rem 0 0.5rem 0;">
    <h4 style="color: white; margin: 0;">Approaching Medal Zone</h4>
</div>
""", unsafe_allow_html=True)

    for athlete in PROJECT_EAST_SQUAD:
        if athlete["status"] != "Rising":
            continue
        in_db = athlete["db_name"] is not None
        db_tag = "" if in_db else f" <span style='color: {GRAY_BLUE}; font-size: 0.75rem;'>NOT IN DATABASE</span>"

        live_pb = "-"
        live_sb = ""
        if in_db:
            pbs = dc.get_ksa_athlete_pbs(athlete["db_name"])
            if pbs is not None and not pbs.empty:
                disc_col = "discipline" if "discipline" in pbs.columns else "event"
                mark_col = "mark" if "mark" in pbs.columns else "result"
                for _, pb_row in pbs.iterrows():
                    if format_event_name(str(pb_row.get(disc_col, ""))) == format_event_name(athlete["event"]):
                        live_pb = str(pb_row.get(mark_col, "-"))
                        break
            recent = _get_recent_marks(athlete["db_name"])
            evt_disp = format_event_name(athlete["event"])
            if evt_disp in recent:
                sb_val = recent[evt_disp].get("sb")
                if sb_val is not None:
                    et = get_event_type(athlete["event"])
                    live_sb = f" | SB: {format_mark_display(sb_val, et)}"

        st.markdown(f"""
<div style="padding: 10px 16px; margin: 4px 0; border-left: 4px solid {GOLD_ACCENT};
     background: #f8f9fa; border-radius: 4px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong>{athlete['name']}</strong>{db_tag}
            <br><span style="color: #666; font-size: 0.85rem;">
                {athlete['event']} | PB: {live_pb}{live_sb} | Medal Std: {athlete['medal_std']} | {athlete['project']}
            </span>
        </div>
        <div style="background: {GOLD_ACCENT}; color: white; padding: 3px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: bold;">
            RISING
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Missing athletes callout
    missing = [a for a in PROJECT_EAST_SQUAD if a["db_name"] is None]
    if missing:
        st.markdown("---")
        st.warning(
            f"**{len(missing)} Project East athletes not found in World Athletics database:** "
            + ", ".join(f"{a['name']} ({a['event']})" for a in missing)
            + ". These athletes may need manual addition or the scraper may need re-running."
        )


# ════════════════════════════════════════════════════════════════════════
# TAB 1: Athlete Readiness Matrix (DYNAMIC - uses recent results)
# ════════════════════════════════════════════════════════════════════════

with tabs[1]:
    render_section_header(
        "Athlete Readiness Matrix",
        "Based on recent competition marks (SB / last 3 average) vs Asian Games 2026 medal targets"
    )

    # Toggle: use PB or recent marks
    use_recent = st.toggle(
        "Use recent form (SB / last 3 avg) instead of all-time PB",
        value=True,
        key="ag_use_recent",
    )
    if use_recent:
        st.caption("Readiness based on best recent competition mark (Season Best or last 3 average)")
    else:
        st.caption("Readiness based on all-time Personal Best")

    if len(ksa) == 0:
        st.warning("No KSA athlete data loaded.")
    else:
        readiness_rows = []
        name_col = "full_name" if "full_name" in ksa.columns else "competitor"

        for _, athlete in ksa.iterrows():
            name = athlete.get(name_col, "Unknown")
            primary_event = athlete.get("primary_event", "")
            world_rank = athlete.get("best_world_rank")

            if not primary_event or pd.isna(primary_event):
                continue

            event_display = format_event_name(str(primary_event))
            event_lookup = _normalize_event_for_lookup(event_display)
            targets = ASIAN_GAMES_2026_TARGETS.get(event_lookup)

            # Get PB from scraped data
            pbs = dc.get_ksa_athlete_pbs(name)
            pb_mark = None
            if pbs is not None and not pbs.empty:
                disc_col = "discipline" if "discipline" in pbs.columns else "event"
                mark_col_pb = "mark" if "mark" in pbs.columns else "result"
                if disc_col in pbs.columns and mark_col_pb in pbs.columns:
                    for _, pb_row in pbs.iterrows():
                        if format_event_name(str(pb_row.get(disc_col, ""))) == event_display:
                            val = pd.to_numeric(pb_row.get(mark_col_pb), errors="coerce")
                            if not pd.isna(val):
                                pb_mark = float(val)
                                break

            if pb_mark is None:
                continue
            if not targets:
                continue

            event_type = get_event_type(str(primary_event))
            lower_is_better = event_type == "time"

            # Get recent marks and decide which mark to use for readiness
            recent = _get_recent_marks(name) if use_recent else {}
            current_mark, mark_source = _get_best_current_mark(
                pb_mark, recent, event_display, event_type
            )

            # SB and last 3 avg for display
            sb_display = "-"
            last3_display = "-"
            n_results = 0
            if event_display in recent:
                r = recent[event_display]
                if r["sb"] is not None:
                    sb_display = format_mark_display(r["sb"], event_type)
                if r["last3_avg"] is not None:
                    last3_display = format_mark_display(r["last3_avg"], event_type)
                n_results = r["n_results"]

            gold_mark = targets["gold"]
            medal_mark = targets["medal"]
            final_mark = targets["final"]

            # Calculate gaps from the chosen mark (current_mark)
            if lower_is_better:
                gold_gap = current_mark - gold_mark
                medal_gap = current_mark - medal_mark
                final_gap = current_mark - final_mark
            else:
                gold_gap = gold_mark - current_mark
                medal_gap = medal_mark - current_mark
                final_gap = final_mark - current_mark

            # Determine readiness status
            if gold_gap <= 0:
                status = "Gold Contender"
            elif medal_gap <= 0:
                status = "Medal Range"
            elif final_gap <= 0:
                status = "Finalist"
            elif abs(final_gap) < abs(final_mark * 0.03):
                status = "Near Final"
            else:
                status = "Development"

            readiness_rows.append({
                "Athlete": name,
                "Event": event_display,
                "PB": format_mark_display(pb_mark, event_type),
                "SB": sb_display,
                "Last 3 Avg": last3_display,
                "# Comps": n_results,
                "Current Mark": format_mark_display(current_mark, event_type),
                "Source": mark_source,
                "Gold Target": format_mark_display(gold_mark, event_type),
                "Gap to Gold": f"{abs(gold_gap):.2f}" if gold_gap > 0 else "Achieved",
                "Medal Target": format_mark_display(medal_mark, event_type),
                "Gap to Medal": f"{abs(medal_gap):.2f}" if medal_gap > 0 else "Achieved",
                "Status": status,
                "World Rank": int(world_rank) if pd.notna(world_rank) else 9999,
            })

        if readiness_rows:
            df_ready = pd.DataFrame(readiness_rows)
            status_order = {"Gold Contender": 0, "Medal Range": 1, "Finalist": 2, "Near Final": 3, "Development": 4}
            df_ready["_sort"] = df_ready["Status"].map(status_order)
            df_ready = df_ready.sort_values(["_sort", "World Rank"])

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                gc = len(df_ready[df_ready["Status"] == "Gold Contender"])
                render_metric_card("Gold Contenders", str(gc), "excellent" if gc > 0 else "neutral")
            with m2:
                mc = len(df_ready[df_ready["Status"].isin(["Gold Contender", "Medal Range"])])
                render_metric_card("Medal Range", str(mc), "good" if mc > 0 else "neutral")
            with m3:
                fc = len(df_ready[df_ready["Status"].isin(["Gold Contender", "Medal Range", "Finalist"])])
                render_metric_card("Potential Finalists", str(fc), "good" if fc > 0 else "neutral")
            with m4:
                render_metric_card("Athletes Tracked", str(len(df_ready)), "neutral")

            # Display table with recent form columns
            if use_recent:
                display_cols = ["Athlete", "Event", "PB", "SB", "Last 3 Avg", "# Comps",
                               "Gold Target", "Gap to Gold", "Medal Target", "Gap to Medal", "Status"]
            else:
                display_cols = ["Athlete", "Event", "PB", "Gold Target", "Gap to Gold",
                               "Medal Target", "Gap to Medal", "Status"]

            st.dataframe(
                df_ready[display_cols],
                hide_index=True,
                column_config={
                    "Athlete": st.column_config.TextColumn("Athlete", width="medium"),
                    "# Comps": st.column_config.NumberColumn("Comps", format=".0f"),
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

with tabs[2]:
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

            # Get all PBs and recent results
            all_pbs = dc.get_ksa_athlete_pbs(selected_athlete)
            all_results = dc.get_ksa_results(athlete_name=selected_athlete, limit=100)
            recent_marks = _get_recent_marks(selected_athlete)

            if all_pbs is not None and not all_pbs.empty:
                disc_col = "discipline" if "discipline" in all_pbs.columns else "event"
                mark_col_pb = "mark" if "mark" in all_pbs.columns else "result"

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

                        # Use recent mark for gap calculation
                        current, source = _get_best_current_mark(
                            pb, recent_marks, card["event_display"], et
                        )

                        gold_mark = targets["gold"]
                        medal_mark = targets["medal"]
                        final_mark = targets["final"]

                        if lib:
                            gold_gap = current - gold_mark
                            medal_gap = current - medal_mark
                            final_gap = current - final_mark
                        else:
                            gold_gap = gold_mark - current
                            medal_gap = medal_mark - current
                            final_gap = final_mark - current

                        # Show SB info
                        sb_info = ""
                        if card["event_display"] in recent_marks:
                            r = recent_marks[card["event_display"]]
                            sb_info = f" | SB: {format_mark_display(r['sb'], et)}" if r["sb"] else ""
                            if r["last3_avg"]:
                                sb_info += f" | Last 3 Avg: {format_mark_display(r['last3_avg'], et)}"

                        with st.expander(
                            f"**{card['event_display']}** — PB: {format_mark_display(pb, et)}{sb_info}",
                            expanded=True,
                        ):
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

                            # Progress bar
                            if lib:
                                worst = final_mark * 1.10
                                if current <= gold_mark:
                                    pct = 100
                                elif current >= worst:
                                    pct = 0
                                else:
                                    pct = max(0, min(100, (worst - current) / (worst - gold_mark) * 100))
                            else:
                                worst = final_mark * 0.90
                                if current >= gold_mark:
                                    pct = 100
                                elif current <= worst:
                                    pct = 0
                                else:
                                    pct = max(0, min(100, (current - worst) / (gold_mark - worst) * 100))

                            bar_color = TEAL_PRIMARY if pct >= 75 else (GOLD_ACCENT if pct >= 50 else GRAY_BLUE)
                            st.markdown(f"""
<div style="margin: 0.5rem 0;">
  <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
    <span style="font-size: 0.8rem; font-weight: 600;">Progress to Gold (using {source})</span>
    <span style="font-size: 0.8rem; font-weight: 700; color: {bar_color};">{pct:.0f}%</span>
  </div>
  <div style="background: #e9ecef; border-radius: 4px; height: 14px; overflow: hidden; position: relative;">
    <div style="width: {pct:.0f}%; height: 100%; background: {bar_color}; border-radius: 4px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

                            # Recent results
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

with tabs[3]:
    render_section_header(
        "Asian Rivals",
        "Top competitors from Asian nations — PB, Season Best & Latest Performance"
    )

    # Gender filter
    rival_gender = st.radio(
        "Gender", ["Men", "Women", "Both"],
        horizontal=True, key="ag_rival_gender",
    )

    rivals = dc.get_rivals(region="asia", limit=200)

    if len(rivals) > 0:
        # Apply gender filter
        gender_col = None
        for gc in ("gender", "Gender"):
            if gc in rivals.columns:
                gender_col = gc
                break
        if gender_col and rival_gender != "Both":
            gender_val = "M" if rival_gender == "Men" else "F"
            rivals = rivals[rivals[gender_col] == gender_val]

        # Event filter
        if "event" in rivals.columns:
            events = sorted(rivals["event"].dropna().unique())
            selected_event = st.selectbox(
                "Filter by event", ["All Events"] + events, key="ag_event"
            )
            if selected_event != "All Events":
                rivals = rivals[rivals["event"] == selected_event]

        # Display columns: focus on performance data (PB, SB, latest), not rankings
        # world_rank and ranking_score from scraper may be unreliable
        # (getWorldRankings API returns Women's-only data)
        perf_cols = [
            "full_name", "country_code", "event",
            "pb_mark", "sb_mark", "best5_avg", "latest_mark", "latest_date",
            "performances_count",
        ]
        available_cols = [c for c in perf_cols if c in rivals.columns]

        col_config = {
            "full_name": st.column_config.TextColumn("Athlete", width="medium"),
            "country_code": st.column_config.TextColumn("Nat"),
            "event": st.column_config.TextColumn("Event"),
            "pb_mark": st.column_config.TextColumn("PB"),
            "sb_mark": st.column_config.TextColumn("SB"),
            "best5_avg": st.column_config.TextColumn("Best 5 Avg"),
            "latest_mark": st.column_config.TextColumn("Latest"),
            "latest_date": st.column_config.TextColumn("Last Date"),
            "performances_count": st.column_config.NumberColumn("# Perfs", format=".0f"),
        }

        # Summary metrics
        r1, r2, r3 = st.columns(3)
        with r1:
            render_metric_card("Total Rivals", str(len(rivals)), "neutral")
        with r2:
            countries = rivals["country_code"].nunique() if "country_code" in rivals.columns else 0
            render_metric_card("Countries", str(countries), "neutral")
        with r3:
            events_count = rivals["event"].nunique() if "event" in rivals.columns else 0
            render_metric_card("Events", str(events_count), "neutral")

        rivals = rivals.copy()

        # Ensure performances_count is numeric (avoids literal "d" format display)
        if "performances_count" in rivals.columns:
            rivals["performances_count"] = pd.to_numeric(
                rivals["performances_count"], errors="coerce"
            )

        # Sort by PB (best performance) instead of unreliable world_rank
        if "pb_mark" in rivals.columns:
            rivals["_pb_num"] = pd.to_numeric(rivals["pb_mark"], errors="coerce")
            rivals = rivals.sort_values("_pb_num", na_position="last")
            rivals = rivals.drop(columns=["_pb_num"])

        st.dataframe(
            rivals[available_cols],
            hide_index=True,
            column_config=col_config,
            height=600,
        )

        # Event-by-event top threat
        if selected_event == "All Events" and "event" in rivals.columns:
            st.markdown("##### Top Threat per Event")
            for evt in events:
                evt_rivals = rivals[rivals["event"] == evt].head(1)
                if len(evt_rivals) > 0:
                    top = evt_rivals.iloc[0]
                    pb_str = str(top.get("pb_mark", "-")) if pd.notna(top.get("pb_mark")) else "-"
                    sb_str = str(top.get("sb_mark", "-")) if pd.notna(top.get("sb_mark")) else "-"
                    st.markdown(
                        f"<div style='padding: 4px 12px; margin: 3px 0; border-left: 3px solid {TEAL_PRIMARY}; "
                        f"background: #f8f9fa; border-radius: 3px; font-size: 0.9rem;'>"
                        f"<strong>{evt}</strong>: {top['full_name']} ({top.get('country_code', '?')}) "
                        f"— PB: {pb_str} | SB: {sb_str}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No rival data loaded. Run: `python -m scrapers.pipeline --daily`")

# ════════════════════════════════════════════════════════════════════════
# TAB 4: Medal Pathways (DYNAMIC - uses recent marks)
# ════════════════════════════════════════════════════════════════════════

with tabs[4]:
    render_section_header(
        "Medal Pathways",
        "Events where KSA has the best chance of Asian Games medals"
    )

    st.markdown(f"""
<div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, #005a51 100%);
     padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid {GOLD_ACCENT};">
    <h4 style="color: white; margin: 0;">Medal Pathway Strategy</h4>
    <p style="color: rgba(255,255,255,0.85); margin: 0.25rem 0 0 0; font-size: 0.9rem;">
        Events ranked by KSA's proximity to medal standards — using best recent mark (SB or PB)
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
                for _, pb_row in pbs.iterrows():
                    if format_event_name(str(pb_row.get(disc_col, ""))) == event_display:
                        val = pd.to_numeric(pb_row.get(mark_col_pb), errors="coerce")
                        if not pd.isna(val):
                            pb_mark = float(val)
                            break

            if pb_mark is None:
                continue

            event_type = get_event_type(str(primary_event))
            lower_is_better = event_type == "time"

            # Use recent form for medal pathway assessment
            recent = _get_recent_marks(name)
            current_mark, source = _get_best_current_mark(
                pb_mark, recent, event_display, event_type
            )

            medal_mark = targets["medal"]
            if lower_is_better:
                medal_gap = current_mark - medal_mark
                gap_pct = medal_gap / medal_mark * 100
            else:
                medal_gap = medal_mark - current_mark
                gap_pct = medal_gap / medal_mark * 100

            # SB display
            sb_str = "-"
            if event_display in recent and recent[event_display]["sb"] is not None:
                sb_str = format_mark_display(recent[event_display]["sb"], event_type)

            pathway_rows.append({
                "Athlete": name,
                "Event": event_display,
                "PB": format_mark_display(pb_mark, event_type),
                "SB": sb_str,
                "Current": format_mark_display(current_mark, event_type),
                "Source": source,
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
                sb_display = f" | SB: {row['SB']}" if row["SB"] != "-" else ""

                st.markdown(f"""
<div style="padding: 10px 16px; margin: 6px 0; border-left: 4px solid {border_color};
     background: #f8f9fa; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <strong>{icon} {row['Athlete']}</strong> — {row['Event']}
        <br><span style="color: #666; font-size: 0.85rem;">PB: {row['PB']}{sb_display} | Medal: {row['Medal Standard']} | {gap_display}</span>
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

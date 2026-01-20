"""
Coach View Module for World Athletics Dashboard

Provides simplified, action-focused interface for coaches including:
- Competition Prep Hub - Select championship, view KSA squad
- Athlete Report Cards - Comprehensive athlete briefings
- Competitor Watch - Monitor rivals and gaps

Adapted from Tilastopaja project to work with World Athletics parquet data.

NOTE: This module is designed to work with both World Athletics (this project)
and Tilastopaja data. Future database combination is planned.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics

# Import our custom modules
from projection_engine import (
    project_performance, calculate_gap, format_gap, detect_trend,
    get_trend_symbol, get_trend_color, calculate_form_score,
    calculate_advancement_probability, METHODOLOGY_NOTES
)
from historical_benchmarks import (
    get_default_benchmarks, get_benchmarks_for_event, format_benchmark_for_display,
    get_event_type, BENCHMARK_METHODOLOGY, BENCHMARK_COLORS
)
from chart_components import (
    season_progression_chart, gap_analysis_chart, probability_gauge,
    competitor_comparison_chart, form_trend_chart, create_form_score_gauge,
    create_last_3_comps_display, COLORS
)

# Team Saudi colors
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'

# Upcoming championships with dates
UPCOMING_CHAMPIONSHIPS = {
    "Asian Games 2026": {
        "date": datetime(2026, 9, 19),
        "type": "Asian Games",
        "venue": "Nagoya, Japan"
    },
    "World Championships 2027": {
        "date": datetime(2027, 8, 14),
        "type": "World Championships",
        "venue": "Beijing, China"
    },
    "LA 2028 Olympics": {
        "date": datetime(2028, 7, 14),
        "type": "Olympics",
        "venue": "Los Angeles, USA"
    }
}

# Asian region country codes for filtering
ASIAN_COUNTRY_CODES = {
    'KSA', 'JPN', 'CHN', 'KOR', 'IND', 'QAT', 'BRN', 'UAE', 'KUW', 'OMA',
    'IRN', 'IRQ', 'SYR', 'JOR', 'LBN', 'PAL', 'YEM', 'THA', 'VIE', 'MAS',
    'SGP', 'INA', 'PHI', 'MYA', 'CAM', 'LAO', 'BRU', 'TLS', 'TPE', 'HKG',
    'MAC', 'MGL', 'PRK', 'PAK', 'AFG', 'BAN', 'NEP', 'SRI', 'MDV', 'BHU',
    'UZB', 'KAZ', 'KGZ', 'TJK', 'TKM'
}


def get_ksa_athletes(df: pd.DataFrame) -> pd.DataFrame:
    """Get all KSA athletes from World Athletics data."""
    if 'nat' in df.columns:
        return df[df['nat'] == 'KSA']
    elif 'nationality' in df.columns:
        return df[df['nationality'] == 'KSA']
    return pd.DataFrame()


def get_athlete_recent_performances(df: pd.DataFrame, athlete_name: str, event: str, limit: int = 10) -> List[Dict]:
    """Get athlete's recent performances in an event from World Athletics data."""
    # Filter by athlete and event
    athlete_data = df[
        (df['competitor'].str.contains(athlete_name, case=False, na=False)) &
        (df['event'] == event)
    ]

    if athlete_data.empty:
        return []

    # Sort by date descending
    athlete_data = athlete_data.copy()
    if 'date' in athlete_data.columns:
        athlete_data['date'] = pd.to_datetime(athlete_data['date'], errors='coerce')
        athlete_data = athlete_data.sort_values('date', ascending=False)

    athlete_data = athlete_data.head(limit)

    performances = []
    result_col = 'result_numeric' if 'result_numeric' in athlete_data.columns else 'result'

    for _, row in athlete_data.iterrows():
        result = row.get(result_col)
        if pd.notna(result):
            performances.append({
                'date': row.get('date'),
                'result': float(result),
                'competition': str(row.get('venue', 'Unknown'))
            })

    return performances


def get_athlete_bests(df: pd.DataFrame, athlete_name: str, event: str) -> Dict:
    """Get athlete's season best, personal best, and averages from World Athletics data."""
    athlete_data = df[
        (df['competitor'].str.contains(athlete_name, case=False, na=False)) &
        (df['event'] == event)
    ]

    if athlete_data.empty:
        return {'sb': None, 'pb': None, 'avg': None, 'pb_date': None}

    result_col = 'result_numeric' if 'result_numeric' in athlete_data.columns else 'result'
    athlete_data = athlete_data.dropna(subset=[result_col])

    if athlete_data.empty:
        return {'sb': None, 'pb': None, 'avg': None, 'pb_date': None}

    event_type = get_event_type(event)
    results = athlete_data[result_col].tolist()

    # Personal Best (all time)
    pb = min(results) if event_type == 'time' else max(results)

    # Season Best (current year)
    athlete_data = athlete_data.copy()
    if 'date' in athlete_data.columns:
        athlete_data['date'] = pd.to_datetime(athlete_data['date'], errors='coerce')
        current_year = datetime.now().year
        season_data = athlete_data[athlete_data['date'].dt.year == current_year]

        if not season_data.empty:
            season_results = season_data[result_col].tolist()
            sb = min(season_results) if event_type == 'time' else max(season_results)
        else:
            sb = pb
    else:
        sb = pb

    # Average of last 5
    recent_data = athlete_data.nlargest(5, 'date') if 'date' in athlete_data.columns else athlete_data.head(5)
    recent_results = recent_data[result_col].tolist()
    avg = statistics.mean(recent_results) if recent_results else None

    # PB date
    pb_row = athlete_data[athlete_data[result_col] == pb]
    pb_date = pb_row['date'].iloc[0] if not pb_row.empty and 'date' in pb_row.columns else None

    return {'sb': sb, 'pb': pb, 'avg': avg, 'pb_date': pb_date}


def show_competition_prep_hub(df: pd.DataFrame):
    """Competition Prep Hub - Central hub for preparing athletes before championships."""

    # Header with Team Saudi styling
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
                padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Competition Prep Hub</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Prepare KSA athletes for upcoming championships
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Competition selector
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_champ = st.selectbox(
            "Select Championship",
            list(UPCOMING_CHAMPIONSHIPS.keys()),
            key="coach_competition_select"
        )

    with col2:
        champ_info = UPCOMING_CHAMPIONSHIPS[selected_champ]
        days_until = (champ_info['date'] - datetime.now()).days

        if days_until > 0:
            st.metric("Days Until", f"{days_until}")
        else:
            st.metric("Status", "Completed" if days_until < -30 else "In Progress")

    # Championship info bar
    st.info(f"**{selected_champ}** | {champ_info['venue']} | {champ_info['date'].strftime('%B %d, %Y')}")

    st.markdown("---")

    # Get KSA athletes
    ksa_df = get_ksa_athletes(df)

    if ksa_df.empty:
        st.warning("No KSA athlete data found in the database.")
        return

    # Filter to active athletes (last 3 years)
    ksa_df = ksa_df.copy()
    if 'date' in ksa_df.columns:
        ksa_df['date'] = pd.to_datetime(ksa_df['date'], errors='coerce')
        cutoff_date = datetime.now() - timedelta(days=365 * 3)
        ksa_df = ksa_df[ksa_df['date'] >= cutoff_date]

    if ksa_df.empty:
        st.warning("No active KSA athletes found (competed in last 3 years).")
        return

    # Group by event and gender
    athlete_events = ksa_df.groupby(['event', 'gender', 'competitor']).size().reset_index(name='count')

    st.subheader("KSA Squad Overview")

    col1, col2 = st.columns(2)
    with col1:
        gender_opts = ['All'] + sorted(athlete_events['gender'].dropna().unique().tolist())
        selected_gender = st.selectbox("Filter by Gender", gender_opts, key="prep_gender")

    with col2:
        if selected_gender != 'All':
            events_for_gender = athlete_events[athlete_events['gender'] == selected_gender]['event'].unique()
        else:
            events_for_gender = athlete_events['event'].unique()
        event_opts = ['All Events'] + sorted(events_for_gender.tolist())
        selected_event = st.selectbox("Filter by Event", event_opts, key="prep_event")

    # Filter data
    filtered = athlete_events.copy()
    if selected_gender != 'All':
        filtered = filtered[filtered['gender'] == selected_gender]
    if selected_event != 'All Events':
        filtered = filtered[filtered['event'] == selected_event]

    if filtered.empty:
        st.info("No athletes found matching filters.")
        return

    # Display athletes by event
    st.markdown("### Athletes")

    for event in sorted(filtered['event'].unique()):
        event_athletes = filtered[filtered['event'] == event]

        with st.expander(f"**{event}** ({len(event_athletes)} athletes)", expanded=True):
            for _, row in event_athletes.iterrows():
                athlete_name = row['competitor']
                gender = row['gender']

                # Get athlete's bests
                bests = get_athlete_bests(df, athlete_name, event)
                event_type = get_event_type(event)

                # Get benchmarks for this event
                benchmarks = get_benchmarks_for_event(event, gender)

                col1, col2, col3, col4, col5 = st.columns([2.5, 1.5, 2, 2, 1.5])

                with col1:
                    st.markdown(f"**{athlete_name}**")

                with col2:
                    if bests['sb']:
                        sb_formatted = format_benchmark_for_display(bests['sb'], event_type)
                        st.caption(f"SB: {sb_formatted}")
                    else:
                        st.caption("SB: N/A")

                with col3:
                    # Gap to entry standard (use medal as proxy)
                    if benchmarks.get('final') and bests['sb']:
                        gap = calculate_gap(bests['sb'], benchmarks['final'], event_type)
                        if gap <= 0:
                            st.success("Final level")
                        else:
                            st.warning(f"{format_gap(gap, event_type)} to final")
                    else:
                        st.caption("Standard: N/A")

                with col4:
                    # Last 3 competitions
                    performances = get_athlete_recent_performances(df, athlete_name, event, 3)
                    if performances:
                        times_list = [format_benchmark_for_display(p['result'], event_type) for p in performances]
                        st.caption(f"Last 3: {', '.join(times_list)}")
                    else:
                        st.caption("Last 3: N/A")

                with col5:
                    # Form trend
                    performances_5 = get_athlete_recent_performances(df, athlete_name, event, 5)
                    if len(performances_5) >= 3:
                        results = [p['result'] for p in performances_5]
                        trend = detect_trend(results, event_type)
                        color = get_trend_color(trend)
                        st.markdown(f"<span style='color: {color}'>{get_trend_symbol(trend)} {trend.title()}</span>",
                                   unsafe_allow_html=True)
                    else:
                        st.caption("Form: N/A")

                # View button to navigate to Athlete Reports
                if st.button("View", key=f"view_{athlete_name}_{event}"):
                    st.session_state['selected_athlete_for_report'] = {
                        'name': athlete_name,
                        'event': event
                    }
                    st.session_state['coach_view_tab'] = "Athlete Reports"
                    st.rerun()

    # Store selection for other tabs
    st.session_state['selected_championship'] = selected_champ
    st.session_state['selected_championship_info'] = champ_info


def show_athlete_report_cards(df: pd.DataFrame):
    """Athlete Report Cards - Comprehensive pre-competition briefings."""

    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
                padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Athlete Report Cards</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Comprehensive performance analysis and projections
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get championship context
    selected_champ = st.session_state.get('selected_championship', 'Asian Games 2026')
    champ_info = st.session_state.get('selected_championship_info', UPCOMING_CHAMPIONSHIPS.get(selected_champ, {}))

    if champ_info:
        days_until = (champ_info.get('date', datetime.now()) - datetime.now()).days
        if days_until > 0:
            st.info(f"**{selected_champ}** | {champ_info.get('venue', 'TBD')} | {days_until} days to go")

    st.markdown("---")

    # Get KSA athletes
    ksa_df = get_ksa_athletes(df)

    if ksa_df.empty:
        st.warning("No KSA athlete data available.")
        return

    # Athlete selector - check for preselected from Competition Prep
    preselected = st.session_state.get('selected_athlete_for_report', None)

    col1, col2 = st.columns(2)

    with col1:
        athlete_names = sorted(ksa_df['competitor'].dropna().unique().tolist())
        default_idx = 0
        if preselected and preselected.get('name') in athlete_names:
            default_idx = athlete_names.index(preselected['name'])
        selected_athlete = st.selectbox("Select Athlete", athlete_names, index=default_idx, key="report_athlete")

    # Get athlete's events
    athlete_data = ksa_df[ksa_df['competitor'] == selected_athlete]

    with col2:
        athlete_events = sorted(athlete_data['event'].dropna().unique().tolist())
        default_event_idx = 0
        if preselected and preselected.get('event') in athlete_events:
            default_event_idx = athlete_events.index(preselected['event'])
        selected_event = st.selectbox("Select Event", athlete_events, index=default_event_idx, key="report_event")

    if not selected_athlete or not selected_event:
        st.info("Please select an athlete and event.")
        return

    # Clear preselected after using it
    if preselected:
        st.session_state['selected_athlete_for_report'] = None

    # Get gender
    gender = athlete_data['gender'].iloc[0] if 'gender' in athlete_data.columns else 'men'
    event_type = get_event_type(selected_event)

    st.markdown("---")

    # Report Header
    st.header(f"{selected_athlete} | {selected_event}")
    st.caption(f"Saudi Arabia | {gender.title()}")

    # === QUALIFICATION STATUS ===
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Summary")

        bests = get_athlete_bests(df, selected_athlete, selected_event)
        benchmarks = get_benchmarks_for_event(selected_event, gender)

        metrics_col1, metrics_col2 = st.columns(2)

        with metrics_col1:
            if bests['sb']:
                st.metric("Season Best", format_benchmark_for_display(bests['sb'], event_type))
            else:
                st.metric("Season Best", "N/A")

            if bests['pb']:
                pb_str = format_benchmark_for_display(bests['pb'], event_type)
                if bests.get('pb_date') and pd.notna(bests['pb_date']):
                    pb_str += f" ({pd.to_datetime(bests['pb_date']).strftime('%b %Y')})"
                st.metric("Personal Best", pb_str)
            else:
                st.metric("Personal Best", "N/A")

        with metrics_col2:
            if benchmarks.get('final'):
                st.metric("Final Line", format_benchmark_for_display(benchmarks['final'], event_type))

            if benchmarks.get('medal'):
                st.metric("Medal Line", format_benchmark_for_display(benchmarks['medal'], event_type))

        # Gap to benchmarks
        if bests['sb'] and benchmarks.get('final'):
            gap = calculate_gap(bests['sb'], benchmarks['final'], event_type)
            if gap <= 0:
                st.success(f"FINAL LEVEL - {format_gap(abs(gap), event_type)} inside final line")
            else:
                st.warning(f"Gap to Final: {format_gap(gap, event_type)}")

    with col2:
        st.subheader("Form Projection")

        performances = get_athlete_recent_performances(df, selected_athlete, selected_event, 10)

        if len(performances) >= 3:
            results = [p['result'] for p in performances]
            projection = project_performance(results, event_type=event_type, is_major_championship=True)

            metrics_col1, metrics_col2 = st.columns(2)

            with metrics_col1:
                st.metric(
                    "Projected",
                    format_benchmark_for_display(projection['projected'], event_type)
                )
                st.metric(
                    "Trend",
                    f"{projection['trend_symbol']} {projection['trend'].title()}"
                )

            with metrics_col2:
                st.metric(
                    "Range (68%)",
                    f"{format_benchmark_for_display(projection['range_low'], event_type)} - {format_benchmark_for_display(projection['range_high'], event_type)}"
                )
                st.metric("Form Score", f"{projection['form_score']}/100")

        else:
            st.info("Insufficient data for projection (need 3+ performances)")

    st.markdown("---")

    # === CHAMPIONSHIP BENCHMARKS ===
    st.subheader("Championship Benchmarks")

    if benchmarks:
        bench_cols = st.columns(4)

        for i, (round_name, label) in enumerate([
            ('medal', 'Medal Zone'),
            ('final', 'Final'),
            ('semi', 'Semi-Final'),
            ('heat', 'Heat Survival')
        ]):
            with bench_cols[i]:
                if round_name in benchmarks:
                    value = benchmarks[round_name]
                    st.metric(label, format_benchmark_for_display(value, event_type))

                    # Gap from athlete
                    if bests['sb']:
                        gap = calculate_gap(bests['sb'], value, event_type)
                        if gap <= 0:
                            st.caption(f"<span style='color: {TEAL_PRIMARY}'>{format_gap(abs(gap), event_type)} ahead</span>",
                                      unsafe_allow_html=True)
                        else:
                            st.caption(f"<span style='color: #dc3545'>{format_gap(gap, event_type)} behind</span>",
                                      unsafe_allow_html=True)
                else:
                    st.metric(label, "N/A")

    st.markdown("---")

    # === LAST 5 RACES ===
    st.subheader("Recent Competition History")

    # Get last 5 performances with full details
    recent_perfs = get_athlete_recent_performances(df, selected_athlete, selected_event, 5)

    if recent_perfs:
        # Build race history table
        race_history = []
        valid_results = []

        for r in recent_perfs:
            race_date = pd.to_datetime(r.get('date'), errors='coerce')
            result_val = r.get('result')
            competition = r.get('competition', 'Unknown')

            if pd.notna(result_val):
                valid_results.append(result_val)

            race_history.append({
                'Date': race_date.strftime('%d %b %Y') if pd.notna(race_date) else 'N/A',
                'Competition': str(competition)[:40] if competition else 'N/A',
                'Result': format_benchmark_for_display(result_val, event_type) if pd.notna(result_val) else 'N/A'
            })

        # Calculate average
        if valid_results:
            avg_result = sum(valid_results) / len(valid_results)
            avg_display = format_benchmark_for_display(avg_result, event_type)

            if event_type == 'time':
                best_recent = min(valid_results)
            else:
                best_recent = max(valid_results)
            best_display = format_benchmark_for_display(best_recent, event_type)
        else:
            avg_result = None
            avg_display = 'N/A'
            best_display = 'N/A'

        # Display metrics
        col_avg1, col_avg2, col_avg3 = st.columns(3)
        with col_avg1:
            st.metric("Last 5 Races Average", avg_display)
        with col_avg2:
            st.metric("Best in Last 5", best_display)
        with col_avg3:
            st.metric("Races Analyzed", len(valid_results))

        # Display race history table
        st.markdown("**Race History**")
        race_df = pd.DataFrame(race_history)
        st.dataframe(race_df, hide_index=True, use_container_width=True)

        # Championship Outlook - compare average vs PB
        if avg_result and benchmarks:
            st.markdown("**Championship Outlook**")
            col_proj1, col_proj2 = st.columns(2)

            with col_proj1:
                st.caption(f"If performing at average ({avg_display}):")
                for round_name, label in [('final', 'Final'), ('semi', 'Semi'), ('heat', 'Heat')]:
                    if round_name in benchmarks:
                        bench_val = benchmarks[round_name]
                        gap = calculate_gap(avg_result, bench_val, event_type)
                        if gap <= 0:
                            st.success(f"✓ {label}: {format_gap(abs(gap), event_type)} inside")
                        else:
                            st.warning(f"✗ {label}: {format_gap(gap, event_type)} outside")

            with col_proj2:
                if bests['pb']:
                    st.caption(f"If performing at PB ({format_benchmark_for_display(bests['pb'], event_type)}):")
                    for round_name, label in [('final', 'Final'), ('semi', 'Semi'), ('heat', 'Heat')]:
                        if round_name in benchmarks:
                            bench_val = benchmarks[round_name]
                            gap = calculate_gap(bests['pb'], bench_val, event_type)
                            if gap <= 0:
                                st.success(f"✓ {label}: {format_gap(abs(gap), event_type)} inside")
                            else:
                                st.warning(f"✗ {label}: {format_gap(gap, event_type)} outside")
    else:
        st.info("No recent competition data available.")

    st.markdown("---")

    # === CHARTS ===
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Season Progression")

        if performances:
            chart = season_progression_chart(
                performances=performances,
                benchmarks=benchmarks,
                event_type=event_type,
                title=f"Season Progression - {selected_athlete}"
            )
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("No performance data for chart.")

    with col2:
        st.subheader("Gap Analysis")

        if bests['sb'] and benchmarks:
            gap_chart = gap_analysis_chart(
                athlete_performance=bests['sb'],
                benchmarks=benchmarks,
                event_type=event_type,
                title='Gap to Championship Benchmarks'
            )
            st.plotly_chart(gap_chart, use_container_width=True)
        else:
            st.info("Insufficient data for gap analysis")

    st.markdown("---")

    # === PROBABILITY AND FORM ===
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Advancement Probability")

        if performances and len(performances) >= 3 and benchmarks:
            results = [p['result'] for p in performances]
            projection = project_performance(results, event_type=event_type)

            probabilities = calculate_advancement_probability(
                projection['projected'],
                benchmarks,
                event_type
            )

            if probabilities:
                prob_chart = probability_gauge(probabilities)
                st.plotly_chart(prob_chart, use_container_width=True)
        else:
            st.info("Insufficient data for probability calculation")

    with col2:
        st.subheader("Current Form")

        if performances and len(performances) >= 3:
            results = [p['result'] for p in performances]
            form_score = calculate_form_score(results, event_type)
            form_gauge = create_form_score_gauge(form_score, title="Form Score")
            st.plotly_chart(form_gauge, use_container_width=True)

            # Form trend chart
            trend_chart = form_trend_chart(performances[:10], event_type=event_type, title="Recent Form Trend")
            st.plotly_chart(trend_chart, use_container_width=True)
        else:
            st.info("Insufficient data for form analysis")

    # === METHODOLOGY ===
    with st.expander("Methodology Notes"):
        st.markdown(METHODOLOGY_NOTES)
        st.markdown(BENCHMARK_METHODOLOGY)

        # Form Score explanation
        st.markdown("""
        **Form Score (0-100)** measures how well an athlete is performing relative to their own capability:

        **Calculation:**
        - Based on recent performance relative to personal range (best to worst)
        - Higher score = performing closer to personal best
        - 85-100: Excellent form
        - 70-84: Good form
        - 50-69: Moderate form
        - 30-49: Low form
        - 0-29: Poor form
        """)


def show_competitor_watch(df: pd.DataFrame):
    """Competitor Watch - Monitor rivals and competitive landscape."""

    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
                padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">Competitor Watch</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Monitor rivals and competitive landscape
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get championship context
    default_champ = st.session_state.get('selected_championship', 'Asian Games 2026')

    col1, col2, col3 = st.columns(3)

    with col1:
        championship_options = list(UPCOMING_CHAMPIONSHIPS.keys())
        default_idx = championship_options.index(default_champ) if default_champ in championship_options else 0
        selected_championship = st.selectbox(
            "Target Championship",
            championship_options,
            index=default_idx,
            key="watch_championship"
        )

    with col2:
        # Handle different column names for gender
        gender_col = 'gender' if 'gender' in df.columns else ('Gender' if 'Gender' in df.columns else None)
        if gender_col:
            gender_opts = sorted(df[gender_col].dropna().unique().tolist())
        else:
            gender_opts = ['Men', 'Women']
        selected_gender = st.selectbox("Gender", gender_opts, key="watch_gender")

    with col3:
        # Handle different column names for event
        event_col = 'event' if 'event' in df.columns else ('Event' if 'Event' in df.columns else ('Event Type' if 'Event Type' in df.columns else None))

        if event_col is None:
            st.error("No Event column found in data. Available columns: " + ", ".join(df.columns.tolist()[:10]))
            return

        if gender_col:
            gender_filtered = df[df[gender_col] == selected_gender]
        else:
            gender_filtered = df

        event_opts = sorted(gender_filtered[event_col].dropna().unique().tolist())
        selected_event = st.selectbox("Event", event_opts, key="watch_event")

    if not selected_event:
        st.info("Please select an event.")
        return

    # Normalize column names for rest of function
    if 'event' not in df.columns and event_col:
        df = df.rename(columns={event_col: 'event'})
    if 'gender' not in df.columns and gender_col:
        df = df.rename(columns={gender_col: 'gender'})

    # Championship info
    champ_info = UPCOMING_CHAMPIONSHIPS.get(selected_championship, {})
    is_asian_event = 'Asian' in selected_championship

    st.info(f"**{selected_championship}** | {champ_info.get('venue', 'TBD')}")

    st.markdown("---")

    # Get KSA athlete for comparison
    ksa_df = get_ksa_athletes(df)
    ksa_in_event = ksa_df[
        (ksa_df['event'] == selected_event) &
        (ksa_df['gender'] == selected_gender)
    ]

    event_type = get_event_type(selected_event)
    ksa_sb = None
    ksa_athlete = None

    if not ksa_in_event.empty:
        # Get best KSA athlete in this event
        result_col = 'result_numeric' if 'result_numeric' in ksa_in_event.columns else 'result'
        ksa_in_event = ksa_in_event.dropna(subset=[result_col])

        if not ksa_in_event.empty:
            if event_type == 'time':
                best_idx = ksa_in_event[result_col].idxmin()
            else:
                best_idx = ksa_in_event[result_col].idxmax()

            ksa_row = ksa_in_event.loc[best_idx]
            ksa_athlete = ksa_row['competitor']
            ksa_sb = ksa_row[result_col]

            st.subheader(f"KSA Athlete: {ksa_athlete}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Season Best", format_benchmark_for_display(ksa_sb, event_type))
            with col2:
                bests = get_athlete_bests(df, ksa_athlete, selected_event)
                if bests['pb']:
                    st.metric("Personal Best", format_benchmark_for_display(bests['pb'], event_type))
    else:
        st.warning("No KSA athlete in this event for comparison.")

    st.markdown("---")

    # Get competitors
    st.subheader(f"Top Competitors - {selected_event}")

    event_data = df[
        (df['event'] == selected_event) &
        (df['gender'] == selected_gender)
    ].copy()

    # Filter to Asian countries for Asian events
    if is_asian_event:
        event_data = event_data[event_data['nat'].isin(ASIAN_COUNTRY_CODES)]
        st.caption("Showing Asian athletes only")

    # Get recent data (last 2 years)
    if 'date' in event_data.columns:
        event_data['date'] = pd.to_datetime(event_data['date'], errors='coerce')
        cutoff_date = datetime.now() - timedelta(days=730)
        event_data = event_data[event_data['date'] >= cutoff_date]

    if event_data.empty:
        st.info("No recent competitor data available.")
        return

    # Get season bests per athlete
    result_col = 'result_numeric' if 'result_numeric' in event_data.columns else 'result'
    event_data = event_data.dropna(subset=[result_col, 'competitor'])

    if event_type == 'time':
        athlete_bests = event_data.groupby('competitor').agg({
            result_col: 'min',
            'nat': 'first'
        }).reset_index()
        athlete_bests = athlete_bests.sort_values(result_col, ascending=True)
    else:
        athlete_bests = event_data.groupby('competitor').agg({
            result_col: 'max',
            'nat': 'first'
        }).reset_index()
        athlete_bests = athlete_bests.sort_values(result_col, ascending=False)

    # Build competitor table
    competitors_data = []
    for i, row in athlete_bests.head(25).iterrows():
        athlete_name = row['competitor']
        country = row['nat']
        sb = row[result_col]

        # Calculate gap from KSA
        gap = None
        gap_formatted = "N/A"
        if ksa_sb and pd.notna(sb):
            gap = calculate_gap(ksa_sb, sb, event_type)
            gap_formatted = format_gap(gap, event_type)

        # Get trend
        performances = get_athlete_recent_performances(df, athlete_name, selected_event, 5)
        if len(performances) >= 3:
            results = [p['result'] for p in performances]
            trend = detect_trend(results, event_type)
        else:
            trend = 'stable'

        competitors_data.append({
            'Rank': len(competitors_data) + 1,
            'Athlete': athlete_name,
            'Country': country,
            'SB': format_benchmark_for_display(sb, event_type) if pd.notna(sb) else 'N/A',
            'Gap': gap_formatted,
            'Trend': f"{get_trend_symbol(trend)} {trend.title()}",
            'Is_KSA': country == 'KSA'
        })

    if competitors_data:
        comp_df = pd.DataFrame(competitors_data)

        # Display with KSA highlighting
        def highlight_ksa(row):
            if row.get('Is_KSA', False) or row.get('Country') == 'KSA':
                return [f'background-color: {TEAL_PRIMARY}20'] * len(row)
            return [''] * len(row)

        display_cols = ['Rank', 'Athlete', 'Country', 'SB', 'Gap', 'Trend']
        styled_df = comp_df[display_cols + ['Is_KSA']].style.apply(highlight_ksa, axis=1)

        st.dataframe(
            styled_df.hide(subset=['Is_KSA'], axis='columns'),
            use_container_width=True,
            height=500
        )

        # Summary insights
        st.markdown("---")
        st.subheader("Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            improving = [c for c in competitors_data if 'Improving' in c['Trend']]
            st.metric("Athletes Improving", len(improving))

        with col2:
            declining = [c for c in competitors_data if 'Declining' in c['Trend']]
            st.metric("Athletes Declining", len(declining))

        with col3:
            if ksa_sb:
                ahead = [c for c in competitors_data if c['Gap'] != 'N/A' and c['Gap'].startswith('-')]
                st.metric("Athletes Ahead of KSA", len(ahead))

    # === BUILD CUSTOM RACE LIST ===
    st.markdown("---")
    st.subheader("Build Custom Race List")
    st.caption(f"Build race simulation for {selected_event} ({selected_gender})")

    # Get all athletes for selection (from all data, not just recent)
    all_event_data = df[
        (df['event'] == selected_event) &
        (df['gender'] == selected_gender)
    ].copy()

    result_col = 'result_numeric' if 'result_numeric' in all_event_data.columns else 'result'
    all_event_data = all_event_data.dropna(subset=[result_col, 'competitor'])

    if event_type == 'time':
        all_athlete_bests = all_event_data.groupby('competitor').agg({
            result_col: 'min',
            'nat': 'first'
        }).reset_index()
        all_athlete_bests = all_athlete_bests.sort_values(result_col, ascending=True)
    else:
        all_athlete_bests = all_event_data.groupby('competitor').agg({
            result_col: 'max',
            'nat': 'first'
        }).reset_index()
        all_athlete_bests = all_athlete_bests.sort_values(result_col, ascending=False)

    # Country filter
    all_countries = sorted(all_athlete_bests['nat'].dropna().unique().tolist())

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        country_filter = st.multiselect(
            "Filter by Country",
            options=all_countries,
            default=[],
            key="race_list_country_filter",
            placeholder="All countries"
        )
    with col_filter2:
        top_n = st.slider("Top N Athletes", min_value=10, max_value=100, value=50, key="race_list_top_n")

    # Filter by country if selected
    if country_filter:
        filtered_bests = all_athlete_bests[all_athlete_bests['nat'].isin(country_filter)]
    else:
        filtered_bests = all_athlete_bests

    # Build options for selection
    top_competitors_for_select = []
    top_sorted = filtered_bests.head(top_n)

    for _, row in top_sorted.iterrows():
        top_competitors_for_select.append({
            'name': row['competitor'],
            'country': row['nat'],
            'best': row[result_col],
            'display': f"{row['competitor']} ({row['nat']}) - {format_benchmark_for_display(row[result_col], event_type)}"
        })

    col_method1, col_method2 = st.columns(2)

    with col_method1:
        st.markdown(f"**Quick Add - Top {top_n} Athletes**")
        if country_filter:
            st.caption(f"Filtered to: {', '.join(country_filter)}")
        quick_select = st.multiselect(
            "Select from top performers",
            options=[a['display'] for a in top_competitors_for_select],
            key="quick_competitor_select",
            label_visibility="collapsed"
        )

        if quick_select:
            if st.button("Add Selected", key="add_quick_select"):
                if 'custom_race_list' not in st.session_state:
                    st.session_state['custom_race_list'] = []
                added = 0
                for sel in quick_select:
                    for a in top_competitors_for_select:
                        if a['display'] == sel:
                            # Check not already in list
                            existing_names = [x['name'] for x in st.session_state['custom_race_list']]
                            if a['name'] not in existing_names:
                                st.session_state['custom_race_list'].append(a)
                                added += 1
                if added:
                    st.success(f"Added {added} athlete(s)")
                    st.rerun()

    with col_method2:
        st.markdown("**Search All Athletes**")
        search_term = st.text_input("Type name to search", key="competitor_search", placeholder="e.g. Johnson", label_visibility="collapsed")

        if search_term and len(search_term) >= 2:
            # Search in all athlete bests
            search_results = all_athlete_bests[
                all_athlete_bests['competitor'].str.contains(search_term, case=False, na=False)
            ].head(20)

            if not search_results.empty:
                search_athletes = [
                    {
                        'name': row['competitor'],
                        'country': row['nat'],
                        'best': row[result_col],
                        'display': f"{row['competitor']} ({row['nat']}) - {format_benchmark_for_display(row[result_col], event_type)}"
                    }
                    for _, row in search_results.iterrows()
                ]

                selected_search = st.multiselect(
                    "Search results",
                    options=[a['display'] for a in search_athletes],
                    key="manual_competitor_select",
                    label_visibility="collapsed"
                )

                if selected_search and st.button("Add from Search", key="add_search_select"):
                    if 'custom_race_list' not in st.session_state:
                        st.session_state['custom_race_list'] = []
                    added = 0
                    for sel in selected_search:
                        for a in search_athletes:
                            if a['display'] == sel:
                                existing_names = [x['name'] for x in st.session_state['custom_race_list']]
                                if a['name'] not in existing_names:
                                    st.session_state['custom_race_list'].append(a)
                                    added += 1
                    if added:
                        st.success(f"Added {added} athlete(s)")
                        st.rerun()
            else:
                st.info("No athletes found.")

    # Display current race list
    if 'custom_race_list' in st.session_state and st.session_state['custom_race_list']:
        st.markdown("#### Your Custom Race List")

        race_list_data = []
        for i, athlete in enumerate(st.session_state['custom_race_list']):
            # Calculate gap from KSA
            gap_val = None
            if ksa_sb and athlete['best']:
                gap_val = calculate_gap(ksa_sb, athlete['best'], event_type)

            race_list_data.append({
                'Lane': i + 1,
                'Athlete': athlete['name'],
                'Country': athlete['country'],
                'Best': format_benchmark_for_display(athlete['best'], event_type) if athlete['best'] else 'N/A',
                'Gap to KSA': format_gap(gap_val, event_type) if gap_val else 'N/A'
            })

        race_df = pd.DataFrame(race_list_data)
        st.dataframe(race_df, hide_index=True, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Race List", key="clear_race_list"):
                st.session_state['custom_race_list'] = []
                st.rerun()
        with col2:
            st.download_button(
                "Export Race List",
                data=race_df.to_csv(index=False),
                file_name=f"race_list_{selected_event}_{selected_gender}.csv",
                mime="text/csv",
                key="export_race_list"
            )


def render_coach_view(df: pd.DataFrame):
    """
    Main entry point for Coach View.
    Renders all Coach View sections.
    """
    # Saudi Arabia header
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <div>
            <h1 style="color: {TEAL_PRIMARY}; margin: 0;">Saudi Athletics</h1>
            <p style="color: #888; margin: 0;">Coach Dashboard - Performance Analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    tab_names = [
        "Competition Prep",
        "Athlete Reports",
        "Competitor Watch"
    ]

    default_tab = st.session_state.get('coach_view_tab', "Competition Prep")
    if default_tab not in tab_names:
        default_tab = "Competition Prep"

    selected_tab = st.selectbox(
        "Navigate to",
        tab_names,
        index=tab_names.index(default_tab),
        key="coach_nav_select"
    )

    st.session_state['coach_view_tab'] = selected_tab

    st.markdown("---")

    # Render selected tab
    if selected_tab == "Competition Prep":
        show_competition_prep_hub(df)
    elif selected_tab == "Athlete Reports":
        show_athlete_report_cards(df)
    elif selected_tab == "Competitor Watch":
        show_competitor_watch(df)

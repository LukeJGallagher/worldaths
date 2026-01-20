# Dashboard Analytics Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add critical analytics features identified in Head Coach + Data Analyst review: Performance Consistency, Near Miss Alerts, Country Comparison, Head-to-Head, and Dashboard Performance optimizations.

**Architecture:** All features integrate into existing `World_Ranking_Deploy_v3.py` tabs. Helper functions added to `analytics_helpers.py` (new file) to keep main file clean. Reuse existing `data_connector.py` queries.

**Tech Stack:** Streamlit, Pandas, Plotly, DuckDB (via data_connector), NumPy for statistics

---

## Task 1: Create Analytics Helpers Module

**Files:**
- Create: `analytics_helpers.py`

**Step 1: Write the analytics helper module with core functions**

```python
"""
Analytics Helper Functions for Saudi Athletics Dashboard

Provides statistical calculations for:
- Performance Consistency Score
- Near Miss Detection
- Head-to-Head Comparison
- Country Benchmarking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# Team Saudi Brand Colors
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'
TEAL_LIGHT = '#009688'
GRAY_BLUE = '#78909C'


def parse_result_to_seconds(result: str, event: str = '') -> Optional[float]:
    """
    Convert result string to numeric value (seconds or meters).

    Args:
        result: Result string like "10.45", "1:59.00", "8.50"
        event: Event name to determine if field event

    Returns:
        Float value or None if unparseable
    """
    if pd.isna(result):
        return None
    result_str = str(result).strip()
    try:
        if ':' in result_str:
            parts = result_str.split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:  # HH:MM:SS for marathon
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(result_str)
    except (ValueError, TypeError):
        return None


def is_field_event(event_name: str) -> bool:
    """Check if event is a field event (higher = better)."""
    if not event_name:
        return False
    event_lower = event_name.lower()
    field_keywords = ['jump', 'vault', 'put', 'throw', 'discus', 'javelin', 'hammer', 'decathlon', 'heptathlon']
    return any(kw in event_lower for kw in field_keywords)


def calculate_consistency_score(performances: List[float], is_field: bool = False) -> Dict:
    """
    Calculate performance consistency score (0-100).

    Higher score = more consistent performer.
    Based on coefficient of variation (CV) of recent performances.

    Args:
        performances: List of numeric performance values
        is_field: If True, higher values are better

    Returns:
        Dict with score, cv, std_dev, interpretation
    """
    if len(performances) < 3:
        return {
            'score': None,
            'cv': None,
            'std_dev': None,
            'interpretation': 'Insufficient data (need 3+ results)',
            'color': GRAY_BLUE
        }

    mean_perf = np.mean(performances)
    std_dev = np.std(performances)

    # Coefficient of variation (CV) - lower is more consistent
    cv = (std_dev / mean_perf) * 100 if mean_perf != 0 else 0

    # Convert CV to 0-100 score (lower CV = higher score)
    # CV of 0% = 100 score, CV of 10% = 0 score
    score = max(0, min(100, 100 - (cv * 10)))

    # Interpretation
    if score >= 85:
        interpretation = 'Elite Consistency'
        color = TEAL_PRIMARY
    elif score >= 70:
        interpretation = 'Very Consistent'
        color = TEAL_LIGHT
    elif score >= 55:
        interpretation = 'Moderately Consistent'
        color = GOLD_ACCENT
    elif score >= 40:
        interpretation = 'Variable'
        color = '#FFA500'
    else:
        interpretation = 'Highly Variable'
        color = '#dc3545'

    return {
        'score': round(score, 1),
        'cv': round(cv, 2),
        'std_dev': round(std_dev, 3),
        'interpretation': interpretation,
        'color': color
    }


def calculate_near_miss(
    athlete_pb: float,
    standard: float,
    is_field: bool = False
) -> Dict:
    """
    Calculate how close an athlete is to a qualification standard.

    Args:
        athlete_pb: Athlete's personal best
        standard: Qualification standard
        is_field: If True, need to exceed standard (higher = better)

    Returns:
        Dict with gap, percentage, status, color
    """
    if athlete_pb is None or standard is None:
        return {
            'gap': None,
            'percentage': None,
            'status': 'No data',
            'color': GRAY_BLUE
        }

    if is_field:
        gap = standard - athlete_pb  # Positive = still need more distance
        qualified = athlete_pb >= standard
    else:
        gap = athlete_pb - standard  # Positive = still need faster time
        qualified = athlete_pb <= standard

    percentage = (abs(gap) / standard) * 100

    if qualified:
        status = 'QUALIFIED'
        color = TEAL_PRIMARY
    elif percentage <= 1.0:
        status = 'Within 1%'
        color = TEAL_LIGHT
    elif percentage <= 2.0:
        status = 'Within 2%'
        color = GOLD_ACCENT
    elif percentage <= 3.0:
        status = 'Within 3%'
        color = '#FFA500'
    elif percentage <= 5.0:
        status = 'Within 5%'
        color = '#FF6B6B'
    else:
        status = f'{percentage:.1f}% away'
        color = GRAY_BLUE

    return {
        'gap': round(gap, 3),
        'percentage': round(percentage, 2),
        'status': status,
        'color': color,
        'qualified': qualified
    }


def head_to_head_comparison(
    athlete1_results: pd.DataFrame,
    athlete2_results: pd.DataFrame,
    event: str
) -> Dict:
    """
    Compare two athletes head-to-head in a specific event.

    Args:
        athlete1_results: DataFrame of athlete 1's results
        athlete2_results: DataFrame of athlete 2's results
        event: Event name to compare

    Returns:
        Dict with comparison stats
    """
    is_field = is_field_event(event)

    # Filter to event
    a1 = athlete1_results[athlete1_results['event'] == event].copy()
    a2 = athlete2_results[athlete2_results['event'] == event].copy()

    if a1.empty or a2.empty:
        return {'error': 'Insufficient data for comparison'}

    # Parse results
    a1['result_numeric'] = a1['result'].apply(lambda x: parse_result_to_seconds(x, event))
    a2['result_numeric'] = a2['result'].apply(lambda x: parse_result_to_seconds(x, event))

    a1 = a1.dropna(subset=['result_numeric'])
    a2 = a2.dropna(subset=['result_numeric'])

    if a1.empty or a2.empty:
        return {'error': 'No valid results to compare'}

    # Calculate stats
    a1_pb = a1['result_numeric'].max() if is_field else a1['result_numeric'].min()
    a2_pb = a2['result_numeric'].max() if is_field else a2['result_numeric'].min()

    a1_avg = a1['result_numeric'].mean()
    a2_avg = a2['result_numeric'].mean()

    a1_recent = a1.nlargest(3, 'date')['result_numeric'].mean() if 'date' in a1.columns else a1_avg
    a2_recent = a2.nlargest(3, 'date')['result_numeric'].mean() if 'date' in a2.columns else a2_avg

    # Direct meetings (same competition/date)
    direct_wins = {'athlete1': 0, 'athlete2': 0, 'meetings': 0}
    if 'date' in a1.columns and 'date' in a2.columns:
        a1['date'] = pd.to_datetime(a1['date'], errors='coerce')
        a2['date'] = pd.to_datetime(a2['date'], errors='coerce')

        common_dates = set(a1['date'].dt.date) & set(a2['date'].dt.date)
        for date in common_dates:
            a1_on_date = a1[a1['date'].dt.date == date]['result_numeric'].iloc[0] if len(a1[a1['date'].dt.date == date]) > 0 else None
            a2_on_date = a2[a2['date'].dt.date == date]['result_numeric'].iloc[0] if len(a2[a2['date'].dt.date == date]) > 0 else None

            if a1_on_date and a2_on_date:
                direct_wins['meetings'] += 1
                if is_field:
                    if a1_on_date > a2_on_date:
                        direct_wins['athlete1'] += 1
                    else:
                        direct_wins['athlete2'] += 1
                else:
                    if a1_on_date < a2_on_date:
                        direct_wins['athlete1'] += 1
                    else:
                        direct_wins['athlete2'] += 1

    # Determine who's better
    if is_field:
        pb_advantage = 'athlete1' if a1_pb > a2_pb else 'athlete2'
        recent_advantage = 'athlete1' if a1_recent > a2_recent else 'athlete2'
    else:
        pb_advantage = 'athlete1' if a1_pb < a2_pb else 'athlete2'
        recent_advantage = 'athlete1' if a1_recent < a2_recent else 'athlete2'

    return {
        'athlete1': {
            'pb': a1_pb,
            'avg': round(a1_avg, 3),
            'recent_avg': round(a1_recent, 3),
            'total_results': len(a1)
        },
        'athlete2': {
            'pb': a2_pb,
            'avg': round(a2_avg, 3),
            'recent_avg': round(a2_recent, 3),
            'total_results': len(a2)
        },
        'direct_meetings': direct_wins,
        'pb_advantage': pb_advantage,
        'recent_advantage': recent_advantage,
        'is_field': is_field
    }


def country_comparison(
    df: pd.DataFrame,
    countries: List[str],
    event: str,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Compare country performance in a specific event.

    Args:
        df: Master rankings DataFrame
        countries: List of country codes to compare
        event: Event name
        top_n: Top N athletes per country

    Returns:
        DataFrame with country comparison stats
    """
    is_field = is_field_event(event)

    # Filter to event and countries
    event_df = df[(df['event'] == event) & (df['nat'].isin(countries))].copy()

    if event_df.empty:
        return pd.DataFrame()

    # Parse results
    event_df['result_numeric'] = event_df['result'].apply(lambda x: parse_result_to_seconds(x, event))
    event_df = event_df.dropna(subset=['result_numeric'])

    results = []
    for country in countries:
        country_df = event_df[event_df['nat'] == country]

        if country_df.empty:
            continue

        # Get top N athletes by best performance
        if is_field:
            top_athletes = country_df.groupby('competitor')['result_numeric'].max().nlargest(top_n)
        else:
            top_athletes = country_df.groupby('competitor')['result_numeric'].min().nsmallest(top_n)

        results.append({
            'Country': country,
            'Athletes': len(country_df['competitor'].unique()),
            'Top 1': top_athletes.iloc[0] if len(top_athletes) > 0 else None,
            'Top 3 Avg': top_athletes.head(3).mean() if len(top_athletes) >= 3 else top_athletes.mean(),
            'Top 5 Avg': top_athletes.head(5).mean() if len(top_athletes) >= 5 else top_athletes.mean(),
            'Depth (Top 10)': len(country_df['competitor'].unique()) if len(country_df['competitor'].unique()) >= 10 else len(country_df['competitor'].unique())
        })

    result_df = pd.DataFrame(results)

    # Sort by Top 1 (best performance)
    if not result_df.empty:
        result_df = result_df.sort_values('Top 1', ascending=not is_field)

    return result_df


def get_regional_rivals() -> List[str]:
    """Return list of KSA's regional rival country codes."""
    return ['KSA', 'BRN', 'QAT', 'UAE', 'KUW', 'OMA', 'IRN', 'JOR']
```

**Step 2: Verify module loads correctly**

Run: `python -c "from analytics_helpers import *; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add analytics_helpers.py
git commit -m "feat: add analytics helpers module with consistency, near miss, H2H functions"
```

---

## Task 2: Add Performance Consistency Score to Athlete Profiles (Tab 2)

**Files:**
- Modify: `World_Ranking_Deploy_v3.py:1019-1130` (trend analysis section)

**Step 1: Add import for analytics_helpers at top of file**

After line 41 (after data_connector imports), add:

```python
# Import analytics helpers
try:
    from analytics_helpers import (
        calculate_consistency_score, calculate_near_miss,
        head_to_head_comparison, country_comparison,
        parse_result_to_seconds, is_field_event, get_regional_rivals
    )
    ANALYTICS_HELPERS_AVAILABLE = True
except ImportError:
    ANALYTICS_HELPERS_AVAILABLE = False
```

**Step 2: Add Consistency Score metric to Athlete Profiles**

In Tab 2 trend analysis section (around line 1085), after the trend_cols metrics, add a new row:

Find this code block (approximately line 1130):
```python
                                        with trend_cols[4]:
                                            avg_result = athlete_results_copy['result_numeric'].mean()
                                            st.markdown(f"""
```

After the closing of trend_cols, add:

```python
                                        # === CONSISTENCY SCORE ===
                                        if ANALYTICS_HELPERS_AVAILABLE and len(athlete_results_copy) >= 5:
                                            recent_perfs = athlete_results_copy.tail(10)['result_numeric'].tolist()
                                            consistency = calculate_consistency_score(recent_perfs, is_field)

                                            if consistency['score'] is not None:
                                                st.markdown("---")
                                                cons_cols = st.columns([2, 1, 1])
                                                with cons_cols[0]:
                                                    st.markdown(f"""
                                                    <div style="background: {consistency['color']}; padding: 1rem; border-radius: 8px;">
                                                        <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Consistency Score</p>
                                                        <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0;">{consistency['score']}/100</p>
                                                        <p style="color: rgba(255,255,255,0.9); margin: 0.25rem 0 0 0; font-size: 0.9rem;">{consistency['interpretation']}</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                with cons_cols[1]:
                                                    st.markdown(f"""
                                                    <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; text-align: center;">
                                                        <p style="color: #aaa; margin: 0; font-size: 0.75rem;">Std Deviation</p>
                                                        <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{consistency['std_dev']}</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                with cons_cols[2]:
                                                    st.markdown(f"""
                                                    <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; text-align: center;">
                                                        <p style="color: #aaa; margin: 0; font-size: 0.75rem;">CV %</p>
                                                        <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0;">{consistency['cv']}%</p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
```

**Step 3: Test locally**

Run: `streamlit run World_Ranking_Deploy_v3.py`
Navigate to Tab 2, select an athlete with 5+ results, verify Consistency Score appears.

**Step 4: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "feat: add Performance Consistency Score to Athlete Profiles"
```

---

## Task 3: Add Near Miss Qualification Alert (Tab 5)

**Files:**
- Modify: `World_Ranking_Deploy_v3.py:1403-1433` (KSA Athletes section in Tab 5)

**Step 1: Add Near Miss section after KSA qualified athletes**

Find this block (around line 1432):
```python
            if not ksa_reserve.empty:
                st.info(f"**{len(ksa_reserve)} KSA athletes in reserve positions** (may qualify if others withdraw)")
```

After the reserve section (around line 1433), add a new Near Miss Analysis section:

```python
        # === NEAR MISS ANALYSIS ===
        st.markdown("---")
        st.subheader("Near Miss Analysis - Athletes Close to Standards")
        st.markdown(f"""
        <p style='color: #ccc; font-size: 0.9em;'>
        KSA athletes within <strong style="color: {GOLD_ACCENT};">5%</strong> of qualification standards who could break through with focused training.
        </p>
        """, unsafe_allow_html=True)

        # Get KSA athletes from master data
        if DATA_CONNECTOR_AVAILABLE and ANALYTICS_HELPERS_AVAILABLE:
            try:
                ksa_rankings = get_ksa_rankings()
                if ksa_rankings is not None and not ksa_rankings.empty:
                    # Get qualification standards
                    qual_stds = qual_standards_df if qual_standards_df is not None else pd.DataFrame()

                    near_miss_data = []

                    # Group by athlete and event
                    for (athlete, event), group in ksa_rankings.groupby(['competitor', 'event']):
                        # Parse results and get PB
                        results = group['result'].apply(lambda x: parse_result_to_seconds(x, event)).dropna()
                        if results.empty:
                            continue

                        is_field = is_field_event(event)
                        pb = results.max() if is_field else results.min()

                        # Look for matching standard (simplified - check qual_stds)
                        # For now use hardcoded Tokyo 2025 standards
                        tokyo_standards = {
                            '100m': 10.00, '200m': 20.24, '400m': 44.90,
                            '800m': 103.50, '1500m': 213.50,  # in seconds
                            'long-jump': 8.27, 'high-jump': 2.33, 'triple-jump': 17.22,
                            'shot-put': 21.10, 'discus-throw': 66.00, 'javelin-throw': 85.20
                        }

                        # Normalize event name
                        event_normalized = event.lower().replace(' ', '-').replace('metres', 'm')
                        standard = tokyo_standards.get(event_normalized)

                        if standard:
                            near_miss = calculate_near_miss(pb, standard, is_field)

                            if not near_miss['qualified'] and near_miss['percentage'] is not None and near_miss['percentage'] <= 5.0:
                                near_miss_data.append({
                                    'Athlete': athlete,
                                    'Event': event,
                                    'PB': pb,
                                    'Standard': standard,
                                    'Gap': near_miss['gap'],
                                    'Status': near_miss['status'],
                                    'Color': near_miss['color']
                                })

                    if near_miss_data:
                        # Sort by closest to standard
                        near_miss_df = pd.DataFrame(near_miss_data).sort_values('Gap', key=abs)

                        # Display as cards
                        for _, row in near_miss_df.head(10).iterrows():
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.3) 100%);
                                        border-left: 4px solid {row['Color']}; padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="color: white; font-size: 1.1rem;">{row['Athlete']}</strong>
                                        <p style="color: #aaa; margin: 0.25rem 0 0 0;">{row['Event']}</p>
                                    </div>
                                    <div style="text-align: right;">
                                        <p style="color: {row['Color']}; font-weight: bold; margin: 0; font-size: 1.2rem;">{row['Status']}</p>
                                        <p style="color: #aaa; margin: 0; font-size: 0.85rem;">PB: {row['PB']:.2f} | Std: {row['Standard']}</p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No KSA athletes currently within 5% of major championship standards.")
            except Exception as e:
                st.warning(f"Near miss analysis unavailable: {str(e)[:100]}")
```

**Step 2: Test locally**

Run: `streamlit run World_Ranking_Deploy_v3.py`
Navigate to Tab 5, scroll to Near Miss section, verify athletes within 5% are shown.

**Step 3: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "feat: add Near Miss Qualification Alert to Tab 5"
```

---

## Task 4: Add Country Comparison Feature (New Section in Tab 4)

**Files:**
- Modify: `World_Ranking_Deploy_v3.py:1308-1335` (Tab 4 - Saudi Athletes Rankings)

**Step 1: Add Country Comparison section to Tab 4**

Find Tab 4 content (around line 1308) and add after the existing content:

```python
        # === REGIONAL COMPARISON ===
        st.markdown("---")
        st.subheader("Regional Comparison - KSA vs Gulf & Middle East")

        if DATA_CONNECTOR_AVAILABLE and ANALYTICS_HELPERS_AVAILABLE:
            try:
                # Get full rankings data
                full_rankings = get_rankings_data()

                if full_rankings is not None and not full_rankings.empty:
                    rival_countries = get_regional_rivals()

                    # Event selector
                    events_available = sorted(full_rankings['event'].dropna().unique())

                    comp_col1, comp_col2 = st.columns([2, 1])
                    with comp_col1:
                        selected_event_comp = st.selectbox(
                            "Select Event for Comparison",
                            events_available,
                            key="country_comp_event"
                        )
                    with comp_col2:
                        selected_gender_comp = st.selectbox(
                            "Gender",
                            ['Men', 'Women'],
                            key="country_comp_gender"
                        )

                    # Filter by gender if column exists
                    filtered_df = full_rankings.copy()
                    if 'gender' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['gender'].str.lower() == selected_gender_comp.lower()]

                    # Get comparison data
                    comparison_df = country_comparison(
                        filtered_df,
                        rival_countries,
                        selected_event_comp,
                        top_n=5
                    )

                    if not comparison_df.empty:
                        # Highlight KSA row
                        def highlight_ksa(row):
                            if row['Country'] == 'KSA':
                                return [f'background-color: {TEAL_PRIMARY}; color: white'] * len(row)
                            return [''] * len(row)

                        st.dataframe(
                            comparison_df.style.apply(highlight_ksa, axis=1).format({
                                'Top 1': '{:.2f}',
                                'Top 3 Avg': '{:.2f}',
                                'Top 5 Avg': '{:.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                        # Visual comparison chart
                        is_field = is_field_event(selected_event_comp)

                        fig_comp = go.Figure()

                        colors = [TEAL_PRIMARY if c == 'KSA' else GRAY_BLUE for c in comparison_df['Country']]

                        fig_comp.add_trace(go.Bar(
                            x=comparison_df['Country'],
                            y=comparison_df['Top 1'],
                            name='Best Performance',
                            marker_color=colors,
                            text=comparison_df['Top 1'].round(2),
                            textposition='outside'
                        ))

                        fig_comp.update_layout(
                            title=f"{selected_event_comp} - Regional Best Performances ({selected_gender_comp})",
                            xaxis_title="Country",
                            yaxis_title="Performance",
                            yaxis=dict(autorange='reversed') if not is_field else dict(),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            height=400
                        )
                        fig_comp.update_xaxes(showgrid=False)
                        fig_comp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

                        st.plotly_chart(fig_comp, use_container_width=True)

                        # KSA position insight
                        ksa_rank = comparison_df[comparison_df['Country'] == 'KSA'].index[0] + 1 if 'KSA' in comparison_df['Country'].values else None
                        if ksa_rank:
                            total_countries = len(comparison_df)
                            if ksa_rank == 1:
                                st.success(f"KSA leads the region in {selected_event_comp}!")
                            elif ksa_rank <= 3:
                                st.info(f"KSA ranks #{ksa_rank} of {total_countries} regional rivals in {selected_event_comp}")
                            else:
                                st.warning(f"KSA ranks #{ksa_rank} of {total_countries} - opportunity for improvement in {selected_event_comp}")
                    else:
                        st.info(f"No data available for {selected_event_comp} comparison")
            except Exception as e:
                st.warning(f"Country comparison unavailable: {str(e)[:100]}")
```

**Step 2: Test locally**

Run: `streamlit run World_Ranking_Deploy_v3.py`
Navigate to Tab 4, scroll to Regional Comparison, select an event, verify chart displays.

**Step 3: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "feat: add Country Comparison feature to Tab 4"
```

---

## Task 5: Add Head-to-Head Comparison (Tab 2)

**Files:**
- Modify: `World_Ranking_Deploy_v3.py:911-1000` (Tab 2 header area)

**Step 1: Add H2H comparison UI after athlete profile header**

Find the Tab 2 content start (around line 911) and add a new expander for H2H:

After the athlete profile display (around line 1200, after the progression chart), add:

```python
                                # === HEAD-TO-HEAD COMPARISON ===
                                st.markdown("---")
                                with st.expander("Head-to-Head Comparison", expanded=False):
                                    st.markdown(f"""
                                    <p style='color: #ccc;'>Compare <strong style='color: {GOLD_ACCENT};'>{athlete_name}</strong> against another athlete</p>
                                    """, unsafe_allow_html=True)

                                    if DATA_CONNECTOR_AVAILABLE and ANALYTICS_HELPERS_AVAILABLE:
                                        try:
                                            # Get list of all athletes for comparison
                                            all_rankings = get_rankings_data()

                                            if all_rankings is not None and not all_rankings.empty:
                                                # Get athletes in same event
                                                if primary_event:
                                                    event_athletes = all_rankings[all_rankings['event'] == primary_event]['competitor'].unique()
                                                    event_athletes = [a for a in event_athletes if a.upper() != athlete_name.upper()]
                                                    event_athletes = sorted(event_athletes)[:100]  # Limit for performance

                                                    if event_athletes:
                                                        h2h_opponent = st.selectbox(
                                                            "Select opponent to compare",
                                                            event_athletes,
                                                            key=f"h2h_opponent_{athlete_name}"
                                                        )

                                                        if st.button("Compare", key=f"h2h_btn_{athlete_name}"):
                                                            # Get opponent data
                                                            opponent_results = all_rankings[all_rankings['competitor'].str.upper() == h2h_opponent.upper()]

                                                            if not opponent_results.empty:
                                                                h2h = head_to_head_comparison(
                                                                    athlete_results,
                                                                    opponent_results,
                                                                    primary_event
                                                                )

                                                                if 'error' not in h2h:
                                                                    # Display comparison
                                                                    h2h_cols = st.columns(3)

                                                                    with h2h_cols[0]:
                                                                        st.markdown(f"""
                                                                        <div style="background: {TEAL_PRIMARY}; padding: 1rem; border-radius: 8px; text-align: center;">
                                                                            <p style="color: white; font-weight: bold; margin: 0;">{athlete_name}</p>
                                                                            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0;">PB: {h2h['athlete1']['pb']:.2f}</p>
                                                                            <p style="color: rgba(255,255,255,0.8); margin: 0;">Recent: {h2h['athlete1']['recent_avg']:.2f}</p>
                                                                        </div>
                                                                        """, unsafe_allow_html=True)

                                                                    with h2h_cols[1]:
                                                                        # Direct meetings
                                                                        meetings = h2h['direct_meetings']
                                                                        st.markdown(f"""
                                                                        <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; text-align: center;">
                                                                            <p style="color: #aaa; margin: 0;">Direct Meetings</p>
                                                                            <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                                                                                {meetings['athlete1']} - {meetings['athlete2']}
                                                                            </p>
                                                                            <p style="color: #aaa; margin: 0;">{meetings['meetings']} races</p>
                                                                        </div>
                                                                        """, unsafe_allow_html=True)

                                                                    with h2h_cols[2]:
                                                                        st.markdown(f"""
                                                                        <div style="background: {GOLD_ACCENT}; padding: 1rem; border-radius: 8px; text-align: center;">
                                                                            <p style="color: white; font-weight: bold; margin: 0;">{h2h_opponent}</p>
                                                                            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0;">PB: {h2h['athlete2']['pb']:.2f}</p>
                                                                            <p style="color: rgba(255,255,255,0.8); margin: 0;">Recent: {h2h['athlete2']['recent_avg']:.2f}</p>
                                                                        </div>
                                                                        """, unsafe_allow_html=True)

                                                                    # Advantage summary
                                                                    pb_winner = athlete_name if h2h['pb_advantage'] == 'athlete1' else h2h_opponent
                                                                    recent_winner = athlete_name if h2h['recent_advantage'] == 'athlete1' else h2h_opponent

                                                                    st.markdown(f"""
                                                                    <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
                                                                        <p style="color: white; margin: 0;">
                                                                            <strong>PB Advantage:</strong> <span style="color: {GOLD_ACCENT};">{pb_winner}</span> |
                                                                            <strong>Recent Form:</strong> <span style="color: {TEAL_LIGHT};">{recent_winner}</span>
                                                                        </p>
                                                                    </div>
                                                                    """, unsafe_allow_html=True)
                                                                else:
                                                                    st.warning(h2h['error'])
                                                            else:
                                                                st.warning("Could not find opponent data")
                                                    else:
                                                        st.info("No other athletes found in this event for comparison")
                                        except Exception as e:
                                            st.warning(f"H2H comparison error: {str(e)[:100]}")
```

**Step 2: Test locally**

Run: `streamlit run World_Ranking_Deploy_v3.py`
Navigate to Tab 2, select an athlete, expand H2H Comparison, select opponent, click Compare.

**Step 3: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "feat: add Head-to-Head Comparison to Athlete Profiles"
```

---

## Task 6: Dashboard Performance Optimization

**Files:**
- Modify: `World_Ranking_Deploy_v3.py:700-720` (lazy loading section)

**Step 1: Add pagination helper for large datasets**

After the existing cache functions, add:

```python
def paginate_dataframe(df: pd.DataFrame, page_size: int = 100, page_num: int = 0) -> pd.DataFrame:
    """Return a page of the dataframe for display."""
    start_idx = page_num * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]
```

**Step 2: Update Tab 3 (Combined Rankings) to use pagination**

Find Tab 3 content (around line 1275) and modify to add pagination:

```python
with tab3:
    st.header("Combined Global Rankings")

    # Lazy load the large rankings data
    with st.spinner("Loading rankings data..."):
        men_df = load_men_rankings()
        women_df = load_women_rankings()

    # Gender filter
    gender_filter = st.radio("Select Gender", ['Men', 'Women'], horizontal=True, key="combined_gender")

    display_df = men_df if gender_filter == 'Men' else women_df

    if display_df is not None and not display_df.empty:
        # Event filter
        events = sorted(display_df['Event Type'].dropna().unique()) if 'Event Type' in display_df.columns else []
        selected_event = st.selectbox("Filter by Event", ['All Events'] + events, key="combined_event")

        if selected_event != 'All Events':
            display_df = display_df[display_df['Event Type'] == selected_event]

        # Pagination
        total_rows = len(display_df)
        page_size = 100
        total_pages = (total_rows // page_size) + (1 if total_rows % page_size > 0 else 0)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page_num = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=max(1, total_pages),
                value=1,
                key="rankings_page"
            ) - 1  # 0-indexed

        st.caption(f"Showing {page_num * page_size + 1}-{min((page_num + 1) * page_size, total_rows)} of {total_rows:,} records")

        # Display paginated data
        page_df = paginate_dataframe(display_df, page_size, page_num)
        st.dataframe(page_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No rankings data available")
```

**Step 3: Test locally**

Run: `streamlit run World_Ranking_Deploy_v3.py`
Navigate to Tab 3, verify pagination controls appear and work.

**Step 4: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "perf: add pagination to Combined Rankings for faster loading"
```

---

## Task 7: Final Integration Test and Push

**Step 1: Run full application test**

Run: `streamlit run World_Ranking_Deploy_v3.py`

Test checklist:
- [ ] Tab 2: Consistency Score appears for athletes with 5+ results
- [ ] Tab 2: H2H Comparison expander works
- [ ] Tab 4: Country Comparison chart displays
- [ ] Tab 5: Near Miss section shows athletes within 5% of standards
- [ ] Tab 3: Pagination works correctly
- [ ] All existing tabs still function

**Step 2: Final commit and push**

```bash
git status
git add .
git commit -m "feat: complete dashboard analytics enhancements - consistency, near miss, H2H, country comparison"
git push origin main
```

---

## Summary of Changes

| Feature | Location | Lines Added |
|---------|----------|-------------|
| Analytics Helpers Module | `analytics_helpers.py` (new) | ~250 |
| Performance Consistency Score | Tab 2 | ~35 |
| Near Miss Qualification Alert | Tab 5 | ~80 |
| Country Comparison | Tab 4 | ~100 |
| Head-to-Head Comparison | Tab 2 | ~90 |
| Pagination for Rankings | Tab 3 | ~30 |

**Total: ~585 lines of new code**

## Dependencies
- No new packages required
- Uses existing: pandas, numpy, plotly, streamlit

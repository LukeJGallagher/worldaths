# Race Intelligence & Enhanced Analytics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add race simulation, competitor intelligence, improved charts, career timelines, and standards progression to the Saudi Athletics Dashboard.

**Architecture:** Extend existing `chart_components.py` for styling, add form calculation functions to `projection_engine.py`, create new helper module `race_intelligence.py`, and add new Tab 11 + enhance Tabs 2 and 5 in `World_Ranking_Deploy_v3.py`.

**Tech Stack:** Streamlit, Plotly, Pandas, DuckDB (via data_connector)

---

## Task 1: Chart Redesign - Update Configuration

**Files:**
- Modify: `chart_components.py:27-55`

**Step 1: Update COLORS dict with official Saudi Green**

Replace lines 27-43 in `chart_components.py`:

```python
# Team Saudi color palette - Official Saudi Green
COLORS = {
    'primary': '#005430',      # Saudi Green (official PMS 3425 C)
    'secondary': '#a08e66',    # Gold accent
    'dark': '#003d1f',         # Dark green
    'light': '#2A8F5C',        # Light green
    'gray': '#78909C',         # Gray blue
    'success': '#005430',      # Saudi Green (positive)
    'warning': '#FFB800',      # Gold/yellow
    'danger': '#dc3545',       # Red
    'medal_gold': '#FFD700',
    'medal_silver': '#C0C0C0',
    'medal_bronze': '#CD7F32',
    'background': 'white',
    'text': '#333333',
    'grid': 'lightgray'
}

# Chart typography - larger for presentations
CHART_FONTS = {
    'title': 20,
    'axis': 14,
    'legend': 13,
    'annotation': 12,
    'tick': 12
}
```

**Step 2: Update get_base_layout() with larger fonts**

Replace lines 46-55 in `chart_components.py`:

```python
def get_base_layout() -> Dict:
    """Get base Plotly layout with Team Saudi styling."""
    return {
        'plot_bgcolor': COLORS['background'],
        'paper_bgcolor': COLORS['background'],
        'font': {
            'family': 'Inter, sans-serif',
            'color': COLORS['text'],
            'size': CHART_FONTS['tick']
        },
        'title': {'font': {'size': CHART_FONTS['title']}},
        'margin': {'l': 60, 'r': 30, 't': 60, 'b': 50},
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'y': -0.15,
            'font': {'size': CHART_FONTS['legend']}
        },
        'xaxis': {'tickfont': {'size': CHART_FONTS['tick']}},
        'yaxis': {'tickfont': {'size': CHART_FONTS['tick']}}
    }
```

**Step 3: Verify changes**

Run: `python -c "from chart_components import COLORS, CHART_FONTS, get_base_layout; print('OK:', COLORS['primary'], CHART_FONTS['title'])"`

Expected: `OK: #005430 20`

**Step 4: Commit**

```bash
git add chart_components.py
git commit -m "style: update chart colors to official Saudi Green and larger fonts"
```

---

## Task 2: Add Rich Tooltip Template to chart_components.py

**Files:**
- Modify: `chart_components.py` (add after CHART_FONTS, around line 60)

**Step 1: Add hover template function**

Add after the CHART_FONTS dict:

```python
def get_performance_hovertemplate(include_venue: bool = True) -> str:
    """Get rich hover template for performance charts."""
    if include_venue:
        return (
            '<b>%{customdata[0]}</b><br>'
            'Result: %{y:.2f}<br>'
            'Venue: %{customdata[1]}<br>'
            'Date: %{customdata[2]}<br>'
            '<extra></extra>'
        )
    return (
        '<b>%{x|%d %b %Y}</b><br>'
        'Result: %{y:.2f}<br>'
        '<extra></extra>'
    )


def add_medal_lines(fig: go.Figure, benchmarks: Dict[str, float], event_type: str = 'time'):
    """Add gold/silver/bronze standard lines to a chart."""
    medal_styles = {
        'gold': {'color': COLORS['medal_gold'], 'dash': 'solid', 'width': 2, 'label': 'Gold'},
        'silver': {'color': COLORS['medal_silver'], 'dash': 'dash', 'width': 2, 'label': 'Silver'},
        'bronze': {'color': COLORS['medal_bronze'], 'dash': 'dot', 'width': 2, 'label': 'Bronze'},
        'final': {'color': COLORS['gray'], 'dash': 'dashdot', 'width': 1.5, 'label': 'Final (8th)'}
    }

    for key, value in benchmarks.items():
        if value is not None and key in medal_styles:
            style = medal_styles[key]
            fig.add_hline(
                y=value,
                line_dash=style['dash'],
                line_color=style['color'],
                line_width=style['width'],
                annotation_text=f"{style['label']}: {value:.2f}",
                annotation_position='right',
                annotation_font_size=CHART_FONTS['annotation']
            )
    return fig
```

**Step 2: Verify changes**

Run: `python -c "from chart_components import get_performance_hovertemplate, add_medal_lines; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add chart_components.py
git commit -m "feat: add rich hover templates and medal line overlays"
```

---

## Task 3: Enhanced Form Score Function

**Files:**
- Modify: `projection_engine.py:195-232`

**Step 1: Replace calculate_form_score with enhanced version**

Replace the existing function at line 195:

```python
def calculate_form_score(
    performances: List[float],
    event_type: str = 'time',
    pb: float = None,
    last_comp_days: int = None
) -> Dict:
    """
    Calculate comprehensive form score (0-100) with status.

    Components:
    - 40% weight: Recent average vs PB (closer = higher)
    - 30% weight: Trend direction (improving/stable/declining)
    - 20% weight: Recency of last competition (within 30 days = bonus)
    - 10% weight: Consistency (low variance = bonus)

    Args:
        performances: List of recent performances (most recent first)
        event_type: 'time', 'distance', or 'points'
        pb: Personal best (optional, calculated from performances if not provided)
        last_comp_days: Days since last competition (optional)

    Returns:
        Dict with 'score', 'status', 'icon', 'color', 'trend'
    """
    if len(performances) < 2:
        return {
            'score': 50.0,
            'status': 'Unknown',
            'icon': '❓',
            'color': '#78909C',
            'trend': 'stable'
        }

    # Calculate PB if not provided
    if pb is None:
        pb = min(performances) if event_type == 'time' else max(performances)

    current = performances[0]
    avg_last_5 = sum(performances[:5]) / min(len(performances), 5)

    # 1. PB proximity score (40%)
    if event_type == 'time':
        pb_score = max(0, 100 - ((current - pb) / pb * 100 * 10))
    else:
        pb_score = max(0, 100 - ((pb - current) / pb * 100 * 10))

    # 2. Trend score (30%)
    trend = detect_trend(performances, event_type)
    trend_scores = {'improving': 100, 'stable': 60, 'declining': 20}
    trend_score = trend_scores.get(trend, 60)

    # 3. Recency score (20%)
    if last_comp_days is not None:
        if last_comp_days <= 14:
            recency_score = 100
        elif last_comp_days <= 30:
            recency_score = 80
        elif last_comp_days <= 60:
            recency_score = 50
        else:
            recency_score = 20
    else:
        recency_score = 60  # Neutral if unknown

    # 4. Consistency score (10%)
    if len(performances) >= 3:
        import statistics
        cv = statistics.stdev(performances) / statistics.mean(performances)
        consistency_score = max(0, 100 - (cv * 500))  # Lower CV = higher score
    else:
        consistency_score = 50

    # Weighted total
    total_score = (
        pb_score * 0.40 +
        trend_score * 0.30 +
        recency_score * 0.20 +
        consistency_score * 0.10
    )
    total_score = max(0, min(100, total_score))

    # Determine status and icon
    if total_score >= 85 and (last_comp_days is None or last_comp_days <= 14):
        status, icon, color = 'Hot', '🔥', '#005430'
    elif total_score >= 70 and trend == 'improving':
        status, icon, color = 'Rising', '📈', '#2A8F5C'
    elif total_score >= 55:
        status, icon, color = 'Stable', '──', '#a08e66'
    elif total_score >= 40:
        status, icon, color = 'Cooling', '📉', '#FFB800'
    else:
        status, icon, color = 'Cold', '❄️', '#dc3545'

    return {
        'score': round(total_score, 1),
        'status': status,
        'icon': icon,
        'color': color,
        'trend': trend,
        'avg_last_5': round(avg_last_5, 2),
        'pb': pb
    }
```

**Step 2: Verify changes**

Run: `python -c "from projection_engine import calculate_form_score; r = calculate_form_score([44.5, 44.8, 45.0, 45.2], 'time', last_comp_days=10); print(r['score'], r['status'], r['icon'])"`

Expected: A score around 70-85, 'Hot' or 'Rising' status

**Step 3: Commit**

```bash
git add projection_engine.py
git commit -m "feat: enhance form score with status icons and weighted components"
```

---

## Task 4: Create race_intelligence.py Module

**Files:**
- Create: `race_intelligence.py`

**Step 1: Create the module with core functions**

```python
"""
Race Intelligence Module for World Athletics Dashboard

Provides:
- Competitor form analysis with best races
- Round-by-round advancement probabilities
- Race preview builder
- Historical standards progression
- Career milestone detection

Uses data_connector for data access and projection_engine for calculations.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Import dependencies
try:
    from data_connector import get_competitors, get_rankings_data, get_benchmarks_data, query
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False

try:
    from projection_engine import calculate_form_score, detect_trend
    PROJECTION_AVAILABLE = True
except ImportError:
    PROJECTION_AVAILABLE = False


# Round qualifying time estimates (based on historical data)
ROUND_STANDARDS = {
    'Asian Games': {
        '100m': {'heat': 10.50, 'semi': 10.25, 'final': 10.10, 'gold': 9.95, 'silver': 10.00, 'bronze': 10.05},
        '200m': {'heat': 21.00, 'semi': 20.60, 'final': 20.35, 'gold': 20.00, 'silver': 20.10, 'bronze': 20.20},
        '400m': {'heat': 46.50, 'semi': 45.50, 'final': 45.00, 'gold': 44.30, 'silver': 44.50, 'bronze': 44.70},
        '800m': {'heat': 108.0, 'semi': 106.0, 'final': 104.5, 'gold': 103.5, 'silver': 104.0, 'bronze': 104.5},
        '1500m': {'heat': 228.0, 'semi': 222.0, 'final': 218.0, 'gold': 214.0, 'silver': 215.0, 'bronze': 216.0},
    },
    'World Championships': {
        '100m': {'heat': 10.20, 'semi': 10.05, 'final': 9.95, 'gold': 9.80, 'silver': 9.85, 'bronze': 9.90},
        '200m': {'heat': 20.50, 'semi': 20.20, 'final': 20.00, 'gold': 19.70, 'silver': 19.80, 'bronze': 19.90},
        '400m': {'heat': 45.50, 'semi': 44.80, 'final': 44.40, 'gold': 43.80, 'silver': 44.00, 'bronze': 44.20},
        '800m': {'heat': 106.0, 'semi': 104.0, 'final': 103.0, 'gold': 102.0, 'silver': 102.5, 'bronze': 103.0},
        '1500m': {'heat': 222.0, 'semi': 216.0, 'final': 212.0, 'gold': 208.0, 'silver': 210.0, 'bronze': 211.0},
    },
    'Olympics': {
        '100m': {'heat': 10.15, 'semi': 10.00, 'final': 9.90, 'gold': 9.75, 'silver': 9.80, 'bronze': 9.85},
        '200m': {'heat': 20.40, 'semi': 20.10, 'final': 19.90, 'gold': 19.60, 'silver': 19.70, 'bronze': 19.80},
        '400m': {'heat': 45.30, 'semi': 44.60, 'final': 44.20, 'gold': 43.50, 'silver': 43.80, 'bronze': 44.00},
        '800m': {'heat': 105.0, 'semi': 103.5, 'final': 102.5, 'gold': 101.5, 'silver': 102.0, 'bronze': 102.5},
        '1500m': {'heat': 220.0, 'semi': 214.0, 'final': 210.0, 'gold': 206.0, 'silver': 208.0, 'bronze': 209.0},
    }
}


def get_competitor_form_cards(
    event: str,
    gender: str = 'Men',
    limit: int = 10,
    season: int = None
) -> List[Dict]:
    """
    Get top competitors with form analysis and best races.

    Returns list of dicts with:
    - athlete_name, country_code
    - form_score, form_status, form_icon, form_color
    - pb, avg_last_5, trend
    - best_2_races: [{result, venue, date}, ...]
    - last_comp: {result, venue, days_ago}
    """
    if not DATA_AVAILABLE:
        return []

    if season is None:
        season = datetime.now().year

    # Get competitors from data
    competitors_df = get_competitors(event, gender, limit * 2, season)

    if competitors_df is None or competitors_df.empty:
        return []

    results = []

    for _, row in competitors_df.head(limit).iterrows():
        athlete_name = row.get('athlete_name', 'Unknown')
        country_code = row.get('country_code', '')

        # Get athlete's recent results
        athlete_results = get_athlete_recent_results(athlete_name, event, limit=10)

        if not athlete_results:
            continue

        performances = [r['result_numeric'] for r in athlete_results if r.get('result_numeric')]

        if not performances:
            continue

        # Calculate form score
        days_since = None
        if athlete_results and athlete_results[0].get('date'):
            try:
                last_date = pd.to_datetime(athlete_results[0]['date'])
                days_since = (datetime.now() - last_date).days
            except:
                pass

        form = calculate_form_score(performances, 'time', last_comp_days=days_since) if PROJECTION_AVAILABLE else {
            'score': 50, 'status': 'Unknown', 'icon': '❓', 'color': '#78909C', 'trend': 'stable'
        }

        # Get best 2 races
        sorted_results = sorted(athlete_results, key=lambda x: x.get('result_numeric', 999))[:2]
        best_2 = [{
            'result': r.get('result', r.get('result_numeric', 0)),
            'venue': r.get('venue', 'Unknown'),
            'date': r.get('date', '')
        } for r in sorted_results]

        # Last competition
        last_comp = {
            'result': athlete_results[0].get('result', ''),
            'venue': athlete_results[0].get('venue', ''),
            'days_ago': days_since or 0
        } if athlete_results else None

        results.append({
            'athlete_name': athlete_name,
            'country_code': country_code,
            'form_score': form['score'],
            'form_status': form['status'],
            'form_icon': form['icon'],
            'form_color': form['color'],
            'pb': min(performances) if performances else None,
            'avg_last_5': form.get('avg_last_5', sum(performances[:5])/min(5, len(performances))),
            'trend': form['trend'],
            'best_2_races': best_2,
            'last_comp': last_comp
        })

    # Sort by form score descending
    results.sort(key=lambda x: x['form_score'], reverse=True)
    return results


def get_athlete_recent_results(athlete_name: str, event: str = None, limit: int = 10) -> List[Dict]:
    """Get athlete's recent competition results."""
    if not DATA_AVAILABLE:
        return []

    try:
        sql = f"""
            SELECT
                result,
                result_numeric,
                date,
                venue,
                event,
                rank
            FROM master
            WHERE competitor ILIKE '%{athlete_name.replace("'", "''")}%'
        """
        if event:
            event_pattern = event.lower().replace(' ', '-').replace('m', '-metres')
            sql += f" AND event ILIKE '%{event_pattern}%'"

        sql += f" ORDER BY date DESC LIMIT {limit}"

        df = query(sql)
        if df is not None and not df.empty:
            return df.to_dict('records')
    except Exception:
        pass

    return []


def calculate_advancement_probability(
    athlete_form_avg: float,
    round_standard: float,
    event_type: str = 'time'
) -> float:
    """
    Calculate probability of advancing past a round.

    Based on gap between athlete's form and round standard.
    """
    if event_type == 'time':
        gap = athlete_form_avg - round_standard  # Positive = slower than needed
        # Convert gap to probability (0.5s slower = ~30% chance)
        if gap <= 0:
            prob = 95 + (gap * 10)  # Faster than standard = high prob
        else:
            prob = 95 - (gap * 60)  # Slower = lower prob
    else:
        gap = round_standard - athlete_form_avg  # Positive = shorter than needed
        if gap <= 0:
            prob = 95 + (gap * 5)
        else:
            prob = 95 - (gap * 30)

    return max(1, min(99, prob))


def build_race_preview(
    athletes: List[Dict],
    championship: str,
    event: str,
    event_type: str = 'time'
) -> Dict:
    """
    Build race preview with round-by-round probabilities.

    Args:
        athletes: List of dicts with 'name', 'form_avg', 'country'
        championship: 'Asian Games', 'World Championships', or 'Olympics'
        event: Event name
        event_type: 'time' or 'distance'

    Returns:
        Dict with per-athlete advancement probabilities and targets
    """
    standards = ROUND_STANDARDS.get(championship, {}).get(event, {})

    if not standards:
        # Use default standards if not found
        standards = {'heat': 99, 'semi': 99, 'final': 99, 'gold': 99, 'silver': 99, 'bronze': 99}

    preview = {
        'championship': championship,
        'event': event,
        'standards': standards,
        'athletes': []
    }

    for athlete in athletes:
        form_avg = athlete.get('form_avg', athlete.get('avg_last_5', 0))

        probs = {}
        for round_name in ['heat', 'semi', 'final']:
            if round_name in standards:
                probs[round_name] = {
                    'probability': calculate_advancement_probability(form_avg, standards[round_name], event_type),
                    'need': standards[round_name]
                }

        # Medal probabilities
        for medal in ['gold', 'silver', 'bronze']:
            if medal in standards:
                probs[medal] = {
                    'probability': calculate_advancement_probability(form_avg, standards[medal], event_type) * 0.5,
                    'need': standards[medal]
                }

        # Calculate gap to bronze
        bronze_std = standards.get('bronze', standards.get('final', 0))
        if event_type == 'time':
            gap = form_avg - bronze_std
        else:
            gap = bronze_std - form_avg

        preview['athletes'].append({
            'name': athlete.get('name', 'Unknown'),
            'country': athlete.get('country', ''),
            'form_avg': form_avg,
            'probabilities': probs,
            'gap_to_bronze': round(gap, 2)
        })

    return preview


def get_career_milestones(athlete_name: str, event: str = None) -> List[Dict]:
    """
    Extract career milestones from athlete's result history.

    Milestone types:
    - pb: Personal best broken
    - medal: Medal at major championship
    - first: Career first (sub-X, DL debut, etc.)
    - title: Championship title
    """
    results = get_athlete_recent_results(athlete_name, event, limit=100)

    if not results:
        return []

    milestones = []
    seen_pbs = set()
    current_pb = None

    # Sort by date ascending to detect PB progression
    results_sorted = sorted(results, key=lambda x: x.get('date', ''), reverse=False)

    for result in results_sorted:
        perf = result.get('result_numeric')
        date = result.get('date', '')
        venue = result.get('venue', '')
        rank = result.get('rank')

        if not perf or not date:
            continue

        year = str(date)[:4] if date else ''

        # Check for PB
        if current_pb is None or perf < current_pb:
            if current_pb is not None:  # Not the first result
                milestones.append({
                    'type': 'pb',
                    'icon': '⭐',
                    'date': date,
                    'year': year,
                    'description': f'PB! {result.get("result", perf)}',
                    'venue': venue
                })
            current_pb = perf

        # Check for medal (rank 1-3 at major venue)
        major_venues = ['Olympic', 'World', 'Asian Games', 'Commonwealth', 'Diamond League']
        if rank and int(rank) <= 3:
            is_major = any(mv.lower() in venue.lower() for mv in major_venues)
            if is_major:
                medal_icons = {1: '🥇', 2: '🥈', 3: '🥉'}
                milestones.append({
                    'type': 'medal',
                    'icon': medal_icons.get(int(rank), '🏅'),
                    'date': date,
                    'year': year,
                    'description': f'{result.get("result", perf)} ({venue})',
                    'venue': venue,
                    'rank': rank
                })

        # Check for title (rank 1)
        if rank and int(rank) == 1:
            title_keywords = ['Championship', 'Games', 'National', 'Trials']
            is_title = any(kw.lower() in venue.lower() for kw in title_keywords)
            if is_title and 'medal' not in [m['type'] for m in milestones if m['date'] == date]:
                milestones.append({
                    'type': 'title',
                    'icon': '🏆',
                    'date': date,
                    'year': year,
                    'description': f'Champion: {venue}',
                    'venue': venue
                })

    # Sort by date descending (most recent first)
    milestones.sort(key=lambda x: x['date'], reverse=True)
    return milestones


def get_standards_progression(event: str, gender: str = 'Men') -> pd.DataFrame:
    """
    Get historical qualification standards for an event.

    Returns DataFrame with Year, Olympic, World Champs, Asian Games columns.
    """
    # Hardcoded historical data (extend benchmarks.parquet in future)
    historical_standards = {
        '100m': {
            'Men': [
                {'Year': 2024, 'Olympic': 10.00, 'World Champs': 10.00, 'Asian Games': 10.25},
                {'Year': 2023, 'Olympic': None, 'World Champs': 10.00, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 10.05, 'Asian Games': 10.30},
                {'Year': 2021, 'Olympic': 10.05, 'World Champs': None, 'Asian Games': None},
                {'Year': 2019, 'Olympic': None, 'World Champs': 10.10, 'Asian Games': None},
            ],
            'Women': [
                {'Year': 2024, 'Olympic': 11.07, 'World Champs': 11.07, 'Asian Games': 11.50},
                {'Year': 2023, 'Olympic': None, 'World Champs': 11.15, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 11.15, 'Asian Games': 11.60},
                {'Year': 2021, 'Olympic': 11.15, 'World Champs': None, 'Asian Games': None},
            ]
        },
        '400m': {
            'Men': [
                {'Year': 2024, 'Olympic': 44.90, 'World Champs': 45.00, 'Asian Games': 46.00},
                {'Year': 2023, 'Olympic': None, 'World Champs': 45.00, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 45.20, 'Asian Games': 46.20},
                {'Year': 2021, 'Olympic': 44.90, 'World Champs': None, 'Asian Games': None},
            ]
        }
    }

    event_data = historical_standards.get(event, {}).get(gender, [])

    if not event_data:
        return pd.DataFrame()

    return pd.DataFrame(event_data)
```

**Step 2: Verify module loads**

Run: `python -c "from race_intelligence import get_competitor_form_cards, build_race_preview, get_career_milestones; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add race_intelligence.py
git commit -m "feat: add race_intelligence module with form cards, race preview, milestones"
```

---

## Task 5: Add Competitor Form Cards to Tab 2

**Files:**
- Modify: `World_Ranking_Deploy_v3.py` (after Tab 2 H2H section, around line 2400)

**Step 1: Add import at top of file (around line 50)**

```python
try:
    from race_intelligence import get_competitor_form_cards, build_race_preview, get_career_milestones, get_standards_progression
    RACE_INTELLIGENCE_AVAILABLE = True
except ImportError:
    RACE_INTELLIGENCE_AVAILABLE = False
```

**Step 2: Add Competitor Form section in Tab 2**

Find the end of the H2H comparison expander in Tab 2 (search for "Head-to-Head" in the file) and add after it:

```python
                    # === TOP COMPETITORS FORM ANALYSIS ===
                    if RACE_INTELLIGENCE_AVAILABLE and primary_event:
                        st.markdown("---")
                        with st.expander("🎯 Top Competitors", expanded=True):
                            st.markdown(f"<p style='color: #888;'>Top competitors for <b style='color: {GOLD_ACCENT};'>{primary_event}</b> ranked by current form</p>", unsafe_allow_html=True)

                            competitors = get_competitor_form_cards(primary_event, gender='Men', limit=5)

                            if competitors:
                                for i, comp in enumerate(competitors, 1):
                                    is_ksa = comp['country_code'] == 'KSA'
                                    bg_color = 'rgba(0, 84, 48, 0.15)' if is_ksa else 'rgba(0,0,0,0.05)'
                                    border_color = TEAL_PRIMARY if is_ksa else comp['form_color']

                                    best_races_html = ""
                                    for race in comp.get('best_2_races', [])[:2]:
                                        best_races_html += f"<div>⭐ {race['result']}  {race['venue'][:20]}  {str(race['date'])[:10]}</div>"

                                    last = comp.get('last_comp', {})
                                    last_html = f"{last.get('result', '')} @ {last.get('venue', '')[:15]} ({last.get('days_ago', 0)}d ago)" if last else ""

                                    st.markdown(f"""
                                    <div style="background: {bg_color}; border-left: 4px solid {border_color};
                                                padding: 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <div>
                                                <b style="font-size: 1.1rem;">{i}. {comp['athlete_name']}</b>
                                                <span style="color: #888;">({comp['country_code']})</span>
                                            </div>
                                            <div style="text-align: right;">
                                                <span style="color: {comp['form_color']}; font-size: 1.2rem; font-weight: bold;">
                                                    {comp['form_icon']} {comp['form_score']:.0f}
                                                </span>
                                            </div>
                                        </div>
                                        <div style="color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
                                            PB: {comp['pb']:.2f} | Avg: {comp['avg_last_5']:.2f} | {comp['form_status']}
                                        </div>
                                        <div style="color: #888; font-size: 0.85rem; margin-top: 0.5rem;">
                                            <b>Best 2:</b><br>{best_races_html}
                                        </div>
                                        <div style="color: #aaa; font-size: 0.8rem; margin-top: 0.25rem;">
                                            Last: {last_html}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No competitor data available for this event")
```

**Step 3: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "feat: add competitor form cards to athlete profiles (Tab 2)"
```

---

## Task 6: Add Career Milestones Timeline to Tab 2

**Files:**
- Modify: `World_Ranking_Deploy_v3.py` (after competitor form cards section)

**Step 1: Add career timeline expander**

```python
                    # === CAREER MILESTONES TIMELINE ===
                    if RACE_INTELLIGENCE_AVAILABLE:
                        st.markdown("---")
                        with st.expander("📅 Career Milestones", expanded=False):
                            milestones = get_career_milestones(athlete_name, primary_event)

                            if milestones:
                                # Group by year
                                years = {}
                                for m in milestones:
                                    year = m.get('year', 'Unknown')
                                    if year not in years:
                                        years[year] = []
                                    years[year].append(m)

                                # Display timeline
                                for year in sorted(years.keys(), reverse=True):
                                    st.markdown(f"### {year}")
                                    for m in years[year]:
                                        icon = m.get('icon', '●')
                                        desc = m.get('description', '')
                                        date_str = str(m.get('date', ''))[:10]

                                        st.markdown(f"""
                                        <div style="display: flex; align-items: center; margin: 0.5rem 0; padding-left: 1rem; border-left: 2px solid {TEAL_PRIMARY};">
                                            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                                            <div>
                                                <b>{desc}</b>
                                                <span style="color: #888; font-size: 0.85rem; margin-left: 0.5rem;">{date_str}</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("No career milestones detected")
```

**Step 2: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "feat: add career milestones timeline to athlete profiles (Tab 2)"
```

---

## Task 7: Add Standards Progression to Tab 5

**Files:**
- Modify: `World_Ranking_Deploy_v3.py` (inside Tab 5, after existing content, around line 2700)

**Step 1: Add standards progression expander**

Find `with tab5:` and add before the closing of the tab:

```python
    # === STANDARDS PROGRESSION HISTORY ===
    st.markdown("---")
    with st.expander("📈 Qualification Standards History", expanded=False):
        st.markdown(f"""
        <p style='color: #888;'>Track how qualification standards have changed over the years</p>
        """, unsafe_allow_html=True)

        prog_col1, prog_col2 = st.columns(2)
        with prog_col1:
            prog_event = st.selectbox("Event", ['100m', '200m', '400m', '800m', '1500m'], key='prog_event')
        with prog_col2:
            prog_gender = st.selectbox("Gender", ['Men', 'Women'], key='prog_gender')

        if RACE_INTELLIGENCE_AVAILABLE:
            standards_df = get_standards_progression(prog_event, prog_gender)

            if not standards_df.empty:
                # Line chart
                import plotly.graph_objects as go
                fig = go.Figure()

                for col in ['Olympic', 'World Champs', 'Asian Games']:
                    if col in standards_df.columns:
                        valid_data = standards_df[standards_df[col].notna()]
                        if not valid_data.empty:
                            fig.add_trace(go.Scatter(
                                x=valid_data['Year'],
                                y=valid_data[col],
                                mode='lines+markers',
                                name=col,
                                line={'width': 3},
                                marker={'size': 10}
                            ))

                fig.update_layout(
                    title=f'{prog_event} {prog_gender} Qualification Standards',
                    xaxis_title='Year',
                    yaxis_title='Standard',
                    yaxis={'autorange': 'reversed'},  # Lower time = better
                    height=400,
                    plot_bgcolor='white',
                    font={'family': 'Inter, sans-serif'}
                )

                st.plotly_chart(fig, use_container_width=True)

                # Table
                st.dataframe(standards_df, use_container_width=True, hide_index=True)

                # Trend insight
                if len(standards_df) >= 2:
                    latest = standards_df.iloc[0]
                    oldest = standards_df.iloc[-1]
                    for col in ['Olympic', 'World Champs']:
                        if col in standards_df.columns and pd.notna(latest[col]) and pd.notna(oldest[col]):
                            change = latest[col] - oldest[col]
                            years = latest['Year'] - oldest['Year']
                            if years > 0:
                                per_year = change / years
                                direction = "tightening" if change < 0 else "loosening"
                                st.info(f"💡 {col} standards {direction} by ~{abs(per_year):.2f}s per year")
            else:
                st.info(f"No historical data available for {prog_event}")
        else:
            st.warning("Race intelligence module not available")
```

**Step 2: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "feat: add standards progression history to Tab 5"
```

---

## Task 8: Create Race Intelligence Tab (Tab 11)

**Files:**
- Modify: `World_Ranking_Deploy_v3.py` (add new tab after tab10)

**Step 1: Update tab creation (line ~1074)**

Change:
```python
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
```

To:
```python
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    'Event Standards & Progression',
    'Athlete Profiles',
    'Combined Rankings',
    'Saudi Athletes Rankings',
    'World Champs Qualification',
    'Major Games Analytics',
    'What It Takes to Win (Live)',
    'AI Analyst',
    'Coach View',
    'Project East 2026',
    'Race Intelligence'  # NEW TAB
])
```

**Step 2: Add Tab 11 content at end of file (before final comments)**

```python
###################################
# Tab 11: Race Intelligence
###################################
with tab11:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_DARK} 100%);
         padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 4px solid {GOLD_ACCENT};">
        <h2 style="color: white; margin: 0;">🏁 Race Intelligence</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Build race previews, analyze competitor form, and calculate advancement probabilities
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not RACE_INTELLIGENCE_AVAILABLE:
        st.warning("Race Intelligence module not available. Check race_intelligence.py")
    else:
        # Event and Championship selection
        ri_col1, ri_col2, ri_col3 = st.columns(3)

        with ri_col1:
            ri_event = st.selectbox("Event", ['100m', '200m', '400m', '800m', '1500m'], key='ri_event')
        with ri_col2:
            ri_gender = st.selectbox("Gender", ['Men', 'Women'], key='ri_gender')
        with ri_col3:
            ri_championship = st.selectbox("Championship", ['Asian Games', 'World Championships', 'Olympics'], key='ri_champ')

        st.markdown("---")

        # Two sections
        form_tab, preview_tab = st.tabs(["📊 Form Rankings", "🎯 Race Preview"])

        with form_tab:
            st.subheader("Current Form Rankings")
            st.caption(f"Athletes ranked by current form, not just PB - {ri_event} {ri_gender}")

            competitors = get_competitor_form_cards(ri_event, ri_gender, limit=20)

            if competitors:
                # Create DataFrame for display
                form_data = []
                for i, c in enumerate(competitors, 1):
                    form_data.append({
                        'Rank': i,
                        'Athlete': c['athlete_name'],
                        'NAT': c['country_code'],
                        'Form': f"{c['form_score']:.0f} {c['form_icon']}",
                        'Avg (5)': f"{c['avg_last_5']:.2f}",
                        'PB': f"{c['pb']:.2f}" if c['pb'] else '-',
                        'Last': f"{c['last_comp']['days_ago']}d" if c.get('last_comp') else '-'
                    })

                form_df = pd.DataFrame(form_data)

                # Highlight KSA athletes
                def highlight_ksa(row):
                    if row['NAT'] == 'KSA':
                        return [f'background-color: rgba(0, 84, 48, 0.2)'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    form_df.style.apply(highlight_ksa, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )
            else:
                st.info("No competitor data available")

        with preview_tab:
            st.subheader("Race Preview Builder")
            st.caption("Select athletes and see round-by-round advancement probabilities")

            # Get available athletes
            if competitors:
                athlete_options = [f"{c['athlete_name']} ({c['country_code']})" for c in competitors[:15]]

                selected_athletes = st.multiselect(
                    "Select Athletes (max 8)",
                    athlete_options,
                    default=athlete_options[:4],
                    max_selections=8,
                    key='ri_athletes'
                )

                if selected_athletes and st.button("Build Race Preview", type="primary"):
                    # Build athlete data for preview
                    preview_athletes = []
                    for sel in selected_athletes:
                        name = sel.split(' (')[0]
                        country = sel.split('(')[1].replace(')', '') if '(' in sel else ''

                        # Find in competitors
                        for c in competitors:
                            if c['athlete_name'] == name:
                                preview_athletes.append({
                                    'name': name,
                                    'country': country,
                                    'form_avg': c['avg_last_5']
                                })
                                break

                    if preview_athletes:
                        preview = build_race_preview(preview_athletes, ri_championship, ri_event)

                        # Display results
                        st.markdown(f"### {ri_championship} - {ri_event}")

                        # Round standards
                        standards = preview.get('standards', {})
                        std_cols = st.columns(6)
                        for i, (round_name, std) in enumerate(standards.items()):
                            with std_cols[i % 6]:
                                st.metric(round_name.title(), f"{std:.2f}")

                        st.markdown("---")

                        # Per-athlete probabilities
                        for athlete in preview['athletes']:
                            is_ksa = athlete['country'] == 'KSA'
                            bg = 'rgba(0, 84, 48, 0.15)' if is_ksa else 'rgba(0,0,0,0.05)'

                            probs = athlete['probabilities']

                            st.markdown(f"""
                            <div style="background: {bg}; padding: 1rem; margin: 0.5rem 0; border-radius: 8px;">
                                <b>{athlete['name']}</b> ({athlete['country']}) - Form: {athlete['form_avg']:.2f}
                            </div>
                            """, unsafe_allow_html=True)

                            prob_cols = st.columns(6)
                            for i, (round_name, prob_data) in enumerate(probs.items()):
                                with prob_cols[i]:
                                    prob = prob_data['probability']
                                    need = prob_data['need']
                                    color = TEAL_PRIMARY if prob >= 70 else (GOLD_ACCENT if prob >= 40 else '#dc3545')
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 0.5rem; background: {color}; color: white; border-radius: 4px;">
                                        <div style="font-size: 1.2rem; font-weight: bold;">{prob:.0f}%</div>
                                        <div style="font-size: 0.8rem;">{round_name.title()}</div>
                                        <div style="font-size: 0.7rem;">Need: {need:.2f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                            # Gap to bronze
                            gap = athlete['gap_to_bronze']
                            gap_color = TEAL_PRIMARY if gap <= 0 else '#dc3545'
                            st.markdown(f"<p style='color: {gap_color}; font-size: 0.9rem;'>Gap to Bronze: {gap:+.2f}s</p>", unsafe_allow_html=True)
            else:
                st.info("Load competitor data first from the Form Rankings tab")
```

**Step 3: Commit**

```bash
git add World_Ranking_Deploy_v3.py
git commit -m "feat: add Race Intelligence tab with form rankings and race preview"
```

---

## Task 9: Final Verification and Testing

**Step 1: Syntax check**

Run: `python -c "import ast; ast.parse(open('World_Ranking_Deploy_v3.py').read()); print('Syntax OK')"`

**Step 2: Import check**

Run: `python -c "from race_intelligence import *; from projection_engine import *; from chart_components import *; print('All imports OK')"`

**Step 3: Run dashboard**

Run: `streamlit run World_Ranking_Deploy_v3.py`

Test checklist:
- [ ] Tab 2: Competitor Form Cards appear with form scores and best races
- [ ] Tab 2: Career Milestones Timeline shows PBs and medals
- [ ] Tab 5: Standards Progression chart displays year-over-year trends
- [ ] Tab 11: Form Rankings table shows athletes ranked by form
- [ ] Tab 11: Race Preview Builder calculates advancement probabilities
- [ ] Charts have larger fonts (20pt titles)
- [ ] Colors use official Saudi Green (#005430)

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete Race Intelligence implementation

- Chart redesign with larger fonts and Saudi Green
- Competitor form cards with best races (Tab 2)
- Career milestones timeline (Tab 2)
- Standards progression history (Tab 5)
- Race Intelligence tab with form rankings (Tab 11)
- Race preview builder with probabilities (Tab 11)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Component | Lines | Files Modified |
|------|-----------|-------|----------------|
| 1-2 | Chart Redesign | ~60 | chart_components.py |
| 3 | Form Score Enhancement | ~80 | projection_engine.py |
| 4 | Race Intelligence Module | ~350 | race_intelligence.py (new) |
| 5 | Competitor Form Cards | ~60 | World_Ranking_Deploy_v3.py |
| 6 | Career Timeline | ~50 | World_Ranking_Deploy_v3.py |
| 7 | Standards Progression | ~70 | World_Ranking_Deploy_v3.py |
| 8 | Race Intelligence Tab | ~200 | World_Ranking_Deploy_v3.py |

**Total: ~870 lines of code across 4 files**

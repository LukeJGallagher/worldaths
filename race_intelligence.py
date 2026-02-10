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

# Event name mapping: UI display name -> database format
EVENT_NAME_MAP = {
    '100m': '100-metres',
    '200m': '200-metres',
    '400m': '400-metres',
    '800m': '800-metres',
    '1500m': '1500-metres',
    '5000m': '5000-metres',
    '10000m': '10000-metres',
    '110m Hurdles': '110-metres-hurdles',
    '100m Hurdles': '100-metres-hurdles',
    '400m Hurdles': '400-metres-hurdles',
    '3000m Steeplechase': '3000-metres-steeplechase',
    'High Jump': 'high-jump',
    'Pole Vault': 'pole-vault',
    'Long Jump': 'long-jump',
    'Triple Jump': 'triple-jump',
    'Shot Put': 'shot-put',
    'Discus Throw': 'discus-throw',
    'Hammer Throw': 'hammer-throw',
    'Javelin Throw': 'javelin-throw',
    'Decathlon': 'decathlon',
    'Heptathlon': 'heptathlon',
    'Marathon': 'marathon',
}

# Current data season (updated as new data is scraped)
CURRENT_DATA_SEASON = 2025

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



# Asian country codes for championship-based filtering
ASIAN_COUNTRY_CODES = {
    'AFG', 'BAN', 'BHR', 'BHU', 'BRU', 'CAM', 'CHN', 'TPE', 'HKG', 'IND',
    'INA', 'IDN', 'IRI', 'IRQ', 'JPN', 'JOR', 'KAZ', 'KOR', 'PRK', 'KUW',
    'KGZ', 'LAO', 'LBN', 'MAC', 'MAS', 'MYA', 'MDV', 'MGL', 'NEP', 'OMA',
    'PAK', 'PLE', 'PHI', 'QAT', 'KSA', 'SIN', 'SRI', 'SYR', 'TJK', 'THA',
    'TLS', 'TKM', 'UAE', 'UZB', 'VIE', 'YEM',
}


def _batch_get_recent_results(athlete_names: List[str], event: str, limit_per_athlete: int = 10) -> Dict[str, List[Dict]]:
    """
    Batch fetch recent results for multiple athletes in a single query.
    Eliminates N+1 query problem in get_competitor_form_cards.

    Returns dict mapping athlete_name -> list of result dicts.
    """
    if not DATA_AVAILABLE or not athlete_names:
        return {}

    try:
        # Build athlete name conditions
        safe_names = [n.replace("'", "''") for n in athlete_names]
        name_conditions = " OR ".join([f"competitor ILIKE '%{n}%'" for n in safe_names])

        # Convert event format
        db_event = EVENT_NAME_MAP.get(event, event.lower().replace(' ', '-'))

        # Event filter
        if db_event in EVENT_NAME_MAP.values():
            event_filter = f"(event = '{db_event}' OR event ILIKE '%{db_event}%')"
        else:
            event_filter = f"event ILIKE '%{db_event}%'"

        sql = f"""
            WITH ranked AS (
                SELECT
                    competitor,
                    result,
                    result_numeric,
                    date,
                    venue,
                    event,
                    rank,
                    ROW_NUMBER() OVER (PARTITION BY competitor ORDER BY date DESC) as rn
                FROM master
                WHERE ({name_conditions})
                    AND {event_filter}
            )
            SELECT competitor, result, result_numeric, date, venue, event, rank
            FROM ranked
            WHERE rn <= {limit_per_athlete}
            ORDER BY competitor, date DESC
        """

        df = query(sql)
        if df is None or df.empty:
            return {}

        # Group by athlete name (using fuzzy matching since ILIKE was used)
        results_by_athlete = {}
        for name in athlete_names:
            name_lower = name.lower()
            mask = df['competitor'].str.lower().str.contains(name_lower, na=False)
            athlete_df = df[mask]
            if not athlete_df.empty:
                results_by_athlete[name] = athlete_df.drop(columns=['competitor'], errors='ignore').to_dict('records')

        return results_by_athlete
    except Exception:
        return {}


def get_competitor_form_cards(
    event: str,
    gender: str = 'Men',
    limit: int = 10,
    season: int = None,
    championship: str = None
) -> List[Dict]:
    """
    Get top competitors with form analysis and best races.

    Args:
        championship: Optional - 'Asian Games', 'World Championships', 'Olympics'.
                      If 'Asian Games', filters to Asian country codes only.

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
        # Use latest data season, not current year (which may not have data yet)
        season = CURRENT_DATA_SEASON

    # Get competitors from data (no season filter for now to get best data coverage)
    competitors_df = get_competitors(event, gender, limit * 2, season=None)

    if competitors_df is None or competitors_df.empty:
        return []

    # Collect athlete names for batch query
    athlete_info = []
    for _, row in competitors_df.head(limit).iterrows():
        athlete_name = row.get('athlete_name', row.get('competitor', 'Unknown'))
        country_code = row.get('country_code', row.get('nat', ''))
        athlete_info.append((athlete_name, country_code))

    # Batch fetch all athletes' recent results in ONE query (instead of N queries)
    all_names = [name for name, _ in athlete_info]
    batch_results = _batch_get_recent_results(all_names, event, limit_per_athlete=10)

    results = []

    for athlete_name, country_code in athlete_info:
        athlete_results = batch_results.get(athlete_name, [])

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
            'score': 50, 'status': 'Unknown', 'icon': '?', 'color': '#78909C', 'trend': 'stable'
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
            'date': athlete_results[0].get('date', ''),
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

    # Filter by championship region if specified
    if championship and championship == 'Asian Games' and results:
        results = [r for r in results if r.get('country_code', '').upper() in ASIAN_COUNTRY_CODES]

    # Sort by form score descending
    results.sort(key=lambda x: x['form_score'], reverse=True)
    return results


def get_athlete_recent_results(athlete_name: str, event: str = None, limit: int = 10) -> List[Dict]:
    """Get athlete's recent competition results."""
    if not DATA_AVAILABLE:
        return []

    try:
        # Escape single quotes in athlete name
        safe_name = athlete_name.replace("'", "''")

        sql = f"""
            SELECT
                result,
                result_numeric,
                date,
                venue,
                event,
                rank
            FROM master
            WHERE competitor ILIKE '%{safe_name}%'
        """
        if event:
            # Convert event format using mapping
            db_event = EVENT_NAME_MAP.get(event, event.lower().replace(' ', '-'))
            # Use exact match for main events, ILIKE for variations
            if db_event in EVENT_NAME_MAP.values():
                sql += f" AND (event = '{db_event}' OR event ILIKE '%{db_event}%')"
            else:
                sql += f" AND event ILIKE '%{db_event}%'"

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
        if rank:
            try:
                rank_int = int(rank)
                if rank_int <= 3:
                    is_major = any(mv.lower() in venue.lower() for mv in major_venues)
                    if is_major:
                        medal_icons = {1: '🥇', 2: '🥈', 3: '🥉'}
                        milestones.append({
                            'type': 'medal',
                            'icon': medal_icons.get(rank_int, '🏅'),
                            'date': date,
                            'year': year,
                            'description': f'{result.get("result", perf)} ({venue})',
                            'venue': venue,
                            'rank': rank
                        })

                # Check for title (rank 1)
                if rank_int == 1:
                    title_keywords = ['Championship', 'Games', 'National', 'Trials']
                    is_title = any(kw.lower() in venue.lower() for kw in title_keywords)
                    if is_title and 'medal' not in [m['type'] for m in milestones if m.get('date') == date]:
                        milestones.append({
                            'type': 'title',
                            'icon': '🏆',
                            'date': date,
                            'year': year,
                            'description': f'Champion: {venue}',
                            'venue': venue
                        })
            except (ValueError, TypeError):
                pass

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
        '200m': {
            'Men': [
                {'Year': 2024, 'Olympic': 20.16, 'World Champs': 20.24, 'Asian Games': 20.80},
                {'Year': 2023, 'Olympic': None, 'World Champs': 20.24, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 20.30, 'Asian Games': 20.90},
                {'Year': 2021, 'Olympic': 20.24, 'World Champs': None, 'Asian Games': None},
            ],
            'Women': [
                {'Year': 2024, 'Olympic': 22.57, 'World Champs': 22.80, 'Asian Games': 23.50},
                {'Year': 2023, 'Olympic': None, 'World Champs': 22.80, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 23.00, 'Asian Games': 23.60},
                {'Year': 2021, 'Olympic': 22.80, 'World Champs': None, 'Asian Games': None},
            ]
        },
        '400m': {
            'Men': [
                {'Year': 2024, 'Olympic': 44.90, 'World Champs': 45.00, 'Asian Games': 46.00},
                {'Year': 2023, 'Olympic': None, 'World Champs': 45.00, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 45.20, 'Asian Games': 46.20},
                {'Year': 2021, 'Olympic': 44.90, 'World Champs': None, 'Asian Games': None},
            ],
            'Women': [
                {'Year': 2024, 'Olympic': 50.40, 'World Champs': 51.00, 'Asian Games': 53.00},
                {'Year': 2023, 'Olympic': None, 'World Champs': 51.00, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 51.35, 'Asian Games': 53.50},
                {'Year': 2021, 'Olympic': 51.00, 'World Champs': None, 'Asian Games': None},
            ]
        },
        '800m': {
            'Men': [
                {'Year': 2024, 'Olympic': 103.90, 'World Champs': 104.50, 'Asian Games': 107.00},
                {'Year': 2023, 'Olympic': None, 'World Champs': 104.50, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 105.00, 'Asian Games': 108.00},
                {'Year': 2021, 'Olympic': 104.20, 'World Champs': None, 'Asian Games': None},
            ],
            'Women': [
                {'Year': 2024, 'Olympic': 118.50, 'World Champs': 120.00, 'Asian Games': 125.00},
                {'Year': 2023, 'Olympic': None, 'World Champs': 120.00, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 121.00, 'Asian Games': 126.00},
                {'Year': 2021, 'Olympic': 119.00, 'World Champs': None, 'Asian Games': None},
            ]
        },
        '1500m': {
            'Men': [
                {'Year': 2024, 'Olympic': 213.00, 'World Champs': 214.50, 'Asian Games': 222.00},
                {'Year': 2023, 'Olympic': None, 'World Champs': 214.50, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 215.00, 'Asian Games': 224.00},
                {'Year': 2021, 'Olympic': 213.50, 'World Champs': None, 'Asian Games': None},
            ],
            'Women': [
                {'Year': 2024, 'Olympic': 239.00, 'World Champs': 242.00, 'Asian Games': 255.00},
                {'Year': 2023, 'Olympic': None, 'World Champs': 242.00, 'Asian Games': None},
                {'Year': 2022, 'Olympic': None, 'World Champs': 244.00, 'Asian Games': 258.00},
                {'Year': 2021, 'Olympic': 240.00, 'World Champs': None, 'Asian Games': None},
            ]
        }
    }

    event_data = historical_standards.get(event, {}).get(gender, [])

    if not event_data:
        return pd.DataFrame()

    return pd.DataFrame(event_data)

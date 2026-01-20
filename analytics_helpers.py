"""
Analytics Helper Functions for Saudi Athletics Dashboard

This module provides utility functions for performance analysis including:
- Result parsing and conversion
- Consistency score calculation
- Near miss detection for qualification standards
- Head-to-head athlete comparisons
- Country benchmarking and comparisons

Team Saudi Brand Colors are included for consistent styling.
"""

import re
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np

# =============================================================================
# Team Saudi Brand Colors
# =============================================================================

TEAL_PRIMARY = '#007167'   # Main brand color, headers, primary buttons
GOLD_ACCENT = '#a08e66'    # Highlights, PB markers, secondary accents
TEAL_DARK = '#005a51'      # Hover states, gradients, secondary elements
TEAL_LIGHT = '#009688'     # Secondary positive, good status
GRAY_BLUE = '#78909C'      # Neutral, needs improvement

# Header gradient for Streamlit components
HEADER_GRADIENT = 'linear-gradient(135deg, #007167 0%, #005a51 100%)'

# Status colors for indicators
STATUS_COLORS = {
    'excellent': TEAL_PRIMARY,
    'good': TEAL_LIGHT,
    'warning': GOLD_ACCENT,
    'danger': '#dc3545',
    'neutral': GRAY_BLUE
}

# =============================================================================
# Field Event Detection
# =============================================================================

# Field events where higher values are better
FIELD_EVENTS = [
    'high jump', 'pole vault', 'long jump', 'triple jump',
    'shot put', 'discus throw', 'hammer throw', 'javelin throw',
    'decathlon', 'heptathlon', 'pentathlon'
]

# Combined events (scored by points)
COMBINED_EVENTS = ['decathlon', 'heptathlon', 'pentathlon']


def is_field_event(event_name: str) -> bool:
    """
    Check if an event is a field event (where higher values are better).

    Args:
        event_name: Name of the event (e.g., "100m", "Long Jump", "Shot Put")

    Returns:
        True if field event (higher = better), False if track event (lower = better)

    Examples:
        >>> is_field_event("100m")
        False
        >>> is_field_event("Long Jump")
        True
        >>> is_field_event("Shot Put")
        True
    """
    if not event_name:
        return False

    event_lower = event_name.lower().strip()

    # Check against known field events
    for field_event in FIELD_EVENTS:
        if field_event in event_lower:
            return True

    return False


def is_combined_event(event_name: str) -> bool:
    """
    Check if an event is a combined/multi event (decathlon, heptathlon, etc.).

    Args:
        event_name: Name of the event

    Returns:
        True if combined event, False otherwise
    """
    if not event_name:
        return False

    event_lower = event_name.lower().strip()

    for combined in COMBINED_EVENTS:
        if combined in event_lower:
            return True

    return False


# =============================================================================
# Result Parsing
# =============================================================================

def parse_result_to_seconds(result: Union[str, float, int], event: str = "") -> Optional[float]:
    """
    Convert a result string to numeric seconds/meters/points.

    Handles various formats:
    - Seconds: "10.45" -> 10.45
    - Minutes:Seconds: "1:59.00" -> 119.00
    - Hours:Minutes:Seconds: "2:05:30.00" -> 7530.00
    - Field events: "8.95" -> 8.95 (meters)
    - Combined events: "8500" -> 8500 (points)
    - Wind-adjusted: "10.45 (+1.5)" -> 10.45

    Args:
        result: Result string or numeric value
        event: Event name for context (optional)

    Returns:
        Numeric value or None if parsing fails

    Examples:
        >>> parse_result_to_seconds("10.45")
        10.45
        >>> parse_result_to_seconds("1:59.00")
        119.0
        >>> parse_result_to_seconds("2:05:30.00")
        7530.0
    """
    if result is None:
        return None

    # Already numeric
    if isinstance(result, (int, float)):
        return float(result) if not pd.isna(result) else None

    # Convert to string and clean
    result_str = str(result).strip()

    if not result_str or result_str.lower() in ['dns', 'dnf', 'dq', 'nr', '-', '']:
        return None

    # Remove wind info in parentheses: "10.45 (+1.5)" -> "10.45"
    result_str = re.sub(r'\s*\([^)]*\)', '', result_str).strip()

    # Remove any trailing letters (like 'A' for altitude, 'h' for hand-timed)
    result_str = re.sub(r'[A-Za-z]+$', '', result_str).strip()

    try:
        # Check for time format with colons
        if ':' in result_str:
            parts = result_str.split(':')

            if len(parts) == 2:
                # Minutes:Seconds format (e.g., "1:59.00")
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds

            elif len(parts) == 3:
                # Hours:Minutes:Seconds format (e.g., "2:05:30.00")
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds

        # Simple numeric value
        return float(result_str)

    except (ValueError, TypeError):
        return None


def format_result(seconds: float, event: str = "") -> str:
    """
    Format a numeric result back to display string.

    Args:
        seconds: Numeric value (seconds, meters, or points)
        event: Event name for context

    Returns:
        Formatted result string
    """
    if seconds is None or pd.isna(seconds):
        return "-"

    # Combined events - return as integer points
    if is_combined_event(event):
        return f"{int(seconds)}"

    # Field events - return with 2 decimal places
    if is_field_event(event):
        return f"{seconds:.2f}"

    # Track events
    if seconds >= 3600:
        # Hours:Minutes:Seconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"

    elif seconds >= 60:
        # Minutes:Seconds
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:05.2f}"

    else:
        # Just seconds
        return f"{seconds:.2f}"


# =============================================================================
# Performance Consistency Score
# =============================================================================

def calculate_consistency_score(
    performances: List[float],
    is_field: bool = False,
    min_performances: int = 3
) -> Optional[Dict]:
    """
    Calculate a 0-100 consistency score based on coefficient of variation.

    Lower CV indicates more consistent performances.
    Score formula: 100 - (CV * 100), clamped to 0-100 range.

    Args:
        performances: List of numeric performance values
        is_field: True if field event (affects interpretation)
        min_performances: Minimum performances required for calculation

    Returns:
        Dictionary with:
        - score: 0-100 consistency score
        - cv: Coefficient of variation
        - mean: Mean performance
        - std: Standard deviation
        - count: Number of performances
        - rating: Text rating (Excellent/Good/Average/Inconsistent)

        Returns None if insufficient data

    Examples:
        >>> calculate_consistency_score([10.45, 10.50, 10.48, 10.52])
        {'score': 99.7, 'cv': 0.003, 'rating': 'Excellent', ...}
    """
    # Filter out None values
    valid_performances = [p for p in performances if p is not None and not pd.isna(p)]

    if len(valid_performances) < min_performances:
        return None

    mean_val = np.mean(valid_performances)
    std_val = np.std(valid_performances, ddof=1)  # Sample std dev

    # Avoid division by zero
    if mean_val == 0:
        return None

    # Coefficient of variation
    cv = std_val / abs(mean_val)

    # Convert to 0-100 score (lower CV = higher score)
    # Using a scaling factor for athletics where CV is typically 1-5%
    score = max(0, min(100, 100 - (cv * 500)))

    # Determine rating
    if score >= 95:
        rating = "Excellent"
    elif score >= 85:
        rating = "Good"
    elif score >= 70:
        rating = "Average"
    else:
        rating = "Inconsistent"

    return {
        'score': round(score, 1),
        'cv': round(cv, 4),
        'mean': round(mean_val, 2),
        'std': round(std_val, 3),
        'count': len(valid_performances),
        'best': min(valid_performances) if not is_field else max(valid_performances),
        'worst': max(valid_performances) if not is_field else min(valid_performances),
        'rating': rating,
        'color': STATUS_COLORS.get(
            'excellent' if score >= 95 else
            'good' if score >= 85 else
            'warning' if score >= 70 else 'danger'
        )
    }


# =============================================================================
# Near Miss Detection
# =============================================================================

def calculate_near_miss(
    athlete_pb: float,
    standard: float,
    is_field: bool = False,
    near_miss_threshold: float = 0.02
) -> Dict:
    """
    Calculate the gap between an athlete's PB and a qualification standard.

    Args:
        athlete_pb: Athlete's personal best
        standard: Qualification standard to compare against
        is_field: True if field event (higher = better)
        near_miss_threshold: Percentage threshold for "near miss" (default 2%)

    Returns:
        Dictionary with:
        - gap: Absolute gap to standard
        - gap_percent: Gap as percentage of standard
        - qualified: True if PB meets/beats standard
        - is_near_miss: True if within threshold but not qualified
        - status: Text status (Qualified/Near Miss/Work Needed)
        - improvement_needed: How much improvement needed (0 if qualified)

    Examples:
        >>> calculate_near_miss(10.15, 10.00, is_field=False)
        {'gap': 0.15, 'qualified': False, 'is_near_miss': True, ...}
    """
    if athlete_pb is None or standard is None:
        return {
            'gap': None,
            'gap_percent': None,
            'qualified': False,
            'is_near_miss': False,
            'status': 'No Data',
            'improvement_needed': None,
            'color': GRAY_BLUE
        }

    if is_field:
        # Field events: higher is better
        gap = standard - athlete_pb
        qualified = athlete_pb >= standard
        improvement_needed = max(0, gap)
    else:
        # Track events: lower is better
        gap = athlete_pb - standard
        qualified = athlete_pb <= standard
        improvement_needed = max(0, gap)

    gap_percent = abs(gap) / standard if standard != 0 else 0
    is_near_miss = not qualified and gap_percent <= near_miss_threshold

    # Determine status
    if qualified:
        status = "Qualified"
        color = TEAL_PRIMARY
    elif is_near_miss:
        status = "Near Miss"
        color = GOLD_ACCENT
    else:
        status = "Work Needed"
        color = GRAY_BLUE

    return {
        'gap': round(abs(gap), 3),
        'gap_percent': round(gap_percent * 100, 2),
        'qualified': qualified,
        'is_near_miss': is_near_miss,
        'status': status,
        'improvement_needed': round(improvement_needed, 3) if improvement_needed > 0 else 0,
        'color': color
    }


# =============================================================================
# Head-to-Head Comparison
# =============================================================================

def head_to_head_comparison(
    athlete1_results: List[Dict],
    athlete2_results: List[Dict],
    event: str
) -> Dict:
    """
    Compare two athletes' performances in head-to-head matchups.

    Args:
        athlete1_results: List of dicts with 'date', 'venue', 'result' for athlete 1
        athlete2_results: List of dicts with 'date', 'venue', 'result' for athlete 2
        event: Event name for proper comparison

    Returns:
        Dictionary with:
        - athlete1_wins: Number of head-to-head wins
        - athlete2_wins: Number of head-to-head wins
        - total_meetings: Total head-to-head competitions
        - meetings: List of individual meeting results
        - athlete1_avg: Average performance in meetings
        - athlete2_avg: Average performance in meetings

    Note:
        Meetings are matched by date and venue.
    """
    is_field = is_field_event(event)

    # Create lookup dictionaries by date+venue
    athlete1_lookup = {}
    for r in athlete1_results:
        key = f"{r.get('date', '')}_{r.get('venue', '')}"
        result = parse_result_to_seconds(r.get('result'), event)
        if result is not None:
            athlete1_lookup[key] = {'result': result, **r}

    athlete2_lookup = {}
    for r in athlete2_results:
        key = f"{r.get('date', '')}_{r.get('venue', '')}"
        result = parse_result_to_seconds(r.get('result'), event)
        if result is not None:
            athlete2_lookup[key] = {'result': result, **r}

    # Find common meetings
    common_keys = set(athlete1_lookup.keys()) & set(athlete2_lookup.keys())

    meetings = []
    athlete1_wins = 0
    athlete2_wins = 0
    athlete1_results_numeric = []
    athlete2_results_numeric = []

    for key in sorted(common_keys):
        r1 = athlete1_lookup[key]
        r2 = athlete2_lookup[key]

        result1 = r1['result']
        result2 = r2['result']

        athlete1_results_numeric.append(result1)
        athlete2_results_numeric.append(result2)

        # Determine winner
        if is_field:
            # Higher is better
            if result1 > result2:
                winner = 1
                athlete1_wins += 1
            elif result2 > result1:
                winner = 2
                athlete2_wins += 1
            else:
                winner = 0  # Tie
        else:
            # Lower is better
            if result1 < result2:
                winner = 1
                athlete1_wins += 1
            elif result2 < result1:
                winner = 2
                athlete2_wins += 1
            else:
                winner = 0  # Tie

        meetings.append({
            'date': r1.get('date'),
            'venue': r1.get('venue'),
            'athlete1_result': format_result(result1, event),
            'athlete2_result': format_result(result2, event),
            'winner': winner,
            'margin': abs(result1 - result2)
        })

    return {
        'athlete1_wins': athlete1_wins,
        'athlete2_wins': athlete2_wins,
        'total_meetings': len(meetings),
        'meetings': meetings,
        'athlete1_avg': round(np.mean(athlete1_results_numeric), 3) if athlete1_results_numeric else None,
        'athlete2_avg': round(np.mean(athlete2_results_numeric), 3) if athlete2_results_numeric else None,
        'win_rate_1': round(athlete1_wins / len(meetings) * 100, 1) if meetings else 0,
        'win_rate_2': round(athlete2_wins / len(meetings) * 100, 1) if meetings else 0
    }


# =============================================================================
# Country Benchmarking
# =============================================================================

def get_regional_rivals() -> List[str]:
    """
    Return list of KSA's regional rival country codes.

    These are Middle East and North African countries that compete
    in similar events and championships.

    Returns:
        List of 3-letter country codes
    """
    return [
        'KSA',  # Saudi Arabia
        'QAT',  # Qatar
        'BRN',  # Bahrain
        'UAE',  # United Arab Emirates
        'KUW',  # Kuwait
        'OMA',  # Oman
        'JOR',  # Jordan
        'EGY',  # Egypt
        'MAR',  # Morocco
        'ALG',  # Algeria
        'TUN',  # Tunisia
        'IRQ',  # Iraq
        'LBN',  # Lebanon
        'SYR',  # Syria
        'PLE',  # Palestine
    ]


def country_comparison(
    df: pd.DataFrame,
    countries: List[str],
    event: str,
    top_n: int = 5,
    year: Optional[int] = None
) -> Dict:
    """
    Compare country performance in a specific event.

    Args:
        df: DataFrame with columns ['nat', 'event', 'result', 'date']
        countries: List of country codes to compare
        event: Event name to filter
        top_n: Number of top athletes per country to consider
        year: Optional year filter (filters by date column)

    Returns:
        Dictionary with:
        - rankings: List of countries sorted by best performance
        - country_stats: Dict of stats per country
        - event: Event name
        - is_field: Whether event is field event

    Example:
        >>> country_comparison(df, ['KSA', 'QAT', 'BRN'], '100m', top_n=3)
    """
    is_field = is_field_event(event)

    # Filter dataframe
    event_df = df[df['event'].str.lower().str.contains(event.lower(), na=False)].copy()

    if year:
        event_df['year'] = pd.to_datetime(event_df['date'], errors='coerce').dt.year
        event_df = event_df[event_df['year'] == year]

    # Filter to specified countries
    event_df = event_df[event_df['nat'].isin(countries)]

    # Parse results
    event_df['result_numeric'] = event_df['result'].apply(
        lambda x: parse_result_to_seconds(x, event)
    )
    event_df = event_df.dropna(subset=['result_numeric'])

    country_stats = {}

    for country in countries:
        country_df = event_df[event_df['nat'] == country]

        if country_df.empty:
            country_stats[country] = {
                'best': None,
                'athletes': 0,
                'top_performers': [],
                'avg_top_n': None
            }
            continue

        # Get best result
        if is_field:
            best_idx = country_df['result_numeric'].idxmax()
            sorted_df = country_df.nlargest(top_n, 'result_numeric')
        else:
            best_idx = country_df['result_numeric'].idxmin()
            sorted_df = country_df.nsmallest(top_n, 'result_numeric')

        best_result = country_df.loc[best_idx, 'result_numeric']

        # Get unique athletes count
        athlete_col = 'competitor' if 'competitor' in country_df.columns else 'athlete'
        if athlete_col in country_df.columns:
            unique_athletes = country_df[athlete_col].nunique()
        else:
            unique_athletes = len(country_df)

        # Top performers
        top_performers = []
        for _, row in sorted_df.head(top_n).iterrows():
            top_performers.append({
                'name': row.get('competitor', row.get('athlete', 'Unknown')),
                'result': format_result(row['result_numeric'], event),
                'result_numeric': row['result_numeric']
            })

        # Average of top N
        top_results = sorted_df['result_numeric'].head(top_n).tolist()
        avg_top_n = np.mean(top_results) if top_results else None

        country_stats[country] = {
            'best': best_result,
            'best_formatted': format_result(best_result, event),
            'athletes': unique_athletes,
            'performances': len(country_df),
            'top_performers': top_performers,
            'avg_top_n': round(avg_top_n, 3) if avg_top_n else None
        }

    # Rank countries by best performance
    ranked_countries = sorted(
        [c for c in countries if country_stats[c]['best'] is not None],
        key=lambda c: country_stats[c]['best'],
        reverse=is_field  # Higher is better for field events
    )

    return {
        'rankings': ranked_countries,
        'country_stats': country_stats,
        'event': event,
        'is_field': is_field,
        'top_n': top_n
    }


# =============================================================================
# Utility Functions
# =============================================================================

def get_performance_trend(
    performances: List[Dict],
    event: str,
    min_points: int = 3
) -> Optional[Dict]:
    """
    Calculate performance trend over time.

    Args:
        performances: List of dicts with 'date' and 'result'
        event: Event name
        min_points: Minimum data points required

    Returns:
        Dictionary with trend direction and statistics
    """
    # Sort by date
    sorted_perfs = sorted(
        [p for p in performances if p.get('date') and p.get('result')],
        key=lambda x: x['date']
    )

    if len(sorted_perfs) < min_points:
        return None

    is_field = is_field_event(event)

    # Parse results
    results = [parse_result_to_seconds(p['result'], event) for p in sorted_perfs]
    results = [r for r in results if r is not None]

    if len(results) < min_points:
        return None

    # Calculate simple linear trend
    x = np.arange(len(results))
    slope, intercept = np.polyfit(x, results, 1)

    # For track events, negative slope is improvement
    # For field events, positive slope is improvement
    if is_field:
        improving = slope > 0
    else:
        improving = slope < 0

    return {
        'slope': round(slope, 4),
        'improving': improving,
        'trend': 'Improving' if improving else 'Declining',
        'first_result': results[0],
        'last_result': results[-1],
        'change': round(results[-1] - results[0], 3),
        'change_percent': round((results[-1] - results[0]) / results[0] * 100, 2)
    }

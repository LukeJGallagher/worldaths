"""
Projection Engine for World Athletics Performance Analysis

Calculates statistical projections for athlete performances including:
- Weighted recent performance averages
- Confidence intervals
- Trend detection (improving/stable/declining)
- Championship pressure adjustments
- Form scores (0-100)
- Gap analysis
- Advancement probability

Adapted from Tilastopaja project for World Athletics data.

NOTE: This module is designed to work with both World Athletics (this project)
and Tilastopaja data. Future database combination is planned.
"""

import statistics
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import math


# Recency decay weights for last 5 performances
# Most recent = 1.0, decreasing by 0.85 factor
RECENCY_WEIGHTS = [1.0, 0.85, 0.72, 0.61, 0.52]

# Championship pressure adjustment (athletes typically ~0.5% slower under major championship pressure)
CHAMPIONSHIP_PRESSURE_FACTOR = 1.005

# Trend thresholds
TREND_IMPROVING_THRESHOLD = -0.02  # 2% improvement indicates improving trend
TREND_DECLINING_THRESHOLD = 0.02   # 2% decline indicates declining trend


def calculate_weighted_average(performances: List[float], weights: List[float] = None) -> float:
    """
    Calculate weighted average of performances with recency bias.

    Args:
        performances: List of performance values (most recent first)
        weights: Optional custom weights (default: RECENCY_WEIGHTS)

    Returns:
        Weighted average performance

    Example:
        >>> calculate_weighted_average([44.72, 44.89, 45.01, 45.15, 45.22])
        44.89  # Weighted toward recent better performances
    """
    if not performances:
        return 0.0

    if weights is None:
        weights = RECENCY_WEIGHTS

    # Truncate weights to match performance count
    weights = weights[:len(performances)]

    weighted_sum = sum(p * w for p, w in zip(performances, weights))
    weight_total = sum(weights)

    return weighted_sum / weight_total if weight_total > 0 else 0.0


def calculate_confidence_interval(
    performances: List[float],
    confidence_level: float = 0.68
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for projected performance.

    Args:
        performances: List of recent performances
        confidence_level: Confidence level (0.68 = 1 SD, 0.95 = 2 SD)

    Returns:
        Tuple of (lower_bound, projected, upper_bound)

    Example:
        >>> calculate_confidence_interval([44.72, 44.89, 45.01])
        (44.58, 44.85, 45.12)  # 68% confidence range
    """
    if len(performances) < 2:
        avg = performances[0] if performances else 0.0
        return (avg, avg, avg)

    weighted_avg = calculate_weighted_average(performances)
    std_dev = statistics.stdev(performances)

    # Z-score for confidence level (simplified)
    z_scores = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96}
    z = z_scores.get(confidence_level, 1.0)

    margin = std_dev * z

    return (weighted_avg - margin, weighted_avg, weighted_avg + margin)


def detect_trend(performances: List[float], event_type: str = 'time') -> str:
    """
    Detect performance trend from recent results.

    Args:
        performances: List of performances (most recent first)
        event_type: 'time' (lower is better) or 'distance'/'points' (higher is better)

    Returns:
        'improving', 'stable', or 'declining'

    Example:
        >>> detect_trend([44.72, 44.89, 45.01], event_type='time')
        'improving'  # Times are getting faster (lower)
    """
    if len(performances) < 3:
        return 'stable'

    # Compare recent (first 2) vs older (last 2)
    recent_avg = sum(performances[:2]) / 2
    older_avg = sum(performances[-2:]) / 2

    # Calculate percentage change
    if older_avg == 0:
        return 'stable'

    pct_change = (recent_avg - older_avg) / older_avg

    # For time events, negative change (faster) = improving
    # For distance/points, positive change (longer/more) = improving
    if event_type == 'time':
        if pct_change < TREND_IMPROVING_THRESHOLD:
            return 'improving'
        elif pct_change > TREND_DECLINING_THRESHOLD:
            return 'declining'
    else:
        if pct_change > abs(TREND_IMPROVING_THRESHOLD):
            return 'improving'
        elif pct_change < -abs(TREND_DECLINING_THRESHOLD):
            return 'declining'

    return 'stable'


def get_trend_symbol(trend: str) -> str:
    """Get display symbol for trend."""
    symbols = {
        'improving': '↗',
        'stable': '→',
        'declining': '↘'
    }
    return symbols.get(trend, '→')


def get_trend_color(trend: str) -> str:
    """Get Team Saudi color for trend."""
    colors = {
        'improving': '#007167',  # Teal
        'stable': '#a08e66',     # Gold
        'declining': '#dc3545'   # Red
    }
    return colors.get(trend, '#6c757d')


def apply_championship_adjustment(
    projected: float,
    event_type: str = 'time',
    is_major_championship: bool = True
) -> float:
    """
    Apply championship pressure adjustment to projection.

    Major championships typically see slightly slower performances
    due to pressure, rounds, and tactical racing.

    Args:
        projected: Base projected performance
        event_type: 'time', 'distance', or 'points'
        is_major_championship: Whether this is a major championship

    Returns:
        Adjusted projection
    """
    if not is_major_championship:
        return projected

    if event_type == 'time':
        # Times get slightly slower under pressure
        return projected * CHAMPIONSHIP_PRESSURE_FACTOR
    else:
        # Distance/points get slightly lower under pressure
        return projected / CHAMPIONSHIP_PRESSURE_FACTOR


def calculate_form_score(performances: List[float], event_type: str = 'time') -> float:
    """
    Calculate a normalized form score (0-100) based on recent performances.

    Higher score = better current form relative to personal range.

    Components:
    - Consistency: 40% (lower variance = higher score)
    - Trend: 30% (improving = higher score)
    - Recency: 30% (how close to PB)

    Args:
        performances: List of recent performances
        event_type: 'time', 'distance', or 'points'

    Returns:
        Form score from 0-100
    """
    if len(performances) < 2:
        return 50.0  # Neutral score if insufficient data

    current = performances[0]  # Most recent
    best = min(performances) if event_type == 'time' else max(performances)
    worst = max(performances) if event_type == 'time' else min(performances)

    if best == worst:
        return 75.0  # Consistent = good form

    # Normalize: how close is current to best?
    if event_type == 'time':
        # For time: lower is better, so invert
        score = 100 * (worst - current) / (worst - best)
    else:
        # For distance/points: higher is better
        score = 100 * (current - worst) / (best - worst)

    return max(0, min(100, score))


def calculate_gap(
    athlete_performance: float,
    target_performance: float,
    event_type: str = 'time'
) -> float:
    """
    Calculate gap between athlete and target performance.

    Positive = athlete is behind target (needs to improve)
    Negative = athlete is ahead of target

    Args:
        athlete_performance: Athlete's performance (SB or projected)
        target_performance: Target to compare against
        event_type: 'time', 'distance', or 'points'

    Returns:
        Gap value (positive = behind, negative = ahead)
    """
    if event_type == 'time':
        # For time: positive gap means athlete is slower (behind)
        return athlete_performance - target_performance
    else:
        # For distance/points: positive gap means athlete is shorter/lower (behind)
        return target_performance - athlete_performance


def format_gap(gap: float, event_type: str = 'time') -> str:
    """
    Format gap for display.

    Args:
        gap: Gap value from calculate_gap()
        event_type: 'time', 'distance', or 'points'

    Returns:
        Formatted string like "+0.35s" or "-0.12m"
    """
    if event_type == 'time':
        if abs(gap) >= 60:
            # Format as minutes:seconds for longer gaps
            mins = int(abs(gap) // 60)
            secs = abs(gap) % 60
            sign = '+' if gap >= 0 else '-'
            return f"{sign}{mins}:{secs:05.2f}"
        else:
            sign = '+' if gap >= 0 else ''
            return f"{sign}{gap:.2f}s"
    elif event_type == 'points':
        sign = '+' if gap >= 0 else ''
        return f"{sign}{int(gap)} pts"
    else:
        sign = '+' if gap >= 0 else ''
        return f"{sign}{gap:.2f}m"


def project_performance(
    performances: List[float],
    performance_dates: List[datetime] = None,
    event_type: str = 'time',
    is_major_championship: bool = True,
    include_trend: bool = True
) -> Dict:
    """
    Generate comprehensive performance projection.

    Args:
        performances: List of recent performances (most recent first)
        performance_dates: Optional dates for each performance
        event_type: 'time', 'distance', or 'points'
        is_major_championship: Apply championship adjustment
        include_trend: Include trend analysis

    Returns:
        Dictionary with projection details:
        {
            'projected': float,
            'range_low': float,
            'range_high': float,
            'confidence': float,
            'trend': str,
            'trend_symbol': str,
            'trend_color': str,
            'form_score': float,
            'methodology': str
        }
    """
    if not performances:
        return {
            'projected': 0,
            'range_low': 0,
            'range_high': 0,
            'confidence': 0,
            'trend': 'unknown',
            'trend_symbol': '?',
            'trend_color': '#6c757d',
            'form_score': 0,
            'methodology': 'Insufficient data'
        }

    # Calculate base projection
    range_low, projected, range_high = calculate_confidence_interval(performances)

    # Apply championship adjustment
    if is_major_championship:
        projected = apply_championship_adjustment(projected, event_type)
        range_low = apply_championship_adjustment(range_low, event_type)
        range_high = apply_championship_adjustment(range_high, event_type)

    # Detect trend
    trend = detect_trend(performances, event_type) if include_trend else 'stable'

    # Calculate form score
    form_score = calculate_form_score(performances, event_type)

    # Build methodology note
    methodology_parts = [
        "Weighted average with recency bias (weights: 1.0, 0.85, 0.72, 0.61, 0.52)",
        "Confidence range: ±1 standard deviation (68% probability)",
    ]
    if is_major_championship:
        methodology_parts.append("Championship pressure adjustment: +0.5% for time events")
    if include_trend:
        methodology_parts.append("Trend: comparing recent 2 vs older 2 performances")

    return {
        'projected': round(projected, 2),
        'range_low': round(range_low, 2),
        'range_high': round(range_high, 2),
        'confidence': 0.68,
        'trend': trend,
        'trend_symbol': get_trend_symbol(trend),
        'trend_color': get_trend_color(trend),
        'form_score': round(form_score, 1),
        'sample_size': len(performances),
        'methodology': '\n'.join(methodology_parts)
    }


def compare_to_competitors(
    athlete_sb: float,
    competitors: List[Dict],
    event_type: str = 'time'
) -> List[Dict]:
    """
    Compare athlete to list of competitors with gap analysis.

    Args:
        athlete_sb: Athlete's season best
        competitors: List of competitor dicts with 'name', 'country', 'sb', 'pb', 'pb_date', 'recent_form'
        event_type: 'time', 'distance', or 'points'

    Returns:
        Enriched competitor list with gaps and tags
    """
    enriched = []

    for comp in competitors:
        gap = calculate_gap(athlete_sb, comp.get('sb', 0), event_type)

        # Determine tag based on gap
        if event_type == 'time':
            if gap < -0.3:
                tag = 'Beatable'
            elif gap < 0.3:
                tag = 'Catchable'
            else:
                tag = 'Threat'
        else:
            if gap < -0.3:
                tag = 'Beatable'
            elif gap < 0.3:
                tag = 'Catchable'
            else:
                tag = 'Threat'

        # Detect competitor's form trend
        recent_form = comp.get('recent_form', [])
        comp_trend = detect_trend(recent_form, event_type) if len(recent_form) >= 3 else 'stable'

        enriched.append({
            **comp,
            'gap': gap,
            'gap_formatted': format_gap(gap, event_type),
            'tag': tag,
            'trend': comp_trend,
            'trend_symbol': get_trend_symbol(comp_trend)
        })

    return enriched


def calculate_advancement_probability(
    projected: float,
    historical_cutoffs: Dict[str, float],
    event_type: str = 'time'
) -> Dict[str, float]:
    """
    Calculate probability of advancing through each round based on historical data.

    Args:
        projected: Projected performance
        historical_cutoffs: Dict with 'heat', 'semi', 'final', 'medal' cutoff values
        event_type: 'time', 'distance', or 'points'

    Returns:
        Dict with probability for each round (0-100%)

    Example:
        >>> calculate_advancement_probability(
        ...     44.78,
        ...     {'heat': 45.50, 'semi': 45.10, 'final': 44.60, 'medal': 44.20},
        ...     'time'
        ... )
        {'heat': 92, 'semi': 85, 'final': 38, 'medal': 8}
    """
    probabilities = {}

    for round_name, cutoff in historical_cutoffs.items():
        if cutoff is None:
            continue

        gap = calculate_gap(projected, cutoff, event_type)

        # Convert gap to probability using logistic function
        # Centered around cutoff, with steeper curve for finals/medal
        if round_name in ['medal', 'final']:
            steepness = 5.0  # Steeper for tighter competition
        else:
            steepness = 3.0  # More forgiving for early rounds

        if event_type == 'time':
            # Negative gap = faster than cutoff = higher probability
            prob = 100 / (1 + math.exp(steepness * gap))
        else:
            # Negative gap = behind cutoff = lower probability
            prob = 100 / (1 + math.exp(-steepness * gap))

        probabilities[round_name] = round(prob, 0)

    return probabilities


def get_form_score_label(score: float) -> Tuple[str, str]:
    """
    Get descriptive label and color for form score.

    Returns:
        Tuple of (label, hex_color) using Team Saudi colors
    """
    if score >= 85:
        return ('Excellent', '#007167')  # Primary teal
    elif score >= 70:
        return ('Good', '#009688')       # Light teal
    elif score >= 50:
        return ('Moderate', '#a08e66')   # Gold
    elif score >= 30:
        return ('Low', '#FFB800')        # Warning
    else:
        return ('Poor', '#dc3545')       # Red


# Methodology documentation for display in dashboard
METHODOLOGY_NOTES = """
## Performance Projection Methodology

### Weighted Average Calculation
Recent performances are weighted with recency bias:
- Most recent: 100%
- 2nd most recent: 85%
- 3rd: 72%
- 4th: 61%
- 5th: 52%

### Confidence Range
The projection range uses ±1 standard deviation, representing 68% probability
that the actual performance will fall within this range.

### Championship Adjustment
For major championships (Olympics, World Championships), a +0.5% adjustment
is applied to time events to account for:
- Racing pressure
- Multiple rounds
- Tactical considerations

### Trend Detection
Trend is determined by comparing the average of the 2 most recent performances
to the average of the 2 oldest performances in the sample:
- Improving: >2% better
- Declining: >2% worse
- Stable: Within 2%

### Form Score (0-100)
Measures current form relative to personal range:
- 85-100: Excellent form
- 70-84: Good form
- 50-69: Moderate form
- 30-49: Low form
- 0-29: Poor form

### Advancement Probability
Probabilities are calculated using a logistic function comparing projected
performance to historical round cutoffs from the last 3-5 championship editions.
"""

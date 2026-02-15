"""
Form Engine - Performance projections and form scoring.

Ported from projection_engine.py with improvements.
Uses weighted recency scoring from Tilastopaja patterns.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Weights for recent performances (most recent first)
RECENCY_WEIGHTS = [1.0, 0.85, 0.72, 0.61, 0.52, 0.44, 0.37, 0.32]

# Championship pressure adjustment
CHAMPIONSHIP_ADJUSTMENT = 1.005


def calculate_form_score(
    recent_marks: List[float],
    pb: Optional[float] = None,
    lower_is_better: bool = True,
) -> float:
    """Calculate 0-100 form score based on recent performances.

    Score components:
    - Consistency (low variance = higher score)
    - Trend (improving = higher score)
    - Proximity to PB
    """
    if not recent_marks or len(recent_marks) < 2:
        return 50.0  # Default neutral

    marks = np.array(recent_marks[:8])  # Max 8 recent marks

    # 1. Consistency (40% of score) - coefficient of variation
    cv = np.std(marks) / np.mean(marks) if np.mean(marks) != 0 else 1.0
    consistency = max(0, min(40, 40 * (1 - cv * 10)))

    # 2. Trend (30% of score) - are marks improving?
    if len(marks) >= 3:
        x = np.arange(len(marks))
        coeffs = np.polyfit(x, marks, 1)
        slope = coeffs[0]

        if lower_is_better:
            trend = max(0, min(30, 15 + slope * -500))
        else:
            trend = max(0, min(30, 15 + slope * 500))
    else:
        trend = 15.0  # Neutral

    # 3. PB proximity (30% of score)
    if pb is not None and pb > 0:
        best_recent = min(marks) if lower_is_better else max(marks)
        if lower_is_better:
            ratio = pb / best_recent if best_recent > 0 else 0
        else:
            ratio = best_recent / pb if pb > 0 else 0
        pb_score = max(0, min(30, ratio * 30))
    else:
        pb_score = 15.0  # Neutral

    return round(consistency + trend + pb_score, 1)


def project_performance(
    recent_marks: List[float],
    weights: Optional[List[float]] = None,
    championship_adjust: bool = False,
    lower_is_better: bool = True,
) -> Dict:
    """Project performance based on weighted recent marks.

    Returns:
        Dict with projected_mark, confidence_low, confidence_high, trend
    """
    if not recent_marks:
        return {"projected_mark": None, "confidence_low": None, "confidence_high": None, "trend": "unknown"}

    marks = recent_marks[:len(RECENCY_WEIGHTS)]
    w = (weights or RECENCY_WEIGHTS)[:len(marks)]

    # Weighted average
    weighted_sum = sum(m * w for m, w in zip(marks, w))
    weight_total = sum(w[:len(marks)])
    projected = weighted_sum / weight_total

    # Championship adjustment
    if championship_adjust:
        if lower_is_better:
            projected *= (2 - CHAMPIONSHIP_ADJUSTMENT)  # Slightly faster
        else:
            projected *= CHAMPIONSHIP_ADJUSTMENT  # Slightly higher

    # Confidence interval (1 std dev)
    std = np.std(marks)
    confidence_low = projected - std
    confidence_high = projected + std

    # Trend detection
    trend = detect_trend(marks, lower_is_better)

    return {
        "projected_mark": round(projected, 3),
        "confidence_low": round(confidence_low, 3),
        "confidence_high": round(confidence_high, 3),
        "trend": trend,
    }


def detect_trend(marks: List[float], lower_is_better: bool = True) -> str:
    """Classify performance trend.

    Returns: 'improving', 'stable', or 'declining'
    """
    if len(marks) < 3:
        return "stable"

    x = np.arange(len(marks))
    coeffs = np.polyfit(x, marks, 1)
    slope = coeffs[0]
    mean = np.mean(marks)

    # Threshold: 2% change is significant
    pct_change = abs(slope * len(marks)) / mean if mean != 0 else 0

    if pct_change < 0.02:
        return "stable"

    if lower_is_better:
        return "improving" if slope < 0 else "declining"
    else:
        return "improving" if slope > 0 else "declining"


def calculate_advancement_probability(
    projected_mark: float,
    standard: float,
    confidence_std: float,
    lower_is_better: bool = True,
) -> float:
    """Estimate probability of meeting a standard.

    Uses normal distribution when scipy is available, with a
    heuristic fallback based on gap-to-standard ratio.
    """
    if confidence_std <= 0:
        return 1.0 if _mark_meets_standard(projected_mark, standard, lower_is_better) else 0.0

    if lower_is_better:
        z = (standard - projected_mark) / confidence_std
    else:
        z = (projected_mark - standard) / confidence_std

    try:
        from scipy.stats import norm
        return round(norm.cdf(z), 3)
    except ImportError:
        # Heuristic fallback: linear approximation of CDF
        return _heuristic_cdf(z)


def _heuristic_cdf(z: float) -> float:
    """Simple sigmoid-like approximation of normal CDF for fallback."""
    # Clamp to reasonable range
    z = max(-4.0, min(4.0, z))
    # Logistic approximation: 1 / (1 + exp(-1.7 * z))
    import math
    return round(1.0 / (1.0 + math.exp(-1.7 * z)), 3)


def calculate_advancement_probability_heuristic(
    athlete_pb: float,
    standard: float,
    lower_is_better: bool = True,
) -> float:
    """Coach-friendly probability estimate based on PB vs standard gap.

    Uses a simpler model: if athlete PB is within X% of the standard,
    estimate based on gap ratio. No scipy needed.

    Returns probability 0.0 to 1.0.
    """
    if standard == 0:
        return 0.0
    if _mark_meets_standard(athlete_pb, standard, lower_is_better):
        return 1.0

    # Calculate gap as percentage of standard
    if lower_is_better:
        gap_pct = (athlete_pb - standard) / standard  # positive = behind
    else:
        gap_pct = (standard - athlete_pb) / standard  # positive = behind

    # Map gap% to probability:
    # 0% gap = 100% (already achieved)
    # 1% gap = ~75%
    # 2% gap = ~50%
    # 5% gap = ~15%
    # 10%+ gap = ~2%
    if gap_pct <= 0:
        return 1.0
    elif gap_pct <= 0.01:
        return round(0.75 + 0.25 * (1 - gap_pct / 0.01), 3)
    elif gap_pct <= 0.02:
        return round(0.50 + 0.25 * (1 - (gap_pct - 0.01) / 0.01), 3)
    elif gap_pct <= 0.05:
        return round(0.15 + 0.35 * (1 - (gap_pct - 0.02) / 0.03), 3)
    elif gap_pct <= 0.10:
        return round(0.02 + 0.13 * (1 - (gap_pct - 0.05) / 0.05), 3)
    else:
        return 0.01


def calculate_gap(
    athlete_mark: float,
    standard: float,
    lower_is_better: bool = True,
) -> float:
    """Calculate gap between athlete mark and a standard.

    Positive = athlete is BETTER than standard
    Negative = athlete needs to improve
    """
    if lower_is_better:
        return standard - athlete_mark  # Positive if athlete is faster
    else:
        return athlete_mark - standard  # Positive if athlete jumped/threw more


def _mark_meets_standard(mark: float, standard: float, lower_is_better: bool) -> bool:
    if lower_is_better:
        return mark <= standard
    return mark >= standard

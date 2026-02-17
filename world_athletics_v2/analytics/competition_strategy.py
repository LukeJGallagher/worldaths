"""
Competition Strategy engine.

Core business logic for the Qualification Pathway Planner:
- Points estimation per competition
- Top-5 displacement calculation
- Competition ranking / recommendation
- Scenario modelling (hypothetical mark → estimated points)
- Qualification status vs championship standards
"""

from __future__ import annotations

import datetime
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────

# Place points (1st place) by WA competition category code
PLACE_POINTS: Dict[str, int] = {
    "OW": 170,  # Olympic / World Championships
    "DF": 140,  # Diamond League Final
    "GW": 100,  # Gold (World) / Diamond League
    "GL": 80,   # Gold Label (Road)
    "A": 60,    # Continental Tour Silver / Area Champs
    "B": 40,    # Continental Tour Bronze / National Champs
    "C": 20,    # National permit
    "D": 10,    # Minor national
    "E": 5,     # Lower-tier
    "F": 0,     # Basic / local
}

# Category ordering for filtering (highest first)
CATEGORY_ORDER = ["OW", "DF", "GW", "GL", "A", "B", "C", "D", "E", "F"]

# Qualification deadlines for target championships
QUALIFICATION_DEADLINES = {
    "World Champs 2025 (Tokyo)": {
        "deadline": "2025-09-01",
        "event_date": "2025-09-13",
    },
    "Asian Games 2026 (Nagoya)": {
        "deadline": "2026-08-01",
        "event_date": "2026-09-19",
    },
    "Olympics 2028 (Los Angeles)": {
        "deadline": "2028-06-01",
        "event_date": "2028-07-14",
    },
}

# Category colors for charts
CATEGORY_COLORS: Dict[str, str] = {
    "OW": "#FFD700",
    "DF": "#C0C0C0",
    "GW": "#a08e66",
    "GL": "#a08e66",
    "A": "#007167",
    "B": "#009688",
    "C": "#78909C",
    "D": "#90A4AE",
    "E": "#B0BEC5",
    "F": "#CFD8DC",
}

CATEGORY_NAMES: Dict[str, str] = {
    "OW": "Olympic/World Champs",
    "DF": "Diamond League Final",
    "GW": "Gold (World)",
    "GL": "Gold Label",
    "A": "Category A",
    "B": "Category B",
    "C": "Category C",
    "D": "Category D",
    "E": "Category E",
    "F": "Category F",
}


# ── Points estimation ─────────────────────────────────────────────────

def estimate_total_points(
    result_score: float,
    category: str,
    place: int = 1,
) -> float:
    """Estimate total WA ranking points for a performance.

    Total = result_score + place_points for the competition category.
    Place points are scaled: 1st gets full value, 8th gets ~40-45%.
    """
    base_pts = PLACE_POINTS.get(category, 0)
    # Scale by finishing place (approximate WA scale)
    if place <= 0:
        place = 1
    place_scale = {1: 1.0, 2: 0.85, 3: 0.75, 4: 0.65, 5: 0.55, 6: 0.50, 7: 0.45, 8: 0.42}
    scale = place_scale.get(place, max(0.3, 1.0 - place * 0.08))
    place_pts = base_pts * scale
    return float(result_score) + place_pts


def calculate_points_gain(
    current_top5: List[float],
    new_score: float,
) -> Dict:
    """Calculate how a new score would affect the top-5 average.

    WA rankings count only the top 5 scoring performances.
    Returns whether the new score displaces the weakest entry.
    """
    top5 = sorted(current_top5, reverse=True)[:5]
    current_avg = np.mean(top5) if top5 else 0.0
    weakest = min(top5) if top5 else 0.0

    if len(top5) < 5:
        # Still building up the top 5
        new_top5 = sorted(top5 + [new_score], reverse=True)[:5]
        new_avg = np.mean(new_top5)
        return {
            "replaces": True,
            "displaced_score": None,
            "new_avg": round(new_avg, 1),
            "current_avg": round(current_avg, 1),
            "improvement": round(new_avg - current_avg, 1),
            "new_top5": new_top5,
        }

    if new_score > weakest:
        new_top5 = sorted(top5[:-1] + [new_score], reverse=True)[:5]
        new_avg = np.mean(new_top5)
        return {
            "replaces": True,
            "displaced_score": round(weakest, 1),
            "new_avg": round(new_avg, 1),
            "current_avg": round(current_avg, 1),
            "improvement": round(new_avg - current_avg, 1),
            "new_top5": new_top5,
        }

    return {
        "replaces": False,
        "displaced_score": None,
        "new_avg": round(current_avg, 1),
        "current_avg": round(current_avg, 1),
        "improvement": 0.0,
        "new_top5": top5,
    }


# ── Competition ranking ───────────────────────────────────────────────

def rank_competitions(
    calendar_df: pd.DataFrame,
    athlete_result_score: float,
    deadline: Optional[str] = None,
    area_preference: Optional[str] = None,
    min_category: str = "F",
    current_top5: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Rank future competitions by estimated points gain.

    Filters calendar to Track & Field competitions, scores each by
    estimated total points and improvement potential.
    """
    if calendar_df.empty:
        return pd.DataFrame()

    df = calendar_df.copy()

    # Filter to Track and Field
    if "disciplines" in df.columns:
        df = df[df["disciplines"].str.contains("Track and Field", case=False, na=False)]

    # Filter future only
    today = datetime.date.today().isoformat()
    if "start_date" in df.columns:
        df = df[df["start_date"] >= today]

    # Filter by deadline
    if deadline and "start_date" in df.columns:
        df = df[df["start_date"] <= deadline]

    # Filter by minimum category
    if min_category != "F" and "ranking_category" in df.columns:
        min_idx = CATEGORY_ORDER.index(min_category) if min_category in CATEGORY_ORDER else len(CATEGORY_ORDER)
        allowed = set(CATEGORY_ORDER[:min_idx + 1])
        df = df[df["ranking_category"].isin(allowed)]

    if df.empty:
        return pd.DataFrame()

    # Calculate estimated points
    df["place_pts"] = df["ranking_category"].map(PLACE_POINTS).fillna(0).astype(int)
    df["est_total"] = athlete_result_score + df["place_pts"]

    # Calculate gain over current weakest top-5
    weakest_top5 = min(current_top5) if current_top5 and len(current_top5) >= 5 else 0
    df["est_gain"] = (df["est_total"] - weakest_top5).clip(lower=0)

    # Days until competition
    if "start_date" in df.columns:
        df["days_until"] = (
            pd.to_datetime(df["start_date"], errors="coerce") - pd.Timestamp.now()
        ).dt.days.clip(lower=0)

    # Priority score (higher = better)
    cat_weight = df["ranking_category"].map(
        {c: (len(CATEGORY_ORDER) - i) * 10 for i, c in enumerate(CATEGORY_ORDER)}
    ).fillna(0)

    region_bonus = 0
    if area_preference and "area" in df.columns:
        region_bonus = (df["area"] == area_preference).astype(int) * 15

    df["priority"] = cat_weight + region_bonus + df["est_gain"] * 0.5

    # Sort by priority descending
    df = df.sort_values("priority", ascending=False)

    return df


# ── Scenario modelling ────────────────────────────────────────────────

def interpolate_result_score(
    marks: List[float],
    scores: List[float],
    hypothetical_mark: float,
    lower_is_better: bool = True,
) -> Optional[float]:
    """Estimate WA result_score for a hypothetical mark.

    Uses linear interpolation from the athlete's own mark/score pairs.
    For time events (lower_is_better=True), faster mark = higher score.
    """
    if len(marks) < 2 or len(scores) < 2:
        return scores[0] if scores else None

    # Sort by mark
    paired = sorted(zip(marks, scores), key=lambda x: x[0])
    m_arr = np.array([p[0] for p in paired])
    s_arr = np.array([p[1] for p in paired])

    # Linear interpolation / extrapolation
    try:
        estimated = float(np.interp(hypothetical_mark, m_arr, s_arr))
        return max(0, round(estimated, 0))
    except Exception:
        return scores[0] if scores else None


# ── Qualification status ──────────────────────────────────────────────

def build_qualification_status(
    athlete_pb_mark: Optional[float],
    athlete_sb_mark: Optional[float],
    event: str,
    event_type: str = "time",
) -> List[Dict]:
    """Compare athlete's PB/SB against championship standards.

    Returns a list of dicts, one per championship, with status info.
    """
    from components.report_components import get_championship_targets

    targets = get_championship_targets(event)
    if not targets:
        return []

    rows = []
    for champ_name, data in targets.items():
        marks = data.get("marks", {})
        deadline = QUALIFICATION_DEADLINES.get(champ_name, {})

        for level, standard in marks.items():
            if standard is None:
                continue

            best_mark = athlete_pb_mark
            gap = None
            status = "unknown"

            if best_mark is not None and standard is not None:
                if event_type == "time":
                    gap = best_mark - standard  # negative = already under
                    if gap <= 0:
                        status = "qualified"
                    elif gap < standard * 0.02:
                        status = "close"
                    else:
                        status = "gap"
                else:
                    gap = standard - best_mark  # negative = already over
                    if gap <= 0:
                        status = "qualified"
                    elif gap < standard * 0.02:
                        status = "close"
                    else:
                        status = "gap"

            rows.append({
                "Championship": champ_name,
                "Level": level.title(),
                "Standard": standard,
                "Athlete PB": best_mark,
                "Gap": abs(gap) if gap is not None else None,
                "Status": status,
                "Deadline": deadline.get("deadline", ""),
            })

    return rows

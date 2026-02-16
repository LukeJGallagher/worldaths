"""
What It Takes to Win - Championship standards analysis.

Ported from what_it_takes_to_win.py with v2 data model.
Calculates medal, finals, semi, and heat standards from historical data.
"""

from typing import Dict, List, Optional
import re
import pandas as pd
import numpy as np

from data.event_utils import normalize_event_for_match, get_event_type


# Round standards (typical marks needed to advance)
ROUND_STANDARDS = {
    "Medal": {"percentile": 3, "description": "Top 3 in final"},
    "Final": {"percentile": 8, "description": "Top 8 - make the final"},
    "Semi-Final": {"percentile": 16, "description": "Top 16 - advance from heats"},
    "Heat": {"percentile": 32, "description": "Top 32 - advance from round 1"},
}


def calculate_standards_from_toplist(
    toplist_df: pd.DataFrame,
    mark_col: str = "mark",
    lower_is_better: bool = True,
) -> Dict[str, Optional[float]]:
    """Calculate round standards from a season toplist.

    Args:
        toplist_df: DataFrame with marks (sorted by rank)
        mark_col: Column containing the mark values
        lower_is_better: True for time events, False for field events

    Returns:
        Dict with Medal, Final, Semi-Final, Heat marks
    """
    # Try to convert marks to numeric
    marks = pd.to_numeric(toplist_df[mark_col], errors="coerce").dropna()

    if len(marks) == 0:
        return {k: None for k in ROUND_STANDARDS}

    if lower_is_better:
        marks = marks.sort_values(ascending=True)
    else:
        marks = marks.sort_values(ascending=False)

    result = {}
    for standard, info in ROUND_STANDARDS.items():
        idx = min(info["percentile"] - 1, len(marks) - 1)
        result[standard] = round(float(marks.iloc[idx]), 3)

    return result


def calculate_historical_standards(
    results_df: pd.DataFrame,
    event: str,
    gender: str,
    championships: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate historical medal/finals standards from championship results.

    Returns DataFrame with columns: championship, year, gold, silver, bronze,
    final_8th, semi_mark, heat_mark
    """
    if len(results_df) == 0:
        return pd.DataFrame()

    event_norm = normalize_event_for_match(event)
    event_type = get_event_type(event)
    lower_is_better = event_type == "time"

    # Filter to matching event
    mask = results_df.apply(
        lambda row: normalize_event_for_match(str(row.get("event", ""))) == event_norm,
        axis=1
    )
    filtered = results_df[mask].copy()

    if len(filtered) == 0:
        return pd.DataFrame()

    # Filter by gender if column exists
    if "gender" in filtered.columns:
        filtered = filtered[filtered["gender"].str.upper() == gender.upper()]

    # Try to get numeric marks
    if "mark" in filtered.columns:
        filtered["mark_numeric"] = pd.to_numeric(filtered["mark"], errors="coerce")
    elif "result_numeric" in filtered.columns:
        filtered["mark_numeric"] = filtered["result_numeric"]
    else:
        return pd.DataFrame()

    # Filter outliers (>5x or <0.2x median)
    median = filtered["mark_numeric"].median()
    if median > 0:
        filtered = filtered[
            (filtered["mark_numeric"] >= median * 0.2) &
            (filtered["mark_numeric"] <= median * 5.0)
        ]

    return filtered


def get_finals_summary_by_place(
    results_df: pd.DataFrame,
    max_place: int = 8,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Aggregate championship final results by finishing position.

    Args:
        results_df: DataFrame with 'pos' and 'result_numeric' columns
        max_place: Maximum place to include (default 8 for finals)
        lower_is_better: True for time events

    Returns:
        DataFrame with columns: place, avg_mark, fastest, slowest, n_results
    """
    if len(results_df) == 0:
        return pd.DataFrame()

    df = results_df.copy()

    # Parse position to numeric (filter to plain numbers = finals only)
    df["pos_str"] = df["pos"].astype(str).str.strip()
    df["pos_numeric"] = pd.to_numeric(df["pos_str"], errors="coerce")
    df = df.dropna(subset=["pos_numeric"])
    df["pos_numeric"] = df["pos_numeric"].astype(int)
    df = df[(df["pos_numeric"] >= 1) & (df["pos_numeric"] <= max_place)]

    marks = pd.to_numeric(df["result_numeric"], errors="coerce").dropna()
    if len(marks) == 0:
        return pd.DataFrame()

    df["result_numeric"] = pd.to_numeric(df["result_numeric"], errors="coerce")
    df = df.dropna(subset=["result_numeric"])

    summary = df.groupby("pos_numeric").agg(
        avg_mark=("result_numeric", "mean"),
        fastest=("result_numeric", "min") if lower_is_better else ("result_numeric", "max"),
        slowest=("result_numeric", "max") if lower_is_better else ("result_numeric", "min"),
        n_results=("result_numeric", "count"),
    ).reset_index()

    summary = summary.rename(columns={"pos_numeric": "place"})
    summary["avg_mark"] = summary["avg_mark"].round(3)
    summary["fastest"] = summary["fastest"].round(3)
    summary["slowest"] = summary["slowest"].round(3)
    summary = summary.sort_values("place")

    return summary


def get_standards_by_year(
    results_df: pd.DataFrame,
    max_place: int = 6,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Get the mark per year per finishing position for trend charts.

    Args:
        results_df: DataFrame with 'pos', 'result_numeric', and 'year' or 'date' columns
        max_place: Maximum place to include
        lower_is_better: True for time events

    Returns:
        Long-format DataFrame with columns: year, place, mark
    """
    if len(results_df) == 0:
        return pd.DataFrame()

    df = results_df.copy()

    # Parse position
    df["pos_str"] = df["pos"].astype(str).str.strip()
    df["pos_numeric"] = pd.to_numeric(df["pos_str"], errors="coerce")
    df = df.dropna(subset=["pos_numeric"])
    df["pos_numeric"] = df["pos_numeric"].astype(int)
    df = df[(df["pos_numeric"] >= 1) & (df["pos_numeric"] <= max_place)]

    # Get year
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df["result_numeric"] = pd.to_numeric(df["result_numeric"], errors="coerce")
    df = df.dropna(subset=["result_numeric"])

    if len(df) == 0:
        return pd.DataFrame()

    # Best mark per year per position
    if lower_is_better:
        trend = df.groupby(["year", "pos_numeric"]).agg(
            mark=("result_numeric", "min")
        ).reset_index()
    else:
        trend = df.groupby(["year", "pos_numeric"]).agg(
            mark=("result_numeric", "max")
        ).reset_index()

    trend = trend.rename(columns={"pos_numeric": "place"})
    trend["mark"] = trend["mark"].round(3)
    trend = trend.sort_values(["year", "place"])

    return trend


def extract_round(pos_str) -> str:
    """Classify the round from a position string.

    Returns one of: 'Final', 'Semi Finals', 'Quarter Finals', 'Heats',
    'Preliminary Round', or 'Unknown'.

    Examples:
        '1'    -> 'Final'
        '2f1'  -> 'Final'
        '1ce1' -> 'Final' (combined event)
        '3sf2' -> 'Semi Finals'
        '2qf1' -> 'Quarter Finals'
        '1h3'  -> 'Heats'
        '5r1'  -> 'Preliminary Round'
    """
    if pd.isna(pos_str):
        return "Unknown"
    p = str(pos_str).strip().lower()
    if any(x in p for x in ("dns", "dnf", "dq", "nm", "nr")):
        return "Unknown"
    # Plain number = final
    if re.match(r"^\d+$", p):
        return "Final"
    # 'f' or 'ce' suffix = final / combined event
    if re.match(r"^\d+(f|ce)\d*$", p):
        return "Final"
    # 'sf' suffix = semi-final
    if re.match(r"^\d+sf\d*$", p):
        return "Semi Finals"
    # 'qf' suffix = quarter-final
    if re.match(r"^\d+qf\d*$", p):
        return "Quarter Finals"
    # 'q' suffix (not qf) = qualifier / preliminary
    if re.match(r"^\d+q\d*$", p):
        return "Preliminary Round"
    # 'h' suffix = heat
    if re.match(r"^\d+h\d*$", p):
        return "Heats"
    # 'r' suffix = round 1 / preliminary
    if re.match(r"^\d+r\d*$", p):
        return "Preliminary Round"
    return "Unknown"


# Canonical ordering from earliest to latest round
ROUND_ORDER = [
    "Preliminary Round",
    "Heats",
    "Quarter Finals",
    "Semi Finals",
    "Final",
]


def get_round_summary(
    results_df: pd.DataFrame,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Aggregate championship results by round stage.

    Args:
        results_df: DataFrame with 'pos' and 'result_numeric' columns
                    (should include ALL rounds, not just finals)
        lower_is_better: True for time events

    Returns:
        DataFrame with columns: round, qualifier_type, avg_mark, fastest, slowest, n_results
        Sorted in natural round order (Preliminary -> Heats -> ... -> Final).
    """
    if len(results_df) == 0:
        return pd.DataFrame()

    df = results_df.copy()
    df["round"] = df["pos"].apply(extract_round)
    df = df[df["round"] != "Unknown"]
    df["result_numeric"] = pd.to_numeric(df["result_numeric"], errors="coerce")
    df = df.dropna(subset=["result_numeric"])

    if len(df) == 0:
        return pd.DataFrame()

    summary = df.groupby("round").agg(
        avg_mark=("result_numeric", "mean"),
        fastest=("result_numeric", "min") if lower_is_better else ("result_numeric", "max"),
        slowest=("result_numeric", "max") if lower_is_better else ("result_numeric", "min"),
        n_results=("result_numeric", "count"),
    ).reset_index()

    summary["avg_mark"] = summary["avg_mark"].round(3)
    summary["fastest"] = summary["fastest"].round(3)
    summary["slowest"] = summary["slowest"].round(3)

    # Qualifier type heuristic
    def _qualifier_type(rnd: str) -> str:
        if rnd == "Final":
            return "Final"
        return "Qualifying"

    summary["qualifier_type"] = summary["round"].apply(_qualifier_type)

    # Sort by canonical round order
    order_map = {r: i for i, r in enumerate(ROUND_ORDER)}
    summary["_order"] = summary["round"].map(order_map).fillna(99)
    summary = summary.sort_values("_order").drop(columns=["_order"])

    return summary


def get_round_trends_by_year(
    results_df: pd.DataFrame,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Get avg/fastest/slowest marks per round per year for trend charts.

    Args:
        results_df: DataFrame with 'pos', 'result_numeric', and 'year'/'date'
        lower_is_better: True for time events

    Returns:
        Long-format DataFrame: year, round, avg_mark, fastest, slowest
    """
    if len(results_df) == 0:
        return pd.DataFrame()

    df = results_df.copy()
    df["round"] = df["pos"].apply(extract_round)
    df = df[df["round"] != "Unknown"]

    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df["result_numeric"] = pd.to_numeric(df["result_numeric"], errors="coerce")
    df = df.dropna(subset=["result_numeric"])

    if len(df) == 0:
        return pd.DataFrame()

    trend = df.groupby(["year", "round"]).agg(
        avg_mark=("result_numeric", "mean"),
        fastest=("result_numeric", "min") if lower_is_better else ("result_numeric", "max"),
        slowest=("result_numeric", "max") if lower_is_better else ("result_numeric", "min"),
    ).reset_index()

    trend["avg_mark"] = trend["avg_mark"].round(3)
    trend["fastest"] = trend["fastest"].round(3)
    trend["slowest"] = trend["slowest"].round(3)
    trend = trend.sort_values(["year", "round"])

    return trend


def gap_analysis(
    athlete_pb: float,
    standards: Dict[str, Optional[float]],
    lower_is_better: bool = True,
) -> Dict[str, Dict]:
    """Calculate gap between athlete PB and each standard.

    Returns dict with gap value and status for each standard level.
    """
    result = {}
    for level, mark in standards.items():
        if mark is None:
            result[level] = {"gap": None, "status": "unknown", "mark": None}
            continue

        if lower_is_better:
            gap = mark - athlete_pb  # Positive = athlete is faster
        else:
            gap = athlete_pb - mark  # Positive = athlete is better

        if gap > 0:
            status = "achieved"
        elif abs(gap) < abs(mark * 0.02):  # Within 2%
            status = "close"
        else:
            status = "needs_work"

        result[level] = {
            "gap": round(gap, 3),
            "status": status,
            "mark": mark,
        }

    return result

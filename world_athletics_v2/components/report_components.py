"""
Shared formatting utilities for Pre-Competition reports.
Used by both PDF (ReportLab) and HTML report generators.
"""

from typing import Dict, List, Optional

import pandas as pd

# Team Saudi brand colors (duplicated from theme.py for standalone use in ReportLab)
TEAL_PRIMARY = "#235032"
TEAL_DARK = "#1a3d25"
GOLD_ACCENT = "#a08e66"
GRAY_BLUE = "#78909C"
STATUS_DANGER = "#dc3545"
STATUS_NEUTRAL = "#6c757d"


# ── Championship Entry Standards ─────────────────────────────────────
# Official WA entry standards for target championships.
# Times in seconds, distances in metres.

# Asian Games 2026 (Nagoya) - by NOC nomination, no published ES.
# Target marks based on recent Asian Games medal lines (Hangzhou 2023).
ASIAN_GAMES_2026_TARGETS = {
    # Event: {"gold": historical gold mark, "medal": bronze-level mark}
    "100m":            {"gold": 9.97,   "medal": 10.07,  "final": 10.15},
    "200m":            {"gold": 20.01,  "medal": 20.32,  "final": 20.55},
    "400m":            {"gold": 44.53,  "medal": 45.12,  "final": 45.80},
    "800m":            {"gold": 104.50, "medal": 105.90, "final": 107.00},
    "1500m":           {"gold": 216.00, "medal": 219.00, "final": 222.00},
    "5000m":           {"gold": 793.00, "medal": 800.00, "final": 810.00},
    "10000m":          {"gold": 1650.0, "medal": 1665.0, "final": 1680.0},
    "110m Hurdles":    {"gold": 13.18,  "medal": 13.42,  "final": 13.55},
    "400m Hurdles":    {"gold": 47.80,  "medal": 48.90,  "final": 49.50},
    "3000m Steeplechase": {"gold": 505.0, "medal": 510.0, "final": 520.0},
    "High Jump":       {"gold": 2.36,   "medal": 2.30,   "final": 2.26},
    "Pole Vault":      {"gold": 5.90,   "medal": 5.75,   "final": 5.60},
    "Long Jump":       {"gold": 8.48,   "medal": 8.15,   "final": 7.95},
    "Triple Jump":     {"gold": 17.19,  "medal": 16.80,  "final": 16.50},
    "Shot Put":        {"gold": 21.40,  "medal": 20.50,  "final": 19.80},
    "Discus Throw":    {"gold": 67.00,  "medal": 63.50,  "final": 60.00},
    "Javelin Throw":   {"gold": 87.80,  "medal": 83.00,  "final": 80.00},
    "Hammer Throw":    {"gold": 78.00,  "medal": 74.00,  "final": 70.00},
    "Decathlon":       {"gold": 8400,   "medal": 8100,   "final": 7800},
}

# LA 2028 Olympics - Entry Standards (estimated, based on WC Tokyo 2025 + Paris 2024).
# These will be updated when official standards are published by WA.
LA_2028_ENTRY_STANDARDS = {
    "100m":            {"entry": 10.00, "medal": 9.85,  "final": 9.97},
    "200m":            {"entry": 20.16, "medal": 19.70, "final": 20.05},
    "400m":            {"entry": 44.90, "medal": 43.80, "final": 44.60},
    "800m":            {"entry": 104.70, "medal": 103.50, "final": 104.20},
    "1500m":           {"entry": 213.50, "medal": 210.00, "final": 212.50},
    "5000m":           {"entry": 785.00, "medal": 775.00, "final": 780.00},
    "10000m":          {"entry": 1630.0, "medal": 1610.0, "final": 1625.0},
    "110m Hurdles":    {"entry": 13.27, "medal": 12.95, "final": 13.15},
    "400m Hurdles":    {"entry": 48.60, "medal": 47.50, "final": 48.20},
    "3000m Steeplechase": {"entry": 495.0, "medal": 488.0, "final": 493.0},
    "High Jump":       {"entry": 2.33,  "medal": 2.36,  "final": 2.31},
    "Pole Vault":      {"entry": 5.82,  "medal": 5.95,  "final": 5.80},
    "Long Jump":       {"entry": 8.25,  "medal": 8.40,  "final": 8.15},
    "Triple Jump":     {"entry": 17.22, "medal": 17.50, "final": 17.00},
    "Shot Put":        {"entry": 21.30, "medal": 22.00, "final": 21.00},
    "Discus Throw":    {"entry": 67.00, "medal": 69.00, "final": 66.00},
    "Javelin Throw":   {"entry": 85.00, "medal": 88.00, "final": 83.00},
    "Hammer Throw":    {"entry": 78.00, "medal": 80.00, "final": 76.00},
    "Decathlon":       {"entry": 8350,  "medal": 8700,  "final": 8300},
}

# WC Tokyo 2025 Entry Standards (official WA published)
WC_TOKYO_2025_ENTRY_STANDARDS = {
    "100m":            {"entry": 10.00, "medal": 9.85,  "final": 9.95},
    "200m":            {"entry": 20.16, "medal": 19.65, "final": 20.00},
    "400m":            {"entry": 44.90, "medal": 43.70, "final": 44.50},
    "800m":            {"entry": 104.70, "medal": 103.50, "final": 104.20},
    "1500m":           {"entry": 213.50, "medal": 210.00, "final": 212.00},
    "5000m":           {"entry": 785.00, "medal": 775.00, "final": 780.00},
    "10000m":          {"entry": 1630.0, "medal": 1610.0, "final": 1625.0},
    "110m Hurdles":    {"entry": 13.27, "medal": 13.00, "final": 13.20},
    "400m Hurdles":    {"entry": 48.60, "medal": 47.50, "final": 48.20},
    "3000m Steeplechase": {"entry": 495.0, "medal": 488.0, "final": 493.0},
    "High Jump":       {"entry": 2.33,  "medal": 2.36,  "final": 2.31},
    "Pole Vault":      {"entry": 5.82,  "medal": 5.95,  "final": 5.80},
    "Long Jump":       {"entry": 8.25,  "medal": 8.40,  "final": 8.15},
    "Triple Jump":     {"entry": 17.22, "medal": 17.50, "final": 17.00},
    "Shot Put":        {"entry": 21.30, "medal": 22.00, "final": 21.00},
    "Discus Throw":    {"entry": 67.00, "medal": 69.00, "final": 66.00},
    "Javelin Throw":   {"entry": 85.00, "medal": 88.00, "final": 83.00},
    "Hammer Throw":    {"entry": 78.00, "medal": 80.00, "final": 76.00},
    "Decathlon":       {"entry": 8350,  "medal": 8700,  "final": 8300},
}


# Qualification deadlines for target championships
QUALIFICATION_DEADLINES = {
    "World Champs 2025 (Tokyo)": {"deadline": "2025-09-01", "event_date": "2025-09-13"},
    "Asian Games 2026 (Nagoya)": {"deadline": "2026-08-01", "event_date": "2026-09-19"},
    "Olympics 2028 (Los Angeles)": {"deadline": "2028-06-01", "event_date": "2028-07-14"},
}


def get_championship_targets(event: str) -> Dict[str, Dict]:
    """Get championship-specific targets for an event.

    Returns dict keyed by championship name, each containing target marks
    and descriptions.
    """
    # Normalize event name for lookup
    event_lookup = _normalize_event_for_lookup(event)

    targets = {}

    ag = ASIAN_GAMES_2026_TARGETS.get(event_lookup)
    if ag:
        targets["Asian Games 2026 (Nagoya)"] = {
            "marks": ag,
            "note": "Based on Hangzhou 2023 medal lines",
        }

    wc = WC_TOKYO_2025_ENTRY_STANDARDS.get(event_lookup)
    if wc:
        targets["World Champs 2025 (Tokyo)"] = {
            "marks": wc,
            "note": "Official WA entry standards",
        }

    la = LA_2028_ENTRY_STANDARDS.get(event_lookup)
    if la:
        targets["Olympics 2028 (Los Angeles)"] = {
            "marks": la,
            "note": "Estimated from recent standards (TBC)",
        }

    return targets


def _normalize_event_for_lookup(event: str) -> str:
    """Normalize event display name to match the standards dict keys.

    Handles gender prefixes (Men's/Women's), multiple display formats,
    and API discipline names.
    """
    # Strip gender prefix first
    for prefix in ("Men's ", "Women's ", "men's ", "women's "):
        if event.startswith(prefix):
            event = event[len(prefix):]
            break

    # Common display -> lookup mappings
    mappings = {
        "100m": "100m", "200m": "200m", "400m": "400m",
        "800m": "800m", "1500m": "1500m", "5000m": "5000m",
        "10000m": "10000m", "10,000m": "10000m",
        # Hurdles (multiple display formats)
        "110mh": "110m Hurdles", "110m hurdles": "110m Hurdles",
        "110m h": "110m Hurdles", "110-metres-hurdles": "110m Hurdles",
        "100mh": "100m Hurdles", "100m hurdles": "100m Hurdles",
        "100m h": "100m Hurdles", "100-metres-hurdles": "100m Hurdles",
        "400mh": "400m Hurdles", "400m hurdles": "400m Hurdles",
        "400m h": "400m Hurdles", "400-metres-hurdles": "400m Hurdles",
        # Steeplechase
        "3000msc": "3000m Steeplechase", "3000m steeplechase": "3000m Steeplechase",
        "3000m sc": "3000m Steeplechase", "3000-metres-steeplechase": "3000m Steeplechase",
        # Jumps
        "high jump": "High Jump", "hj": "High Jump", "high-jump": "High Jump",
        "pole vault": "Pole Vault", "pv": "Pole Vault", "pole-vault": "Pole Vault",
        "long jump": "Long Jump", "lj": "Long Jump", "long-jump": "Long Jump",
        "triple jump": "Triple Jump", "tj": "Triple Jump", "triple-jump": "Triple Jump",
        # Throws
        "shot put": "Shot Put", "sp": "Shot Put", "shot-put": "Shot Put",
        "discus throw": "Discus Throw", "discus": "Discus Throw", "discus-throw": "Discus Throw",
        "javelin throw": "Javelin Throw", "javelin": "Javelin Throw", "javelin-throw": "Javelin Throw",
        "hammer throw": "Hammer Throw", "hammer": "Hammer Throw", "hammer-throw": "Hammer Throw",
        # Combined
        "decathlon": "Decathlon",
        # WA API format "Overall Ranking" → skip
        "overall ranking": "",
    }
    # Try exact match first
    if event in ASIAN_GAMES_2026_TARGETS:
        return event
    # Try lowercase lookup
    return mappings.get(event.lower(), event)


def format_mark_display(mark: float, event_type: str) -> str:
    """Format a mark for display. Time events show 's', field events show 'm'."""
    if mark is None:
        return "-"
    if event_type == "time":
        if mark >= 60:
            minutes = int(mark // 60)
            seconds = mark - (minutes * 60)
            return f"{minutes}:{seconds:05.2f}"
        return f"{mark:.2f}s"
    elif event_type == "points":
        return f"{int(mark)} pts"
    else:
        return f"{mark:.2f}m"


def format_gap_display(gap: float, event_type: str) -> str:
    """Format gap display for coaches. Shows how far to go, always positive."""
    if gap is None:
        return "-"
    if gap > 0:
        return f"Ahead by {abs(gap):.2f}"
    elif gap < 0:
        return f"{abs(gap):.2f} to go"
    return "On target"


def get_status_color(status: str) -> str:
    """Return hex color based on gap status."""
    colors = {
        "achieved": TEAL_PRIMARY,
        "close": GOLD_ACCENT,
        "needs_work": GRAY_BLUE,  # Neutral gray-blue, not red (coaching context)
        "unknown": STATUS_NEUTRAL,
    }
    return colors.get(status, STATUS_NEUTRAL)


def get_status_label(status: str) -> str:
    """Return a coach-friendly status label."""
    labels = {
        "achieved": "Achieved",
        "close": "Within Range",
        "needs_work": "Needs Work",
        "unknown": "-",
    }
    return labels.get(status, status.replace("_", " ").title())


def get_trend_arrow(trend: str) -> str:
    """Return arrow character for trend direction."""
    arrows = {"improving": "\u2191", "stable": "\u2192", "declining": "\u2193", "unknown": "?"}
    return arrows.get(trend, "?")


def get_trend_color(trend: str) -> str:
    """Return color for trend display."""
    colors = {
        "improving": TEAL_PRIMARY,
        "stable": GOLD_ACCENT,
        "declining": GRAY_BLUE,  # Neutral gray-blue, not red
    }
    return colors.get(trend, STATUS_NEUTRAL)


def build_standards_rows(
    standards: Dict,
    athlete_mark: float,
    event_type: str,
    lower_is_better: bool,
) -> list:
    """Build formatted rows for standards table with gaps and status."""
    rows = []
    for level, info in standards.items():
        if info.get("mark") is None:
            continue
        mark = info["mark"]
        gap = info.get("gap")
        status = info.get("status", "unknown")
        rows.append(
            {
                "Level": level,
                "Standard": format_mark_display(mark, event_type),
                "Athlete": format_mark_display(athlete_mark, event_type),
                "Gap": format_gap_display(gap, event_type),
                "Status": get_status_label(status),
                "Color": get_status_color(status),
            }
        )
    return rows


def build_championship_target_rows(
    targets: Dict[str, Dict],
    athlete_mark: float,
    event_type: str,
    lower_is_better: bool,
) -> List[Dict]:
    """Build formatted rows for championship-specific targets.

    Args:
        targets: From get_championship_targets()
        athlete_mark: Athlete PB or SB in seconds/metres
        event_type: 'time', 'distance', or 'points'
        lower_is_better: True for time events

    Returns:
        List of dicts with Championship, Target, Level, Athlete, Gap, Status, Color
    """
    rows = []
    if athlete_mark is None:
        return rows

    for champ_name, champ_data in targets.items():
        marks = champ_data.get("marks", {})
        note = champ_data.get("note", "")

        for level, target_mark in marks.items():
            if target_mark is None:
                continue

            # Calculate gap
            if lower_is_better:
                gap = target_mark - athlete_mark  # Positive = athlete is faster
            else:
                gap = athlete_mark - target_mark  # Positive = athlete is better

            if gap > 0:
                status = "achieved"
            elif abs(gap) < abs(target_mark * 0.02):
                status = "close"
            else:
                status = "needs_work"

            rows.append({
                "Championship": champ_name,
                "Level": level.replace("_", " ").title(),
                "Target": format_mark_display(target_mark, event_type),
                "Athlete": format_mark_display(athlete_mark, event_type),
                "Gap": format_gap_display(gap, event_type),
                "Status": get_status_label(status),
                "Color": get_status_color(status),
                "Note": note,
            })

    return rows


def build_rivals_rows(rivals_df: pd.DataFrame) -> list:
    """Build formatted rows for rivals table."""
    if rivals_df is None or rivals_df.empty:
        return []

    name_col = (
        "full_name" if "full_name" in rivals_df.columns
        else "athlete" if "athlete" in rivals_df.columns
        else "competitor" if "competitor" in rivals_df.columns
        else None
    )
    country_col = (
        "country_code" if "country_code" in rivals_df.columns
        else "country" if "country" in rivals_df.columns
        else "nat" if "nat" in rivals_df.columns
        else None
    )
    rank_col = (
        "world_rank" if "world_rank" in rivals_df.columns
        else "rank" if "rank" in rivals_df.columns
        else None
    )
    score_col = (
        "ranking_score" if "ranking_score" in rivals_df.columns
        else "resultscore" if "resultscore" in rivals_df.columns
        else None
    )

    rows = []
    for _, row in rivals_df.head(10).iterrows():
        rows.append(
            {
                "Name": row.get(name_col, "-") if name_col else "-",
                "Country": row.get(country_col, "-") if country_col else "-",
                "World Rank": row.get(rank_col, "-") if rank_col else "-",
                "Score": row.get(score_col, "-") if score_col else "-",
            }
        )
    return rows

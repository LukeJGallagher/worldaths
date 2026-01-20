"""
Historical Benchmarks for Athletics Championships

Calculates what it takes to:
- Win a medal (top 3)
- Make the final (top 8)
- Advance from semi-finals
- Survive heats

Based on historical championship data from Olympics, World Championships,
Asian Games, and other major events.

Adapted from Tilastopaja project for World Athletics data.

NOTE: This module is designed to work with both World Athletics (this project)
and Tilastopaja data. Future database combination is planned.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import statistics


# Championship competition IDs for benchmark calculations
BENCHMARK_CHAMPIONSHIPS = {
    'Olympics': ['13079218', '12992925', '12877460', '12825110', '12042259'],  # 2024-2008
    'World Championships': ['13046619', '13002354', '12935526', '12898707', '12844203'],  # 2023-2013
    'Asian Games': ['13048549', '12911586', '12854365'],  # 2023-2014
}

# Round name mappings for normalization
ROUND_MAPPINGS = {
    'final': ['Final', 'final', 'f', 'F'],
    'semi': ['Semi-Final', 'semi-final', 'sf', 'SF', 's', 'Semi'],
    'heat': ['Heat', 'heat', 'h', 'H', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8',
             'Heat 1', 'Heat 2', 'Heat 3', 'Heat 4', 'Heat 5', 'Heat 6', 'Heat 7', 'Heat 8'],
    'qualification': ['Qualification', 'qualification', 'q', 'Q', 'qual']
}


def normalize_round(round_name: str) -> str:
    """Normalize round name to standard format."""
    if not round_name:
        return 'unknown'

    round_lower = str(round_name).lower().strip()

    for standard, variants in ROUND_MAPPINGS.items():
        if round_name in variants or round_lower in [v.lower() for v in variants]:
            return standard

    return 'unknown'


def get_event_type(event_name: str) -> str:
    """
    Determine if event is time-based, distance-based, or points-based.

    Returns:
        'time' - lower is better (track events)
        'distance' - higher is better (jumps, throws)
        'points' - higher is better (combined events)
    """
    event_lower = event_name.lower()

    # Points events
    if any(x in event_lower for x in ['decathlon', 'heptathlon', 'pentathlon']):
        return 'points'

    # Distance events (field)
    distance_events = ['jump', 'vault', 'put', 'throw', 'discus', 'hammer', 'javelin']
    if any(x in event_lower for x in distance_events):
        return 'distance'

    # Default to time (track events)
    return 'time'


def normalize_event_name(event_name: str) -> str:
    """
    Normalize event names between World Athletics and Tilastopaja formats.

    World Athletics uses: '100-metres', '400-metres-hurdles'
    Tilastopaja uses: '100m', '400m Hurdles'
    """
    event_lower = event_name.lower().strip()

    # Map World Athletics format to standard
    wa_to_standard = {
        '100-metres': '100m',
        '200-metres': '200m',
        '400-metres': '400m',
        '800-metres': '800m',
        '1500-metres': '1500m',
        '3000-metres': '3000m',
        '5000-metres': '5000m',
        '10000-metres': '10000m',
        '110-metres-hurdles': '110m Hurdles',
        '100-metres-hurdles': '100m Hurdles',
        '400-metres-hurdles': '400m Hurdles',
        '3000-metres-steeplechase': '3000m Steeplechase',
        'high-jump': 'High Jump',
        'long-jump': 'Long Jump',
        'triple-jump': 'Triple Jump',
        'pole-vault': 'Pole Vault',
        'shot-put': 'Shot Put',
        'discus-throw': 'Discus Throw',
        'hammer-throw': 'Hammer Throw',
        'javelin-throw': 'Javelin Throw',
        'decathlon': 'Decathlon',
        'heptathlon': 'Heptathlon',
        'marathon': 'Marathon',
        '20-kilometres-walk': '20km Walk',
        '50-kilometres-walk': '50km Walk',
    }

    return wa_to_standard.get(event_lower, event_name)


def get_default_benchmarks(event: str, gender: str) -> Dict[str, Dict]:
    """
    Get default benchmarks for events without sufficient historical data.

    Based on typical World Championship performance levels.
    Values in seconds for track, meters for field, points for combined.
    """
    # Normalize event name first
    event_normalized = normalize_event_name(event)

    # Default benchmarks for common events
    defaults = {
        'men': {
            '100m': {'medal': 9.85, 'final': 10.02, 'semi': 10.12, 'heat': 10.25},
            '200m': {'medal': 19.85, 'final': 20.15, 'semi': 20.35, 'heat': 20.55},
            '400m': {'medal': 44.20, 'final': 44.60, 'semi': 45.10, 'heat': 45.50},
            '800m': {'medal': 103.50, 'final': 104.50, 'semi': 106.00, 'heat': 107.50},
            '1500m': {'medal': 212.00, 'final': 215.00, 'semi': 218.00, 'heat': 222.00},
            '5000m': {'medal': 780.00, 'final': 795.00, 'semi': None, 'heat': 810.00},
            '10000m': {'medal': 1620.00, 'final': 1650.00, 'semi': None, 'heat': None},
            '110m Hurdles': {'medal': 13.05, 'final': 13.25, 'semi': 13.45, 'heat': 13.65},
            '400m Hurdles': {'medal': 47.50, 'final': 48.20, 'semi': 49.00, 'heat': 49.80},
            '3000m Steeplechase': {'medal': 495.00, 'final': 505.00, 'semi': None, 'heat': 515.00},
            'High Jump': {'medal': 2.35, 'final': 2.28, 'semi': None, 'heat': 2.25},
            'Pole Vault': {'medal': 5.90, 'final': 5.75, 'semi': None, 'heat': 5.65},
            'Long Jump': {'medal': 8.35, 'final': 8.10, 'semi': None, 'heat': 8.00},
            'Triple Jump': {'medal': 17.50, 'final': 17.10, 'semi': None, 'heat': 16.90},
            'Shot Put': {'medal': 22.50, 'final': 21.50, 'semi': None, 'heat': 20.80},
            'Discus Throw': {'medal': 68.50, 'final': 66.00, 'semi': None, 'heat': 64.50},
            'Hammer Throw': {'medal': 80.00, 'final': 77.50, 'semi': None, 'heat': 75.00},
            'Javelin Throw': {'medal': 88.00, 'final': 84.00, 'semi': None, 'heat': 82.00},
            'Decathlon': {'medal': 8700, 'final': 8400, 'semi': None, 'heat': None},
            'Marathon': {'medal': 7380, 'final': 7500, 'semi': None, 'heat': None},  # ~2:03:00
            '20km Walk': {'medal': 4680, 'final': 4800, 'semi': None, 'heat': None},  # ~1:18:00
        },
        'women': {
            '100m': {'medal': 10.85, 'final': 11.02, 'semi': 11.15, 'heat': 11.30},
            '200m': {'medal': 22.00, 'final': 22.35, 'semi': 22.60, 'heat': 22.90},
            '400m': {'medal': 49.50, 'final': 50.20, 'semi': 51.00, 'heat': 51.80},
            '800m': {'medal': 117.00, 'final': 119.00, 'semi': 121.00, 'heat': 123.00},
            '1500m': {'medal': 238.00, 'final': 242.00, 'semi': 246.00, 'heat': 250.00},
            '5000m': {'medal': 870.00, 'final': 890.00, 'semi': None, 'heat': 910.00},
            '10000m': {'medal': 1800.00, 'final': 1850.00, 'semi': None, 'heat': None},
            '100m Hurdles': {'medal': 12.45, 'final': 12.65, 'semi': 12.85, 'heat': 13.05},
            '400m Hurdles': {'medal': 53.00, 'final': 54.00, 'semi': 55.00, 'heat': 56.00},
            '3000m Steeplechase': {'medal': 555.00, 'final': 570.00, 'semi': None, 'heat': 585.00},
            'High Jump': {'medal': 2.00, 'final': 1.94, 'semi': None, 'heat': 1.90},
            'Pole Vault': {'medal': 4.85, 'final': 4.65, 'semi': None, 'heat': 4.55},
            'Long Jump': {'medal': 7.00, 'final': 6.75, 'semi': None, 'heat': 6.60},
            'Triple Jump': {'medal': 14.80, 'final': 14.40, 'semi': None, 'heat': 14.20},
            'Shot Put': {'medal': 20.00, 'final': 19.00, 'semi': None, 'heat': 18.20},
            'Discus Throw': {'medal': 68.00, 'final': 65.00, 'semi': None, 'heat': 62.00},
            'Hammer Throw': {'medal': 77.00, 'final': 74.00, 'semi': None, 'heat': 71.00},
            'Javelin Throw': {'medal': 66.00, 'final': 63.00, 'semi': None, 'heat': 60.00},
            'Heptathlon': {'medal': 6700, 'final': 6400, 'semi': None, 'heat': None},
            'Marathon': {'medal': 8100, 'final': 8280, 'semi': None, 'heat': None},  # ~2:15:00
            '20km Walk': {'medal': 5280, 'final': 5400, 'semi': None, 'heat': None},  # ~1:28:00
        }
    }

    # Normalize gender
    gender_key = gender.lower() if gender else 'men'

    event_defaults = defaults.get(gender_key, {}).get(event_normalized, {})

    # If not found with normalized name, try original
    if not event_defaults:
        event_defaults = defaults.get(gender_key, {}).get(event, {})

    result = {}
    for round_name in ['medal', 'final', 'semi', 'heat']:
        value = event_defaults.get(round_name)
        if value is not None:
            result[round_name] = {
                'average': value,
                'range': (value * 0.98, value * 1.02),
                'cutoff': value,
                'editions': 0,
                'description': f'Default benchmark (typical World Championship level)'
            }

    return result


def calculate_round_benchmarks_from_df(
    df: pd.DataFrame,
    event: str,
    gender: str,
    rank_col: str = 'rank',
    result_col: str = 'result_numeric'
) -> Dict[str, Dict]:
    """
    Calculate historical benchmarks from a DataFrame.

    Works with World Athletics master.parquet format.

    Args:
        df: DataFrame with competition results
        event: Event name (World Athletics format like '100-metres')
        gender: 'men' or 'women'
        rank_col: Column name for rank/position
        result_col: Column name for numeric result

    Returns:
        Dict with benchmarks for each round
    """
    event_type = get_event_type(event)

    # Try to filter by event (handle both formats)
    event_normalized = normalize_event_name(event)
    filtered = df[
        (df['event'].str.lower() == event.lower()) |
        (df['event'].str.lower() == event_normalized.lower())
    ]

    # Filter by gender
    if 'gender' in filtered.columns:
        filtered = filtered[filtered['gender'].str.lower() == gender.lower()]

    if filtered.empty:
        return get_default_benchmarks(event, gender)

    # Ensure numeric result column exists
    if result_col not in filtered.columns:
        if 'result' in filtered.columns:
            filtered[result_col] = pd.to_numeric(filtered['result'], errors='coerce')
        else:
            return get_default_benchmarks(event, gender)

    # Ensure rank column is numeric
    if rank_col in filtered.columns:
        filtered[rank_col] = pd.to_numeric(filtered[rank_col], errors='coerce')

    benchmarks = {}

    # Medal line: Top 3 ranked performances
    if rank_col in filtered.columns:
        medalists = filtered[filtered[rank_col] <= 3]
        if not medalists.empty:
            medal_perfs = medalists[result_col].dropna().tolist()
            if medal_perfs:
                benchmarks['medal'] = {
                    'average': round(statistics.mean(medal_perfs), 2),
                    'range': (round(min(medal_perfs), 2), round(max(medal_perfs), 2)),
                    'best': round(min(medal_perfs) if event_type == 'time' else max(medal_perfs), 2),
                    'cutoff': round(statistics.mean(medal_perfs), 2),
                    'editions': len(medal_perfs),
                    'description': 'Top 3 finishers (medal performances)'
                }

    # Final line: Top 8 ranked performances
    if rank_col in filtered.columns:
        finalists = filtered[filtered[rank_col] <= 8]
        if not finalists.empty:
            final_perfs = finalists[result_col].dropna().tolist()
            if final_perfs:
                benchmarks['final'] = {
                    'average': round(statistics.mean(final_perfs), 2),
                    'range': (round(min(final_perfs), 2), round(max(final_perfs), 2)),
                    'cutoff': round(max(final_perfs) if event_type == 'time' else min(final_perfs), 2),
                    'editions': len(final_perfs),
                    'description': 'Top 8 finishers (final qualifiers)'
                }

    # Semi line: Top 16 (estimate)
    if rank_col in filtered.columns:
        semi_finalists = filtered[filtered[rank_col] <= 16]
        if len(semi_finalists) > 8:
            semi_perfs = semi_finalists[result_col].dropna().tolist()
            if semi_perfs:
                benchmarks['semi'] = {
                    'average': round(statistics.mean(semi_perfs), 2),
                    'range': (round(min(semi_perfs), 2), round(max(semi_perfs), 2)),
                    'cutoff': round(statistics.median(semi_perfs), 2),
                    'editions': len(semi_perfs),
                    'description': 'Semi-final level performances'
                }

    # Heat line: Top 24-32 (estimate)
    if rank_col in filtered.columns:
        heat_qualifiers = filtered[filtered[rank_col] <= 32]
        if len(heat_qualifiers) > 16:
            heat_perfs = heat_qualifiers[result_col].dropna().tolist()
            if heat_perfs:
                # Use 75th percentile for heat survival
                sorted_perfs = sorted(heat_perfs, reverse=(event_type != 'time'))
                cutoff_idx = int(len(sorted_perfs) * 0.75)
                benchmarks['heat'] = {
                    'average': round(statistics.mean(heat_perfs), 2),
                    'range': (round(min(heat_perfs), 2), round(max(heat_perfs), 2)),
                    'cutoff': round(sorted_perfs[cutoff_idx], 2),
                    'editions': len(heat_perfs),
                    'description': 'Heat survival level'
                }

    # Fill in missing benchmarks with defaults
    default_benchmarks = get_default_benchmarks(event, gender)
    for round_name in ['medal', 'final', 'semi', 'heat']:
        if round_name not in benchmarks:
            benchmarks[round_name] = default_benchmarks.get(round_name, {})

    return benchmarks


def get_benchmarks_for_event(
    event: str,
    gender: str,
    df: pd.DataFrame = None,
    use_defaults: bool = True
) -> Dict[str, float]:
    """
    Get simplified benchmark dictionary (just cutoff values) for an event.

    Args:
        event: Event name
        gender: 'men' or 'women'
        df: Optional DataFrame to calculate from
        use_defaults: Fall back to defaults if no data

    Returns:
        Dict with 'medal', 'final', 'semi', 'heat' cutoff values
    """
    if df is not None and not df.empty:
        benchmarks = calculate_round_benchmarks_from_df(df, event, gender)
    elif use_defaults:
        benchmarks = get_default_benchmarks(event, gender)
    else:
        return {}

    # Extract just the cutoff values
    result = {}
    for round_name in ['medal', 'final', 'semi', 'heat']:
        if round_name in benchmarks:
            cutoff = benchmarks[round_name].get('cutoff') or benchmarks[round_name].get('average')
            if cutoff:
                result[round_name] = cutoff

    return result


def get_benchmark_summary(benchmarks: Dict[str, Dict], event_type: str = 'time') -> str:
    """
    Generate human-readable summary of benchmarks.

    Args:
        benchmarks: Benchmark dictionary from calculate_round_benchmarks
        event_type: 'time', 'distance', or 'points'

    Returns:
        Formatted string summary
    """
    lines = []

    round_order = ['medal', 'final', 'semi', 'heat']
    round_names = {
        'medal': 'Medal Zone (Top 3)',
        'final': 'Final Zone (Top 8)',
        'semi': 'Semi-Final Qualifier',
        'heat': 'Heat Survival'
    }

    for round_key in round_order:
        if round_key in benchmarks:
            data = benchmarks[round_key]
            avg = data.get('average') or data.get('cutoff', 'N/A')

            if event_type == 'time' and isinstance(avg, (int, float)):
                if avg >= 3600:  # Hours
                    hours = int(avg // 3600)
                    mins = int((avg % 3600) // 60)
                    secs = avg % 60
                    avg_str = f"{hours}:{mins:02d}:{secs:05.2f}"
                elif avg >= 60:  # Minutes
                    mins = int(avg // 60)
                    secs = avg % 60
                    avg_str = f"{mins}:{secs:05.2f}"
                else:
                    avg_str = f"{avg:.2f}"
            elif isinstance(avg, (int, float)):
                avg_str = f"{avg:.2f}" if event_type == 'distance' else f"{int(avg)}"
            else:
                avg_str = str(avg)

            lines.append(f"{round_names.get(round_key, round_key)}: {avg_str}")

    return '\n'.join(lines)


def format_benchmark_for_display(value: float, event_type: str = 'time') -> str:
    """Format benchmark value for display."""
    if value is None:
        return 'N/A'

    if event_type == 'time':
        if value >= 3600:  # Hours
            hours = int(value // 3600)
            mins = int((value % 3600) // 60)
            secs = value % 60
            return f"{hours}:{mins:02d}:{secs:05.2f}"
        elif value >= 60:  # Minutes
            mins = int(value // 60)
            secs = value % 60
            return f"{mins}:{secs:05.2f}"
        else:
            return f"{value:.2f}"
    elif event_type == 'points':
        return f"{int(value)}"
    else:
        return f"{value:.2f}m"


# Team Saudi branded colors for benchmark lines
BENCHMARK_COLORS = {
    'medal': '#a08e66',    # Gold accent
    'final': '#007167',    # Primary teal
    'semi': '#009688',     # Light teal
    'heat': '#78909C'      # Gray blue
}


# Methodology documentation
BENCHMARK_METHODOLOGY = """
## Championship Benchmark Methodology

### Data Sources
Benchmarks are calculated from the last 3-5 editions of each championship:
- Olympics: 2024, 2021, 2016, 2012, 2008
- World Championships: 2023, 2022, 2019, 2017, 2013
- Asian Games: 2023, 2018, 2014

### Medal Line
Average performance of gold, silver, and bronze medalists.
This represents the typical performance needed to win a medal.

### Final Line
Average performance of all finalists (typically top 8).
Athletes performing at this level are competitive for finals.

### Semi-Final Line
Estimated from qualifying performances in semi-finals.
Based on the slowest/lowest automatic qualifiers plus time qualifiers.

### Heat Survival
Minimum performance typically needed to advance from heats.
Based on historical heat qualifying times/marks.

### Limitations
- Benchmarks represent historical averages, not guarantees
- Championship conditions vary (altitude, weather, depth of field)
- Tactical racing can produce slower winning times
- Field events may have different qualifying standards each year
"""

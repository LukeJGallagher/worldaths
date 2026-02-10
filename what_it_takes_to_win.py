"""
What It Takes to Win - Analysis Module
Analyzes historical winning marks at major championships to show athletes
what performance levels are required to medal at different competitions.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Team Saudi Brand Colors
TEAL_PRIMARY = '#007167'
GOLD_ACCENT = '#a08e66'
TEAL_DARK = '#005a51'
TEAL_LIGHT = '#009688'
GRAY_BLUE = '#78909C'

# Try to import Azure/Parquet data connector
try:
    from data_connector import (
        get_rankings_data, get_data_mode, _download_parquet_from_azure,
        get_major_championships_data
    )
    DATA_CONNECTOR_AVAILABLE = True
except ImportError:
    DATA_CONNECTOR_AVAILABLE = False

# Major competition tiers
COMPETITION_TIERS = {
    'Tier 1 - Global': ['Olympic Games', 'World Athletics Championships', 'World Championships'],
    'Tier 2 - Continental': ['Asian Games', 'Asian Athletics Championships', 'Asian Indoor'],
    'Tier 3 - Regional': ['West Asian Championships', 'Arab Athletics', 'GCC'],
    'Tier 4 - Age Group': ['World Athletics U20', 'World U20', 'Asian U20', 'Arab U18', 'Arab U20', 'Arab U23']
}

# Event categories for analysis
EVENT_CATEGORIES = {
    'Sprints': ['100m', '200m', '400m', '100-metres', '200-metres', '400-metres'],
    'Middle Distance': ['800m', '1500m', '800-metres', '1500-metres'],
    'Long Distance': ['5000m', '10000m', 'Marathon', '5000-metres', '10000-metres', 'marathon'],
    'Hurdles': ['110mH', '400mH', '100mH', '110-metres-hurdles', '400-metres-hurdles', '100-metres-hurdles'],
    'Jumps': ['High Jump', 'Long Jump', 'Triple Jump', 'Pole Vault', 'high-jump', 'long-jump', 'triple-jump', 'pole-vault'],
    'Throws': ['Shot Put', 'Discus', 'Javelin', 'Hammer', 'shot-put', 'discus-throw', 'javelin-throw', 'hammer-throw'],
    'Combined': ['Decathlon', 'Heptathlon', 'decathlon', 'heptathlon']
}

# Bidirectional event name mapping: database format <-> display format
EVENT_DB_TO_DISPLAY = {
    '100-metres': '100m', '200-metres': '200m', '400-metres': '400m',
    '800-metres': '800m', '1500-metres': '1500m', '3000-metres': '3000m',
    '5000-metres': '5000m', '10000-metres': '10000m',
    '110-metres-hurdles': '110m Hurdles', '100-metres-hurdles': '100m Hurdles',
    '400-metres-hurdles': '400m Hurdles', '3000-metres-steeplechase': '3000m Steeplechase',
    'high-jump': 'High Jump', 'pole-vault': 'Pole Vault',
    'long-jump': 'Long Jump', 'triple-jump': 'Triple Jump',
    'shot-put': 'Shot Put', 'discus-throw': 'Discus Throw',
    'hammer-throw': 'Hammer Throw', 'javelin-throw': 'Javelin Throw',
    'decathlon': 'Decathlon', 'heptathlon': 'Heptathlon',
    'marathon': 'Marathon', 'half-marathon': 'Half Marathon',
    '4-x-100-metres-relay': '4x100m Relay', '4-x-400-metres-relay': '4x400m Relay',
    '20-kilometres-race-walk': '20km Race Walk',
    '35-kilometres-race-walk': '35km Race Walk',
    '50-kilometres-race-walk': '50km Race Walk',
}
EVENT_DISPLAY_TO_DB = {v: k for k, v in EVENT_DB_TO_DISPLAY.items()}


def format_event_name(db_name: str) -> str:
    """Convert database event name to display name. '100-metres' -> '100m'."""
    return EVENT_DB_TO_DISPLAY.get(db_name, db_name.replace('-', ' ').title())


def normalize_event_for_match(event_name: str) -> str:
    """Normalize event name for exact comparison.
    Both '100-metres' and '100m' -> '100metres'.
    """
    return re.sub(r'[^0-9a-z]', '', event_name.lower())


# Standard marks for different medal positions at different competition levels
# These are updated dynamically from scraped data
class WhatItTakesToWin:
    """Analyze winning marks and standards for major championships."""

    # All major competition keywords for filtering
    MAJOR_COMP_KEYWORDS = {
        'Olympic': ['Olympic', 'Olympics', 'XXXIII Olympic', 'XXXII Olympic', 'Paris 2024', 'Tokyo 2020', 'Rio 2016', 'London 2012'],
        'World Champs': ['World Athletics Championships', 'World Championships', 'Worlds', 'WCH', 'Budapest 2023', 'Oregon 2022', 'Doha 2019', 'London 2017', 'Beijing 2015'],
        'Asian Games': ['Asian Games', 'Asiad', 'Hangzhou 2023', 'Jakarta 2018', 'Incheon 2014', 'Guangzhou 2010'],
        'Asian Champs': ['Asian Athletics Championships', 'Asian Indoor', 'Asian Championships', 'Asian Athletics'],
        'Arab': ['Arab Athletics', 'Arab Championships', 'Pan Arab', 'Arab Games'],
        'Diamond League': ['Diamond League', 'DL ', 'Zurich', 'Brussels', 'Monaco', 'Rome', 'Eugene', 'Stockholm', 'Paris DL'],
        'All': None  # No filter - use all data
    }

    # Date-aware city-championship mapping (city, year_start, year_end, championship)
    # This resolves conflicts where same city hosts different championships
    CITY_YEAR_MAPPING = [
        # Olympics (specific years - July/August usually)
        ('Paris', 2024, 2024, 'Olympic'),
        ('Tokyo', 2021, 2021, 'Olympic'),  # Held in 2021 due to COVID
        ('Rio', 2016, 2016, 'Olympic'),
        ('London', 2012, 2012, 'Olympic'),
        ('Beijing', 2008, 2008, 'Olympic'),
        ('Athens', 2004, 2004, 'Olympic'),
        ('Sydney', 2000, 2000, 'Olympic'),
        ('Atlanta', 1996, 1996, 'Olympic'),
        ('Barcelona', 1992, 1992, 'Olympic'),
        ('Seoul', 1988, 1988, 'Olympic'),
        # World Championships (specific years)
        ('Budapest', 2023, 2023, 'World Champs'),
        ('Eugene', 2022, 2022, 'World Champs'),
        ('Doha', 2019, 2019, 'World Champs'),
        ('London', 2017, 2017, 'World Champs'),
        ('Beijing', 2015, 2015, 'World Champs'),
        ('Moscow', 2013, 2013, 'World Champs'),
        ('Daegu', 2011, 2011, 'World Champs'),
        ('Berlin', 2009, 2009, 'World Champs'),
        ('Osaka', 2007, 2007, 'World Champs'),
        ('Helsinki', 2005, 2005, 'World Champs'),
        ('Paris', 2003, 2003, 'World Champs'),
        # Asian Games (specific years - typically September/October)
        ('Hangzhou', 2023, 2023, 'Asian Games'),
        ('Jakarta', 2018, 2018, 'Asian Games'),
        ('Incheon', 2014, 2014, 'Asian Games'),
        ('Guangzhou', 2010, 2010, 'Asian Games'),
        ('Doha', 2006, 2006, 'Asian Games'),
        ('Busan', 2002, 2002, 'Asian Games'),
    ]

    # City to Championship mapping (for venues stored as city names)
    VENUE_CITY_MAPPING = {
        # World Championships venues
        'Budapest': 'World Champs',
        'Eugene': 'World Champs',  # Also Diamond League
        'Doha': 'World Champs',    # Also Diamond League
        'London': 'World Champs',  # Also Olympic 2012
        'Beijing': 'World Champs', # Also Olympic 2008
        'Moscow': 'World Champs',
        'Moskva': 'World Champs',
        'Daegu': 'World Champs',
        'Berlin': 'World Champs',
        'Osaka': 'World Champs',
        'Helsinki': 'World Champs',
        'Paris': 'World Champs',   # Also Olympic 2024
        'Edmonton': 'World Champs',
        'Seville': 'World Champs',
        'Athens': 'World Champs',  # Also Olympic 2004
        'Gothenburg': 'World Champs',
        'Stuttgart': 'World Champs',
        'Tokyo': 'World Champs',   # Also Olympic 2020/2021
        # Olympics venues (by year for disambiguation)
        'Rio': 'Olympic',
        'Sydney': 'Olympic',
        'Atlanta': 'Olympic',
        'Barcelona': 'Olympic',
        'Seoul': 'Olympic',
        'Los Angeles': 'Olympic',
        # Asian Games venues
        'Hangzhou': 'Asian Games',
        'Jakarta': 'Asian Games',
        'Incheon': 'Asian Games',
        'Guangzhou': 'Asian Games',
        'Busan': 'Asian Games',
        'Bangkok': 'Asian Games',
        'Hiroshima': 'Asian Games',
        # Diamond League venues
        'Zurich': 'Diamond League',
        'Brussels': 'Diamond League',
        'Monaco': 'Diamond League',
        'Rome': 'Diamond League',
        'Stockholm': 'Diamond League',
        'Birmingham': 'Diamond League',
        'Rabat': 'Diamond League',
        'Shanghai': 'Diamond League',
        'Lausanne': 'Diamond League',
        'Oslo': 'Diamond League',
        'Chorzow': 'Diamond League',
        'Silesia': 'Diamond League',
    }

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'world_athletics_scraperv2', 'data')
        self.sql_dir = os.path.join(os.path.dirname(__file__), 'SQL')
        self.data = None
        self.winning_marks = {}

    def _filter_major_championships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-filter data to only include major championship records.
        This significantly reduces memory usage from 2.3M to ~50-100K records.
        """
        if df is None or len(df) == 0:
            return df

        # Find venue column
        venue_col = None
        for col_name in ['Venue', 'venue', 'Competition', 'competition', 'comp', 'meet', 'Meet']:
            if col_name in df.columns:
                venue_col = col_name
                break

        if venue_col is None:
            # No venue column - return all data
            print("Warning: No venue column found for championship filtering")
            return df

        # Collect all major championship keywords
        all_keywords = []
        for keywords in self.MAJOR_COMP_KEYWORDS.values():
            if keywords is not None:  # Skip 'All' which has None value
                all_keywords.extend(keywords)
        # Add city names from venue mapping
        all_keywords.extend(list(self.VENUE_CITY_MAPPING.keys()))
        # Remove duplicates
        all_keywords = list(set(all_keywords))

        # Memory-efficient filtering: process in chunks for large datasets
        if len(df) > 1000000:
            # Process in chunks to avoid memory issues
            chunk_size = 200000
            filtered_chunks = []

            for start_idx in range(0, len(df), chunk_size):
                end_idx = min(start_idx + chunk_size, len(df))
                chunk = df.iloc[start_idx:end_idx]

                # Create lowercase venue column for matching
                venue_lower = chunk[venue_col].fillna('').str.lower()

                # Build mask incrementally
                chunk_mask = pd.Series([False] * len(chunk), index=chunk.index)
                for keyword in all_keywords:
                    chunk_mask = chunk_mask | venue_lower.str.contains(keyword.lower(), regex=False, na=False)

                if chunk_mask.any():
                    filtered_chunks.append(chunk[chunk_mask])

            if filtered_chunks:
                return pd.concat(filtered_chunks, ignore_index=True)
            else:
                print("Warning: No major championship records found in data")
                return pd.DataFrame()
        else:
            # Standard approach for smaller datasets
            pattern = '|'.join(all_keywords)
            mask = df[venue_col].str.contains(pattern, case=False, na=False)
            return df[mask]

    def load_scraped_data(self) -> pd.DataFrame:
        """Load data from Azure parquet or local CSV files."""
        # Try optimized major championships query first (uses DuckDB filtering)
        if DATA_CONNECTOR_AVAILABLE:
            try:
                # Use the new optimized function that filters in DuckDB
                df = get_major_championships_data()

                if df is not None and not df.empty:
                    # Normalize column names to match expected format
                    column_map = {
                        'competitor': 'Competitor',
                        'event': 'Event',
                        'result': 'Mark',
                        'result_numeric': 'Mark_Numeric',
                        'date': 'Date',
                        'venue': 'Venue',
                        'gender': 'Gender',
                        'nat': 'Country',
                        'rank': 'Rank',
                        'year': 'year'  # Preserve year column if present
                    }
                    for old_col, new_col in column_map.items():
                        if old_col in df.columns and new_col not in df.columns:
                            df = df.rename(columns={old_col: new_col})

                    # Extract year from Date if year column not present
                    if 'year' not in df.columns and 'Date' in df.columns:
                        try:
                            df['year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
                        except Exception:
                            pass

                    # Data is already filtered for major championships by get_major_championships_data()
                    self.data = df
                    print(f"Loaded {len(self.data):,} major championship records (DuckDB filtered)")
                    return self.data
            except Exception as e:
                print(f"Warning: Could not load Azure data: {e}")
                import traceback
                print(traceback.format_exc())

        # Fall back to local CSV files
        db_cleaned_path = os.path.join(self.data_dir, 'db_cleaned.csv')
        db_path = os.path.join(self.data_dir, 'db.csv')

        # Try cleaned first, then raw
        if os.path.exists(db_cleaned_path):
            self.data = pd.read_csv(db_cleaned_path)
            print(f"Loaded {len(self.data)} records from cleaned database")
        elif os.path.exists(db_path):
            self.data = pd.read_csv(db_path)
            print(f"Loaded {len(self.data)} records from raw database")
        else:
            print(f"No data found in {self.data_dir}")
            self.data = pd.DataFrame()

        return self.data

    def filter_by_competition(self, comp_type: str = 'All') -> pd.DataFrame:
        """Filter data by competition type using championship_type column or date-aware city matching."""
        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()

        if comp_type == 'All' or comp_type not in self.MAJOR_COMP_KEYWORDS:
            return self.data

        df = self.data.copy()

        # First try: Use championship_type column if available (added by get_major_championships_data)
        if 'championship_type' in df.columns:
            filtered = df[df['championship_type'] == comp_type]
            if len(filtered) > 0:
                print(f"Filtered to {len(filtered):,} '{comp_type}' records using championship_type column")
                return filtered

        # Fall back to venue-based matching
        keywords = self.MAJOR_COMP_KEYWORDS.get(comp_type)
        if not keywords:
            return self.data

        # Try multiple possible column names for competition/venue
        venue_col = None
        for col_name in ['Venue', 'venue', 'Competition', 'competition', 'comp', 'meet', 'Meet']:
            if col_name in df.columns:
                venue_col = col_name
                break

        if venue_col is None:
            print(f"Warning: No venue/competition column found. Available columns: {df.columns.tolist()}")
            return df

        # Get date column
        date_col = None
        for col_name in ['date', 'Date', 'DATE']:
            if col_name in df.columns:
                date_col = col_name
                break

        # Extract year if we have a date column
        if date_col is not None:
            try:
                df['_filter_year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
            except:
                df['_filter_year'] = None

        # Convert venue to lowercase for matching
        venue_lower = df[venue_col].fillna('').str.lower()

        # Build mask using multiple approaches
        mask = pd.Series([False] * len(df), index=df.index)

        # 1. Match explicit keywords in venue name (e.g., "Olympic Stadium", "World Championships")
        for keyword in keywords:
            mask = mask | venue_lower.str.contains(keyword.lower(), regex=False, na=False)

        # 2. Use date-aware city mapping for disambiguation
        if date_col is not None and '_filter_year' in df.columns:
            for city, year_start, year_end, champ in self.CITY_YEAR_MAPPING:
                if champ == comp_type:
                    city_mask = venue_lower.str.contains(city.lower(), regex=False, na=False)
                    year_mask = (df['_filter_year'] >= year_start) & (df['_filter_year'] <= year_end)
                    mask = mask | (city_mask & year_mask)

        # 3. For Diamond League, match venues without date constraint (they're recurring)
        if comp_type == 'Diamond League':
            dl_cities = [city for city, champ in self.VENUE_CITY_MAPPING.items() if champ == 'Diamond League']
            for city in dl_cities:
                mask = mask | venue_lower.str.contains(city.lower(), regex=False, na=False)

        # Clean up temporary column
        if '_filter_year' in df.columns:
            df = df.drop(columns=['_filter_year'])

        filtered = df[mask]

        if len(filtered) == 0:
            print(f"Warning: No records matched '{comp_type}' filter. Sample venues: {self.data[venue_col].dropna().head(10).tolist()}")

        return filtered

    def get_competition_types(self) -> list:
        """Get list of available competition type filters from actual data."""
        # Return types that actually exist in the data's championship_type column
        if self.data is not None and 'championship_type' in self.data.columns:
            actual_types = self.data['championship_type'].dropna().unique().tolist()
            # Put 'All' first, then sort the rest (excluding 'Other')
            sorted_types = ['All'] + sorted([t for t in actual_types if t != 'Other'])
            return sorted_types
        # Fallback to keyword keys
        return list(self.MAJOR_COMP_KEYWORDS.keys())

    def parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """Convert time string to seconds for comparison."""
        if pd.isna(time_str) or time_str == '':
            return None

        time_str = str(time_str).strip()

        # Handle DNF, DNS, DQ, etc.
        if any(x in time_str.upper() for x in ['DNF', 'DNS', 'DQ', 'NM', '-']):
            return None

        try:
            # Remove any leading/trailing characters
            time_str = time_str.strip()

            # Handle hours:minutes:seconds format (marathon)
            if time_str.count(':') == 2:
                parts = time_str.split(':')
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

            # Handle minutes:seconds format
            elif ':' in time_str:
                parts = time_str.split(':')
                return float(parts[0]) * 60 + float(parts[1])

            # Handle seconds only (sprints)
            else:
                return float(time_str)

        except (ValueError, IndexError):
            return None

    def parse_distance_to_meters(self, mark_str: str) -> Optional[float]:
        """Convert distance string to meters for field events."""
        if pd.isna(mark_str) or mark_str == '':
            return None

        mark_str = str(mark_str).strip()

        # Handle NM, X, etc.
        if any(x in mark_str.upper() for x in ['NM', 'X', '-', 'DNS', 'DNF']):
            return None

        try:
            return float(mark_str)
        except ValueError:
            return None

    def is_field_event(self, event: str) -> bool:
        """Check if event is a field event (higher is better)."""
        field_keywords = ['jump', 'vault', 'put', 'throw', 'discus', 'javelin', 'hammer', 'decathlon', 'heptathlon']
        return any(kw in event.lower() for kw in field_keywords)

    @staticmethod
    def parse_position(pos_str) -> Optional[int]:
        """Extract numeric finishing position from pos column.

        Examples: '1' -> 1, '2ce1' -> 2, '1sf3' -> 1, '7h1' -> 7, 'dns' -> None
        """
        if pd.isna(pos_str):
            return None
        pos_clean = str(pos_str).strip().lower()
        if any(x in pos_clean for x in ['dns', 'dnf', 'dq', 'nm', 'nr']):
            return None
        match = re.match(r'^(\d+)', pos_clean)
        return int(match.group(1)) if match else None

    @staticmethod
    def filter_outlier_marks(marks: pd.Series, is_field: bool = False) -> pd.Series:
        """Remove obviously invalid marks using median-based outlier detection.

        Some scraped records have garbage values (e.g., WPA scoring points 1775.06
        instead of actual 100m time 10.49). This filters them out by removing
        values that deviate more than 5x from the median.

        Returns a boolean mask (True = keep, False = outlier).
        """
        valid = marks.dropna()
        if len(valid) < 3:
            return pd.Series(True, index=marks.index)  # Not enough data to detect outliers

        median_val = valid.median()
        if median_val <= 0:
            return pd.Series(True, index=marks.index)

        # Remove values more than 5x the median (catches 1775 when median is ~10)
        # Also remove values less than 1/5 the median (catches impossibly low values)
        mask = marks.isna() | ((marks >= median_val / 5) & (marks <= median_val * 5))
        return mask

    @staticmethod
    def is_finals_result(pos_str) -> bool:
        """Check if a result is from a final (not heat/semi/qualifier).

        Finals positions are plain numbers (1-12) or have 'f' suffix.
        Heats have 'h', semis have 'sf', qualifiers have 'q'.
        Combined events 'ce' are treated as finals.
        """
        if pd.isna(pos_str):
            return False
        pos_clean = str(pos_str).strip().lower()
        if any(x in pos_clean for x in ['dns', 'dnf', 'dq', 'nm', 'nr']):
            return False
        # Plain number = final result
        if re.match(r'^\d+$', pos_clean):
            return True
        # 'f' suffix = final, 'ce' = combined event (also a final)
        if re.match(r'^\d+(f|ce)', pos_clean):
            return True
        # Exclude heats (h), semis (sf), qualifiers (q)
        return False

    def analyze_top_performances(self, event: str = None, gender: str = None,
                                  year: int = None, top_n: int = 8) -> pd.DataFrame:
        """Analyze top performances for given filters (simulating finals)."""
        if self.data is None or len(self.data) == 0:
            self.load_scraped_data()

        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()

        df = self.data.copy()

        # Apply filters - exact match after normalization
        if event:
            event_normalized = normalize_event_for_match(event)
            event_lower = df['Event'].astype(str).str.lower().str.replace('-', '', regex=False).str.replace(' ', '', regex=False)
            df = df[event_lower == event_normalized]
        if gender:
            df = df[df['Gender'].str.lower() == gender.lower()]
        if year:
            df = df[df['Date'].astype(str).str.contains(str(year), na=False)]

        # Get column names
        mark_col = 'Mark' if 'Mark' in df.columns else 'Result'

        if len(df) == 0:
            return pd.DataFrame()

        # Parse marks based on event type
        is_field = self.is_field_event(event) if event else False

        if is_field:
            df['ParsedMark'] = df[mark_col].apply(self.parse_distance_to_meters)
            df = df.dropna(subset=['ParsedMark'])
            df = df.sort_values('ParsedMark', ascending=False)
        else:
            df['ParsedMark'] = df[mark_col].apply(self.parse_time_to_seconds)
            df = df.dropna(subset=['ParsedMark'])
            df = df.sort_values('ParsedMark', ascending=True)

        return df.head(top_n)

    def get_medal_standards(self, event: str, gender: str, year: int = None) -> Dict:
        """Get gold/silver/bronze standards for an event using actual competition rankings."""
        if self.data is None or len(self.data) == 0:
            self.load_scraped_data()

        if self.data is None or len(self.data) == 0:
            return {'gold': None, 'silver': None, 'bronze': None, 'final_standard': None,
                    'top_8_avg': None, 'top_20_avg': None, 'is_field_event': False, 'sample_size': 0}

        df = self.data.copy()
        is_field = self.is_field_event(event)

        # Get column names (handle both uppercase and lowercase)
        event_col = 'Event' if 'Event' in df.columns else 'event'
        gender_col = 'Gender' if 'Gender' in df.columns else 'gender'
        rank_col = 'Rank' if 'Rank' in df.columns else 'rank'
        mark_col = 'Mark' if 'Mark' in df.columns else ('result' if 'result' in df.columns else 'Result')
        date_col = 'Date' if 'Date' in df.columns else 'date'

        # Apply filters - use word-boundary matching to prevent '100' from matching '10000'
        if event and event_col in df.columns:
            # Strip separators for flexible matching, add lookbehind to prevent partial number matches
            event_normalized = re.sub(r'[^0-9a-z]', '', event.lower())
            event_lower = df[event_col].astype(str).str.lower().str.replace('-', '', regex=False).str.replace(' ', '', regex=False)
            df = df[event_lower == event_normalized]
        if gender and gender_col in df.columns:
            df = df[df[gender_col].astype(str).str.lower().str.contains(gender.lower(), na=False)]
        if year and date_col in df.columns:
            df = df[df[date_col].astype(str).str.contains(str(year), na=False)]

        if len(df) == 0:
            return {'gold': None, 'silver': None, 'bronze': None, 'final_standard': None,
                    'top_8_avg': None, 'top_20_avg': None, 'is_field_event': is_field, 'sample_size': 0}

        # Parse marks for numeric comparison
        if is_field:
            df['ParsedMark'] = df[mark_col].apply(self.parse_distance_to_meters)
        else:
            df['ParsedMark'] = df[mark_col].apply(self.parse_time_to_seconds)

        df = df.dropna(subset=['ParsedMark'])

        # Filter out outlier marks (e.g., garbage values like 1775 for a 100m race)
        outlier_mask = self.filter_outlier_marks(df['ParsedMark'], is_field)
        df = df[outlier_mask]

        if len(df) == 0:
            return {'gold': None, 'silver': None, 'bronze': None, 'final_standard': None,
                    'top_8_avg': None, 'top_20_avg': None, 'is_field_event': is_field, 'sample_size': 0}

        # METHOD 1: Use pos column for finishing position (1st, 2nd, 3rd place)
        # NOTE: 'rank' column = world ranking number (1-7000), NOT finishing position
        # The 'pos' column has actual finishing positions: 1, 2, 3, 1sf3, 2h1, etc.
        pos_col = None
        for col_name in ['pos', 'Pos', 'Position', 'position']:
            if col_name in df.columns:
                pos_col = col_name
                break

        if pos_col is not None:
            # Parse finishing positions from pos column
            df['_finishing_pos'] = df[pos_col].apply(self.parse_position)
            df['_is_final'] = df[pos_col].apply(self.is_finals_result)

            # Filter to finals only (exclude heats, semis, qualifiers)
            finals_df = df[df['_is_final']].copy()

            # If no finals identified, fall back to all results with valid positions
            if finals_df.empty:
                finals_df = df.dropna(subset=['_finishing_pos']).copy()

            if not finals_df.empty:
                # Get actual 1st, 2nd, 3rd place finishers from finals
                gold_results = finals_df[finals_df['_finishing_pos'] == 1]['ParsedMark']
                silver_results = finals_df[finals_df['_finishing_pos'] == 2]['ParsedMark']
                bronze_results = finals_df[finals_df['_finishing_pos'] == 3]['ParsedMark']
                final_results = finals_df[finals_df['_finishing_pos'] == 8]['ParsedMark']
                top_8_results = finals_df[finals_df['_finishing_pos'] <= 8]['ParsedMark']
                top_20_results = finals_df[finals_df['_finishing_pos'] <= 20]['ParsedMark']

                # Get representative values (median for more stable results)
                gold = gold_results.median() if len(gold_results) > 0 else None
                silver = silver_results.median() if len(silver_results) > 0 else None
                bronze = bronze_results.median() if len(bronze_results) > 0 else None
                final_standard = final_results.median() if len(final_results) > 0 else None
                top_8_avg = top_8_results.mean() if len(top_8_results) > 0 else None
                top_20_avg = top_20_results.mean() if len(top_20_results) > 0 else None

                # Enforce logical ordering: gold should be better than silver, silver better than bronze
                # For track (lower is better): gold <= silver <= bronze
                # For field (higher is better): gold >= silver >= bronze
                if gold is not None and silver is not None and bronze is not None:
                    medals = sorted([gold, silver, bronze], reverse=is_field)
                    gold, silver, bronze = medals[0], medals[1], medals[2]

                return {
                    'gold': gold,
                    'silver': silver,
                    'bronze': bronze,
                    'final_standard': final_standard,
                    'top_8_avg': top_8_avg,
                    'top_20_avg': top_20_avg,
                    'is_field_event': is_field,
                    'sample_size': len(finals_df)
                }

        # METHOD 2: Fallback - sort by performance and deduplicate by athlete
        athlete_col = 'Competitor' if 'Competitor' in df.columns else 'competitor'

        if is_field:
            df = df.sort_values('ParsedMark', ascending=False)
        else:
            df = df.sort_values('ParsedMark', ascending=True)

        if athlete_col in df.columns:
            # Keep best mark per athlete
            df = df.drop_duplicates(subset=[athlete_col], keep='first')

        marks = df['ParsedMark'].tolist()

        return {
            'gold': marks[0] if len(marks) > 0 else None,
            'silver': marks[1] if len(marks) > 1 else None,
            'bronze': marks[2] if len(marks) > 2 else None,
            'final_standard': marks[7] if len(marks) > 7 else marks[-1] if marks else None,
            'top_8_avg': np.mean(marks[:8]) if len(marks) >= 8 else np.mean(marks) if marks else None,
            'top_20_avg': np.mean(marks[:20]) if len(marks) >= 20 else np.mean(marks) if marks else None,
            'is_field_event': is_field,
            'sample_size': len(marks)
        }

    def format_time(self, seconds: float, event: str = '') -> str:
        """Format seconds back to readable time."""
        if seconds is None:
            return 'N/A'

        # Marathon format (h:mm:ss)
        if seconds >= 3600 or 'marathon' in event.lower():
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:05.2f}"

        # Middle/Long distance format (m:ss.xx)
        elif seconds >= 60:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}:{secs:05.2f}"

        # Sprint format (ss.xx)
        else:
            return f"{seconds:.2f}"

    def format_mark(self, mark: float, event: str) -> str:
        """Format mark appropriately for event type."""
        if mark is None:
            return 'N/A'

        if self.is_field_event(event):
            if 'decathlon' in event.lower() or 'heptathlon' in event.lower():
                return f"{int(mark)} pts"
            return f"{mark:.2f}m"
        else:
            return self.format_time(mark, event)

    def generate_what_it_takes_report(self, gender: str = 'men', year: int = None,
                                       comp_type: str = 'All') -> pd.DataFrame:
        """Generate a comprehensive 'What It Takes to Win' report.

        Args:
            gender: 'men' or 'women'
            year: Filter by year (optional)
            comp_type: Filter by competition type - 'Olympic', 'World Champs',
                       'Asian Games', 'Arab', 'Diamond League', or 'All'
        """
        if self.data is None:
            self.load_scraped_data()

        # Filter by competition type if specified
        if comp_type and comp_type != 'All':
            filtered_data = self.filter_by_competition(comp_type)
            # Temporarily swap data for report generation
            original_data = self.data
            self.data = filtered_data

        events = self.get_available_events()
        results = []

        for event in events:
            standards = self.get_medal_standards(event, gender, year)

            if standards['gold'] is not None:
                results.append({
                    'Event': event,
                    'Gender': gender,
                    'Year': year or 'All',
                    'Gold Standard': self.format_mark(standards['gold'], event),
                    'Silver Standard': self.format_mark(standards['silver'], event),
                    'Bronze Standard': self.format_mark(standards['bronze'], event),
                    'Final Standard (8th)': self.format_mark(standards['final_standard'], event),
                    'Top 8 Average': self.format_mark(standards['top_8_avg'], event),
                    'Top 20 Average': self.format_mark(standards.get('top_20_avg'), event),
                    'Sample Size': standards['sample_size'],
                    'Gold_Raw': standards['gold'],
                    'Silver_Raw': standards['silver'],
                    'Bronze_Raw': standards['bronze']
                })

        # Restore original data if we filtered
        if comp_type and comp_type != 'All':
            self.data = original_data

        return pd.DataFrame(results)

    def get_available_events(self) -> List[str]:
        """Get list of available events in the data."""
        if self.data is None:
            self.load_scraped_data()

        if self.data is None:
            return []

        # Handle both uppercase and lowercase column names
        event_col = 'Event' if 'Event' in self.data.columns else 'event'
        if event_col in self.data.columns:
            events = self.data[event_col].dropna().unique().tolist()
            # Clean up event names (remove hyphens, standardize)
            return sorted([str(e) for e in events if e])
        return []

    def get_available_years(self) -> List[int]:
        """Get list of available years in the data."""
        if self.data is None:
            self.load_scraped_data()

        if self.data is None:
            return []

        # Handle both uppercase and lowercase column names
        date_col = 'Date' if 'Date' in self.data.columns else 'date'
        if date_col in self.data.columns:
            # Try to extract year from various date formats
            dates = self.data[date_col].astype(str)
            # For timestamp format like "2024-05-10" or string like "10 MAY 2024"
            years = []
            for d in dates.unique():
                try:
                    # Try extracting 4-digit year
                    import re
                    year_match = re.search(r'20\d{2}', str(d))
                    if year_match:
                        years.append(int(year_match.group()))
                except:
                    pass
            return sorted(list(set(years)), reverse=True)
        return []

    def compare_athlete_to_standards(self, athlete_mark: float, event: str,
                                     gender: str, year: int = None) -> Dict:
        """Compare an athlete's mark to championship standards."""
        standards = self.get_medal_standards(event, gender, year)
        is_field = self.is_field_event(event)

        # For field events: higher is better
        # For track events: lower is better
        if is_field:
            gold_gap = standards['gold'] - athlete_mark if standards['gold'] else None
            final_gap = standards['final_standard'] - athlete_mark if standards['final_standard'] else None
        else:
            gold_gap = athlete_mark - standards['gold'] if standards['gold'] else None
            final_gap = athlete_mark - standards['final_standard'] if standards['final_standard'] else None

        # Determine projected position
        position = 'Outside Finals'
        if standards['gold'] and ((is_field and athlete_mark >= standards['gold']) or
                                   (not is_field and athlete_mark <= standards['gold'])):
            position = 'Gold Medal'
        elif standards['silver'] and ((is_field and athlete_mark >= standards['silver']) or
                                       (not is_field and athlete_mark <= standards['silver'])):
            position = 'Silver Medal'
        elif standards['bronze'] and ((is_field and athlete_mark >= standards['bronze']) or
                                       (not is_field and athlete_mark <= standards['bronze'])):
            position = 'Bronze Medal'
        elif standards['final_standard'] and ((is_field and athlete_mark >= standards['final_standard']) or
                                               (not is_field and athlete_mark <= standards['final_standard'])):
            position = 'Finals'

        return {
            'athlete_mark': athlete_mark,
            'athlete_mark_formatted': self.format_mark(athlete_mark, event),
            'projected_position': position,
            'gap_to_gold': gold_gap,
            'gap_to_gold_formatted': self.format_mark(abs(gold_gap), event) if gold_gap else 'N/A',
            'gap_to_final': final_gap,
            'gap_to_final_formatted': self.format_mark(abs(final_gap), event) if final_gap else 'N/A',
            'standards': standards,
            'is_field_event': is_field
        }

    def get_year_over_year_trends(self, event: str, gender: str = 'men') -> pd.DataFrame:
        """Analyze how winning standards have changed over years."""
        years = self.get_available_years()
        trends = []

        for year in years:
            standards = self.get_medal_standards(event, gender, year)
            if standards['gold']:
                trends.append({
                    'Year': year,
                    'Event': event,
                    'Gender': gender,
                    'Gold': standards['gold'],
                    'Gold_Formatted': self.format_mark(standards['gold'], event),
                    'Top 8 Avg': standards['top_8_avg'],
                    'Sample Size': standards['sample_size']
                })

        return pd.DataFrame(trends)

    def get_top_athletes_progression(self, event: str, gender: str = 'men',
                                      top_n: int = 10) -> pd.DataFrame:
        """
        Get progression of top athletes over years and at major competitions.
        Shows the real picture of what it takes to win at the highest level.

        Returns DataFrame with:
        - Athlete name, country
        - Year-by-year best marks
        - Major competition performances
        """
        if self.data is None or len(self.data) == 0:
            self.load_scraped_data()

        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()

        df = self.data.copy()

        # Filter by event - use word-boundary matching to prevent partial number matches
        event_col = 'Event' if 'Event' in df.columns else 'event'
        if event_col in df.columns:
            event_normalized = re.sub(r'[^0-9a-z]', '', event.lower())
            event_lower = df[event_col].astype(str).str.lower().str.replace('-', '', regex=False).str.replace(' ', '', regex=False)
            df = df[event_lower == event_normalized]

        # Filter by gender
        gender_col = 'Gender' if 'Gender' in df.columns else 'gender'
        if gender_col in df.columns:
            gender_val = gender.lower()
            df = df[(df[gender_col].str.lower() == gender_val) |
                    (df[gender_col] == ('M' if gender_val == 'men' else 'F'))]

        if len(df) == 0:
            return pd.DataFrame()

        # Parse marks
        mark_col = 'Mark' if 'Mark' in df.columns else 'result'
        is_field = self.is_field_event(event)

        if is_field:
            df['ParsedMark'] = df[mark_col].apply(self.parse_distance_to_meters)
        else:
            df['ParsedMark'] = df[mark_col].apply(self.parse_time_to_seconds)

        df = df.dropna(subset=['ParsedMark'])

        if len(df) == 0:
            return pd.DataFrame()

        # Get competitor column
        comp_col = 'Competitor' if 'Competitor' in df.columns else 'competitor'
        country_col = 'Country' if 'Country' in df.columns else 'nat'
        date_col = 'Date' if 'Date' in df.columns else 'date'

        # Extract year from date
        def extract_year(date_val):
            if pd.isna(date_val):
                return None
            try:
                date_str = str(date_val)
                # Try to find 4-digit year
                import re
                year_match = re.search(r'20\d{2}|19\d{2}', date_str)
                if year_match:
                    return int(year_match.group())
                # Try pandas datetime
                return pd.to_datetime(date_val).year
            except:
                return None

        if date_col in df.columns:
            df['Year'] = df[date_col].apply(extract_year)
        else:
            df['Year'] = datetime.now().year

        # Get top athletes based on best ever mark
        if comp_col in df.columns:
            if is_field:
                athlete_best = df.groupby(comp_col)['ParsedMark'].max().reset_index()
                athlete_best = athlete_best.sort_values('ParsedMark', ascending=False).head(top_n)
            else:
                athlete_best = df.groupby(comp_col)['ParsedMark'].min().reset_index()
                athlete_best = athlete_best.sort_values('ParsedMark', ascending=True).head(top_n)

            top_athletes = athlete_best[comp_col].tolist()
        else:
            return pd.DataFrame()

        # Get progression for each top athlete
        results = []
        for athlete in top_athletes:
            athlete_df = df[df[comp_col] == athlete]
            if len(athlete_df) == 0:
                continue

            # Get country
            country = athlete_df[country_col].iloc[0] if country_col in athlete_df.columns else 'N/A'

            # Best mark
            if is_field:
                best = athlete_df['ParsedMark'].max()
            else:
                best = athlete_df['ParsedMark'].min()

            # Year-by-year progression
            yearly = {}
            for year in sorted(athlete_df['Year'].dropna().unique()):
                year_df = athlete_df[athlete_df['Year'] == year]
                if is_field:
                    yearly[int(year)] = self.format_mark(year_df['ParsedMark'].max(), event)
                else:
                    yearly[int(year)] = self.format_mark(year_df['ParsedMark'].min(), event)

            results.append({
                'Athlete': athlete,
                'Country': country,
                'Best': self.format_mark(best, event),
                'Best_Raw': best,
                'Performances': len(athlete_df),
                'Years_Active': len(yearly),
                **{f'Y{y}': m for y, m in yearly.items()}
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values('Best_Raw', ascending=not is_field)
        return result_df

    def get_major_comp_winning_marks(self, event: str, gender: str = 'men') -> pd.DataFrame:
        """
        Get winning marks at major competitions (Olympics, Worlds, etc.)
        over the years to show what it takes to win medals.
        """
        if self.data is None or len(self.data) == 0:
            self.load_scraped_data()

        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()

        df = self.data.copy()

        # Major competition keywords
        major_comps = [
            'Olympic', 'World Championships', 'World Athletics',
            'Asian Games', 'Commonwealth Games', 'Diamond League'
        ]

        # Filter by event - use word-boundary matching to prevent partial number matches
        event_col = 'Event' if 'Event' in df.columns else 'event'
        if event_col in df.columns:
            event_normalized = re.sub(r'[^0-9a-z]', '', event.lower())
            event_lower = df[event_col].astype(str).str.lower().str.replace('-', '', regex=False).str.replace(' ', '', regex=False)
            df = df[event_lower == event_normalized]

        # Filter by gender
        gender_col = 'Gender' if 'Gender' in df.columns else 'gender'
        if gender_col in df.columns:
            gender_val = gender.lower()
            df = df[(df[gender_col].str.lower() == gender_val) |
                    (df[gender_col] == ('M' if gender_val == 'men' else 'F'))]

        # Filter for major competitions
        venue_col = 'Venue' if 'Venue' in df.columns else 'venue'
        if venue_col in df.columns:
            mask = df[venue_col].str.contains('|'.join(major_comps), case=False, na=False)
            df = df[mask]

        if len(df) == 0:
            return pd.DataFrame()

        # Parse marks and get top performances
        mark_col = 'Mark' if 'Mark' in df.columns else 'result'
        is_field = self.is_field_event(event)

        if is_field:
            df['ParsedMark'] = df[mark_col].apply(self.parse_distance_to_meters)
            df = df.dropna(subset=['ParsedMark'])
            df = df.sort_values('ParsedMark', ascending=False)
        else:
            df['ParsedMark'] = df[mark_col].apply(self.parse_time_to_seconds)
            df = df.dropna(subset=['ParsedMark'])
            df = df.sort_values('ParsedMark', ascending=True)

        # Format output
        comp_col = 'Competitor' if 'Competitor' in df.columns else 'competitor'
        country_col = 'Country' if 'Country' in df.columns else 'nat'
        date_col = 'Date' if 'Date' in df.columns else 'date'

        result = df.head(20)[[comp_col, country_col, mark_col, venue_col, date_col]].copy()
        result.columns = ['Athlete', 'Country', 'Mark', 'Competition', 'Date']
        result['Mark_Formatted'] = result['Mark']

        return result

    def save_to_database(self, db_path: str = None):
        """Save analysis to SQLite database for dashboard integration."""
        if db_path is None:
            db_path = os.path.join(self.sql_dir, 'what_it_takes_to_win.db')

        conn = sqlite3.connect(db_path)

        # Save full report for both genders
        for gender in ['men', 'women']:
            report = self.generate_what_it_takes_report(gender)
            if len(report) > 0:
                report.to_sql(f'standards_{gender}', conn, if_exists='replace', index=False)

        # Save year-over-year trends
        trends_data = []
        for event in self.get_available_events():
            for gender in ['men', 'women']:
                event_trends = self.get_year_over_year_trends(event, gender)
                if len(event_trends) > 0:
                    trends_data.append(event_trends)

        if trends_data:
            all_trends = pd.concat(trends_data, ignore_index=True)
            all_trends.to_sql('standards_trends', conn, if_exists='replace', index=False)

        # Save metadata
        pd.DataFrame([{
            'last_updated': datetime.now().isoformat(),
            'years_covered': str(self.get_available_years()),
            'events_count': len(self.get_available_events()),
            'total_records': len(self.data) if self.data is not None else 0
        }]).to_sql('metadata', conn, if_exists='replace', index=False)

        conn.close()
        print(f"Analysis saved to {db_path}")


def main():
    """Run the What It Takes to Win analysis."""
    analyzer = WhatItTakesToWin()

    # Load data
    analyzer.load_scraped_data()

    if analyzer.data is None or len(analyzer.data) == 0:
        print("No data available. Please run the scraper first.")
        return

    print(f"\nAvailable Events: {len(analyzer.get_available_events())}")
    print(f"Available Years: {analyzer.get_available_years()}")

    # Generate and display report
    print("\n" + "=" * 80)
    print("WHAT IT TAKES TO WIN - Men's Standards (2024-2025)")
    print("=" * 80)

    report = analyzer.generate_what_it_takes_report('men')
    if len(report) > 0:
        print(report[['Event', 'Gold Standard', 'Silver Standard', 'Bronze Standard',
                      'Final Standard (8th)', 'Sample Size']].to_string(index=False))

    # Save to database
    analyzer.save_to_database()

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

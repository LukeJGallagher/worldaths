"""
Update KSA Athlete Profiles from Scraped World Athletics Data
Syncs the latest performance data from db_cleaned.csv to athlete profiles database.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'world_athletics_scraperv2', 'data')
PROFILES_DB = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')

# Event name mapping (scraped format to display format)
EVENT_MAPPING = {
    '100-metres': '100m',
    '200-metres': '200m',
    '400-metres': '400m',
    '800-metres': '800m',
    '1500-metres': '1500m',
    '5000-metres': '5000m',
    '10000-metres': '10000m',
    '110-metres-hurdles': '110mH',
    '100-metres-hurdles': '100mH',
    '400-metres-hurdles': '400mH',
    '3000-metres-steeplechase': '3000mSC',
    'high-jump': 'High Jump',
    'long-jump': 'Long Jump',
    'triple-jump': 'Triple Jump',
    'pole-vault': 'Pole Vault',
    'shot-put': 'Shot Put',
    'discus-throw': 'Discus',
    'javelin-throw': 'Javelin',
    'hammer-throw': 'Hammer',
    'decathlon': 'Decathlon',
    'heptathlon': 'Heptathlon',
    'marathon': 'Marathon',
    '400-metres-short-track': '400m (Indoor)',
    '800-metres-short-track': '800m (Indoor)',
    '1500-metres-short-track': '1500m (Indoor)',
    '5000-metres-short-track': '5000m (Indoor)',
}


def normalize_event_name(event: str) -> str:
    """Convert scraped event name to display format."""
    return EVENT_MAPPING.get(event, event)


def parse_time_to_seconds(time_str: str) -> Optional[float]:
    """Convert time string to seconds for comparison."""
    if pd.isna(time_str) or time_str == '':
        return None

    time_str = str(time_str).strip()

    # Handle DNF, DNS, DQ, etc.
    if any(x in time_str.upper() for x in ['DNF', 'DNS', 'DQ', 'NM', '-']):
        return None

    try:
        # Handle hours:minutes:seconds format (marathon)
        if time_str.count(':') == 2:
            parts = time_str.split(':')
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

        # Handle minutes:seconds format
        elif ':' in time_str:
            parts = time_str.split(':')
            return float(parts[0]) * 60 + float(parts[1])

        # Handle seconds only (sprints) or distance (field events)
        else:
            return float(time_str)

    except (ValueError, IndexError):
        return None


def is_field_event(event: str) -> bool:
    """Check if event is a field event (higher is better)."""
    field_keywords = ['jump', 'vault', 'put', 'throw', 'discus', 'javelin', 'hammer', 'decathlon', 'heptathlon']
    return any(kw in event.lower() for kw in field_keywords)


def get_better_mark(mark1: str, mark2: str, event: str) -> str:
    """Return the better mark based on event type."""
    val1 = parse_time_to_seconds(mark1)
    val2 = parse_time_to_seconds(mark2)

    if val1 is None:
        return mark2
    if val2 is None:
        return mark1

    if is_field_event(event):
        return mark1 if val1 > val2 else mark2
    else:
        return mark1 if val1 < val2 else mark2


def load_scraped_data() -> pd.DataFrame:
    """Load the cleaned scraped data."""
    db_cleaned_path = os.path.join(DATA_DIR, 'db_cleaned.csv')

    if os.path.exists(db_cleaned_path):
        df = pd.read_csv(db_cleaned_path)
        print(f"Loaded {len(df)} records from cleaned database")
        return df
    else:
        print(f"No data found at {db_cleaned_path}")
        return pd.DataFrame()


def ensure_tables_exist(conn: sqlite3.Connection):
    """Ensure all required tables exist with proper schema."""
    cursor = conn.cursor()

    # Check and add missing columns to ksa_athletes
    try:
        cursor.execute("ALTER TABLE ksa_athletes ADD COLUMN world_athletics_id TEXT")
    except:
        pass

    try:
        cursor.execute("ALTER TABLE ksa_athletes ADD COLUMN profile_url TEXT")
    except:
        pass

    try:
        cursor.execute("ALTER TABLE ksa_athletes ADD COLUMN updated_at TEXT")
    except:
        pass

    try:
        cursor.execute("ALTER TABLE ksa_athletes ADD COLUMN best_score REAL")
    except:
        pass

    try:
        cursor.execute("ALTER TABLE ksa_athletes ADD COLUMN best_world_rank INTEGER")
    except:
        pass

    # Add scraped_results table for detailed performance tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id TEXT,
            event_name TEXT,
            mark TEXT,
            result_score REAL,
            world_rank INTEGER,
            competition_date TEXT,
            venue TEXT,
            wind TEXT,
            gender TEXT,
            environment TEXT,
            year INTEGER,
            scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(athlete_id, event_name, mark, competition_date)
        )
    """)

    conn.commit()


def update_athletes_from_scraped(df: pd.DataFrame, conn: sqlite3.Connection):
    """Update athlete profiles from scraped data."""
    cursor = conn.cursor()

    # Filter KSA athletes
    ksa_df = df[df['Nat'] == 'KSA'].copy()
    print(f"\nProcessing {len(ksa_df)} KSA performance records")
    print(f"Unique athletes: {ksa_df['Competitor'].nunique()}")

    # Get best performance per athlete per event
    best_performances = ksa_df.groupby(['Competitor', 'Event']).agg({
        'Mark': 'first',
        'Rank': 'min',
        'ResultScore': 'max',
        'DOB': 'first',
        'CompetitorURL': 'first',
        'Date': 'first',
        'Venue': 'first',
        'Wind': 'first',
        'Gender': 'first',
        'Environment': 'first'
    }).reset_index()

    # Track stats
    athletes_added = 0
    athletes_updated = 0
    results_added = 0
    pbs_updated = 0

    # Process each athlete
    unique_athletes = best_performances.groupby('Competitor').agg({
        'ResultScore': 'max',
        'Rank': 'min',
        'DOB': 'first',
        'CompetitorURL': 'first',
        'Event': 'first',  # Primary event (best scored)
        'Gender': 'first'
    }).reset_index()

    for _, athlete_row in unique_athletes.iterrows():
        athlete_name = athlete_row['Competitor']
        dob = athlete_row['DOB']
        profile_url = athlete_row['CompetitorURL']
        primary_event = normalize_event_name(athlete_row['Event'])
        best_score = athlete_row['ResultScore']
        best_rank = int(athlete_row['Rank']) if pd.notna(athlete_row['Rank']) else None
        gender = 'men' if athlete_row['Gender'] == 'men' else 'women'

        # Extract athlete ID from URL
        athlete_id = None
        if pd.notna(profile_url):
            match = re.search(r'-(\d+)$', str(profile_url))
            if match:
                athlete_id = match.group(1)

        # Create standardized athlete_id if not from URL
        if not athlete_id:
            athlete_id = athlete_name.lower().replace(' ', '_').replace("'", "").replace('-', '_')

        # Check if athlete exists
        cursor.execute("SELECT athlete_id, best_score FROM ksa_athletes WHERE full_name = ?", (athlete_name,))
        existing = cursor.fetchone()

        if existing:
            existing_id = existing[0]
            existing_score = existing[1] or 0

            # Update if we have better data
            if best_score and (best_score > existing_score):
                cursor.execute("""
                    UPDATE ksa_athletes SET
                        date_of_birth = COALESCE(?, date_of_birth),
                        primary_event = ?,
                        profile_url = COALESCE(?, profile_url),
                        world_athletics_id = COALESCE(?, world_athletics_id),
                        best_score = ?,
                        best_world_rank = ?,
                        status = 'active',
                        updated_at = ?
                    WHERE full_name = ?
                """, (dob, primary_event, profile_url, athlete_id, best_score, best_rank,
                      datetime.now().isoformat(), athlete_name))
                athletes_updated += 1

            athlete_id = existing_id
        else:
            # Check if athlete_id already exists (different name, same ID)
            cursor.execute("SELECT athlete_id FROM ksa_athletes WHERE athlete_id = ?", (athlete_id,))
            id_exists = cursor.fetchone()

            if id_exists:
                # Generate unique ID
                athlete_id = f"{athlete_id}_{len(athlete_name)}"

            # Generate profile image URL
            profile_img = f"https://ui-avatars.com/api/?name={athlete_name.replace(' ', '+')}&background=007167&color=fff&size=128"

            # Insert new athlete
            try:
                cursor.execute("""
                    INSERT INTO ksa_athletes
                    (athlete_id, full_name, gender, date_of_birth, primary_event,
                     profile_image_url, country_code, status, profile_url, world_athletics_id,
                     best_score, best_world_rank, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'KSA', 'active', ?, ?, ?, ?, ?, ?)
                """, (athlete_id, athlete_name, gender, dob, primary_event, profile_img,
                      profile_url, athlete_id, best_score, best_rank,
                      datetime.now().isoformat(), datetime.now().isoformat()))
                athletes_added += 1
            except sqlite3.IntegrityError:
                # Update existing record instead
                cursor.execute("""
                    UPDATE ksa_athletes SET
                        date_of_birth = COALESCE(?, date_of_birth),
                        primary_event = ?,
                        best_score = COALESCE(?, best_score),
                        best_world_rank = COALESCE(?, best_world_rank),
                        status = 'active',
                        updated_at = ?
                    WHERE athlete_id = ?
                """, (dob, primary_event, best_score, best_rank,
                      datetime.now().isoformat(), athlete_id))
                athletes_updated += 1

        # Add all performance records for this athlete
        athlete_perfs = best_performances[best_performances['Competitor'] == athlete_name]

        for _, perf in athlete_perfs.iterrows():
            event = perf['Event']
            display_event = normalize_event_name(event)

            # Extract year from date
            year = None
            if pd.notna(perf['Date']):
                date_str = str(perf['Date'])
                if len(date_str) >= 4:
                    year_match = re.search(r'(\d{4})', date_str)
                    if year_match:
                        year = int(year_match.group(1))

            # Insert scraped result
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO scraped_results
                    (athlete_id, event_name, mark, result_score, world_rank,
                     competition_date, venue, wind, gender, environment, year)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (athlete_id, display_event, perf['Mark'], perf['ResultScore'],
                      int(perf['Rank']) if pd.notna(perf['Rank']) else None,
                      perf['Date'], perf['Venue'], perf['Wind'], gender,
                      perf['Environment'], year))
                results_added += 1
            except sqlite3.IntegrityError:
                pass

            # Update rankings
            cursor.execute("""
                INSERT OR REPLACE INTO athlete_rankings
                (athlete_id, event_name, world_rank, ranking_score, rank_date)
                VALUES (?, ?, ?, ?, ?)
            """, (athlete_id, display_event,
                  int(perf['Rank']) if pd.notna(perf['Rank']) else None,
                  perf['ResultScore'],
                  datetime.now().strftime('%Y-%m-%d')))

            # Update PBs
            cursor.execute("""
                SELECT pb_result FROM athlete_pbs
                WHERE athlete_id = ? AND event_name = ?
            """, (athlete_id, display_event))
            existing_pb = cursor.fetchone()

            current_mark = perf['Mark']
            if existing_pb:
                existing_mark = existing_pb[0]
                better = get_better_mark(current_mark, existing_mark, event)
                if better == current_mark and current_mark != existing_mark:
                    cursor.execute("""
                        UPDATE athlete_pbs SET
                            pb_result = ?,
                            pb_date = ?,
                            pb_venue = ?
                        WHERE athlete_id = ? AND event_name = ?
                    """, (current_mark, perf['Date'], perf['Venue'], athlete_id, display_event))
                    pbs_updated += 1
            else:
                cursor.execute("""
                    INSERT INTO athlete_pbs
                    (athlete_id, event_name, pb_result, pb_date, pb_venue, is_indoor)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (athlete_id, display_event, current_mark, perf['Date'],
                      perf['Venue'], 1 if 'indoor' in str(perf['Environment']).lower() else 0))
                pbs_updated += 1

    conn.commit()

    print(f"\n=== Update Summary ===")
    print(f"Athletes added: {athletes_added}")
    print(f"Athletes updated: {athletes_updated}")
    print(f"Results added: {results_added}")
    print(f"PBs added/updated: {pbs_updated}")


def calculate_progression(conn: sqlite3.Connection):
    """Calculate year-over-year progression for each athlete."""
    cursor = conn.cursor()

    print("\nCalculating athlete progression...")

    # Clear existing progression
    cursor.execute("DELETE FROM athlete_progression")

    # Calculate progression from scraped results (using existing schema columns)
    cursor.execute("""
        INSERT INTO athlete_progression (athlete_id, event_name, year, best_result)
        SELECT
            athlete_id,
            event_name,
            year,
            mark as best_result
        FROM scraped_results
        WHERE year IS NOT NULL
        GROUP BY athlete_id, event_name, year
        ORDER BY athlete_id, event_name, year
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM athlete_progression")
    count = cursor.fetchone()[0]
    print(f"Added {count} progression records")


def print_summary(conn: sqlite3.Connection):
    """Print summary of database contents."""
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("KSA ATHLETE PROFILES - DATABASE SUMMARY")
    print("=" * 60)

    # Table counts
    tables = ['ksa_athletes', 'athlete_rankings', 'athlete_pbs', 'scraped_results', 'athlete_progression']
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"{table}: {count} rows")
        except:
            print(f"{table}: table not found")

    # Top athletes by score
    print("\n--- Top 15 KSA Athletes by World Athletics Score ---")
    cursor.execute("""
        SELECT full_name, primary_event, best_score, best_world_rank
        FROM ksa_athletes
        WHERE best_score IS NOT NULL
        ORDER BY best_score DESC
        LIMIT 15
    """)
    for row in cursor.fetchall():
        rank_str = f"#{row[3]}" if row[3] else "N/A"
        print(f"  {row[0][:35]:35} | {row[1]:15} | Score: {row[2]:4.0f} | Rank: {rank_str}")

    # Events covered
    print("\n--- Events with KSA Athletes ---")
    cursor.execute("""
        SELECT event_name, COUNT(DISTINCT athlete_id) as athletes, MAX(ranking_score) as best_score
        FROM athlete_rankings
        GROUP BY event_name
        ORDER BY best_score DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:20} | {row[1]} athletes | Best: {row[2]:.0f}")

    # Recent PBs
    print("\n--- Recent Personal Bests (2024-2025) ---")
    cursor.execute("""
        SELECT a.full_name, p.event_name, p.pb_result, p.pb_date
        FROM athlete_pbs p
        JOIN ksa_athletes a ON p.athlete_id = a.athlete_id
        WHERE p.pb_date LIKE '%2024%' OR p.pb_date LIKE '%2025%'
        ORDER BY p.pb_date DESC
        LIMIT 15
    """)
    for row in cursor.fetchall():
        print(f"  {row[0][:30]:30} | {row[1]:15} | {row[2]:10} | {row[3]}")


def main():
    """Main function to update KSA athlete profiles."""
    print("=" * 60)
    print("Updating KSA Athlete Profiles from Scraped Data")
    print("=" * 60)

    # Load scraped data
    df = load_scraped_data()
    if df.empty:
        print("No data to process. Exiting.")
        return

    # Connect to profiles database
    conn = sqlite3.connect(PROFILES_DB)

    # Ensure tables exist
    ensure_tables_exist(conn)

    # Update athletes
    update_athletes_from_scraped(df, conn)

    # Calculate progression
    calculate_progression(conn)

    # Print summary
    print_summary(conn)

    conn.close()
    print("\n" + "=" * 60)
    print("Profile update complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

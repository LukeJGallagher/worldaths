"""
Populate KSA Athlete Profiles with Real Data
Extracts data from rankings database and World Athletics website
"""

import sqlite3
import pandas as pd
import os
import re
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
PROFILES_DB = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')

# Major games categorization
MAJOR_GAMES = {
    'Olympic': ['Olympic Games', 'XXXIII Olympic', 'XXXII Olympic', 'XXXI Olympic'],
    'World Championships': ['World Athletics Championships', 'World Championships'],
    'World U20': ['World Athletics U20', 'World U20 Championships'],
    'Asian Games': ['Asian Games'],
    'Asian Championships': ['Asian Athletics Championships', 'Asian Indoor'],
    'West Asian': ['West Asian Championships'],
    'Arab Championships': ['Arab Athletics', 'Arab U18', 'Arab U20', 'Arab U23'],
    'GCC': ['GCC Youth Games', 'GCC Championships']
}

def get_major_game_category(competition_name):
    """Categorize competition into major games type."""
    for category, keywords in MAJOR_GAMES.items():
        for keyword in keywords:
            if keyword.lower() in competition_name.lower():
                return category
    return 'Other'


def populate_from_rankings():
    """Populate athlete profiles from rankings database."""
    print("=" * 60)
    print("Populating KSA Athlete Profiles from Rankings")
    print("=" * 60)

    # Connect to databases
    profiles_conn = sqlite3.connect(PROFILES_DB)
    profiles_cursor = profiles_conn.cursor()

    rankings_db = os.path.join(SQL_DIR, 'rankings_men_all_events.db')
    rankings_conn = sqlite3.connect(rankings_db)

    # Get KSA athletes from rankings (use DISTINCT on Name to avoid duplicates)
    df = pd.read_sql("""
        SELECT Name, DOB, [Profile URL], Event, Score, Rank
        FROM rankings_men_all_events
        WHERE Country='KSA'
        GROUP BY Name, Event
    """, rankings_conn)

    print(f"Found {len(df)} KSA athlete records in rankings")

    # Group by athlete to get unique athletes with their best event
    athletes = df.groupby('Name').agg({
        'DOB': 'first',
        'Profile URL': 'first',
        'Event': lambda x: x.iloc[0],  # Primary event (first/best ranked)
        'Score': 'max',
        'Rank': 'min'
    }).reset_index()

    print(f"Processing {len(athletes)} unique athletes")

    for _, row in athletes.iterrows():
        athlete_name = row['Name']
        profile_url = row['Profile URL']
        dob = row['DOB']
        primary_event = row['Event']
        score = row['Score']

        # Extract athlete ID from URL
        athlete_id = None
        if profile_url:
            match = re.search(r'-(\d+)$', profile_url)
            if match:
                athlete_id = match.group(1)

        # Create standardized athlete_id if not from URL
        if not athlete_id:
            athlete_id = athlete_name.lower().replace(' ', '_').replace("'", "").replace('-', '_')

        # Check if athlete exists
        profiles_cursor.execute("SELECT athlete_id FROM ksa_athletes WHERE full_name = ?", (athlete_name,))
        existing = profiles_cursor.fetchone()

        if existing:
            # Update existing athlete
            profiles_cursor.execute("""
                UPDATE ksa_athletes SET
                    date_of_birth = ?,
                    primary_event = ?,
                    profile_url = ?,
                    world_athletics_id = ?,
                    updated_at = ?
                WHERE full_name = ?
            """, (dob, primary_event, profile_url, athlete_id, datetime.now().isoformat(), athlete_name))
            print(f"  * Updated: {athlete_name} ({primary_event})")
            athlete_id = existing[0]  # Use existing athlete_id
        else:
            # Insert new athlete
            profiles_cursor.execute("""
                INSERT INTO ksa_athletes (athlete_id, full_name, gender, date_of_birth, primary_event, profile_url, world_athletics_id, updated_at)
                VALUES (?, ?, 'men', ?, ?, ?, ?, ?)
            """, (athlete_id, athlete_name, dob, primary_event, profile_url, athlete_id, datetime.now().isoformat()))
            print(f"  + Added: {athlete_name} ({primary_event})")

        # Add ranking record
        profiles_cursor.execute("""
            INSERT OR REPLACE INTO athlete_rankings
            (athlete_id, event_name, world_rank, ranking_score, rank_date)
            VALUES (?, ?, ?, ?, ?)
        """, (athlete_id, primary_event, int(row['Rank']) if pd.notna(row['Rank']) else None, score, datetime.now().strftime('%Y-%m-%d')))

    profiles_conn.commit()
    print(f"\nUpdated {len(athletes)} athletes in profiles database")

    rankings_conn.close()
    profiles_conn.close()


def populate_results_from_modal():
    """Populate athlete results from KSA modal results database."""
    print("\n" + "=" * 60)
    print("Populating Results from Modal Results Database")
    print("=" * 60)

    profiles_conn = sqlite3.connect(PROFILES_DB)
    profiles_cursor = profiles_conn.cursor()

    modal_db = os.path.join(SQL_DIR, 'ksa_modal_results_men.db')
    modal_conn = sqlite3.connect(modal_db)

    # Get all results
    df = pd.read_sql("SELECT * FROM ksa_modal_results_men", modal_conn)
    print(f"Found {len(df)} performance records")

    # Get athlete mapping
    profiles_cursor.execute("SELECT athlete_id, full_name FROM ksa_athletes")
    athlete_map = {row[1].upper(): row[0] for row in profiles_cursor.fetchall()}

    results_added = 0
    pbs_added = 0

    for _, row in df.iterrows():
        athlete_name = row['Athlete']
        if not athlete_name:
            continue

        # Find athlete ID
        athlete_id = athlete_map.get(athlete_name.upper())
        if not athlete_id:
            # Try partial match
            for name, aid in athlete_map.items():
                if name in athlete_name.upper() or athlete_name.upper() in name:
                    athlete_id = aid
                    break

        if not athlete_id:
            continue

        event = row.get('Event Type', '')
        result = row.get('Result', '')
        competition = row.get('Competition', '')
        date = row.get('Date', '')
        place = row.get('Pl.', '')
        round_type = row.get('Type', '')
        wind = row.get('Wind', '')

        # Get major game category
        game_category = get_major_game_category(competition)

        # Insert result (check for existing first)
        try:
            # Convert place to integer if possible
            place_int = None
            if place:
                match = re.match(r'(\d+)', str(place))
                if match:
                    place_int = int(match.group(1))

            # Check if this result already exists
            profiles_cursor.execute("""
                SELECT result_id FROM athlete_results
                WHERE athlete_id = ? AND event_name = ? AND result_value = ? AND competition_name = ? AND round = ?
            """, (athlete_id, event, result, competition, round_type))

            if profiles_cursor.fetchone() is None:
                profiles_cursor.execute("""
                    INSERT INTO athlete_results
                    (athlete_id, event_name, result_value, competition_name, competition_date, place, round, wind, game_category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (athlete_id, event, result, competition, date, place_int, round_type, wind, game_category))
                results_added += 1
        except sqlite3.IntegrityError:
            pass  # Duplicate

    profiles_conn.commit()
    print(f"Added {results_added} results to athlete_results")

    # Calculate PBs
    print("\nCalculating Personal Bests...")
    profiles_cursor.execute("""
        INSERT OR REPLACE INTO athlete_pbs (athlete_id, event_name, pb_result, pb_date, pb_venue)
        SELECT
            athlete_id,
            event_name,
            MIN(result_value) as pb_result,
            competition_date,
            competition_name
        FROM athlete_results
        WHERE result_value IS NOT NULL AND result_value != ''
        GROUP BY athlete_id, event_name
    """)
    profiles_conn.commit()

    profiles_cursor.execute("SELECT COUNT(*) FROM athlete_pbs")
    pb_count = profiles_cursor.fetchone()[0]
    print(f"Calculated {pb_count} personal bests")

    modal_conn.close()
    profiles_conn.close()


def add_missing_columns():
    """Add any missing columns to the database."""
    conn = sqlite3.connect(PROFILES_DB)
    cursor = conn.cursor()

    # Check and add columns to ksa_athletes
    try:
        cursor.execute("ALTER TABLE ksa_athletes ADD COLUMN world_athletics_id TEXT")
        print("Added world_athletics_id column")
    except:
        pass

    try:
        cursor.execute("ALTER TABLE ksa_athletes ADD COLUMN profile_url TEXT")
        print("Added profile_url column")
    except:
        pass

    try:
        cursor.execute("ALTER TABLE ksa_athletes ADD COLUMN updated_at TEXT")
        print("Added updated_at column")
    except:
        pass

    # Add game_category to athlete_results
    try:
        cursor.execute("ALTER TABLE athlete_results ADD COLUMN game_category TEXT")
        print("Added game_category column to athlete_results")
    except:
        pass

    conn.commit()
    conn.close()


def print_summary():
    """Print summary of populated data."""
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    conn = sqlite3.connect(PROFILES_DB)
    cursor = conn.cursor()

    tables = ['ksa_athletes', 'athlete_rankings', 'athlete_results', 'athlete_pbs']

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} rows")

    # Show major games breakdown
    print("\n--- Results by Major Games Category ---")
    cursor.execute("""
        SELECT game_category, COUNT(*) as count
        FROM athlete_results
        WHERE game_category IS NOT NULL
        GROUP BY game_category
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} results")

    # Show top athletes by results
    print("\n--- Top Athletes by Number of Results ---")
    cursor.execute("""
        SELECT a.full_name, a.primary_event, COUNT(r.result_id) as result_count
        FROM ksa_athletes a
        LEFT JOIN athlete_results r ON a.athlete_id = r.athlete_id
        GROUP BY a.athlete_id
        ORDER BY result_count DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]} ({row[1]}): {row[2]} results")

    conn.close()


if __name__ == '__main__':
    # First add any missing columns
    add_missing_columns()

    # Populate from rankings
    populate_from_rankings()

    # Populate results from modal results
    populate_results_from_modal()

    # Print summary
    print_summary()

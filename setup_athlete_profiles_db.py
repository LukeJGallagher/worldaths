"""
Setup script for KSA Athlete Profiles Database
Creates SQLite tables for athlete profiles, rankings breakdown, and benchmark results
"""

import sqlite3
import os
from datetime import datetime

# Database path
SQL_DIR = os.path.join(os.path.dirname(__file__), 'SQL')
os.makedirs(SQL_DIR, exist_ok=True)

DB_PATH = os.path.join(SQL_DIR, 'ksa_athlete_profiles.db')

def create_tables():
    """Create all tables for the athlete profiles system."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Master list of KSA athletes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ksa_athletes (
            athlete_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            gender TEXT,
            date_of_birth TEXT,
            primary_event TEXT,
            profile_image_url TEXT,
            country_code TEXT DEFAULT 'KSA',
            status TEXT DEFAULT 'active',
            last_scraped TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Athlete rankings - current WPA ranking snapshots
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS athlete_rankings (
            ranking_id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id TEXT,
            event_name TEXT,
            world_rank INTEGER,
            ranking_score REAL,
            average_score REAL,
            rank_date TEXT,
            scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (athlete_id) REFERENCES ksa_athletes(athlete_id)
        )
    ''')

    # 3. Ranking breakdown - the 5 results that make up ranking score
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ranking_breakdown (
            breakdown_id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id TEXT,
            event_name TEXT,
            competition_date TEXT,
            competition_name TEXT,
            result_value TEXT,
            result_score REAL,
            placing INTEGER,
            place_score REAL,
            performance_score REAL,
            competition_category TEXT,
            rank_date TEXT,
            FOREIGN KEY (athlete_id) REFERENCES ksa_athletes(athlete_id)
        )
    ''')

    # 4. All competition results (historical)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS athlete_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id TEXT,
            competition_id TEXT,
            competition_name TEXT,
            competition_date TEXT,
            competition_type TEXT,
            event_name TEXT,
            round TEXT,
            result_value TEXT,
            wind TEXT,
            place INTEGER,
            points REAL,
            venue TEXT,
            country TEXT DEFAULT 'KSA',
            FOREIGN KEY (athlete_id) REFERENCES ksa_athletes(athlete_id)
        )
    ''')

    # 5. Personal bests by event
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS athlete_pbs (
            pb_id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id TEXT,
            event_name TEXT,
            pb_result TEXT,
            pb_date TEXT,
            pb_venue TEXT,
            is_indoor INTEGER DEFAULT 0,
            FOREIGN KEY (athlete_id) REFERENCES ksa_athletes(athlete_id)
        )
    ''')

    # 6. Year-by-year progression
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS athlete_progression (
            progression_id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id TEXT,
            event_name TEXT,
            year INTEGER,
            best_result TEXT,
            indoor_best TEXT,
            FOREIGN KEY (athlete_id) REFERENCES ksa_athletes(athlete_id)
        )
    ''')

    # 7. Benchmark results - top world performances for comparison
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_results (
            benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
            competition_name TEXT,
            competition_date TEXT,
            event_name TEXT,
            round TEXT,
            place INTEGER,
            athlete_name TEXT,
            country TEXT,
            result_value TEXT,
            scraped_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rankings_athlete ON athlete_rankings(athlete_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rankings_event ON athlete_rankings(event_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_athlete ON athlete_results(athlete_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_competition ON athlete_results(competition_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_breakdown_athlete ON ranking_breakdown(athlete_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmark_event ON benchmark_results(event_name)')

    conn.commit()
    conn.close()

    print(f"Database created at: {DB_PATH}")
    print("Tables created:")
    print("  - ksa_athletes (master athlete list)")
    print("  - athlete_rankings (WPA ranking snapshots)")
    print("  - ranking_breakdown (5 best results breakdown)")
    print("  - athlete_results (all competition results)")
    print("  - athlete_pbs (personal bests)")
    print("  - athlete_progression (year-by-year)")
    print("  - benchmark_results (world benchmarks)")


def seed_initial_athletes():
    """Seed database with known KSA athletes from existing data."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get unique athletes from existing ksa_modal_results_men
    men_db = os.path.join(SQL_DIR, 'ksa_modal_results_men.db')
    women_db = os.path.join(SQL_DIR, 'ksa_modal_results_women.db')

    athletes_to_add = []

    # Read from men's results
    if os.path.exists(men_db):
        men_conn = sqlite3.connect(men_db)
        men_cursor = men_conn.cursor()
        men_cursor.execute('SELECT DISTINCT Athlete FROM ksa_modal_results_men')
        for row in men_cursor.fetchall():
            athletes_to_add.append((row[0], 'men'))
        men_conn.close()
        print(f"Found {len(athletes_to_add)} male athletes")

    # Read from women's results
    women_count = 0
    if os.path.exists(women_db):
        try:
            women_conn = sqlite3.connect(women_db)
            women_cursor = women_conn.cursor()
            women_cursor.execute('SELECT DISTINCT Athlete FROM ksa_modal_results_women')
            for row in women_cursor.fetchall():
                athletes_to_add.append((row[0], 'women'))
                women_count += 1
            women_conn.close()
            print(f"Found {women_count} female athletes")
        except sqlite3.OperationalError:
            print("Women's database exists but has no data yet")

    # Insert athletes (without IDs for now - will be populated by scraper)
    for name, gender in athletes_to_add:
        cursor.execute('''
            INSERT OR IGNORE INTO ksa_athletes (athlete_id, full_name, gender, status)
            VALUES (?, ?, ?, ?)
        ''', (name.replace(' ', '_').lower(), name, gender, 'active'))

    conn.commit()
    conn.close()

    print(f"Seeded {len(athletes_to_add)} athletes into database")


def verify_database():
    """Verify database setup and show table counts."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n=== Database Verification ===")

    tables = [
        'ksa_athletes',
        'athlete_rankings',
        'ranking_breakdown',
        'athlete_results',
        'athlete_pbs',
        'athlete_progression',
        'benchmark_results'
    ]

    for table in tables:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} rows")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")

    conn.close()


if __name__ == '__main__':
    print("Setting up KSA Athlete Profiles Database...")
    print("=" * 50)

    create_tables()
    seed_initial_athletes()
    verify_database()

    print("\n Setup complete!")

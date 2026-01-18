"""
Data Connector - Dual-mode DuckDB wrapper for Azure Blob / Local Parquet files.

Automatically detects AZURE_STORAGE_CONNECTION_STRING to choose mode:
- Azure mode: Queries Parquet files directly from Azure Blob Storage
- Local mode: Queries Parquet files from local Data/parquet folder

Usage:
    from data_connector import query, get_ksa_athletes, get_competitors, get_benchmarks

    # Simple query
    df = query("SELECT * FROM master WHERE country_code = 'KSA' LIMIT 10")

    # Helper functions
    ksa_athletes = get_ksa_athletes()
    competitors = get_competitors(event='100m', gender='Men', top_n=20)
"""
import os
import duckdb
import pandas as pd
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()

# Configuration
CONN_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "personal-data"
AZURE_FOLDER = "athletics"

# Local paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_PARQUET_DIR = os.path.join(BASE_DIR, "Data", "parquet")

# Data mode
DATA_MODE = "azure" if CONN_STRING else "local"


def get_data_mode() -> str:
    """Return current data mode: 'azure' or 'local'."""
    return DATA_MODE


def get_base_path() -> str:
    """Get base path for Parquet files based on data mode."""
    if DATA_MODE == "azure":
        return f"az://{CONTAINER_NAME}/{AZURE_FOLDER}"
    else:
        return LOCAL_PARQUET_DIR


def get_connection():
    """Create DuckDB connection with Azure extension if needed."""
    con = duckdb.connect()

    if DATA_MODE == "azure":
        con.execute("INSTALL azure; LOAD azure;")
        con.execute(f"""
            CREATE SECRET azure_secret (
                TYPE AZURE,
                CONNECTION_STRING '{CONN_STRING}'
            );
        """)

    return con


def query(sql: str, params: dict = None) -> pd.DataFrame:
    """
    Execute SQL query against Parquet files.

    Table aliases available:
    - master: All 2.3M scraped records
    - ksa_profiles: 152 KSA athlete profiles
    - benchmarks: Championship standards

    Example:
        df = query("SELECT * FROM master WHERE event = 'Long Jump' LIMIT 100")
    """
    base_path = get_base_path()

    # Replace table aliases with actual paths
    sql = sql.replace("FROM master", f"FROM '{base_path}/master.parquet'")
    sql = sql.replace("FROM ksa_profiles", f"FROM '{base_path}/ksa_profiles.parquet'")
    sql = sql.replace("FROM benchmarks", f"FROM '{base_path}/benchmarks.parquet'")

    # Also handle JOIN clauses
    sql = sql.replace("JOIN master", f"JOIN '{base_path}/master.parquet'")
    sql = sql.replace("JOIN ksa_profiles", f"JOIN '{base_path}/ksa_profiles.parquet'")
    sql = sql.replace("JOIN benchmarks", f"JOIN '{base_path}/benchmarks.parquet'")

    con = get_connection()
    try:
        if params:
            result = con.execute(sql, params).fetchdf()
        else:
            result = con.execute(sql).fetchdf()
        return result
    finally:
        con.close()


def get_ksa_athletes() -> pd.DataFrame:
    """Get all KSA athlete profiles."""
    return query("SELECT * FROM ksa_profiles ORDER BY best_world_rank ASC NULLS LAST")


def get_athlete_by_id(athlete_id: str) -> pd.DataFrame:
    """Get single athlete profile by ID."""
    return query(f"SELECT * FROM ksa_profiles WHERE athlete_id = '{athlete_id}'")


def get_athlete_results(athlete_name: str, event: str = None) -> pd.DataFrame:
    """Get competition results for an athlete."""
    sql = f"""
        SELECT * FROM master
        WHERE competitor ILIKE '%{athlete_name}%'
    """
    if event:
        sql += f" AND event = '{event}'"
    sql += " ORDER BY date DESC"
    return query(sql)


def get_competitors(event: str, gender: str = 'Men', top_n: int = 20,
                    season: int = None, region: str = None) -> pd.DataFrame:
    """
    Get top competitors for an event.

    Args:
        event: Event name (e.g., '100m', 'Long Jump')
        gender: 'Men' or 'Women'
        top_n: Number of top athletes to return
        season: Filter by year (e.g., 2025)
        region: Filter by region/country code
    """
    # Normalize event name for matching
    event_patterns = {
        '100m': '100-metres',
        '200m': '200-metres',
        '400m': '400-metres',
        '800m': '800-metres',
        '1500m': '1500-metres',
        '5000m': '5000-metres',
        '10000m': '10000-metres',
        '110m Hurdles': '110-metres-hurdles',
        '100m Hurdles': '100-metres-hurdles',
        '400m Hurdles': '400-metres-hurdles',
        '3000m Steeplechase': '3000-metres-steeplechase',
        'High Jump': 'high-jump',
        'Pole Vault': 'pole-vault',
        'Long Jump': 'long-jump',
        'Triple Jump': 'triple-jump',
        'Shot Put': 'shot-put',
        'Discus Throw': 'discus-throw',
        'Hammer Throw': 'hammer-throw',
        'Javelin Throw': 'javelin-throw',
        'Decathlon': 'decathlon',
        'Heptathlon': 'heptathlon',
        'Marathon': 'marathon',
    }

    event_filter = event_patterns.get(event, event.lower().replace(' ', '-'))

    sql = f"""
        SELECT
            competitor as athlete_name,
            nat as country_code,
            result,
            result_numeric,
            date,
            venue,
            event,
            rank
        FROM master
        WHERE event ILIKE '%{event_filter}%'
    """

    if gender:
        gender_code = 'M' if gender == 'Men' else 'F'
        sql += f" AND (gender = '{gender}' OR gender = '{gender_code}')"

    if season:
        sql += f" AND EXTRACT(YEAR FROM date) = {season}"

    if region:
        sql += f" AND nat = '{region}'"

    # Get best result per athlete
    sql = f"""
        WITH ranked AS (
            {sql}
        )
        SELECT DISTINCT ON (athlete_name) *
        FROM ranked
        ORDER BY athlete_name, result_numeric ASC
        LIMIT {top_n}
    """

    # DuckDB doesn't support DISTINCT ON, use different approach
    sql = f"""
        SELECT
            competitor as athlete_name,
            nat as country_code,
            MIN(result_numeric) as best_result,
            FIRST(result) as result,
            FIRST(date) as date,
            FIRST(venue) as venue,
            event,
            MIN(rank) as best_rank
        FROM master
        WHERE event ILIKE '%{event_filter}%'
    """

    if gender:
        gender_code = 'M' if gender == 'Men' else 'F'
        sql += f" AND (gender = '{gender}' OR gender = '{gender_code}')"

    if season:
        sql += f" AND TRY_CAST(date AS DATE) IS NOT NULL AND EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) = {season}"

    if region:
        sql += f" AND nat = '{region}'"

    sql += f"""
        GROUP BY competitor, nat, event
        ORDER BY best_result ASC
        LIMIT {top_n}
    """

    return query(sql)


def get_head_to_head(athlete1: str, athlete2: str, event: str = None) -> pd.DataFrame:
    """
    Get head-to-head results between two athletes.

    Returns competitions where both athletes competed.
    """
    sql = f"""
        WITH a1 AS (
            SELECT date, venue, result, result_numeric, event
            FROM master
            WHERE competitor ILIKE '%{athlete1}%'
        ),
        a2 AS (
            SELECT date, venue, result, result_numeric, event
            FROM master
            WHERE competitor ILIKE '%{athlete2}%'
        )
        SELECT
            a1.date,
            a1.venue,
            a1.event,
            '{athlete1}' as athlete1,
            a1.result as result1,
            a1.result_numeric as numeric1,
            '{athlete2}' as athlete2,
            a2.result as result2,
            a2.result_numeric as numeric2,
            CASE
                WHEN a1.result_numeric < a2.result_numeric THEN '{athlete1}'
                WHEN a2.result_numeric < a1.result_numeric THEN '{athlete2}'
                ELSE 'TIE'
            END as winner
        FROM a1
        JOIN a2 ON a1.date = a2.date AND a1.venue = a2.venue AND a1.event = a2.event
    """

    if event:
        sql += f" WHERE a1.event ILIKE '%{event}%'"

    sql += " ORDER BY a1.date DESC"

    return query(sql)


def get_benchmarks(event: str = None, gender: str = None) -> pd.DataFrame:
    """Get championship benchmark standards."""
    sql = "SELECT * FROM benchmarks WHERE source_table != 'metadata'"

    if event:
        sql += f" AND Event = '{event}'"
    if gender:
        sql += f" AND Gender = '{gender}'"

    return query(sql)


def get_gap_analysis(athlete_name: str, event: str, gender: str = 'Men') -> dict:
    """
    Calculate gap between athlete's PB and various targets.

    Returns dict with gaps to:
    - World #1, #10, #50
    - Championship standards (Gold, Final)
    - Asian/Regional leaders
    """
    # Get athlete's best result
    athlete_results = get_athlete_results(athlete_name, event)
    if athlete_results.empty:
        return {"error": f"No results found for {athlete_name} in {event}"}

    athlete_pb = athlete_results['result_numeric'].min()

    # Get top competitors
    top_10 = get_competitors(event, gender, top_n=10)
    top_50 = get_competitors(event, gender, top_n=50)

    # Get benchmarks
    benchmarks = get_benchmarks(event, gender)

    result = {
        "athlete": athlete_name,
        "event": event,
        "pb": athlete_pb,
        "gaps": {}
    }

    if not top_10.empty:
        result["gaps"]["world_1"] = top_10.iloc[0]['best_result'] - athlete_pb
        result["gaps"]["world_10"] = top_10.iloc[-1]['best_result'] - athlete_pb if len(top_10) >= 10 else None

    if not top_50.empty and len(top_50) >= 50:
        result["gaps"]["world_50"] = top_50.iloc[-1]['best_result'] - athlete_pb

    if not benchmarks.empty:
        for _, row in benchmarks.iterrows():
            if 'Gold_Raw' in row and pd.notna(row['Gold_Raw']):
                result["gaps"]["gold_standard"] = row['Gold_Raw'] - athlete_pb
            if 'Final Standard (8th)' in row and pd.notna(row['Final Standard (8th)']):
                result["gaps"]["final_standard"] = row['Final Standard (8th)'] - athlete_pb

    return result


def get_event_list() -> list:
    """Get list of unique events in the database."""
    df = query("SELECT DISTINCT event FROM master ORDER BY event")
    return df['event'].tolist()


def get_country_list() -> list:
    """Get list of unique countries in the database."""
    df = query("SELECT DISTINCT nat FROM master WHERE nat IS NOT NULL ORDER BY nat")
    return df['nat'].tolist()


def test_connection() -> dict:
    """Test database connectivity and return diagnostic info."""
    result = {
        "mode": DATA_MODE,
        "azure_configured": bool(CONN_STRING),
        "base_path": get_base_path(),
        "connection_test": "not_run",
        "tables": {}
    }

    try:
        # Test each table
        for table in ["master", "ksa_profiles", "benchmarks"]:
            try:
                df = query(f"SELECT COUNT(*) as cnt FROM {table}")
                result["tables"][table] = int(df['cnt'].iloc[0])
            except Exception as e:
                result["tables"][table] = f"ERROR: {str(e)}"

        result["connection_test"] = "success"
    except Exception as e:
        result["connection_test"] = "failed"
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("DATA CONNECTOR TEST")
    print("=" * 60)

    info = test_connection()
    print(f"\nData Mode: {info['mode']}")
    print(f"Azure Configured: {info['azure_configured']}")
    print(f"Base Path: {info['base_path']}")
    print(f"Connection Test: {info['connection_test']}")

    print("\nTable Row Counts:")
    for table, count in info['tables'].items():
        print(f"  - {table}: {count:,}" if isinstance(count, int) else f"  - {table}: {count}")

    print("\n" + "=" * 60)
    print("SAMPLE QUERIES")
    print("=" * 60)

    # Test KSA athletes
    print("\nTop 5 KSA Athletes by World Rank:")
    ksa = get_ksa_athletes().head(5)
    print(ksa[['full_name', 'primary_event', 'best_world_rank']].to_string(index=False))

    # Test competitors
    print("\nTop 10 Men's 100m (2025):")
    competitors = get_competitors('100m', 'Men', top_n=10, season=2025)
    if not competitors.empty:
        print(competitors[['athlete_name', 'country_code', 'result', 'best_rank']].to_string(index=False))
    else:
        print("  No data for 2025 season yet")

    print("\n" + "=" * 60)

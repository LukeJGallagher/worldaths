"""
Data Connector - Dual-mode DuckDB wrapper for Azure Blob / Local Parquet files.

Automatically detects AZURE_STORAGE_CONNECTION_STRING to choose mode:
- Azure mode: Downloads Parquet from Azure Blob, queries with DuckDB
- Local mode: Queries Parquet files from local Data/parquet folder

Performance optimizations:
- TTL-based caching (1 hour default) to avoid repeated downloads
- Singleton pattern for master data (only download once per session)
- Streamlit cache integration when available

Usage:
    from data_connector import query, get_ksa_athletes, get_competitors, get_benchmarks

    # Simple query
    df = query("SELECT * FROM master WHERE country_code = 'KSA' LIMIT 10")

    # Helper functions
    ksa_athletes = get_ksa_athletes()
    competitors = get_competitors(event='100m', gender='Men', top_n=20)
"""
import os
import time
import duckdb
import pandas as pd
from dotenv import load_dotenv
from functools import lru_cache
from io import BytesIO

# Load environment variables
load_dotenv()

# Configuration - lazy loaded
CONTAINER_NAME = "personal-data"
AZURE_FOLDER = "athletics"

# Cache TTL in seconds (1 hour default)
CACHE_TTL_SECONDS = 3600

# Force local mode for faster development (set FORCE_LOCAL_MODE=true in .env)
FORCE_LOCAL_MODE = os.getenv("FORCE_LOCAL_MODE", "").lower() in ("true", "1", "yes")

# Local paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_PARQUET_DIR = os.path.join(BASE_DIR, "Data", "parquet")

# Azure blob client (for direct download)
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

# Cache for downloaded parquet data with timestamps
# Format: {'blob_name': {'data': DataFrame, 'timestamp': float}}
_PARQUET_CACHE = {}

# Try to use Streamlit caching for persistence across reruns
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Connection string cache (lazy-loaded)
_CONN_STRING_CHECKED = False
_CONN_STRING = None

# Persistent DuckDB connection (session-scoped, with pre-registered views)
_PERSISTENT_CON = None


def _get_connection_string():
    """Get Azure connection string from env or Streamlit secrets (lazy-loaded)."""
    global _CONN_STRING, _CONN_STRING_CHECKED

    # Only use cache if we previously found a valid connection string
    if _CONN_STRING_CHECKED and _CONN_STRING:
        return _CONN_STRING

    # Try environment variable first
    conn_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    # Try Streamlit secrets if not in environment
    if not conn_str:
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                try:
                    conn_str = st.secrets.get('AZURE_STORAGE_CONNECTION_STRING')
                except (FileNotFoundError, KeyError, AttributeError):
                    # No secrets.toml file or key not found - gracefully skip
                    pass
        except ImportError:
            pass

    # Cache only if we found something
    if conn_str:
        _CONN_STRING = conn_str
        _CONN_STRING_CHECKED = True

    return conn_str


def get_data_mode() -> str:
    """Return current data mode: 'azure' or 'local'.

    Set FORCE_LOCAL_MODE=true in .env for faster local development.
    """
    if FORCE_LOCAL_MODE:
        return "local"
    return "azure" if _get_connection_string() else "local"


def get_base_path() -> str:
    """Get base path for Parquet files based on data mode."""
    if get_data_mode() == "azure":
        return f"az://{CONTAINER_NAME}/{AZURE_FOLDER}"
    else:
        return LOCAL_PARQUET_DIR


def _download_blob_raw(blob_name: str, conn_str: str) -> bytes:
    """Download raw bytes from Azure - used by cached wrapper."""
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(CONTAINER_NAME)
    blob_path = f"{AZURE_FOLDER}/{blob_name}"
    blob_client = container_client.get_blob_client(blob_path)
    return blob_client.download_blob().readall()


# Streamlit-cached version (persists across reruns)
if STREAMLIT_AVAILABLE:
    @st.cache_data(ttl=3600, show_spinner=f"Loading data from Azure...")
    def _cached_download_parquet(blob_name: str) -> pd.DataFrame:
        """Streamlit-cached Azure download - persists across reruns."""
        conn_str = _get_connection_string()
        if not conn_str or not AZURE_SDK_AVAILABLE:
            return None
        try:
            data = _download_blob_raw(blob_name, conn_str)
            return pd.read_parquet(BytesIO(data))
        except Exception as e:
            print(f"Azure download error for {blob_name}: {e}")
            return None
else:
    _cached_download_parquet = None


def _download_parquet_from_azure(blob_name: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Download parquet file from Azure Blob Storage using SDK.

    Uses Streamlit cache (if available) or TTL-based caching.
    Data is downloaded once and cached for the session.

    Args:
        blob_name: Name of the parquet file to download
        force_refresh: If True, bypass cache and re-download

    Returns:
        DataFrame or None if download fails
    """
    # Use Streamlit cache if available (best for Streamlit apps)
    if STREAMLIT_AVAILABLE and _cached_download_parquet and not force_refresh:
        df = _cached_download_parquet(blob_name)
        if df is not None:
            return df.copy()

    # Fallback to manual cache for non-Streamlit or force_refresh
    global _PARQUET_CACHE
    current_time = time.time()

    # Check manual cache with TTL
    if not force_refresh and blob_name in _PARQUET_CACHE:
        cache_entry = _PARQUET_CACHE[blob_name]
        cache_age = current_time - cache_entry['timestamp']

        # Master.parquet is large - cache for entire session (no TTL check)
        if blob_name == 'master.parquet' or cache_age < CACHE_TTL_SECONDS:
            return cache_entry['data'].copy()

    conn_str = _get_connection_string()
    if not conn_str or not AZURE_SDK_AVAILABLE:
        return None

    try:
        data = _download_blob_raw(blob_name, conn_str)
        df = pd.read_parquet(BytesIO(data))

        # Cache with timestamp
        _PARQUET_CACHE[blob_name] = {
            'data': df,
            'timestamp': current_time
        }

        return df.copy()
    except Exception as e:
        print(f"Azure download error for {blob_name}: {e}")
        return None


def clear_cache(blob_name: str = None):
    """
    Clear the parquet cache and invalidate DuckDB connection.

    Args:
        blob_name: Specific blob to clear, or None to clear all
    """
    global _PARQUET_CACHE
    if blob_name:
        _PARQUET_CACHE.pop(blob_name, None)
    else:
        _PARQUET_CACHE.clear()
    # Invalidate DuckDB views so they pick up new data on next query
    close_connection()


def get_cache_status() -> dict:
    """Get cache status for debugging."""
    current_time = time.time()
    status = {}
    for name, entry in _PARQUET_CACHE.items():
        age_seconds = current_time - entry['timestamp']
        status[name] = {
            'rows': len(entry['data']),
            'age_seconds': round(age_seconds, 1),
            'age_minutes': round(age_seconds / 60, 1)
        }
    return status


def _get_persistent_connection():
    """Get or create a persistent DuckDB connection with pre-registered parquet views.

    Views registered: master, ksa_profiles, benchmarks, road_to_tokyo.
    Small files (ksa_profiles, benchmarks, road_to_tokyo) are pre-warmed.
    master.parquet (67MB) is registered but not pre-warmed (lazy loaded on first query).
    """
    global _PERSISTENT_CON
    if _PERSISTENT_CON is not None:
        try:
            _PERSISTENT_CON.execute("SELECT 1")
            return _PERSISTENT_CON
        except Exception:
            _PERSISTENT_CON = None

    con = duckdb.connect()
    base_path = LOCAL_PARQUET_DIR

    # Register parquet files as views for clean SQL (no string replacement needed)
    for name in ['master', 'ksa_profiles', 'benchmarks', 'road_to_tokyo']:
        parquet_path = os.path.join(base_path, f'{name}.parquet').replace('\\', '/')
        if os.path.exists(parquet_path):
            try:
                con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM '{parquet_path}'")
            except Exception as e:
                print(f"Warning: Could not register view '{name}': {e}")

    # Pre-warm small file metadata (forces DuckDB to read parquet headers)
    for name in ['ksa_profiles', 'benchmarks', 'road_to_tokyo']:
        try:
            con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        except Exception:
            pass

    _PERSISTENT_CON = con
    return con


def get_connection():
    """Get persistent DuckDB connection (backward compatible)."""
    return _get_persistent_connection()


def close_connection():
    """Close persistent DuckDB connection (call on cleanup)."""
    global _PERSISTENT_CON
    if _PERSISTENT_CON is not None:
        try:
            _PERSISTENT_CON.close()
        except Exception:
            pass
        _PERSISTENT_CON = None


def query(sql: str, params: dict = None) -> pd.DataFrame:
    """
    Execute SQL query against Parquet files via persistent DuckDB connection.

    Table names available (registered as views):
    - master: All 2.3M scraped records
    - ksa_profiles: 152 KSA athlete profiles
    - benchmarks: Championship standards
    - road_to_tokyo: Qualification tracking

    Example:
        df = query("SELECT * FROM master WHERE event = 'Long Jump' LIMIT 100")
    """
    con = _get_persistent_connection()
    try:
        if params:
            return con.execute(sql, params).fetchdf()
        else:
            return con.execute(sql).fetchdf()
    except duckdb.ConnectionException:
        # Connection went stale — recreate and retry
        global _PERSISTENT_CON
        _PERSISTENT_CON = None
        con = _get_persistent_connection()
        if params:
            return con.execute(sql, params).fetchdf()
        else:
            return con.execute(sql).fetchdf()


def get_ksa_athletes() -> pd.DataFrame:
    """Get all KSA athlete profiles."""
    # Try direct Azure download first (avoids DuckDB SSL issues on Streamlit Cloud)
    if get_data_mode() == "azure":
        df = _download_parquet_from_azure("ksa_profiles.parquet")
        if df is not None:
            # Sort by best_world_rank
            if 'best_world_rank' in df.columns:
                df = df.sort_values('best_world_rank', na_position='last')
            return df

    # Fall back to DuckDB query (for local mode)
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


def get_rankings_data(gender: str = None, country: str = None, limit: int = None) -> pd.DataFrame:
    """
    Get rankings data from master.parquet.

    This provides similar data to the old SQLite rankings tables.
    Returns athlete rankings with event, country, result, etc.

    Args:
        gender: Filter by 'Men' or 'Women' (optional)
        country: Filter by country code like 'KSA' (optional)
        limit: Maximum rows to return (optional, for performance)
    """
    # Build SQL query with filters - avoid loading full dataset
    conditions = []

    if gender:
        gender_code = 'M' if gender == 'Men' else 'F'
        conditions.append(f"(gender = '{gender}' OR gender = '{gender_code}')")

    if country:
        conditions.append(f"(nat ILIKE '%{country}%' OR Country ILIKE '%{country}%')")

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    sql = f"SELECT * FROM master WHERE {where_clause}"

    if limit:
        sql += f" LIMIT {limit}"

    # Use DuckDB to query parquet directly (much faster than loading full file)
    return query(sql)


def get_ksa_rankings() -> pd.DataFrame:
    """
    Get KSA athlete rankings from master.parquet.

    Returns filtered view of master data for KSA athletes only,
    formatted similar to old SQLite rankings tables.
    """
    # Try direct Azure download first (avoids DuckDB SSL issues on Streamlit Cloud)
    if get_data_mode() == "azure":
        df = _download_parquet_from_azure("master.parquet")
        if df is not None and not df.empty:
            # Filter for KSA athletes
            mask = pd.Series([False] * len(df))
            for col in ['nat', 'Nat', 'Country', 'country']:
                if col in df.columns:
                    mask = mask | df[col].astype(str).str.upper().str.contains('KSA', na=False)
            df = df[mask]
            return df

    # Fall back to DuckDB query (for local mode)
    # Note: master.parquet only has 'nat' column, not 'Country'
    return query("SELECT * FROM master WHERE nat ILIKE '%KSA%'")


def get_benchmarks_data() -> pd.DataFrame:
    """
    Get benchmarks/qualification standards from benchmarks.parquet.
    """
    # Try direct Azure download first
    if get_data_mode() == "azure":
        df = _download_parquet_from_azure("benchmarks.parquet")
        if df is not None:
            # Filter out metadata rows
            if 'source_table' in df.columns:
                df = df[df['source_table'] != 'metadata']
            return df

    # Fall back to DuckDB for local mode
    return query("SELECT * FROM benchmarks WHERE source_table != 'metadata'")


def get_ksa_asian_games_candidates(min_results: int = 2) -> pd.DataFrame:
    """
    Dynamically discover all KSA athletes from the database,
    compute PB/SB per event, and return ranked results.

    Args:
        min_results: Minimum number of results to include athlete-event (default 2)

    Returns:
        DataFrame with: athlete_name, event, event_display, pb, sb,
        result_count, latest_date, is_field
    """
    import re as _re
    ksa_data = get_ksa_rankings()
    if ksa_data is None or ksa_data.empty:
        return pd.DataFrame()

    # Find column names dynamically
    athlete_col = next((c for c in ['competitor', 'Competitor', 'full_name'] if c in ksa_data.columns), None)
    event_col = next((c for c in ['event', 'Event'] if c in ksa_data.columns), None)
    result_col = next((c for c in ['result', 'Result', 'Mark'] if c in ksa_data.columns), None)
    date_col = next((c for c in ['date', 'Date'] if c in ksa_data.columns), None)

    if not all([athlete_col, event_col, result_col]):
        return pd.DataFrame()

    # Import parse helpers
    try:
        from what_it_takes_to_win import WhatItTakesToWin, format_event_name
        wittw = WhatItTakesToWin()
    except ImportError:
        return pd.DataFrame()

    # Drop rows without athlete or event
    ksa_data = ksa_data.dropna(subset=[athlete_col, event_col])

    # Parse year for SB calculation
    current_year = pd.Timestamp.now().year
    if date_col:
        ksa_data['_year'] = pd.to_datetime(ksa_data[date_col], errors='coerce').dt.year

    candidates = []
    for (athlete, event), group in ksa_data.groupby([athlete_col, event_col]):
        if len(group) < min_results:
            continue

        is_field = wittw.is_field_event(str(event))

        # Parse marks through WITTW methods (not result_numeric which has garbage)
        if is_field:
            group = group.copy()
            group['_parsed'] = group[result_col].apply(wittw.parse_distance_to_meters)
        else:
            group = group.copy()
            group['_parsed'] = group[result_col].apply(wittw.parse_time_to_seconds)

        valid = group.dropna(subset=['_parsed'])
        if valid.empty:
            continue

        # Filter outliers
        outlier_mask = WhatItTakesToWin.filter_outlier_marks(valid['_parsed'], is_field)
        valid = valid[outlier_mask]
        if valid.empty:
            continue

        # PB: best result
        pb = valid['_parsed'].max() if is_field else valid['_parsed'].min()

        # SB: best result from current year
        sb = pb
        if date_col and '_year' in valid.columns:
            current = valid[valid['_year'] >= current_year]
            if not current.empty:
                sb = current['_parsed'].max() if is_field else current['_parsed'].min()

        # Latest date
        latest_date = valid[date_col].max() if date_col and date_col in valid.columns else None

        candidates.append({
            'athlete_name': str(athlete),
            'event': str(event),
            'event_display': format_event_name(str(event)),
            'pb': pb,
            'sb': sb,
            'result_count': len(valid),
            'latest_date': latest_date,
            'is_field': is_field,
        })

    if not candidates:
        return pd.DataFrame()

    return pd.DataFrame(candidates)


def get_road_to_tokyo_data(federation: str = None) -> pd.DataFrame:
    """
    Get Road to Tokyo qualification data from road_to_tokyo.parquet.

    Args:
        federation: Filter by federation code like 'KSA' (optional)

    Returns:
        DataFrame with qualification tracking data including:
        - Event_Type, Actual_Event_Name
        - Federation, Athlete
        - Qualification_Status, Status, Details
    """
    # Try direct Azure download first
    if get_data_mode() == "azure":
        df = _download_parquet_from_azure("road_to_tokyo.parquet")
        if df is not None:
            if federation:
                df = df[df['Federation'].str.upper().str.contains(federation.upper(), na=False)]
            return df

    # Fall back to local parquet or DuckDB
    base_path = get_base_path()
    parquet_path = f"{base_path}/road_to_tokyo.parquet"
    try:
        con = get_connection()
        sql = f"SELECT * FROM '{parquet_path}'"
        if federation:
            sql += f" WHERE Federation ILIKE '%{federation}%'"
        return con.execute(sql).fetchdf()
    except Exception:
        return pd.DataFrame()


def get_major_championships_data() -> pd.DataFrame:
    """
    Get only major championship data from master.parquet using efficient DuckDB filtering.

    Uses date-aware city matching to identify championship records since venues
    are stored as stadium names (e.g., "Stade de France, Paris (FRA)") not
    championship names (e.g., "Paris 2024 Olympics").

    Returns:
        DataFrame with major championship records and a 'championship_type' column
    """
    # Date-aware city-year to championship mapping
    # Format: (city_pattern, year, championship_type)
    championship_city_years = [
        # Olympics
        ('paris', 2024, 'Olympic'),
        ('tokyo', 2021, 'Olympic'),
        ('rio', 2016, 'Olympic'),
        ('london', 2012, 'Olympic'),
        ('beijing', 2008, 'Olympic'),
        ('athens', 2004, 'Olympic'),
        ('sydney', 2000, 'Olympic'),
        # World Championships
        ('budapest', 2023, 'World Champs'),
        ('eugene', 2022, 'World Champs'),
        ('doha', 2019, 'World Champs'),
        ('london', 2017, 'World Champs'),
        ('beijing', 2015, 'World Champs'),
        ('moscow', 2013, 'World Champs'),
        ('daegu', 2011, 'World Champs'),
        ('berlin', 2009, 'World Champs'),
        ('osaka', 2007, 'World Champs'),
        # Asian Games
        ('hangzhou', 2023, 'Asian Games'),
        ('jakarta', 2018, 'Asian Games'),
        ('incheon', 2014, 'Asian Games'),
        ('guangzhou', 2010, 'Asian Games'),
    ]

    # Diamond League cities (recurring venues, no year constraint)
    diamond_league_cities = [
        'zurich', 'brussels', 'monaco', 'rome', 'stockholm', 'oslo',
        'lausanne', 'rabat', 'shanghai', 'birmingham', 'chorzow',
        'silesia', 'xiamen', 'suzhou'
    ]

    # Build SQL with date-aware matching
    city_year_conditions = []
    for city, year, champ in championship_city_years:
        city_year_conditions.append(
            f"(LOWER(venue) LIKE '%{city}%' AND EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) = {year})"
        )

    # Diamond League cities (any year from 2010+)
    dl_conditions = []
    for city in diamond_league_cities:
        dl_conditions.append(f"LOWER(venue) LIKE '%{city}%'")
    dl_clause = f"(({' OR '.join(dl_conditions)}) AND EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) >= 2010)"

    # Explicit keyword matches (for venues that DO contain championship names)
    keyword_conditions = [
        "LOWER(venue) LIKE '%olympic%'",
        "LOWER(venue) LIKE '%world championship%'",
        "LOWER(venue) LIKE '%asian games%'",
        "LOWER(venue) LIKE '%diamond league%'",
        "LOWER(venue) LIKE '%asian athletics%'",
        "LOWER(venue) LIKE '%arab%'"
    ]

    all_conditions = city_year_conditions + [dl_clause] + keyword_conditions
    where_clause = " OR ".join(all_conditions)

    # Build CASE statement dynamically from championship_city_years (DRY - single source of truth)
    case_branches = []
    for city, year, champ in championship_city_years:
        case_branches.append(
            f"WHEN (LOWER(venue) LIKE '%{city}%' AND EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) = {year}) THEN '{champ}'"
        )
    case_branches_sql = "\n                ".join(case_branches)

    # Diamond League cities for CASE
    dl_case_likes = " OR ".join(f"LOWER(venue) LIKE '%{c}%'" for c in diamond_league_cities[:6])

    sql = f"""
        SELECT *,
            CASE
                {case_branches_sql}
                WHEN ({dl_case_likes})
                    THEN 'Diamond League'
                WHEN LOWER(venue) LIKE '%olympic%' THEN 'Olympic'
                WHEN LOWER(venue) LIKE '%world championship%' THEN 'World Champs'
                WHEN LOWER(venue) LIKE '%asian games%' THEN 'Asian Games'
                WHEN LOWER(venue) LIKE '%diamond league%' THEN 'Diamond League'
                ELSE 'Other'
            END as championship_type
        FROM master
        WHERE {where_clause}
    """

    try:
        df = query(sql)
        if df is not None and not df.empty:
            print(f"Loaded {len(df):,} major championship records via DuckDB filter")
            return df
    except Exception as e:
        print(f"DuckDB championship query failed: {e}")

    # Fall back to downloading entire file and filtering (slower, more memory)
    if get_data_mode() == "azure":
        df = _download_parquet_from_azure("master.parquet")
        if df is not None and not df.empty:
            # Apply same filtering logic in pandas
            venue_col = 'venue' if 'venue' in df.columns else 'Venue'
            if venue_col in df.columns:
                df['_year'] = pd.to_datetime(df.get('date', df.get('Date')), errors='coerce').dt.year
                venue_lower = df[venue_col].fillna('').str.lower()

                # Track championship type for each row (needed for filtering later)
                df['championship_type'] = 'Other'

                # Assign championship types based on city+year matching
                for city, year, champ in championship_city_years:
                    city_year_mask = (venue_lower.str.contains(city, na=False)) & (df['_year'] == year)
                    df.loc[city_year_mask, 'championship_type'] = champ

                # Diamond League cities (any year from 2010+)
                for city in diamond_league_cities:
                    dl_mask = (venue_lower.str.contains(city, na=False)) & (df['_year'] >= 2010)
                    # Only set if not already assigned a more specific type
                    df.loc[dl_mask & (df['championship_type'] == 'Other'), 'championship_type'] = 'Diamond League'

                # Filter to only major championships (exclude 'Other')
                df = df[df['championship_type'] != 'Other'].drop(columns=['_year'], errors='ignore')
                print(f"Loaded {len(df):,} major championship records via Azure + filter")
                return df

    return pd.DataFrame()


def get_athlete_race_history(athlete_name: str, event: str = None, limit: int = 10) -> pd.DataFrame:
    """Get recent race history for an athlete with full context."""
    # Escape single quotes in name
    safe_name = athlete_name.replace("'", "''")
    sql = f"""
        SELECT date, event, result, result_numeric, venue, pos, environment, wind, resultscore
        FROM master
        WHERE competitor ILIKE '%{safe_name}%'
    """
    if event:
        safe_event = event.replace("'", "''")
        sql += f" AND event ILIKE '%{safe_event}%'"
    sql += f" ORDER BY date DESC LIMIT {limit}"
    try:
        return query(sql)
    except Exception:
        return pd.DataFrame()


def get_pb_details_per_event(athlete_name: str) -> pd.DataFrame:
    """Get PB, SB, PB date, PB venue for each event athlete competed in.
    Uses a single query with window functions (no N+1 loop)."""
    safe_name = athlete_name.replace("'", "''")
    current_year = pd.Timestamp.now().year
    sql = f"""
        WITH athlete_results AS (
            SELECT
                event,
                result,
                result_numeric,
                date,
                venue,
                environment,
                EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) as yr
            FROM master
            WHERE competitor ILIKE '%{safe_name}%'
                AND result_numeric IS NOT NULL
        ),
        event_stats AS (
            SELECT
                event,
                MIN(result_numeric) as pb_numeric,
                COUNT(*) as competitions,
                MIN(CASE WHEN yr >= {current_year} THEN result_numeric END) as sb_numeric,
                MIN(CASE WHEN LOWER(environment) = 'indoor' THEN result_numeric END) as indoor_pb,
                MIN(CASE WHEN LOWER(environment) = 'outdoor' THEN result_numeric END) as outdoor_pb
            FROM athlete_results
            GROUP BY event
        ),
        pb_rows AS (
            SELECT
                ar.event,
                ar.result as pb_result,
                ar.date as pb_date,
                ar.venue as pb_venue,
                ROW_NUMBER() OVER (PARTITION BY ar.event ORDER BY ar.date DESC) as rn
            FROM athlete_results ar
            JOIN event_stats es ON ar.event = es.event AND ar.result_numeric = es.pb_numeric
        )
        SELECT
            es.event,
            pr.pb_result,
            pr.pb_date,
            pr.pb_venue,
            es.pb_numeric,
            es.competitions,
            es.sb_numeric,
            es.indoor_pb,
            es.outdoor_pb
        FROM event_stats es
        LEFT JOIN pb_rows pr ON es.event = pr.event AND pr.rn = 1
        ORDER BY es.competitions DESC
    """
    try:
        df = query(sql)
        if df.empty:
            return df

        pb_details = []
        for _, row in df.iterrows():
            pb_details.append({
                'Event': row['event'],
                'PB': row['pb_result'] if pd.notna(row.get('pb_result')) else f"{row['pb_numeric']:.2f}",
                'PB Date': str(row['pb_date'])[:10] if pd.notna(row.get('pb_date')) else 'N/A',
                'PB Venue': str(row['pb_venue'])[:35] if pd.notna(row.get('pb_venue')) else 'N/A',
                'SB': f"{row['sb_numeric']:.2f}" if pd.notna(row.get('sb_numeric')) else '-',
                'Indoor PB': f"{row['indoor_pb']:.2f}" if pd.notna(row.get('indoor_pb')) else '-',
                'Outdoor PB': f"{row['outdoor_pb']:.2f}" if pd.notna(row.get('outdoor_pb')) else '-',
                'Competitions': int(row['competitions'])
            })

        return pd.DataFrame(pb_details)
    except Exception:
        return pd.DataFrame()


def get_year_by_year_progression(athlete_name: str, event: str) -> pd.DataFrame:
    """Get athlete's best mark per year for progression table."""
    safe_name = athlete_name.replace("'", "''")
    safe_event = event.replace("'", "''")
    sql = f"""
        SELECT
            EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) as year,
            MIN(result_numeric) as best_mark,
            COUNT(*) as competitions,
            AVG(result_numeric) as avg_mark
        FROM master
        WHERE competitor ILIKE '%{safe_name}%'
            AND event ILIKE '%{safe_event}%'
            AND result_numeric IS NOT NULL
            AND TRY_CAST(date AS DATE) IS NOT NULL
        GROUP BY EXTRACT(YEAR FROM TRY_CAST(date AS DATE))
        ORDER BY year DESC
    """
    try:
        return query(sql)
    except Exception:
        return pd.DataFrame()


def test_connection() -> dict:
    """Test database connectivity and return diagnostic info."""
    result = {
        "mode": get_data_mode(),
        "azure_configured": bool(_get_connection_string()),
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

"""
Convert existing CSV/DB data to Parquet format and upload to Azure Blob Storage.

Data sources for KSA Competitor Intelligence Dashboard:
1. master.parquet - 2.3M scraped records from db_cleaned.csv
2. ksa_profiles.parquet - 152 KSA athlete profiles from ksa_athlete_profiles.db
3. benchmarks.parquet - Championship standards from what_it_takes_to_win.db
"""
import os
import sys
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
SQL_DIR = os.path.join(BASE_DIR, "SQL")
SCRAPER_DATA = os.path.join(BASE_DIR, "world_athletics_scraperv2", "data")

# Source files
MASTER_CSV = os.path.join(SCRAPER_DATA, "db_cleaned.csv")
KSA_PROFILES_DB = os.path.join(SQL_DIR, "ksa_athlete_profiles.db")
BENCHMARKS_DB = os.path.join(SQL_DIR, "what_it_takes_to_win.db")

# Output Parquet files (local)
OUTPUT_DIR = os.path.join(DATA_DIR, "parquet")
MASTER_PARQUET = os.path.join(OUTPUT_DIR, "master.parquet")
KSA_PROFILES_PARQUET = os.path.join(OUTPUT_DIR, "ksa_profiles.parquet")
BENCHMARKS_PARQUET = os.path.join(OUTPUT_DIR, "benchmarks.parquet")

# Azure config
CONN_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "personal-data"
AZURE_FOLDER = "athletics"


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def convert_master_data():
    """Convert db_cleaned.csv to master.parquet."""
    print("\n" + "=" * 60)
    print("1. CONVERTING MASTER DATA (db_cleaned.csv)")
    print("=" * 60)

    if not os.path.exists(MASTER_CSV):
        print(f"   ERROR: {MASTER_CSV} not found")
        return False

    print(f"   Reading {MASTER_CSV}...")
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    print(f"   Loaded {len(df):,} records")

    # Show columns
    print(f"   Columns: {list(df.columns)}")

    # Standardize column names for the app
    column_mapping = {
        'athleteid': 'athlete_id',
        'athletename': 'athlete_name',
        'country': 'country_code',
        'nationality': 'country_code',
        'gender': 'gender',
        'event': 'event',
        'eventname': 'event',
        'result': 'result',
        'performance': 'result',
        'mark': 'result',
        'date': 'date',
        'competition': 'competition',
        'competitionname': 'competition',
        'venue': 'venue',
        'round': 'round',
        'position': 'position',
        'place': 'position',
        'wind': 'wind',
        'sb': 'season_best',
        'pb': 'personal_best',
    }

    # Rename columns (case-insensitive)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Parse result to numeric
    if 'result' in df.columns:
        df['result_numeric'] = df['result'].apply(parse_result_to_numeric)

    # Ensure date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Map gender (handle lowercase from source)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.strip().str.lower()
        df['gender'] = df['gender'].map({'m': 'Men', 'f': 'Women', 'men': 'Men', 'women': 'Women'})
        print(f"   Gender values: {df['gender'].value_counts().to_dict()}")

    print(f"   Final columns: {list(df.columns)}")
    print(f"   Sample data:")
    print(df.head(3).to_string())

    # Save as Parquet
    print(f"\n   Saving to {MASTER_PARQUET}...")
    df.to_parquet(MASTER_PARQUET, index=False, compression='snappy')

    file_size = os.path.getsize(MASTER_PARQUET) / (1024 * 1024)
    print(f"   Saved: {file_size:.2f} MB")
    return True


def convert_ksa_profiles():
    """Convert ksa_athlete_profiles.db to ksa_profiles.parquet."""
    print("\n" + "=" * 60)
    print("2. CONVERTING KSA PROFILES (ksa_athlete_profiles.db)")
    print("=" * 60)

    if not os.path.exists(KSA_PROFILES_DB):
        print(f"   ERROR: {KSA_PROFILES_DB} not found")
        return False

    print(f"   Reading {KSA_PROFILES_DB}...")
    conn = sqlite3.connect(KSA_PROFILES_DB)

    # Check available tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"   Tables: {list(tables['name'])}")

    # Read main athlete table
    df = pd.read_sql("SELECT * FROM ksa_athletes", conn)
    print(f"   Loaded {len(df):,} athletes")

    # Try to get PBs
    try:
        pbs = pd.read_sql("SELECT * FROM personal_bests", conn)
        print(f"   Loaded {len(pbs):,} personal bests")
        # Merge PB info
        if not pbs.empty:
            pb_pivot = pbs.pivot_table(index='athlete_id', columns='event', values='mark', aggfunc='first')
            pb_pivot.columns = [f"pb_{col}" for col in pb_pivot.columns]
            df = df.merge(pb_pivot, left_on='athlete_id', right_index=True, how='left')
    except Exception as e:
        print(f"   No personal_bests table: {e}")

    conn.close()

    print(f"   Final columns: {list(df.columns)}")
    print(f"   Sample data:")
    print(df.head(3).to_string())

    # Save as Parquet
    print(f"\n   Saving to {KSA_PROFILES_PARQUET}...")
    df.to_parquet(KSA_PROFILES_PARQUET, index=False, compression='snappy')

    file_size = os.path.getsize(KSA_PROFILES_PARQUET) / (1024 * 1024)
    print(f"   Saved: {file_size:.2f} MB")
    return True


def convert_benchmarks():
    """Convert what_it_takes_to_win.db to benchmarks.parquet."""
    print("\n" + "=" * 60)
    print("3. CONVERTING BENCHMARKS (what_it_takes_to_win.db)")
    print("=" * 60)

    if not os.path.exists(BENCHMARKS_DB):
        print(f"   ERROR: {BENCHMARKS_DB} not found")
        return False

    print(f"   Reading {BENCHMARKS_DB}...")
    conn = sqlite3.connect(BENCHMARKS_DB)

    # Check available tables
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"   Tables: {list(tables['name'])}")

    # Read all tables and combine
    dfs = []
    for table in tables['name']:
        try:
            df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
            df['source_table'] = table
            dfs.append(df)
            print(f"   - {table}: {len(df)} rows")
        except Exception as e:
            print(f"   - {table}: Error - {e}")

    conn.close()

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        print(f"   Combined: {len(df):,} rows")
    else:
        print("   No data found")
        return False

    print(f"   Final columns: {list(df.columns)}")

    # Save as Parquet
    print(f"\n   Saving to {BENCHMARKS_PARQUET}...")
    df.to_parquet(BENCHMARKS_PARQUET, index=False, compression='snappy')

    file_size = os.path.getsize(BENCHMARKS_PARQUET) / (1024 * 1024)
    print(f"   Saved: {file_size:.2f} MB")
    return True


def parse_result_to_numeric(value):
    """Parse result string to numeric (seconds for time, meters for distance)."""
    if pd.isna(value) or value == '':
        return None

    try:
        value = str(value).strip().upper()

        # Remove common suffixes
        for suffix in ['A', 'H', 'W', 'I', 'M', 'Q', 'R']:
            value = value.rstrip(suffix)

        value = value.strip()

        # Skip invalid results
        if value in ['DNF', 'DNS', 'DQ', 'NM', 'NH', '', '-']:
            return None

        # Time format (MM:SS.ms or HH:MM:SS.ms)
        if ':' in value:
            parts = value.split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

        # Plain number (distance/points)
        return float(value)

    except (ValueError, TypeError):
        return None


def upload_to_azure():
    """Upload Parquet files to Azure Blob Storage."""
    print("\n" + "=" * 60)
    print("4. UPLOADING TO AZURE BLOB STORAGE")
    print("=" * 60)

    if not CONN_STRING:
        print("   ERROR: AZURE_STORAGE_CONNECTION_STRING not set")
        return False

    try:
        from azure.storage.blob import BlobServiceClient

        blob_service = BlobServiceClient.from_connection_string(CONN_STRING)
        container_client = blob_service.get_container_client(CONTAINER_NAME)

        # Create backup of existing files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        files_to_upload = [
            (MASTER_PARQUET, f"{AZURE_FOLDER}/master.parquet"),
            (KSA_PROFILES_PARQUET, f"{AZURE_FOLDER}/ksa_profiles.parquet"),
            (BENCHMARKS_PARQUET, f"{AZURE_FOLDER}/benchmarks.parquet"),
        ]

        for local_path, blob_path in files_to_upload:
            if not os.path.exists(local_path):
                print(f"   SKIP: {local_path} does not exist")
                continue

            # Backup existing if present
            try:
                existing_blob = container_client.get_blob_client(blob_path)
                if existing_blob.exists():
                    backup_path = f"{AZURE_FOLDER}/backups/{os.path.basename(blob_path)}_{timestamp}"
                    backup_blob = container_client.get_blob_client(backup_path)
                    backup_blob.start_copy_from_url(existing_blob.url)
                    print(f"   Backed up: {blob_path} -> {backup_path}")
            except Exception as e:
                pass  # No existing file to backup

            # Upload new file
            blob_client = container_client.get_blob_client(blob_path)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            file_size = os.path.getsize(local_path) / (1024 * 1024)
            print(f"   Uploaded: {blob_path} ({file_size:.2f} MB)")

        return True

    except ImportError:
        print("   ERROR: azure-storage-blob not installed")
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def verify_azure_upload():
    """Verify uploaded files using DuckDB."""
    print("\n" + "=" * 60)
    print("5. VERIFYING AZURE DATA WITH DUCKDB")
    print("=" * 60)

    try:
        import duckdb

        con = duckdb.connect()
        con.execute("INSTALL azure; LOAD azure;")
        con.execute(f"""
            CREATE SECRET azure_secret (
                TYPE AZURE,
                CONNECTION_STRING '{CONN_STRING}'
            );
        """)

        files = [
            f"az://{CONTAINER_NAME}/{AZURE_FOLDER}/master.parquet",
            f"az://{CONTAINER_NAME}/{AZURE_FOLDER}/ksa_profiles.parquet",
            f"az://{CONTAINER_NAME}/{AZURE_FOLDER}/benchmarks.parquet",
        ]

        for file_path in files:
            try:
                result = con.execute(f"SELECT COUNT(*) as cnt FROM '{file_path}'").fetchone()
                print(f"   {os.path.basename(file_path)}: {result[0]:,} rows")
            except Exception as e:
                print(f"   {os.path.basename(file_path)}: ERROR - {e}")

        con.close()
        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("PARQUET CONVERSION & AZURE UPLOAD")
    print("=" * 60)
    print(f"Started: {datetime.now()}")

    ensure_output_dir()

    # Convert each data source
    success_master = convert_master_data()
    success_profiles = convert_ksa_profiles()
    success_benchmarks = convert_benchmarks()

    # Upload to Azure
    if any([success_master, success_profiles, success_benchmarks]):
        upload_to_azure()
        verify_azure_upload()

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now()}")

    # Summary
    print("\nLocal Parquet files:")
    for f in [MASTER_PARQUET, KSA_PROFILES_PARQUET, BENCHMARKS_PARQUET]:
        if os.path.exists(f):
            size = os.path.getsize(f) / (1024 * 1024)
            print(f"   {os.path.basename(f)}: {size:.2f} MB")
        else:
            print(f"   {os.path.basename(f)}: NOT CREATED")

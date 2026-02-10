"""
Memory-Efficient Historical Data Merge using DuckDB

Uses DuckDB's out-of-core processing to merge large historical CSV files
with the existing master.parquet without loading everything into memory.

Usage:
    python merge_historical_duckdb.py                    # Merge and upload
    python merge_historical_duckdb.py --dry-run          # Preview only
    python merge_historical_duckdb.py --local-only       # No Azure upload
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:
    logger.error("DuckDB not installed. Run: pip install duckdb")
    sys.exit(1)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_PARQUET = os.path.join(BASE_DIR, 'Data', 'parquet', 'master.parquet')
CSV_GLOB_PATTERN = os.path.join(BASE_DIR, 'world_athletics_scraperv2', 'data', '*', '*', '*.csv')
OUTPUT_PARQUET = os.path.join(BASE_DIR, 'Data', 'parquet', 'master_merged.parquet')
BACKUP_DIR = os.path.join(BASE_DIR, 'Data', 'parquet', 'backups')


def load_env():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, '.env'))
    except ImportError:
        pass


def get_azure_connection():
    """Get Azure Blob Storage connection string."""
    conn_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not conn_str:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'AZURE_STORAGE_CONNECTION_STRING' in st.secrets:
                conn_str = st.secrets['AZURE_STORAGE_CONNECTION_STRING']
        except:
            pass
    return conn_str


def analyze_current_data(con):
    """Analyze current master.parquet coverage."""
    logger.info("Analyzing current master.parquet...")

    result = con.execute(f"""
        SELECT
            COUNT(*) as total_rows,
            MIN(EXTRACT(YEAR FROM TRY_CAST(date AS DATE))) as min_year,
            MAX(EXTRACT(YEAR FROM TRY_CAST(date AS DATE))) as max_year
        FROM read_parquet('{MASTER_PARQUET.replace(os.sep, "/")}')
    """).fetchone()

    logger.info(f"  Current master: {result[0]:,} rows, years {result[1]}-{result[2]}")

    # Year distribution
    year_dist = con.execute(f"""
        SELECT
            EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) as year,
            COUNT(*) as count
        FROM read_parquet('{MASTER_PARQUET.replace(os.sep, "/")}')
        GROUP BY EXTRACT(YEAR FROM TRY_CAST(date AS DATE))
        HAVING year IS NOT NULL
        ORDER BY year
    """).fetchall()

    return result[0], year_dist


def analyze_csv_data(con, csv_pattern):
    """Analyze scraped CSV data coverage."""
    logger.info("Analyzing scraped CSV files...")

    # Convert Windows path to forward slashes for DuckDB
    csv_pattern_unix = csv_pattern.replace(os.sep, "/")

    try:
        result = con.execute(f"""
            SELECT
                COUNT(*) as total_rows
            FROM read_csv_auto('{csv_pattern_unix}',
                               ignore_errors=true,
                               union_by_name=true)
        """).fetchone()

        logger.info(f"  Scraped CSVs: {result[0]:,} rows")
        return result[0]
    except Exception as e:
        logger.error(f"Error reading CSVs: {e}")
        return 0


def merge_data(con, csv_pattern, output_path, dry_run=False):
    """Merge master.parquet with CSV files using DuckDB SQL."""
    logger.info("Starting merge operation...")

    csv_pattern_unix = csv_pattern.replace(os.sep, "/")
    master_path_unix = MASTER_PARQUET.replace(os.sep, "/")
    output_path_unix = output_path.replace(os.sep, "/")

    # Master.parquet schema: rank, result, wind, competitor, competitorurl, dob, nat, pos,
    #                        "unnamed: 8", venue, date, resultscore, age, event, environment,
    #                        gender, result_numeric, year
    merge_query = f"""
        WITH csv_data AS (
            SELECT
                TRY_CAST("Rank" AS BIGINT) as rank,
                TRY_CAST("Mark" AS DOUBLE) as result,
                TRY_CAST("Wind" AS DOUBLE) as wind,
                "Competitor" as competitor,
                "CompetitorURL" as competitorurl,
                "DOB" as dob,
                "Nat" as nat,
                "Pos" as pos,
                CAST(NULL AS DOUBLE) as "unnamed: 8",
                "Venue" as venue,
                TRY_CAST("Date" AS TIMESTAMP) as date,
                TRY_CAST("ResultScore" AS BIGINT) as resultscore,
                TRY_CAST("Age" AS DOUBLE) as age,
                "Event" as event,
                "Environment" as environment,
                "Gender" as gender,
                TRY_CAST("Mark" AS DOUBLE) as result_numeric,
                EXTRACT(YEAR FROM TRY_CAST("Date" AS DATE)) as year
            FROM read_csv_auto('{csv_pattern_unix}',
                               ignore_errors=true,
                               union_by_name=true,
                               header=true)
            WHERE "Competitor" IS NOT NULL
        ),
        master_data AS (
            SELECT
                rank, result, wind, competitor, competitorurl, dob, nat, pos,
                "unnamed: 8", venue, date, resultscore, age, event, environment,
                gender, result_numeric, year
            FROM read_parquet('{master_path_unix}')
        ),
        combined AS (
            SELECT * FROM master_data
            UNION ALL
            SELECT * FROM csv_data
        ),
        deduplicated AS (
            SELECT DISTINCT ON (competitor, event, date, result, venue) *
            FROM combined
            ORDER BY competitor, event, date, result, venue, rank
        )
        SELECT * FROM deduplicated
        ORDER BY date DESC NULLS LAST
    """

    # Count results first
    count_query = f"""
        WITH csv_data AS (
            SELECT
                "Competitor" as competitor,
                "Mark" as result,
                "Date" as date,
                "Venue" as venue,
                "Event" as event
            FROM read_csv_auto('{csv_pattern_unix}',
                               ignore_errors=true,
                               union_by_name=true,
                               header=true)
            WHERE "Competitor" IS NOT NULL
        ),
        master_data AS (
            SELECT competitor, result, date, venue, event
            FROM read_parquet('{master_path_unix}')
        ),
        combined AS (
            SELECT * FROM master_data
            UNION ALL
            SELECT * FROM csv_data
        ),
        deduplicated AS (
            SELECT DISTINCT competitor, event, date, result, venue
            FROM combined
        )
        SELECT COUNT(*) FROM deduplicated
    """

    try:
        # Get expected row count
        result_count = con.execute(count_query).fetchone()[0]
        logger.info(f"  Expected merged rows: {result_count:,}")

        if dry_run:
            logger.info("[DRY RUN] Would write merged data to parquet")
            return result_count

        # Execute merge and write to parquet
        logger.info("  Writing merged data to parquet...")
        con.execute(f"""
            COPY ({merge_query})
            TO '{output_path_unix}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # Verify the output
        verify_count = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{output_path_unix}')
        """).fetchone()[0]

        logger.info(f"  Verified output: {verify_count:,} rows")
        return verify_count

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise


def create_backup():
    """Create backup of current master.parquet."""
    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(BACKUP_DIR, f'master_backup_{timestamp}.parquet')

    logger.info(f"Creating backup: {backup_path}")

    import shutil
    shutil.copy2(MASTER_PARQUET, backup_path)

    return backup_path


def replace_master(output_path):
    """Replace master.parquet with merged file."""
    import shutil

    logger.info(f"Replacing master.parquet with merged file...")
    shutil.move(output_path, MASTER_PARQUET)
    logger.info("  Master.parquet updated successfully")


def upload_to_azure(local_path, dry_run=False):
    """Upload master.parquet to Azure Blob Storage."""
    conn_str = get_azure_connection()

    if not conn_str:
        logger.warning("Azure connection string not found. Skipping upload.")
        return False

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.error("azure-storage-blob not installed.")
        return False

    if dry_run:
        logger.info("[DRY RUN] Would upload to Azure Blob Storage")
        return True

    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_name = 'personal-data'
        blob_name = 'athletics/master.parquet'

        logger.info(f"Uploading to Azure: {blob_name}")
        container_client = blob_service.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)

        with open(local_path, 'rb') as f:
            blob_client.upload_blob(f, overwrite=True)

        logger.info("  Upload complete!")
        return True

    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        return False


def print_year_comparison(before_dist, con, output_path):
    """Print year distribution comparison."""
    output_unix = output_path.replace(os.sep, "/")

    after_dist = con.execute(f"""
        SELECT
            EXTRACT(YEAR FROM TRY_CAST(date AS DATE)) as year,
            COUNT(*) as count
        FROM read_parquet('{output_unix}')
        GROUP BY EXTRACT(YEAR FROM TRY_CAST(date AS DATE))
        HAVING year IS NOT NULL
        ORDER BY year
    """).fetchall()

    before_dict = {int(y): c for y, c in before_dist if y is not None}
    after_dict = {int(y): c for y, c in after_dist if y is not None}

    all_years = sorted(set(before_dict.keys()) | set(after_dict.keys()))

    logger.info("")
    logger.info("Year Distribution Comparison:")
    logger.info("-" * 50)
    logger.info(f"{'Year':<8} {'Before':>12} {'After':>12} {'New Rows':>12}")
    logger.info("-" * 50)

    for year in all_years:
        before = before_dict.get(year, 0)
        after = after_dict.get(year, 0)
        new_rows = after - before
        marker = " *NEW*" if new_rows > 0 else ""
        logger.info(f"{year:<8} {before:>12,} {after:>12,} {new_rows:>+12,}{marker}")

    logger.info("-" * 50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Memory-efficient historical data merge using DuckDB'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview changes without saving or uploading'
    )
    parser.add_argument(
        '--local-only', action='store_true',
        help='Merge only, skip Azure upload'
    )
    parser.add_argument(
        '--skip-backup', action='store_true',
        help='Skip creating backup (not recommended)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    load_env()

    logger.info("")
    logger.info("=" * 60)
    logger.info("MEMORY-EFFICIENT HISTORICAL DATA MERGE (DuckDB)")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN MODE - No changes will be made]")

    # Initialize DuckDB with memory limits
    con = duckdb.connect(database=':memory:')
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")

    # Analyze current data
    master_rows, year_dist = analyze_current_data(con)

    # Analyze CSV data
    csv_rows = analyze_csv_data(con, CSV_GLOB_PATTERN)

    if csv_rows == 0:
        logger.error("No CSV data found. Check the path:")
        logger.error(f"  {CSV_GLOB_PATTERN}")
        sys.exit(1)

    # Create backup
    if not args.dry_run and not args.skip_backup:
        create_backup()

    # Merge data
    merged_rows = merge_data(con, CSV_GLOB_PATTERN, OUTPUT_PARQUET, dry_run=args.dry_run)

    if not args.dry_run:
        # Print year comparison
        print_year_comparison(year_dist, con, OUTPUT_PARQUET)

        # Replace master
        replace_master(OUTPUT_PARQUET)

        # Upload to Azure
        if not args.local_only:
            upload_to_azure(MASTER_PARQUET)
        else:
            logger.info("Skipping Azure upload (--local-only)")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("MERGE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original master.parquet:  {master_rows:>12,} rows")
    logger.info(f"Scraped CSV data:         {csv_rows:>12,} rows")
    logger.info(f"Merged (deduplicated):    {merged_rows:>12,} rows")
    logger.info(f"Net new rows:             {merged_rows - master_rows:>+12,} rows")
    logger.info("=" * 60)

    con.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()

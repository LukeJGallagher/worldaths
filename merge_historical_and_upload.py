"""
Merge Historical Data and Upload to Azure

This script merges newly scraped historical data (2003-2019) with the existing
master.parquet and uploads to Azure Blob Storage.

Usage:
    python merge_historical_and_upload.py                    # Merge and upload
    python merge_historical_and_upload.py --local-only       # Merge only, no upload
    python merge_historical_and_upload.py --dry-run          # Preview without changes
    python merge_historical_and_upload.py --backup           # Create backup before merge

The script will:
1. Load existing master.parquet (2.3M rows, 2020-2025)
2. Load historical CSV from scraper (2003-2019)
3. Align column names and data types
4. Combine and deduplicate
5. Save updated master.parquet locally
6. Upload to Azure Blob Storage (with backup)
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_PARQUET = os.path.join(BASE_DIR, 'Data', 'parquet', 'master.parquet')
HISTORICAL_CSV = os.path.join(BASE_DIR, 'world_athletics_scraperv2', 'data', 'historical', 'db_cleaned.csv')
BACKUP_DIR = os.path.join(BASE_DIR, 'Data', 'parquet', 'backups')

# Column mapping: scraper CSV -> master.parquet
COLUMN_MAPPING = {
    'Rank': 'rank',
    'Mark': 'result',
    'Wind': 'wind',
    'Competitor': 'competitor',
    'CompetitorURL': 'competitorurl',
    'DOB': 'dob',
    'Nat': 'nat',
    'Pos': 'pos',
    '': 'unnamed: 8',
    'Venue': 'venue',
    'Date': 'date',
    'ResultScore': 'resultscore',
    'Age': 'age',
    'Event': 'event',
    'Environment': 'environment',
    'Gender': 'gender'
}


def load_env():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, '.env'))
    except ImportError:
        logger.warning("python-dotenv not installed, using system env vars only")


def get_azure_connection():
    """Get Azure Blob Storage connection string."""
    conn_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not conn_str:
        # Try Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'AZURE_STORAGE_CONNECTION_STRING' in st.secrets:
                conn_str = st.secrets['AZURE_STORAGE_CONNECTION_STRING']
        except:
            pass
    return conn_str


def load_master_parquet() -> pd.DataFrame:
    """Load existing master.parquet file."""
    if not os.path.exists(MASTER_PARQUET):
        logger.error(f"Master parquet not found: {MASTER_PARQUET}")
        sys.exit(1)

    logger.info(f"Loading master.parquet from {MASTER_PARQUET}")
    df = pd.read_parquet(MASTER_PARQUET)
    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def load_historical_csv(csv_path: str = None) -> pd.DataFrame:
    """Load historical CSV from scraper output."""
    path = csv_path or HISTORICAL_CSV

    if not os.path.exists(path):
        logger.error(f"Historical CSV not found: {path}")
        logger.error("Run the scraper first: python world_athletics_scraperv2/scrape_historical.py")
        sys.exit(1)

    logger.info(f"Loading historical CSV from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Loaded {len(df):,} rows")

    # Rename columns to match master.parquet
    df.columns = [COLUMN_MAPPING.get(col, col.lower()) for col in df.columns]

    return df


def align_datatypes(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Align data types of df to match reference_df."""
    logger.info("Aligning data types...")

    for col in df.columns:
        if col not in reference_df.columns:
            continue

        ref_dtype = reference_df[col].dtype

        try:
            if pd.api.types.is_datetime64_any_dtype(ref_dtype):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif pd.api.types.is_numeric_dtype(ref_dtype):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif ref_dtype == 'object':
                df[col] = df[col].astype(str).replace('nan', '')
        except Exception as e:
            logger.warning(f"Could not convert {col}: {e}")

    return df


def add_result_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Add result_numeric column if missing."""
    if 'result_numeric' not in df.columns:
        logger.info("Adding result_numeric column...")
        df['result_numeric'] = pd.to_numeric(df['result'], errors='coerce')
    return df


def merge_dataframes(master_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Merge master and historical dataframes, removing duplicates."""
    logger.info("Merging dataframes...")

    # Ensure both have the same columns
    master_cols = set(master_df.columns)
    hist_cols = set(historical_df.columns)

    # Add missing columns to historical
    for col in master_cols - hist_cols:
        historical_df[col] = np.nan
        logger.info(f"  Added missing column to historical: {col}")

    # Reorder columns to match master
    historical_df = historical_df[master_df.columns]

    # Concatenate
    combined = pd.concat([master_df, historical_df], ignore_index=True)
    logger.info(f"  Combined: {len(combined):,} rows")

    # Remove duplicates based on key columns
    dedup_cols = ['competitor', 'event', 'date', 'result', 'venue']
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=dedup_cols, keep='first')
    after_dedup = len(combined)

    if before_dedup > after_dedup:
        logger.info(f"  Removed {before_dedup - after_dedup:,} duplicates")

    # Sort by date descending
    combined = combined.sort_values('date', ascending=False).reset_index(drop=True)

    return combined


def create_backup(df: pd.DataFrame) -> str:
    """Create a backup of the current master.parquet."""
    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(BACKUP_DIR, f'master_backup_{timestamp}.parquet')

    logger.info(f"Creating backup: {backup_path}")
    df.to_parquet(backup_path, index=False)

    return backup_path


def save_master_parquet(df: pd.DataFrame, dry_run: bool = False) -> str:
    """Save merged dataframe to master.parquet."""
    if dry_run:
        logger.info(f"[DRY RUN] Would save {len(df):,} rows to {MASTER_PARQUET}")
        return MASTER_PARQUET

    logger.info(f"Saving {len(df):,} rows to {MASTER_PARQUET}")
    df.to_parquet(MASTER_PARQUET, index=False)

    return MASTER_PARQUET


def upload_to_azure(local_path: str, dry_run: bool = False) -> bool:
    """Upload master.parquet to Azure Blob Storage."""
    conn_str = get_azure_connection()

    if not conn_str:
        logger.warning("Azure connection string not found. Skipping upload.")
        logger.warning("Set AZURE_STORAGE_CONNECTION_STRING in .env or environment")
        return False

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        logger.error("azure-storage-blob not installed. Run: pip install azure-storage-blob")
        return False

    if dry_run:
        logger.info("[DRY RUN] Would upload to Azure Blob Storage")
        return True

    try:
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_name = 'personal-data'
        blob_name = 'athletics/master.parquet'

        # Create backup in Azure first
        backup_blob_name = f'athletics/backups/master_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'

        logger.info(f"Creating Azure backup: {backup_blob_name}")
        container_client = blob_service.get_container_client(container_name)

        # Copy current blob to backup
        try:
            source_blob = container_client.get_blob_client(blob_name)
            backup_blob = container_client.get_blob_client(backup_blob_name)
            backup_blob.start_copy_from_url(source_blob.url)
            logger.info("  Azure backup created")
        except Exception as e:
            logger.warning(f"  Could not create Azure backup: {e}")

        # Upload new file
        logger.info(f"Uploading to Azure: {blob_name}")
        blob_client = container_client.get_blob_client(blob_name)

        with open(local_path, 'rb') as f:
            blob_client.upload_blob(f, overwrite=True)

        logger.info("  Upload complete!")
        return True

    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        return False


def print_summary(master_df: pd.DataFrame, historical_df: pd.DataFrame, merged_df: pd.DataFrame):
    """Print summary of the merge operation."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("MERGE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Existing master.parquet:  {len(master_df):>12,} rows")
    logger.info(f"New historical data:      {len(historical_df):>12,} rows")
    logger.info(f"Combined (after dedup):   {len(merged_df):>12,} rows")
    logger.info(f"Net new rows added:       {len(merged_df) - len(master_df):>12,} rows")
    logger.info("")

    # Year distribution
    merged_df['year'] = pd.to_datetime(merged_df['date']).dt.year
    year_counts = merged_df['year'].value_counts().sort_index()

    logger.info("Year Distribution:")
    for year, count in year_counts.items():
        bar = '#' * int(count / 50000)
        logger.info(f"  {int(year)}: {count:>8,} {bar}")

    logger.info("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge historical data and upload to Azure'
    )
    parser.add_argument(
        '--local-only', action='store_true',
        help='Merge only, skip Azure upload'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview changes without saving or uploading'
    )
    parser.add_argument(
        '--backup', action='store_true',
        help='Create local backup before merge'
    )
    parser.add_argument(
        '--historical-csv', type=str,
        help='Path to historical CSV (default: scraperv2/data/historical/db_cleaned.csv)'
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
    logger.info("MERGE HISTORICAL DATA AND UPLOAD TO AZURE")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN MODE - No changes will be made]")

    # Load data
    master_df = load_master_parquet()
    historical_df = load_historical_csv(args.historical_csv)

    # Align data types
    historical_df = align_datatypes(historical_df, master_df)
    historical_df = add_result_numeric(historical_df)

    # Create backup if requested
    if args.backup and not args.dry_run:
        create_backup(master_df)

    # Merge
    merged_df = merge_dataframes(master_df, historical_df)

    # Print summary
    print_summary(master_df, historical_df, merged_df)

    # Save locally
    if not args.dry_run:
        if not args.skip_backup:
            create_backup(master_df)
        save_master_parquet(merged_df)
    else:
        save_master_parquet(merged_df, dry_run=True)

    # Upload to Azure
    if not args.local_only:
        upload_to_azure(MASTER_PARQUET, dry_run=args.dry_run)
    else:
        logger.info("Skipping Azure upload (--local-only)")

    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()

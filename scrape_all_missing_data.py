"""
Scrape All Missing Data - Fill the Gaps in Master Parquet

This script orchestrates scraping of all missing data:
1. Historical years (2003-2019) - 17 years of data
2. Year 2023 - Missing from current dataset
3. Merges everything into master.parquet and uploads to Azure

Current data coverage in master.parquet:
- 2001-2002: 102 records (sparse)
- 2020-2025: 2,118 records (good)
- MISSING: 2003-2019, 2023

Usage:
    python scrape_all_missing_data.py --test       # Test mode (quick validation)
    python scrape_all_missing_data.py --year 2023  # Scrape just 2023
    python scrape_all_missing_data.py --full       # Full historical scrape (overnight)
    python scrape_all_missing_data.py --merge-only # Just merge existing CSVs

Estimated times:
- Test mode: ~5 minutes
- Single year (2023): ~30 minutes
- Full historical (2003-2019): ~8-12 hours (run overnight)
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
SCRAPER_DIR = BASE_DIR / "world_athletics_scraperv2"
DATA_DIR = BASE_DIR / "Data" / "parquet"
HISTORICAL_DIR = SCRAPER_DIR / "data" / "historical"
OUTPUT_2023_DIR = SCRAPER_DIR / "data" / "2023"


def run_scraper(years, output_dir, performances=7000, test_mode=False):
    """Run the historical scraper for specified years."""
    scraper_script = SCRAPER_DIR / "scrape_historical.py"

    if not scraper_script.exists():
        logger.error(f"Scraper not found: {scraper_script}")
        return False

    for year in years:
        logger.info(f"Scraping year {year}...")

        cmd = [
            sys.executable, str(scraper_script),
            "--year", str(year),
            "--performances", str(500 if test_mode else performances),
            "--output", str(output_dir)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRAPER_DIR))
            if result.returncode != 0:
                logger.error(f"Scraper failed for {year}: {result.stderr}")
                return False
            logger.info(f"Completed {year}")
        except Exception as e:
            logger.error(f"Error running scraper for {year}: {e}")
            return False

    return True


def merge_data():
    """Merge scraped data with existing master.parquet."""
    merge_script = BASE_DIR / "merge_historical_and_upload.py"

    if not merge_script.exists():
        logger.error(f"Merge script not found: {merge_script}")
        return False

    logger.info("Merging historical data with master.parquet...")

    cmd = [sys.executable, str(merge_script)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Merge failed: {result.stderr}")
            return False
        logger.info("Merge completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        return False


def check_current_coverage():
    """Check current data coverage in master.parquet."""
    import pandas as pd

    master_path = DATA_DIR / "master.parquet"
    if not master_path.exists():
        logger.warning("master.parquet not found")
        return

    df = pd.read_parquet(master_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year

    logger.info("\nCurrent data coverage in master.parquet:")
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info("\nRecords by year:")

    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        if pd.notna(year):
            logger.info(f"  {int(year)}: {count:,}")

    # Identify gaps
    all_years = set(range(2003, 2026))
    existing_years = set(year_counts[year_counts > 100].index.dropna().astype(int))
    missing = sorted(all_years - existing_years)

    if missing:
        logger.info(f"\nMissing years (< 100 records): {missing}")
    else:
        logger.info("\nNo significant gaps found!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Scrape all missing World Athletics data'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Test mode: scrape 2023 only with 500 performances'
    )
    parser.add_argument(
        '--year', type=int,
        help='Scrape a single specific year'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Full scrape: all historical years 2003-2019 + 2023'
    )
    parser.add_argument(
        '--merge-only', action='store_true',
        help='Skip scraping, just merge existing data'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Just check current data coverage'
    )
    parser.add_argument(
        '--performances', type=int, default=7000,
        help='Number of performances per event (default: 7000)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("WORLD ATHLETICS DATA GAP FILLER")
    logger.info("=" * 60)

    # Check current coverage
    check_current_coverage()

    if args.check:
        return

    if args.merge_only:
        merge_data()
        check_current_coverage()
        return

    # Determine what to scrape
    if args.test:
        years = [2023]
        performances = 500
        output_dir = OUTPUT_2023_DIR
        logger.info("\nTEST MODE: Scraping 2023 with 500 performances")
    elif args.year:
        years = [args.year]
        performances = args.performances
        output_dir = SCRAPER_DIR / "data" / str(args.year)
        logger.info(f"\nSingle year mode: {args.year}")
    elif args.full:
        # Full historical + 2023
        years = list(range(2003, 2020)) + [2023]  # 2003-2019 and 2023
        performances = args.performances
        output_dir = HISTORICAL_DIR
        logger.info(f"\nFULL MODE: Scraping 2003-2019 and 2023 ({len(years)} years)")
        logger.info("This will take several hours. Press Ctrl+C to cancel.")

        import time
        for i in range(10, 0, -1):
            print(f"Starting in {i}...", end='\r')
            time.sleep(1)
        print()
    else:
        # Default: just 2023 (most critical for Asian Games)
        years = [2023]
        performances = args.performances
        output_dir = OUTPUT_2023_DIR
        logger.info("\nDefault mode: Scraping 2023 (for Asian Games data)")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run scraper
    start_time = datetime.now()
    logger.info(f"\nStarted at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    success = run_scraper(years, output_dir, performances, test_mode=args.test)

    if success:
        # Merge data
        merge_data()

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"\nCompleted in {duration}")

        # Show updated coverage
        check_current_coverage()
    else:
        logger.error("Scraping failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

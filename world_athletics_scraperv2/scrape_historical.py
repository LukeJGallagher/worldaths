"""
Historical Data Scraper - Fill the 2003-2019 gap

This script runs the scraper for historical years (2003-2019) to fill
the data gap in the master.parquet dataset.

Usage:
    python scrape_historical.py                  # Scrape all 2003-2019
    python scrape_historical.py --start 2010     # Start from 2010
    python scrape_historical.py --year 2015      # Scrape single year
    python scrape_historical.py --test           # Test mode (2019 only, fewer performances)

The script will:
1. Scrape data year by year (oldest first)
2. Save individual CSV files per event/year
3. Combine into db.csv and db_cleaned.csv when complete
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from main import WorldAthleticsApp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Scrape historical World Athletics data (2003-2019)'
    )
    parser.add_argument(
        '--start', type=int, default=2003,
        help='Start year (default: 2003)'
    )
    parser.add_argument(
        '--end', type=int, default=2019,
        help='End year inclusive (default: 2019)'
    )
    parser.add_argument(
        '--year', type=int,
        help='Scrape a single specific year'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Test mode: scrape 2019 only with fewer performances'
    )
    parser.add_argument(
        '--performances', type=int, default=7000,
        help='Number of performances per event (default: 7000)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory (default: data/historical/)'
    )
    return parser.parse_args()


def main():
    """Run historical scraper."""
    args = parse_args()

    # Determine years to scrape
    if args.test:
        years = [2019]
        num_performances = 500  # Smaller sample for testing
        logger.info("TEST MODE: Scraping 2019 only with 500 performances")
    elif args.year:
        years = [args.year]
        num_performances = args.performances
        logger.info(f"Single year mode: {args.year}")
    else:
        years = list(range(args.start, args.end + 1))
        num_performances = args.performances
        logger.info(f"Scraping years {args.start}-{args.end}")

    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(config.OUTPUT_DIR, 'historical')

    os.makedirs(output_dir, exist_ok=True)

    # Calculate time estimate
    events_count = len(config.EVENTS)
    genders_count = len(config.DEFAULT_GENDER)
    total_combos = len(years) * events_count * genders_count
    est_time_hours = (total_combos * 0.5) / 60 / 60  # ~0.5s per combo average

    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"HISTORICAL DATA SCRAPER")
    logger.info(f"=" * 60)
    logger.info(f"Years to scrape:     {years[0]}-{years[-1]} ({len(years)} years)")
    logger.info(f"Events:              {events_count}")
    logger.info(f"Genders:             {genders_count}")
    logger.info(f"Total combinations:  {total_combos}")
    logger.info(f"Performances/event:  {num_performances}")
    logger.info(f"Output directory:    {output_dir}")
    logger.info(f"=" * 60)
    logger.info(f"")

    # Confirm before starting long scrape
    if len(years) > 3 and not args.test:
        logger.info("This is a large scrape. Starting in 5 seconds...")
        logger.info("Press Ctrl+C to cancel.")
        import time
        time.sleep(5)

    # Initialize app with custom output directory
    app = WorldAthleticsApp(output_dir=output_dir)

    # Run scraper
    start_time = datetime.now()
    logger.info(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        combined, cleaned = app.run(
            years=years,
            num_performances=num_performances
        )

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info(f"")
        logger.info(f"=" * 60)
        logger.info(f"SCRAPE COMPLETE")
        logger.info(f"=" * 60)
        logger.info(f"Duration:      {duration}")
        logger.info(f"Combined DB:   {combined}")
        logger.info(f"Cleaned DB:    {cleaned}")
        logger.info(f"=" * 60)

        if combined:
            # Show row count
            import pandas as pd
            try:
                df = pd.read_csv(combined)
                logger.info(f"Total rows:    {len(df):,}")
                logger.info(f"Year range:    {df['Date'].str[-4:].min()}-{df['Date'].str[-4:].max()}")
            except Exception as e:
                logger.warning(f"Could not read stats: {e}")

    except KeyboardInterrupt:
        logger.warning("Scrape interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Scrape failed: {e}")
        raise


if __name__ == "__main__":
    main()

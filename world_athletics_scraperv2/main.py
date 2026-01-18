"""
Main application module for World Athletics Top Lists Scraper.
Coordinates the scraping process and handles concurrent operations.
"""

import os
import sys
import time
import logging
import concurrent.futures
import itertools
from datetime import datetime
from typing import Optional, List, Tuple

import config
from scraper import WorldAthleticsScraper
from storage import DataStorage

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,           # change to DEBUG for more detail
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WorldAthleticsApp:
    """
    Main application class for World Athletics Top Lists Scraper.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        # resolve to absolute path so later glob searches work regardless of CWD
        self.output_dir = os.path.abspath(output_dir or config.OUTPUT_DIR)

        # components
        self.scraper = WorldAthleticsScraper(
            base_url=config.BASE_URL,
            max_retries=config.MAX_RETRIES,
            request_delay=config.REQUEST_DELAY,
        )
        self.storage = DataStorage(output_dir=self.output_dir)
        self.headers = config.HEADERS

    # ──────────────────────────────────────────────────────────────────────
    # Low‑level helpers
    # ──────────────────────────────────────────────────────────────────────
    def _scrape_page(
        self,
        event: str,
        event_category: str,
        gender: str,
        age_category: str,
        year: int,
        page: int,
        best_results_only: bool,
        track_size: str,
        region_type: str,
    ):
        """Scrape a single results page and return a list of rows."""
        return self.scraper.scrape_event(
            event_category=event_category,
            event=event,
            gender=gender,
            age_category=age_category,
            year=year,
            page=page,
            region_type=region_type,
            best_results_only=best_results_only,
            track_size=track_size,
        )

    def _scrape_event(
        self,
        event: str,
        gender: str,
        age_category: str,
        year: int,
        best_results_only: bool,
        track_size: str,
        region_type: str,
        num_performances: int,
    ) -> Optional[str]:
        """Scrape **all pages** for one parameter combination."""
        event_category = config.EVENTS.get(event)
        if not event_category:
            logger.error("Unknown event: %s", event)
            return None

        pages = config.calculate_pages(num_performances)
        all_rows: List[List[str]] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.MAX_CONCURRENT_REQUESTS
        ) as pool:
            futures = [
                pool.submit(
                    self._scrape_page,
                    event,
                    event_category,
                    gender,
                    age_category,
                    year,
                    p,
                    best_results_only,
                    track_size,
                    region_type,
                )
                for p in pages
            ]

            for future in concurrent.futures.as_completed(futures):
                rows = future.result()
                if rows:
                    all_rows.extend(rows)

        if not all_rows:
            logger.warning("No rows scraped for %s %s %d", event, gender, year)
            return None

        return self.storage.save_results(
            results=all_rows,
            event=event,
            gender=gender,
            year=year,
            age_category=age_category,
            best_results_only=best_results_only,
            track_size=track_size,
            region_type=region_type,
            headers=self.headers,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public runner
    # ──────────────────────────────────────────────────────────────────────
    def run(
        self,
        years: Optional[list] = None,
        events: Optional[list] = None,
        genders: Optional[list] = None,
        age_categories: Optional[list] = None,
        best_results_only: Optional[list] = None,
        track_sizes: Optional[list] = None,
        region_types: Optional[list] = None,
        num_performances: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Scrape every combination and build combined / cleaned DBs."""
        # apply defaults
        years = years or config.DEFAULT_YEARS
        events = events or list(config.EVENTS.keys())
        genders = genders or config.DEFAULT_GENDER
        age_categories = age_categories or config.DEFAULT_AGE_CATEGORY
        best_results_only = best_results_only or config.DEFAULT_BEST_RESULTS_ONLY
        track_sizes = track_sizes or config.DEFAULT_TRACK_SIZE
        region_types = region_types or config.DEFAULT_REGION_TYPE
        num_performances = num_performances or config.DEFAULT_NUM_PERFORMANCES

        start = time.time()
        logger.info(
            "Starting scraper (years=%s, events=%d, genders=%s)",
            years,
            len(events),
            genders,
        )

        combos = list(
            itertools.product(
                years,
                events,
                age_categories,
                best_results_only,
                track_sizes,
                region_types,
                genders,
            )
        )
        logger.info("Total combinations: %d", len(combos))

        for idx, (yr, ev, age_cat, best_only, track_sz, region, gen) in enumerate(
            combos, start=1
        ):
            logger.info(
                "▶ [%d/%d] %s %s %d %s",
                idx,
                len(combos),
                ev,
                gen,
                yr,
                age_cat,
            )
            self._scrape_event(
                event=ev,
                gender=gen,
                age_category=age_cat,
                year=yr,
                best_results_only=best_only,
                track_size=track_sz,
                region_type=region,
                num_performances=num_performances,
            )

        # ── Combine CSVs ────────────────────────────────────────────────
        combined_path = os.path.join(self.output_dir, config.DB_FILENAME)
        combined = self.storage.combine_csv_files(combined_path)

        if combined is None:  # nothing was scraped
            logger.error("❌ No CSV files were written – skipping clean step.")
            return None, None

        cleaned_path = os.path.join(self.output_dir, config.DB_CLEANED_FILENAME)
        cleaned = self.storage.clean_database(combined, cleaned_path)

        logger.info("Finished in %.1fs", time.time() - start)
        logger.info("Combined DB ➜ %s", combined)
        logger.info("Cleaned   DB ➜ %s", cleaned)
        return combined, cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WorldAthleticsApp().run()

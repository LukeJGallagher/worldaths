"""
Storage utilities for World Athletics Top Lists Scraper.
Handles CSV writing, merging, and cleaning.
"""

import csv
import glob
import os
import logging
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataStorage:
    """Write individual CSVs, merge them, clean big DB."""

    # ────────────────────────────────────────────────────────────────────
    # Construction
    # ────────────────────────────────────────────────────────────────────
    def __init__(self, output_dir: str = "data") -> None:
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # 1 – Write one event CSV
    # ────────────────────────────────────────────────────────────────────
    def save_results(
        self,
        results: List[List[str]],
        event: str,
        gender: str,
        year: int,
        age_category: str,
        best_results_only: bool,
        track_size: str,
        region_type: str,
        headers: Iterable[str],
    ) -> Optional[str]:
        if not results:
            logger.warning("No rows to save for %s‑%s‑%d", event, gender, year)
            return None

        folder = os.path.join(self.output_dir, event, gender)
        os.makedirs(folder, exist_ok=True)

        fname = f"{year}-{age_category}-{best_results_only}-{track_size}-{region_type}.csv"
        path = os.path.join(folder, fname)

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            writer.writerows(results)

        logger.info("✓ Saved %d rows → %s", len(results), path)
        return path

    # ────────────────────────────────────────────────────────────────────
    # 2 – Combine many CSVs into one
    # ────────────────────────────────────────────────────────────────────
    def combine_csv_files(
        self, output_file: str, input_pattern: Optional[str] = None
    ) -> Optional[str]:
        input_pattern = input_pattern or os.path.join(self.output_dir, "**", "*.csv")
        csv_files = glob.glob(input_pattern, recursive=True)

        if not csv_files:
            logger.warning("No CSV files found for %s", input_pattern)
            return None

        logger.info("Combining %d CSV files → %s", len(csv_files), output_file)

        header_written = False
        rows_written = 0

        with open(output_file, "w", newline="", encoding="utf-8") as out_fh:
            writer = csv.writer(out_fh)

            for path in csv_files:
                with open(path, newline="", encoding="utf-8") as fh:
                    reader = csv.reader(fh)
                    try:
                        first_row = next(reader)
                    except StopIteration:
                        logger.warning("Empty file skipped: %s", path)
                        continue  # move on to next file

                    if not header_written:
                        writer.writerow(first_row)  # write header once
                        header_written = True
                    else:
                        # assume all files share the same header; skip this row
                        pass

                    # write remaining rows
                    row_count = 0
                    for row in reader:
                        writer.writerow(row)
                        row_count += 1
                    rows_written += row_count

        if not header_written:
            logger.error("All CSVs were empty – combined DB not created.")
            os.remove(output_file)  # avoid leaving a 0‑byte file
            return None

        logger.info("✓ Combined %d data rows", rows_written)
        return output_file

    # ────────────────────────────────────────────────────────────────────
    # 3 – Clean combined DB
    # ────────────────────────────────────────────────────────────────────
    def clean_database(self, input_file: str, output_file: str) -> str:
        logger.info("Cleaning %s → %s", input_file, output_file)

        df = pd.read_csv(input_file, dtype={"Mark": "string"})

        # remove non‑numeric chars from “Mark” and convert to float
        df["Mark"] = (
            df["Mark"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
        ).astype(float, errors="ignore")

        # drop rows missing Age
        if "Age" in df.columns:
            df.dropna(subset=["Age"], inplace=True)

        df.to_csv(output_file, index=False)
        logger.info("✓ Cleaned DB saved")
        return output_file

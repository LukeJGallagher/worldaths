World Athletics Top‑Lists & Athlete Results Scraper
===================================================

This project has two main components:

1. Top List Scraper (main.py + scraper.py)
2. Athlete Results Fetcher (athlete_results.py)

It downloads World Athletics top performance lists, builds a combined database,
and fetches competition-level history for every athlete listed.

Output is saved under `data/`:
- data/db.csv                    → combined top list results
- data/db_cleaned.csv           → cleaned version with extra fields
- data/athlete_results/<id>.csv → per-athlete results
- data/athlete_results/master_athlete_results.csv → all athletes merged

------------------------
QUICK START
------------------------

1. Create and activate a Python environment:

    python -m venv venv
    source venv/bin/activate         # On Windows: venv\Scripts\activate

2. Install dependencies:

    pip install -r requirements.txt

3. Run the top-lists scraper:

    python main.py

4. Then run the athlete history fetcher:

    python athlete_results.py

------------------------
CONFIGURATION
------------------------

You can change the following in `config.py`:
- DEFAULT_YEARS
- DEFAULT_NUM_PERFORMANCES
- EVENTS list

These control which years, how many athletes per event, and which events get scraped.

The script will generate two CSVs:
- db.csv – raw combined file
- db_cleaned.csv – cleaned version with extra columns (event, environment, etc.)

------------------------
ATHLETE RESULTS SCRIPT
------------------------

`athlete_results.py` takes `db_cleaned.csv` and pulls detailed competition
history for every athlete using the World Athletics private AppSync API.

Each result includes:
- AthleteName
- DOB
- Gender
- CompetitorURL

These allow you to cross-reference against other datasets.

Compressed data (brotli, zstd, gzip, deflate) is handled automatically.
TLS verification is disabled by default to work on macOS.

------------------------
NOTES
------------------------

- Sleep time of 0.4s is included between API calls to avoid rate-limiting.
- Output files go to the `data/` folder by default.
- You can change output directories in `athlete_results.py` by modifying `OUT_DIR`.

------------------------
LICENSE
------------------------

MIT License – feel free to adapt or extend.

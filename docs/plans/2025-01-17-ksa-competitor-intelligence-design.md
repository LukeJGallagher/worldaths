# KSA Competitor Intelligence App - Design Document

**Date:** 2025-01-17
**Status:** Approved
**Author:** Claude + Luke Gallagher

---

## Overview

A Streamlit app providing complete competitor analysis for KSA athletes using World Athletics scraped data. Features pre-competition intel, gap analysis, and event scouting with a hybrid Azure/local data approach.

---

## Architecture

### Data Storage (Dual Mode)

**Azure Mode** (production/Streamlit Cloud):
```
Azure Container: personal-data
└── athletics/
    ├── master.parquet          # 2.3M scraped records
    ├── ksa_profiles.parquet    # 152 KSA athlete profiles
    ├── benchmarks.parquet      # Championship standards
    └── backups/
        └── backup_YYYYMMDD.parquet
```

**Local Mode** (development/offline):
```
world_athletics/
└── Data/
    ├── master.parquet
    ├── ksa_profiles.parquet
    ├── benchmarks.parquet
    └── backups/
```

### Mode Detection

```python
DATA_MODE = "azure" if os.getenv('AZURE_STORAGE_CONNECTION_STRING') else "local"
```

### Tech Stack

- **DuckDB** - Query Parquet directly from Azure or local
- **Streamlit** - UI framework
- **Parquet** - Columnar storage (fast, compact)
- **GitHub Actions** - Weekly automated scraping

---

## Data Model

### master.parquet (2.3M records)

| Column | Type | Description |
|--------|------|-------------|
| athlete_id | STRING | World Athletics ID |
| athlete_name | STRING | Full name |
| country_code | STRING | 3-letter code (KSA, USA, etc.) |
| gender | STRING | Men / Women |
| event | STRING | Normalized (100m, Long Jump, etc.) |
| result | STRING | Raw result (10.05, 8.12m, etc.) |
| result_numeric | FLOAT | Parsed for sorting (seconds/meters) |
| date | DATE | Competition date |
| competition | STRING | Competition name |
| venue | STRING | City/stadium |
| round | STRING | Heat/Semi/Final |
| position | INT | Place in round |
| wind | FLOAT | Wind reading (nullable) |
| season_best | BOOL | SB marker |
| personal_best | BOOL | PB marker |

### ksa_profiles.parquet (152 athletes)

| Column | Type | Description |
|--------|------|-------------|
| athlete_id | STRING | World Athletics ID |
| name | STRING | Full name |
| dob | DATE | Date of birth |
| primary_event | STRING | Main event |
| pb | STRING | Personal best |
| wpa_score | INT | World Athletics Points |
| world_rank | INT | Current world ranking |
| photo_url | STRING | Profile image URL |

### benchmarks.parquet (~500 records)

| Column | Type | Description |
|--------|------|-------------|
| championship | STRING | Olympics/World Champs/Asian Games |
| year | INT | Edition year |
| event | STRING | Event name |
| gender | STRING | Men / Women |
| medal_mark | FLOAT | Typical medal-winning performance |
| final_cutoff | FLOAT | Typical mark to make final |
| qualifying_standard | FLOAT | Entry standard |

---

## App Features

### Tab 1: Pre-Competition Intel

**Use case:** "Who is Mohammed TOLU racing against in Tokyo 2025 discus?"

Features:
- Select KSA athlete and event
- Show athlete card (PB, SB, world rank, form trend)
- Display top 20 competitors with SB, PB, head-to-head record
- Expandable head-to-head history for each competitor

### Tab 2: Gap Analysis

**Use case:** "How far is our 400m runner from the world top 10?"

Features:
- Visual gap bars showing athlete PB vs targets
- Targets: Championship standard, World #10, Asian #1
- Gap calculation in absolute terms and percentage
- Progression assessment based on historical improvement rates

### Tab 3: Event Scouting

**Use case:** "Who are the rising competitors in Asian 100m?"

Features:
- Filter by event, gender, region, age group, season
- Ranked list of top performers with trends
- "Rising Threats" section highlighting athletes with >2% improvement
- Click-through to athlete details

---

## File Structure

```
world_athletics/
├── ksa_competitor_intelligence.py    # Main Streamlit app
├── data_connector.py                 # Azure/Local dual-mode data access
├── convert_to_parquet.py             # One-time: convert existing data to Parquet
├── scraper_athletics.py              # Weekly scraper (GitHub Actions)
├── Data/                             # Local Parquet files
│   ├── master.parquet
│   ├── ksa_profiles.parquet
│   └── benchmarks.parquet
├── .github/
│   └── workflows/
│       └── athletics_scraper.yml     # Weekly data refresh
├── docs/
│   └── plans/
│       └── 2025-01-17-ksa-competitor-intelligence-design.md
└── requirements.txt
```

---

## Implementation Order

| Step | File | Description |
|------|------|-------------|
| 1 | `convert_to_parquet.py` | Convert db_cleaned.csv, ksa_athlete_profiles.db, what_it_takes_to_win.db → Parquet |
| 2 | `data_connector.py` | DuckDB wrapper with Azure/local auto-detection |
| 3 | `ksa_competitor_intelligence.py` | Main app with 3 tabs |
| 4 | `scraper_athletics.py` | Automated weekly refresh |
| 5 | `athletics_scraper.yml` | GitHub Actions workflow |

---

## Dependencies

```
streamlit>=1.30.0
duckdb>=0.10.0
pandas>=2.0.0
azure-storage-blob>=12.19.0
pyarrow>=14.0.0
```

---

## Styling

Team Saudi theme:
- Primary Teal: `#007167`
- Gold Accent: `#a08e66`
- Dark Teal: `#005a51`
- Header gradient: `linear-gradient(135deg, #007167 0%, #005a51 100%)`

---

## Data Sources

1. **Primary:** `db_cleaned.csv` (2.3M scraped records from world_athletics_scraperv2)
2. **Secondary:** `ksa_athlete_profiles.db` (152 KSA athletes)
3. **Tertiary:** `what_it_takes_to_win.db` (championship benchmarks)
4. **Optional:** GraphQL API for live enrichment (graceful fallback if blocked)

---

## GitHub Actions Workflow

```yaml
name: Athletics Data Scraper
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
  workflow_dispatch:

jobs:
  run-athletics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install duckdb azure-storage-blob pandas pyarrow
      - name: Run Athletics Scraper
        env:
          AZURE_STORAGE_CONNECTION_STRING: ${{ secrets.AZURE_STORAGE_CONNECTION_STRING }}
        run: python scraper_athletics.py
```

---

## Future Enhancements

- Live GraphQL API enrichment when available
- Coach View (simplified pre-competition briefing mode)
- Relay team analysis
- Historical championship progression charts

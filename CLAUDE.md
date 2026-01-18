# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

World Athletics data scraping and visualization system for Team Saudi. Scrapes athlete rankings, results, and qualification data from worldathletics.org, stores in Parquet/SQLite databases, and displays via Streamlit dashboards.

## Common Commands

### Running Dashboards
```bash
streamlit run World_Ranking_Dash.py      # Local development
streamlit run World_Ranking_Deploy.py    # Production version
```

### Data Pipeline (in order)
```bash
python "World Rankings_All.py"           # Scrape global rankings → Data/*.csv
python "World Rankings_KSA_Ave_both.py"  # Scrape KSA athlete details → Data/*.csv
python "SQL Converter2.py"               # Convert CSVs → SQL/*.db
```

### Parquet Data Management
```bash
python convert_to_parquet.py             # Convert to Parquet + upload to Azure
python backup_manager.py validate        # Check data before upload
python backup_manager.py upload          # Validate + backup + upload
python backup_manager.py list            # List Azure backups
python backup_manager.py baseline        # Create protected snapshot
python backup_manager.py restore --baseline  # Restore from baseline
```

### Scrapers (require Chrome/chromedriver)
```bash
python "World Rankings_All.py"           # Global rankings, all events
python "World Rankings_KSA_Ave_both.py"  # KSA-specific modal results
python road_to_scraper6.py               # Road to Tokyo qualification (interactive)
```

## Architecture

### Data Flow
```
worldathletics.org (Selenium scrapers)
        ↓
   Data/*.csv (raw scraped data)
        ↓
   SQL/*.db (SQLite - legacy)  OR  Data/parquet/*.parquet (modern)
        ↓
   Azure Blob Storage (athletics/)  ← for Streamlit Cloud
        ↓
   Streamlit Dashboard (DuckDB queries Parquet directly)
```

### Azure Blob Structure
```
personal-data/athletics/
├── master.parquet          # 2.3M scraped records
├── ksa_profiles.parquet    # 152 KSA athlete profiles
├── benchmarks.parquet      # Championship standards
├── baseline/               # Protected backups (never auto-deleted)
└── backups/                # Rolling backups (7 daily, 4 weekly)
```

### Key Files

| File | Purpose |
|------|---------|
| `World_Ranking_Dash.py` | Main dashboard (development) |
| `World_Ranking_Deploy.py` | Main dashboard (production - push to GitHub) |
| `convert_to_parquet.py` | Convert CSV/DB → Parquet + Azure upload |
| `backup_manager.py` | Backup validation, retention, restore |
| `data_connector.py` | DuckDB wrapper for Azure/local Parquet |
| `World Rankings_All.py` | Global rankings scraper |
| `World Rankings_KSA_Ave_both.py` | KSA athlete detail scraper |

### Parquet Schema

**master.parquet** (2.3M rows):
- rank, result, wind, competitor, competitorurl, dob, nat, pos, venue, date
- resultscore, age, event, environment, gender, result_numeric

**ksa_profiles.parquet** (152 rows):
- athlete_id, full_name, gender, date_of_birth, primary_event
- profile_image_url, country_code, status, best_score, best_world_rank

**benchmarks.parquet** (49 rows):
- Event, Gender, Year, Gold/Silver/Bronze Standard, Final Standard (8th), Top 8 Average

### Backup Validation Thresholds
- master.parquet: minimum 2,000,000 rows
- ksa_profiles.parquet: minimum 100 rows
- benchmarks.parquet: minimum 40 rows

## Environment Variables

```bash
# .env file (local) or Streamlit secrets (cloud)
AZURE_STORAGE_CONNECTION_STRING=...  # For Parquet/Blob storage
AZURE_SQL_CONN=...                   # For SQL database (if used)
```

## Development Workflow

1. Edit `World_Ranking_Dash.py` for local development
2. Test with `streamlit run World_Ranking_Dash.py`
3. Copy finalized code to `World_Ranking_Deploy.py`
4. Push to GitHub for Streamlit Cloud deployment

## Scraper Configuration

Scrapers have configurable parameters at the top of each file:
- `rank_date` - Date for rankings query
- `num_pages` - Pages to scrape per event (default: 3)
- `event_lists` - Events to scrape by gender

## Additional Documentation

- `AZURE_BLOB_DEPLOYMENT_GUIDE.md` - Azure Blob Storage + Parquet + DuckDB setup guide
- `docs/plans/` - Design documents for planned features

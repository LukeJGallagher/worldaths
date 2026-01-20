# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

World Athletics data scraping and visualization system for Team Saudi. Scrapes athlete rankings, results, and qualification data from worldathletics.org, stores in Parquet/SQLite databases, and displays via Streamlit dashboards with 9 analytics tabs including an AI chatbot and Coach View.

## Common Commands

### Running Dashboards
```bash
streamlit run World_Ranking_Dash.py      # Local development
streamlit run World_Ranking_Deploy_v3.py # Production version (9 tabs with AI + Coach View)
```

### Data Pipeline (in order)
```bash
python "World Rankings_All.py"           # Scrape global rankings → Data/*.csv
python "World Rankings_KSA_Ave_both.py"  # Scrape KSA athlete details → Data/*.csv
python "SQL Converter2.py"               # Convert CSVs → SQL/*.db
python convert_to_parquet.py             # Convert to Parquet + upload to Azure
```

### Backup Management
```bash
python backup_manager.py validate        # Check data before upload
python backup_manager.py upload          # Validate + backup + upload
python backup_manager.py list            # List Azure backups
python backup_manager.py baseline        # Create protected snapshot
python backup_manager.py restore --baseline  # Restore from baseline
```

## Architecture

### Dashboard Structure (9 Tabs)

| Tab | Name | Key Module/Function |
|-----|------|---------------------|
| 1 | Event Standards & Progression | `WhatItTakesToWin` class |
| 2 | Athlete Profiles | `get_ksa_athletes()` |
| 3 | Combined Rankings | `get_rankings_data()` |
| 4 | Saudi Athletes Rankings | `get_ksa_rankings()` |
| 5 | World Champs Qualification | `get_road_to_tokyo_data()` |
| 6 | Major Games Analytics | `MajorGamesAnalyzer` class |
| 7 | What It Takes to Win (Live) | `WhatItTakesToWin.generate_report()` |
| 8 | AI Analyst | `OpenRouterClient` (free LLM models) |
| 9 | Coach View | `render_coach_view()` (competition prep, projections) |

### Data Flow
```
worldathletics.org (Selenium scrapers)
        ↓
   Data/*.csv (raw scraped data)
        ↓
   Data/parquet/*.parquet (modern) OR SQL/*.db (legacy)
        ↓
   Azure Blob Storage (athletics/)  ← for Streamlit Cloud
        ↓
   Streamlit Dashboard (DuckDB queries Parquet directly)
```

### Key Modules

| File | Purpose |
|------|---------|
| `World_Ranking_Deploy_v3.py` | Main dashboard with 8 tabs |
| `data_connector.py` | DuckDB wrapper - Azure/local dual-mode |
| `athletics_analytics_agents.py` | Analytics classes (SprintsAnalyzer, MajorGamesAnalyzer, etc.) |
| `what_it_takes_to_win.py` | `WhatItTakesToWin` class for medal standards |
| `convert_to_parquet.py` | CSV/DB → Parquet + Azure upload |
| `backup_manager.py` | Backup validation, retention, restore |

### data_connector.py Key Functions
```python
get_ksa_athletes()           # KSA athlete profiles
get_rankings_data()          # Global rankings (2.3M records)
get_ksa_rankings()           # KSA-filtered rankings
get_benchmarks_data()        # Championship standards
get_road_to_tokyo_data()     # Qualification data
get_competitors()            # Top competitors by event
get_head_to_head()           # H2H comparison
get_gap_analysis()           # Gap to qualification standards
get_data_mode()              # Returns 'azure' or 'local'
query(sql)                   # Raw DuckDB SQL query
```

### Azure Blob Structure
```
personal-data/athletics/
├── master.parquet          # 2.3M scraped records
├── ksa_profiles.parquet    # 152 KSA athlete profiles
├── benchmarks.parquet      # Championship standards
├── road_to_tokyo.parquet   # Qualification tracking
├── baseline/               # Protected backups (never auto-deleted)
└── backups/                # Rolling backups (7 daily, 4 weekly)
```

### AI Chatbot (Tab 8)

Uses OpenRouter free models via OpenAI SDK:
- `OPENROUTER_API_KEY` in `.env` or Streamlit secrets
- Free models: DeepSeek R1 70B, Llama 3.1, Gemma 2, Mistral 7B, Qwen 2
- RAG context from `AthleticsContextBuilder` class
- **Note**: `st.chat_input()` doesn't work in tabs - uses `st.text_input()` + button

## Environment Variables

```bash
# .env file (local) or Streamlit secrets (cloud)
AZURE_STORAGE_CONNECTION_STRING=...  # For Parquet/Blob storage
AZURE_SQL_CONN=...                   # For SQL database (optional)
OPENROUTER_API_KEY=...               # For AI chatbot (Tab 8)
```

## Parquet Schema

**master.parquet** (2.3M rows):
- rank, result, wind, competitor, competitorurl, dob, nat, pos, venue, date
- resultscore, age, event, environment, gender, result_numeric

**ksa_profiles.parquet** (152 rows):
- athlete_id, full_name, gender, date_of_birth, primary_event
- profile_image_url, country_code, status, best_score, best_world_rank

**benchmarks.parquet** (49 rows):
- Event, Gender, Year, Gold/Silver/Bronze Standard, Final Standard (8th), Top 8 Average

## Backup Validation Thresholds
- master.parquet: minimum 2,000,000 rows
- ksa_profiles.parquet: minimum 100 rows
- benchmarks.parquet: minimum 40 rows

## Development Workflow

1. Edit `World_Ranking_Dash.py` for local development
2. Test with `streamlit run World_Ranking_Dash.py`
3. Copy finalized code to `World_Ranking_Deploy_v3.py`
4. Push to GitHub for Streamlit Cloud deployment

## Team Saudi Theme

```python
TEAL_PRIMARY = '#007167'   # Main brand color, headers
GOLD_ACCENT = '#a08e66'    # Highlights, PB markers
TEAL_DARK = '#005a51'      # Hover states, gradients
TEAL_LIGHT = '#009688'     # Secondary positive
GRAY_BLUE = '#78909C'      # Neutral
```

## GitHub Actions Workflows

| Workflow | Schedule | Description |
|----------|----------|-------------|
| `weekly_sync.yml` | Sundays 2am UTC | Automated data sync with backup |
| `backup_baseline.yml` | Manual | Create protected baseline snapshot |
| `restore_backup.yml` | Manual | Restore from backup or baseline |

## Coach View Module (NEW)

Analytics modules ported from Tilastopaja project:

| Module | Purpose |
|--------|---------|
| `projection_engine.py` | Form scores, trend detection, projections, advancement probability |
| `historical_benchmarks.py` | Medal/final/semi/heat lines from championship data |
| `chart_components.py` | Plotly charts with Team Saudi styling |
| `coach_view.py` | Competition prep, athlete reports, competitor watch |

**Future Database Combination**: These modules are designed to work with both World Athletics and Tilastopaja data. A future merge will combine the databases for unified analysis.

## Historical Data Scraper

Scripts for filling the 2003-2019 data gap:

```bash
cd world_athletics_scraperv2
python scrape_historical.py --test           # Test mode (2019 only)
python scrape_historical.py --year 2015      # Single year
python scrape_historical.py                  # Full 2003-2019 (overnight)

# After scraping:
cd ..
python merge_historical_and_upload.py --dry-run  # Preview merge
python merge_historical_and_upload.py            # Merge + upload to Azure
```

## Additional Documentation

- `AZURE_BLOB_DEPLOYMENT_GUIDE.md` - Azure Blob Storage + Parquet + DuckDB setup
- `docs/plans/` - Design documents for planned features

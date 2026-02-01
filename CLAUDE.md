# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

World Athletics data scraping and visualization system for Team Saudi. Scrapes athlete rankings, results, and qualification data from worldathletics.org, stores in Parquet/SQLite databases, and displays via Streamlit dashboards with 10 analytics tabs including an AI chatbot with document RAG, Coach View, and Project East 2026 strategy planning.

## Common Commands

### Running Dashboards
```bash
streamlit run World_Ranking_Dash.py      # Local development
streamlit run World_Ranking_Deploy_v3.py # Production version (10 tabs)
```

### Data Pipeline (in order)
```bash
python "World Rankings_All.py"           # Scrape global rankings -> Data/*.csv
python "World Rankings_KSA_Ave_both.py"  # Scrape KSA athlete details -> Data/*.csv
python "SQL Converter2.py"               # Convert CSVs -> SQL/*.db
python convert_to_parquet.py             # Convert to Parquet + upload to Azure
```

### NotebookLM Briefings
```bash
python generate_briefings.py             # Generate briefings from Parquet data
python generate_briefings.py --upload    # Generate + upload to NotebookLM

# Manual upload (if auth expired, run: notebooklm login)
notebooklm source add "briefings/00_combined_briefing.md" --notebook d7034cab-0282-4b95-b960-d8f5e40d90e1
```

**Generated Files:**
- `briefings/00_combined_briefing.md` - All data combined
- `briefings/01_athlete_overview.md` - Top 20 athletes by rank
- `briefings/02_gap_analysis.md` - Gap to medal standards
- `briefings/03_competitor_intelligence.md` - Key rivals
- `briefings/04_asian_games_focus.md` - Project East 2026 targets
- `briefings/athletes/*.md` - 29 individual athlete profiles

### Backup Management
```bash
python backup_manager.py validate        # Check data before upload
python backup_manager.py upload          # Validate + backup + upload
python backup_manager.py list            # List Azure backups
python backup_manager.py baseline        # Create protected snapshot
python backup_manager.py restore --baseline  # Restore from baseline
```

### Testing Data Connector
```bash
python data_connector.py                 # Run diagnostics and sample queries
```

## Architecture

### Dashboard Structure (10 Tabs)

| Tab | Name | Key Module/Function |
|-----|------|---------------------|
| 1 | Event Standards & Progression | `WhatItTakesToWin` class - championship filtering, 1-8th progression |
| 2 | Athlete Profiles | `get_ksa_athletes()` - WPA ranking points, multi-event summary |
| 3 | Combined Rankings | `get_rankings_data()` - paginated, sorted by rank |
| 4 | Saudi Athletes Rankings | `get_ksa_rankings()` - country comparison |
| 5 | World Champs Qualification | `get_road_to_tokyo_data()` - near miss alerts |
| 6 | Major Games Analytics | `MajorGamesAnalyzer` class |
| 7 | What It Takes to Win (Live) | `WhatItTakesToWin.generate_report()` |
| 8 | AI Analyst | `OpenRouterClient` + `DocumentRAG` for semantic search |
| 9 | Coach View | `render_coach_view()` - competition prep, projections |
| 10 | Project East 2026 | Asian Games strategy with cached data loading |

### Data Flow
```
worldathletics.org (Selenium scrapers)
        |
   Data/*.csv (raw scraped data)
        |
   Data/parquet/*.parquet (modern) OR SQL/*.db (legacy)
        |
   Azure Blob Storage (athletics/)  <- for Streamlit Cloud
        |
   Streamlit Dashboard (DuckDB queries Parquet directly)
```

### Key Modules

| File | Purpose |
|------|---------|
| `World_Ranking_Deploy_v3.py` | Main dashboard with 10 tabs |
| `data_connector.py` | DuckDB wrapper - Azure/local dual-mode with TTL caching |
| `athletics_analytics_agents.py` | Analytics classes (SprintsAnalyzer, MajorGamesAnalyzer) |
| `athletics_chatbot.py` | AI chatbot with OpenRouter LLM integration |
| `document_rag.py` | RAG module for semantic search over PDFs and Parquet data |
| `what_it_takes_to_win.py` | `WhatItTakesToWin` class for medal standards |
| `coach_view.py` | Competition prep, athlete reports, competitor watch |
| `convert_to_parquet.py` | CSV/DB -> Parquet + Azure upload |
| `backup_manager.py` | Backup validation, retention, restore |

### data_connector.py Key Functions
```python
# Core data access
get_ksa_athletes()           # KSA athlete profiles (sorted by world rank)
get_rankings_data()          # Global rankings (2.3M records)
get_ksa_rankings()           # KSA-filtered rankings from master
get_benchmarks_data()        # Championship standards
get_road_to_tokyo_data()     # Qualification tracking data

# Analysis helpers
get_competitors()            # Top competitors by event/gender/season
get_head_to_head()           # H2H comparison between athletes
get_gap_analysis()           # Gap to qualification standards
get_athlete_results()        # Competition results for an athlete
get_event_list()             # Unique events in database
get_country_list()           # Unique countries in database

# Utilities
get_data_mode()              # Returns 'azure' or 'local'
query(sql)                   # Raw DuckDB SQL query
clear_cache()                # Clear TTL cache
get_cache_status()           # Debug cache state
test_connection()            # Diagnostics
```

### Azure Blob Structure
```
personal-data/athletics/
├── master.parquet           # 2.3M scraped records
├── ksa_profiles.parquet     # 152 KSA athlete profiles
├── benchmarks.parquet       # Championship standards
├── road_to_tokyo.parquet    # Qualification tracking
├── documents/               # PDF rulebooks for RAG
├── embeddings/              # Vector embeddings for semantic search
├── baseline/                # Protected backups (never auto-deleted)
└── backups/                 # Rolling backups (7 daily, 4 weekly)
```

### AI Chatbot (Tab 8)

**Three AI Backend Modes:**
- **Hybrid** (Recommended): NotebookLM documents + Live database combined
- **NotebookLM**: Fast, citation-backed answers from uploaded briefings/rulebooks
- **OpenRouter**: General LLM with live Parquet data context

**NotebookLM Integration:**
```bash
# Install (one-time)
pip install notebooklm-py

# Authenticate (opens browser for Google sign-in)
notebooklm login

# Check authentication status
notebooklm list

# Re-authenticate if expired
notebooklm logout && notebooklm login
```

**NotebookLM Notebook ID:** `d7034cab-0282-4b95-b960-d8f5e40d90e1`

**Upload briefings to NotebookLM:**
```bash
# Generate briefings from Parquet data
python generate_briefings.py

# Upload to NotebookLM (after authentication)
notebooklm source add "briefings/00_combined_briefing.md" --notebook d7034cab-0282-4b95-b960-d8f5e40d90e1

# Upload athlete profiles
for f in briefings/athletes/*.md; do notebooklm source add "$f" --notebook d7034cab-0282-4b95-b960-d8f5e40d90e1; done
```

**OpenRouter Configuration:**
- `OPENROUTER_API_KEY` in `.env` or Streamlit secrets
- Default model: Llama 3.2 3B (fastest) - configurable in UI
- RAG context from `AthleticsContextBuilder` class

**Performance Optimizations:**
- **Response caching** with 5-minute TTL (`ResponseCache` class) - instant repeated queries
- **Query intent classification** (`classify_query_intent()`) - builds smaller context for simple queries
- **Lazy RAG loading** - only loads embeddings when "Rulebooks" knowledge source selected
- Streaming responses for real-time display (no waiting for full response)
- Context caching (FIFO, 50 entries max)
- Filter dropdown caching via `get_ai_filter_options()` (30 min TTL)
- Temperature 0.3 for factual analytics (reduced hallucination)

**Query Intent Types:**
- `athlete` - Detects KSA athlete names, focuses context on athlete data
- `event` - Detects event names (100m, pole vault, etc.), focuses on rankings
- `standards` - Qualification/medal keywords, includes benchmarks
- `comparison` - vs/compare/gap keywords, includes both rankings and benchmarks
- `general` - Uses minimal context for faster response

**Quick Action Buttons:**
- Medal Gap Analysis - Gaps to Asian Games medal standards
- Top Rivals - Key Asian competitors
- Form Trends - Recent performance analysis
- Qualification Status - Progress toward entry standards

**Available Models:**
- Llama 3.2 3B (Fastest)
- DeepSeek Chat v3 (Fast)
- Gemini 2.0 Flash (Google)
- Llama 3.3 70B (Best Quality)
- Qwen 2.5 VL 7B (Multilingual)
- DeepSeek R1 (Reasoning)

**Note**: `st.chat_input()` doesn't work in tabs - uses `st.text_input()` + button

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

**Official Brand Colors** (use these, not teal):
```python
SAUDI_GREEN = '#005430'       # Official PMS 3425 C - Main brand color
GOLD_ACCENT = '#a08e66'       # Highlights, PB markers
SAUDI_GREEN_DARK = '#003d1f'  # Hover states, gradients
SAUDI_GREEN_LIGHT = '#2A8F5C' # Secondary positive
GRAY_BLUE = '#78909C'         # Neutral
```

See global `CLAUDE.md` for full theme documentation including header templates, metric cards, and Plotly styling.

## GitHub Actions Workflows

| Workflow | Schedule | Description |
|----------|----------|-------------|
| `weekly_sync.yml` | Sundays 2am UTC | Automated data sync with backup |
| `backup_baseline.yml` | Manual | Create protected baseline snapshot |
| `restore_backup.yml` | Manual | Restore from backup or baseline |

## Coach View Module

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

## Known Issues / Notes

- Championship key mismatch: Use `'World Champs'` not `'World Championships'` in qualification tab filters
- `st.chat_input()` doesn't work inside tabs - chatbot uses `st.text_input()` with button
- DuckDB Azure extension has SSL issues on Streamlit Cloud - use `_download_parquet_from_azure()` instead
- **Column naming**: `pos` column has finishing positions (1, 2, 3...), `rank` column has world ranking numbers (1012, 4954...)
- **Event name formats**: Dropdowns show '100m' but data uses '100-metres' - use normalized matching with regex
- **Theme colors**: Code uses `TEAL_PRIMARY = '#007167'`, not the official Saudi green `#005430` - both are acceptable

## Performance Notes

**Startup Time Optimization:**
- `sentence_transformers` import is now lazy-loaded in `document_rag.py` (saves ~12 seconds)
- Model `SentenceTransformer('all-MiniLM-L6-v2')` only loads on first RAG search (saves ~30 seconds)
- Azure Parquet downloads are cached with TTL - first load takes ~10-15 seconds per file

**Known Bottlenecks:**
- Initial Azure data load: ~30-40 seconds total for all parquet files (network latency)
- master.parquet: 33MB, downloads in ~10 chunks
- First page render after data load can be slow due to Streamlit rerun

## Additional Documentation

- `AZURE_BLOB_DEPLOYMENT_GUIDE.md` - Azure Blob Storage + Parquet + DuckDB setup
- `docs/plans/` - Design documents for planned features

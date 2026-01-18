# World Athletics Data System

World Athletics data scraping and visualization system for Team Saudi. Scrapes athlete rankings, results, and qualification data from worldathletics.org.

## Features

- **Data Pipeline**: Automated scraping from World Athletics website
- **Cloud Storage**: Azure Blob Storage with Parquet format
- **Dashboards**: Streamlit-based analytics dashboards
- **Backup System**: Automated backups with validation and retention policies

## Quick Start

### Local Development

1. Clone the repository
2. Copy `.env.example` to `.env` and add your Azure credentials
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the dashboard:
   ```bash
   streamlit run World_Ranking_Dash.py
   ```

### Data Management

```bash
# Validate local data before upload
python backup_manager.py validate

# Upload with automatic backup
python backup_manager.py upload

# List all backups
python backup_manager.py list

# Create protected baseline
python backup_manager.py baseline

# Restore from baseline
python backup_manager.py restore --baseline
```

## GitHub Actions Workflows

| Workflow | Schedule | Description |
|----------|----------|-------------|
| `weekly_sync.yml` | Sundays 2am UTC | Automated data sync with backup |
| `backup_baseline.yml` | Manual | Create protected baseline snapshot |
| `restore_backup.yml` | Manual | Restore from backup or baseline |

### Required Secrets

Add these in GitHub repo Settings > Secrets:

- `AZURE_STORAGE_CONNECTION_STRING` - Azure Blob Storage connection string

## Azure Blob Structure

```
personal-data/athletics/
├── master.parquet          # 2.3M scraped records
├── ksa_profiles.parquet    # KSA athlete profiles
├── benchmarks.parquet      # Championship standards
├── baseline/               # Protected backups
└── backups/                # Rolling backups (7 daily, 4 weekly)
```

## Backup Policy

- **Validation**: Blocks uploads if data is below minimum thresholds
- **Rolling Retention**: 7 daily + 4 weekly backups
- **Baseline**: Protected snapshot that never auto-deletes
- **Restore**: Can restore from any backup point

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development guide
- [AZURE_BLOB_DEPLOYMENT_GUIDE.md](AZURE_BLOB_DEPLOYMENT_GUIDE.md) - Azure setup guide

## Team Saudi Theme

Primary Teal: `#007167` | Gold Accent: `#a08e66` | Dark Teal: `#005a51`

# Azure Blob Storage Deployment Guide

A step-by-step guide to add Azure Blob Storage with Parquet + DuckDB to any Python project.

---

## Overview

| Component | Purpose |
|-----------|---------|
| Azure Blob Storage | Cloud storage for data (free tier: 5 GB) |
| Parquet format | Compressed columnar storage (10x smaller than CSV) |
| DuckDB | Fast SQL queries on Parquet data |
| GitHub Actions | Automated weekly data sync |

---

## Step 1: Azure Storage Account Setup

### 1.1 Create Storage Account (One-time)

1. Go to [Azure Portal](https://portal.azure.com)
2. Click **Create a resource** > **Storage account**
3. Settings:
   - **Subscription**: Your subscription
   - **Resource group**: Create new or use existing
   - **Storage account name**: `yourprojectname` (lowercase, no spaces)
   - **Region**: Choose closest to you
   - **Performance**: Standard
   - **Redundancy**: LRS (Locally-redundant storage) - cheapest
4. Click **Review + Create** > **Create**

### 1.2 Get Connection String

1. Go to your Storage Account
2. Click **Access keys** (left sidebar under Security + networking)
3. Click **Show** next to Connection string
4. Copy the entire connection string

It looks like:
```
DefaultEndpointsProtocol=https;AccountName=yourproject;AccountKey=abc123...;EndpointSuffix=core.windows.net
```

### 1.3 Create Container

1. In Storage Account, click **Containers** (left sidebar)
2. Click **+ Container**
3. Name: `your-data` (e.g., `fencing-data`, `taekwondo-data`)
4. Access level: **Private**
5. Click **Create**

---

## Step 2: Project Setup

### 2.1 Install Dependencies

Add to your `requirements.txt`:
```
azure-storage-blob>=12.19.0
azure-identity>=1.15.0
pandas>=2.0.0
pyarrow>=14.0.0
duckdb>=0.9.0
python-dotenv>=1.0.0
```

Install:
```bash
pip install -r requirements.txt
```

### 2.2 Create .env File

Create `.env` in your project root:
```env
# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=yourproject;AccountKey=YOUR_KEY_HERE;EndpointSuffix=core.windows.net

# Optional: Force local data (skip Azure)
# FORCE_LOCAL_DATA=true
```

### 2.3 Create .env.template

Create `.env.template` for others to copy:
```env
# Azure Blob Storage - Get from Azure Portal > Storage Account > Access Keys
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here

# Optional: Force local data instead of Azure
# FORCE_LOCAL_DATA=true
```

### 2.4 Update .gitignore

Add these lines to `.gitignore`:
```gitignore
# Environment variables - NEVER COMMIT
.env
.env.local
*.env

# Data files - stored in Azure, not git
*.csv
*.parquet
archive/

# Local databases
*.db
*.sqlite
```

---

## Step 3: Add blob_storage.py

Copy this file to your project and update the configuration section:

```python
"""
Azure Blob Storage Module
Supports Parquet files with DuckDB queries
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional
from io import BytesIO

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Azure imports
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Warning: azure-storage-blob not installed")

# DuckDB import
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Warning: duckdb not installed")


# =============================================================================
# CONFIGURATION - UPDATE THESE FOR YOUR PROJECT
# =============================================================================

CONTAINER_NAME = "your-data"           # Your container name
MASTER_FILE = "master.parquet"         # Main data file
STORAGE_ACCOUNT_URL = "https://yourproject.blob.core.windows.net/"

# =============================================================================


# Connection string (lazy-loaded)
_CONN_STRING = None


def _get_connection_string() -> Optional[str]:
    """Get Azure Storage connection string from env or Streamlit secrets."""
    global _CONN_STRING

    if _CONN_STRING is not None:
        return _CONN_STRING

    # Try environment variable
    _CONN_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    # Try Streamlit secrets
    if not _CONN_STRING:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'AZURE_STORAGE_CONNECTION_STRING' in st.secrets:
                _CONN_STRING = st.secrets['AZURE_STORAGE_CONNECTION_STRING']
        except:
            pass

    return _CONN_STRING


def _use_azure() -> bool:
    """Check if Azure should be used."""
    if os.getenv('FORCE_LOCAL_DATA', '').lower() in ('true', '1', 'yes'):
        return False
    return bool(_get_connection_string()) and AZURE_AVAILABLE


def get_blob_service() -> Optional['BlobServiceClient']:
    """Get Azure Blob Service client."""
    if not AZURE_AVAILABLE:
        return None

    conn_str = _get_connection_string()
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)

    return None


def get_container_client(create_if_missing: bool = True):
    """Get container client."""
    blob_service = get_blob_service()
    if not blob_service:
        return None

    container = blob_service.get_container_client(CONTAINER_NAME)

    if create_if_missing:
        try:
            if not container.exists():
                container.create_container()
                print(f"Created container: {CONTAINER_NAME}")
        except:
            pass

    return container


def download_parquet(blob_path: str) -> Optional[pd.DataFrame]:
    """Download a parquet file from Azure."""
    container = get_container_client()
    if not container:
        return None

    try:
        blob_client = container.get_blob_client(blob_path)
        data = blob_client.download_blob().readall()
        return pd.read_parquet(BytesIO(data))
    except Exception as e:
        print(f"Error downloading {blob_path}: {e}")
        return None


def _clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame for Parquet compatibility."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('').astype(str)
            df[col] = df[col].replace('nan', '')
    return df


def upload_parquet(df: pd.DataFrame, blob_path: str, overwrite: bool = True) -> bool:
    """Upload DataFrame as parquet to Azure."""
    container = get_container_client()
    if not container:
        return False

    try:
        # Clean data
        df = _clean_dataframe_for_parquet(df)

        buffer = BytesIO()
        df.to_parquet(buffer, index=False, compression='gzip')
        buffer.seek(0)

        file_size_mb = buffer.getbuffer().nbytes / (1024 * 1024)
        print(f"Uploading {len(df):,} rows ({file_size_mb:.1f} MB)...")

        blob_client = container.get_blob_client(blob_path)
        blob_client.upload_blob(
            buffer,
            overwrite=overwrite,
            max_concurrency=4,
            timeout=600
        )
        print(f"Uploaded to {blob_path}")
        return True
    except Exception as e:
        print(f"Error uploading: {e}")
        return False


def create_backup() -> Optional[str]:
    """Create backup of master file."""
    container = get_container_client()
    if not container:
        return None

    try:
        blob_client = container.get_blob_client(MASTER_FILE)
        if not blob_client.exists():
            print("No master file to backup")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"backups/backup_{timestamp}.parquet"

        backup_client = container.get_blob_client(backup_path)
        backup_client.start_copy_from_url(blob_client.url)

        print(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Backup error: {e}")
        return None


def load_data() -> pd.DataFrame:
    """Load data from Azure (or local fallback)."""
    if _use_azure():
        print("Loading from Azure Blob Storage...")
        df = download_parquet(MASTER_FILE)
        if df is not None and not df.empty:
            print(f"Loaded {len(df):,} rows from Azure")
            return df
        print("Azure empty, falling back to local...")

    return _load_local_csv()


def _load_local_csv() -> pd.DataFrame:
    """Load from local CSV files (fallback)."""
    from pathlib import Path
    import re

    all_dfs = []
    csv_files = sorted(Path('.').glob('*.csv'))

    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)

            # Extract year from filename if present
            year_match = re.search(r'(\d{4})', f.name)
            if year_match and 'year' not in df.columns:
                df['year'] = int(year_match.group(1))

            all_dfs.append(df)
            print(f"Loaded {len(df):,} rows from {f.name}")
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"Total: {len(combined):,} rows")
        return combined

    return pd.DataFrame()


def save_data(df: pd.DataFrame, append: bool = False) -> bool:
    """Save data to Azure."""
    if not _use_azure():
        print("Azure not configured, saving locally")
        df.to_parquet('data.parquet', index=False)
        return True

    if append:
        existing = download_parquet(MASTER_FILE)
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            # Remove duplicates if you have an ID column
            # df = df.drop_duplicates(subset=['id'], keep='last')

    return upload_parquet(df, MASTER_FILE)


def migrate_csv_to_azure() -> bool:
    """Migrate local CSV files to Azure."""
    print("=" * 60)
    print("MIGRATING CSV TO AZURE BLOB STORAGE")
    print("=" * 60)

    df = _load_local_csv()
    if df.empty:
        print("No data to migrate")
        return False

    # Create backup first
    create_backup()

    # Upload
    success = upload_parquet(df, MASTER_FILE)

    if success:
        print(f"\nMigration complete: {len(df):,} rows uploaded")

    return success


def get_storage_usage() -> dict:
    """Get storage usage statistics."""
    container = get_container_client()
    if not container:
        return {'error': 'Not connected'}

    total_size = 0
    files = []

    for blob in container.list_blobs():
        size_mb = blob.size / (1024 * 1024)
        total_size += blob.size
        files.append({'name': blob.name, 'size_mb': round(size_mb, 2)})

    return {
        'total_mb': round(total_size / (1024 * 1024), 2),
        'total_gb': round(total_size / (1024 * 1024 * 1024), 3),
        'free_tier_limit_gb': 5,
        'percent_used': round((total_size / (5 * 1024 * 1024 * 1024)) * 100, 1),
        'files': files
    }


# =============================================================================
# DUCKDB QUERY SUPPORT
# =============================================================================

_duckdb_conn = None
_duckdb_ready = False


def get_duckdb_connection():
    """Get DuckDB connection with data loaded."""
    global _duckdb_conn, _duckdb_ready

    if not DUCKDB_AVAILABLE:
        print("DuckDB not available")
        return None

    if _duckdb_conn is not None and _duckdb_ready:
        return _duckdb_conn

    try:
        _duckdb_conn = duckdb.connect(':memory:')

        print("Loading data into DuckDB...")
        df = load_data()

        if df.empty:
            print("No data available")
            return None

        _duckdb_conn.register('data', df)
        _duckdb_ready = True

        print(f"DuckDB ready with {len(df):,} rows in 'data' table")
        return _duckdb_conn

    except Exception as e:
        print(f"DuckDB error: {e}")
        return None


def query(sql: str) -> Optional[pd.DataFrame]:
    """Execute SQL query against data.

    The data is available as the 'data' table.

    Examples:
        query("SELECT * FROM data LIMIT 10")
        query("SELECT year, COUNT(*) FROM data GROUP BY year")
    """
    conn = get_duckdb_connection()
    if conn is None:
        return None

    try:
        return conn.execute(sql).fetchdf()
    except Exception as e:
        print(f"Query error: {e}")
        return None


def refresh_data():
    """Reload data from Azure into DuckDB."""
    global _duckdb_conn, _duckdb_ready

    if _duckdb_conn is not None:
        _duckdb_conn.close()
    _duckdb_conn = None
    _duckdb_ready = False

    return get_duckdb_connection()


# =============================================================================
# TEST CONNECTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AZURE BLOB STORAGE CONNECTION TEST")
    print("=" * 60)

    if _use_azure():
        print(f"\nContainer: {CONTAINER_NAME}")
        print(f"Storage URL: {STORAGE_ACCOUNT_URL}")

        usage = get_storage_usage()
        if 'error' not in usage:
            print(f"\nStorage Usage: {usage['total_mb']:.2f} MB ({usage['percent_used']:.1f}% of free tier)")
            print(f"\nFiles:")
            for f in usage['files']:
                print(f"  - {f['name']}: {f['size_mb']} MB")
            print("\nConnection: SUCCESS")
        else:
            print(f"\nConnection: FAILED - {usage['error']}")
    else:
        print("\nAzure not configured. Set AZURE_STORAGE_CONNECTION_STRING in .env")

    print("=" * 60)
```

---

## Step 4: Migrate Existing Data

### 4.1 Test Connection

```bash
python blob_storage.py
```

Expected output:
```
============================================================
AZURE BLOB STORAGE CONNECTION TEST
============================================================

Container: your-data
Storage URL: https://yourproject.blob.core.windows.net/

Storage Usage: 0.00 MB (0.0% of free tier)

Files:

Connection: SUCCESS
============================================================
```

### 4.2 Migrate CSV Data

```python
from blob_storage import migrate_csv_to_azure

migrate_csv_to_azure()
```

This will:
1. Load all local CSV files
2. Create a backup (if master exists)
3. Upload as compressed Parquet

---

## Step 5: GitHub Actions (Automated Sync)

### 5.1 Create Workflow File

Create `.github/workflows/sync.yml`:

```yaml
name: Data Sync

on:
  schedule:
    # Run weekly on Sunday at 2 AM UTC
    - cron: '0 2 * * 0'
  workflow_dispatch:
    inputs:
      full_refresh:
        description: 'Full data refresh'
        required: false
        default: 'false'

jobs:
  sync:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run sync
        env:
          AZURE_STORAGE_CONNECTION_STRING: ${{ secrets.AZURE_STORAGE_CONNECTION_STRING }}
        run: python your_scraper.py

      - name: Verify data
        env:
          AZURE_STORAGE_CONNECTION_STRING: ${{ secrets.AZURE_STORAGE_CONNECTION_STRING }}
        run: python -c "from blob_storage import get_storage_usage; print(get_storage_usage())"
```

### 5.2 Add GitHub Secret

1. Go to your GitHub repo
2. Click **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Name: `AZURE_STORAGE_CONNECTION_STRING`
5. Value: Your connection string from `.env`

---

## Step 6: Usage Examples

### Load Data

```python
from blob_storage import load_data

df = load_data()
print(f"Loaded {len(df):,} rows")
```

### SQL Queries with DuckDB

```python
from blob_storage import query

# Count by year
df = query("SELECT year, COUNT(*) as count FROM data GROUP BY year ORDER BY year")

# Filter data
df = query("SELECT * FROM data WHERE country = 'KSA' ORDER BY date DESC")

# Aggregations
df = query("""
    SELECT
        athlete_name,
        COUNT(*) as competitions,
        MIN(time) as best_time
    FROM data
    GROUP BY athlete_name
    ORDER BY best_time
""")
```

### Check Storage Usage

```python
from blob_storage import get_storage_usage

usage = get_storage_usage()
print(f"Using {usage['total_mb']:.1f} MB ({usage['percent_used']:.1f}% of 5 GB free tier)")
```

### Manual Backup

```python
from blob_storage import create_backup

backup_path = create_backup()
print(f"Backup created: {backup_path}")
```

---

## Quick Checklist

- [ ] Create Azure Storage Account
- [ ] Create Container
- [ ] Copy connection string to `.env`
- [ ] Add `blob_storage.py` to project
- [ ] Update configuration in `blob_storage.py`:
  - [ ] `CONTAINER_NAME`
  - [ ] `MASTER_FILE`
  - [ ] `STORAGE_ACCOUNT_URL`
- [ ] Add dependencies to `requirements.txt`
- [ ] Update `.gitignore` (exclude .env, *.csv, archive/)
- [ ] Run `python blob_storage.py` to test
- [ ] Run migration: `migrate_csv_to_azure()`
- [ ] Add GitHub secret for Actions
- [ ] Create `.github/workflows/sync.yml`

---

## Step 7: Streamlit Cloud Deployment

### 7.1 Configure Streamlit Secrets

1. Deploy your app to [Streamlit Cloud](https://share.streamlit.io)
2. Go to your app's **Settings** (gear icon in bottom right, or "Manage app")
3. Click **Secrets** in the left sidebar
4. Add your connection string in **TOML format**:

```toml
AZURE_STORAGE_CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=yourproject;AccountKey=YOUR_KEY_HERE;EndpointSuffix=core.windows.net'
```

**Important:** Use single quotes `'...'` to avoid TOML parsing issues with the `==` in the AccountKey.

5. Click **Save**
6. Reboot the app

### 7.2 Verify Secrets Are Loaded

Add this debug code temporarily to check secrets are working:

```python
import streamlit as st

st.write("Secrets check:")
if 'AZURE_STORAGE_CONNECTION_STRING' in st.secrets:
    st.success("Connection string found!")
else:
    st.error("Connection string NOT found")
    st.write(f"Available secrets: {list(st.secrets.keys())}")
```

### 7.3 Create secrets.toml.example

Create `.streamlit/secrets.toml.example` for documentation:

```toml
# Streamlit Cloud Secrets Template
# Copy this to secrets.toml for local testing
# On Streamlit Cloud: Settings > Secrets

AZURE_STORAGE_CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=YOUR_ACCOUNT;AccountKey=YOUR_KEY;EndpointSuffix=core.windows.net'
```

### 7.4 Update .gitignore

```gitignore
# Streamlit secrets (NEVER COMMIT)
.streamlit/secrets.toml
```

### 7.5 Headless Environment Detection

The `blob_storage.py` module automatically detects Streamlit Cloud and skips interactive browser authentication. It checks for:
- `/mount/src` directory (Streamlit Cloud runs from here)
- `CI` or `GITHUB_ACTIONS` environment variables
- `STREAMLIT_SERVER_HEADLESS` environment variable

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `azure-storage-blob not installed` | Run `pip install azure-storage-blob` |
| `Connection string not found` | Check `.env` file exists and has correct format |
| `Container not found` | Create container in Azure Portal |
| `Upload timeout` | Large file - increase `timeout` parameter |
| `Parquet type error` | Data has mixed types - `_clean_dataframe_for_parquet()` handles this |
| `Invalid TOML format` (Streamlit) | Use single quotes around connection string |
| `InteractiveBrowserCredential failed` | Not on Streamlit Cloud - add connection string to secrets |
| `No data found` on Streamlit Cloud | Check secrets are configured correctly (Settings > Secrets) |

---

## Cost (Free Tier)

| Resource | Free Tier Limit | Typical Usage |
|----------|-----------------|---------------|
| Storage | 5 GB | ~50 MB per sport |
| Reads | 20,000/month | ~1000/month |
| Writes | 10,000/month | ~100/month |

You can run 50-100 sport projects on the free tier.

---

## Projects Using This

- Athletics (World Rankings)
- Swimming Analytics
- Fencing Performance
- Taekwondo Data

---

*Last updated: January 2026*

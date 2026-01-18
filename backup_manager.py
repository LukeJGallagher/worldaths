"""
Backup Manager for Athletics Parquet Data

Features:
- Pre-upload validation (minimum row counts)
- Rolling retention (7 daily, 4 weekly backups)
- Protected baseline snapshot
- Restore capability

Usage:
    python backup_manager.py validate     # Check new data before upload
    python backup_manager.py upload       # Validate + upload + manage backups
    python backup_manager.py list         # List all backups
    python backup_manager.py restore      # Restore from backup
    python backup_manager.py baseline     # Create/update baseline snapshot
"""

import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# Azure config
CONN_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "personal-data"
AZURE_FOLDER = "athletics"
BACKUP_FOLDER = f"{AZURE_FOLDER}/backups"
BASELINE_FOLDER = f"{AZURE_FOLDER}/baseline"

# Local paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(BASE_DIR, "Data", "parquet")

# Minimum row counts for validation (prevents uploading empty/corrupted data)
MIN_ROW_COUNTS = {
    "master.parquet": 2_000_000,      # Should have ~2.3M records
    "ksa_profiles.parquet": 100,       # Should have ~152 KSA athletes
    "benchmarks.parquet": 40,          # Should have ~49 benchmark records
}

# Retention policy
DAILY_RETENTION = 7   # Keep last 7 daily backups
WEEKLY_RETENTION = 4  # Keep last 4 weekly backups


def get_blob_service():
    """Get Azure Blob Service client."""
    if not CONN_STRING:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set in environment")

    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(CONN_STRING)


def validate_local_data():
    """Validate local parquet files before upload."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("VALIDATING LOCAL DATA")
    print("=" * 60)

    all_valid = True
    results = {}

    for filename, min_rows in MIN_ROW_COUNTS.items():
        local_path = os.path.join(PARQUET_DIR, filename)

        if not os.path.exists(local_path):
            print(f"  {filename}: MISSING")
            results[filename] = {"status": "missing", "rows": 0}
            all_valid = False
            continue

        try:
            df = pd.read_parquet(local_path)
            row_count = len(df)

            if row_count >= min_rows:
                print(f"  {filename}: {row_count:,} rows (min: {min_rows:,})")
                results[filename] = {"status": "valid", "rows": row_count}
            else:
                print(f"  {filename}: {row_count:,} rows - BELOW MINIMUM ({min_rows:,})")
                results[filename] = {"status": "below_minimum", "rows": row_count}
                all_valid = False

        except Exception as e:
            print(f"  {filename}: ERROR - {e}")
            results[filename] = {"status": "error", "rows": 0, "error": str(e)}
            all_valid = False

    if all_valid:
        print("\n  All files VALID for upload")
    else:
        print("\n  VALIDATION FAILED - Upload blocked")

    return all_valid, results


def list_backups():
    """List all backups in Azure."""
    print("\n" + "=" * 60)
    print("AZURE BACKUPS")
    print("=" * 60)

    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    # Get all backups
    backups = defaultdict(list)

    for blob in container_client.list_blobs(name_starts_with=BACKUP_FOLDER):
        if '.parquet' in blob.name and '.placeholder' not in blob.name:
            # Parse filename: master.parquet_20260118_085811
            parts = blob.name.split('/')[-1].rsplit('_', 2)
            if len(parts) >= 3:
                base_name = parts[0]  # e.g., master.parquet
                date_str = parts[1]   # e.g., 20260118
                time_str = parts[2]   # e.g., 085811

                backups[base_name].append({
                    'name': blob.name,
                    'date': date_str,
                    'time': time_str,
                    'size_mb': blob.size / (1024 * 1024),
                    'last_modified': blob.last_modified,
                })

    # Display backups by file
    for base_name in sorted(backups.keys()):
        print(f"\n  {base_name}:")
        for backup in sorted(backups[base_name], key=lambda x: x['date'] + x['time'], reverse=True):
            print(f"    {backup['date']} {backup['time']} - {backup['size_mb']:.2f} MB")

    # Check baseline
    print(f"\n  BASELINE:")
    for blob in container_client.list_blobs(name_starts_with=BASELINE_FOLDER):
        if blob.name.endswith('.parquet'):
            size_mb = blob.size / (1024 * 1024)
            print(f"    {blob.name.split('/')[-1]} - {size_mb:.2f} MB ({blob.last_modified.strftime('%Y-%m-%d')})")

    return backups


def create_backup(blob_service, container_client, filename):
    """Create a timestamped backup of current Azure file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    source_path = f"{AZURE_FOLDER}/{filename}"
    backup_path = f"{BACKUP_FOLDER}/{filename}_{timestamp}"

    try:
        source_blob = container_client.get_blob_client(source_path)
        if source_blob.exists():
            backup_blob = container_client.get_blob_client(backup_path)
            backup_blob.start_copy_from_url(source_blob.url)
            print(f"    Backed up: {filename} -> {backup_path.split('/')[-1]}")
            return backup_path
    except Exception as e:
        print(f"    Backup failed for {filename}: {e}")

    return None


def cleanup_old_backups(container_client):
    """Apply retention policy - keep 7 daily + 4 weekly backups."""
    print("\n  Applying retention policy...")

    # Get all backups grouped by base filename
    backups_by_file = defaultdict(list)

    for blob in container_client.list_blobs(name_starts_with=BACKUP_FOLDER):
        if blob.name.endswith('.parquet') and '.placeholder' not in blob.name:
            parts = blob.name.split('/')[-1].rsplit('_', 2)
            if len(parts) >= 3:
                base_name = parts[0]
                try:
                    backup_date = datetime.strptime(parts[1], "%Y%m%d")
                    backups_by_file[base_name].append({
                        'name': blob.name,
                        'date': backup_date,
                        'blob': blob,
                    })
                except ValueError:
                    pass

    now = datetime.now()
    deleted_count = 0

    for base_name, backups in backups_by_file.items():
        # Sort by date, newest first
        backups.sort(key=lambda x: x['date'], reverse=True)

        keep_set = set()

        # Keep last 7 daily backups
        daily_kept = 0
        for backup in backups:
            if daily_kept < DAILY_RETENTION:
                keep_set.add(backup['name'])
                daily_kept += 1

        # Keep last 4 weekly backups (oldest backup from each week)
        weekly_kept = 0
        weeks_seen = set()
        for backup in sorted(backups, key=lambda x: x['date']):
            week_key = backup['date'].isocalendar()[:2]  # (year, week)
            if week_key not in weeks_seen and weekly_kept < WEEKLY_RETENTION:
                keep_set.add(backup['name'])
                weeks_seen.add(week_key)
                weekly_kept += 1

        # Delete backups not in keep set
        for backup in backups:
            if backup['name'] not in keep_set:
                try:
                    blob_client = container_client.get_blob_client(backup['name'])
                    blob_client.delete_blob()
                    print(f"    Deleted: {backup['name'].split('/')[-1]}")
                    deleted_count += 1
                except Exception as e:
                    print(f"    Failed to delete {backup['name']}: {e}")

    print(f"    Cleaned up {deleted_count} old backups")


def upload_with_backup():
    """Validate, backup, upload, and cleanup."""
    print("\n" + "=" * 60)
    print("UPLOAD WITH BACKUP")
    print("=" * 60)

    # Step 1: Validate local data
    is_valid, results = validate_local_data()

    if not is_valid:
        print("\n  UPLOAD ABORTED - Validation failed")
        print("  Fix the data issues and try again")
        return False

    # Step 2: Connect to Azure
    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    # Step 3: Backup existing files
    print("\n  Creating backups...")
    for filename in MIN_ROW_COUNTS.keys():
        create_backup(blob_service, container_client, filename)

    # Step 4: Upload new files
    print("\n  Uploading new files...")
    for filename in MIN_ROW_COUNTS.keys():
        local_path = os.path.join(PARQUET_DIR, filename)
        blob_path = f"{AZURE_FOLDER}/{filename}"

        if os.path.exists(local_path):
            blob_client = container_client.get_blob_client(blob_path)
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)

            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"    Uploaded: {filename} ({size_mb:.2f} MB)")

    # Step 5: Cleanup old backups
    cleanup_old_backups(container_client)

    print("\n  Upload complete!")
    return True


def create_baseline():
    """Create or update the baseline snapshot (protected backup)."""
    print("\n" + "=" * 60)
    print("CREATING BASELINE SNAPSHOT")
    print("=" * 60)

    # Validate first
    is_valid, results = validate_local_data()

    if not is_valid:
        print("\n  BASELINE ABORTED - Data validation failed")
        return False

    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    for filename in MIN_ROW_COUNTS.keys():
        local_path = os.path.join(PARQUET_DIR, filename)
        baseline_path = f"{BASELINE_FOLDER}/{filename}"

        if os.path.exists(local_path):
            blob_client = container_client.get_blob_client(baseline_path)
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)

            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  Baseline created: {filename} ({size_mb:.2f} MB)")

    print(f"\n  Baseline snapshot saved to {BASELINE_FOLDER}/")
    print("  This backup will NOT be auto-deleted by retention policy")
    return True


def restore_backup(backup_name=None, use_baseline=False):
    """Restore data from a backup."""
    print("\n" + "=" * 60)
    print("RESTORE FROM BACKUP")
    print("=" * 60)

    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    if use_baseline:
        # Restore from baseline
        print("  Restoring from BASELINE...")
        source_folder = BASELINE_FOLDER

        for blob in container_client.list_blobs(name_starts_with=source_folder):
            if blob.name.endswith('.parquet'):
                filename = blob.name.split('/')[-1]
                dest_path = f"{AZURE_FOLDER}/{filename}"

                source_blob = container_client.get_blob_client(blob.name)
                dest_blob = container_client.get_blob_client(dest_path)
                dest_blob.start_copy_from_url(source_blob.url)

                print(f"    Restored: {filename}")

        print("\n  Baseline restored successfully!")
        return True

    elif backup_name:
        # Restore specific backup
        print(f"  Restoring from backup: {backup_name}")

        # Find the backup
        for blob in container_client.list_blobs(name_starts_with=BACKUP_FOLDER):
            if backup_name in blob.name:
                # Extract original filename (e.g., master.parquet from master.parquet_20260118_085811)
                parts = blob.name.split('/')[-1].rsplit('_', 2)
                original_name = parts[0] + '.parquet'
                dest_path = f"{AZURE_FOLDER}/{original_name}"

                source_blob = container_client.get_blob_client(blob.name)
                dest_blob = container_client.get_blob_client(dest_path)
                dest_blob.start_copy_from_url(source_blob.url)

                print(f"    Restored: {original_name} from {blob.name.split('/')[-1]}")

        print("\n  Restore complete!")
        return True

    else:
        # Interactive restore - list available backups
        print("  Available backups:")
        backups = list_backups()

        print("\n  Usage:")
        print("    python backup_manager.py restore --baseline")
        print("    python backup_manager.py restore --backup master.parquet_20260118")
        return False


def print_usage():
    """Print usage instructions."""
    print("""
Athletics Backup Manager

Commands:
    python backup_manager.py validate     Validate local parquet files
    python backup_manager.py upload       Validate + upload + manage backups
    python backup_manager.py list         List all Azure backups
    python backup_manager.py baseline     Create protected baseline snapshot
    python backup_manager.py restore      Restore from backup

Restore options:
    --baseline                Restore from baseline snapshot
    --backup <name>           Restore specific backup (partial match)

Examples:
    python backup_manager.py restore --baseline
    python backup_manager.py restore --backup master.parquet_20260118

Retention Policy:
    - 7 daily backups retained
    - 4 weekly backups retained
    - Baseline snapshot never auto-deleted
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1].lower()

    try:
        if command == "validate":
            is_valid, _ = validate_local_data()
            sys.exit(0 if is_valid else 1)

        elif command == "upload":
            success = upload_with_backup()
            sys.exit(0 if success else 1)

        elif command == "list":
            list_backups()

        elif command == "baseline":
            success = create_baseline()
            sys.exit(0 if success else 1)

        elif command == "restore":
            if "--baseline" in sys.argv:
                restore_backup(use_baseline=True)
            elif "--backup" in sys.argv:
                idx = sys.argv.index("--backup")
                if idx + 1 < len(sys.argv):
                    restore_backup(backup_name=sys.argv[idx + 1])
                else:
                    print("Error: --backup requires a backup name")
                    sys.exit(1)
            else:
                restore_backup()

        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

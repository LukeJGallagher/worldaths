"""
Test Azure Blob Storage connection and list available data.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONN_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if not CONN_STRING:
    print("ERROR: AZURE_STORAGE_CONNECTION_STRING not found in .env")
    exit(1)

print("=" * 60)
print("AZURE BLOB STORAGE CONNECTION TEST")
print("=" * 60)

# Test 1: Basic connection with azure-storage-blob
print("\n1. Testing azure-storage-blob connection...")
try:
    from azure.storage.blob import BlobServiceClient

    blob_service = BlobServiceClient.from_connection_string(CONN_STRING)
    account_name = blob_service.account_name
    print(f"   Connected to account: {account_name}")

    # List containers
    print("\n2. Listing containers...")
    containers = list(blob_service.list_containers())
    for container in containers:
        print(f"   - {container['name']}")

    # Check for personal-data container
    print("\n3. Checking 'personal-data' container...")
    try:
        container_client = blob_service.get_container_client("personal-data")

        # List blobs in athletics folder
        print("\n4. Listing blobs in athletics/ folder...")
        blobs = list(container_client.list_blobs(name_starts_with="athletics/"))

        if blobs:
            for blob in blobs[:20]:  # First 20
                size_mb = blob.size / (1024 * 1024)
                print(f"   - {blob.name} ({size_mb:.2f} MB)")
            if len(blobs) > 20:
                print(f"   ... and {len(blobs) - 20} more files")
        else:
            print("   No files found in athletics/ folder")
            print("   (This is expected if data hasn't been uploaded yet)")

    except Exception as e:
        print(f"   Container 'personal-data' not found or error: {e}")
        print("   You may need to create this container first")

except ImportError:
    print("   ERROR: azure-storage-blob not installed")
    print("   Run: pip install azure-storage-blob")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: DuckDB Azure extension
print("\n" + "=" * 60)
print("5. Testing DuckDB Azure extension...")
try:
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL azure; LOAD azure;")
    print("   DuckDB Azure extension loaded successfully")

    # Try to set up Azure secret
    con.execute(f"""
        CREATE SECRET azure_secret (
            TYPE AZURE,
            CONNECTION_STRING '{CONN_STRING}'
        );
    """)
    print("   Azure secret created successfully")

    # Try to list files
    print("\n6. Testing DuckDB query to Azure...")
    try:
        result = con.execute("""
            SELECT * FROM glob('az://personal-data/athletics/*.parquet')
        """).fetchall()

        if result:
            print(f"   Found {len(result)} parquet files:")
            for row in result:
                print(f"   - {row[0]}")
        else:
            print("   No parquet files found in athletics/ folder")

    except Exception as e:
        print(f"   Query error (expected if no data yet): {e}")

    con.close()

except ImportError:
    print("   ERROR: duckdb not installed")
    print("   Run: pip install duckdb")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

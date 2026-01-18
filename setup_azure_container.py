"""
Create Azure Blob container and folder structure for athletics data.
"""
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContainerClient

# Load environment variables
load_dotenv()

CONN_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = "personal-data"

def create_container():
    """Create the personal-data container if it doesn't exist."""
    blob_service = BlobServiceClient.from_connection_string(CONN_STRING)

    print(f"Creating container: {CONTAINER_NAME}")
    try:
        container_client = blob_service.create_container(CONTAINER_NAME)
        print(f"   Container '{CONTAINER_NAME}' created successfully")
    except Exception as e:
        if "ContainerAlreadyExists" in str(e):
            print(f"   Container '{CONTAINER_NAME}' already exists")
        else:
            raise e

    return blob_service.get_container_client(CONTAINER_NAME)


def create_folder_structure(container_client: ContainerClient):
    """Create folder structure by uploading placeholder files."""
    folders = [
        "athletics/backups/.placeholder",
    ]

    print("\nCreating folder structure...")
    for folder_path in folders:
        try:
            blob_client = container_client.get_blob_client(folder_path)
            blob_client.upload_blob(b"", overwrite=True)
            print(f"   Created: {folder_path}")
        except Exception as e:
            print(f"   Error creating {folder_path}: {e}")


def list_contents(container_client: ContainerClient):
    """List all contents in the container."""
    print(f"\nContents of '{CONTAINER_NAME}':")
    blobs = list(container_client.list_blobs())
    if blobs:
        for blob in blobs:
            print(f"   - {blob.name}")
    else:
        print("   (empty)")


if __name__ == "__main__":
    print("=" * 60)
    print("AZURE BLOB STORAGE SETUP")
    print("=" * 60)

    container_client = create_container()
    create_folder_structure(container_client)
    list_contents(container_client)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run convert_to_parquet.py to create local Parquet files")
    print("2. Upload Parquet files to Azure using upload script")

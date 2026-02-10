"""
Download World Athletics rulebook PDFs and upload to Azure Blob Storage.

This script downloads key rulebook documents from worldathletics.org and
uploads them to Azure Blob storage for RAG indexing.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base URL for World Athletics downloads
BASE_URL = "https://worldathletics.org/download/download"

# Documents to download (English versions)
DOCUMENTS = {
    "Competition_Technical_Rules_2025.pdf": {
        "filename": "89292580-aaca-4ea3-9c96-f4d05afb1bf8.pdf",
        "urlslug": "C1.1 & C2.1 - Competition Rules & Technical Rules",
        "description": "Competition & Technical Rules (Dec 2025)"
    },
    "Athletic_Shoe_Regulations_2026.pdf": {
        "filename": "3b2559ca-694d-4efb-80bb-4b17cdca0cb5.pdf",
        "urlslug": "C2.1A",
        "description": "Athletic Shoe Regulations (Jan 2026)"
    },
    "Mechanical_Aids_Regulations.pdf": {
        "filename": "66b24876-a6d2-46a0-9f33-c311ab921b18.pdf",
        "urlslug": "C2.1B",
        "description": "Mechanical Aids Regulations"
    },
    "Eligibility_Rules_2025.pdf": {
        "filename": "17b72e0c-15c3-47da-ba2a-4dc06c67d1e7.pdf",
        "urlslug": "C3.3A",
        "description": "Eligibility Rules (Dec 2025)"
    },
    "Male_Female_Category_Regulations.pdf": {
        "filename": "f84d186d-64a5-42c7-9a93-9100f20c628b.pdf",
        "urlslug": "C3.5A",
        "description": "Male and Female Category Regulations (Dec 2025)"
    }
}

# Local documents folder
DOCS_FOLDER = Path(__file__).parent / "documents"


def download_documents():
    """Download all rulebook PDFs to local documents folder."""
    DOCS_FOLDER.mkdir(exist_ok=True)

    downloaded = []
    failed = []

    for local_name, doc_info in DOCUMENTS.items():
        local_path = DOCS_FOLDER / local_name

        # Skip if already downloaded
        if local_path.exists():
            print(f"[OK] Already exists: {local_name}")
            downloaded.append(local_name)
            continue

        # Construct download URL
        url = f"{BASE_URL}?filename={doc_info['filename']}&urlslug={doc_info['urlslug']}"

        print(f"Downloading: {doc_info['description']}...")
        try:
            response = requests.get(url, timeout=60, allow_redirects=True)
            response.raise_for_status()

            # Verify it's a PDF
            if b'%PDF' not in response.content[:10]:
                print(f"  [WARN] {local_name} may not be a valid PDF")

            # Save locally
            with open(local_path, 'wb') as f:
                f.write(response.content)

            size_kb = len(response.content) / 1024
            print(f"  [OK] Saved: {local_name} ({size_kb:.1f} KB)")
            downloaded.append(local_name)

        except Exception as e:
            print(f"  [FAIL] {local_name} - {e}")
            failed.append(local_name)

    return downloaded, failed


def upload_to_azure():
    """Upload documents to Azure Blob Storage."""
    try:
        from document_rag import get_rag

        rag = get_rag()
        # Check if Azure is available via the vector store's azure helper
        if not rag.vector_store.azure.is_available():
            print("\n[WARN] Azure Blob Storage not available. Documents saved locally only.")
            print("  Set AZURE_STORAGE_CONNECTION_STRING in .env to enable cloud storage.")
            return False

        print("\nUploading to Azure Blob Storage...")

        for pdf_file in DOCS_FOLDER.glob("*.pdf"):
            result = rag.upload_document(str(pdf_file))
            if result.get("success"):
                print(f"  [OK] Uploaded: {pdf_file.name}")
            else:
                print(f"  [FAIL] {pdf_file.name} - {result.get('error')}")

        return True

    except ImportError:
        print("\n[WARN] document_rag module not available. Install dependencies:")
        print("  pip install sentence-transformers pypdf")
        return False
    except Exception as e:
        print(f"\n[WARN] Upload failed: {e}")
        print("  Documents saved locally and will be indexed from local folder.")
        return False


def index_documents():
    """Index all documents for RAG search."""
    try:
        from document_rag import get_rag

        rag = get_rag()

        print("\nIndexing documents for RAG search...")
        result = rag.index_documents(str(DOCS_FOLDER))

        # Keys are 'processed' and 'chunks' in the result dict
        chunks = result.get('chunks', 0)
        processed = result.get('processed', 0)
        errors = result.get('errors', [])

        print(f"  [OK] Indexed {chunks} chunks from {processed} documents")
        if errors:
            for err in errors:
                print(f"  [WARN] {err}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("World Athletics Rulebook Downloader")
    print("=" * 60)

    # Step 1: Download documents
    print("\n1. Downloading rulebook PDFs...")
    downloaded, failed = download_documents()
    print(f"\n   Downloaded: {len(downloaded)}, Failed: {len(failed)}")

    if not downloaded:
        print("\n[FAIL] No documents downloaded. Check your internet connection.")
        return

    # Step 2: Upload to Azure (optional)
    print("\n2. Uploading to Azure Blob Storage...")
    upload_to_azure()

    # Step 3: Index for RAG
    print("\n3. Indexing documents for search...")
    index_documents()

    print("\n" + "=" * 60)
    print("Complete! Documents are ready for RAG search in Tab 8.")
    print("=" * 60)


if __name__ == "__main__":
    main()
